"""`ecnode check`, `ecnode policy-template`, `ecnode version`, and `--stdio` as a real child process."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ecnode import __version__, proto
from ecnode.__main__ import main
from ecnode.cli import candidate_roots, check, policy_template
from ecnode.gpus import Gpus
from ecnode.policy import parse

from .conftest import ARGV_SECRET, ENV_VALUE, GATEWAY_IMAGE, manage_policy, write_policy

SRC = Path(__file__).resolve().parents[1] / "src"


def test_check_prints_a_readable_summary_without_secrets(agent_for, machine, fakebin):
    a = agent_for(manage_policy(machine, mode="monitor"))
    out = io.StringIO()
    assert check(None, out, agent=a) == 0
    text = out.getvalue()
    for frag in (f"ecnode {__version__}", "Test Linux 1.0", "GPUs     2 via nvidia-smi", "NVIDIA Test GPU",
                 "docker   29.0.0 · context testctx · rootless yes · nvidia runtime yes", "mode MONITOR",
                 f"hf         hf-home  {machine.data / 'hf'}", "Data Drive 8TB",
                 "from 10.0.0.9:/export/models", "running  3 containers (2 engines, 1 console-owned)"):
        assert frag in text, frag
    assert ENV_VALUE not in text and ARGV_SECRET not in text


def test_check_flags_a_refused_policy(machine, fakebin, tmp_path, prober):
    from ecnode.agent import Agent
    from ecnode.disks import Disks
    from ecnode.policy import PolicyStore

    path = write_policy(tmp_path / "p" / "policy.json", manage_policy(machine), mode=0o666)
    a = Agent(PolicyStore(str(path), home=str(tmp_path), etc_path=str(tmp_path / "none")), paths=machine.paths,
              gpus=Gpus(use_nvml=False), prober=prober.probe, metrics_get=prober.get)
    a.disk = Disks(machine.paths, statvfs=machine.statvfs)
    out = io.StringIO()
    assert check(None, out, agent=a) == 1
    assert "! policy file" in out.getvalue() and "writable" in out.getvalue()
    assert set(prober.tried) == {8000, 8080, 11434, 30000}     # the default ports, refused by the test prober


def test_candidate_roots_point_inside_caches_and_ollama_homes(tmp_path):
    home = tmp_path / "home"
    (home / ".cache" / "huggingface" / "hub").mkdir(parents=True)
    (home / ".ollama" / "models" / "manifests").mkdir(parents=True)
    (home / "models").mkdir()
    roots = candidate_roots(env={}, home=str(home))
    assert [(r["id"], r["kind"], r["path"]) for r in roots] == [
        ("hf", "hf-hub", str(home / ".cache" / "huggingface" / "hub")),
        ("ollama", "ollama", str(home / ".ollama" / "models")),
        ("models", "auto", str(home / "models"))]
    assert all(r["writable"] is False for r in roots)


def test_policy_template_is_a_valid_monitor_policy(fakebin, tmp_path):
    home = tmp_path / "home"
    (home / ".cache" / "huggingface" / "hub").mkdir(parents=True)
    out = io.StringIO()
    assert policy_template(out, gpus=Gpus(use_nvml=False), env={}, home=str(home)) == 0
    data = json.loads(out.getvalue())
    p = parse(data, None, home=str(home))
    assert p.mode == "monitor" and p.docker_context == "testctx" and p.gateway_image == GATEWAY_IMAGE
    assert [r.id for r in p.model_roots] == ["hf"] and len(data["_detected_gpus"]) == 2
    assert not list(tmp_path.glob("**/policy.json"))          # it writes nothing


def test_template_prefers_an_existing_rootless_context(fakebin):
    from ecnode.cli import detect_context

    fakebin.update(context="default", contexts=["default", "rootless"])
    assert detect_context() == "rootless"
    fakebin.update(context="default", contexts=["default"])
    assert detect_context() is None
    fakebin.update(context="lab-remote", contexts=["default", "lab-remote", "rootless"])
    assert detect_context() == "lab-remote"


def test_version_and_usage(capsys):
    assert main(["version"]) == 0 and capsys.readouterr().out.strip() == f"ecnode {__version__}"
    assert main([]) == 2
    with pytest.raises(SystemExit):
        main(["--stdio", "version"])


CHILD = textwrap.dedent("""
    import os, sys
    sys.path.insert(0, {src!r})
    from pathlib import Path
    import ecnode.agent as ag
    from ecnode.disks import Disks
    from ecnode.hostinfo import Paths
    paths = Paths(proc=Path({proc!r}), sys=Path({sysd!r}), etc=Path({etc!r}), state=Path({state!r}))
    fake_vfs = lambda p: os.statvfs_result((4096, 4096, 1000, 500, 500, 10, 5, 5, 0, 255))
    real_init = ag.Agent.__init__
    def init(self, policies, **kw):
        real_init(self, policies, paths=paths, **kw)
        self.disk = Disks(paths, statvfs=fake_vfs)
    ag.Agent.__init__ = init
    real_sample = ag.Agent.sample
    def noisy_sample(self, seq):
        print("a stray print from inside the node")       # must land on stderr, not in the frame stream
        return real_sample(self, seq)
    ag.Agent.sample = noisy_sample
    print("login banner noise before the node starts", flush=True)
    from ecnode.__main__ import main
    sys.exit(main(["--stdio", "--policy", {policy!r}]))
""")


def test_stdio_child_process_protects_its_stream(machine, fakebin, tmp_path):
    policy = write_policy(tmp_path / "p" / "policy.json", manage_policy(machine))
    code = CHILD.format(src=str(SRC), proc=str(machine.paths.proc), sysd=str(machine.paths.sys),
                        etc=str(machine.paths.etc), state=str(machine.paths.state), policy=str(policy))
    child = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, env=os.environ.copy())
    try:
        assert child.stdout is not None and child.stdin is not None
        skipped = proto.find_magic_sync(child.stdout)
        assert skipped == len(b"login banner noise before the node starts\n")
        hello = proto.read_frame_sync(child.stdout)
        assert hello is not None and hello.msg["type"] == "hello" and hello.msg["policy"]["mode"] == "manage"
        child.stdin.write(proto.encode_json(proto.req(1, "node.info")))
        child.stdin.flush()
        kinds = []
        while True:
            f = proto.read_frame_sync(child.stdout)
            assert f is not None
            kinds.append(f.msg["type"])
            if f.msg["type"] == "res":
                assert f.msg["ok"] and f.msg["id"] == 1
                break
        while "sample" not in kinds:                   # a sample (with its stray print) arrives intact
            f = proto.read_frame_sync(child.stdout)
            assert f is not None
            kinds.append(f.msg["type"])
        child.stdin.close()                            # EOF: the node exits by itself
        assert child.wait(timeout=20) == 0
        assert b"a stray print from inside the node" in child.stderr.read()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
