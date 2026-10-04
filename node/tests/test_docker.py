"""docker.* ops against the fake docker CLI: run-spec rules, argv hardening, secrets, ownership, gateway."""
from __future__ import annotations

import json
import os
import stat
from typing import Any

import pytest

from ecnode import ops, proto
from ecnode.errors import OpError
from ecnode.ops import HANDLERS, Call

from .conftest import ARGV_SECRET, CID_ENGINE, CID_EXT, ENV_VALUE, GATEWAY_IMAGE, GPU_A, GPU_B, manage_policy
from .test_fsops import FakeStream

SECRET = "hub-secret-value-for-the-engine"


def call(a, op: str, args: dict[str, Any], stream: FakeStream | None = None) -> dict[str, Any]:
    cleaned = proto.clean_args(op, args)
    assert not proto.missing_args(op, cleaned), proto.missing_args(op, cleaned)
    return HANDLERS[op](a, Call(op, cleaned, stream or FakeStream()))


def err(a, op, args, stream=None) -> OpError:
    with pytest.raises(OpError) as e:
        call(a, op, args, stream)
    return e.value


def spec(machine, **over: Any) -> dict[str, Any]:
    s = {"name": "ec-test-1", "instance": "inst_9", "image": "vllm/vllm-openai:v0.11.0",
         "argv": ["--model", "org/m", "--port", "8000"], "container_port": 8000, "gpu_uuids": [GPU_A, GPU_B],
         "labels": {"engine-console.role": "engine", "engine-console.instance": "inst_9",
                    "engine-console.engine": "vllm", "engine-console.gpu_ids": "0,1"},
         "env": {"HF_HUB_OFFLINE": "1"}, "secret_env": {"VLLM_API_KEY": SECRET},
         "mounts": [{"root": "hf", "rel": "", "dest": "/root/.cache/huggingface", "ro": True}],
         "volumes": [{"name": "ec-vllm-cache", "dest": "/root/.cache/vllm"}], "shm_size": "16g", "ipc_host": True,
         "files": {"/etc/engine/chat.jinja": "{{ messages }}"}}
    s.update(over)
    return s


@pytest.fixture
def agent(agent_for, machine):
    a = agent_for(manage_policy(machine))
    a.hello()            # learns the GPUs (fake nvidia-smi)
    return a


def test_render_shows_the_hardened_argv_without_secret_values(agent, machine, fakebin):
    argv = call(agent, "docker.render", {"spec": spec(machine)})["argv"]
    assert argv[:4] == ["docker", "--context", "testctx", "run"]
    joined = " ".join(argv)
    for frag in ("--cap-drop ALL", "--security-opt no-new-privileges:true", "--pids-limit 4096", "--restart no",
                 "--pull never", "--network engine-console-engines", "--network-alias ec-test-1",
                 f'--gpus "device={GPU_A},{GPU_B}"', "--ipc host", "--shm-size 16g",
                 "--label engine-console=1", "--label engine-console.instance=inst_9", "-e HF_HUB_OFFLINE=1",
                 f"-v {machine.data / 'hf'}:/root/.cache/huggingface:ro", "-v ec-vllm-cache:/root/.cache/vllm"):
        assert frag in joined, frag
    assert argv[argv.index("-e", argv.index("HF_HUB_OFFLINE=1")) + 1] == "VLLM_API_KEY"
    assert SECRET not in joined
    assert argv[-5:] == ["vllm/vllm-openai:v0.11.0", "--model", "org/m", "--port", "8000"]
    files = machine.paths.state / "instances" / "inst_9" / "files"
    assert f"{files}/0-chat.jinja:/etc/engine/chat.jinja:ro" in joined
    assert not files.exists()                                 # render changes nothing
    assert not any(c["argv"][2:3] == ["run"] for c in fakebin.calls())


def test_run_passes_secrets_only_through_the_child_environment(agent, machine, fakebin):
    res = call(agent, "docker.run", {"spec": spec(machine)})
    assert len(res["id"]) == 64
    runs = [c for c in fakebin.calls() if "run" in c["argv"][:3]]
    assert len(runs) == 1 and SECRET not in " ".join(runs[0]["argv"])
    assert runs[0]["env"] == {"VLLM_API_KEY": SECRET}             # docker read it from its own environment
    assert fakebin.state()["networks"]["engine-console-engines"] == {"internal": True}
    fdir = machine.paths.state / "instances" / "inst_9" / "files"
    assert (fdir / "0-chat.jinja").read_text() == "{{ messages }}"
    for d in (machine.paths.state, fdir.parent, fdir):
        assert stat.S_IMODE(os.stat(d).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(fdir / "0-chat.jinja").st_mode) == 0o644


@pytest.mark.parametrize("over,code,fragment", [
    ({"name": "my-engine"}, proto.BAD_ARGS, "spec.name"),
    ({"name": "ec-UPPER"}, proto.BAD_ARGS, "spec.name"),
    ({"instance": "bad id!"}, proto.BAD_ARGS, "spec.instance"),
    ({"image": "evil/vllm/vllm-openai:x"}, proto.POLICY_DENIED, "not allowed"),
    ({"image": "vllm/vllm-openai-evil:x"}, proto.POLICY_DENIED, "not allowed"),
    ({"image": "--privileged"}, proto.BAD_ARGS, "image reference"),
    ({"container_port": 0}, proto.BAD_ARGS, "container_port"),
    ({"gpu_uuids": ["GPU-not-here"]}, proto.BAD_ARGS, "not a GPU of this machine"),
    ({"network_alias": "x"}, proto.BAD_ARGS, "network_alias"),
    ({"labels": {"com.evil": "1"}}, proto.BAD_ARGS, "engine-console."),
    ({"labels": {"engine-console.role": "gateway"}}, proto.BAD_ARGS, "must be 'engine'"),
    ({"labels": {"engine-console.x": "a\nb"}}, proto.BAD_ARGS, "control character"),
    ({"env": {"1BAD": "x"}}, proto.BAD_ARGS, "env name"),
    ({"env": {"HF_TOKEN": "x"}}, proto.BAD_ARGS, "secret_env"),
    ({"secret_env": {"LD_PRELOAD": "/tmp/x.so"}}, proto.POLICY_DENIED, "docker CLI"),
    ({"secret_env": {"DOCKER_HOST": "tcp://x"}}, proto.POLICY_DENIED, "docker CLI"),
    ({"secret_env": {"PATH": "/tmp"}}, proto.POLICY_DENIED, "docker CLI"),
    ({"env": {"X": "1"}, "secret_env": {"X": "2"}}, proto.BAD_ARGS, "both"),
    ({"mounts": [{"root": "nope", "rel": "", "dest": "/m"}]}, proto.NOT_FOUND, "model root"),
    ({"mounts": [{"root": "hf", "rel": "../..", "dest": "/m"}]}, proto.BAD_ARGS, "'..'"),
    ({"mounts": [{"root": "hf", "rel": "", "dest": "relative"}]}, proto.BAD_ARGS, "absolute"),
    ({"mounts": [{"root": "hf", "rel": "", "dest": "/m:rw,z"}]}, proto.BAD_ARGS, "':'"),
    ({"mounts": [{"root": "hf", "rel": "", "dest": "/a/../etc"}]}, proto.BAD_ARGS, "'..'"),
    ({"mounts": [{"root": "hf", "rel": "", "dest": "/"}]}, proto.BAD_ARGS, "not '/'"),
    ({"volumes": [{"name": "data", "dest": "/d"}]}, proto.BAD_ARGS, "ec-<name>"),
    ({"volumes": [{"name": "ec-x", "dest": "/root/.cache/huggingface"}]}, proto.BAD_ARGS, "used twice"),
    ({"shm_size": "16G"}, proto.BAD_ARGS, "shm_size"),
    ({"shm_size": "65g"}, proto.POLICY_DENIED, "max_shm_gib"),
    ({"files": {"relative.txt": "x"}}, proto.BAD_ARGS, "absolute"),
])
def test_run_spec_rules(agent, machine, over, code, fragment):
    e = err(agent, "docker.run", {"spec": spec(machine, **over)})
    assert e.code == code and fragment in e.message, e.message


def test_mount_symlink_escape_and_writable_rules(agent, machine, tmp_path):
    (machine.data / "hf" / "esc").symlink_to(tmp_path)
    e = err(agent, "docker.render", {"spec": spec(machine, mounts=[{"root": "hf", "rel": "esc", "dest": "/m"}])})
    assert e.code == proto.POLICY_DENIED
    (machine.data / "ro" / "m").mkdir()
    argv = call(agent, "docker.render", {"spec": spec(machine, mounts=[
        {"root": "ro", "rel": "m", "dest": "/models", "ro": False},       # read-only root: stays :ro
        {"root": "hf", "rel": "", "dest": "/hf", "ro": False}])})["argv"]  # writable root and ro false: rw
    assert f"{machine.data / 'ro' / 'm'}:/models:ro" in argv and f"{machine.data / 'hf'}:/hf" in argv


def test_empty_rel_is_the_root_itself_for_mounts_only(agent, machine):
    argv = call(agent, "docker.render", {"spec": spec(machine, mounts=[
        {"root": "hf", "rel": "", "dest": "/hf"}])})["argv"]
    assert f"{machine.data / 'hf'}:/hf:ro" in argv
    for rel in (".", "./", "x/../"):
        e = err(agent, "docker.render", {"spec": spec(machine, mounts=[{"root": "hf", "rel": rel, "dest": "/hf"}])})
        assert e.code == proto.BAD_ARGS, rel


def test_docker_context_falls_back_to_rootless_only_when_default_is_unusable(fakebin, tmp_path):
    from ecnode.docker import effective_context

    fakebin.update(context="default", contexts=["default", "rootless", "desktop-linux"])
    nosock = str(tmp_path / "no-such.sock")
    assert effective_context("docker", "lab", env={}, default_socket=nosock) == "lab"           # policy wins
    assert effective_context("docker", None, env={}, default_socket=nosock) == "rootless"
    assert effective_context("docker", None, env={"DOCKER_HOST": "unix:///x"}, default_socket=nosock) is None
    assert effective_context("docker", None, env={"DOCKER_CONTEXT": "x"}, default_socket=nosock) is None
    usable = tmp_path / "docker.sock"
    usable.write_text("")
    usable.chmod(0o600)                                       # a default socket this user may use: keep "default"
    assert effective_context("docker", None, env={}, default_socket=str(usable)) is None
    fakebin.update(context="lab-remote")                      # a current context chosen in the CLI config
    assert effective_context("docker", None, env={}, default_socket=nosock) is None
    fakebin.update(context="default", contexts=["default"])  # no rootless context to fall back to
    assert effective_context("docker", None, env={}, default_socket=nosock) is None


def test_ipc_host_needs_the_policy(agent_for, machine):
    a = agent_for(manage_policy(machine, allow_ipc_host=False))
    a.hello()
    assert err(a, "docker.render", {"spec": spec(machine)}).code == proto.POLICY_DENIED
    argv = call(a, "docker.render", {"spec": spec(machine, ipc_host=False)})["argv"]
    assert argv[argv.index("--ipc") + 1] == "private"


def test_existing_container_name_is_a_conflict_and_missing_image_not_found(agent, machine, fakebin):
    assert err(agent, "docker.run", {"spec": spec(machine, name="ec-owned-1")}).code == proto.CONFLICT
    assert err(agent, "docker.run", {"spec": spec(machine, image="vllm/vllm-openai:v9")}).code == proto.NOT_FOUND


def test_engine_network_that_is_not_internal_is_refused(agent, machine, fakebin):
    fakebin.update(networks={"engine-console-engines": {"internal": False}})
    assert err(agent, "net.ensure_internal", {}).code == proto.CONFLICT
    assert err(agent, "docker.run", {"spec": spec(machine)}).code == proto.CONFLICT
    fakebin.update(networks={})
    assert call(agent, "net.ensure_internal", {}) == {"network": "engine-console-engines"}


# ── ownership ────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("op", ["docker.stop", "docker.start", "docker.rm", "docker.logs"])
def test_only_console_owned_containers(agent, op):
    assert err(agent, op, {"name": "other-engine"}).code == proto.NOT_OWNED
    assert err(agent, op, {"name": "no-such"}).code == proto.NOT_FOUND
    assert err(agent, op, {"name": "--all"}).code == proto.BAD_ARGS


def test_role_and_instance_must_match_exactly(agent, fakebin):
    assert err(agent, "docker.stop", {"name": "ec-owned-1", "role": "gateway"}).code == proto.NOT_OWNED
    assert err(agent, "docker.stop", {"name": "ec-owned-1", "instance": "inst_2"}).code == proto.NOT_OWNED
    assert call(agent, "docker.stop", {"name": "ec-owned-1", "instance": "inst_1", "timeout_s": 5}) == {}
    stop = [c["argv"] for c in fakebin.calls() if "stop" in c["argv"]][-1]
    assert stop[-3:] == ["-t", "5", CID_ENGINE]                       # acted on the verified id, not the name


def test_rm_removes_the_instance_files(agent, machine, fakebin):
    call(agent, "docker.run", {"spec": spec(machine)})
    assert (machine.paths.state / "instances" / "inst_9").exists()
    call(agent, "docker.rm", {"name": "ec-test-1", "instance": "inst_9"})
    assert not (machine.paths.state / "instances" / "inst_9").exists()


def test_logs_are_scrubbed_and_tail_clamped(agent, fakebin):
    res = call(agent, "docker.logs", {"name": "ec-owned-1", "tail": 999999})
    assert "INFO ready" in res["text"] and ARGV_SECRET not in res["text"] and res["truncated"] is False
    assert [c["argv"] for c in fakebin.calls() if "logs" in c["argv"]][-1][-3:] == ["--tail", "5000", CID_ENGINE]


def test_inspect_is_sanitized(agent):
    info = call(agent, "docker.inspect", {"name": "other-engine"})["info"]
    blob = json.dumps(info)
    assert ENV_VALUE not in blob and "topsecretvalue" not in blob
    assert info["engine"] == "sglang" and info["managed"] is False and info["compose_project"] == "lab"


# ── images ───────────────────────────────────────────────────────────────────────────────────────────────
def test_images_and_presence(agent):
    imgs = call(agent, "docker.images", {})["images"]
    assert {i["ref"] for i in imgs} == {"vllm/vllm-openai:v0.11.0", GATEWAY_IMAGE}
    assert call(agent, "docker.image_present", {"image": "vllm/vllm-openai:v0.11.0"})["present"] is True
    assert call(agent, "docker.image_present", {"image": "vllm/vllm-openai:nope"}) == {"present": False, "id": None}
    assert err(agent, "docker.image_present", {"image": "-x"}).code == proto.BAD_ARGS


def test_pull_rules_and_progress(agent_for, machine, fakebin):
    a = agent_for(manage_policy(machine, allow_pull=False))
    assert err(a, "docker.pull", {"image": "vllm/vllm-openai:v1"}).code == proto.POLICY_DENIED
    a = agent_for(manage_policy(machine))
    assert err(a, "docker.pull", {"image": "evil/vllm/vllm-openai:x"}).code == proto.POLICY_DENIED
    s = FakeStream()
    res = call(a, "docker.pull", {"image": "vllm/vllm-openai:v1"}, s)
    assert res["id"] == "sha256:" + "ef" * 32 and any("Downloaded" in e["line"] for e in s.events)


# ── engines the console did not create ──────────────────────────────────────────────────────────────────
def test_external_control_rules(agent_for, machine, fakebin):
    a = agent_for(manage_policy(machine, external_control="none"))
    assert err(a, "docker.external", {"name": "other-engine", "action": "stop"}).code == proto.POLICY_DENIED
    scripts = [{"name": "sg", "containers": ["other-*"], "actions": [{"name": "up", "argv": ["/bin/true"]}]}]
    a = agent_for(manage_policy(machine, scripts=scripts))
    e = err(a, "docker.external", {"name": "other-engine", "action": "stop"})
    assert e.code == proto.POLICY_DENIED and "script" in e.message
    a = agent_for(manage_policy(machine))
    assert err(a, "docker.external", {"name": "ec-owned-1", "action": "stop"}).code == proto.POLICY_DENIED
    assert err(a, "docker.external", {"name": "grafana", "action": "stop"}).code == proto.POLICY_DENIED
    assert err(a, "docker.external", {"name": "other-engine", "action": "kill"}).code == proto.BAD_ARGS
    assert call(a, "docker.external", {"name": "other-engine", "action": "stop"}) == {}
    assert [c["argv"] for c in fakebin.calls() if "stop" in c["argv"]][-1][-1] == CID_EXT


# ── gateway ──────────────────────────────────────────────────────────────────────────────────────────────
def gw(**over: Any) -> dict[str, Any]:
    return {"name": "ec-owned-1-gw", "instance": "inst_1", "host_port": 18005, "engine_host": "ec-owned-1",
            "engine_port": 8000, **over}


def test_gateway_rules(agent, monkeypatch):
    monkeypatch.setattr(ops, "port_free", lambda port: True)
    assert err(agent, "gateway.start", gw(host_port=18100)).code == proto.POLICY_DENIED
    assert err(agent, "gateway.start", gw(host_port=8080)).code == proto.POLICY_DENIED
    assert err(agent, "gateway.start", gw(engine_host="other-engine")).code == proto.NOT_OWNED
    assert err(agent, "gateway.start", gw(engine_host="Bad;Host")).code == proto.BAD_ARGS
    assert err(agent, "gateway.start", gw(name="gw")).code == proto.BAD_ARGS
    monkeypatch.setattr(ops, "port_free", lambda port: False)
    assert err(agent, "gateway.start", gw()).code == proto.PORT_IN_USE


def test_gateway_without_an_image_in_the_policy(agent_for, machine, monkeypatch):
    monkeypatch.setattr(ops, "port_free", lambda port: True)
    a = agent_for(manage_policy(machine, gateway_image=None))
    assert err(a, "gateway.start", gw()).code == proto.POLICY_DENIED


def test_gateway_start_creates_connects_and_starts(agent, fakebin, monkeypatch):
    monkeypatch.setattr(ops, "port_free", lambda port: True)
    assert len(call(agent, "gateway.start", gw())["id"]) == 64
    argvs = [c["argv"] for c in fakebin.calls()]
    create = next(a for a in argvs if "create" in a[:3])
    joined = " ".join(create)
    for frag in ("--network bridge", "--read-only", "--cap-drop ALL", "--user 101:101", "--pull never",
                 "-p 127.0.0.1:18005:8080", "--label engine-console.role=gateway",
                 "--label engine-console.instance=inst_1", GATEWAY_IMAGE):
        assert frag in joined, frag
    conf = next(x for x in create if x.startswith("NGINX_CONF="))
    assert "daemon off;" in conf and "proxy_pass ec-owned-1:8000;" in conf
    assert any(a[2:5] == ["network", "connect", "engine-console-engines"] for a in argvs)
    assert any(a[2:4] == ["start", "ec-owned-1-gw"] for a in argvs)


def test_gateway_is_removed_when_it_cannot_start(agent, fakebin, monkeypatch):
    monkeypatch.setattr(ops, "port_free", lambda port: True)
    docker_cls = type(agent.docker())
    real_ok = docker_cls.ok

    def failing_connect(self, *a, **k):
        if a[:2] == ("network", "connect"):
            raise OpError(proto.DOCKER_ERROR, "boom")
        return real_ok(self, *a, **k)

    monkeypatch.setattr(docker_cls, "ok", failing_connect)
    assert err(agent, "gateway.start", gw()).code == proto.DOCKER_ERROR
    assert any(c["argv"][2:5] == ["rm", "-f", "ec-owned-1-gw"] for c in fakebin.calls())


def test_ulimits_are_allowlisted_validated_and_rendered(agent, machine):
    argv = call(agent, "docker.render", {"spec": spec(machine, ulimits={"stack": "67108864",
                                                                          "nofile": "65536:65536"})})["argv"]
    joined = " ".join(argv)
    assert "--ulimit stack=67108864" in joined and "--ulimit nofile=65536:65536" in joined
    assert err(agent, "docker.render", {"spec": spec(machine, ulimits={"memlock": "-1"})}).code == proto.BAD_ARGS
    assert err(agent, "docker.render", {"spec": spec(machine, ulimits={"stack": "lots"})}).code == proto.BAD_ARGS
    assert "--ulimit" not in " ".join(call(agent, "docker.render", {"spec": spec(machine)})["argv"])
