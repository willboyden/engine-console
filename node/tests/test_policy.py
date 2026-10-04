"""The node policy: lookup order, refused files (-> monitor mode with a reason), schema, image patterns."""
from __future__ import annotations

import json
import os

import pytest

from ecnode import policy as pol
from ecnode.policy import PolicyStore, image_matches, load, parse

from .conftest import write_policy

GOOD = {"version": 1, "mode": "manage", "engine_images": ["vllm/vllm-openai:*"], "discovery_ports": [],
        "model_roots": [{"id": "hf", "path": "/data/hf", "kind": "hf-home", "writable": True}]}


def _load(tmp_path, path=None, **kw):
    return load(path, home=str(tmp_path / "home"), etc_path=str(tmp_path / "etc" / "policy.json"), **kw)


# ── lookup order ─────────────────────────────────────────────────────────────────────────────────────────
def test_no_file_means_monitor_mode_with_a_reason(tmp_path):
    p = _load(tmp_path)
    assert p.mode == "monitor" and p.path is None and "no policy file" in (p.error or "")


def test_etc_file_wins_over_the_user_file(tmp_path):
    user = write_policy(tmp_path / "home" / ".config" / "ecnode" / "policy.json", {**GOOD, "mode": "monitor"})
    assert _load(tmp_path).path == str(user)
    etc = write_policy(tmp_path / "etc" / "policy.json", GOOD)
    # the etc location must be root-owned: a user-owned file there is refused (monitor mode, with the reason)
    p = load(None, home=str(tmp_path / "home"), etc_path=str(etc), system_dirs=(str(tmp_path / "etc") + "/",))
    assert p.path == str(etc) and p.mode == "monitor" and "owned by root" in (p.error or "")
    # outside the system dirs the same file is fine
    assert _load(tmp_path).path == str(etc) and _load(tmp_path).mode == "manage"


def test_explicit_path_is_used_and_missing_is_reported(tmp_path):
    p = _load(tmp_path, str(tmp_path / "missing.json"))
    assert p.mode == "monitor" and "does not exist" in (p.error or "")
    good = write_policy(tmp_path / "x" / "p.json", GOOD)
    assert _load(tmp_path, str(good)).mode == "manage"


# ── refused files ────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", [0o664, 0o646, 0o666])
def test_group_or_world_writable_file_is_refused(tmp_path, mode):
    p = _load(tmp_path, str(write_policy(tmp_path / "x" / "p.json", GOOD, mode=mode)))
    assert p.mode == "monitor" and "writable" in (p.error or "")


def test_symlinked_policy_is_refused(tmp_path):
    real = write_policy(tmp_path / "x" / "real.json", GOOD)
    link = tmp_path / "x" / "link.json"
    link.symlink_to(real)
    p = _load(tmp_path, str(link))
    assert p.mode == "monitor" and "symlink" in (p.error or "")


def test_policy_in_a_world_writable_directory_is_refused(tmp_path):
    d = tmp_path / "open"
    path = write_policy(d / "p.json", GOOD)
    d.chmod(0o777)
    try:
        p = _load(tmp_path, str(path))
    finally:
        d.chmod(0o700)
    assert p.mode == "monitor" and "others can write" in (p.error or "")


def test_group_writable_directory_is_fine_for_a_user_policy_but_not_under_etc(tmp_path):
    d = tmp_path / "grp"
    path = write_policy(d / "p.json", GOOD)
    d.chmod(0o775)
    assert _load(tmp_path, str(path)).mode == "manage"
    p = _load(tmp_path, str(path), system_dirs=(str(d) + "/",), uid=os.getuid())
    assert p.mode == "monitor"     # (user-owned under a system dir is refused for ownership first)


def test_fifo_is_refused_without_hanging(tmp_path):
    fifo = tmp_path / "x" / "fifo.json"
    fifo.parent.mkdir()
    os.mkfifo(fifo, 0o644)
    p = _load(tmp_path, str(fifo))
    assert p.mode == "monitor" and "regular file" in (p.error or "")


@pytest.mark.parametrize("text,why", [
    ("{not json", "not valid JSON"),
    ('{"mode": "manage", "allow_delte": true}', "unknown key"),
    ('{"mode": "admin"}', "monitor\" or \"manage"),
    ('{"version": 2}', "version"),
    ('{"max_shm_gib": NaN}', "not valid JSON"),
    ('[1, 2]', "one JSON object"),
])
def test_bad_content_means_monitor_mode_with_the_reason(tmp_path, text, why):
    path = tmp_path / "x" / "p.json"
    path.parent.mkdir(mode=0o700)
    path.write_text(text)
    path.chmod(0o644)
    p = _load(tmp_path, str(path))
    assert p.mode == "monitor" and why in (p.error or ""), p.error
    assert p.model_roots == () and p.scripts == ()            # nothing from a refused file is used


def test_oversized_file_is_refused(tmp_path):
    path = write_policy(tmp_path / "x" / "p.json", {**GOOD, "_pad": "x" * (pol.MAX_POLICY_BYTES + 10)})
    assert "larger than" in (_load(tmp_path, str(path)).error or "")


# ── schema ───────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("change,fragment", [
    ({"model_roots": [{"id": "x", "path": "relative"}]}, "absolute path"),
    ({"model_roots": [{"id": "x", "path": "/"}]}, "too broad"),
    ({"model_roots": [{"id": "x", "path": "/etc"}]}, "too broad"),
    ({"model_roots": [{"id": "x", "path": "/data/../etc"}]}, "'..'"),
    ({"model_roots": [{"id": "x", "path": "/data/a:b"}]}, "':'"),
    ({"model_roots": [{"id": "Bad Id", "path": "/data/x"}]}, "not a valid value"),
    ({"model_roots": [{"id": "x", "path": "/data/x"}, {"id": "x", "path": "/data/y"}]}, "used twice"),
    ({"model_roots": [{"id": "x", "path": "/data/x", "kind": "zip"}]}, "kind must be"),
    ({"engine_images": ["*"]}, "literal repository"),
    ({"engine_images": ["vllm/*:*"]}, "literal repository"),
    ({"gateway_image": "nginx:*"}, "exact image reference"),
    ({"port_range": [18100, 18000]}, "port_range[1]"),
    ({"port_range": [80, 90]}, "port_range[0]"),
    ({"discovery_ports": [0]}, "discovery_ports[0]"),
    ({"docker": {"binary": "../docker"}}, "absolute path"),
    ({"docker": {"context": "-H tcp://x"}}, "not a valid value"),
    ({"external_control": "all"}, "external_control"),
    ({"max_shm_gib": 0}, "max_shm_gib"),
    ({"allow_pull": "yes"}, "true or false"),
    ({"scripts": [{"name": "s", "actions": [{"name": "up", "argv": ["model.sh", "up"]}]}]}, "absolute path"),
    ({"scripts": [{"name": "s", "actions": [{"name": "up", "argv": []}]}]}, "1-64 non-empty"),
    ({"scripts": [{"name": "s", "status_action": "nope", "actions": [{"name": "up", "argv": ["/bin/true"]}]}]},
     "status_action"),
])
def test_schema_errors_name_the_field(change, fragment):
    with pytest.raises(pol.PolicyError) as e:
        parse({**GOOD, **change}, "/x", home="/home/someone")
    assert fragment in str(e.value)


def test_home_directory_itself_is_not_a_root():
    with pytest.raises(pol.PolicyError):
        parse({"model_roots": [{"id": "h", "path": "/home/someone"}]}, None, home="/home/someone")
    parse({"model_roots": [{"id": "h", "path": "/home/someone/models"}]}, None, home="/home/someone")


def test_comment_keys_and_defaults():
    p = parse({"_comment": "hi", "mode": "manage"}, "/x")
    assert p.manage and p.engine_images == () and p.allow_ipc_host is False and p.allow_delete is False
    assert p.discovery_ports == pol.DEFAULT_DISCOVERY and p.port_range == (18000, 18099)
    assert parse({"discovery_ports": []}, None).discovery_ports == ()


def test_summary_never_contains_script_argv():
    p = parse({**GOOD, "scripts": [{"name": "big", "containers": ["lab-sglang-*"], "status_action": "status",
                                    "actions": [{"name": "status", "argv": ["/opt/lab/secret-path.sh", "status"]},
                                                {"name": "switch", "argv": ["/opt/lab/use.sh"], "args": ["a", "b"]}]}]},
              "/x")
    s = json.dumps(p.summary())
    assert "secret-path" not in s and "/opt/lab" not in s
    assert p.summary()["scripts"][0]["actions"][1]["args"] == ["a", "b"]


# ── image patterns ───────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("pattern,image,ok", [
    ("vllm/vllm-openai:*", "vllm/vllm-openai:v0.11.0", True),
    ("vllm/vllm-openai:*", "vllm/vllm-openai:latest", True),
    ("vllm/vllm-openai:*", "evil/vllm/vllm-openai:x", False),
    ("vllm/vllm-openai:*", "vllm/vllm-openai-evil:x", False),
    ("vllm/vllm-openai:*", "docker.io/vllm/vllm-openai:x", False),
    ("vllm/vllm-openai:*", "vllm/vllm-openai", False),                       # no tag
    ("vllm/vllm-openai:*", "vllm/vllm-openai:x@sha256:" + "a" * 64, False),  # tag + digest is not "a tag"
    ("vllm/vllm-openai:*", "vllm/vllm-openai:x y", False),
    ("vllm/vllm-openai:*", "vllm/vllm-openai:-x", False),
    ("vllm/vllm-openai:*", "--help", False),
    ("lmsysorg/sglang@sha256:*", "lmsysorg/sglang@sha256:" + "b" * 64, True),
    ("lmsysorg/sglang@sha256:*", "lmsysorg/sglang:latest", False),
    ("lmsysorg/sglang@sha256:*", "lmsysorg/sglang@sha256:short", False),
    ("nvcr.io/nvidia/tensorrt-llm/release:*", "nvcr.io/nvidia/tensorrt-llm/release:1.0", True),
    ("nvcr.io/nvidia/tensorrt-llm/release:*", "nvcr.io/nvidia/tensorrt-llm/release-x:1.0", False),
    ("localhost:5000/eng:*", "localhost:5000/eng:t1", True),
    ("ghcr.io/ggml-org/llama.cpp:server-cuda", "ghcr.io/ggml-org/llama.cpp:server-cuda", True),
    ("ghcr.io/ggml-org/llama.cpp:server-cuda", "ghcr.io/ggml-org/llama.cpp:server", False),
])
def test_image_patterns_compare_the_repository_literally(pattern, image, ok):
    assert pol.valid_image_pattern(pattern)
    assert image_matches(pattern, image) is ok


# ── reloads ──────────────────────────────────────────────────────────────────────────────────────────────
def test_store_reloads_an_edited_file(tmp_path):
    path = write_policy(tmp_path / "x" / "p.json", GOOD)
    clock = [0.0]
    store = PolicyStore(str(path), home=str(tmp_path / "home"), etc_path=str(tmp_path / "etc" / "p.json"),
                        recheck_s=2.0, clock=lambda: clock[0])
    assert store.current().manage
    write_policy(path, {**GOOD, "mode": "monitor"})
    os.utime(path, ns=(1, 1))                      # a different mtime even within the same clock tick
    assert store.current().manage                  # not re-checked yet
    clock[0] = 5.0
    assert store.current().mode == "monitor"


def test_shipped_examples_are_valid_policies():
    from pathlib import Path

    ex = Path(__file__).resolve().parents[1] / "examples"
    for name, mode in (("policy.monitor.json", "monitor"), ("policy.manage.json", "manage")):
        p = parse(json.loads((ex / name).read_text()), name, home="/home/someone")
        assert p.mode == mode and p.model_roots
