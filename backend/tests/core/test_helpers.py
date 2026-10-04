"""The pure hub-side helpers that survive the move to nodes: snippets, names, the policy's image rule, inventory readers."""
from __future__ import annotations

import pytest
import yaml

from engine_console.domain.fleet import HostConfig
from engine_console.services import nodeview as nv
from engine_console.services.discovery import (
    EXT,
    SCRIPT,
    clean_text,
    display_name,
    parse_generic_metrics,
    slug,
)
from engine_console.services.docker import clean_render, command_snippets, gateway_name

from .fakefleet import FakeNode, ports


# ---- docker.py ----------------------------------------------------------------------------------------------------------------------
def test_gateway_name_is_the_engine_name_plus_gw() -> None:
    assert gateway_name("ec-qwen-abc123") == "ec-qwen-abc123-gw"


@pytest.mark.parametrize(("argv", "expected"), [
    (["run", "-d", "--name", "ec-x", "img:1"], "docker run -d --name ec-x img:1"),
    (["docker", "run", "-d", "img:1"], "docker run -d img:1"),
    (["/usr/bin/docker", "--context", "rootless", "run", "-d", "img:1"], "docker run -d img:1"),       # the context is the machine's own
    (["docker", "--context", "a", "--context", "b", "run", "x y"], "docker run 'x y'")])
def test_the_nodes_rendering_is_normalised_into_one_copyable_line(argv: list[str], expected: str) -> None:
    assert clean_render(argv) == expected


SPEC = {"name": "ec-x", "image": "img:1", "argv": ["--model", "a b", "--port", "8000"], "gpu_uuids": ["GPU-1", "GPU-2"],
        "labels": {"engine-console.instance": "i1"}, "env": {"A": "b"}, "secret_env": {"VLLM_API_KEY": "SHOULD-NEVER-APPEAR"},
        "mounts": [{"root": "hf", "rel": "", "dest": "/root/.cache/huggingface", "ro": True},
                   {"root": "gguf", "rel": "a/b", "dest": "/models", "ro": False}, {"root": "unknown", "rel": "", "dest": "/x", "ro": True}],
        "volumes": [{"name": "ec-vllm-cache", "dest": "/root/.cache/vllm"}], "ipc_host": True, "shm_size": "16g",
        "files": {"/etc/x.json": "{}"}}


def test_command_snippets_have_no_secret_value_and_no_unknown_mount() -> None:
    out = command_snippets(SPEC, ["docker", "run", "-d", "img:1"], host_paths={"hf": "/data/hf", "gguf": "/data/gguf"},
                           network="engines")
    blob = repr(out)
    assert "SHOULD-NEVER-APPEAR" not in blob and out["docker_run"] == "docker run -d img:1"
    assert out["engine_cli"] == "--model 'a b' --port 8000"
    text = out["compose_yaml"]
    assert text.startswith("# generated config files the console mounts: /etc/x.json")
    doc = yaml.safe_load(text)
    svc = doc["services"]["ec-x"]
    assert svc["environment"] == {"A": "b", "VLLM_API_KEY": "${VLLM_API_KEY}"}
    assert svc["volumes"] == ["/data/hf:/root/.cache/huggingface:ro", "/data/gguf/a/b:/models", "ec-vllm-cache:/root/.cache/vllm"]
    assert svc["ipc"] == "host" and svc["shm_size"] == "16g" and svc["cap_drop"] == ["ALL"] and svc["pids_limit"] == 4096
    assert svc["labels"] == {"engine-console": "1", "engine-console.instance": "i1"} and svc["networks"] == ["engines"]
    assert svc["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == ["GPU-1", "GPU-2"]
    assert doc["networks"] == {"engines": {"external": True}}


def test_command_snippets_without_a_rendering_leave_docker_run_empty_and_default_the_network() -> None:
    out = command_snippets({**SPEC, "files": {}, "ipc_host": False}, None)
    assert out["docker_run"] == "" and "engine-console-engines" in out["compose_yaml"] and not out["compose_yaml"].startswith("#")
    assert yaml.safe_load(out["compose_yaml"])["services"]["ec-x"]["ipc"] == "private"


# ---- discovery.py ---------------------------------------------------------------------------------------------------------------------------
def test_text_hygiene_helpers() -> None:
    assert (EXT, SCRIPT) == ("ext-", "scr-")
    assert slug("My Engine!") == "my-engine" and slug("../../etc") == "etc" and slug("") == "unknown" and len(slug("x" * 99)) == 48
    assert display_name("a\x00b\x1b[0m c") == "ab0m c" and display_name("!!!") == "external" and len(display_name("x" * 99)) == 64
    assert clean_text("a\nb\x00c", 5) == "abc" and clean_text("x" * 300) == "x" * 200


def test_generic_metrics_map_only_what_it_knows_and_survives_garbage() -> None:
    text = ("# HELP x\nllamacpp:requests_processing 3\nllamacpp:requests_deferred{a=\"b\"} 1\nllamacpp:tokens_predicted_total 9.5\n"
            "other 5\nllamacpp:prompt_tokens_total notanumber\n\n")
    assert parse_generic_metrics("llamacpp", text) == {"requests_running": 3.0, "requests_waiting": 1.0, "generation_tokens_total": 9.5}
    assert parse_generic_metrics("unknown-engine", text) == {} and parse_generic_metrics("llamacpp", "") == {}


# ---- nodeview.py ----------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("image", "patterns", "ok"), [
    ("vllm/vllm-openai:v0.23.0", ["vllm/vllm-openai:*"], True), ("vllm/vllm-openai:v0.23.0", ["vllm/vllm-openai:v0.23.0"], True),
    ("vllm/vllm-openai:v0.24.0", ["vllm/vllm-openai:v0.23.0"], False), ("vllm/vllm-openai:evil/x", ["vllm/vllm-openai:*"], False),
    ("vllm/vllm-openai-evil:1", ["vllm/vllm-openai:*"], False), ("other/vllm/vllm-openai:1", ["vllm/vllm-openai:*"], False),
    ("vllm/vllm-openai:1:2", ["vllm/vllm-openai:*"], False), ("vllm/vllm-openai:", ["vllm/vllm-openai:*"], False),
    ("vllm/vllm-openai@sha256:" + "a" * 64, ["vllm/vllm-openai@sha256:*"], True),
    ("vllm/vllm-openai@sha256:" + "A" * 64, ["vllm/vllm-openai@sha256:*"], False), ("vllm/vllm-openai@sha256:abc", ["vllm/vllm-openai@sha256:*"], False),
    ("vllm/vllm-openai:1", ["vllm/vllm-openai@sha256:*"], False), ("a/b:1", ["*:*"], False), ("a/b:1", ["a/*:*"], False),
    ("localhost:5000/eng:1", ["localhost:5000/eng:*"], True), ("a/b:1", [], False), ("nginx:1", ["nginx@sha256:*"], False)])
def test_the_policys_image_rule_is_a_literal_repository_never_a_prefix(image: str, patterns: list[str], ok: bool) -> None:
    assert nv.image_allowed(image, patterns) is ok


def test_policy_readers_tolerate_a_node_that_said_nothing_or_nonsense() -> None:
    node = FakeNode(HostConfig(id="x", label="X"))
    assert nv.mode(node) == "manage" and nv.arch(node) == "x86_64" and nv.port_range(node, (1, 2)) == (18000, 18099)
    node.policy.update(mode="weird", port_range=[5, 3], external_control=None)
    assert nv.mode(node) == "monitor" and nv.port_range(node, (1, 2)) == (1, 2) and nv.external_control(node) == "none"
    node.policy["port_range"] = [70000, 70001]
    assert nv.port_range(node, (1, 2)) == (1, 2)
    node.hello["host"]["arch"] = None
    assert nv.arch(node) == "x86_64"


def test_host_ports_only_count_what_is_reachable_on_loopback() -> None:
    c = {"ports": [{"host_ip": "0.0.0.0", "host_port": 1}, {"host_ip": "127.0.0.1", "host_port": 2},  # noqa: S104
                   {"host_ip": "10.0.0.5", "host_port": 3}, {"host_ip": "", "host_port": 4}, {"host_ip": "::", "host_port": 5},
                   {"host_ip": "0.0.0.0", "host_port": 1}, "junk", {"host_ip": "0.0.0.0", "host_port": "7"}]}  # noqa: S104
    assert nv.host_ports(c) == [1, 2, 4, 5]


def test_ports_in_use_combines_containers_and_endpoints() -> None:
    node = FakeNode(HostConfig(id="x", label="X"))
    node.add_container("a", host_ports=ports(18000))
    node.add_endpoint(18005)
    assert nv.ports_in_use(node) == {18000, 18005}


def test_numbers_from_untrusted_data() -> None:
    assert nv.num(True) is None and nv.num("1") is None and nv.num(float("nan")) is None and nv.num(3) == 3.0
    assert nv.gib(2 * 1024**3) == 2.0 and nv.gib(None) is None
    assert nv.ok_status(200) and nv.ok_status(299) and not nv.ok_status(300) and not nv.ok_status(True) and not nv.ok_status(None)
    assert nv.auth_status(401) and nv.auth_status(403) and not nv.auth_status(404)
