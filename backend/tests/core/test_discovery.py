"""External engine discovery: classification, redaction, probing safety, monitor-only enforcement."""
from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from engine_console.services.discovery import (
    DiscoveryService,
    classify,
    parse_inspect,
    parse_params,
    redact_argv,
)

from .conftest import Env
from .fakes import FakeAdapter
from .test_api_smoke import P, problem_code, sse_engine

SECRET_ENV = "HF_TOKEN=hf_supersecretenvtoken123456"
SECRET_ARG = "sk-live-argvsecret-987654321"
LEAKS = ("hf_supersecret", "argvsecret", "s3cr3t-key", "Bearer abc")


def ports(host_port: int, container_port: int = 8000, ip: str = "0.0.0.0") -> dict[str, Any]:  # noqa: S104
    return {f"{container_port}/tcp": [{"HostIp": ip, "HostPort": str(host_port)}]}


def models(*ids: str, owned_by: str = "org") -> Any:
    return lambda r: httpx.Response(200, json={"object": "list", "data": [{"id": i, "owned_by": owned_by} for i in ids]})


def serve(env: Env, port: int, *, models_h: Any = None, health: int | None = 200, metrics: str | None = None) -> None:
    if models_h:
        env.port_routes[(port, "/v1/models")] = models_h
    if health is not None:
        env.port_routes[(port, "/health")] = lambda r: httpx.Response(health, text="ok")
    if metrics is not None:
        env.port_routes[(port, "/metrics")] = lambda r: httpx.Response(200, text=metrics)


def add(env: Env, name: str, image: str, argv: list[str], pmap: dict[str, Any] | None = None, **extra: Any) -> None:
    env.runner.containers[name] = {"running": True, "exit": 0, "labels": {}, "image": image, "argv": argv,
                                   "ports": pmap or {}, "env": [SECRET_ENV, "VLLM_API_KEY=s3cr3t-key"], **extra}


def discover(client: TestClient) -> list[dict[str, Any]]:
    r = client.post(f"{P}/instances/discover")
    assert r.status_code == 200, r.text
    return list(r.json()["items"])


def by_name(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {i["name"]: i for i in items}


# ---- signatures (data-driven) ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("image", "cmd", "engine"), [
    ("vllm/vllm-openai:v0.23.0", "", "vllm"), ("registry.example/team/vllm-openai:1", "", "vllm"),
    ("python:3.12", "python -m vllm.entrypoints.openai.api_server --model m", "vllm"),
    ("img", "vllm serve org/m --port 8000", "vllm"),
    ("lmsysorg/sglang:v0.5.14", "", "sglang"), ("img", "python3 -m sglang.launch_server --model-path m", "sglang"),
    ("ollama/ollama:0.5", "", "ollama"), ("img", "/bin/ollama serve", "ollama"),
    ("ghcr.io/ggml-org/llama.cpp:server-cuda", "", "llamacpp"), ("img", "/app/llama-server -m x.gguf", "llamacpp"),
    ("nvcr.io/nvidia/tensorrt-llm/release:1.0", "", "trtllm"), ("img", "trtllm-serve model", "trtllm"),
    ("img", "python -m dynamo.frontend --http-port 8000", "dynamo"),
])
def test_classify_known_engines(image: str, cmd: str, engine: str) -> None:
    assert classify(image, cmd) == engine


@pytest.mark.parametrize(("image", "cmd"), [
    ("ghcr.io/berriai/litellm:main", "litellm --config c.yaml"), ("ghcr.io/open-webui/open-webui:main", ""),
    ("myorg/vllm-router:1", "vllm-router"), ("postgres:16", "postgres"), ("grafana/grafana", ""),
    ("prom/prometheus", ""), ("nginx", ""), ("redis:7", ""), ("busybox", "sleep 1000"),
    ("myorg/some-llm-gateway:1", "gateway --upstream vllm"), ("nvidia/nemo-relay", "")])
def test_classify_never_matches_non_engines(image: str, cmd: str) -> None:
    assert classify(image, cmd) is None


# ---- redaction happens before parsing ----------------------------------------------------------------------------------------
def test_redact_argv_then_parse() -> None:
    raw = ["vllm", "serve", "org/model", "--api-key", SECRET_ARG, "--token=abc123", "--hf-token", "hf_abcdefgh12345",
           "--max-num-batched-tokens", "8192", "--tokenizer", "org/tok", "--port", "8000", "VLLM_API_KEY=zzz",
           "--enable-prefix-caching", "--served-model-name=chat", "Bearer abcdefghijkl"]
    red = redact_argv(raw)
    text = " ".join(red)
    assert SECRET_ARG not in text and "abc123" not in text and "hf_abcdefgh" not in text and "zzz" not in text
    assert "8192" in text and "org/tok" in text                        # "tokens"/"tokenizer" are not secrets
    params = parse_params(red)
    assert params["model"] == "org/model" and params["port"] == "8000" and params["max_num_batched_tokens"] == "8192"
    assert params["api_key"] == "[redacted]" and params["enable_prefix_caching"] is True
    assert params["served_model_name"] == "chat" and SECRET_ARG not in json.dumps(params)


def test_parse_inspect_never_reads_env_and_sanitises_hostile_data() -> None:
    doc = {"Name": "/../../etc; rm -rf $(id)\n", "Path": "vllm", "Args": ["serve", "m", "--api-key", SECRET_ARG],
           "Config": {"Image": "vllm/vllm-openai:1", "Env": [SECRET_ENV],
                      "Labels": {"k" * 500: "v" * 10_000_000, **{f"l{i}": "x" for i in range(200)}}},
           "HostConfig": {"NetworkMode": "bridge"},
           "NetworkSettings": {"Ports": {"8000/tcp": [{"HostIp": "192.168.1.5", "HostPort": "8000"},
                                                      {"HostIp": "127.0.0.1", "HostPort": "notanumber"},
                                                      {"HostIp": "127.0.0.1", "HostPort": "99999"},
                                                      {"HostIp": "127.0.0.1", "HostPort": "18010"}, "junk", None]}}}
    v = parse_inspect(doc)
    assert v is not None
    dump = repr(v)
    assert SECRET_ENV.split("=")[1] not in dump and SECRET_ARG not in dump and "\n" not in v.name
    assert v.ports == [18010] and len(v.labels) <= 50 and all(len(x) <= 200 for x in v.labels.values())
    assert parse_inspect({"Name": ""}) is None


# ---- end to end through the API -----------------------------------------------------------------------------------------------------
def seed_world(env: Env) -> None:
    add(env, "vllm-a", "vllm/vllm-openai:v0.23.0", ["vllm", "serve", "org/big", "--api-key", SECRET_ARG, "--port", "8000"],
        ports(18500))
    serve(env, 18500, models_h=models("org/big"), metrics="running 2\nwaiting 1\nprompt_total 10\ngen_total 20\nkv 33\n")
    add(env, "sgl-b", "lmsysorg/sglang:v0.5.14", ["python3", "-m", "sglang.launch_server", "--model-path", "m"],
        ports(18501, 30000))
    env.port_routes[(18501, "/v1/models")] = lambda r: httpx.Response(401, text="no")
    env.port_routes[(18501, "/health")] = lambda r: httpx.Response(401, text="no")
    add(env, "ollama-c", "ollama/ollama:0.5", ["/bin/ollama", "serve"], ports(18502, 11434))
    serve(env, 18502, models_h=models("llama3", owned_by="library"), health=None)
    add(env, "router", "ghcr.io/berriai/litellm:main", ["litellm"], ports(4000, 4000))
    serve(env, 4000, models_h=models("gpt-alias"), metrics="x 1\n")
    add(env, "vllm-noport", "vllm/vllm-openai:v0.23.0", ["vllm", "serve", "m"])
    add(env, "postgres", "postgres:16", ["postgres"], ports(5432, 5432))
    env.runner.containers["mine"] = {"running": True, "exit": 0, "labels": {"engine-console": "1"}, "image": "vllm/vllm-openai:1",
                                     "argv": ["vllm"], "ports": ports(18503)}
    serve(env, 18503, models_h=models("owned"))
    env.cfg.discovery_ports = [4000, 9100, 9101, 18500, 18503]
    serve(env, 9100, models_h=models("bare-model"), metrics="vllm:num_requests_running 1\n")    # bare port, vllm markers
    env.port_routes[(9101, "/health")] = lambda r: httpx.Response(200, text="ok")               # health only: not an engine


def test_discovery_lists_external_engines_and_ignores_everything_else(client: TestClient, env: Env) -> None:
    seed_world(env)
    items = discover(client)
    got = by_name(items)
    assert set(got) == {"vllm-a", "sgl-b", "ollama-c", "vllm-noport", "127.0.0.1:9100"}, set(got)
    a = got["vllm-a"]
    assert (a["id"], a["managed"], a["source"], a["engine"], a["state"]) == ("ext-vllm-a", False, "external", "vllm", "ready")
    assert a["endpoint"] == "http://127.0.0.1:18500" and a["port"] == 18500 and a["served_models"] == ["org/big"]
    assert a["container_name"] == "vllm-a" and a["image"] == "vllm/vllm-openai:v0.23.0" and a["repo_id"] == "org/big"
    assert a["params"]["model"] == "org/big" and a["params"]["api_key"] == "[redacted]" and a["params"]["port"] == "8000"
    assert isinstance(a["history_since"], float) and a["gpu_ids"] == []
    assert got["sgl-b"]["state"] == "auth_required" and "API key" in got["sgl-b"]["state_reason"]
    assert got["ollama-c"]["engine"] == "ollama" and got["ollama-c"]["served_models"] == ["llama3"]
    assert got["vllm-noport"]["state"] == "unreachable" and "no published host port" in got["vllm-noport"]["state_reason"]
    assert got["vllm-noport"]["port"] is None and got["vllm-noport"]["endpoint"] is None
    bare = got["127.0.0.1:9100"]
    assert bare["engine"] == "vllm" and bare["container_name"] is None and bare["id"] == "ext-127-0-0-1-9100"
    body = json.dumps(items)
    for leak in (*LEAKS, "HF_TOKEN", "VLLM_API_KEY", "sk-live", "hf_super"):
        assert leak not in body, leak
    assert "gpt-alias" not in body and "postgres" not in body                       # router / database never listed


def test_instances_and_get_prefilled_at_startup_and_persist_nothing(client: TestClient, env: Env) -> None:
    seed_world(env)
    discover(client)
    listed = client.get(f"{P}/instances").json()["items"]
    assert {i["id"] for i in listed} >= {"ext-vllm-a", "ext-sgl-b"}
    one = client.get(f"{P}/instances/ext-vllm-a").json()
    assert one["state"] == "ready" and one["managed"] is False
    assert client.get(f"{P}/instances/ext-nope").status_code == 404
    rows = json.dumps([dict(r) for r in env.container.store.all("SELECT * FROM instances")])
    assert "vllm-a" not in rows                                                     # nothing external is persisted


def test_console_owned_instances_are_never_double_listed(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    env.engine_up()
    mine = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B",
                                               "params": {"max_model_len": 4096}}).json()
    # its labelled containers exist in docker and its published port answers like an engine
    serve(env, mine["port"], models_h=models("Qwen/Qwen3-32B"), metrics="running 1\n")
    env.cfg.discovery_ports = [mine["port"]]
    items = discover(client)
    assert [i["id"] for i in items] == [mine["id"]] and items[0]["managed"] is True and items[0]["source"] == "console"


def test_container_disappears_after_one_stopped_cycle(client: TestClient, env: Env) -> None:
    seed_world(env)
    discover(client)
    env.runner.containers.pop("vllm-a")
    got = by_name(discover(client))
    assert got["vllm-a"]["state"] == "stopped" and "no longer running" in got["vllm-a"]["state_reason"]
    assert "vllm-a" not in by_name(discover(client))


def test_disabled_discovery_lists_nothing_and_never_calls_docker_ps(client: TestClient, env: Env) -> None:
    seed_world(env)
    env.cfg.discovery_enabled = False
    n = len(env.runner.calls)
    assert discover(client) == []
    assert not any(c[3] == "ps" for c in env.runner.calls[n:])


def test_supervisor_tick_rediscovers_on_interval(client: TestClient, env: Env) -> None:
    seed_world(env)
    env.cfg.discovery_interval_s = 0.0
    client.portal.call(env.container.lifecycle.tick)  # type: ignore[union-attr]
    assert "ext-vllm-a" in {i["id"] for i in client.get(f"{P}/instances").json()["items"]}


# ---- monitor-only enforcement ---------------------------------------------------------------------------------------------------------------
def test_every_mutating_instance_route_is_409_for_external_engines(client: TestClient, env: Env) -> None:
    seed_world(env)
    discover(client)
    paths = [(m.upper(), p) for p, ops in client.app.openapi()["paths"].items() for m in ops  # type: ignore[attr-defined]
             if "/instances/{iid}" in p and m.upper() in ("POST", "PUT", "PATCH", "DELETE")]
    assert {m for m, _ in paths} == {"POST", "PATCH", "DELETE"} and len(paths) >= 5
    for method, path in paths:
        r = client.request(method, path.replace("{iid}", "ext-vllm-a"), json={"pinned": True})
        assert r.status_code == 409 and problem_code(r) == "instance_not_managed", (method, path, r.text)
    for path in ("logs", "logs/stream", "command"):
        r = client.get(f"{P}/instances/ext-vllm-a/{path}")
        assert r.status_code == 409 and problem_code(r) == "instance_not_managed", path
    assert not any(c[3] in ("stop", "rm", "run", "create", "start") for c in env.runner.calls)   # docker untouched


def test_viewer_cannot_force_discovery(env: Env) -> None:
    from engine_console.main import create_app
    with TestClient(create_app(env.cfg, env.container), client=("192.168.1.9", 1)) as remote:
        boot = (env.cfg.data_dir / "bootstrap-admin.key").read_text().strip()
        v = remote.post(f"{P}/keys", json={"name": "v", "role": "viewer"}, headers={"Authorization": f"Bearer {boot}"}).json()["secret"]
        assert remote.post(f"{P}/instances/discover", headers={"Authorization": f"Bearer {v}"}).status_code == 403


# ---- metrics, chat, bench on external engines ------------------------------------------------------------------------------------------------
def test_metrics_scraped_for_ready_external_only_with_history_since(client: TestClient, env: Env) -> None:
    fake_vllm = FakeAdapter()
    fake_vllm.id = "vllm"                         # stands in for the real adapter's parse_metrics
    env.container.adapters.register(fake_vllm)
    seed_world(env)
    discover(client)
    client.portal.call(env.container.metrics.scrape_once)  # type: ignore[union-attr]
    ts = client.get(f"{P}/metrics/instances/ext-vllm-a").json()
    assert len(ts["points"]) == 1 and ts["points"][0]["values"]["requests_running"] == 2
    assert ts["history_since"] == client.get(f"{P}/instances/ext-vllm-a").json()["history_since"]
    assert client.get(f"{P}/metrics/instances/ext-sgl-b").json()["points"] == []           # auth_required: no scraping
    assert client.get(f"{P}/metrics/instances/ext-ollama-c").json()["points"] == []        # no /metrics: nothing fabricated


def test_chat_and_bench_can_target_external_with_confirmation(client: TestClient, env: Env) -> None:
    seed_world(env)
    discover(client)
    env.port_routes[(18500, "/v1/chat/completions")] = sse_engine
    r = client.post(f"{P}/chat/completions", json={"instance_id": "ext-vllm-a", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and json.loads(env.engine_requests[-1].content)["model"] == "org/big"
    assert client.post(f"{P}/chat/completions", json={"instance_id": "ext-sgl-b", "messages": [{"role": "user", "content": "x"}]}).status_code == 409
    no = client.post(f"{P}/bench", json={"instance_id": "ext-vllm-a"})
    assert no.status_code == 400 and problem_code(no) == "confirm_external_required"
    assert client.post(f"{P}/bench", json={"instance_id": "ext-sgl-b", "confirm_external": True}).status_code == 409   # not ready
    ok = client.post(f"{P}/bench", json={"instance_id": "ext-vllm-a", "confirm_external": True,
                                          "suite": {"concurrency": [1], "prompt_tokens": 14, "max_tokens": 2, "rounds": 1}})
    assert ok.status_code == 202 and "external engine" in ok.json()["suite"]["note"]
    client.portal.call(env.container.bench.wait, ok.json()["id"])  # type: ignore[union-attr]
    assert client.get(f"{P}/bench/{ok.json()['id']}").json()["state"] == "completed"


# ---- hostile inputs ---------------------------------------------------------------------------------------------------------------------------
def test_hostile_container_name_and_huge_labels_are_neutralised(client: TestClient, env: Env) -> None:
    hostile = "../../etc; rm -rf $(id) `x`"
    add(env, hostile, "vllm/vllm-openai:1", ["vllm", "serve", "m"], ports(18600), labels={"a": "b" * 5_000_000})
    serve(env, 18600, models_h=models("m\x00\x1b[31m" + "x" * 1000))
    items = discover(client)
    ext = next(i for i in items if i["source"] == "external")
    assert re.fullmatch(r"ext-[a-z0-9-]{1,80}", ext["id"]) and ".." not in ext["id"] and "/" not in ext["id"]
    assert re.fullmatch(r"[A-Za-z0-9 _.\-]{1,64}", ext["name"])
    assert all("\x00" not in m and "\x1b" not in m and len(m) <= 200 for m in ext["served_models"])
    assert client.get(f"{P}/instances/{ext['id']}").status_code == 200


def test_colliding_slugs_get_distinct_stable_ids(client: TestClient, env: Env) -> None:
    for n in ("My Engine", "my-engine"):
        add(env, n, "vllm/vllm-openai:1", ["vllm"], ports(18700 if n == "My Engine" else 18701))
    serve(env, 18700, models_h=models("a"))
    serve(env, 18701, models_h=models("b"))
    ids = [i["id"] for i in discover(client)]
    assert len(ids) == len(set(ids)) == 2
    assert ids == [i["id"] for i in discover(client)]                              # stable across refreshes


def test_redirecting_endpoint_is_not_followed_and_not_listed(client: TestClient, env: Env) -> None:
    add(env, "redir", "vllm/vllm-openai:1", ["vllm"], ports(18800))
    for path in ("/v1/models", "/health", "/metrics"):
        env.port_routes[(18800, path)] = lambda r: httpx.Response(302, headers={"location": "http://169.254.169.254/latest"})
    env.cfg.discovery_ports = [18801]
    env.port_routes[(18801, "/v1/models")] = lambda r: httpx.Response(307, headers={"location": "http://127.0.0.1:18500/v1/models"})
    env.port_routes[(18801, "/health")] = lambda r: httpx.Response(307, headers={"location": "http://127.0.0.1:18500/health"})
    items = discover(client)
    assert [i["state"] for i in items] == ["unreachable"] and items[0]["name"] == "redir"    # signature-matched, but not answering
    assert not any(r.url.host != "127.0.0.1" or r.method != "GET" for r in env.engine_requests)
    assert not any(r.url.port == 18500 for r in env.engine_requests)                          # the redirect target was never fetched


async def test_probe_is_get_only_loopback_only_and_body_capped(env: Env) -> None:
    seen: list[httpx.Request] = []

    def big(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return httpx.Response(200, content=b"x" * (3 * 1024 * 1024))

    http = httpx.AsyncClient(transport=httpx.MockTransport(big))
    svc = DiscoveryService(env.container.docker, http, env.cfg)
    pr = await svc.probe_port(18900)
    assert len(pr.models.body) == 1024 * 1024 and pr.model_names() == []            # capped, unparsable, no crash
    assert {r.method for r in seen} == {"GET"} and {r.url.host for r in seen} == {"127.0.0.1"}
    assert {r.url.path for r in seen} == {"/v1/models", "/health", "/metrics"}
    assert all(r.extensions.get("timeout") for r in seen)                          # short timeouts on every probe
    await http.aclose()


async def test_unreachable_port_yields_no_instance(env: Env) -> None:
    def boom(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    svc = DiscoveryService(env.container.docker, httpx.AsyncClient(transport=httpx.MockTransport(boom)), env.cfg)
    env.cfg.discovery_ports = [8000, 30000]
    assert await svc.refresh() == []
