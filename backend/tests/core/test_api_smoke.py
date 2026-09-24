"""One smoke test per router through the real FastAPI app (TestClient), with every port faked."""
from __future__ import annotations

import json
import time
from typing import Any

import httpx
from fastapi.testclient import TestClient

from .conftest import QWEN_CFG, Env

P = "/api/v1"


def problem_code(r: httpx.Response) -> str:
    assert r.headers["content-type"].startswith("application/problem+json"), r.text
    body = r.json()
    assert body["status"] == r.status_code
    return str(body["code"])


def tick(client: TestClient, env: Env, n: int = 1) -> None:
    for _ in range(n):
        client.portal.call(env.container.lifecycle.tick)  # type: ignore[union-attr]


def make_ready_instance(client: TestClient, env: Env, name: str = "qwen", repo: str = "Qwen/Qwen3-32B") -> dict[str, Any]:
    if not env.container.downloads.is_cached(repo):
        env.seed_local_model(repo)
    env.engine_up()
    r = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": repo, "name": name,
                                            "params": {"max_model_len": 8192}, "gpu_ids": [0] if name == "qwen" else [1]})
    assert r.status_code == 201, r.text
    inst = r.json()
    env.runner.logs[inst["container_name"]] = "loading weights 40%\n"
    tick(client, env)
    assert client.get(f"{P}/instances/{inst['id']}").json()["state"] == "ready"
    return dict(client.get(f"{P}/instances/{inst['id']}").json())


# ---- system ------------------------------------------------------------------------------------
def test_health_hardware_engines(client: TestClient) -> None:
    assert client.get(f"{P}/health").json()["status"] == "ok"
    hwr = client.get(f"{P}/hardware").json()
    assert [g["uuid"] for g in hwr["gpus"]] == ["GPU-uuid-0", "GPU-uuid-1"]
    assert hwr["source"] == "fake"
    eng = client.get(f"{P}/engines").json()
    assert eng[0]["id"] == "fake" and eng[0]["presets"][0]["name"] == "balanced"
    params = client.get(f"{P}/engines/fake/params").json()
    assert {p["key"] for p in params} == {"max_model_len", "tp", "mem_fraction", "kv_dtype", "api_key", "leaky_key", "image", "trust_remote_code"}
    assert problem_code(client.get(f"{P}/engines/nope/params")) == "unknown_engine"


def test_hf_search_detail_and_fit(client: TestClient, env: Env) -> None:
    env.hub.add("Qwen/Qwen3-32B", {"config.json": b"{}", "model.safetensors": b"x" * 100}, QWEN_CFG,
                params={"BF16": 32_762_000_000})
    s = client.get(f"{P}/hf/search", params={"q": "qwen", "limit": 5}).json()
    assert s["items"][0]["repo_id"] == "Qwen/Qwen3-32B"
    assert s["items"][0]["approx_size_gib"] == round(32_762_000_000 * 2 / 1024**3, 2)
    assert client.get(f"{P}/hf/search", params={"q": "qwen", "max_gib": 1}).json()["items"] == []
    d = client.get(f"{P}/hf/models/Qwen/Qwen3-32B").json()
    assert d["info"]["num_layers"] == 64 and d["info"]["num_kv_heads"] == 8 and "model card" in d["card"]
    assert problem_code(client.get(f"{P}/hf/models/does/not-exist")) == "not_found"
    r = client.post(f"{P}/fit", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 8192}})
    assert r.status_code == 200, r.text
    fit = r.json()
    assert fit["verdict"] in ("fits", "tight") and fit["per_gpu"][0]["uuid"] == "GPU-uuid-0"
    bad = client.post(f"{P}/fit", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"bogus": 1}})
    assert bad.status_code == 422 and problem_code(bad) == "invalid_params"
    assert bad.json()["errors"] == ["unknown parameter 'bogus'"]
    env.hub.gated.add("meta/gated")
    assert problem_code(client.get(f"{P}/hf/models/meta/gated")) == "hf_gated"


# ---- downloads / models ----------------------------------------------------------------------------
def test_downloads_and_models(client: TestClient, env: Env) -> None:
    env.hub.add("org/tiny", {"config.json": json.dumps(QWEN_CFG).encode(), "model.safetensors": b"w" * 64}, QWEN_CFG)
    r = client.post(f"{P}/downloads", json={"repo_id": "org/tiny"})
    assert r.status_code == 201
    did = r.json()["id"]
    client.portal.call(env.container.downloads.wait, did)  # type: ignore[union-attr]
    d = client.get(f"{P}/downloads/{did}").json()
    assert d["state"] == "completed" and d["done_bytes"] == d["total_bytes"] > 0
    assert client.get(f"{P}/downloads").json()["items"][0]["id"] == did
    snap = client.get(f"{P}/downloads/stream", params={"once": "true"})
    assert snap.headers["content-type"].startswith("text/event-stream") and "event: snapshot" in snap.text
    models = client.get(f"{P}/models").json()["items"]
    assert models[0]["repo_id"] == "org/tiny" and models[0]["engines_that_fit"] == ["fake"]
    assert client.post(f"{P}/downloads/{did}/pause").status_code == 409
    assert client.delete(f"{P}/downloads/{did}").status_code == 204
    assert client.delete(f"{P}/models/org/tiny").status_code == 204
    assert client.get(f"{P}/models").json()["items"] == []
    assert problem_code(client.delete(f"{P}/models/org/tiny")) == "not_found"
    assert problem_code(client.post(f"{P}/downloads", json={"repo_id": "../etc"})) == "validation_error"


# ---- instances -------------------------------------------------------------------------------------
def test_instance_lifecycle_end_to_end(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    r = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 8192}})
    assert r.status_code == 201, r.text
    inst = r.json()
    assert inst["state"] == "starting" and 18000 <= inst["port"] <= 18099 and inst["gpu_uuids"] == ["GPU-uuid-0"]
    run = next(c for c in env.runner.calls if c[3] == "run")
    assert run[:4] == ["docker", "--context", "rootless", "run"]
    assert run[run.index("--gpus") + 1] == "device=GPU-uuid-0"
    assert "-p" not in run and "ai-lab.console=1" in run       # engines publish nothing; the gateway does
    assert run[run.index("--network") + 1] == "ai-lab-engines"
    assert "FAKE_API_KEY=s3cr3t-key-value" not in " ".join(run)      # secret env is by name only
    assert "FAKE_API_KEY" in run
    run_env = env.runner.envs[env.runner.calls.index(run)]
    assert run_env is not None and run_env["FAKE_API_KEY"] == "s3cr3t-key-value"
    assert "HF_HUB_OFFLINE=1" in run
    name = inst["container_name"]
    # loading -> ready
    env.runner.logs[name] = "boot\nloading weights 55%\n"
    tick(client, env)
    got = client.get(f"{P}/instances/{inst['id']}").json()
    assert got["state"] == "loading" and got["progress_pct"] == 55.0
    env.engine_up()
    tick(client, env)
    assert client.get(f"{P}/instances/{inst['id']}").json()["state"] == "ready"
    assert client.get(f"{P}/instances").json()["items"][0]["state"] == "ready"
    # logs are scrubbed of the secret value
    env.runner.logs[name] = "args: api_key=s3cr3t-key-value token hf_abcdefghijkl1234\n"
    logs = client.get(f"{P}/instances/{inst['id']}/logs").text
    assert "s3cr3t" not in logs and "hf_abcdef" not in logs and "[redacted]" in logs
    once = client.get(f"{P}/instances/{inst['id']}/logs/stream", params={"once": "true"})
    assert "event: log" in once.text and "s3cr3t" not in once.text
    cmd = client.get(f"{P}/instances/{inst['id']}/command").json()
    assert "docker run" in cmd["docker_run"] and "s3cr3t" not in json.dumps(cmd) and "--model" in cmd["engine_cli"]
    assert "GPU-uuid-0" in cmd["compose_yaml"]
    assert client.patch(f"{P}/instances/{inst['id']}", json={"pinned": True}).json()["pinned"] is True
    # stop / start / restart
    assert client.post(f"{P}/instances/{inst['id']}/stop").json()["state"] == "stopped"
    assert client.post(f"{P}/instances/{inst['id']}/start").json()["state"] == "starting"
    assert client.post(f"{P}/instances/{inst['id']}/restart").json()["state"] == "starting"
    # crash detection captures logs
    env.runner.logs[name] = "CUDA error: out of memory\n"
    env.runner.crash(name)
    tick(client, env)
    failed = client.get(f"{P}/instances/{inst['id']}").json()
    assert failed["state"] == "failed" and "code 137" in failed["error"] and "out of memory" in failed["last_logs"]
    assert client.delete(f"{P}/instances/{inst['id']}").status_code == 204
    assert problem_code(client.get(f"{P}/instances/{inst['id']}")) == "not_found"


def test_instance_preflight_blocks(client: TestClient, env: Env) -> None:
    body = {"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {}}
    r = client.post(f"{P}/instances", json=body)
    assert r.status_code == 409 and problem_code(r) == "preflight_failed"
    codes = {c["code"]: c["level"] for c in r.json()["preflight"]["checks"]}
    assert codes["model_not_cached"] == "block"
    env.seed_local_model()
    env.runner.missing_images.add("fake/engine:1.0")
    pf = client.post(f"{P}/instances/preflight", json=body).json()
    assert pf["ok"] is False and {c["code"] for c in pf["checks"] if c["level"] == "block"} == {"image_missing"}
    env.runner.missing_images.clear()
    assert client.post(f"{P}/instances/preflight", json={**body, "params": {"tp": 1, "max_model_len": 4096}}).json()["ok"]
    env.probe.free = 10.0   # other tenants ate the VRAM
    pf = client.post(f"{P}/instances/preflight", json=body).json()
    assert pf["ok"] is False and pf["fit"]["verdict"] == "wont_fit"
    assert problem_code(client.post(f"{P}/instances", json={**body, "params": {"nope": 1}})) == "preflight_failed"


def test_docker_failure_marks_failed(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    env.runner.fail_run = "no such gpu"
    r = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B"})
    assert r.status_code == 502 and problem_code(r) == "docker_error"
    assert client.get(f"{P}/instances").json()["items"][0]["state"] == "failed"


def test_two_instances_get_distinct_ports_and_ttl_stop(client: TestClient, env: Env) -> None:
    a = make_ready_instance(client, env, "qwen")
    b = make_ready_instance(client, env, "qwen-b")
    assert a["port"] != b["port"] and a["gpu_ids"] == [0] and b["gpu_ids"] == [1]
    client.patch(f"{P}/instances/{a['id']}", json={"ttl_idle_s": 1})
    env.container.store.execute("UPDATE instances SET last_request_at=? WHERE id=?", (time.time() - 100, a["id"]))
    tick(client, env)
    assert client.get(f"{P}/instances/{a['id']}").json()["state"] == "stopped"
    assert client.get(f"{P}/instances/{b['id']}").json()["state"] == "ready"


# ---- profiles ----------------------------------------------------------------------------------------
def test_profiles_crud_diff_yaml(client: TestClient) -> None:
    r = client.post(f"{P}/profiles", json={"name": "long", "engine": "fake", "preset": "long-context",
                                           "params": {"tp": 2}})
    assert r.status_code == 201, r.text
    a = r.json()
    assert a["params"] == {"mem_fraction": 0.9, "max_model_len": 8192, "tp": 2}
    b = client.post(f"{P}/profiles", json={"name": "fast", "engine": "fake", "params": {"tp": 1, "kv_dtype": "fp8"}}).json()
    assert problem_code(client.post(f"{P}/profiles", json={"name": "long", "engine": "fake"})) == "profile_exists"
    assert problem_code(client.post(f"{P}/profiles", json={"name": "x", "engine": "fake", "preset": "zzz"})) == "unknown_preset"
    assert client.post(f"{P}/profiles", json={"name": "x", "engine": "fake", "params": {"bad": 1}}).status_code == 422
    d = client.get(f"{P}/profiles/{a['id']}/diff/{b['id']}").json()
    assert d["changed"] == {"tp": {"a": 2, "b": 1}}
    assert d["only_a"] == {"mem_fraction": 0.9, "max_model_len": 8192} and d["only_b"] == {"kv_dtype": "fp8"}
    y = client.get(f"{P}/profiles/{a['id']}/export")
    assert "max_model_len: 8192" in y.text
    imp = client.post(f"{P}/profiles/import", content=y.text.replace("name: long", "name: long2"),
                      headers={"content-type": "text/yaml"})
    assert imp.status_code == 201 and imp.json()["name"] == "long2"
    assert problem_code(client.post(f"{P}/profiles/import", content="- just\n- a list", headers={"content-type": "text/yaml"})) == "invalid_yaml"
    assert client.put(f"{P}/profiles/{b['id']}", json={"name": "fast2", "engine": "fake", "params": {"tp": 1}}).json()["name"] == "fast2"
    assert len(client.get(f"{P}/profiles", params={"limit": 2}).json()["items"]) == 2
    assert client.get(f"{P}/profiles", params={"limit": 1}).json()["next_cursor"] == "1"
    assert client.delete(f"{P}/profiles/{b['id']}").status_code == 204
    assert problem_code(client.get(f"{P}/profiles/{b['id']}/export")) == "not_found"


def test_instance_from_profile(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    p = client.post(f"{P}/profiles", json={"name": "p", "engine": "fake", "params": {"max_model_len": 4096}}).json()
    inst = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "profile_id": p["id"]}).json()
    assert inst["params"] == {"max_model_len": 4096} and inst["profile_id"] == p["id"]


# ---- metrics / prometheus --------------------------------------------------------------------------------
def test_metrics_series_stream_and_prometheus(client: TestClient, env: Env) -> None:
    inst = make_ready_instance(client, env)
    body = iter(["running 2\nwaiting 1\nprompt_total 100\ngen_total 50\nkv 12.5\n",
                 "running 3\nwaiting 0\nprompt_total 300\ngen_total 150\nkv 20\nunknown_key 9\n"])
    env.engine_routes["/metrics"] = lambda r: httpx.Response(200, text=next(body))
    client.portal.call(env.container.metrics.scrape_once)  # type: ignore[union-attr]
    time.sleep(0.05)
    client.portal.call(env.container.metrics.scrape_once)  # type: ignore[union-attr]
    ts = client.get(f"{P}/metrics/instances/{inst['id']}", params={"window": "15m"}).json()
    assert ts["resolution"] == "raw" and len(ts["points"]) == 2
    last = ts["points"][-1]["values"]
    assert last["requests_running"] == 3 and last["kv_cache_usage_pct"] == 20
    assert last["generation_tps"] > 0 and "unknown_key" not in last   # derived from counter deltas
    hourly = client.get(f"{P}/metrics/instances/{inst['id']}", params={"window": "1h"}).json()
    assert hourly["resolution"] == "1h" and hourly["points"][0]["values"]["requests_running"] == 2.5
    assert problem_code(client.get(f"{P}/metrics/instances/{inst['id']}", params={"window": "soon"})) == "invalid_window"
    assert "event: metrics" in client.get(f"{P}/metrics/stream", params={"once": "true"}).text
    prom = client.get("/metrics")
    assert prom.status_code == 200 and 'engine_console_instances{state="ready"} 1.0' in prom.text
    assert "engine_console_http_requests_total" in prom.text


# ---- bench ------------------------------------------------------------------------------------------------
def sse_engine(request: httpx.Request) -> httpx.Response:
    req = json.loads(request.content)
    chunks = [{"choices": [{"delta": {"content": w}}]} for w in ("Hel", "lo", " there")]
    chunks.append({"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 3}})
    if req.get("stream"):
        text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json={"choices": [{"message": {"content": "Hello there", "reasoning_content": "hmm"}}],
                                     "usage": {"prompt_tokens": 20, "completion_tokens": 3}})


def test_bench(client: TestClient, env: Env) -> None:
    inst = make_ready_instance(client, env)
    env.engine_routes["/v1/chat/completions"] = sse_engine
    r = client.post(f"{P}/bench", json={"instance_id": inst["id"], "suite": {"concurrency": [1, 2], "prompt_tokens": 28,
                                                                             "max_tokens": 4, "rounds": 1, "prefix_cache": True}})
    assert r.status_code == 202, r.text
    bid = r.json()["id"]
    client.portal.call(env.container.bench.wait, bid)  # type: ignore[union-attr]
    run = client.get(f"{P}/bench/{bid}").json()
    assert run["state"] == "completed", run
    assert [lv["concurrency"] for lv in run["results"]["concurrency"]] == [1, 2]
    assert run["results"]["single"]["errors"] == 0 and run["results"]["single"]["ttft_p50_s"] is not None
    assert run["results"]["concurrency"][1]["throughput_tps"] > 0 and "speedup" in run["results"]["prefix_cache"]
    assert client.get(f"{P}/bench").json()["items"][0]["id"] == bid
    assert "event: snapshot" in client.get(f"{P}/bench/{bid}/stream", params={"once": "true"}).text
    assert problem_code(client.post(f"{P}/bench", json={"instance_id": inst["id"], "suite": "nope"})) == "unknown_suite"
    client.post(f"{P}/instances/{inst['id']}/stop")
    assert problem_code(client.post(f"{P}/bench", json={"instance_id": inst["id"]})) == "instance_not_ready"


# ---- chat / arena / usage ------------------------------------------------------------------------------------
def test_chat_conversations_prompts(client: TestClient, env: Env) -> None:
    inst = make_ready_instance(client, env)
    env.engine_routes["/v1/chat/completions"] = sse_engine
    conv = client.post(f"{P}/conversations", json={"title": "t", "instance_id": inst["id"]}).json()
    msgs = [{"role": "user", "content": "hi"}]
    r = client.post(f"{P}/chat/completions", json={"instance_id": inst["id"], "conversation_id": conv["id"],
                                                   "messages": msgs})
    assert r.json()["choices"][0]["message"]["content"] == "Hello there"
    with client.stream("POST", f"{P}/chat/completions", json={"instance_id": inst["id"], "conversation_id": conv["id"],
                                                             "messages": msgs, "stream": True}) as s:
        assert s.headers["content-type"].startswith("text/event-stream")
        text = "".join(s.iter_text())
    assert "Hel" in text and "[DONE]" in text
    forwarded = json.loads(env.engine_requests[-1].content)
    assert forwarded["model"] == "Qwen/Qwen3-32B" and "instance_id" not in forwarded
    assert forwarded["stream_options"] == {"include_usage": True}
    full = client.get(f"{P}/conversations/{conv['id']}").json()
    assert [m["role"] for m in full["messages"]] == ["user", "assistant", "user", "assistant"]
    assert full["messages"][3]["content"] == "Hello there" and full["messages"][1]["reasoning"] == "hmm"
    assert client.get(f"{P}/conversations").json()["items"][0]["id"] == conv["id"]
    assert client.delete(f"{P}/conversations/{conv['id']}").status_code == 204
    assert problem_code(client.post(f"{P}/chat/completions", json={"messages": msgs})) == "missing_instance_id"
    assert problem_code(client.post(f"{P}/chat/completions", json={"instance_id": "nope", "messages": msgs})) == "not_found"
    u = client.get(f"{P}/usage", params={"group_by": "model"}).json()
    assert u[0]["key"] == "Qwen/Qwen3-32B" and u[0]["requests"] == 2 and u[0]["prompt_tokens"] == 40
    assert client.get(f"{P}/usage", params={"group_by": "day"}).status_code == 200
    assert client.get(f"{P}/usage", params={"group_by": "hour"}).json()[0]["key"].endswith(":00Z")
    assert problem_code(client.get(f"{P}/usage", params={"group_by": "x"})) == "invalid_group_by"
    csv = client.get(f"{P}/usage/export.csv")
    assert csv.headers["content-type"].startswith("text/csv") and csv.text.splitlines()[0].startswith("key,requests")
    pr = client.post(f"{P}/prompts", json={"title": "Sys", "content": "You are terse.", "tags": ["a"]}).json()
    assert client.put(f"{P}/prompts/{pr['id']}", json={"title": "Sys2", "content": "c"}).json()["title"] == "Sys2"
    assert client.get(f"{P}/prompts").json()["items"][0]["title"] == "Sys2"
    assert client.delete(f"{P}/prompts/{pr['id']}").status_code == 204


def test_arena_blind_vote_and_elo(client: TestClient, env: Env) -> None:
    a = make_ready_instance(client, env, "qwen")
    b = make_ready_instance(client, env, "qwen-b", "org/other")
    env.engine_routes["/v1/chat/completions"] = sse_engine
    m = client.post(f"{P}/arena/matches", json={"prompt": "hello", "instance_a": a["id"], "instance_b": b["id"]}).json()
    assert m["blind"] and m["model_a"] is None and m["response_a"] == "Hello there"
    v = client.post(f"{P}/arena/matches/{m['id']}/vote", json={"winner": "a"}).json()
    assert v["model_a"] == "Qwen/Qwen3-32B" and v["winner"] == "a"
    assert problem_code(client.post(f"{P}/arena/matches/{m['id']}/vote", json={"winner": "b"})) == "already_voted"
    lb = client.get(f"{P}/arena/leaderboard").json()
    assert (lb[0]["model"], lb[0]["rating"], lb[0]["wins"]) == ("Qwen/Qwen3-32B", 1016.0, 1)
    assert (lb[1]["model"], lb[1]["rating"], lb[1]["losses"]) == ("org/other", 984.0, 1)
    same = make_ready_instance(client, env, "qwen-c")
    m2 = client.post(f"{P}/arena/matches", json={"prompt": "x", "instance_a": a["id"], "instance_b": same["id"]}).json()
    client.post(f"{P}/arena/matches/{m2['id']}/vote", json={"winner": "a"})
    assert client.get(f"{P}/arena/leaderboard").json()[0]["rating"] == 1016.0   # self-play leaves ratings alone
    assert problem_code(client.post(f"{P}/arena/matches", json={"prompt": "x", "instance_a": a["id"], "instance_b": a["id"]})) == "same_instance"


# ---- admin: settings, keys, audit, auth ---------------------------------------------------------------------------
def test_settings_keys_audit(client: TestClient, env: Env) -> None:
    s = client.get(f"{P}/settings").json()
    assert s["hf_token"] == "[unset]" and s["docker_context"] == "rootless"
    upd = client.put(f"{P}/settings", json={"idle_ttl_s": 600, "image_pins": {"fake": "fake/engine:2.0"}}).json()
    assert upd["idle_ttl_s"] == 600 and upd["image_pins"] == {"fake": "fake/engine:2.0"}
    assert problem_code(client.put(f"{P}/settings", json={"image_pins": {"fake": "fake/engine:latest"}})) == "invalid_setting"
    assert client.get(f"{P}/engines").json()[0]["default_image"] == "fake/engine:2.0"
    k = client.post(f"{P}/keys", json={"name": "ci", "role": "viewer"}).json()
    assert k["secret"].startswith("ec_") and len(k["secret"]) > 30
    listed = client.get(f"{P}/keys").json()
    assert all(x["secret"] is None for x in listed) and {x["name"] for x in listed} == {"bootstrap-admin", "ci"}
    assert client.delete(f"{P}/keys/{k['id']}").status_code == 204
    assert problem_code(client.delete(f"{P}/keys/{k['id']}")) == "not_found"
    audit = client.get(f"{P}/audit").json()["items"]
    assert audit[0]["method"] == "DELETE" and audit[0]["actor"] == "loopback"
    creates = [a for a in audit if a["path"].endswith("/keys") and a["method"] == "POST"]
    assert creates and json.dumps(audit).count(k["secret"]) == 0     # secrets never reach the audit log
    put = next(a for a in audit if a["method"] == "PUT" and a["status"] == 200)
    assert put["params"]["body"]["idle_ttl_s"] == 600


def test_hf_token_status_never_leaks(env: Env, client: TestClient) -> None:
    env.cfg.hf_token = "hf_supersecrettokenvalue123"
    text = json.dumps(client.get(f"{P}/settings").json())
    assert "[set, 27 chars]" in text and "supersecret" not in text


def test_auth_roles_for_remote_clients(env: Env) -> None:
    from engine_console.main import create_app
    app = create_app(env.cfg, env.container)
    with TestClient(app, client=("192.168.1.9", 4000)) as remote:
        assert remote.get(f"{P}/health").status_code == 200            # liveness is open
        r = remote.get(f"{P}/instances")
        assert r.status_code == 401 and problem_code(r) == "unauthorized"
        assert remote.get("/metrics").status_code == 401
        boot = (env.cfg.data_dir / "bootstrap-admin.key").read_text().strip()
        assert oct((env.cfg.data_dir / "bootstrap-admin.key").stat().st_mode & 0o777) == "0o600"
        adm = {"Authorization": f"Bearer {boot}"}
        assert remote.get(f"{P}/instances", headers=adm).status_code == 200
        viewer = remote.post(f"{P}/keys", json={"name": "v", "role": "viewer"}, headers=adm).json()["secret"]
        vh = {"Authorization": f"Bearer {viewer}"}
        assert remote.get(f"{P}/instances", headers=vh).status_code == 200
        forbid = remote.post(f"{P}/profiles", json={"name": "n", "engine": "fake"}, headers=vh)
        assert forbid.status_code == 403 and problem_code(forbid) == "forbidden"
        assert remote.get(f"{P}/keys", headers=vh).status_code == 403     # admin-only read
        assert remote.get(f"{P}/instances", headers={"Authorization": "Bearer ec_wrong"}).status_code == 401
        assert remote.post(f"{P}/fit", json={"engine": "fake", "repo_id": "a/b"}, headers=vh).status_code != 403
        denied = [a for a in env.container.audit.list_page(50, None).items if a["status"] in (401, 403)]
        assert denied, "denied mutating attempts are audited too"


def test_x_forwarded_for_disables_loopback_trust(client: TestClient) -> None:
    assert client.get(f"{P}/instances", headers={"X-Forwarded-For": "8.8.8.8"}).status_code == 401


# ---- static + errors -------------------------------------------------------------------------------------------
def test_static_spa_and_error_shapes(client: TestClient) -> None:
    assert "console" in client.get("/").text
    assert client.get("/js/app.js").text == "console.log(1)"
    assert client.get("/models/deep/link").status_code == 404           # only index + asset dirs are served
    assert client.get("/..%2f..%2fetc/passwd").status_code == 404
    nf = client.get(f"{P}/nope")
    assert nf.status_code == 404 and problem_code(nf) == "not_found"
    v = client.post(f"{P}/fit", json={"engine": 5})
    assert v.status_code == 422 and problem_code(v) == "validation_error" and "input" not in json.dumps(v.json())
    assert client.get(f"{P}/openapi.json").json()["info"]["title"] == "Engine Console"
    assert client.get("/api/docs").status_code == 404
