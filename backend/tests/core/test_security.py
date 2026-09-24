"""Regression tests for the security review findings (one block per finding)."""
from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import ValidationError

from engine_console.config import Settings
from engine_console.domain.errors import Forbidden
from engine_console.domain.models import DownloadRequest
from engine_console.domain.repo_id import check_repo_id
from engine_console.main import create_app
from engine_console.services.docker import DockerCli
from engine_console.services.hf_http import HfHttpClient

from .conftest import Env
from .test_api_smoke import P, make_ready_instance, problem_code, sse_engine

SHA = "c0ffee00000000000000000000000000000000ff"


# ---- 1. Host allowlist, Origin, CSRF header, content-type ----------------------------------------------------
def test_host_allowlist_returns_421(client: TestClient) -> None:
    assert client.get(f"{P}/health", headers={"Host": "evil.example:8791"}).status_code == 421
    assert client.get("/", headers={"Host": "attacker.test"}).status_code == 421      # static too (rebinding)
    assert client.get(f"{P}/health", headers={"Host": "localhost:8791"}).status_code == 200
    assert client.get(f"{P}/health", headers={"Host": "127.0.0.1:8791"}).status_code == 200


def test_unsafe_methods_need_custom_header_and_same_origin(env: Env) -> None:
    app = create_app(env.cfg, env.container)
    with TestClient(app, client=("127.0.0.1", 1)) as bare:      # no default X-Engine-Console header
        body = {"name": "n", "engine": "fake"}
        r = bare.post(f"{P}/profiles", json=body)
        assert r.status_code == 403 and problem_code(r) == "csrf_header_required"
        ok = bare.post(f"{P}/profiles", json=body, headers={"X-Engine-Console": "1"})
        assert ok.status_code == 201
        cross = bare.post(f"{P}/profiles", json={**body, "name": "m"},
                          headers={"X-Engine-Console": "1", "Origin": "https://evil.example"})
        assert cross.status_code == 403 and problem_code(cross) == "cross_origin"
        null = bare.delete(f"{P}/profiles/{ok.json()['id']}", headers={"X-Engine-Console": "1", "Origin": "null"})
        assert null.status_code == 403
        same = bare.post(f"{P}/profiles", json={**body, "name": "s"},
                         headers={"X-Engine-Console": "1", "Origin": "http://testserver"})
        assert same.status_code == 201
        assert bare.get(f"{P}/profiles").status_code == 200                # safe methods are exempt
        assert bare.get(f"{P}/profiles", headers={"Origin": "https://evil.example"}).status_code == 200
        pre = bare.options(f"{P}/profiles", headers={"Origin": "https://evil.example",
                                                     "Access-Control-Request-Method": "POST"})
        assert "access-control-allow-origin" not in pre.headers        # no CORS is ever granted


def test_bearer_clients_are_exempt_from_custom_header(env: Env) -> None:
    app = create_app(env.cfg, env.container)
    with TestClient(app, client=("192.168.1.9", 1)) as remote:
        key = (env.cfg.data_dir / "bootstrap-admin.key").read_text().strip()
        r = remote.post(f"{P}/profiles", json={"name": "n", "engine": "fake"}, headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 201


def test_profile_import_requires_json_or_yaml_content_type(client: TestClient) -> None:
    r = client.post(f"{P}/profiles/import", content="name: x\nengine: fake\n", headers={"content-type": "text/plain"})
    assert r.status_code == 415 and problem_code(r) == "unsupported_media_type"
    r = client.post(f"{P}/profiles/import", content="name: x\nengine: fake\n", headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 415
    r = client.post(f"{P}/profiles/import", content="name: x\nengine: fake\n", headers={"content-type": "application/yaml"})
    assert r.status_code == 201


# ---- 2. hf_cache_dir ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["relative/path", "/tmp/a:/etc", "/tmp/a,b", "/tmp/a\nb"])
def test_cache_dir_validated_at_startup(bad: str) -> None:
    with pytest.raises(ValidationError):
        Settings(hf_cache_dir=bad)


def test_cache_dir_symlink_rejected(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(ValidationError):
        Settings(hf_cache_dir=tmp_path / "link")


def test_cache_dir_not_settable_via_api_and_mount_is_readonly(client: TestClient, env: Env) -> None:
    before = client.get(f"{P}/settings").json()["hf_cache_dir"]
    r = client.put(f"{P}/settings", json={"hf_cache_dir": "/etc"})
    assert r.status_code == 200 and r.json()["hf_cache_dir"] == before
    assert Settings().hf_cache_readonly is True
    env.seed_local_model()
    client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 4096}})
    run = next(c for c in env.runner.calls if c[3] == "run")
    mount = run[run.index("-v") + 1]
    assert mount.endswith("/root/.cache/huggingface:ro")


# ---- 3. hostile hub -------------------------------------------------------------------------------------------------
async def run_dl(env: Env, repo: str = "org/evil") -> Any:
    d = await env.container.downloads.create(DownloadRequest(repo_id=repo))
    await env.container.downloads.wait(d.id)
    return env.container.downloads.get(d.id)


@pytest.mark.parametrize("path", ["../escape.txt", "/etc/passwd", "a/../../b", "a\\b", "a\x00b", "a//b", "./a", ""])
async def test_hostile_paths_rejected_before_any_write(env: Env, path: str) -> None:
    env.hub.add("org/evil", {"config.json": b"{}"}, {"m": 1})
    env.hub.repos["org/evil"][0].files.append(type(env.hub.repos["org/evil"][0].files[0])(path=path, size=1))
    d = await run_dl(env)
    assert d.state == "failed" and d.error_code == "hf_bad_path"
    assert not (env.cfg.hf_cache_dir / "hub" / "models--org--evil" / "blobs").exists()
    assert not (env.tmp / "escape.txt").exists()


@pytest.mark.parametrize("sha", ["../../../../etc/x", "ABCD" * 16, "a" * 63, "g" * 64])
async def test_hostile_sha256_rejected(env: Env, sha: str) -> None:
    env.hub.add("org/evil", {"config.json": b"{}"}, {"m": 1})
    env.hub.repos["org/evil"][0].files[0].sha256 = sha
    d = await run_dl(env)
    assert d.state == "failed" and d.error_code == "hf_bad_hash"


@pytest.mark.parametrize("sha", ["abc", "../x", "Z" * 40, "a" * 41])
async def test_hostile_commit_sha_rejected(env: Env, sha: str) -> None:
    env.hub.add("org/evil", {"config.json": b"{}"}, {"m": 1}, sha=sha)
    d = await run_dl(env)
    assert d.state == "failed" and d.error_code == "hf_bad_sha"


async def test_stream_longer_than_declared_size_is_cut_off(env: Env) -> None:
    env.hub.add("org/evil", {"config.json": b"12345678"}, {"m": 1})
    env.hub.repos["org/evil"][0].files[0].size = 4     # declares 4, server streams 8
    d = await run_dl(env)
    assert d.state == "failed" and d.error_code == "size_mismatch"


def test_revision_and_repo_validation(client: TestClient) -> None:
    for rev in ("../x", "a b", "x" * 101, "a;b", "a/b"):
        assert client.post(f"{P}/downloads", json={"repo_id": "a/b", "revision": rev}).status_code == 422
    assert client.get(f"{P}/hf/models/a/b", params={"revision": "../x"}).status_code == 422


async def test_download_queue_is_capped(env: Env) -> None:
    env.hub.add("org/m", {"config.json": b"{}"}, {"m": 1})
    for i in range(env.cfg.max_download_queue):
        env.container.store.execute(
            "INSERT INTO downloads(id,repo_id,state,created_at,updated_at) VALUES(?,?,'paused',1,1)", (f"d{i}", "a/b"))
    from engine_console.domain.errors import Conflict
    with pytest.raises(Conflict) as ei:
        await env.container.downloads.create(DownloadRequest(repo_id="org/m"))
    assert ei.value.code == "download_queue_full"


# ---- 4. RepoId ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["a--b/c", "a/b--c", "../a/b", "a/../b", "a/b/c", "a", "/a/b", "a/b\n", "-a/b", "a/.hidden",
                                 "a/b c", "a/b?x=1", "a/" + "x" * 97])
def test_repo_id_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        check_repo_id(bad)


@pytest.mark.parametrize("good", ["Qwen/Qwen3-32B", "org/m", "a/b.c_d-e", "openai/gpt-oss-120b"])
def test_repo_id_accepts(good: str) -> None:
    assert check_repo_id(good) == good


def test_every_route_validates_repo_id(client: TestClient, env: Env) -> None:
    bad = "a--b/c"
    assert client.post(f"{P}/fit", json={"engine": "fake", "repo_id": bad}).status_code == 422
    assert client.post(f"{P}/downloads", json={"repo_id": bad}).status_code == 422
    assert client.post(f"{P}/instances", json={"engine": "fake", "repo_id": bad}).status_code == 422
    assert client.post(f"{P}/instances/preflight", json={"engine": "fake", "repo_id": bad}).status_code == 422
    assert client.post(f"{P}/profiles", json={"name": "n", "engine": "fake", "repo_id": bad}).status_code == 422
    assert client.get(f"{P}/hf/models/{bad}").status_code == 422
    assert client.delete(f"{P}/models/{bad}").status_code == 422


@respx.mock
async def test_hub_urls_quote_repo_id() -> None:
    c = HfHttpClient("https://huggingface.co", ["huggingface.co"], lambda: None)
    route = respx.get(url__regex=r"https://huggingface.co/.*").mock(return_value=httpx.Response(404))
    with pytest.raises(Exception):  # noqa: B017, PT011 - 404 -> NotFound
        await c.model("org/na me?x#y", None)
    raw = route.calls.last.request.url.raw_path.decode()
    assert "%20" in raw and "%3Fx" in raw and "%23y" in raw and raw.count("?") == 1   # only the real query separator
    await c.aclose()


# ---- 5. secret params ------------------------------------------------------------------------------------------------------
SECRET = "sk-live-abcdef123456"


def test_api_key_never_stored_or_returned(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    env.engine_up()
    r = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B",
                                            "params": {"max_model_len": 4096, "api_key": SECRET}})
    assert r.status_code == 201, r.text
    inst = r.json()
    assert inst["params"]["api_key"] == "[set]" and SECRET not in r.text
    rows = env.container.store.all("SELECT * FROM instances")
    assert SECRET not in json.dumps([dict(x) for x in rows])
    sf = env.cfg.data_dir / "secrets" / f"{inst['id']}.json"
    assert stat.S_IMODE(sf.stat().st_mode) == 0o600 and json.loads(sf.read_text()) == {"api_key": SECRET}
    run = next(c for c in env.runner.calls if c[3] == "run")
    assert SECRET not in " ".join(run) and "FAKE_API_KEY" in run
    assert env.runner.envs[env.runner.calls.index(run)]["FAKE_API_KEY"] == SECRET   # env value only
    cmd = client.get(f"{P}/instances/{inst['id']}/command")
    assert SECRET not in cmd.text and SECRET not in client.get(f"{P}/instances").text
    env.runner.logs[inst["container_name"]] = f"starting with key {SECRET}\n"
    assert SECRET not in client.get(f"{P}/instances/{inst['id']}/logs").text
    audit = json.dumps(client.get(f"{P}/audit").json())
    assert SECRET not in audit and "[redacted]" in audit
    # a stop/start cycle re-reads the secret file rather than needing the API to hold the value
    client.post(f"{P}/instances/{inst['id']}/stop")
    assert client.post(f"{P}/instances/{inst['id']}/start").status_code == 200
    run2 = [c for c in env.runner.calls if c[3] == "run"][-1]
    assert env.runner.envs[env.runner.calls.index(run2)]["FAKE_API_KEY"] == SECRET
    client.delete(f"{P}/instances/{inst['id']}")
    assert not sf.exists()


def test_profile_and_export_drop_secrets_and_placeholder_must_be_resolved(client: TestClient, env: Env) -> None:
    p = client.post(f"{P}/profiles", json={"name": "k", "engine": "fake", "params": {"api_key": SECRET, "tp": 1}})
    assert p.status_code == 201 and p.json()["params"]["api_key"] == "[set]" and SECRET not in p.text
    assert SECRET not in client.get(f"{P}/profiles/{p.json()['id']}/export").text
    env.seed_local_model()
    body = {"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "profile_id": p.json()["id"]}
    r = client.post(f"{P}/instances", json=body)
    assert r.status_code == 409 and "placeholder" in json.dumps(r.json()["preflight"])
    ok = client.post(f"{P}/instances", json={**body, "params": {"api_key": SECRET}})
    assert ok.status_code == 201 and SECRET not in ok.text


def test_secret_that_would_land_in_argv_is_refused(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    r = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B",
                                            "params": {"leaky_key": SECRET}})
    assert r.status_code == 409 and SECRET not in r.text
    assert any(c["code"] == "secret_in_argv" and c["level"] == "block" for c in r.json()["preflight"]["checks"])
    assert not any(c[3] == "run" for c in env.runner.calls)


def test_bench_params_do_not_carry_secrets(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    env.engine_up()
    env.engine_routes["/v1/chat/completions"] = sse_engine
    inst = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B",
                                               "params": {"api_key": SECRET, "max_model_len": 4096}}).json()
    client.portal.call(env.container.lifecycle.tick)  # type: ignore[union-attr]
    r = client.post(f"{P}/bench", json={"instance_id": inst["id"]})
    client.portal.call(env.container.bench.wait, r.json()["id"])  # type: ignore[union-attr]
    assert SECRET not in client.get(f"{P}/bench").text


# ---- 6. image control -------------------------------------------------------------------------------------------------------
def test_per_instance_image_param_is_rejected(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    r = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B",
                                            "params": {"image": "evil/miner:1"}})
    assert r.status_code == 409 and "image override" in json.dumps(r.json()["preflight"])
    assert client.post(f"{P}/profiles", json={"name": "i", "engine": "fake", "params": {"image": "x/y:1"}}).status_code == 422
    assert not any(c[3] == "run" for c in env.runner.calls)


def test_image_allowlist_enforced_for_pins_and_defaults(client: TestClient, env: Env) -> None:
    r = client.put(f"{P}/settings", json={"image_pins": {"fake": "evil/miner:1"}})
    assert r.status_code == 400 and problem_code(r) == "image_not_allowed"
    assert client.put(f"{P}/settings", json={"image_pins": {"fake": "fake/engine:2"}}).status_code == 200
    env.seed_local_model()
    env.cfg.engine_image_allowlist[:] = ["only/this:"]
    pf = client.post(f"{P}/instances/preflight", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B"}).json()
    assert any(c["code"] == "image_not_allowed" and c["level"] == "block" for c in pf["checks"])
    assert client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B"}).status_code == 409


def test_engine_image_allowlist_is_exact_match_with_adapter_pins() -> None:
    from engine_console.adapters import load_default_adapters
    from engine_console.services.settings import SettingsService
    from engine_console.services.store import Store
    reg = load_default_adapters()
    svc = SettingsService(Store(":memory:"), Settings(), lambda: {a.default_image for a in reg.list()})
    assert Settings().engine_image_allowlist == []
    for a in reg.list():
        assert svc.engine_image_allowed(a.default_image)                       # pinned adapter defaults
    assert not svc.engine_image_allowed("vllm/vllm-openai:evil")                # no prefix matching
    assert not svc.engine_image_allowed("vllm/vllm-openai:v0.23.0-evil")
    assert not svc.engine_image_allowed(Settings().gateway_image)               # gateway is not an engine image


# ---- 8. label verification -----------------------------------------------------------------------------------------------------
async def test_docker_refuses_unlabelled_containers(env: Env) -> None:
    env.runner.containers["postgres"] = {"running": True, "labels": {}, "exit": 0}
    d = DockerCli(env.runner)
    for op in (d.stop("postgres"), d.remove("postgres"), d.logs("postgres")):
        with pytest.raises(Forbidden):
            await op
    with pytest.raises(Forbidden):
        async for _ in d.follow_logs("postgres"):
            pass
    assert not any(c[3] in ("stop", "rm") for c in env.runner.calls)
    await d.stop("does-not-exist")            # missing container is a harmless no-op


def test_launch_will_not_rm_an_unlabelled_namesake(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    inst = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 4096}}).json()
    victim = "someone-elses-container"
    env.runner.containers[victim] = {"running": True, "labels": {}, "exit": 0}

    async def force() -> None:
        from engine_console.services.docker import ContainerSpec
        spec = ContainerSpec(name=victim, image="fake/engine:1.0", argv=[], host_port=18050, container_port=8000,
                             gpu_uuids=["GPU-uuid-0"], network="engines")
        await env.container.lifecycle._launch(inst["id"], spec, {})  # noqa: SLF001

    with pytest.raises(Forbidden):
        client.portal.call(force)  # type: ignore[union-attr]
    assert env.runner.containers[victim]["running"] is True
    assert not any(c[3] == "rm" and c[-1] == victim for c in env.runner.calls)
    assert client.get(f"{P}/instances/{inst['id']}").json()["state"] == "failed"


async def test_adopt_skips_out_of_range_ports(env: Env) -> None:
    for i, port in enumerate(("22", "99999", "abc", "18010")):
        env.runner.containers[f"ec-{i}"] = {"running": True, "exit": 0, "labels": {
            "engine-console": "1", "engine-console.instance": f"inst_{i}", "engine-console.engine": "fake",
            "engine-console.model": "a/b", "engine-console.port": port}}
    assert await env.container.lifecycle.adopt() == 1
    assert env.container.lifecycle.get("inst_3").port == 18010


# ---- 9. audit hygiene ----------------------------------------------------------------------------------------------------------------
def test_audit_chat_is_metadata_only_and_query_redacted(client: TestClient, env: Env) -> None:
    inst = make_ready_instance(client, env)
    env.engine_routes["/v1/chat/completions"] = sse_engine
    client.post(f"{P}/chat/completions", json={"instance_id": inst["id"], "messages": [{"role": "user", "content": "my private diary"}]})
    client.post(f"{P}/prompts?access_token=hf_abcdefgh123456&x=1", json={"title": "t", "content": "secret prompt text"})
    dump = json.dumps(client.get(f"{P}/audit").json())
    assert "private diary" not in dump and "secret prompt text" not in dump and "hf_abcdefgh" not in dump
    chat = next(a for a in client.get(f"{P}/audit").json()["items"] if a["path"].endswith("/chat/completions"))
    assert chat["params"]["body"]["messages"] == 1 and "messages" in chat["params"]["body"]["fields"]
    q = next(a for a in client.get(f"{P}/audit").json()["items"] if a["path"].endswith("/prompts"))
    assert q["params"]["query"]["access_token"] == "[redacted]" and q["params"]["query"]["x"] == "1"


def test_audit_records_key_id_as_principal(env: Env) -> None:
    app = create_app(env.cfg, env.container)
    with TestClient(app, client=("192.168.1.9", 1)) as remote:
        key = (env.cfg.data_dir / "bootstrap-admin.key").read_text().strip()
        remote.post(f"{P}/profiles", json={"name": "n", "engine": "fake"}, headers={"Authorization": f"Bearer {key}"})
        kid = env.container.settings.list_keys()[0].id
        top = env.container.audit.list_page(5, None).items[0]
        assert top["actor"] == kid and key not in json.dumps(top)


# ---- 10. size caps, stream limits, bench clamps ----------------------------------------------------------------------------------------
def test_body_caps(client: TestClient, env: Env) -> None:
    big = {"engine": "fake", "repo_id": "a/b", "params": {"pad": "x" * (1024 * 1024 + 10)}}
    r = client.post(f"{P}/fit", json=big)
    assert r.status_code == 413 and problem_code(r) == "payload_too_large"
    chunked = client.post(f"{P}/fit", content=iter([b"x" * 600_000, b"y" * 600_000]), headers={"content-type": "application/json"})
    assert chunked.status_code == 413                     # no Content-Length: counted while streaming
    imp = client.post(f"{P}/profiles/import", content=b"a: " + b"x" * (256 * 1024 + 1), headers={"content-type": "text/yaml"})
    assert imp.status_code == 413
    chat_body = {"instance_id": "nope", "messages": [{"role": "user", "content": "x" * (2 * 1024 * 1024)}]}
    assert client.post(f"{P}/chat/completions", json=chat_body).status_code == 404   # past the size gate: chat has a larger cap
    huge = {"instance_id": "nope", "messages": [{"role": "user", "content": "x" * (33 * 1024 * 1024)}]}
    assert client.post(f"{P}/chat/completions", json=huge).status_code == 413


def test_sse_connections_are_capped_and_released(client: TestClient, env: Env) -> None:
    env.container.sse.limit = 1
    assert env.container.sse.acquire()          # simulate one open stream
    r = client.get(f"{P}/metrics/stream", params={"once": "true"})
    assert r.status_code == 429 and problem_code(r) == "too_many_streams"
    env.container.sse.release()
    assert client.get(f"{P}/metrics/stream", params={"once": "true"}).status_code == 200
    assert env.container.sse.active == 0


def test_sse_idle_timeout_ends_silent_stream(client: TestClient, env: Env) -> None:
    env.cfg.sse_idle_timeout_s = 0.05
    r = client.get(f"{P}/downloads/stream")     # no `once`: only the idle timeout can end this
    assert r.status_code == 200 and "event: snapshot" in r.text
    assert env.container.sse.active == 0
    _ = asyncio


@pytest.mark.parametrize("suite", [{"concurrency": [1000]}, {"concurrency": list(range(1, 20))}, {"prompt_tokens": 10**7},
                                   {"max_tokens": 10**6}, {"rounds": 500}, {"concurrency": [True]}, {"surprise": 1}])
def test_bench_suite_is_clamped(client: TestClient, env: Env, suite: dict[str, Any]) -> None:
    inst = make_ready_instance(client, env)
    r = client.post(f"{P}/bench", json={"instance_id": inst["id"], "suite": suite})
    assert r.status_code == 400 and problem_code(r) == "invalid_suite"


# ---- container hardening flags ---------------------------------------------------------------------------------------------------------
def test_run_args_drop_caps_and_limit_pids(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 4096}})
    run = next(c for c in env.runner.calls if c[3] == "run")
    assert run[run.index("--cap-drop") + 1] == "ALL"
    assert run[run.index("--pids-limit") + 1] == "4096"
    assert "no-new-privileges:true" in run
