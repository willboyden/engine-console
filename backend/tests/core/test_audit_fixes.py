"""Regression tests for the pre-release audit findings."""
from __future__ import annotations

import logging
import os
import re
import stat
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from engine_console.api.deps import enforce_roles
from engine_console.api.security import CSP, MUTATING, viewer_may
from engine_console.config import GATEWAY_IMAGE
from engine_console.domain.errors import EgressDenied, Forbidden
from engine_console.main import create_app
from engine_console.services.hf_http import EgressConfig, HfHttpClient
from engine_console.services.secrets_store import SecretStore
from engine_console.services.settings import Principal

from .conftest import Env
from .test_api_smoke import P, problem_code

DEPLOY = Path(__file__).resolve().parents[3] / "deploy" / "engine-console.service"


# ---- 1. no Swagger page; openapi behind auth -------------------------------------------------------------------
def test_no_docs_pages_and_openapi_needs_auth(env: Env) -> None:
    app = create_app(env.cfg, env.container)
    with TestClient(app, client=("192.168.1.9", 1)) as remote:
        for path in ("/api/docs", "/api/redoc", f"{P}/openapi.json"):
            assert remote.get(path).status_code in (401, 404)
        assert remote.get(f"{P}/openapi.json").status_code == 401
    with TestClient(app, client=("127.0.0.1", 1)) as local:
        assert local.get("/api/docs").status_code == 404 and local.get(f"{P}/openapi.json").status_code == 200


# ---- 2. security headers on every response -----------------------------------------------------------------------
EXPECTED = {"x-content-type-options": "nosniff", "referrer-policy": "no-referrer", "x-frame-options": "DENY",
            "permissions-policy": "camera=(), microphone=(), geolocation=()", "content-security-policy": CSP}


def check_headers(r: httpx.Response) -> None:
    for k, v in EXPECTED.items():
        assert r.headers[k] == v, (k, r.request.url)


def test_security_headers_everywhere(client: TestClient, env: Env) -> None:
    for r in (client.get(f"{P}/health"), client.get(f"{P}/nope"), client.get("/"), client.get("/js/x.js"),
              client.get(f"{P}/instances", headers={"Host": "evil.example"}), client.post(f"{P}/fit", json={"engine": 3}),
              client.get("/metrics")):
        check_headers(r)
    assert client.get(f"{P}/health").headers["cache-control"] == "no-store"
    assert client.get(f"{P}/nope").headers["cache-control"] == "no-store"
    assert client.get("/").headers.get("cache-control") != "no-store"       # only /api/* is no-store
    app = create_app(env.cfg, env.container)
    with TestClient(app, client=("192.168.1.9", 1)) as remote:
        r = remote.get(f"{P}/instances")
        assert r.status_code == 401
        check_headers(r)
    assert "script-src 'self'" in CSP and "frame-ancestors 'none'" in CSP and "unsafe-inline" not in CSP


# ---- 3. static allowlist ------------------------------------------------------------------------------------------
def test_static_serves_only_shell_and_asset_dirs(client: TestClient, env: Env) -> None:
    fe = env.tmp / "fe"
    for d, f in (("js", "a.js"), ("css", "a.css"), ("i18n", "en.json"), ("dev", "mock.mjs"), ("tests", "t.mjs")):
        (fe / d).mkdir(exist_ok=True)
        (fe / d / f).write_text("x")
    (fe / "package.json").write_text("{}")
    (fe / "secret.txt").write_text("nope")
    for ok in ("/", "/index.html", "/js/a.js", "/css/a.css", "/i18n/en.json"):
        assert client.get(ok).status_code == 200, ok
    for bad in ("/dev/mock.mjs", "/tests/t.mjs", "/package.json", "/secret.txt", "/js", "/js/", "/js/missing.js",
                "/js/../secret.txt", "/js/..%2fsecret.txt", "/%2e%2e/etc/passwd", "/app.js"):
        assert client.get(bad).status_code == 404, bad
    (fe / "js" / "link").symlink_to("/etc/hostname")
    assert client.get("/js/link").status_code == 404                        # symlink escape


# ---- 5. HF client: trust_env off, no https downgrade ----------------------------------------------------------------
async def test_direct_client_ignores_proxy_env_and_ca_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://evil:3128")
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent")
    c = HfHttpClient("https://huggingface.co", ["huggingface.co"], lambda: None, egress=EgressConfig())
    built = c._client  # noqa: SLF001
    assert built._trust_env is False  # type: ignore[attr-defined]  # noqa: SLF001
    await c.aclose()


@respx.mock
async def test_redirect_to_http_is_refused() -> None:
    c = HfHttpClient("https://huggingface.co", ["huggingface.co"], lambda: None,
                     client=httpx.AsyncClient(follow_redirects=True))
    respx.get("https://huggingface.co/a/b/resolve/main/f").mock(
        return_value=httpx.Response(302, headers={"location": "http://huggingface.co/a/b/f"}))
    downgraded = respx.get("http://huggingface.co/a/b/f").mock(return_value=httpx.Response(200, content=b"x"))
    with pytest.raises(EgressDenied, match="non-https"):
        await c.open_file("a/b", "main", "f", 0)
    assert not downgraded.called
    await c.aclose()


# ---- 6. roles: every mutating route --------------------------------------------------------------------------------------
def concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "x", path)


def test_viewer_gets_403_on_every_mutating_route_except_the_allowed_set(env: Env) -> None:
    app = create_app(env.cfg, env.container)
    # enumerate from the OpenAPI document so newly added routes are picked up automatically
    routes = [(m.upper(), p) for p, ops in app.openapi()["paths"].items() for m in ops if m.upper() in MUTATING]
    assert len(routes) >= 30, "route enumeration looks wrong"
    with TestClient(app, client=("192.168.1.9", 1)) as remote:
        boot = (env.cfg.data_dir / "bootstrap-admin.key").read_text().strip()
        secret = remote.post(f"{P}/keys", json={"name": "v", "role": "viewer"}, headers={"Authorization": f"Bearer {boot}"}).json()["secret"]
        vh = {"Authorization": f"Bearer {secret}"}
        allowed = []
        for method, path in routes:
            r = remote.request(method, concrete(path), headers=vh, json={})
            if viewer_may(method, concrete(path)):
                allowed.append((method, path))
                assert r.status_code != 403, (method, path)
            else:
                assert r.status_code == 403, (method, path, r.status_code)
        assert {p for _, p in allowed} == {f"{P}/chat/completions", f"{P}/fit", f"{P}/conversations",
                                           f"{P}/arena/matches", f"{P}/arena/matches/{{mid}}/vote"}


def test_viewer_cannot_delete_conversations_or_touch_prompts(env: Env) -> None:
    assert not viewer_may("DELETE", f"{P}/conversations/x") and not viewer_may("POST", f"{P}/prompts")
    assert not viewer_may("POST", f"{P}/conversations/x/../keys") and not viewer_may("POST", f"{P}/fitx")
    assert viewer_may("POST", f"{P}/fit") and not viewer_may("GET", f"{P}/fit")


def test_enforce_roles_dependency_is_independent_of_the_middleware() -> None:
    from starlette.requests import Request

    def req(method: str, path: str, role: str) -> Request:
        return Request({"type": "http", "method": method, "path": path, "headers": [], "query_string": b"",
                        "state": {"principal": Principal("k", role)}})
    enforce_roles(req("GET", f"{P}/keys", "viewer"))
    enforce_roles(req("DELETE", f"{P}/keys/x", "admin"))
    enforce_roles(req("POST", f"{P}/chat/completions", "viewer"))
    for method, path in (("DELETE", f"{P}/conversations/x"), ("POST", f"{P}/instances"), ("PUT", f"{P}/settings")):
        with pytest.raises(Forbidden):
            enforce_roles(req(method, path, "viewer"))
    with pytest.raises(Forbidden):
        enforce_roles(req("POST", f"{P}/instances", "public"))


# ---- 7. images -------------------------------------------------------------------------------------------------------------------
def test_gateway_image_cannot_be_pinned_as_an_engine_image(client: TestClient) -> None:
    r = client.put(f"{P}/settings", json={"image_pins": {"fake": GATEWAY_IMAGE}})
    assert r.status_code == 400 and problem_code(r) == "image_not_allowed"
    assert client.put(f"{P}/settings", json={"image_pins": {"fake": "fake/engine:2.0-evil"}}).status_code == 400
    assert client.put(f"{P}/settings", json={"image_pins": {"fake": "fake/engine:2.0"}}).status_code == 200


def test_gateway_has_its_own_check(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    env.cfg.gateway_image = "fake/engine:2.0"     # allowed as an engine pin, but not on the gateway list
    r = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 4096}})
    assert r.status_code == 422 and problem_code(r) == "image_not_allowed"
    assert not any(c[3] == "create" for c in env.runner.calls)


# ---- 8. adoption label validation ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("labels", [
    {"ai-lab.console.model": "a/b;rm -rf"}, {"ai-lab.console.model": "a--b/c"}, {"ai-lab.console.instance": "inst x"},
    {"ai-lab.console.instance": "../x"}, {"ai-lab.console.name": "a\nb"},
    {"ai-lab.console.name": "x" * 200}, {"ai-lab.console.instance": "i" * 100}])
async def test_adopt_skips_invalid_labels_without_raising(env: Env, labels: dict[str, str], caplog: pytest.LogCaptureFixture) -> None:
    good = {"ai-lab.console": "1", "ai-lab.console.instance": "inst_ok", "ai-lab.console.engine": "fake",
            "ai-lab.console.model": "a/b", "ai-lab.console.port": "18010", "ai-lab.console.name": "ok"}
    env.runner.containers["ec-bad"] = {"running": True, "exit": 0, "labels": {**good, **labels}}
    with caplog.at_level(logging.WARNING):
        assert await env.container.lifecycle.adopt() == 0
    assert "not adopting" in caplog.text
    env.runner.containers["ec-bad"]["labels"] = good
    assert await env.container.lifecycle.adopt() == 1


def test_instance_names_are_constrained(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    for bad in ("a\nb", "x;y", "../a", "", "a" * 65):
        r = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "name": bad})
        assert r.status_code == 422, bad


# ---- 9. bounded metric labels --------------------------------------------------------------------------------------------------------------
def test_method_metric_label_is_bounded(client: TestClient) -> None:
    for m in ("BREW", "PROPFIND", "X" * 30):
        client.request(m, f"{P}/health")
    text = client.get("/metrics").text
    assert 'method="OTHER"' in text and "BREW" not in text and "PROPFIND" not in text and "XXXXXXXX" not in text


# ---- 10. IPC mode / trust_remote_code ------------------------------------------------------------------------------------------------------
def run_args(client: TestClient, env: Env, params: dict[str, object]) -> list[str]:
    if not env.container.downloads.is_cached("Qwen/Qwen3-32B"):
        env.seed_local_model()
    r = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": params})
    assert r.status_code == 201, r.text
    return next(c for c in env.runner.calls if c[3] == "run")


def test_single_gpu_uses_private_ipc_and_multi_gpu_uses_host(client: TestClient, env: Env) -> None:
    a = run_args(client, env, {"max_model_len": 4096})
    assert a[a.index("--ipc") + 1] == "private" and a[a.index("--shm-size") + 1] == "16g"
    env.runner.calls.clear()
    b = run_args(client, env, {"max_model_len": 4096, "tp": 2})
    assert b[b.index("--ipc") + 1] == "host"


def test_trust_remote_code_is_off_by_default_and_warns_when_on(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    body: dict[str, object] = {"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 4096}}
    checks = client.post(f"{P}/instances/preflight", json=body).json()["checks"]
    assert not any(c["code"] == "trust_remote_code" for c in checks)
    body["params"] = {"max_model_len": 4096, "trust_remote_code": True}
    checks = client.post(f"{P}/instances/preflight", json=body).json()["checks"]
    assert any(c["code"] == "trust_remote_code" and c["level"] == "warn" for c in checks)
    run_args(client, env, {"max_model_len": 4096})
    assert not any("--trust-remote-code" in c for c in env.runner.calls if c[3] == "run")


# ---- 11. secret file modes, health warning, startup log ---------------------------------------------------------------------------------------
def test_secret_files_are_forced_to_0600_even_if_preexisting(tmp_path: Path) -> None:
    d = tmp_path / "s"
    d.mkdir(mode=0o755)
    f = d / "inst_1.json"
    f.write_text("{}")
    f.chmod(0o644)
    SecretStore(d).put("inst_1", {"api_key": "k"})
    assert stat.S_IMODE(f.stat().st_mode) == 0o600 and stat.S_IMODE(d.stat().st_mode) == 0o700


def test_bootstrap_key_rewritten_with_0600(env: Env) -> None:
    (env.cfg.data_dir).mkdir(parents=True, exist_ok=True)
    key = env.cfg.data_dir / "bootstrap-admin.key"
    key.write_text("old")
    key.chmod(0o666)
    env.container.settings.ensure_bootstrap_key(env.cfg.data_dir)
    assert stat.S_IMODE(os.stat(key).st_mode) == 0o600


def test_health_warns_about_loopback_trust(client: TestClient, env: Env) -> None:
    assert any("trust_loopback" in w for w in client.get(f"{P}/health").json()["warnings"])
    env.cfg.trust_loopback = False
    assert not any("trust_loopback" in w for w in client.get(f"{P}/health").json()["warnings"])


def test_startup_tells_operator_to_delete_bootstrap_key(env: Env, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="engine_console"), TestClient(create_app(env.cfg, env.container)):
        pass
    assert "DELETE this file" in caplog.text and "first start" in caplog.text
    assert (env.cfg.data_dir / "bootstrap-admin.key").read_text().strip() not in caplog.text


# ---- 12. systemd unit -------------------------------------------------------------------------------------------------------------------------------
def test_service_unit_hardening() -> None:
    text = DEPLOY.read_text()
    live = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    for want in ("RestrictNamespaces=yes", "RestrictRealtime=yes", "ProtectClock=yes", "ProtectHostname=yes",
                 "ProtectProc=invisible", "UMask=0077", "CapabilityBoundingSet=", "SystemCallFilter=@system-service",
                 "SystemCallArchitectures=native", "NoNewPrivileges=yes"):
        assert want in live, want
    assert not any(ln.startswith("MemoryDenyWriteExecute") for ln in live)
    assert not any("EGRESS_PROXY" in ln or "REQUIRE_EGRESS" in ln for ln in live), "unit must not change egress"
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(("WorkingDirectory=", "ExecStart=", "ReadWritePaths=")):
            assert "CHANGEME" in lines[i - 1] or "CHANGEME" in lines[i - 2], ln
