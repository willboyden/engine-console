"""Engine isolation (internal network + per-instance gateway) and console egress fail-closed tests."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from engine_console.config import GATEWAY_IMAGE, Settings
from engine_console.domain.errors import EgressDenied, EgressProxyUnavailable
from engine_console.services.docker import (
    GATEWAY_SH,
    GatewaySpec,
    build_gateway_create_args,
    gateway_config,
)
from engine_console.services.hf_http import EgressConfig, HfHttpClient, public_proxy

from .conftest import Env
from .test_api_smoke import P, problem_code, tick

# A throwaway self-signed certificate (no private key kept): only used so an SSL context can be built.
TEST_CA = """-----BEGIN CERTIFICATE-----
MIIDBTCCAe2gAwIBAgIUCtZZanOajHEtt8R6GS0GjwC+x+EwDQYJKoZIhvcNAQEL
BQAwEjEQMA4GA1UEAwwHdGVzdC1jYTAeFw0yNjA5MjQwMzU2MjRaFw0zNjA5MjEw
MzU2MjRaMBIxEDAOBgNVBAMMB3Rlc3QtY2EwggEiMA0GCSqGSIb3DQEBAQUAA4IB
DwAwggEKAoIBAQDmD6fL5c6Fa8/gwfZ8ICOeTApiS5P3dTnCV5ShZCFjAeMslalB
FBHUb6F5GvTfDtKUW1JYHh+wJSNdvFJr1N1UKJcYhYkRYpEcvJ+WYVF49EyeTe43
PXRQwtj3jP52cEJDvQ6LqLQUIIPyk55xmqJlrRPjFYshYl7FoN1VMdV2lAmkIorN
ESLRiudH7aw1KqUFCKVW8B6GuMnDrUQvZa9gqbuKKqn3EhVn1AiqaWgjOjDCL/NL
bqZUZ0Li7DiUUvW4/RoHXMzNoOZQlBCG+lqJkR2rOS0An1dYaV4GGbPFolpsZUdw
KNMyA1flIqYNnZXpWEuw+CzCdojyd/ouRW5/AgMBAAGjUzBRMB0GA1UdDgQWBBQF
NzdJOmBeCM6T+H+IGRckrKvhqjAfBgNVHSMEGDAWgBQFNzdJOmBeCM6T+H+IGRck
rKvhqjAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUAA4IBAQB/OOWMxOGA
50BkyxkR93rEi4dFDoRDDqPiX93ywBkPBOjbCVnThTp10qv28iaCufdd2saob+f4
XfIQyMFTNYv6rO07pAlG4voz0zbDbMbhOGaUh7+lv2T5cgTndKcbwF5yzCi1TKGV
s9yNPfTD167nUNz8jlgpmoRfVrZ18WC7ibCzniL3etSSV7e6Y4Fnk6t1+x+1+65d
xSOlZHyI1qHImdkLD5GeWeEHEq+lKU7phP1YxAYzhOvQgLgp12K/M9VnAQYWYx2H
6PEuWxrdG9Igd9QX1Qo9DgsrCADr2nzPyBuEQx/vQch6O3MGS3CA2AtkeS1lzTgC
nL32Ohbc45Ve
-----END CERTIFICATE-----
"""


def create_engine(client: TestClient, env: Env, **kw: object) -> dict[str, object]:
    env.seed_local_model()
    r = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B",
                                            "params": {"max_model_len": 4096}, **kw})
    assert r.status_code == 201, r.text
    return dict(r.json())


# ---- engine on the internal net, gateway on loopback ---------------------------------------------------------------
def test_engine_is_internal_only_and_gateway_is_the_only_bridged_container(client: TestClient, env: Env) -> None:
    inst = create_engine(client, env)
    calls = env.runner.calls
    run = next(c for c in calls if c[3] == "run")
    assert run[run.index("--network") + 1] == "ai-lab-engines"
    assert run[run.index("--network-alias") + 1] == inst["container_name"]
    assert "-p" not in run and "--publish" not in run and "bridge" not in run and "ai-lab" not in run
    assert run[run.index("--cap-drop") + 1] == "ALL"
    gw = next(c for c in calls if c[3] == "create")
    assert gw[gw.index("--network") + 1] == "bridge"
    assert gw[gw.index("-p") + 1] == f"127.0.0.1:{inst['port']}:8080"        # loopback only
    assert [a for a in gw if a == "-p"] == ["-p"] and "0.0.0.0" not in " ".join(gw)
    connect = next(c for c in calls if c[3:5] == ["network", "connect"])
    assert connect[-2:] == ["ai-lab-engines", f"{inst['container_name']}-gw"]
    # order: engine first, gateway created, attached to the internal net, then started
    idx = {k: next(i for i, c in enumerate(calls) if c[3] == k or c[3:5] == k.split()) for k in ("run", "create", "network connect", "start")}
    assert idx["run"] < idx["create"] < idx["network connect"] < idx["start"]
    assert sum(1 for c in calls if "--network" in c and c[c.index("--network") + 1] == "bridge") == 1


def test_gateway_hardening_flags_labels_and_no_shell_interpolation(client: TestClient, env: Env) -> None:
    inst = create_engine(client, env)
    gw = next(c for c in env.runner.calls if c[3] == "create")
    for flag, val in (("--cap-drop", "ALL"), ("--security-opt", "no-new-privileges:true"), ("--user", "101:101")):
        assert gw[gw.index(flag) + 1] == val
    assert "--read-only" in gw and gw.count("--tmpfs") == 3
    assert any(a.startswith("/var/cache/nginx:") for a in gw) and any(a.startswith("/var/run:") for a in gw)
    labels = [gw[i + 1] for i, a in enumerate(gw) if a == "--label"]
    assert {"ai-lab.console=1", "ai-lab.console.role=gateway", f"ai-lab.console.instance={inst['id']}"} <= set(labels)
    assert gw[gw.index("--entrypoint") + 1] == "/bin/sh"
    assert gw[-3:] == ["fake/gateway:1", "-c", GATEWAY_SH]                    # constant shell text, image is the pinned one
    conf = next(a for a in gw if a.startswith("NGINX_CONF="))
    assert f"proxy_pass {inst['container_name']}:8000;" in conf and "listen 8080;" in conf
    assert inst["container_name"] not in GATEWAY_SH


def test_default_gateway_image_is_the_pinned_digest_and_allowlisted() -> None:
    s = Settings()
    assert s.gateway_image == GATEWAY_IMAGE and GATEWAY_IMAGE.startswith("nginx@sha256:62ff2089")
    assert GATEWAY_IMAGE in s.gateway_image_allowlist and GATEWAY_IMAGE not in s.engine_image_allowlist


def test_internal_endpoint_field(client: TestClient, env: Env) -> None:
    inst = create_engine(client, env)
    got = client.get(f"{P}/instances/{inst['id']}").json()
    assert got["internal_endpoint"] == f"http://{inst['container_name']}:8000" and got["container_port"] == 8000


def test_network_created_internal_when_absent(client: TestClient, env: Env) -> None:
    env.runner.networks.clear()
    create_engine(client, env)
    create = next(c for c in env.runner.calls if c[3:5] == ["network", "create"])
    assert "--internal" in create and create[-1] == "ai-lab-engines"
    assert env.runner.networks["ai-lab-engines"] is True


def test_existing_non_internal_network_is_refused(client: TestClient, env: Env) -> None:
    env.runner.networks["ai-lab-engines"] = False
    env.seed_local_model()
    r = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 4096}})
    assert r.status_code == 409 and problem_code(r) == "engine_network_not_internal"
    assert not any(c[3] in ("run", "create") for c in env.runner.calls)
    assert env.runner.networks["ai-lab-engines"] is False                     # left untouched


def test_gateway_failure_tears_the_engine_down(client: TestClient, env: Env) -> None:
    env.runner.fail_connect = True
    env.seed_local_model()
    r = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 4096}})
    assert r.status_code == 502
    assert not any(v["running"] for v in env.runner.containers.values())
    assert client.get(f"{P}/instances").json()["items"][0]["state"] == "failed"


def test_stop_and_remove_handle_both_containers(client: TestClient, env: Env) -> None:
    inst = create_engine(client, env)
    name = str(inst["container_name"])
    n0 = len(env.runner.calls)
    client.post(f"{P}/instances/{inst['id']}/stop")
    stops = [c[-1] for c in env.runner.calls[n0:] if c[3] == "stop"]
    assert stops == [f"{name}-gw", name]                                     # gateway first
    client.delete(f"{P}/instances/{inst['id']}")
    assert name not in env.runner.containers and f"{name}-gw" not in env.runner.containers


def test_supervisor_fails_instance_whose_gateway_is_gone_or_foreign(client: TestClient, env: Env) -> None:
    inst = create_engine(client, env)
    name = str(inst["container_name"])
    env.engine_up()
    tick(client, env)
    assert client.get(f"{P}/instances/{inst['id']}").json()["state"] == "ready"
    env.runner.containers[f"{name}-gw"]["labels"]["ai-lab.console.instance"] = "inst_someone_else"
    tick(client, env)
    got = client.get(f"{P}/instances/{inst['id']}").json()
    assert got["state"] == "failed" and "gateway" in got["error"]
    assert not any(c[3] == "stop" and c[-1] == f"{name}-gw" for c in env.runner.calls)   # a foreign gateway is never touched


async def test_adoption_ignores_gateways_and_only_adopts_engines(env: Env) -> None:
    lab = {"ai-lab.console": "1", "ai-lab.console.instance": "inst_x", "ai-lab.console.engine": "fake",
           "ai-lab.console.model": "a/b", "ai-lab.console.port": "18020"}
    env.runner.containers["ec-x-gw"] = {"running": True, "exit": 0, "labels": {**lab, "ai-lab.console.role": "gateway"}}
    assert await env.container.lifecycle.adopt() == 0
    env.runner.containers["ec-x"] = {"running": True, "exit": 0, "labels": {**lab, "ai-lab.console.role": "engine"}}
    assert await env.container.lifecycle.adopt() == 1


# ---- gateway config injection ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("host", ["a;b", "a b", "a\n}", "$(id)", "a{b}", "A", "", "x" * 64, "-a", "a\x00", "a:80", "a/b", "a#b"])
def test_hostile_gateway_host_names_are_rejected(host: str) -> None:
    with pytest.raises(ValueError):
        gateway_config(host, 8000)


@pytest.mark.parametrize("port", ["80;", "80", True, False, 0, -1, 65536, 80.0, None])
def test_hostile_gateway_ports_are_rejected(port: object) -> None:
    with pytest.raises(ValueError):
        gateway_config("ec-ok-abc123", port)  # type: ignore[arg-type]


def test_gateway_create_args_reject_hostile_inputs() -> None:
    base = dict(name="ec-x-gw", image="fake/gw:1", host_port=18001, engine_host="ec-x", engine_port=8000,
                internal_network="ai-lab-engines", instance_id="inst_x")
    assert build_gateway_create_args(GatewaySpec(**base))[0] == "create"  # type: ignore[arg-type]
    for bad in ({"engine_host": "x; proxy_pass evil:1"}, {"engine_port": "8000; x"}, {"host_port": "18001:22"}, {"host_port": True}):
        with pytest.raises(ValueError):
            build_gateway_create_args(GatewaySpec(**{**base, **bad}))  # type: ignore[arg-type]


def test_gateway_config_is_a_pure_tcp_forward() -> None:
    conf = gateway_config("ec-ok-abc123", 8000)
    assert conf.count("proxy_pass") == 1 and "http {" not in conf and "stream {" in conf and conf.count(";") == conf.count(";")
    assert "ec-ok-abc123:8000;" in conf and "buffer" not in conf.lower()


# ---- console egress ---------------------------------------------------------------------------------------------------
def egress_client(tmp_path: Path, proxy: str | None, ca: bool = True, require: bool = False,
                  injected: httpx.AsyncClient | None = None) -> HfHttpClient:
    bundle = tmp_path / "ca.pem"
    if ca:
        bundle.write_text(TEST_CA)
    return HfHttpClient("https://huggingface.co", ["huggingface.co"], lambda: None, client=injected,
                        egress=EgressConfig(proxy=proxy, ca_bundle=bundle, require_proxy=require))


async def test_missing_ca_fails_closed_without_any_connection(tmp_path: Path) -> None:
    hits: list[httpx.Request] = []
    direct = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: hits.append(r) or httpx.Response(200, json=[])))
    c = egress_client(tmp_path, "http://127.0.0.1:1", ca=False, injected=direct)
    with pytest.raises(EgressProxyUnavailable) as ei:
        await c.search(q="x", task=None, library=None, quant=None, sort="downloads", limit=1)
    assert ei.value.status == 503 and ei.value.code == "egress_proxy_unavailable" and hits == []


async def test_unreachable_proxy_fails_closed_never_direct(tmp_path: Path) -> None:
    hits: list[httpx.Request] = []
    direct = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: hits.append(r) or httpx.Response(200, json=[])))
    c = egress_client(tmp_path, "http://127.0.0.1:1", injected=direct)       # nothing listens on port 1
    with pytest.raises(EgressProxyUnavailable):
        await c.model("org/m", None)
    with pytest.raises(EgressProxyUnavailable):
        await c.open_file("org/m", "main", "f", 0)
    assert hits == [], "the injected direct client must never be used when a proxy is configured"
    await c.aclose()


async def test_require_proxy_without_proxy_refuses(tmp_path: Path) -> None:
    c = egress_client(tmp_path, None, require=True)
    with pytest.raises(EgressProxyUnavailable):
        await c.card("org/m", None)


async def test_bad_ca_file_fails_closed(tmp_path: Path) -> None:
    c = egress_client(tmp_path, "http://127.0.0.1:1", ca=False)
    (tmp_path / "ca.pem").write_text("not a certificate")
    with pytest.raises(EgressProxyUnavailable):
        await c.card("org/m", None)


async def test_proxied_client_ignores_env_and_uses_the_proxy(tmp_path: Path) -> None:
    c = egress_client(tmp_path, "http://127.0.0.1:8082")
    built = c._client  # noqa: SLF001
    assert built is not c._injected and c._built is built  # noqa: SLF001
    assert built._trust_env is False  # type: ignore[attr-defined]  # noqa: SLF001
    await c.aclose()


def test_proxy_block_page_maps_to_egress_denied() -> None:
    c = HfHttpClient("https://huggingface.co", ["huggingface.co"], lambda: None)
    req = httpx.Request("GET", "https://huggingface.co/x")
    blocked = httpx.Response(403, text="egress blocked by addon_guard: host not in allowlist.txt\n", request=req)
    with pytest.raises(EgressDenied):
        c._raise(blocked, "org/m")  # noqa: SLF001


def test_health_and_settings_show_egress_mode(client: TestClient, env: Env) -> None:
    h = client.get(f"{P}/health").json()
    assert h["egress_mode"] == "direct" and any("DIRECT" in w for w in h["warnings"])
    assert client.get(f"{P}/settings").json()["egress_mode"] == "direct"
    env.cfg.egress_proxy = "http://user:hunter2@127.0.0.1:8082"
    h2 = client.get(f"{P}/health").json()
    assert h2["egress_mode"] == "proxied" and not any("egress" in w.lower() for w in h2["warnings"])
    s = client.get(f"{P}/settings").json()
    assert s["egress_proxy"] == "http://127.0.0.1:8082" and "hunter2" not in str(s) and s["require_egress_proxy"] is False


def test_public_proxy_strips_credentials() -> None:
    assert public_proxy("http://u:p@h:1") == "http://h:1" and public_proxy(None) is None


def test_egress_failure_is_a_503_problem_over_the_api(env: Env, tmp_path: Path) -> None:
    from engine_console.main import create_app
    env.container.hf._hub = egress_client(tmp_path, "http://127.0.0.1:1", ca=False)  # noqa: SLF001
    with TestClient(create_app(env.cfg, env.container), client=("127.0.0.1", 1), headers={"X-Engine-Console": "1"}) as c:
        r = c.get(f"{P}/hf/models/org/m")
        assert r.status_code == 503 and problem_code(r) == "egress_proxy_unavailable"


def test_default_ca_is_the_certificate_never_the_private_key() -> None:
    ca = Settings().egress_ca_bundle
    assert ca is None or (ca.name == "mitmproxy-ca-cert.pem" and ca.is_file())
    assert Settings().egress_proxy is None and Settings().require_egress_proxy is False


# ---- the real addon_guard allowlist ---------------------------------------------------------------------------------------
def load_addon_guard(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    stub = types.ModuleType("mitmproxy")
    stub.http = types.ModuleType("mitmproxy.http")  # type: ignore[attr-defined]
    stub.http.HTTPFlow = object  # type: ignore[attr-defined]  # only used in an annotation
    monkeypatch.setitem(sys.modules, "mitmproxy", stub)
    monkeypatch.setitem(sys.modules, "mitmproxy.http", stub.http)  # type: ignore[attr-defined]
    path = Path(__file__).resolve().parents[5] / "security" / "egress" / "mitmproxy" / "addon_guard.py"
    spec = importlib.util.spec_from_file_location("addon_guard_under_test", path)
    assert spec and spec.loader, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("host", ["huggingface.co", "cdn-lfs.huggingface.co", "cdn-lfs-us-1.hf.co", "cdn-lfs.hf.co",
                                  "cas-bridge.xethub.hf.co", "transfer.xethub.hf.co", "x.hf.co"])
def test_console_hosts_are_covered_by_the_real_allowlist(monkeypatch: pytest.MonkeyPatch, host: str) -> None:
    assert load_addon_guard(monkeypatch)._allowed(host) is True  # noqa: SLF001


@pytest.mark.parametrize("host", ["evil.example", "huggingface.co.evil.example", "nothf.co", "hf.co.evil.example"])
def test_real_allowlist_still_blocks_lookalikes(monkeypatch: pytest.MonkeyPatch, host: str) -> None:
    assert load_addon_guard(monkeypatch)._allowed(host) is False  # noqa: SLF001


def test_gateway_config_runs_in_foreground() -> None:
    # Regression: without `daemon off;` nginx forks and PID 1 exits, so the gateway container dies at once.
    from engine_console.services.docker import gateway_config
    assert "daemon off;" in gateway_config("engine-a", 8000)
