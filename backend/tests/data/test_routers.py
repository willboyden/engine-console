"""The data routes over HTTP: /library, /downloads, /models and PUT /settings, in a small app with the real error
handlers and the real role checks (a viewer may read, only an admin may change anything)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import Depends, FastAPI
from starlette.types import ASGIApp, Receive, Scope, Send

from engine_console.api.deps import enforce_roles
from engine_console.api.errors import install_error_handlers
from engine_console.api.routers import admin, downloads, library, system
from engine_console.domain.models import FitReport
from engine_console.services.common import SseLimiter
from engine_console.services.hf import HfService
from engine_console.services.settings import Principal

from .conftest import World
from .fakes import COMMIT, FakeNode, RootCfg, seed_hf_cache
from .helpers import FILES, TOTAL

P = "/api/v1"


class FakeFit:
    """assess_info as H2a's FitService has it: (engine, info, params, gpu_ids, concurrency=1, *, host_id)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def assess_info(self, engine: str, info: Any, params: dict[str, Any], gpu_ids: list[int], concurrency: int = 1, *,
                    host_id: str = "local") -> FitReport:
        self.calls.append((engine, host_id))
        if engine == "boom":
            raise RuntimeError("one odd model")
        verdict = "fits" if (engine == "vllm" and host_id == "local") else "wont_fit"
        return FitReport(verdict=verdict, confidence="low", tp=1, tp_required=None, concurrency=1, max_len=None, per_gpu=[],  # type: ignore[arg-type]
                         max_context_at_current_concurrency=None, max_concurrency_at_current_context=None)


class AsRole:
    """What SecurityMiddleware does for the routes: put the caller's principal in the ASGI scope."""

    def __init__(self, app: ASGIApp, role: str) -> None:
        self.app, self.role = app, role

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})["principal"] = Principal("tester", self.role)
        await self.app(scope, receive, send)


class Env:
    def __init__(self, world: World, role: str) -> None:
        self.world = world
        self.fit = FakeFit()
        c = SimpleNamespace(library=world.library, downloads=world.downloads, settings=world.settings, adapters=world.adapters,
                            fit=self.fit, bus=world.bus, sse=SseLimiter(8), cfg=SimpleNamespace(sse_idle_timeout_s=2.0),
                            hf=HfService(world.hub))
        app = FastAPI()
        app.state.container = c
        install_error_handlers(app)
        for mod in (downloads, library, admin, system):
            app.include_router(mod.router, prefix=P, dependencies=[Depends(enforce_roles)])
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=AsRole(app, role)), base_url="http://testserver")


async def env(tmp: Path, role: str = "admin") -> Env:
    local = FakeNode(tmp, "local", label="This machine", roots=[RootCfg("hf", "hf-home"), RootCfg("ro", "plain", writable=False)],
                     addresses=["10.0.0.1"])
    other = FakeNode(tmp, "spark", label="Spark", roots=[RootCfg("hf", "hf-hub")], addresses=["10.0.0.2"])
    seed_hf_cache(local.roots["hf"].path, "org/m", FILES)                  # type: ignore[arg-type]
    local.scan()
    seed_hf_cache(other.roots["hf"].path, "org/s", {"w.safetensors": b"s" * 40}, layout="hf-hub")   # type: ignore[arg-type]
    other.scan()
    w = World(tmp, [local, other])
    w.hub.add("org/m", FILES)
    w.hub.add("org/new", FILES)
    await w.start()
    return Env(w, role)


def problem(r: httpx.Response) -> dict[str, Any]:
    assert r.headers["content-type"].startswith("application/problem+json"), r.text
    body: dict[str, Any] = r.json()
    assert body["status"] == r.status_code and body["code"] and body["detail"]
    return body


# ---- GET /library ------------------------------------------------------------------------------------------------------------------
async def test_get_library_lists_models_with_copies(tmp_path: Path) -> None:
    e = await env(tmp_path)
    r = await e.client.get(f"{P}/library")
    assert r.status_code == 200
    body = r.json()
    assert body["next_cursor"] is None and [m["key"] for m in body["items"]] == ["hf:org/m", "hf:org/s"]
    m = body["items"][0]
    assert m["format"] == "hf" and m["engines"] == ["vllm"] and m["size_bytes"] == TOTAL
    c = m["copies"][0]
    assert c["host_id"] == "local" and c["host_label"] == "This machine" and c["root_id"] == "hf" and c["visible_on"] == ["local"]
    assert c["writable"] is True and c["deletable"] is True and c["in_use_by"] == [] and c["drive"]["mount"] == "/"
    assert c["id"] and len(c["id"]) == 16 and c["path"].endswith("hub/models--org--m")
    await e.world.stop()


async def test_get_library_filters_and_paginates(tmp_path: Path) -> None:
    e = await env(tmp_path)
    assert [m["name"] for m in (await e.client.get(f"{P}/library", params={"host": "spark"})).json()["items"]] == ["org/s"]
    assert [m["name"] for m in (await e.client.get(f"{P}/library", params={"q": "org/m"})).json()["items"]] == ["org/m"]
    assert (await e.client.get(f"{P}/library", params={"format": "gguf"})).json()["items"] == []
    page = (await e.client.get(f"{P}/library", params={"limit": 1})).json()
    assert len(page["items"]) == 1 and page["next_cursor"] == "1"
    rest = (await e.client.get(f"{P}/library", params={"limit": 1, "cursor": page["next_cursor"]})).json()
    assert [m["name"] for m in rest["items"]] == ["org/s"] and rest["next_cursor"] is None
    for bad in ({"host": "Bad_Host"}, {"host": "-x"}, {"format": "nope"}, {"q": "x" * 201}):
        r = await e.client.get(f"{P}/library", params=bad)
        assert r.status_code == 422 and problem(r)["code"] == "validation_error"
    await e.world.stop()


async def test_a_viewer_can_read_the_library(tmp_path: Path) -> None:
    e = await env(tmp_path, "viewer")
    assert (await e.client.get(f"{P}/library")).status_code == 200
    assert (await e.client.get(f"{P}/models")).status_code == 200
    await e.world.stop()


# ---- rescan, delete, copy -------------------------------------------------------------------------------------------------------------
async def test_rescan(tmp_path: Path) -> None:
    e = await env(tmp_path)
    r = await e.client.post(f"{P}/library/rescan")
    assert r.status_code == 202 and r.json() == {"hosts": ["local", "spark"]}
    r = await e.client.post(f"{P}/library/rescan", json={"host_id": "spark"})
    assert r.status_code == 202 and r.json() == {"hosts": ["spark"]}
    r = await e.client.post(f"{P}/library/rescan", json={"host_id": "nobody"})
    assert r.status_code == 404 and problem(r)["code"] == "unknown_host"
    r = await e.client.post(f"{P}/library/rescan", json={"host_id": "Bad Id"})
    assert r.status_code == 422
    e.world.fleet.node("spark").state = "down"
    r = await e.client.post(f"{P}/library/rescan", json={"host_id": "spark"})
    assert r.status_code == 503 and problem(r)["code"] == "node_unavailable" and problem(r)["detail"].startswith("Spark: ")
    await e.world.stop()


async def test_delete_a_copy(tmp_path: Path) -> None:
    e = await env(tmp_path)
    cid = (await e.client.get(f"{P}/library", params={"q": "org/m"})).json()["items"][0]["copies"][0]["id"]
    local = e.world.fleet.node("local")
    e.world.busy[("local", (await e.client.get(f"{P}/library", params={"q": "org/m"})).json()["items"][0]["copies"][0]["path"])] = ["inst_1"]
    r = await e.client.delete(f"{P}/library/copies/{cid}")
    assert r.status_code == 409 and problem(r)["code"] == "model_in_use" and "inst_1" in problem(r)["detail"]
    e.world.busy.clear()
    local.allow_delete = False
    r = await e.client.delete(f"{P}/library/copies/{cid}")
    assert r.status_code == 403 and problem(r)["code"] == "policy_denied" and problem(r)["detail"].startswith("This machine: ")
    local.allow_delete = True
    r = await e.client.delete(f"{P}/library/copies/{cid}")
    assert r.status_code == 204 and r.content == b""
    assert [m["name"] for m in (await e.client.get(f"{P}/library")).json()["items"]] == ["org/s"]
    r = await e.client.delete(f"{P}/library/copies/{cid}")
    assert r.status_code == 404 and problem(r)["code"] == "copy_not_found"
    assert (await e.client.delete(f"{P}/library/copies/not-an-id")).status_code == 404
    await e.world.stop()


async def test_copy_a_copy(tmp_path: Path) -> None:
    e = await env(tmp_path)
    cid = (await e.client.get(f"{P}/library", params={"q": "org/m"})).json()["items"][0]["copies"][0]["id"]
    r = await e.client.post(f"{P}/library/copies/{cid}/copy", json={"host_id": "spark", "root_id": "hf"})
    assert r.status_code == 201
    d = r.json()
    assert d["kind"] == "copy" and d["source_copy_id"] == cid and d["host_id"] == "spark" and d["root_id"] == "hf" and d["repo_id"] == "org/m"
    await e.world.downloads.wait(d["id"])
    got = (await e.client.get(f"{P}/downloads/{d['id']}")).json()
    assert got["state"] == "completed" and got["total_bytes"] == TOTAL
    r = await e.client.post(f"{P}/library/copies/{cid}/copy", json={"host_id": "local", "root_id": "hf"})
    assert r.status_code == 409 and problem(r)["code"] == "copy_same_location"
    r = await e.client.post(f"{P}/library/copies/{cid}/copy", json={"host_id": "Bad Id"})
    assert r.status_code == 422
    r = await e.client.post(f"{P}/library/copies/{'0' * 16}/copy", json={"host_id": "spark"})
    assert r.status_code == 404 and problem(r)["code"] == "copy_not_found"
    await e.world.stop()


@pytest.mark.parametrize(("method", "path", "body"), [
    ("POST", "/library/rescan", None), ("DELETE", "/library/copies/" + "0" * 16, None),
    ("POST", "/library/copies/" + "0" * 16 + "/copy", {"host_id": "spark"}), ("DELETE", "/models/org/m", None),
    ("PUT", "/settings", {"idle_ttl_s": 5}),
])
async def test_a_viewer_cannot_change_anything(tmp_path: Path, method: str, path: str, body: Any) -> None:
    e = await env(tmp_path, "viewer")
    r = await e.client.request(method, P + path, json=body)
    assert r.status_code == 403 and problem(r)["code"] == "forbidden"
    await e.world.stop()


# ---- /downloads --------------------------------------------------------------------------------------------------------------------------
async def test_download_to_a_machine_and_a_root(tmp_path: Path) -> None:
    e = await env(tmp_path)
    r = await e.client.post(f"{P}/downloads", json={"repo_id": "org/new", "host_id": "spark", "root_id": "hf"})
    assert r.status_code == 201
    d = r.json()
    assert (d["host_id"], d["root_id"], d["kind"], d["repo_id"], d["source_copy_id"]) == ("spark", "hf", "hf", "org/new", None)
    await e.world.downloads.wait(d["id"])
    done = (await e.client.get(f"{P}/downloads/{d['id']}")).json()
    assert done["state"] == "completed" and done["host_id"] == "spark"
    assert [x["id"] for x in (await e.client.get(f"{P}/downloads")).json()["items"]] == [d["id"]]
    snap = await e.client.get(f"{P}/downloads/stream", params={"once": "true"})
    assert snap.status_code == 200 and "event: snapshot" in snap.text
    data = json.loads(next(line for line in snap.text.splitlines() if line.startswith("data: "))[len("data: "):])
    assert data[0]["host_id"] == "spark" and data[0]["kind"] == "hf"
    assert (await e.client.delete(f"{P}/downloads/{d['id']}")).status_code == 204
    assert (await e.client.get(f"{P}/downloads/{d['id']}")).status_code == 404
    await e.world.stop()


async def test_download_defaults_to_the_local_machine_and_validates_input(tmp_path: Path) -> None:
    e = await env(tmp_path)
    r = await e.client.post(f"{P}/downloads", json={"repo_id": "org/new"})
    assert r.status_code == 201 and r.json()["host_id"] == "local" and r.json()["root_id"] == "hf"
    await e.world.downloads.wait(r.json()["id"])
    for body, status, code in (
            ({"repo_id": "../etc"}, 422, "validation_error"), ({"repo_id": "a/b", "host_id": "BAD"}, 422, "validation_error"),
            ({"repo_id": "a/b", "host_id": "nobody"}, 404, "unknown_host"),
            ({"repo_id": "a/b", "root_id": "ro"}, 409, "root_not_writable"), ({"repo_id": "a/b", "root_id": "zzz"}, 422, "unknown_root")):
        r = await e.client.post(f"{P}/downloads", json=body)
        assert r.status_code == status and problem(r)["code"] == code, body
    await e.world.stop()


async def test_pause_resume_cancel_routes(tmp_path: Path) -> None:
    e = await env(tmp_path)
    e.world.hub.gate = asyncio.Event()
    d = (await e.client.post(f"{P}/downloads", json={"repo_id": "org/new", "host_id": "spark"})).json()
    await asyncio.sleep(0.02)
    assert (await e.client.post(f"{P}/downloads/{d['id']}/pause")).json()["state"] == "paused"
    assert (await e.client.post(f"{P}/downloads/{d['id']}/pause")).status_code == 409
    e.world.hub.gate.set()
    assert (await e.client.post(f"{P}/downloads/{d['id']}/resume")).json()["state"] in ("queued", "running")
    await e.world.downloads.wait(d["id"])
    assert (await e.client.post(f"{P}/downloads/{d['id']}/cancel")).status_code == 409           # already completed
    await e.world.stop()


# ---- GET /hf/models/{repo}: gguf_groups arrive without the route being touched --------------------------------------------------------
async def test_the_hf_model_route_returns_the_gguf_groups(tmp_path: Path) -> None:
    e = await env(tmp_path)
    e.world.hub.add("org/g-GGUF", {"g-Q4_K_M.gguf": b"1234", "g-Q8_0-00001-of-00002.gguf": b"12345", "g-Q8_0-00002-of-00002.gguf": b"678", "README.md": b"x"})
    r = await e.client.get(f"{P}/hf/models/org/g-GGUF")
    assert r.status_code == 200
    assert r.json()["gguf_groups"] == [
        {"quant": "Q4_K_M", "files": ["g-Q4_K_M.gguf"], "total_bytes": 4},
        {"quant": "Q8_0", "files": ["g-Q8_0-00001-of-00002.gguf", "g-Q8_0-00002-of-00002.gguf"], "total_bytes": 8}]
    assert (await e.client.get(f"{P}/hf/models/org/m")).json()["gguf_groups"] == []
    await e.world.stop()


# ---- /models (v1 shape) --------------------------------------------------------------------------------------------------------------------
async def test_models_keeps_the_v1_shape_with_a_host_on_every_row(tmp_path: Path) -> None:
    e = await env(tmp_path)
    r = await e.client.get(f"{P}/models")
    assert r.status_code == 200
    items = r.json()["items"]
    assert [(m["repo_id"], m["host_id"]) for m in items] == [("org/m", "local"), ("org/s", "spark")]
    assert set(items[0]) == {"repo_id", "size_bytes", "revisions", "last_used", "path", "engines_that_fit", "host_id"}
    assert items[0]["revisions"] == [COMMIT] and items[0]["size_bytes"] == TOTAL and items[0]["last_used"] == 2.0
    assert items[0]["engines_that_fit"] == ["vllm"] and items[1]["engines_that_fit"] == []        # judged on its own machine
    assert ("vllm", "local") in e.fit.calls and ("vllm", "spark") in e.fit.calls
    await e.world.stop()


async def test_one_model_that_cannot_be_judged_does_not_blank_the_list(tmp_path: Path) -> None:
    from .fakes import StubAdapter
    e = await env(tmp_path)
    e.world.adapters.register(StubAdapter("boom", frozenset({"hf"})))
    items = (await e.client.get(f"{P}/models")).json()["items"]
    assert items[0]["engines_that_fit"] == ["vllm"]
    await e.world.stop()


async def test_models_pages(tmp_path: Path) -> None:
    e = await env(tmp_path)
    page = (await e.client.get(f"{P}/models", params={"limit": 1})).json()
    assert [m["repo_id"] for m in page["items"]] == ["org/m"] and page["next_cursor"] == "1"
    await e.world.stop()


async def test_delete_model_removes_this_machines_cache_only(tmp_path: Path) -> None:
    e = await env(tmp_path)
    r = await e.client.delete(f"{P}/models/org/s")                       # only spark has it
    assert r.status_code == 404 and problem(r)["code"] == "not_found"
    local = (await e.client.get(f"{P}/library", params={"q": "org/m"})).json()["items"][0]["copies"][0]
    e.world.busy[("local", local["path"])] = ["inst_1"]
    r = await e.client.delete(f"{P}/models/org/m")
    assert r.status_code == 409 and problem(r)["code"] == "model_in_use"
    e.world.busy.clear()
    assert (await e.client.delete(f"{P}/models/org/m")).status_code == 204
    assert [m["repo_id"] for m in (await e.client.get(f"{P}/models")).json()["items"]] == ["org/s"]
    assert problem(await e.client.delete(f"{P}/models/org/m"))["code"] == "not_found"
    assert (await e.client.delete(f"{P}/models/..%2Fetc")).status_code in (404, 422)
    r = await e.client.delete(f"{P}/models/a--b/c")
    assert r.status_code == 422                                           # the repo id rule, as in v1
    await e.world.stop()


# ---- PUT /settings: host_image_pins ----------------------------------------------------------------------------------------------------------------
async def test_put_settings_host_image_pins(tmp_path: Path) -> None:
    e = await env(tmp_path)
    assert (await e.client.get(f"{P}/settings")).json()["host_image_pins"] == {}
    pins = {"spark": {"vllm": "vllm/vllm-openai:v0.12.0-arm64"}, "local": {"sglang": "lmsysorg/sglang:v0.5.1"}}
    r = await e.client.put(f"{P}/settings", json={"host_image_pins": pins})
    assert r.status_code == 200 and r.json()["host_image_pins"] == pins
    assert (await e.client.get(f"{P}/settings")).json()["host_image_pins"] == pins
    assert e.world.settings.image_pin("vllm", "spark") == "vllm/vllm-openai:v0.12.0-arm64"
    assert e.world.settings.image_pin("vllm", "local") is None
    r = await e.client.put(f"{P}/settings", json={"idle_ttl_s": 30})                  # another setting leaves the pins alone
    assert r.json()["host_image_pins"] == pins and r.json()["idle_ttl_s"] == 30
    r = await e.client.put(f"{P}/settings", json={"host_image_pins": {}})
    assert r.json()["host_image_pins"] == {}
    for bad in ({"Spark": {"vllm": "a/b:1"}}, {"spark": {"vllm": "a/b:latest"}}, {"spark": {"vllm": "a/b"}},
                {"spark": {"vllm": "a/b:1 --privileged"}}, {"spark": {"bad engine": "a/b:1"}}):
        r = await e.client.put(f"{P}/settings", json={"host_image_pins": bad})
        assert r.status_code == 400 and problem(r)["code"] == "invalid_setting", bad
    r = await e.client.put(f"{P}/settings", json={"host_image_pins": {"spark": "not a map"}})
    assert r.status_code == 422
    assert (await e.client.get(f"{P}/settings")).json()["host_image_pins"] == {}          # a refused change changed nothing
    await e.world.stop()
