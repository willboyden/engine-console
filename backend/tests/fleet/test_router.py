"""The hosts router through a minimal app: the same dependencies, error handlers and role checks as the real one."""
from __future__ import annotations

import asyncio
import json
import types
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send

from engine_console.api.deps import enforce_roles
from engine_console.api.errors import install_error_handlers
from engine_console.api.routers import hosts as hosts_router
from engine_console.api.routers.hosts import fleet_events, router
from engine_console.config import Settings
from engine_console.domain.fleet import HostConfig
from engine_console.services.common import EventBus, SseLimiter
from engine_console.services.fleet import FleetService
from engine_console.services.settings import Principal

from .builders import disks, hello, mount, root, sample
from .harness import wait_until
from .test_fleet_service import FakeLink

P = "/api/v1"
A = HostConfig(id="gpu-box-2", label="GPU box 2", ssh="ops@10.0.0.5")
L = HostConfig(id="local", label="Workstation")


class AsRole:
    """Stands in for SecurityMiddleware: puts a principal (taken from the X-Test-Role header) on the request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            role = dict(scope["headers"]).get(b"x-test-role", b"admin").decode()
            scope.setdefault("state", {})["principal"] = Principal("tester", role)
        await self.app(scope, receive, send)


class Rig:
    def __init__(self, cfg: Settings, limit: int = 4) -> None:
        self.bus = EventBus()
        self.links: dict[str, FakeLink] = {}

        def factory(h: HostConfig) -> FakeLink:
            link = FakeLink(h, hello=hello(roots=[root("hf", "/data/hf")]), sample=sample(), disks=disks([mount("/"), mount("/data")]))
            link.results["docker.images"] = {"images": [{"ref": "vllm/vllm-openai:v0.1", "size": 5}]}
            self.links[h.id] = link
            return link

        self.fleet = FleetService(cfg, self.bus, hosts=[L, A], link_factory=factory)
        app = FastAPI()
        install_error_handlers(app)
        app.include_router(router, prefix=P, dependencies=[Depends(enforce_roles)])
        app.state.container = types.SimpleNamespace(
            fleet=self.fleet, bus=self.bus, sse=SseLimiter(limit), cfg=types.SimpleNamespace(sse_idle_timeout_s=5.0))
        app.add_middleware(AsRole)
        self.app = app


@pytest.fixture
def rig(make_settings: Callable[..., Settings]) -> Rig:
    return Rig(make_settings())


@pytest.fixture
def client(rig: Rig) -> Iterator[TestClient]:
    with TestClient(rig.app) as c:
        yield c


def problem(r: Any) -> dict[str, Any]:
    assert r.headers["content-type"].startswith("application/problem+json"), r.text
    body: dict[str, Any] = r.json()
    assert body["status"] == r.status_code
    return body


# ── GET /hosts, /hosts/{id} ─────────────────────────────────────────────────────────────────────────────
def test_list_hosts_is_a_page_of_summaries(client: TestClient) -> None:
    r = client.get(f"{P}/hosts")
    assert r.status_code == 200
    body = r.json()
    assert body["next_cursor"] is None and [h["id"] for h in body["items"]] == ["local", "gpu-box-2"]
    first = body["items"][0]
    assert first["label"] == "Workstation" and first["remote"] is False and first["link"]["state"] == "up"
    assert first["gpus"][0]["total_gib"] == 48.0 and first["memory"]["total_gib"] == 256.0 and first["counts"]["models"] == 0
    assert body["items"][1]["remote"] is True
    assert "disks" not in first                                      # the list carries summaries, not details


def test_host_detail(client: TestClient) -> None:
    r = client.get(f"{P}/hosts/gpu-box-2")
    assert r.status_code == 200
    d = r.json()
    assert d["id"] == "gpu-box-2" and [m["mount"] for m in d["disks"]] == ["/", "/data"]
    assert d["roots"][0]["id"] == "hf" and d["roots"][0]["mount"]["mount"] == "/data" and d["roots"][0]["visible_on"] == ["gpu-box-2"]
    assert d["images"][0]["ref"] == "vllm/vllm-openai:v0.1" and d["addresses"] == ["10.0.0.5/24 (eth0)"] and d["scripts"] == []


def test_an_unknown_host_is_a_404_problem(client: TestClient) -> None:
    r = client.get(f"{P}/hosts/nope")
    assert r.status_code == 404 and problem(r)["code"] == "unknown_host"
    assert problem(client.post(f"{P}/hosts/nope/reconnect"))["code"] == "unknown_host"


def test_the_link_error_and_notes_are_visible_to_the_page(rig: Rig, client: TestClient) -> None:
    rig.links["gpu-box-2"].state = "down"
    rig.links["gpu-box-2"].error = "ssh refused the login to ops@10.0.0.5"
    h = client.get(f"{P}/hosts/gpu-box-2").json()
    assert h["link"]["state"] == "down" and h["link"]["error"] == "ssh refused the login to ops@10.0.0.5"
    assert h["gpus"][0]["used_gib"] is None and h["memory"] is None      # nothing live from a host that is down


# ── POST /hosts/{id}/reconnect ──────────────────────────────────────────────────────────────────────────
def test_reconnect_is_admin_only_and_returns_the_summary(rig: Rig, client: TestClient) -> None:
    r = client.post(f"{P}/hosts/gpu-box-2/reconnect")
    assert r.status_code == 200 and r.json()["id"] == "gpu-box-2" and rig.links["gpu-box-2"].reconnects == 1
    for role in ("viewer", "public"):
        denied = client.post(f"{P}/hosts/gpu-box-2/reconnect", headers={"x-test-role": role})
        assert denied.status_code == 403 and problem(denied)["code"] == "forbidden"
    assert rig.links["gpu-box-2"].reconnects == 1                    # the refused requests did nothing


def test_the_route_checks_for_admin_itself(rig: Rig) -> None:
    """The real app also puts enforce_roles in front of every router; the route must not depend on that."""
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(router, prefix=P)                              # no router-level dependency at all
    app.state.container = rig.app.state.container
    app.add_middleware(AsRole)
    with TestClient(app) as c:
        assert c.post(f"{P}/hosts/gpu-box-2/reconnect", headers={"x-test-role": "viewer"}).status_code == 403
        assert c.post(f"{P}/hosts/gpu-box-2/reconnect", headers={"x-test-role": "public"}).status_code == 403
        assert rig.links["gpu-box-2"].reconnects == 0
        assert c.post(f"{P}/hosts/gpu-box-2/reconnect", headers={"x-test-role": "admin"}).status_code == 200


def test_reconnecting_a_disabled_host_is_a_409(rig: Rig, client: TestClient) -> None:
    rig.links["local"].state = "disabled"
    r = client.post(f"{P}/hosts/local/reconnect")
    assert r.status_code == 409 and problem(r)["code"] == "host_disabled"


def test_viewers_may_read_hosts(client: TestClient) -> None:
    assert client.get(f"{P}/hosts", headers={"x-test-role": "viewer"}).status_code == 200
    assert client.get(f"{P}/hosts/local", headers={"x-test-role": "viewer"}).status_code == 200


# ── GET /fleet/stream ───────────────────────────────────────────────────────────────────────────────────
def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        assert lines[0].startswith("event: ") and lines[1].startswith("data: ") and len(lines) == 2, block
        out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


def test_stream_once_sends_every_host_and_closes(client: TestClient) -> None:
    r = client.get(f"{P}/fleet/stream?once=true")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    events = parse_sse(r.text)
    assert [(e, d["id"]) for e, d in events] == [("host", "local"), ("host", "gpu-box-2")]
    assert events[0][1]["link"]["state"] == "up" and events[0][1]["gpus"]


def test_the_stream_counts_against_the_connection_cap(make_settings: Callable[..., Settings]) -> None:
    capped = Rig(make_settings(), limit=0)
    with TestClient(capped.app) as c:
        r = c.get(f"{P}/fleet/stream?once=true")
    assert r.status_code == 429 and problem(r)["code"] == "too_many_streams"


def test_the_slot_is_given_back_when_the_stream_ends(make_settings: Callable[..., Settings]) -> None:
    one = Rig(make_settings(), limit=1)
    with TestClient(one.app) as c:
        for _ in range(3):
            assert c.get(f"{P}/fleet/stream?once=true").status_code == 200
    assert one.app.state.container.sse.active == 0


# ── the event generator itself (an endless stream cannot be read through TestClient) ────────────────────
async def next_chunk(gen: Any, timeout: float = 5.0) -> tuple[str, dict[str, Any]]:
    chunk = await asyncio.wait_for(gen.__anext__(), timeout)
    (event,) = parse_sse(chunk)
    return event


async def test_events_start_with_a_snapshot_then_report_changes(rig: Rig) -> None:
    gen = fleet_events(rig.fleet, rig.bus, poll_s=0.05, refresh_s=60)
    try:
        first = [await next_chunk(gen), await next_chunk(gen)]
        assert [d["id"] for _, d in first] == ["local", "gpu-box-2"]
        quiet = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0.3)
        assert not quiet.done()                                      # nothing changed: nothing is sent
        rig.links["gpu-box-2"].state = "down"
        rig.links["gpu-box-2"].error = "the node exited (3)"
        event, d = await next_chunk(_Resume(quiet))
        assert event == "host" and d["id"] == "gpu-box-2" and d["link"]["state"] == "down" and d["link"]["error"] == "the node exited (3)"
    finally:
        await gen.aclose()


class _Resume:
    """Lets a test finish reading an `__anext__` that is already in flight."""

    def __init__(self, fut: asyncio.Future[str]) -> None:
        self.fut = fut

    async def __anext__(self) -> str:
        return await self.fut


async def test_a_change_to_live_numbers_is_sent_without_any_link_event(rig: Rig) -> None:
    gen = fleet_events(rig.fleet, rig.bus, poll_s=0.05, refresh_s=60)
    try:
        await next_chunk(gen)
        await next_chunk(gen)
        rig.links["local"].data["sample"] = sample(gpus={"GPU-A": {"util_pct": 99.0, "mem_used_bytes": 1}, "GPU-B": {}})
        event, d = await next_chunk(gen)
        assert d["id"] == "local" and d["gpus"][0]["util_pct"] == 99.0
    finally:
        await gen.aclose()


async def test_every_host_is_resent_at_least_every_refresh_interval(rig: Rig) -> None:
    gen = fleet_events(rig.fleet, rig.bus, poll_s=0.05, refresh_s=0.2)
    try:
        ids = [(await next_chunk(gen))[1]["id"] for _ in range(6)]
    finally:
        await gen.aclose()
    assert ids[:2] == ["local", "gpu-box-2"] and sorted(ids[2:4]) == ["gpu-box-2", "local"] and len(ids) == 6   # unchanged hosts, resent


async def test_the_default_refresh_is_five_seconds() -> None:
    assert hosts_router.STREAM_REFRESH_S == 5.0 and hosts_router.STREAM_POLL_S <= 1.0


async def test_a_bus_event_wakes_the_stream_before_the_next_poll(rig: Rig) -> None:
    gen = fleet_events(rig.fleet, rig.bus, poll_s=30.0, refresh_s=300)
    try:
        await next_chunk(gen)
        await next_chunk(gen)
        waiting = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0.1)                                      # the generator is now parked on the bus wake-up
        rig.links["gpu-box-2"].state = "stale"
        rig.bus.publish("fleet", {"id": "gpu-box-2"})                 # what a link does on a state change
        chunk = await asyncio.wait_for(waiting, 2.0)
        (event,) = parse_sse(chunk)
        assert event[1]["id"] == "gpu-box-2" and event[1]["link"]["state"] == "stale"
    finally:
        await gen.aclose()


async def test_closing_the_stream_unsubscribes_from_the_bus(rig: Rig) -> None:
    gen = fleet_events(rig.fleet, rig.bus, poll_s=0.05, refresh_s=60)
    await next_chunk(gen)
    await next_chunk(gen)
    await asyncio.sleep(0.15)                                         # parked in its wait: the subscription exists
    assert len(rig.bus._subs["fleet"]) == 1
    await gen.aclose()
    await wait_until(lambda: len(rig.bus._subs["fleet"]) == 0, what="the subscription to go away")


async def test_cancelling_the_reader_cleans_up_too(rig: Rig) -> None:
    gen = fleet_events(rig.fleet, rig.bus, poll_s=0.05, refresh_s=60)

    async def read_forever() -> None:
        async for _ in gen:
            pass

    task = asyncio.create_task(read_forever())
    await wait_until(lambda: rig.bus._subs.get("fleet"), what="the subscription")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await wait_until(lambda: len(rig.bus._subs["fleet"]) == 0, what="the subscription to go away")


async def test_once_never_subscribes(rig: Rig) -> None:
    chunks = [c async for c in fleet_events(rig.fleet, rig.bus, once=True)]
    assert len(chunks) == 2 and not rig.bus._subs.get("fleet")
