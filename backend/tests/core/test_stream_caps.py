"""Review finding L4: the stream caps count per API key as well as in total, streamed chat replies are capped too, a slot
is given back however a stream ends, and the /hf routes (which spend the operator's Hugging Face token) are admin-only.

Streams are driven straight through the ASGI app: TestClient reads every response to its end, so it can neither keep a
stream open nor go away in the middle of one."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine_console.main import create_app
from engine_console.services.common import SseLimiter

from .conftest import Env
from .test_api_smoke import P, problem_code
from .test_nodehttp_chat_bench import engine, ready

SSE = f"{P}/downloads/stream"        # no `once`: it stays open until the client leaves or the idle timeout
CHAT = f"{P}/chat/completions"


class Conn:
    """One request through the ASGI app. `hold=True` makes the client stop reading after the first body chunk (the next
    send blocks, as it would for a reader that stopped); `leave()` makes it go away. `spec` is the ASGI spec version the
    server claims: 2.3 (uvicorn) reports a disconnect through receive(), 2.4 only by failing the next send."""

    def __init__(self, app: FastAPI, method: str, path: str, *, key: str | None = None, body: Any = None,
                 hold: bool = False, spec: str = "2.3", client: str = "127.0.0.1") -> None:
        headers = [(b"host", b"testserver"), (b"x-engine-console", b"1"), (b"content-type", b"application/json")]
        if key:
            headers.append((b"authorization", f"Bearer {key}".encode()))
        self.scope: dict[str, Any] = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": spec}, "http_version": "1.1", "method": method,
            "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
            "headers": headers, "client": (client, 50000), "server": ("testserver", 80)}
        self.body = json.dumps(body).encode() if body is not None else b""
        self.hold, self.spec = hold, spec
        self.gone, self.started, self.first_chunk = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.status = 0
        self.received = b""
        self.error: BaseException | None = None
        self._asked = False
        self.task = asyncio.create_task(self._run(app))

    async def _run(self, app: FastAPI) -> None:
        try:
            await app(self.scope, self._receive, self._send)
        except Exception as e:   # noqa: BLE001 - a client that left mid-write surfaces here (spec 2.4)
            self.error = e

    async def _receive(self) -> dict[str, Any]:
        if not self._asked:
            self._asked = True
            return {"type": "http.request", "body": self.body, "more_body": False}
        await self.gone.wait()
        return {"type": "http.disconnect"}

    async def _send(self, msg: dict[str, Any]) -> None:
        if self.gone.is_set() and self.spec == "2.4":
            raise OSError("the client went away")
        if msg["type"] == "http.response.start":
            self.status = msg["status"]
            self.started.set()
        elif msg["type"] == "http.response.body" and msg.get("body"):
            self.received += msg["body"]
            if self.first_chunk.is_set() and self.hold:
                await self.gone.wait()             # stopped reading: the stream stays open, holding its slot
                if self.spec == "2.4":
                    raise OSError("the client went away")
            self.first_chunk.set()

    async def opened(self) -> int:
        await asyncio.wait_for(self.started.wait(), 5)
        if self.status == 200:
            await asyncio.wait_for(self.first_chunk.wait(), 5)
        return self.status

    def problem(self) -> dict[str, Any]:
        doc: dict[str, Any] = json.loads(self.received)
        return doc

    async def leave(self, nudge: Callable[[], None] | None = None) -> None:
        """Go away. Under spec 2.4 an idle stream notices only at its next write: `nudge` makes it write."""
        self.gone.set()
        if nudge is not None:
            nudge()
        await asyncio.wait_for(self.task, 5)


@pytest.fixture
def capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    """The core test environment with small caps, set the way an operator sets them: in the environment."""
    for k, v in {"MAX_SSE_CONNECTIONS": "3", "MAX_SSE_PER_KEY": "2", "MAX_CHAT_STREAMS": "3",
                 "MAX_CHAT_STREAMS_PER_KEY": "2"}.items():
        monkeypatch.setenv(k, v)
    return Env(tmp_path)


def key(env: Env, name: str, role: str = "viewer") -> tuple[str, str]:
    info = env.container.settings.create_key(name, role)
    assert info.secret is not None
    return info.id, info.secret


async def settle() -> None:
    for _ in range(20):        # let cancelled tasks and async-generator cleanups run
        await asyncio.sleep(0)


# ---- the limiter itself -----------------------------------------------------------------------------------------------------------
def test_limiter_counts_per_key_and_in_total_and_forgets_empty_keys() -> None:
    lim = SseLimiter(3, per_key=2)
    assert lim.acquire("a") and lim.acquire("a") and not lim.acquire("a")      # a's share is full
    assert lim.key_full("a") and not lim.key_full("b")
    assert lim.acquire("b") and not lim.acquire("c")                            # the total is full
    lim.release("a")
    assert lim.acquire("c") and lim.active == 3
    for k in ("a", "b", "c"):
        lim.release(k)
    assert lim.active == 0 and lim._by_key == {}                                # no empty buckets kept per key id
    assert SseLimiter(8).per_key == 8                                           # no per-key share given: only the total


# ---- SSE: one key cannot hold every slot --------------------------------------------------------------------------------------
async def test_one_key_cannot_hold_every_sse_slot(capped: Env) -> None:
    env = capped
    app = create_app(env.cfg, env.container)
    _, a = key(env, "a")
    _, b = key(env, "b")
    held = [Conn(app, "GET", SSE, key=a) for _ in range(2)]
    assert [await c.opened() for c in held] == [200, 200]
    third = Conn(app, "GET", SSE, key=a)
    assert await third.opened() == 429
    p = third.problem()
    # the same code the UI already treats as "do not retry", plus which cap is full
    assert p["code"] == "too_many_streams" and p["scope"] == "key" and p["limit"] == 2 and "this API key" in p["detail"]
    other = Conn(app, "GET", SSE, key=b)
    assert await other.opened() == 200              # another key still gets its streams
    for c in (*held, other):
        await c.leave()
    await settle()
    assert env.container.sse.active == 0
    again = Conn(app, "GET", SSE, key=a)
    assert await again.opened() == 200              # a's slots came back
    await again.leave()


async def test_the_total_sse_cap_still_holds_across_keys(capped: Env) -> None:
    env = capped
    app = create_app(env.cfg, env.container)
    held = [Conn(app, "GET", SSE, key=key(env, f"k{i}")[1]) for i in range(3)]
    assert [await c.opened() for c in held] == [200, 200, 200]
    late = Conn(app, "GET", SSE, key=key(env, "late")[1])
    assert await late.opened() == 429
    assert late.problem()["code"] == "too_many_streams" and late.problem()["scope"] == "total"
    for c in held:
        await c.leave()


async def test_loopback_requests_without_a_key_share_one_bucket(capped: Env) -> None:
    env = capped
    assert env.cfg.trust_loopback          # the core test env trusts loopback: no key needed
    app = create_app(env.cfg, env.container)
    first = [Conn(app, "GET", SSE) for _ in range(2)]
    assert [await c.opened() for c in first] == [200, 200]
    third = Conn(app, "GET", SSE)
    assert await third.opened() == 429 and third.problem()["scope"] == "key"
    keyed = Conn(app, "GET", SSE, key=key(env, "k")[1])
    assert await keyed.opened() == 200     # a key has its own share
    for c in (*first, keyed):
        await c.leave()


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
async def test_an_sse_slot_comes_back_when_the_client_leaves_mid_stream(env: Env, spec: str) -> None:
    app = create_app(env.cfg, env.container)
    c = Conn(app, "GET", SSE, key=key(env, "k")[1], spec=spec)
    assert await c.opened() == 200 and env.container.sse.active == 1
    await c.leave(nudge=lambda: env.container.bus.publish("downloads", {"id": "d1"}))
    await settle()
    assert env.container.sse.active == 0


async def test_an_sse_slot_comes_back_when_the_client_leaves_before_the_first_byte(env: Env) -> None:
    # Spec 2.4 reports a gone client by failing a send. Failing the very first one means the body never starts, so neither
    # its `finally` nor a background task runs: the slot used to leak until the process restarted.
    app = create_app(env.cfg, env.container)
    c = Conn(app, "GET", SSE, key=key(env, "k")[1], spec="2.4")
    c.gone.set()
    await asyncio.wait_for(c.task, 5)
    await settle()
    assert c.status == 0 and env.container.sse.active == 0


# ---- streamed chat replies -------------------------------------------------------------------------------------------------------
async def test_streamed_chat_replies_are_capped_per_key_and_the_non_streaming_path_is_not(capped: Env) -> None:
    env = capped
    inst = await ready(env)
    env.node.http_chunk = 8                 # many small frames: a reply is still streaming while the test looks
    env.node.serve(None, "/v1/chat/completions", engine)
    app = create_app(env.cfg, env.container)
    _, a = key(env, "a")
    _, b = key(env, "b")
    body = {"instance_id": inst.id, "messages": [{"role": "user", "content": "hi"}], "stream": True}
    held = [Conn(app, "POST", CHAT, key=a, body=body, hold=True) for _ in range(2)]
    assert [await c.opened() for c in held] == [200, 200]
    third = Conn(app, "POST", CHAT, key=a, body=body)
    assert await third.opened() == 429
    p = third.problem()
    assert p["code"] == "too_many_chat_streams" and p["scope"] == "key" and p["limit"] == 2
    plain = Conn(app, "POST", CHAT, key=a, body={**body, "stream": False})
    assert await plain.opened() == 200      # a non-streaming reply holds no stream slot
    other = Conn(app, "POST", CHAT, key=b, body=body, hold=True)
    assert await other.opened() == 200
    for c in (*held, other):
        await c.leave()
    await settle()
    assert env.container.chat_streams.active == 0


async def test_the_total_chat_stream_cap_holds_across_keys(capped: Env) -> None:
    env = capped
    inst = await ready(env)
    env.node.http_chunk = 8
    env.node.serve(None, "/v1/chat/completions", engine)
    app = create_app(env.cfg, env.container)
    body = {"instance_id": inst.id, "messages": [{"role": "user", "content": "hi"}], "stream": True}
    first = [Conn(app, "POST", CHAT, key=key(env, f"k{i}")[1], body=body, hold=True) for i in range(3)]
    assert [await c.opened() for c in first] == [200, 200, 200]
    late = Conn(app, "POST", CHAT, key=key(env, "late")[1], body=body)
    assert await late.opened() == 429 and late.problem()["scope"] == "total"
    for c in first:
        await c.leave()


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
async def test_a_chat_slot_comes_back_when_the_client_leaves_and_the_engine_request_is_stopped(env: Env, spec: str) -> None:
    inst = await ready(env)
    env.node.http_chunk = 8
    env.node.serve(None, "/v1/chat/completions", engine)
    app = create_app(env.cfg, env.container)
    body = {"instance_id": inst.id, "messages": [{"role": "user", "content": "hi"}], "stream": True}
    c = Conn(app, "POST", CHAT, key=key(env, "k")[1], body=body, hold=True, spec=spec)
    assert await c.opened() == 200 and env.container.chat_streams.active == 1
    await c.leave()
    await settle()
    assert env.container.chat_streams.active == 0
    assert env.node.streams[-1].cancelled                                  # the node was told to stop the engine request
    assert env.container.usage.query("model")[0].requests == 1            # and the partial reply is still accounted for


async def test_a_chat_slot_comes_back_on_a_normal_end_and_on_a_refusal_before_streaming(env: Env) -> None:
    inst = await ready(env)
    env.node.serve(None, "/v1/chat/completions", engine)
    app = create_app(env.cfg, env.container)
    k = key(env, "k")[1]
    body = {"instance_id": inst.id, "messages": [{"role": "user", "content": "hi"}], "stream": True}
    done = Conn(app, "POST", CHAT, key=k, body=body)
    assert await done.opened() == 200
    await asyncio.wait_for(done.task, 5)
    assert b"[DONE]" in done.received and env.container.chat_streams.active == 0
    missing = Conn(app, "POST", CHAT, key=k, body={**body, "instance_id": "nope"})
    assert await missing.opened() == 404
    await asyncio.wait_for(missing.task, 5)
    env.node.serve(None, "/v1/chat/completions", lambda r: httpx.Response(500, text="engine fell over"))
    failed = Conn(app, "POST", CHAT, key=k, body=body)
    assert await failed.opened() == 502
    await asyncio.wait_for(failed.task, 5)
    assert env.container.chat_streams.active == 0


# ---- /hf/* spends the operator's token: admin only -----------------------------------------------------------------------------------
def test_a_viewer_fit_never_asks_hugging_face(env: Env) -> None:
    env.seed_hub_model("Qwen/Qwen3-32B")
    app = create_app(env.cfg, env.container)
    _, viewer = key(env, "v")
    _, admin = key(env, "adm", "admin")
    body = {"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 8192}}
    with TestClient(app, client=("192.0.2.7", 1)) as remote:
        before = list(env.hub.calls)
        r = remote.post(f"{P}/fit", json=body, headers={"Authorization": f"Bearer {viewer}", "X-Engine-Console": "1"})
        assert r.status_code == 422 and problem_code(r) == "model_info_unavailable", r.text   # not downloaded anywhere
        assert env.hub.calls == before                                   # the operator's token was not spent
        ok = remote.post(f"{P}/fit", json=body, headers={"Authorization": f"Bearer {admin}", "X-Engine-Console": "1"})
        assert ok.status_code == 200, ok.text
        assert env.hub.calls != before


def test_the_hf_routes_are_admin_only(env: Env) -> None:
    env.seed_hub_model("Qwen/Qwen3-32B")
    app = create_app(env.cfg, env.container)
    _, viewer = key(env, "v")
    _, admin = key(env, "adm", "admin")
    with TestClient(app, client=("192.0.2.7", 1)) as remote:          # not loopback: only the keys count
        before = list(env.hub.calls)
        for path in (f"{P}/hf/search", f"{P}/hf/models/Qwen/Qwen3-32B"):
            r = remote.get(path, params={"q": "qwen"}, headers={"Authorization": f"Bearer {viewer}"})
            assert r.status_code == 403 and problem_code(r) == "forbidden", path
        assert env.hub.calls == before                                   # nothing reached Hugging Face for the viewer
        ok = remote.get(f"{P}/hf/search", params={"q": "qwen"}, headers={"Authorization": f"Bearer {admin}"})
        assert ok.status_code == 200 and ok.json()["items"]
        assert remote.get(f"{P}/hf/models/Qwen/Qwen3-32B", headers={"Authorization": f"Bearer {admin}"}).status_code == 200
        # every /hf route, enumerated from OpenAPI so a new one cannot skip the check
        hf = [p for p in app.openapi()["paths"] if p.startswith(f"{P}/hf/")]
        assert len(hf) == 2
        for path in hf:
            concrete = path.replace("{repo_id}", "a/b")
            assert remote.get(concrete, headers={"Authorization": f"Bearer {viewer}"}).status_code == 403, path
