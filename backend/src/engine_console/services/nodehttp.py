"""An engine's HTTP API, reached through its host's node (`http.request`), for chat, benchmarks and health probes.

The node connects to 127.0.0.1:<port> on ITS machine and relays: one event {status, headers}, the body as DATA
frames, then a result. A request body is uploaded with `write()` + `end()` first. Nothing here can reach an address
other than a loopback port the node's policy allows (the node refuses anything else); a remote engine therefore
needs no network exposure at all.

Everything coming back is untrusted: bodies are read under a cap, lines are decoded incrementally."""
from __future__ import annotations

import asyncio
import codecs
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from engine_console.domain.errors import UpstreamError
from engine_console.domain.fleet import FleetPort, NodeError, NodeStream, StreamItem
from engine_console.services.nodeerr import SLACK_S, NodeCaller, shielded

UPLOAD_CHUNK = 256 * 1024            # well under the protocol's 1 MiB DATA frame
MAX_BUFFERED = 16 * 1024 * 1024      # a non-streaming engine answer larger than this is refused, not buffered
JSON_HEADERS = {"content-type": "application/json", "accept": "application/json"}
MAX_LINE = 1024 * 1024               # characters in one streamed line (an SSE event is a few hundred)
MAX_STREAM = 256 * 1024 * 1024       # bytes in one streamed answer
_EOL = re.compile(r"\r\n|\r|\n")
_LONG_LINE = f"the engine sent a line longer than {MAX_LINE // 1024} KiB; the stream was cut off"


@dataclass
class HttpResult:
    status: int
    headers: dict[str, str]
    body: bytes


@dataclass
class HttpStream:
    """An open engine response. Iterate `chunks()` / `lines()` once; always `aclose()` (cancels the node request when
    the caller stops early, e.g. a browser that went away mid-stream)."""
    status: int
    headers: dict[str, str]
    _stream: NodeStream
    _it: AsyncIterator[StreamItem]
    _caller: NodeCaller
    _host_id: str
    _done: bool = False
    _decoder: codecs.IncrementalDecoder = field(default_factory=lambda: codecs.getincrementaldecoder("utf-8")("replace"))

    async def chunks(self) -> AsyncIterator[bytes]:
        try:
            async for item in self._it:
                if item.kind == "data" and item.data:
                    yield item.data
            try:
                await self._stream.result()
            except NodeError as e:
                raise self._caller.problem(e) from None
            self._done = True
        finally:
            await self.aclose()

    async def lines(self) -> AsyncIterator[str]:
        """Text lines (no terminator) split on \\n, \\r\\n or \\r, across chunk boundaries.

        Linear in the stream: each chunk is scanned once and a partial line is kept as pieces, never re-joined and
        re-scanned (security review M2: a hostile engine streaming bytes with no newline made this quadratic). A line
        over MAX_LINE characters or a stream over MAX_STREAM bytes is cut off with an UpstreamError."""
        pending: list[str] = []
        plen = total = 0
        after_cr = False                     # the last text ended with \r: a leading \n belongs to that terminator
        async for chunk in self.chunks():
            total += len(chunk)
            if total > MAX_STREAM:
                raise UpstreamError(f"the engine's stream passed {MAX_STREAM // 2**20} MiB; it was cut off")
            text = self._decoder.decode(chunk)
            if not text:
                continue
            if after_cr and text[0] == "\n":
                text = text[1:]
            after_cr = text.endswith("\r")
            segs = _EOL.split(text)
            if len(segs) > 1:
                pending.append(segs[0])
                if plen + len(segs[0]) > MAX_LINE:
                    raise UpstreamError(_LONG_LINE)
                yield "".join(pending)
                for seg in segs[1:-1]:
                    if len(seg) > MAX_LINE:
                        raise UpstreamError(_LONG_LINE)
                    yield seg
                pending, plen = [], 0
            pending.append(segs[-1])
            plen += len(segs[-1])
            if plen > MAX_LINE:
                raise UpstreamError(_LONG_LINE)
        tail = "".join(pending) + self._decoder.decode(b"", final=True)
        if tail:
            yield tail

    async def read(self, limit: int = MAX_BUFFERED) -> bytes:
        out = bytearray()
        try:
            async for chunk in self.chunks():
                out += chunk
                if len(out) > limit:
                    raise UpstreamError(f"the engine's answer is larger than {limit // 1024} KiB; refusing to buffer it")
        finally:
            await self.aclose()   # over the cap: tell the node to stop sending (a no-op once the response is complete)
        return bytes(out)

    async def aclose(self) -> None:
        if not self._done:
            self._done = True
            await shielded(self._stream.cancel())


class NodeHttp:
    def __init__(self, fleet: FleetPort) -> None:
        self._caller = NodeCaller(fleet)

    async def open(self, host_id: str, port: int, method: str, path: str, *, body: bytes | None = None,
                   headers: dict[str, str] | None = None, timeout_s: float = 600.0) -> HttpStream:
        """Send the request and wait for the response head (status + headers); the body is left to read."""
        args: dict[str, Any] = {"port": int(port), "method": method, "path": path, "timeout_s": timeout_s}
        if headers:
            args["headers"] = headers
        if body:
            args["body_len"] = len(body)
        stream = self._caller.open(host_id, "http.request", args)
        try:
            return await asyncio.wait_for(self._start(stream, host_id, body), timeout_s + SLACK_S)
        except TimeoutError:
            await shielded(stream.cancel())
            raise self._caller.problem(NodeError(host_id, "timeout", f"the engine did not answer in {timeout_s:.0f}s")) from None
        except BaseException:
            await shielded(stream.cancel())
            raise

    async def _start(self, stream: NodeStream, host_id: str, body: bytes | None) -> HttpStream:
        try:
            if body:
                for i in range(0, len(body), UPLOAD_CHUNK):
                    await stream.write(body[i:i + UPLOAD_CHUNK])
                await stream.end()
            it = stream.__aiter__()
            while True:
                try:
                    item = await it.__anext__()
                except StopAsyncIteration:
                    await stream.result()   # raises the node's reason when it refused; an answer without a head is a fault
                    raise NodeError(host_id, "http_error", "the node answered without an HTTP status") from None
                ev = item.event if item.kind == "event" else None
                if ev and isinstance(ev.get("status"), int):
                    hdrs = ev.get("headers")
                    return HttpStream(int(ev["status"]), {str(k).lower(): str(v) for k, v in hdrs.items()}
                                      if isinstance(hdrs, dict) else {}, stream, it, self._caller, host_id)
        except NodeError as e:
            raise self._caller.problem(e) from None

    async def request(self, host_id: str, port: int, method: str, path: str, *, body: bytes | None = None,
                      headers: dict[str, str] | None = None, timeout_s: float = 30.0,
                      max_bytes: int = MAX_BUFFERED) -> HttpResult:
        """One buffered request/response (health probes, non-streaming chat)."""
        resp = await self.open(host_id, port, method, path, body=body, headers=headers, timeout_s=timeout_s)
        data = await resp.read(max_bytes)
        return HttpResult(resp.status, resp.headers, data)
