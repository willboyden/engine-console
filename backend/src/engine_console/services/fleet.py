"""The hub's side of the node protocol: one `NodeLink` per machine, owned by the `FleetService`.

A node is an `ecnode --stdio` process: a local subprocess for host "local", `ssh` with the hub's restricted key for
every other machine. The hub talks to it in length-prefixed frames (`ecnode.proto`, the locked wire contract) and
treats everything it says as untrusted: each pushed message is rebuilt from a whitelist (`proto.sanitize_*`), a node
that floods, goes silent, breaks the framing or sends more data than the credit it was granted is dropped, and
whatever goes wrong the pending requests fail with `NodeUnavailable` instead of hanging.

Everything runs on the event loop; nothing here blocks it. Every wait has a timeout, no shell is involved, and the
ssh command line and the key path are never logged above debug level.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
import re
import signal
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ecnode import proto

from engine_console.config import Settings
from engine_console.domain.errors import Conflict, NotFound
from engine_console.domain.fleet import (
    LOCAL,
    HostConfig,
    HostDetail,
    HostSummary,
    ImageInfo,
    LinkState,
    LinkStatus,
    NodeClient,
    NodeError,
    NodeKind,
    NodeStream,
    NodeUnavailable,
    StreamItem,
)
from engine_console.services import fleet_status
from engine_console.services.common import EventBus, scrub
from engine_console.services.hosts import (
    HostsError,
    default_hosts,
    load_hosts,
    local_argv,
    local_env,
    ssh_argv,
    ssh_env,
)

log = logging.getLogger("engine_console.fleet")

READ_LIMIT = proto.MAX_DATA + 4096     # StreamReader buffer: a whole DATA frame fits, so reading it never stalls
STDERR_KEEP = 2048                     # bytes of the node's stderr kept to explain why a link dropped
MAX_STREAM_EVENTS = 10_000             # events queued for a consumer that is not reading; more are dropped
CREDIT_BATCH = proto.WINDOW // 4       # credit goes back to the node in chunks this big, as the consumer reads
MAX_CREDIT = 1 << 30                   # a node cannot make the hub's send window absurd
ORPHAN_MAX_S = 600.0                   # a cancelled request the node never answered is forgotten after this
NODE_LOG_BURST = 20                    # node `log` messages written to the hub's log per 10 s, the rest are dropped
IMAGES_TTL_S = 60.0
_CODE = re.compile(r"[a-z][a-z0-9_]{0,31}")


@dataclass(frozen=True)
class LinkTiming:
    """Every duration of the link. The defaults are the contract's; tests shrink them so a drop, a silence or a
    backoff does not take half a minute."""
    hello_timeout_s: float = proto.HELLO_TIMEOUT_S
    ping_s: float = proto.PING_S
    silent_s: float = proto.SILENT_S
    stale_s: float = 3 * proto.SAMPLE_S         # no sample for this long: the host shows as `stale`
    backoff_min_s: float = 1.0
    backoff_max_s: float = 60.0
    stable_s: float = 60.0                      # up this long: the next failure starts the backoff over
    fix_retry_s: float = 300.0                  # needs a person (login, host key, install, version): rarely retried
    flood_bytes: int = 20 * 1024 * 1024         # of JSON messages from one node ...
    flood_window_s: float = 10.0                # ... in this window
    flood_frames: int = 50_000                  # JSON messages (and empty DATA frames) in that window
    exit_grace_s: float = 3.0                   # after the node's pipes close: time to exit on its own


def _text(v: object, limit: int = 300) -> str:
    """Untrusted text made safe for a message or a log line: secrets masked, control characters (newlines
    included) turned into spaces, whitespace collapsed, length capped."""
    s = "".join(ch if ch.isprintable() else " " for ch in scrub(str(v)))
    return " ".join(s.split())[:limit]


def _code(v: object) -> str:
    """A node error code. Well-formed codes pass (a newer node may add some); anything else is `internal`."""
    return v if isinstance(v, str) and _CODE.fullmatch(v) else proto.INTERNAL


def _kill_proc(p: asyncio.subprocess.Process | None) -> None:
    """SIGKILL the child's whole process group (it is its own session, so ssh and anything it spawned go too)."""
    if p is None or p.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        # never signal a group that is not the child's own: that could be the hub's
        if os.getpgid(p.pid) == p.pid:
            os.killpg(p.pid, signal.SIGKILL)
        else:
            p.kill()


class _Truncated(proto.ProtoError):
    """The stream ended inside a frame: the far side died mid-write. Its exit status explains more than the framing."""


async def _read_frame(r: asyncio.StreamReader) -> tuple[proto.Frame, int] | None:
    """`proto.read_frame_async` plus the frame's size on the wire, which the flood limit needs. Same primitives,
    same errors: a clean end between frames is None, anything else broken is ProtoError."""
    try:
        hdr = await r.readexactly(5)
    except asyncio.IncompleteReadError as e:
        if not e.partial:
            return None
        raise _Truncated("stream ended inside a frame header") from None
    kind, n = proto.parse_header(hdr)
    try:
        body = await r.readexactly(n)
    except asyncio.IncompleteReadError:
        raise _Truncated("stream ended inside a frame") from None
    return proto.decode_body(kind, body), 5 + n


async def _drain_stderr(stream: asyncio.StreamReader, tail: bytearray) -> None:
    with contextlib.suppress(Exception):    # losing the diagnostics must never take the link down
        while chunk := await stream.read(4096):
            tail += chunk
            del tail[:-STDERR_KEEP]


def explain_exit(host: HostConfig, rc: int | None, stderr: str, remote_cmd: str | None,
                 fix_after_s: float, mask: Sequence[str] = ()) -> tuple[str, float | None]:
    """Why a node process ended, in words a person can act on, and a forced retry delay (None: ordinary backoff).
    Failures that only a person can fix (login refused, host key, node not installed) retry rarely. Anything in
    `mask` (the hub key's path: ssh quotes it in its complaints) is replaced before the text goes anywhere."""
    for secret in mask:
        if secret:
            stderr = stderr.replace(secret, "<hub key>")
    text = stderr.strip()
    last = _text(text.splitlines()[-1]) if text else ""
    dest = host.ssh
    if dest is not None and rc == 255:
        if "Permission denied" in text:
            return (f"ssh refused the login to {dest}: the hub's key is not authorized there. Run "
                    f"`engine-console hosts init-key --from <this machine's address>` and add the line it prints to "
                    f"authorized_keys of the login user on {host.label}", fix_after_s)
        if "UNPROTECTED PRIVATE KEY" in text or "bad permissions" in text:
            return "the hub's ssh key file is readable by other users, so ssh refuses it: chmod 600 the key", fix_after_s
        if "Host key verification failed" in text or "IDENTIFICATION HAS CHANGED" in text:
            return (f"the host key of {dest} is unknown or has changed. Check it, then connect once by hand to "
                    f"trust it: ssh {dest} true", fix_after_s)
        if "Could not resolve hostname" in text:
            return f"cannot resolve {dest}: check the name in hosts.json", None
        if any(s in text for s in ("No route to host", "Connection timed out", "Connection refused",
                                   "Network is unreachable", "Operation timed out")):
            return f"cannot reach {dest}: {last}", None
        return f"ssh to {dest} failed: {last or 'exit 255'}", None
    if dest is not None and rc == 127:       # the remote shell's "command not found"; matching text would also match a node's own errors
        where = f" (expected {remote_cmd.split()[0]})" if remote_cmd else ""
        return f"the ecnode node is not installed on {host.label}{where}: run its install.sh there", fix_after_s
    if dest is None and "No module named" in text:
        return f"the ecnode package is missing from the console's Python environment: {last}", None
    if rc is not None and rc > 0:
        return f"the node exited ({rc}){': ' + last if last else ''}", None
    if rc is not None and rc < 0 and rc != -signal.SIGKILL:
        return f"the node was stopped by signal {-rc}{': ' + last if last else ''}", None
    return last or "the connection closed", None


# ── requests and streams ─────────────────────────────────────────────────────────────────────────────────
class _Request:
    """One request in flight, seen from the hub: its answer, its stream of events and DATA, and both credit windows."""
    __slots__ = ("rid", "op", "done", "res", "unavailable", "queue", "queued_events", "cancel_sent", "cancel_at",
                 "uploading", "ended", "send_credit", "credit_ev", "recv_window", "unacked")

    def __init__(self, rid: int, op: str, streaming: bool) -> None:
        self.rid, self.op = rid, op
        self.done = asyncio.Event()
        self.res: dict[str, Any] | None = None
        self.unavailable: str | None = None           # set when the link dropped before the node answered
        self.queue: asyncio.Queue[StreamItem | None] | None = asyncio.Queue() if streaming else None
        self.queued_events = 0
        self.cancel_sent = False
        self.cancel_at = 0.0
        self.uploading = False
        self.ended = False
        self.send_credit = proto.WINDOW               # DATA the hub may still send (the node's credit to us)
        self.credit_ev = asyncio.Event()              # set when more credit arrives or the request ends
        self.recv_window = proto.WINDOW               # DATA the node may still send (our credit to it)
        self.unacked = 0                              # bytes the consumer read that the node has not been credited for

    @property
    def finished(self) -> bool:
        return self.done.is_set()

    def finish(self, res: dict[str, Any] | None, unavailable: str | None = None) -> None:
        if self.done.is_set():
            return
        self.res, self.unavailable = res, unavailable
        self.done.set()
        self.credit_ev.set()
        if self.queue is not None:
            self.queue.put_nowait(None)               # end-of-stream marker, behind everything already queued

    def outcome(self, host_id: str) -> dict[str, Any]:
        """The result, or the NodeError / NodeUnavailable the request ended with."""
        if self.unavailable is not None:
            raise NodeUnavailable(host_id, self.unavailable)
        res = self.res or {}
        if res.get("ok") is True:
            result = res.get("result")
            return result if isinstance(result, dict) else {}
        raw = res.get("error")
        err: dict[str, Any] = raw if isinstance(raw, dict) else {}
        raise NodeError(host_id, _code(err.get("code")), _text(err.get("message") or "the node refused", 1000))


class FailedStream:
    """A stream that never started (link down, too many requests, bad arguments). `open()` returns one instead of
    raising, so a caller handles every failure in one place. Like a request the node refused, it yields nothing and
    `result()` raises the reason (so do `write()` and `end()`)."""

    def __init__(self, error: NodeError) -> None:
        self._error = error

    def __aiter__(self) -> AsyncIterator[StreamItem]:
        return self._nothing()

    @staticmethod
    async def _nothing() -> AsyncIterator[StreamItem]:
        return
        yield   # unreachable: it only makes this an (empty) async generator

    async def write(self, data: bytes) -> None:
        raise self._error

    async def end(self) -> None:
        raise self._error

    async def result(self) -> dict[str, Any]:
        raise self._error

    async def cancel(self) -> None:
        return None


class _Stream:
    """`NodeStream` over one request. Reading DATA hands credit back to the node, so a slow consumer slows only its
    own stream: nothing here ever blocks the link's reader.

    Iteration yields the node's events and DATA in order and simply ENDS when the request does: at the node's res
    (whether it succeeded or not) or when the link drops. Call `result()` afterwards: it returns the result or raises
    the NodeError / NodeUnavailable the request ended with. Stopping the iteration early cancels the request."""

    def __init__(self, link: NodeLink, req: _Request) -> None:
        self._link, self._req = link, req
        self._wlock = asyncio.Lock()
        self._loop = asyncio.get_running_loop()

    def __aiter__(self) -> AsyncIterator[StreamItem]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[StreamItem]:
        req, q = self._req, self._req.queue
        assert q is not None
        try:
            while True:
                item = await q.get()
                if item is None:
                    q.put_nowait(None)                # a second iteration ends at once instead of waiting forever
                    break
                if item.kind == "data":
                    self._consumed(len(item.data))
                else:
                    req.queued_events -= 1
                yield item
        finally:
            if not req.finished:                      # the consumer stopped early: tell the node, do not leak it
                self._cancel_nowait()

    def _consumed(self, n: int) -> None:
        req = self._req
        req.unacked += n
        if req.unacked >= CREDIT_BATCH and not req.finished:
            grant, req.unacked = req.unacked, 0
            req.recv_window += grant
            self._link.send_nowait(proto.credit(req.rid, grant))

    async def write(self, data: bytes) -> None:
        """Upload bytes as DATA frames of at most MAX_DATA, waiting for the node's credit when its window is used up.
        One writer at a time (a lock keeps one call's bytes contiguous)."""
        req = self._req
        async with self._wlock:
            if req.ended:
                raise NodeError(self._link.host.id, proto.BAD_ARGS, "write after end()")
            req.uploading = True
            view = memoryview(data)
            pos = 0
            while pos < len(view):
                if req.finished:
                    req.outcome(self._link.host.id)   # the node answered (usually refused) before the upload was done
                    raise NodeError(self._link.host.id, proto.CONFLICT, "the node already finished this request")
                if req.send_credit <= 0:
                    req.credit_ev.clear()
                    await req.credit_ev.wait()
                    continue
                n = min(len(view) - pos, proto.MAX_DATA, req.send_credit)
                req.send_credit -= n
                await self._link.send_data(proto.encode_data(req.rid, bytes(view[pos:pos + n])))
                pos += n

    async def end(self) -> None:
        req = self._req
        if req.ended:
            return
        if req.finished:
            req.outcome(self._link.host.id)
            return
        req.ended = True
        self._link.send_nowait(proto.end(req.rid))

    async def result(self) -> dict[str, Any]:
        await self._req.done.wait()
        return self._req.outcome(self._link.host.id)

    async def cancel(self) -> None:
        self._cancel_nowait()

    def _cancel_nowait(self) -> None:
        self._link.cancel_request(self._req)

    def __del__(self) -> None:
        # An upload that was started and then dropped without end() or cancel() would hold a request slot on the
        # node (and an open partial file) until the link drops. Nothing else is cancelled here: a request whose
        # consumer simply stopped waiting for its answer is still wanted.
        req = getattr(self, "_req", None)
        if req is None or not req.uploading or req.ended or req.finished:
            return
        with contextlib.suppress(Exception):        # at interpreter exit the loop may be gone: nothing left to tell
            try:
                running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is self._loop:
                self._link.cancel_request(req)
            else:                                   # dropped on another thread: transport writes belong to the loop's thread
                self._loop.call_soon_threadsafe(self._link.cancel_request, req)


@dataclass
class _End:
    """How one connection attempt ended."""
    error: str
    wait: float | None = None           # forced retry delay; None = ordinary exponential backoff
    incompatible: bool = False
    up_s: float = 0.0                   # how long the link was up


@dataclass
class _Talk:
    err: str | None = None              # why the conversation ended, when we know (None: the node closed its pipes)
    wait: float | None = None
    incompatible: bool = False
    up_s: float = 0.0
    was_up: bool = False


class _NodeFatal(Exception):
    """The node said `fatal` before its hello."""


class NodeLink:
    """One machine's node, kept connected: spawn, handshake, read, ping, and reconnect with backoff, forever.

    Implements `domain.fleet.NodeClient`. `argv` is a command (or a function returning one, called on every
    attempt, so a key created after the first failure is picked up); it may raise `HostsError`, which is shown as
    the link's error and retried rarely because only a person can fix it."""

    def __init__(self, host: HostConfig, argv: Sequence[str] | Callable[[], Sequence[str]],
                 env: Mapping[str, str] | None, bus: EventBus, *, timing: LinkTiming | None = None,
                 on_change: Callable[[], None] | None = None, remote_cmd: str | None = None,
                 mask: Sequence[str] = (), clock: Callable[[], float] = time.monotonic,
                 wallclock: Callable[[], float] = time.time) -> None:
        self.host = host
        self._mask = tuple(mask)
        if callable(argv):
            self._argv: Callable[[], Sequence[str]] = argv
        else:
            fixed = list(argv)
            self._argv = lambda: fixed
        self._env = dict(env) if env is not None else None
        self._bus = bus
        self._t = timing or LinkTiming()
        self._on_change = on_change
        self._remote_cmd = remote_cmd
        self._clock, self._wallclock = clock, wallclock
        self._state: LinkState = "connecting"
        self._error: str | None = None
        self._notes: list[str] = []
        self._retry_at: float | None = None
        self._up_mono: float | None = None
        self._connected_wall: float | None = None
        self._last_seen: float | None = None
        self._last_rx = clock()
        self._last_sample_mono: float | None = None
        self._rtt_ms: int | None = None
        self._skew_s: float | None = None
        self._latest: dict[str, dict[str, Any]] = {}
        self._pending: dict[int, _Request] = {}
        self._next_id = 1
        self._pings: dict[int, float] = {}
        self._next_ping = 1
        self._proc: asyncio.subprocess.Process | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._stopping = False
        self._drop_reason: str | None = None
        self._backoff = self._t.backoff_min_s
        self._log_window = clock()
        self._log_count = 0
        self.last_wait_s: float | None = None        # the delay chosen before the next attempt (diagnostics, tests)

    # ── NodeClient ───────────────────────────────────────────────────────────────────────────────────────
    def link(self) -> LinkStatus:
        now = self._clock()
        state = self._state
        if state == "up":
            ref = self._last_sample_mono if self._last_sample_mono is not None else self._up_mono
            if ref is not None and now - ref > self._t.stale_s:
                state = "stale"
        live = state in ("up", "stale")
        return LinkStatus(
            state=state, error=self._error, retry_in_s=None if self._retry_at is None else max(0, round(self._retry_at - now)),
            notes=list(self._notes), last_seen=self._last_seen, connected_since=self._connected_wall if live else None,
            rtt_ms=self._rtt_ms if live else None, skew_s=self._skew_s if live else None)

    def latest(self, kind: NodeKind) -> dict[str, Any] | None:
        return self._latest.get(kind)

    @property
    def inflight(self) -> int:
        """Requests the node is (as far as the hub knows) still working on, cancelled ones included."""
        return len(self._pending)

    async def call(self, op: str, args: dict[str, Any] | None = None, *, timeout: float = 30.0) -> dict[str, Any]:
        req = self._begin(op, args, streaming=False)
        try:
            await asyncio.wait_for(req.done.wait(), timeout)
        except TimeoutError:
            self.cancel_request(req)
            raise NodeError(self.host.id, proto.TIMEOUT,
                            f"{op} did not answer within {timeout:g} s; it may or may not have taken effect") from None
        except asyncio.CancelledError:
            self.cancel_request(req)                  # the caller went away: the node should stop working for it
            raise
        result = req.outcome(self.host.id)
        if op == "node.info":
            self._refresh_hello(result)
        return result

    def open(self, op: str, args: dict[str, Any] | None = None) -> NodeStream:
        try:
            return _Stream(self, self._begin(op, args, streaming=True))
        except NodeError as e:
            return FailedStream(e)

    # ── lifecycle ────────────────────────────────────────────────────────────────────────────────────────
    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"link-{self.host.id}")

    async def stop(self) -> None:
        """Close the node's stdin (it exits on EOF, and a script it is running is not killed under it), give it a
        moment, then kill what is left."""
        self._stopping = True
        self._wake.set()
        proc, task = self._proc, self._task
        if self._writer is not None:
            with contextlib.suppress(Exception):
                self._writer.close()
        if task is not None:
            await asyncio.wait({task}, timeout=self._t.exit_grace_s)
            if not task.done():
                _kill_proc(proc)
                await asyncio.wait({task}, timeout=5.0)
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        _kill_proc(self._proc)
        self._fail_all("the console is shutting down")
        self._latest.pop("sample", None)
        self._to("down", error="stopped")

    def reconnect(self) -> None:
        """Reconnect now: skip the wait, forget the backoff, and drop a connection that is up."""
        self._backoff = self._t.backoff_min_s
        self._wake.set()
        proc = self._proc
        if proc is None:
            return
        if self._state == "up":
            self._drop_reason = "reconnecting at your request"
            if self._writer is not None:
                with contextlib.suppress(Exception):
                    self._writer.close()              # a node exits on EOF; the timer below catches one that does not
            asyncio.get_running_loop().call_later(self._t.exit_grace_s, _kill_proc, proc)
        elif self._state == "connecting":
            self._kill("reconnecting at your request")   # no hello yet: nothing to be graceful about, start over

    # ── the connect loop ─────────────────────────────────────────────────────────────────────────────────
    async def _run(self) -> None:
        t = self._t
        while not self._stopping:
            self._wake.clear()
            self._to("connecting")
            try:
                end = await self._session()
            except asyncio.CancelledError:
                raise
            except Exception as e:   # noqa: BLE001 - nothing may end this loop but stop()
                log.exception("link %s: internal error", self.host.id)
                end = _End(f"internal error in the link ({type(e).__name__})")
            if self._stopping:
                return
            if self._wake.is_set():
                continue                                   # reconnect() asked for a retry now: not a failure to back off from
            if end.wait is None:
                if end.up_s >= t.stable_s:
                    self._backoff = t.backoff_min_s
                wait = self._backoff
                self._backoff = min(t.backoff_max_s, self._backoff * 2)
            else:
                wait = end.wait
            self.last_wait_s = wait
            self._to("incompatible" if end.incompatible else "down", error=end.error, wait=wait)
            log.warning("link %s: %s: %s (retry in %.0f s)", self.host.id, self._state, _text(end.error), wait)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), wait)

    async def _session(self) -> _End:
        t = self._t
        try:
            argv = [str(a) for a in self._argv()]
        except HostsError as e:
            return _End(str(e), wait=t.fix_retry_s)        # e.g. no hub key yet: only a person can fix it
        if not argv:
            return _End("there is no command to run for this host")
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env=self._env, start_new_session=True, limit=READ_LIMIT)
        except OSError as e:
            return _End(f"could not start {os.path.basename(argv[0])}: {e.strerror or type(e).__name__}")
        log.debug("link %s: started %s (pid %s)", self.host.id, os.path.basename(argv[0]), proc.pid)
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        self._proc, self._writer, self._drop_reason = proc, proc.stdin, None
        if self._stopping:                                 # stop() ran while the child was being spawned
            _kill_proc(proc)
        tail = bytearray()
        drain = asyncio.create_task(_drain_stderr(proc.stderr, tail))
        try:
            talk = await self._converse(proc.stdout)
            reason = talk.err or self._drop_reason or "the node closed the connection"
            # From here nothing may be sent, and nobody should wait for the process to be reaped below.
            self._writer = None
            self._fail_all(reason)
            self._latest.pop("sample", None)
            if talk.was_up:
                self._to("down", error=reason)
            await self._reap(proc, drain)
        finally:
            _kill_proc(proc)                               # never leave a child behind, even when cancelled
            drain.cancel()
            self._proc, self._writer = None, None
        err = talk.err or self._drop_reason
        if err is not None:
            return _End(err, wait=talk.wait, incompatible=talk.incompatible, up_s=talk.up_s)
        msg, wait = explain_exit(self.host, proc.returncode, tail.decode("utf-8", "replace"), self._remote_cmd,
                                 t.fix_retry_s, self._mask)
        return _End(msg, wait=wait, up_s=talk.up_s)

    async def _reap(self, proc: asyncio.subprocess.Process, drain: asyncio.Task[None]) -> None:
        """Let a node that closed its pipes exit by itself, so its real exit status (ssh's 255, a shell's 127) and its
        last words on stderr can explain the drop; kill it if it does not."""
        if proc.stdin is not None:
            with contextlib.suppress(Exception):
                proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), self._t.exit_grace_s)
        except TimeoutError:
            _kill_proc(proc)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), 5.0)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(drain, 2.0)

    async def _converse(self, reader: asyncio.StreamReader) -> _Talk:
        t = self._t
        try:
            hello, skipped = await asyncio.wait_for(self._handshake(reader), t.hello_timeout_s)
        except TimeoutError:
            self._kill()
            return _Talk(err=f"no answer from the node within {t.hello_timeout_s:g} s")
        except _NodeFatal as e:
            return _Talk(err=str(e))
        except (proto.ProtoError, OSError) as e:
            if isinstance(e, (OSError, _Truncated)) or reader.at_eof():
                # The far side simply closed (ssh failing, a shell exiting). Do not kill it: its exit status and its
                # last words on stderr explain why far better than "closed before the protocol".
                return _Talk(err=None)
            self._kill()
            return _Talk(err=f"protocol error from the node: {_text(e)}")
        if hello["proto"] != proto.PROTO:
            self._latest["hello"] = hello                  # so the page can show which version the far side runs
            return _Talk(err=self._mismatch(hello), wait=t.fix_retry_s, incompatible=True)
        self._connected(hello, skipped)
        up_at = self._clock()
        watch = asyncio.create_task(self._watchdog(), name=f"watch-{self.host.id}")
        pinger = asyncio.create_task(self._pinger(), name=f"ping-{self.host.id}")
        try:
            err = await self._read_loop(reader)
        finally:
            for task in (pinger, watch):
                task.cancel()
            await asyncio.gather(pinger, watch, return_exceptions=True)
        return _Talk(err=err, up_s=self._clock() - up_at, was_up=True)

    async def _handshake(self, reader: asyncio.StreamReader) -> tuple[dict[str, Any], int]:
        skipped = await proto.find_magic_async(reader)
        got = await _read_frame(reader)
        if got is None:
            raise proto.ProtoError("the node closed before sending its hello")
        frame, _ = got
        msg = frame.msg if frame.kind == proto.KIND_JSON else None
        if msg is None:
            raise proto.ProtoError("the node's first frame was not a message")
        if msg.get("type") == "fatal":
            raise _NodeFatal(f"the node gave up: {_text(proto.clean(msg.get('error'), proto.S(1000)) or 'no reason given')}")
        if msg.get("type") != "hello":
            raise proto.ProtoError(f"the node's first message was {_text(msg.get('type'), 40)!r}, not hello")
        hello = proto.sanitize_push(msg)
        assert hello is not None
        return hello, skipped

    def _mismatch(self, hello: dict[str, Any]) -> str:
        theirs = hello["proto"]
        side = "the Engine Console hub" if isinstance(theirs, int) and theirs > proto.PROTO else f"ecnode on {self.host.label}"
        return (f"{self.host.label} runs ecnode {_text(hello.get('version') or '?', 32)} (protocol {theirs}); this "
                f"Engine Console speaks protocol {proto.PROTO}. Update {side}.")

    def _connected(self, hello: dict[str, Any], skipped: int) -> None:
        notes: list[str] = []
        if skipped:
            notes.append(f"ignored {skipped} byte(s) the far side printed before the node started "
                         "(a shell startup file writing to stdout?)")
        if self.host.ssh is not None and hello.get("forced_cmd") is not True:
            notes.append("reached with an UNRESTRICTED ssh key: restrict it to the node command in authorized_keys "
                         "(`engine-console hosts init-key` prints the line)")
        now = self._clock()
        self._latest["hello"] = hello
        self._notes = notes
        self._up_mono, self._last_rx, self._last_sample_mono = now, now, None
        self._connected_wall = self._last_seen = self._wallclock()
        self._rtt_ms = self._skew_s = None
        self._pings.clear()
        self._wake.clear()                                 # a reconnect request has been served
        self._to("up")
        self._publish("hello", hello)
        log.info("link %s: up (ecnode %s, %d GPU(s))", self.host.id, _text(hello.get("version") or "?", 32),
                 len(hello.get("gpus") or []))

    def _refresh_hello(self, result: dict[str, Any]) -> None:
        """`node.info` re-reads the node's policy: show the fresh facts without waiting for a reconnect."""
        if result.get("proto") != proto.PROTO or "hello" not in self._latest:
            return
        hello = {**result, "type": "hello", "gpus": [g for g in result.get("gpus") or [] if g.get("uuid")]}
        self._latest["hello"] = hello
        self._publish("hello", hello)
        self._changed()

    # ── state ────────────────────────────────────────────────────────────────────────────────────────────
    def _to(self, state: LinkState, *, error: str | None = None, wait: float | None = None) -> None:
        before = (self._state, self._error, tuple(self._notes))
        self._state = state
        if state == "up":
            self._error = None
        elif error is not None:
            self._error = error
        self._retry_at = None if wait is None else self._clock() + wait
        if state != "up":
            self._up_mono = None
        if (self._state, self._error, tuple(self._notes)) != before:
            self._changed()

    def _changed(self) -> None:
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception:   # noqa: BLE001 - a listener bug must not break the link
                log.exception("link %s: change listener failed", self.host.id)

    def _publish(self, kind: str, msg: dict[str, Any]) -> None:
        self._bus.publish(f"node.{kind}", {"host_id": self.host.id, "msg": msg})

    def _kill(self, reason: str | None = None) -> None:
        if reason is not None and self._drop_reason is None:
            self._drop_reason = reason
        _kill_proc(self._proc)

    def _fail_all(self, reason: str) -> None:
        pending, self._pending = self._pending, {}
        for req in pending.values():
            req.finish(None, reason)
        self._pings.clear()

    # ── liveness ─────────────────────────────────────────────────────────────────────────────────────────
    async def _pinger(self) -> None:
        while True:
            self._ping()                                   # the first one at once, so the page has an rtt right away
            await asyncio.sleep(self._t.ping_s)

    def _ping(self) -> None:
        if len(self._pings) > 16:
            self._pings.clear()                            # a node that never answers cannot grow this
        pid = self._next_ping
        self._next_ping = pid % proto.MAX_ID + 1
        self._pings[pid] = self._clock()
        self.send_nowait(proto.ping(pid))                  # tiny and never awaited: not stuck behind an upload

    async def _watchdog(self) -> None:
        tick = max(0.02, min(1.0, self._t.silent_s / 5))
        while True:
            await asyncio.sleep(tick)
            now = self._clock()
            if now - self._last_rx > self._t.silent_s:
                self._kill(f"no message from the node for {self._t.silent_s:g} s")
                return
            for rid, req in list(self._pending.items()):   # cancelled long ago and never answered: forget it
                if req.cancel_sent and now - req.cancel_at > ORPHAN_MAX_S:
                    self._pending.pop(rid, None)
                    req.finish(None, "the node never answered a cancelled request")

    # ── reading ──────────────────────────────────────────────────────────────────────────────────────────
    async def _read_loop(self, reader: asyncio.StreamReader) -> str | None:
        """Route frames until the node closes (None) or misbehaves (the reason, after killing it)."""
        t = self._t
        win_start, win_bytes, win_frames, since_yield = self._clock(), 0, 0, 0
        while True:
            try:
                got = await _read_frame(reader)
            except (_Truncated, OSError):
                return None                                # the node died mid-frame: its exit status says why
            except proto.ProtoError as e:
                self._kill()
                return f"protocol error from the node: {_text(e)}"
            if got is None:
                return None
            frame, size = got
            now = self._clock()
            self._last_rx = now
            self._last_seen = self._wallclock()
            # The flood limit is for JSON messages, which nothing else bounds. DATA is bounded by the credit this
            # hub grants (a model copy legitimately moves far more than 20 MB in 10 s).
            if frame.kind == proto.KIND_JSON or not frame.data:
                if now - win_start > t.flood_window_s:
                    win_start, win_bytes, win_frames = now, 0, 0
                win_bytes, win_frames = win_bytes + size, win_frames + 1
                if win_bytes > t.flood_bytes or win_frames > t.flood_frames:
                    self._kill()
                    return (f"the node sent over {t.flood_bytes / 2**20:g} MB or {t.flood_frames} messages in "
                            f"{t.flood_window_s:g} s; dropped it")
            err = self._on_json(frame.msg or {}, now) if frame.kind == proto.KIND_JSON else self._on_data(frame.stream, frame.data)
            if err is not None:
                self._kill()
                return err
            since_yield += 1
            if since_yield >= 64:                          # a node writing flat out must not starve the loop
                since_yield = 0
                await asyncio.sleep(0)

    def _on_json(self, msg: dict[str, Any], now: float) -> str | None:
        """Handle one message; returns a reason to drop the link, or None."""
        kind = msg.get("type")
        if kind == "res":
            req = self._request(msg.get("id"))
            if req is not None:                            # unknown ids (late answers to forgotten requests) are ignored
                del self._pending[req.rid]
                req.finish(proto.sanitize_res(msg, req.op))
            return None
        if kind == "event":
            req = self._request(msg.get("id"))
            if req is None or req.queue is None or req.cancel_sent:
                return None
            ev = proto.sanitize_event(msg, req.op)["event"]
            if any(v is not None for v in ev.values()) and req.queued_events < MAX_STREAM_EVENTS:
                req.queued_events += 1
                req.queue.put_nowait(StreamItem("event", event=ev))
            return None
        try:
            out = proto.sanitize_push(msg)
        except proto.ProtoError as e:
            return f"protocol error from the node: {_text(e)}"
        if out is None:
            return None                                    # a message type this hub does not know: ignore it
        t = out["type"]
        if t == "sample":
            announced = {g["uuid"] for g in (self._latest.get("hello") or {}).get("gpus") or []}
            out["gpus"] = {u: g for u, g in (out.get("gpus") or {}).items() if u in announced}   # no new GPUs from a sample
            self._latest["sample"] = out
            self._last_sample_mono = now
            if isinstance(out.get("t"), (int, float)):
                self._skew_s = round(out["t"] - self._wallclock(), 1)
            self._publish("sample", out)
        elif t in ("disks", "inventory"):
            self._latest[t] = out
            self._publish(t, out)
        elif t == "hello":                                 # a re-sent hello: same rules as the first
            if out["proto"] != proto.PROTO:
                return self._mismatch(out)
            self._latest["hello"] = out
            self._publish("hello", out)
            self._changed()
        elif t in ("metrics", "models_changed"):
            self._publish(t, out)
        elif t == "pong":
            t0 = self._pings.pop(out["id"], None) if isinstance(out.get("id"), int) else None
            if t0 is not None:
                self._rtt_ms = round((now - t0) * 1000)
        elif t == "credit":
            nbytes = out.get("bytes")
            req = self._request(out.get("id"))
            if req is not None and isinstance(nbytes, int) and nbytes > 0:
                req.send_credit = min(req.send_credit + nbytes, MAX_CREDIT)
                req.credit_ev.set()
        elif t == "log":
            self._node_log(out)
        elif t == "fatal":
            return f"the node gave up: {_text(out.get('error') or 'no reason given', 300)}"
        return None

    def _request(self, rid: object) -> _Request | None:
        """The request in flight with this id, or None (a node can name any id it likes)."""
        return self._pending.get(rid) if isinstance(rid, int) and proto.valid_id(rid) else None

    def _on_data(self, stream: int, data: bytes) -> str | None:
        req = self._pending.get(stream)
        if req is None or req.queue is None:
            return None                                    # unknown stream, or not one the hub opened for DATA: drop
        if len(data) > req.recv_window:
            return f"the node sent more data on stream {stream} than the credit this hub granted it"
        req.recv_window -= len(data)
        if data and not req.cancel_sent:
            req.queue.put_nowait(StreamItem("data", data=data))
        return None

    def _node_log(self, out: dict[str, Any]) -> None:
        """A node's own log line, rate limited: it is untrusted text and must not be able to fill the hub's log."""
        now = self._clock()
        if now - self._log_window > 10.0:
            self._log_window, self._log_count = now, 0
        self._log_count += 1
        if self._log_count <= NODE_LOG_BURST:
            level = logging.WARNING if out.get("level") in ("warn", "warning", "error") else logging.DEBUG
            log.log(level, "node %s: %s", self.host.id, _text(out.get("msg") or "", 300))

    # ── writing ──────────────────────────────────────────────────────────────────────────────────────────
    def _alloc_id(self) -> int:
        while True:
            rid = self._next_id
            self._next_id = rid % proto.MAX_ID + 1
            if rid not in self._pending:                   # never reuse an id that is still in flight
                return rid

    def _begin(self, op: str, args: dict[str, Any] | None, *, streaming: bool) -> _Request:
        hid = self.host.id
        if op not in proto.OPS:
            raise NodeError(hid, proto.UNKNOWN_OP, f"{op!r} is not an operation of the node protocol")
        w = self._writer
        if self._state != "up" or w is None or w.is_closing():
            why = self._error or self._state
            raise NodeUnavailable(hid, f"{self.host.label} is not connected ({_text(why, 200)})")
        if len(self._pending) >= proto.MAX_INFLIGHT:
            raise NodeError(hid, proto.BUSY, f"{self.host.label} already has {proto.MAX_INFLIGHT} requests in flight")
        rid = self._alloc_id()
        try:
            frame = proto.encode_json(proto.req(rid, op, args))
        except (proto.ProtoError, TypeError, ValueError) as e:   # NaN, a value JSON cannot carry, a message too big
            raise NodeError(hid, proto.BAD_ARGS, f"cannot send {op}: {_text(e, 200)}") from None
        req = _Request(rid, op, streaming)
        self._pending[rid] = req
        w.write(frame)
        return req

    def send_nowait(self, msg: dict[str, Any]) -> None:
        """Queue one small control frame (ping, cancel, end, credit). Never waits: control frames must not sit
        behind a bulk transfer's back-pressure. A dead link is ignored: its requests are failed elsewhere."""
        w = self._writer
        if w is None or w.is_closing():
            return
        try:
            w.write(proto.encode_json(msg))
        except (proto.ProtoError, TypeError, ValueError):
            log.error("link %s: could not encode a %s frame", self.host.id, msg.get("type"))

    async def send_data(self, frame: bytes) -> None:
        w = self._writer
        if w is None or w.is_closing():
            raise NodeUnavailable(self.host.id, "the connection dropped")
        try:
            w.write(frame)
            await w.drain()
        except (ConnectionError, OSError):
            raise NodeUnavailable(self.host.id, "the connection dropped") from None

    def cancel_request(self, req: _Request) -> None:
        """Ask the node to stop a request. It still answers with exactly one res, so the request stays registered
        (its id cannot be reused, and it still counts toward the node's limit) until that answer arrives."""
        if req.finished or req.cancel_sent:
            return
        req.cancel_sent, req.cancel_at = True, self._clock()
        self.send_nowait(proto.cancel(req.rid))


# ── disabled hosts ───────────────────────────────────────────────────────────────────────────────────────
class DisabledLink:
    """The link of a host that is switched off (`enabled: false`, or the whole fleet is): it never connects."""

    def __init__(self, host: HostConfig, reason: str) -> None:
        self.host = host
        self._reason = reason

    def link(self) -> LinkStatus:
        return LinkStatus(state="disabled", error=self._reason)

    def latest(self, kind: NodeKind) -> dict[str, Any] | None:
        return None

    async def call(self, op: str, args: dict[str, Any] | None = None, *, timeout: float = 30.0) -> dict[str, Any]:
        raise NodeUnavailable(self.host.id, f"{self.host.label} is disabled: {self._reason}")

    def open(self, op: str, args: dict[str, Any] | None = None) -> NodeStream:
        return FailedStream(NodeUnavailable(self.host.id, f"{self.host.label} is disabled: {self._reason}"))

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def reconnect(self) -> None:
        return None


class LinkLike(NodeClient, Protocol):
    """What the FleetService needs from a link: a NodeClient that can be started, stopped and kicked."""
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def reconnect(self) -> None: ...


class FleetService:
    """Every configured machine's link, and the hub's one picture of the fleet (implements `domain.fleet.FleetPort`).

    Raises `HostsError` from the constructor when hosts.json is unacceptable: the console must refuse to start
    rather than guess which machines it may run ssh against. With `cfg.fleet_enabled` false no link ever connects
    (every host is `disabled`), which keeps unit tests of unrelated services away from subprocesses."""

    WATCH_S = 1.0     # how often the watcher looks for link changes nobody announced (a class attribute so tests can shorten it)

    def __init__(self, cfg: Settings, bus: EventBus, *, hosts: list[HostConfig] | None = None,
                 link_factory: Callable[[HostConfig], LinkLike] | None = None) -> None:
        self._cfg, self._bus = cfg, bus
        if hosts is None:
            hosts = load_hosts(cfg.hosts_file) if cfg.fleet_enabled else default_hosts()
        if len({h.id for h in hosts}) != len(hosts):
            raise ValueError("two hosts share an id")
        self._hosts = list(hosts)
        make = link_factory or self._make_link
        self._links: dict[str, LinkLike] = {h.id: make(h) for h in self._hosts}
        self._resolver: fleet_status.InstanceResolver | None = None
        self._counts: fleet_status.CountsProvider | None = None
        self._images: dict[str, tuple[float, list[ImageInfo]]] = {}
        self._image_locks: dict[str, asyncio.Lock] = {}
        self._seen: dict[str, tuple[str, str | None]] = {}
        self._watcher: asyncio.Task[None] | None = None

    # ── FleetPort ────────────────────────────────────────────────────────────────────────────────────────
    def hosts(self) -> list[HostConfig]:
        return list(self._hosts)

    def node(self, host_id: str) -> NodeClient:
        return self._link(host_id)

    # ── wiring from other services ───────────────────────────────────────────────────────────────────────
    def set_instance_resolver(self, fn: fleet_status.InstanceResolver | None) -> None:
        """fn(host_id, container_name) -> console instance id, to name the instance behind each GPU process."""
        self._resolver = fn

    def set_counts_provider(self, fn: fleet_status.CountsProvider | None) -> None:
        """fn(host_id) -> HostCounts (engines running/total, models), shown in the host summary."""
        self._counts = fn

    # ── lifecycle ────────────────────────────────────────────────────────────────────────────────────────
    async def start(self) -> None:
        for link in self._links.values():
            await link.start()
        if self._watcher is None:
            self._watcher = asyncio.create_task(self._watch(), name="fleet-watch")

    async def stop(self) -> None:
        if self._watcher is not None:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._watcher
            self._watcher = None
        await asyncio.gather(*(link.stop() for link in self._links.values()), return_exceptions=True)

    async def _watch(self) -> None:
        """Links announce their own changes; this catches the ones nobody announces (a link turning `stale` is just
        time passing) and links from a custom factory that do not announce at all."""
        while True:
            await asyncio.sleep(self.WATCH_S)
            for hid, link in self._links.items():
                st = link.link()
                if self._seen.get(hid) != (st.state, st.error):
                    self._publish(hid)

    def _publish(self, host_id: str) -> None:
        try:
            s = self.summary(host_id)
        except Exception:   # noqa: BLE001 - a summary bug must not break the link that called us
            log.exception("could not build the summary of host %s", host_id)
            return
        self._seen[host_id] = (s.link.state, s.link.error)
        self._bus.publish("fleet", s.model_dump(mode="json"))

    def _make_link(self, host: HostConfig) -> LinkLike:
        cfg = self._cfg
        if not cfg.fleet_enabled:
            return DisabledLink(host, "the fleet is switched off (FLEET_ENABLED=false)")
        if not host.enabled:
            return DisabledLink(host, "disabled in hosts.json")
        notify = functools.partial(self._publish, host.id)
        if host.id == LOCAL:
            return NodeLink(host, lambda: local_argv(cfg), local_env(), self._bus, on_change=notify)
        return NodeLink(host, lambda: self._remote_argv(host), ssh_env(), self._bus, on_change=notify,
                        remote_cmd=cfg.remote_node_command, mask=[os.fspath(cfg.hub_key_file)])

    def _remote_argv(self, host: HostConfig) -> list[str]:
        # Built on every attempt (and raising NoHubKey while there is no key), so running `hosts init-key` fixes the
        # link by itself at its next retry or on a manual reconnect.
        return ssh_argv(host, self._cfg.hub_key_file, self._cfg.remote_node_command)

    # ── what the API shows ───────────────────────────────────────────────────────────────────────────────
    def _link(self, host_id: str) -> LinkLike:
        link = self._links.get(host_id)
        if link is None:
            raise NotFound(f"no host {host_id[:40]!r} is configured", code="unknown_host")
        return link

    def _data(self, host_id: str) -> fleet_status.HostData:
        link = self._link(host_id)
        return fleet_status.HostData(host=link.host, link=link.link(), hello=link.latest("hello"),
                                     sample=link.latest("sample"), disks=link.latest("disks"),
                                     inventory=link.latest("inventory"))

    def summary(self, host_id: str) -> HostSummary:
        return fleet_status.build_summary(self._data(host_id), resolver=self._resolver, counts=self._counts)

    def summaries(self) -> list[HostSummary]:
        return [self.summary(h.id) for h in self._hosts]

    async def detail(self, host_id: str) -> HostDetail:
        self._link(host_id)
        images = await self._images_of(host_id)
        fleet = {h.id: self._data(h.id) for h in self._hosts}
        return fleet_status.build_detail(host_id, fleet, images=images, resolver=self._resolver, counts=self._counts)

    async def _images_of(self, host_id: str) -> list[ImageInfo]:
        """Local images are not part of a node's pushed inventory: ask (at most once a minute per host)."""
        link = self._link(host_id)
        cached = self._images.get(host_id)
        if cached is not None and time.monotonic() < cached[0]:
            return cached[1]
        facts = fleet_status.docker_facts(link.latest("hello"))
        if link.link().state not in ("up", "stale") or facts is None or not facts.available:
            return cached[1] if cached is not None else []
        lock = self._image_locks.setdefault(host_id, asyncio.Lock())
        async with lock:
            cached = self._images.get(host_id)
            if cached is not None and time.monotonic() < cached[0]:
                return cached[1]
            try:
                images = fleet_status.images_from_result(await link.call("docker.images", timeout=5.0))
                self._images[host_id] = (time.monotonic() + IMAGES_TTL_S, images)
            except NodeError:
                old = cached[1] if cached is not None else []
                self._images[host_id] = (time.monotonic() + 10.0, old)   # do not hammer a node that cannot answer
                images = old
            return images

    def reconnect(self, host_id: str) -> HostSummary:
        link = self._link(host_id)
        if link.link().state == "disabled":
            raise Conflict(f"{link.host.label} is disabled", code="host_disabled")
        link.reconnect()
        return self.summary(host_id)

