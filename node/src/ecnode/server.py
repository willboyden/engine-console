"""The stdio protocol server (docs/FLEET-PROTOCOL.md §2): MAGIC + hello, then frames both ways.

Threads: one reader (frames from the hub), one writer (every frame to the hub goes through its queue, so frames
never interleave), a watchdog (a write blocked for WRITE_STALL_S ends the process; HUB_SILENCE_S without any hub
frame ends the session), the push loops (sample, disks, inventory, metrics), the model scanner, and one thread
per running request (at most proto.MAX_INFLIGHT; more get `busy`).

Flow control: every stream starts with proto.WINDOW bytes of credit each way. Downloads (node -> hub) never put
more DATA on a stream than the hub granted; uploads (hub -> node) are buffered only up to the credit this node
granted, and credit is handed back as the handler consumes, so a slow disk slows only its own upload. The hub
drops a node that exceeds credit or floods it with JSON, so bulk JSON (events, inventory, metrics, disks) is
rate-limited here to well under the hub's limit; control frames (res, pong, credit, sample) are never held back.
"""
from __future__ import annotations

import collections
import contextlib
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import IO, Any

from . import proto
from .agent import Agent
from .errors import Cancelled, OpError
from .models import Scanner
from .ops import HANDLERS, Call
from .util import scrub

DATA_QUEUE_MAX = 4 * 1024 * 1024        # DATA bytes waiting in the writer before producers wait
JSON_QUEUE_MAX = 64 * 1024 * 1024
WRITE_BATCH = 1024 * 1024
GRANT_STEP = 1024 * 1024                # upload credit is handed back in steps of at least this
UPLOAD_IDLE_S = 300.0
CREDIT_WAIT_S = 300.0
BULK_WINDOW_S = 10.0                    # bulk JSON budget per window (the hub allows 20 MB / 50 000 frames per 10 s)
BULK_BYTES = 8 * 1024 * 1024
BULK_FRAMES = 20_000
LINE_OPS = frozenset({"docker.logs_follow", "docker.pull", "scripts.run"})   # events that can come by the thousand
FIRST_INVENTORY_WAIT_S = 5.0


@dataclass
class Timing:
    sample_s: float = proto.SAMPLE_S
    disks_s: float = proto.DISKS_S
    inventory_s: float = proto.INVENTORY_S
    inventory_poll_s: float = 10.0
    hub_silence_s: float = proto.HUB_SILENCE_S
    write_stall_s: float = 10.0
    watchdog_s: float = 1.0


def _say(msg: str) -> None:
    with contextlib.suppress(OSError, ValueError):
        print(f"ecnode: {msg}", file=sys.stderr, flush=True)


# ── the writer ───────────────────────────────────────────────────────────────────────────────────────────
class Writer:
    """Single serialized writer. put() never interleaves frames; DATA and bulk JSON wait for room."""

    def __init__(self, out: IO[bytes], on_closed: Callable[[], None], clock: Callable[[], float] = time.monotonic,
                 bulk_bytes: int = BULK_BYTES, bulk_frames: int = BULK_FRAMES) -> None:
        self._out = out
        self._on_closed = on_closed
        self._clock = clock
        self._cv = threading.Condition()
        self._q: collections.deque[tuple[bytes, bool]] = collections.deque()
        self._data = 0
        self._json = 0
        self._closing = False
        self.closed = False
        self.writing_since: float | None = None
        self._bulk_bytes, self._bulk_frames = bulk_bytes, bulk_frames
        self._win_start = clock()
        self._win_bytes = 0
        self._win_frames = 0

    def put(self, frame: bytes, *, data: bool = False, bulk: bool = False,
            abort: Callable[[], bool] | None = None) -> bool:
        with self._cv:
            while not self.closed and not self._closing and self._must_wait(len(frame), data, bulk):
                if abort is not None and abort():
                    return False
                self._cv.wait(0.25)
            if self.closed or self._closing:
                return False
            if bulk:
                self._win_bytes += len(frame)
                self._win_frames += 1
            self._q.append((frame, data))
            if data:
                self._data += len(frame)
            else:
                self._json += len(frame)
            self._cv.notify_all()
            return True

    def _must_wait(self, n: int, data: bool, bulk: bool) -> bool:
        if data:
            return self._data > 0 and self._data + n > DATA_QUEUE_MAX
        if self._json > 0 and self._json + n > JSON_QUEUE_MAX:
            return True
        return bulk and not self._bulk_room(n)

    def _bulk_room(self, n: int) -> bool:
        now = self._clock()
        if now - self._win_start >= BULK_WINDOW_S:
            self._win_start, self._win_bytes, self._win_frames = now, 0, 0
        return self._win_frames < self._bulk_frames and (self._win_bytes + n <= self._bulk_bytes or not self._win_bytes)

    def run(self) -> None:
        while True:
            with self._cv:
                while not self._q and not self.closed and not self._closing:
                    self._cv.wait(0.5)
                if not self._q:
                    self.closed = True
                    self._cv.notify_all()
                    return
                batch: list[bytes] = []
                size = 0
                while self._q and size < WRITE_BATCH:
                    frame, is_data = self._q.popleft()
                    batch.append(frame)
                    size += len(frame)
                    if is_data:
                        self._data -= len(frame)
                    else:
                        self._json -= len(frame)
                self._cv.notify_all()
            try:
                self.writing_since = self._clock()
                self._out.write(b"".join(batch))
                self._out.flush()
            except (OSError, ValueError):
                with self._cv:
                    self.closed = True
                    self._q.clear()
                    self._cv.notify_all()
                self._on_closed()
                return
            finally:
                self.writing_since = None

    def close(self, timeout: float = 2.0) -> None:
        """Stop accepting frames, write what is queued (for at most `timeout`), stop."""
        deadline = self._clock() + timeout
        with self._cv:
            self._closing = True
            self._cv.notify_all()
            while self._q and not self.closed and self._clock() < deadline:
                self._cv.wait(0.1)
            self.closed = True
            self._cv.notify_all()


# ── one request ──────────────────────────────────────────────────────────────────────────────────────────
class Request:
    """A request in flight: its cancel state, both credit windows and its upload buffer (errors.Stream)."""

    def __init__(self, server: Server, rid: int, op: str, upload: bool) -> None:
        self.server, self.rid, self.op, self.upload = server, rid, op, upload
        self.cv = threading.Condition()
        self._cancelled = False
        self._hooks: list[Callable[[], None]] = []
        self.send_credit = proto.WINDOW          # DATA this node may still send (the hub's credit)
        self.chunks: collections.deque[bytes] = collections.deque()
        self.received = 0                        # upload bytes received
        self.granted = proto.WINDOW              # upload credit granted to the hub, in total
        self.consumed = 0                        # upload bytes the handler took
        self.ended = False
        self.violation: str | None = None

    # Stream protocol
    @property
    def cancelled(self) -> bool:
        return self._cancelled or self.server.done.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise Cancelled

    def event(self, ev: dict[str, Any]) -> None:
        spec = proto.EVENTS.get(self.op, {})
        self.server.send(proto.event(self.rid, proto.clean(ev, spec) or {}), bulk=self.op in LINE_OPS,
                         abort=lambda: self.cancelled)

    def send_data(self, data: bytes) -> None:
        view = memoryview(data)
        while len(view):
            with self.cv:
                deadline = time.monotonic() + CREDIT_WAIT_S
                while self.send_credit <= 0 and not self.cancelled:
                    if time.monotonic() > deadline:
                        raise OpError(proto.TIMEOUT, "the hub stopped granting credit for this stream")
                    self.cv.wait(0.25)
                self.check()
                n = min(len(view), self.send_credit, proto.MAX_DATA)
                self.send_credit -= n
            if not self.server.writer.put(proto.encode_data(self.rid, bytes(view[:n])), data=True,
                                          abort=lambda: self.cancelled):
                raise Cancelled
            view = view[n:]

    def read_upload(self) -> bytes:
        with self.cv:
            deadline = time.monotonic() + UPLOAD_IDLE_S
            while not self.chunks and not self.ended and not self.cancelled and self.violation is None:
                if time.monotonic() > deadline:
                    raise OpError(proto.TIMEOUT, f"no upload data from the hub for {UPLOAD_IDLE_S:g} s")
                self.cv.wait(0.25)
            self.check()
            if self.violation is not None:
                raise OpError(proto.BAD_ARGS, self.violation)
            if not self.chunks:
                return b""
            chunk = self.chunks.popleft()
            self.consumed += len(chunk)
            grant = self.consumed + proto.WINDOW - self.granted
            if grant >= GRANT_STEP and not self.ended:
                self.granted += grant
            else:
                grant = 0
        if grant:
            self.server.send(proto.credit(self.rid, grant))
        return chunk

    def drain_upload(self) -> None:
        with contextlib.suppress(Cancelled, OpError):
            while self.read_upload():
                pass

    def on_cancel(self, fn: Callable[[], None]) -> None:
        with self.cv:
            if not self.cancelled:
                self._hooks.append(fn)
                return
        fn()

    # the reader's side
    def cancel(self) -> threading.Thread | None:
        with self.cv:
            if self._cancelled:
                return None
            self._cancelled = True
            hooks, self._hooks = self._hooks, []
            self.cv.notify_all()
        if not hooks:
            return None
        # hooks may block (killing a process waits for it): never on the reader thread
        th = threading.Thread(target=_run_hooks, args=(hooks,), name="ecnode-cancel", daemon=True)
        th.start()
        return th

    def add_credit(self, n: int) -> None:
        with self.cv:
            self.send_credit = min(self.send_credit + n, 1 << 40)
            self.cv.notify_all()

    def add_data(self, data: bytes) -> None:
        with self.cv:
            if not self.upload or self.ended:
                return
            self.received += len(data)
            if self.received > self.granted:
                self.violation = "the hub sent more upload data than the credit this node granted"
                self.chunks.clear()
            elif self.violation is None and data:
                self.chunks.append(data)
            self.cv.notify_all()

    def end(self) -> None:
        with self.cv:
            self.ended = True
            self.cv.notify_all()


def _run_hooks(hooks: list[Callable[[], None]]) -> None:
    for fn in hooks:
        try:
            fn()
        except Exception as e:     # a hook failing must not take the node down
            _say(f"cancel hook failed: {type(e).__name__}")


# ── the server ───────────────────────────────────────────────────────────────────────────────────────────
class Server:
    def __init__(self, agent: Agent, inp: IO[bytes], out: IO[bytes], *, timing: Timing | None = None,
                 on_stall: Callable[[], None] | None = None, clock: Callable[[], float] = time.monotonic,
                 scanner: bool = True) -> None:
        self.agent = agent
        self.inp = inp
        self.t = timing or Timing()
        self.clock = clock
        self.done = threading.Event()
        self.exit_code = 0
        self.writer = Writer(out, self._closed, clock)
        self.on_stall = on_stall or (lambda: os._exit(3))
        self.last_rx = clock()
        self.requests: dict[int, Request] = {}
        self.lock = threading.Lock()
        self._seq = 0
        self._use_scanner = scanner
        self._last_inventory_key: Any = None
        self._last_inventory_push = -1e18

    # ── sending ──────────────────────────────────────────────────────────────────────────────────────────
    def send(self, msg: dict[str, Any], *, bulk: bool = False, abort: Callable[[], bool] | None = None) -> bool:
        try:
            frame = proto.encode_json(msg)
        except (proto.ProtoError, ValueError, TypeError) as e:
            _say(f"dropped an unsendable {msg.get('type')} message: {e}")
            if msg.get("type") == "res":
                frame = proto.encode_json(proto.res_err(int(msg["id"]), proto.INTERNAL,
                                                        "the result was too large or malformed to send"))
            else:
                return False
        return self.writer.put(frame, bulk=bulk, abort=abort)

    def push(self, msg: dict[str, Any], *, bulk: bool = False) -> bool:
        t = msg["type"]
        out = proto.clean(msg, proto.PUSH[t]) or {}
        out["type"] = t
        return self.send(out, bulk=bulk)

    def log(self, level: str, text: str) -> None:
        _say(text)
        self.push({"type": "log", "level": level, "msg": scrub(text)[:1000]})

    def _closed(self) -> None:
        self.done.set()

    # ── lifecycle ────────────────────────────────────────────────────────────────────────────────────────
    def run(self) -> int:
        self.writer.put(proto.MAGIC)
        threading.Thread(target=self.writer.run, name="ecnode-writer", daemon=True).start()
        try:
            self.push(self.agent.hello())
        except Exception as e:     # still speak the protocol: say why, then go
            self.push({"type": "fatal", "error": f"the node could not describe this machine: {type(e).__name__}"})
            self.writer.close()
            return 1
        if self._use_scanner:
            self.agent.scanner = Scanner(lambda: self.agent.policy, self._models_changed)
            self.agent.scanner.start()
        self.last_rx = self.clock()
        for name, fn in (("reader", self._reader), ("watchdog", self._watchdog), ("sampler", self._sampler),
                         ("disks", self._disks), ("inventory", self._inventory), ("metrics", self._metrics)):
            threading.Thread(target=fn, name=f"ecnode-{name}", daemon=True).start()
        self.done.wait()
        self.shutdown()
        return self.exit_code

    def shutdown(self) -> None:
        self.done.set()
        with self.lock:
            reqs = list(self.requests.values())
        hooks = [th for th in (r.cancel() for r in reqs) if th is not None]
        if self.agent.scanner is not None:
            self.agent.scanner.stop()
        deadline = time.monotonic() + 8.0      # child processes (logs -f, pulls, scripts) are stopped, not orphaned
        for th in hooks:
            th.join(max(0.0, deadline - time.monotonic()))
        self.writer.close()

    def _models_changed(self, generation: int) -> None:
        self.push({"type": "models_changed", "generation": generation, "t": round(time.time(), 3)})

    # ── reading ──────────────────────────────────────────────────────────────────────────────────────────
    def _reader(self) -> None:
        try:
            while not self.done.is_set():
                frame = proto.read_frame_sync(self.inp)
                if frame is None:
                    break
                self.last_rx = self.clock()
                if frame.kind == proto.KIND_DATA:
                    with self.lock:
                        req = self.requests.get(frame.stream)
                    if req is not None:
                        req.add_data(frame.data)
                    continue
                self._on_message(frame.msg or {})
        except proto.ProtoError as e:
            _say(f"protocol error from the hub, closing: {e}")
            self.exit_code = 2
        except (OSError, ValueError) as e:
            _say(f"reading from the hub failed, closing: {type(e).__name__}")
        self.done.set()

    def _on_message(self, msg: dict[str, Any]) -> None:
        t = msg.get("type")
        rid = msg.get("id")
        if t == "req":
            self._on_req(msg)
            return
        if not isinstance(rid, int) or not proto.valid_id(rid):
            return
        if t == "ping":
            self.send(proto.pong(rid))
            return
        with self.lock:
            req = self.requests.get(rid)
        if req is None:
            return                     # finished, refused or unknown: nothing to do
        if t == "cancel":
            req.cancel()
        elif t == "credit":
            n = msg.get("bytes")
            if isinstance(n, int) and not isinstance(n, bool) and 0 < n <= 1 << 40:
                req.add_credit(n)
        elif t == "end":
            req.end()

    def _on_req(self, msg: dict[str, Any]) -> None:
        rid = msg.get("id")
        if not isinstance(rid, int) or not proto.valid_id(rid):
            return
        op = msg.get("op")
        if not isinstance(op, str) or op not in proto.OPS:
            self.send(proto.res_err(rid, proto.UNKNOWN_OP, f"this node does not know the op {str(op)[:64]!r} "
                                                            f"(ecnode protocol {proto.PROTO})"))
            return
        policy = self.agent.policy
        if op not in proto.MONITOR_OPS and not policy.manage:
            why = f" ({policy.error})" if policy.error else ""
            self.send(proto.res_err(rid, proto.MONITOR_ONLY, f"this node's policy is in monitor mode{why}, so "
                                    f"{op}, which would change the machine, is refused; an administrator can set "
                                    f"\"mode\": \"manage\" in {policy.where()}"))
            return
        args = proto.clean_args(op, msg.get("args"))
        missing = proto.missing_args(op, args)
        if missing:
            self.send(proto.res_err(rid, proto.BAD_ARGS, f"{op} needs {', '.join(missing)}"))
            return
        upload = op == "fs.write" or (op == "http.request" and (args.get("body_len") or 0) > 0)
        req = Request(self, rid, op, upload)
        with self.lock:
            dup = rid in self.requests
            full = not dup and len(self.requests) >= proto.MAX_INFLIGHT
            if not dup and not full:
                self.requests[rid] = req
        if dup:
            self.log("warn", f"ignored request {rid}: that id is already in flight")
        elif full:
            self.send(proto.res_err(rid, proto.BUSY, f"this node is already running {proto.MAX_INFLIGHT} requests; "
                                                     "try again shortly"))
        else:
            threading.Thread(target=self._handle, args=(req, args), name=f"ecnode-req-{op}", daemon=True).start()

    def _handle(self, req: Request, args: dict[str, Any]) -> None:
        try:
            result = HANDLERS[req.op](self.agent, Call(req.op, args, req))
            msg = proto.res_ok(req.rid, proto.clean(result, proto.RESULTS[req.op]) or {})
        except Cancelled:
            msg = proto.res_err(req.rid, proto.CANCELLED, "cancelled")
        except OpError as e:
            msg = proto.res_err(req.rid, e.code, scrub(e.message))
        except Exception as e:     # a handler bug must cost one request, not the node, and never leak a trace
            where = traceback.extract_tb(e.__traceback__)[-1] if e.__traceback__ else None
            loc = f" at {os.path.basename(where.filename)}:{where.lineno}" if where else ""
            _say(f"internal error in {req.op}: {type(e).__name__}{loc}")
            msg = proto.res_err(req.rid, proto.INTERNAL, f"the node hit an unexpected error in {req.op} "
                                                         f"({type(e).__name__}); see the node's stderr")
        if req.cancelled and not msg["ok"] and msg["error"]["code"] != proto.CANCELLED:
            msg = proto.res_err(req.rid, proto.CANCELLED, "cancelled")
        self.send(msg)
        with self.lock:
            self.requests.pop(req.rid, None)

    # ── background loops ─────────────────────────────────────────────────────────────────────────────────
    def _every(self, period: Callable[[], float], fn: Callable[[], None], what: str, *, first: float = 0.0) -> None:
        nxt = self.clock() + first
        while not self.done.wait(max(0.0, nxt - self.clock())):
            started = self.clock()
            try:
                fn()
            except Exception as e:   # one bad round must not end the stream
                _say(f"{what}: {type(e).__name__}: {scrub(str(e))[:200]}")
            nxt = max(started + period(), self.clock())

    def _watchdog(self) -> None:
        while not self.done.wait(self.t.watchdog_s):
            now = self.clock()
            since = self.writer.writing_since
            if since is not None and now - since > self.t.write_stall_s:
                _say("a write to the hub has been blocked too long; exiting")
                self.on_stall()
                self.done.set()
                return
            if now - self.last_rx > self.t.hub_silence_s:
                _say(f"nothing from the hub for {self.t.hub_silence_s:g} s; exiting")
                self.done.set()
                return

    def _sampler(self) -> None:
        # The first samples should already name the container of each GPU process and carry the engines' cgroup
        # memory: wait (briefly) for the first inventory, which knows the containers.
        self.agent.first_inventory.wait(FIRST_INVENTORY_WAIT_S)

        def one() -> None:
            self._seq += 1
            self.push(self.agent.sample(self._seq))
        self._every(lambda: self.t.sample_s, one, "sampler")

    def _disks(self) -> None:
        def one() -> None:
            self.push(self.agent.disks(), bulk=True)
        self._every(lambda: self.t.disks_s, one, "disks")

    def _inventory(self) -> None:
        def one() -> None:
            inv = self.agent.refresh_inventory()
            key = _inventory_key(inv.msg)
            now = self.clock()
            due = key != self._last_inventory_key or now - self._last_inventory_push >= self.t.inventory_s
            if due and self.push({"type": "inventory", **inv.msg}, bulk=True):
                self._last_inventory_key, self._last_inventory_push = key, now
        self._every(lambda: self.t.inventory_poll_s, one, "inventory")

    def _metrics(self) -> None:
        def one() -> None:
            for msg in self.agent.scrape():
                self.push(msg, bulk=True)
        self._every(lambda: self.agent.metrics_conf.interval_s or 1.0, one, "metrics", first=1.0)


def _inventory_key(msg: dict[str, Any]) -> Any:
    """What counts as an inventory change: everything except timestamps and docker's "Up 5 minutes" text."""
    containers = [{k: v for k, v in c.items() if k != "status"} for c in msg.get("containers") or []]
    scripts = [{k: v for k, v in s.items() if k != "checked_at"} for s in msg.get("scripts") or []]
    return repr((containers, msg.get("endpoints"), scripts))
