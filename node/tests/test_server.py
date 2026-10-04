"""End to end: the real node server loop over OS pipes, driven by a hub that speaks real proto frames."""
from __future__ import annotations

import contextlib
import hashlib
import os
import queue
import threading
import time
from typing import IO, Any

import pytest

from ecnode import ops, proto
from ecnode.server import Server, Timing

from .conftest import GPU_A, manage_policy

FAST = Timing(sample_s=0.2, disks_s=0.5, inventory_s=1.0, inventory_poll_s=0.3, hub_silence_s=30.0,
              write_stall_s=10.0, watchdog_s=0.05)
STEP = 1024 * 1024


class Hub:
    """The hub's end of the pipes. Checks, frame by frame, that the node never exceeds the credit granted."""

    def __init__(self, to_node: IO[bytes], from_node: IO[bytes]) -> None:
        self.to = to_node
        self.src = from_node
        self.frames: queue.Queue[proto.Frame | None] = queue.Queue()
        self.pushed: list[dict[str, Any]] = []
        self.skipped: int | None = None
        self._rid = 100
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        try:
            self.skipped = proto.find_magic_sync(self.src)
            while (f := proto.read_frame_sync(self.src)) is not None:
                self.frames.put(f)
        except (proto.ProtoError, OSError, ValueError):
            pass
        self.frames.put(None)

    def rid(self) -> int:
        self._rid += 1
        return self._rid

    def send(self, msg: dict[str, Any]) -> None:
        self.to.write(proto.encode_json(msg))
        self.to.flush()

    def next(self, timeout: float = 10.0) -> proto.Frame | None:
        return self.frames.get(timeout=timeout)

    def push_of(self, kind: str, timeout: float = 10.0) -> dict[str, Any]:
        """The next pushed message of a kind (earlier ones of other kinds are kept in self.pushed)."""
        for m in self.pushed:
            if m.get("type") == kind:
                self.pushed.remove(m)
                return m
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            f = self.next(deadline - time.monotonic())
            assert f is not None, "the node closed the stream"
            if f.kind == proto.KIND_JSON and f.msg and f.msg.get("type") == kind:
                return f.msg
            if f.kind == proto.KIND_JSON and f.msg:
                self.pushed.append(f.msg)
        raise AssertionError(f"no {kind} within {timeout} s")

    def call(self, op: str, args: dict[str, Any] | None = None, *, upload: bytes | None = None, grant: bool = True,
             cancel_after: int | None = None, timeout: float = 30.0) -> dict[str, Any]:
        """One request. Uploads respect the node's credit; downloads are credited back as they are consumed, and the
        test fails the moment the node sends more than it was granted."""
        rid = self.rid()
        self.send(proto.req(rid, op, args or {}))
        credit, sent, ended = proto.WINDOW, 0, upload is None
        window, got, acked = proto.WINDOW, 0, 0
        out: dict[str, Any] = {"events": [], "data": bytearray(), "credit_frames": 0, "credit_bytes": 0,
                               "max_window_used": 0, "res": None}
        deadline = time.monotonic() + timeout
        while out["res"] is None:
            if not ended:
                assert upload is not None
                while sent < len(upload) and credit > 0:
                    n = min(proto.MAX_DATA, credit, len(upload) - sent)
                    self.to.write(proto.encode_data(rid, upload[sent:sent + n]))
                    sent, credit = sent + n, credit - n
                self.to.flush()
                if sent == len(upload):
                    self.send(proto.end(rid))
                    ended = True
            if cancel_after is not None and len(out["events"]) >= cancel_after:
                self.send(proto.cancel(rid))
                cancel_after = None
            f = self.next(max(0.1, deadline - time.monotonic()))
            assert f is not None, "the node closed the stream"
            if f.kind == proto.KIND_DATA:
                assert f.stream == rid
                got += len(f.data)
                assert got <= window, f"node sent {got} bytes with only {window} granted"
                out["max_window_used"] = max(out["max_window_used"], got - acked)
                out["data"] += f.data
                if grant and got - acked >= STEP:
                    window += got - acked
                    self.send(proto.credit(rid, got - acked))
                    acked = got
                continue
            msg = f.msg or {}
            if msg.get("id") != rid or msg.get("type") not in ("credit", "event", "res"):
                self.pushed.append(msg)
            elif msg["type"] == "credit":
                credit += msg["bytes"]
                out["credit_frames"] += 1
                out["credit_bytes"] += msg["bytes"]
            elif msg["type"] == "event":
                out["events"].append(msg["event"])
            else:
                out["res"] = msg
            assert time.monotonic() < deadline, "timed out"
        out["data"] = bytes(out["data"])
        return out


@pytest.fixture
def node(agent_for, machine):
    """start(policy) -> (server, hub, thread). Closes the pipes at the end."""
    made: list[tuple[Server, IO[bytes], threading.Thread]] = []

    def start(policy: dict[str, Any] | None, *, timing: Timing = FAST, on_stall=None, scanner: bool = True):
        a = agent_for(policy)
        r1, w1 = os.pipe()
        r2, w2 = os.pipe()
        srv = Server(a, os.fdopen(r1, "rb"), os.fdopen(w2, "wb"), timing=timing,
                     on_stall=on_stall or (lambda: None), scanner=scanner)
        th = threading.Thread(target=srv.run, daemon=True)
        th.start()
        hub_out = os.fdopen(w1, "wb")
        made.append((srv, hub_out, th))
        return srv, Hub(hub_out, os.fdopen(r2, "rb")), th

    yield start
    for _srv, hub_out, th in made:
        with contextlib.suppress(OSError):
            hub_out.close()
        th.join(15)
        assert not th.is_alive(), "the node did not exit after stdin closed"


def test_session_hello_ping_sample_requests_and_eof(node, machine, fakebin):
    srv, hub, th = node(manage_policy(machine))
    hello = hub.push_of("hello")
    assert hub.skipped == 0 and hello["proto"] == proto.PROTO and hello["policy"]["mode"] == "manage"
    assert proto.sanitize_push(hello)["gpus"][0]["name"] == "NVIDIA Test GPU"
    hub.send(proto.ping(7))
    assert hub.push_of("pong") == {"type": "pong", "id": 7}
    sample = proto.sanitize_push(hub.push_of("sample"))      # the FIRST sample already knows the containers
    assert sample["mem"]["total"] == 263000000 * 1024 and sample["cpu"]["load1"] == 0.5
    assert sample["seq"] == 1 and sample["gpus"][GPU_A]["procs"][0]["container"] == "ec-owned-1"
    assert sample["containers"]["ec-owned-1"]["mem"]["source"] == "cgroup"
    disks = hub.push_of("disks")
    assert {m["mount"] for m in disks["mounts"]} >= {"/", str(machine.data)}
    inv = hub.push_of("inventory")
    assert {c["name"] for c in inv["containers"]} >= {"ec-owned-1", "other-engine"}
    info = hub.call("node.info")["res"]
    assert info["ok"] and info["result"]["node_id"] == hello["node_id"] and "type" not in info["result"]
    assert hub.push_of("models_changed")["generation"] >= 1
    listed = hub.call("models.list", {"limit": 5})["res"]
    assert listed["ok"] and isinstance(listed["result"]["entries"], list)
    hub.to.close()                                    # stdin EOF: the node exits
    th.join(10)
    assert not th.is_alive() and srv.exit_code == 0


def test_upload_bigger_than_the_window_flows_on_credit(node, machine):
    _srv, hub, _th = node(manage_policy(machine))
    hub.push_of("hello")
    payload = os.urandom(proto.WINDOW + 3 * 1024 * 1024 + 123)
    sha = hashlib.sha256(payload).hexdigest()
    out = hub.call("fs.write", {"root": "gguf", "rel": "big/model.gguf", "size": len(payload), "sha256": sha},
                   upload=payload)
    assert out["res"]["ok"], out["res"]
    assert out["res"]["result"] == {"size": len(payload), "sha256": sha}
    assert out["credit_frames"] >= 1 and out["credit_bytes"] >= len(payload) - proto.WINDOW
    assert (machine.data / "gguf" / "big" / "model.gguf").read_bytes() == payload

    # the same bytes back: a download that respects the hub's credit
    down = hub.call("fs.read", {"root": "gguf", "rel": "big/model.gguf"})
    assert down["res"]["ok"] and down["data"] == payload and down["res"]["result"]["size"] == len(payload)


def test_download_stops_at_the_credit_until_more_is_granted(node, machine):
    _srv, hub, _th = node(manage_policy(machine))
    hub.push_of("hello")
    blob = os.urandom(proto.WINDOW + 2 * STEP)
    (machine.data / "gguf" / "x.gguf").write_bytes(blob)
    rid = hub.rid()
    hub.send(proto.req(rid, "fs.read", {"root": "gguf", "rel": "x.gguf"}))
    got = 0
    deadline = time.monotonic() + 15
    while got < proto.WINDOW and time.monotonic() < deadline:
        f = hub.next(5)
        if f is not None and f.kind == proto.KIND_DATA:
            got += len(f.data)
    assert got == proto.WINDOW
    quiet_until = time.monotonic() + 1.0             # for a second: pushes yes, DATA no (no credit left)
    while time.monotonic() < quiet_until:
        try:
            f = hub.next(0.1)
        except queue.Empty:
            continue
        assert f is not None and f.kind == proto.KIND_JSON, "DATA beyond the granted credit"
    hub.send(proto.credit(rid, len(blob)))
    while True:
        f = hub.next(10)
        assert f is not None
        if f.kind == proto.KIND_DATA:
            got += len(f.data)
        elif (f.msg or {}).get("type") == "res" and f.msg["id"] == rid:
            assert f.msg["ok"] and got == len(blob)
            break


def test_cancel_stops_a_stream_and_answers_cancelled(node, machine):
    _srv, hub, _th = node(manage_policy(machine))
    hub.push_of("hello")
    out = hub.call("docker.logs_follow", {"name": "ec-owned-1", "tail": 10}, cancel_after=3)
    assert out["res"]["ok"] is False and out["res"]["error"]["code"] == proto.CANCELLED
    lines = [e["line"] for e in out["events"]]
    assert "INFO ready" in lines and not any("sk-test-not-a-real-key" in line for line in lines)


def test_unknown_op_bad_args_and_unknown_messages(node, machine):
    _srv, hub, _th = node(manage_policy(machine))
    hub.push_of("hello")
    assert hub.call("shell.exec", {"cmd": "id"})["res"]["error"]["code"] == proto.UNKNOWN_OP
    res = hub.call("fs.read", {"root": "gguf"})["res"]
    assert res["error"]["code"] == proto.BAD_ARGS and "rel" in res["error"]["message"]
    hub.send({"type": "who-knows", "id": 5})
    hub.send(proto.cancel(999_999))                    # cancel / credit / end for unknown ids are ignored
    hub.send(proto.credit(999_999, 5))
    hub.send(proto.end(999_999))
    hub.send(proto.ping(8))
    assert hub.push_of("pong")["id"] == 8


def test_monitor_mode_refuses_every_op_that_changes_the_machine(node, machine):
    _srv, hub, _th = node(manage_policy(machine, mode="monitor"), scanner=False)
    hub.push_of("hello")
    for op in sorted(proto.OPS - proto.MONITOR_OPS):
        res = hub.call(op, {})["res"]
        assert res["ok"] is False and res["error"]["code"] == proto.MONITOR_ONLY, op
        assert "monitor mode" in res["error"]["message"]
    assert hub.call("node.info")["res"]["ok"]
    assert hub.call("docker.images")["res"]["ok"]


def test_no_policy_file_is_monitor_mode_too(node, fakebin):
    _srv, hub, _th = node(None, scanner=False)
    hello = hub.push_of("hello")
    assert hello["policy"]["mode"] == "monitor" and "no policy file" in hello["policy"]["error"]
    res = hub.call("docker.run", {"spec": {"name": "ec-x"}})["res"]
    assert res["error"]["code"] == proto.MONITOR_ONLY


def test_busy_when_every_slot_is_taken(node, machine, monkeypatch):
    monkeypatch.setattr(proto, "MAX_INFLIGHT", 1)
    _srv, hub, _th = node(manage_policy(machine), scanner=False)
    hub.push_of("hello")
    rid = hub.rid()
    hub.send(proto.req(rid, "docker.logs_follow", {"name": "ec-owned-1"}))
    time.sleep(0.5)
    res = hub.call("node.info")["res"]
    assert res["error"]["code"] == proto.BUSY
    hub.send(proto.cancel(rid))


def test_refused_upload_answers_at_once_and_drops_the_rest(node, machine):
    _srv, hub, _th = node(manage_policy(machine), scanner=False)
    hub.push_of("hello")
    out = hub.call("fs.write", {"root": "ro", "rel": "x.bin", "size": 3 * STEP}, upload=b"z" * (3 * STEP))
    assert out["res"]["error"]["code"] == proto.POLICY_DENIED
    hub.send(proto.ping(9))                           # the link is still fine
    assert hub.push_of("pong")["id"] == 9


def test_upload_beyond_credit_fails_that_request_only(node, machine, monkeypatch):
    real = ops.fs_write

    def slow(*a, **k):                                # the node grants nothing while the hub floods it
        time.sleep(1.0)
        return real(*a, **k)

    monkeypatch.setattr(ops, "fs_write", slow)
    _srv, hub, _th = node(manage_policy(machine), scanner=False)
    hub.push_of("hello")
    rid = hub.rid()
    size = proto.WINDOW + 4 * STEP
    hub.send(proto.req(rid, "fs.write", {"root": "gguf", "rel": "greedy.bin", "size": size}))
    chunk = b"q" * proto.MAX_DATA
    for _ in range(size // proto.MAX_DATA):           # ignores credit on purpose
        hub.to.write(proto.encode_data(rid, chunk))
    hub.to.flush()
    hub.send(proto.end(rid))
    deadline = time.monotonic() + 15
    res = None
    while res is None and time.monotonic() < deadline:
        f = hub.next(5)
        if f is not None and f.kind == proto.KIND_JSON and f.msg.get("type") == "res" and f.msg["id"] == rid:
            res = f.msg
    assert res is not None and res["ok"] is False and res["error"]["code"] == proto.BAD_ARGS
    assert "credit" in res["error"]["message"]
    assert not (machine.data / "gguf" / "greedy.bin").exists()
    hub.send(proto.ping(10))
    assert hub.push_of("pong")["id"] == 10


def test_hub_silence_ends_the_session(node, machine):
    srv, hub, th = node(manage_policy(machine), scanner=False,
                        timing=Timing(sample_s=0.2, hub_silence_s=0.6, watchdog_s=0.05))
    hub.push_of("hello")
    th.join(10)
    assert not th.is_alive()


def test_protocol_error_from_the_hub_drops_the_link(node, machine):
    srv, hub, th = node(manage_policy(machine), scanner=False)
    hub.push_of("hello")
    hub.to.write(b"\x00\x00\x00\x05\x09junk")          # unknown frame kind
    hub.to.flush()
    th.join(10)
    assert not th.is_alive() and srv.exit_code == 2


def test_blocked_writer_trips_the_watchdog(agent_for, machine):
    gate = threading.Event()
    stalled = threading.Event()

    class Stuck:
        def write(self, b: bytes) -> int:
            gate.wait(10)
            return len(b)

        def flush(self) -> None:
            pass

    r, w = os.pipe()
    srv = Server(agent_for(manage_policy(machine)), os.fdopen(r, "rb"), Stuck(),  # type: ignore[arg-type]
                 timing=Timing(write_stall_s=0.3, watchdog_s=0.05), on_stall=stalled.set, scanner=False)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    try:
        assert stalled.wait(5), "a blocked write did not trip the watchdog"
    finally:
        gate.set()
        os.close(w)
        th.join(10)
