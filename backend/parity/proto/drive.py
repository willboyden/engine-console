"""Hub-side ecnode protocol driver (reference). Uses ONLY ecnode.proto, the locked wire module.

    "$(node backend/parity/lib/python-release.mjs)/node/.venv/bin/python" backend/parity/proto/drive.py \
        --node '<argv as JSON>' --scenario <file.json>

Spawns the node (it inherits this process's environment: the caller builds a hermetic one), skips the preamble,
reads hello, waits for the first sample, disks and inventory pushes, then runs the scenario's steps and prints a
JSON transcript on stdout. backend/crates/ec-proto/src/bin/ecproto-drive.rs is the same driver in Rust: the two must
produce the same transcript for the same node (backend/parity/proto/matrix.mjs checks it). Step kinds and record
shapes: backend/parity/proto/NORMALISE.md.

Credit is handled like the hub's _Stream: downloads are credited back in CREDIT_BATCH (WINDOW/4) steps as they are
consumed; uploads never exceed the node's credit. Every frame the node sends beyond its credit is recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
from typing import Any

from ecnode import proto

CREDIT_BATCH = proto.WINDOW // 4
_BASE = bytes((i * 31 + 7) % 251 for i in range(251))


def pattern(n: int) -> bytes:
    """The upload payload: byte i is (i * 31 + 7) % 251."""
    return (_BASE * (n // 251 + 1))[:n]


class Req:
    def __init__(self, rid: int, upload: bytes | None, grant: bool, cancel_after: int | None,
                 cancel_after_events: int | None = None) -> None:
        self.rid = rid
        self.events: list[Any] = []
        self.res: dict[str, Any] | None = None
        self.hasher = hashlib.sha256()
        self.got = 0
        self.recv_window = proto.WINDOW
        self.unacked = 0
        self.violation = False
        self.credit_from_node = 0
        self.upload = upload
        self.sent = 0
        self.send_credit = proto.WINDOW
        self.ended = upload is None
        self.grant = grant
        self.cancel_after = cancel_after
        self.cancel_after_events = cancel_after_events
        self.cancel_sent = False


class Driver:
    def __init__(self, argv: list[str]) -> None:
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.q: queue.Queue[proto.Frame | None] = queue.Queue()
        self.skipped: int | None = None
        self.err = bytearray()
        self.hello: dict[str, Any] | None = None
        self.first: dict[str, Any] = {}
        self.counts: dict[str, int] = {}
        self.last: dict[str, Any] = {}
        self.pongs: dict[int, Any] = {}
        self.active: dict[int, Req] = {}
        self.closed = False
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()

    def _read(self) -> None:
        assert self.p.stdout is not None
        try:
            self.skipped = proto.find_magic_sync(self.p.stdout)
            while (f := proto.read_frame_sync(self.p.stdout)) is not None:
                self.q.put(f)
        except (proto.ProtoError, OSError, ValueError):
            pass
        self.q.put(None)

    def _read_err(self) -> None:
        assert self.p.stderr is not None
        while chunk := self.p.stderr.read(4096):
            self.err += chunk

    # ── sending ──────────────────────────────────────────────────────────────────────────────────────────────
    def raw(self, b: bytes) -> None:
        if self.closed:
            return
        try:
            assert self.p.stdin is not None
            self.p.stdin.write(b)
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            self.closed = True

    def send(self, msg: dict[str, Any]) -> None:
        self.raw(proto.encode_json(msg))

    # ── receiving ────────────────────────────────────────────────────────────────────────────────────────────
    def pump(self, timeout: float) -> bool:
        """Handle one frame (or none within timeout); False once the node's stream ended."""
        for r in list(self.active.values()):
            self._feed_upload(r)
        try:
            f = self.q.get(timeout=timeout)
        except queue.Empty:
            return True
        if f is None:
            self.q.put(None)
            return False
        if f.kind == proto.KIND_DATA:
            r = self.active.get(f.stream)
            if r is not None:
                self._data(r, f.data)
            return True
        msg = f.msg or {}
        t, rid = msg.get("type"), msg.get("id")
        r = self.active.get(rid) if isinstance(rid, int) and not isinstance(rid, bool) else None
        if t in ("res", "event", "credit") and r is not None:
            if t == "res":
                r.res = msg
            elif t == "event":
                r.events.append(msg.get("event"))
                if r.cancel_after_events is not None and not r.cancel_sent and len(r.events) >= r.cancel_after_events:
                    r.cancel_sent = True
                    self.send(proto.cancel(r.rid))
            else:
                r.credit_from_node += int(msg.get("bytes") or 0)
                r.send_credit += int(msg.get("bytes") or 0)
            return True
        if t == "pong":
            self.pongs[rid] = msg
            return True
        if t == "hello" and self.hello is None:
            self.hello = msg
            return True
        if isinstance(t, str):
            self.counts[t] = self.counts.get(t, 0) + 1
            self.first.setdefault(t, msg)
            self.last[t] = msg
        return True

    def _data(self, r: Req, data: bytes) -> None:
        r.got += len(data)
        r.hasher.update(data)
        if len(data) > r.recv_window:
            r.violation = True
        r.recv_window -= len(data)
        r.unacked += len(data)
        if r.cancel_after is not None and not r.cancel_sent and r.got >= r.cancel_after:
            r.cancel_sent = True
            self.send(proto.cancel(r.rid))
        if r.grant and r.unacked >= CREDIT_BATCH and r.res is None and not r.cancel_sent:
            self.send(proto.credit(r.rid, r.unacked))
            r.recv_window += r.unacked
            r.unacked = 0

    def _feed_upload(self, r: Req) -> None:
        if r.ended or r.upload is None or r.res is not None:
            return
        while r.sent < len(r.upload) and r.send_credit > 0:
            n = min(proto.MAX_DATA, r.send_credit, len(r.upload) - r.sent)
            self.raw(proto.encode_data(r.rid, r.upload[r.sent:r.sent + n]))
            r.sent += n
            r.send_credit -= n
        if r.sent == len(r.upload):
            self.send(proto.end(r.rid))
            r.ended = True

    def wait(self, pred: Any, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not pred():
            if time.monotonic() > deadline:
                return False
            if not self.pump(0.05):
                return pred()
        return True


def record_req(r: Req, timed_out: bool) -> dict[str, Any]:
    rec: dict[str, Any] = {"res": r.res, "events": r.events, "credit_from_node": r.credit_from_node,
                           "credit_violation": r.violation}
    if timed_out:
        rec["timeout"] = True
    if r.cancel_after is not None:
        rec["data"] = {"bytes_ge_cancel": r.got >= r.cancel_after}
    elif r.cancel_after_events is not None:
        rec["events"] = r.events[:r.cancel_after_events]     # more may arrive before the cancel lands
        rec["data"] = None
    elif r.got:
        rec["data"] = {"bytes": r.got, "sha256": r.hasher.hexdigest()}
    else:
        rec["data"] = None
    return rec


def run(node_argv: list[str], scenario: dict[str, Any]) -> dict[str, Any]:
    d = Driver(node_argv)
    out: dict[str, Any] = {"scenario": scenario["name"], "steps": []}
    d.wait(lambda: d.hello is not None, 25)
    out["skipped"] = d.skipped
    out["hello"] = d.hello
    d.wait(lambda: all(k in d.first for k in ("sample", "disks", "inventory")), 15)
    out["initial"] = {k: d.first.get(k) for k in ("sample", "disks", "inventory")}
    next_id = 1
    for step in scenario["steps"]:
        kind = step["do"]
        rec: dict[str, Any] = {"do": kind}
        if kind == "ping":
            d.send(proto.ping(step["id"]))
            ok = d.wait(lambda: step["id"] in d.pongs, 10)
            rec["pong"] = d.pongs.get(step["id"]) if ok else None
        elif kind == "req":
            rid = step.get("id", next_id)
            next_id = max(next_id, rid) + 1
            msg: dict[str, Any] = {"type": "req", "id": rid}
            if not step.get("op_missing"):
                msg["op"] = step["op"]
            upload = pattern(step["upload"]) if "upload" in step else None
            if not step.get("args_missing"):
                args = step.get("args", {})
                if isinstance(args, dict) and args.get("sha256") == "$upload_sha256" and upload is not None:
                    args = {**args, "sha256": hashlib.sha256(upload).hexdigest()}
                msg["args"] = args
            r = Req(rid, upload, step.get("grant", True), step.get("cancel_after_bytes"),
                    step.get("cancel_after_events"))
            d.active[rid] = r
            d.send(msg)
            done = d.wait(lambda: r.res is not None, step.get("timeout", 30))
            del d.active[rid]
            rec.update(id=rid, **record_req(r, not done))
        elif kind == "send_reqs":
            for i in range(step["count"]):
                rid = step["from"] + i
                args = json.loads(json.dumps(step["args"]).replace("{i}", str(i)))
                d.active[rid] = Req(rid, None, True, None)
                d.send(proto.req(rid, step["op"], args))
            rec["sent"] = step["count"]
        elif kind == "send_req":
            d.send(proto.req(step["id"], step["op"], step.get("args", {})))
        elif kind == "cancel_all":
            ids = [step["from"] + i for i in range(step["count"])]
            for rid in ids:
                d.send(proto.cancel(rid))
            d.wait(lambda: all(d.active[i].res is not None for i in ids), 30)
            rec["res"] = [d.active[i].res for i in ids]
            for i in ids:
                del d.active[i]
        elif kind == "send":
            d.send(step["msg"])
        elif kind == "raw":
            d.raw(bytes.fromhex(step["hex"]))
        elif kind == "sleep":
            end = time.monotonic() + step["s"]
            while time.monotonic() < end:
                d.pump(0.05)
        elif kind == "await_push":
            t, n = step["type"], step.get("count", 1)
            ok = d.wait(lambda: d.counts.get(t, 0) >= n, step.get("timeout", 10))
            rec["msg"] = d.last.get(t) if ok else None
            rec["timeout"] = not ok
        elif kind in ("close", "wait_exit"):
            if kind == "close" and d.p.stdin is not None:
                d.closed = True
                try:
                    d.p.stdin.close()
                except OSError:
                    pass
            deadline = time.monotonic() + step.get("timeout", 20)
            while d.p.poll() is None and time.monotonic() < deadline:
                d.pump(0.05)
            code = d.p.poll()
            if code is None:
                d.p.kill()
                d.p.wait()
            time.sleep(0.2)                       # let the stderr reader catch the last words
            text = d.err.decode("utf-8", "replace")
            rec["exit"] = code if code is not None else "timeout"
            rec["stderr_contains"] = {s: s in text for s in step.get("stderr_contains", [])}
        else:
            raise SystemExit(f"unknown step {kind!r}")
        out["steps"].append(rec)
    if d.p.poll() is None:
        d.p.kill()
        d.p.wait()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True, help="the node's argv, as a JSON list")
    ap.add_argument("--scenario", required=True)
    a = ap.parse_args()
    with open(a.scenario) as f:
        scenario = json.load(f)
    t = run(json.loads(a.node), scenario)
    sys.stdout.write(json.dumps(t, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
