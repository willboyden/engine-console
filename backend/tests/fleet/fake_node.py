"""A frame-level fake `ecnode --stdio` for the hub's link tests.

Run as a program (`python fake_node.py '<scenario json>'`) it speaks the real wire protocol with the real
`ecnode.proto` framing, so the hub's `NodeLink` exercises its true subprocess path: spawn, preamble, hello, pushes,
requests, DATA with credit, cancel, silence, flood, crash. The scenario (a JSON object, see DEFAULTS) scripts what it
does; `log` names a JSONL file the node appends everything it saw and did to, so a test can assert on the far side.

Imported as a module it only offers `pattern()` and `read_log()`: the deterministic file content the node streams and a
log reader.

Requests are served in one thread each, so a hanging request never blocks pings, samples or other requests.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ecnode import proto

GIB = 1024**3
DEFAULTS: dict[str, Any] = {
    "log": None,                  # path of a JSONL file the node appends its observations to
    "junk": "",                   # text printed before MAGIC (a login shell's noise)
    "junk_lines": 0,              # this many 1 KiB lines of junk, and no MAGIC after them
    "stderr": "",                 # written to stderr at start
    "stderr_noise": 0,            # this many "noise line" lines are written to stderr before `stderr`
    "exit_before_hello": None,    # exit code: die after the stderr text without speaking
    "hello_delay_s": 0.0,
    "first_frame": "hello",       # "hello" | "sample" | "fatal" | "none" (MAGIC and then nothing)
    "proto": proto.PROTO,
    "version": "0.4.0-test",
    "forced_cmd": True,
    "gpus": 2,
    "hello_patch": {},            # deep-merged into the hello
    "hello_extra": {},            # top-level keys added to the hello (fields the hub must drop)
    "info_mode": None,            # node.info answers with this policy mode
    "samples": True,
    "sample_interval_s": 0.05,
    "samples_stop_after_s": None,
    "rogue_gpu": True,            # samples carry a GPU uuid the hello never announced
    "silent_after_hello": False,  # stop talking and reading entirely
    "no_pong": False,
    "exit_after_s": None,
    "exit_code": 3,
    "pushes": [],                 # [{"after_s": 0.1, "msg": {...}} | {"after_s": 0.1, "raw": "<json text>"} | {"after_s", "data_stream", "bytes"}]
    "push_disks": None,
    "push_inventory": None,
    "flood": None,                # {"after_s": 0.1, "frame_kb": 64, "count": 100}
    "ignore_cancel": False,
    "ignore_credit": False,       # fs.read sends everything at once, credit or not
    "data_chunk": 256 * 1024,
    "upload_hold_s": 0.0,         # fs.write does not read the upload (or grant credit) for this long
    "upload_consume_delay_s": 0.0,
    "credit_batch": 1024 * 1024,  # fs.write grants credit after consuming this much
    "crash_after_bytes": None,    # fs.read dies after sending this much
    "follow_delay_s": 0.0,
    "images": [{"ref": "vllm/vllm-openai:v0.1", "id": "sha256:abc", "size": 10 * GIB, "created": 1.7e9, "arch": "amd64"}],
    "hostile_images": False,
    "spawn_child": False,         # start a grandchild (sleep) in the node's process group and log its pid
}


def pattern(offset: int, n: int) -> bytes:
    """Bytes offset..offset+n of the node's deterministic test file: each 4 KiB block carries its own index."""
    if n <= 0:
        return b""
    first, last = offset // 4096, (offset + n - 1) // 4096
    blob = b"".join(bytes([k % 251]) * 4096 for k in range(first, last + 1))
    start = offset - first * 4096
    return blob[start:start + n]


def read_log(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


class Req:
    """One request being served: its cancel flag, the node's send credit for it, and the hub's upload."""

    def __init__(self, rid: int, op: str, args: dict[str, Any]) -> None:
        self.rid, self.op, self.args = rid, op, args
        self.cancel = threading.Event()
        self.cv = threading.Condition()
        self.credit = proto.WINDOW                       # what the hub has granted this node to send on the stream
        self.inbox: queue.Queue[tuple[str, bytes]] = queue.Queue()
        self.received = 0                                # upload bytes that arrived
        self.granted = 0                                 # credit this node handed back for the upload


class FakeNode:
    def __init__(self, sc: dict[str, Any]) -> None:
        self.sc = {**DEFAULTS, **sc}
        self.out = sys.stdout.buffer
        self.out_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.reqs: dict[int, Req] = {}
        self.t0 = time.monotonic()

    # ── output ────────────────────────────────────────────────────────────────────────────────────────────
    def log(self, ev: str, **kw: Any) -> None:
        path = self.sc["log"]
        if not path:
            return
        with self.log_lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ev": ev, "t": round(time.monotonic() - self.t0, 3), **kw}) + "\n")

    def write(self, data: bytes) -> None:
        try:
            with self.out_lock:
                self.out.write(data)
                self.out.flush()
        except (BrokenPipeError, OSError):
            os._exit(0)                                  # the hub is gone

    def send(self, msg: dict[str, Any]) -> None:
        self.write(proto.encode_json(msg))

    def send_raw_json(self, text: str) -> None:
        body = text.encode()
        self.write(struct.pack(">IB", len(body) + 1, proto.KIND_JSON) + body)

    def send_data(self, stream: int, payload: bytes) -> None:
        self.write(proto.encode_data(stream, payload))

    # ── messages ──────────────────────────────────────────────────────────────────────────────────────────
    def hello(self) -> dict[str, Any]:
        gpus = [{"index": i, "uuid": f"GPU-{i:04d}", "name": "Test GPU 96GB", "total_bytes": 96 * GIB,
                 "compute_capability": "12.0", "unified_memory": False, "pci_bus_id": f"0000:0{i}:00.0"}
                for i in range(self.sc["gpus"])]
        msg: dict[str, Any] = {
            "type": "hello", "proto": self.sc["proto"], "version": self.sc["version"], "node_id": "feedc0de12345678",
            "host": {"hostname": "gpu-box-2", "arch": "x86_64", "os": "Ubuntu 24.04", "kernel": "6.8.0-test",
                     "cpu_model": "Test CPU", "cpu_cores": 16, "cpu_threads": 32, "ram_total_bytes": 256 * GIB,
                     "boot_time": 1.7e9, "python": "3.12.3"},
            "gpu": {"driver": "570.00", "cuda": "12.8", "source": "nvml"},
            "gpus": gpus,
            "docker": {"available": True, "context": "default", "rootless": True, "version": "27.0.0",
                       "nvidia_runtime": True},
            "addresses": [{"iface": "eth0", "addr": "10.0.0.5", "prefix": 24}],
            "policy": {"path": "/etc/ecnode/policy.json", "mode": "monitor", "engine_images": ["vllm/vllm-openai:*"],
                       "gateway_image": "nginx@sha256:00", "engine_network": "engine-console-engines",
                       "port_range": [18000, 18099], "discovery_ports": [8000],
                       "model_roots": [{"id": "hf", "path": "/data/hf", "kind": "hf-home", "writable": True,
                                        "label": "HF cache"}],
                       "scripts": [], "allow_pull": False, "allow_delete": False, "allow_ipc_host": True,
                       "external_control": "none", "max_shm_gib": 64},
            "forced_cmd": self.sc["forced_cmd"], "via_ssh": False,
        }
        _merge(msg, self.sc["hello_patch"])
        msg.update(self.sc["hello_extra"])
        return msg

    def sample(self, seq: int) -> dict[str, Any]:
        gpus: dict[str, Any] = {
            f"GPU-{i:04d}": {"util_pct": 12.5, "mem_used_bytes": 10 * GIB, "mem_total_bytes": 96 * GIB, "temp_c": 41.0,
                             "power_w": 123.0, "power_limit_w": 300.0, "fan_pct": 30.0,
                             "procs": [{"pid": 1000 + i, "used_bytes": 8 * GIB, "container": "ec-engine-1", "name": "python3"}]}
            for i in range(self.sc["gpus"])}
        if self.sc["rogue_gpu"]:
            gpus["GPU-ROGUE"] = {"util_pct": 99.0}
        return {"type": "sample", "seq": seq, "t": time.time(), "gpus": gpus,
                "mem": {"total": 256 * GIB, "available": 200 * GIB, "free": 150 * GIB, "cached": 40 * GIB,
                        "shmem": GIB, "swap_total": 8 * GIB, "swap_free": 8 * GIB},
                "cpu": {"load1": 1.0, "load5": 0.8, "load15": 0.5, "util_pct": 7.5},
                "containers": {"ec-engine-1": {"mem": {"current": 30 * GIB, "anon": 20 * GIB, "file": 9 * GIB,
                                                      "shmem": GIB, "kernel": GIB, "source": "cgroup"}, "cpu_pct": 55.0}}}

    # ── background behaviours ─────────────────────────────────────────────────────────────────────────────
    def sampler(self) -> None:
        seq = 0
        stop_after = self.sc["samples_stop_after_s"]
        while True:
            if stop_after is not None and time.monotonic() - self.t0 > stop_after:
                return
            seq += 1
            self.send(self.sample(seq))
            time.sleep(self.sc["sample_interval_s"])

    def scheduled(self) -> None:
        pushes = sorted(self.sc["pushes"], key=lambda p: p.get("after_s", 0))
        for p in pushes:
            time.sleep(max(0.0, p.get("after_s", 0) - (time.monotonic() - self.t0)))
            if "raw" in p:
                self.send_raw_json(p["raw"])
            elif "data_stream" in p:
                self.send_data(p["data_stream"], b"x" * p.get("bytes", 10))
            else:
                self.send(p["msg"])

    def flooder(self) -> None:
        f = self.sc["flood"]
        time.sleep(f.get("after_s", 0))
        body = {"type": "log", "level": "info", "msg": "x" * (f.get("frame_kb", 1) * 1024)}
        for _ in range(f.get("count", 100)):
            self.send(body)

    def exiter(self) -> None:
        time.sleep(self.sc["exit_after_s"])
        sys.stderr.write("fake node crashing on purpose\n")
        sys.stderr.flush()
        self.log("exit", code=self.sc["exit_code"])
        os._exit(self.sc["exit_code"])

    # ── requests ──────────────────────────────────────────────────────────────────────────────────────────
    def serve(self, r: Req) -> None:
        try:
            getattr(self, "op_" + r.op.replace(".", "_"), self.op_unknown)(r)
        finally:
            self.reqs.pop(r.rid, None)

    def op_unknown(self, r: Req) -> None:
        self.send(proto.res_err(r.rid, proto.UNKNOWN_OP, f"the fake node does not do {r.op}"))

    def cancelled(self, r: Req) -> None:
        if not self.sc["ignore_cancel"]:
            self.send(proto.res_err(r.rid, proto.CANCELLED, "cancelled"))

    def op_node_info(self, r: Req) -> None:
        h = self.hello()
        h.pop("type")
        if self.sc["info_mode"]:
            h["policy"]["mode"] = self.sc["info_mode"]
        self.send(proto.res_ok(r.rid, h))

    def op_metrics_configure(self, r: Req) -> None:
        self.send(proto.res_ok(r.rid, {}))

    def op_docker_images(self, r: Req) -> None:
        if self.sc["hostile_images"]:
            self.send(proto.res_ok(r.rid, {"images": [{"ref": "r" * 5000, "id": "i\x00d", "size": 5, "evil": "x"}],
                                           "evil_top": {"deep": [1, 2]}}))
        else:
            self.send(proto.res_ok(r.rid, {"images": self.sc["images"]}))

    def op_docker_image_present(self, r: Req) -> None:
        if r.args.get("image") == "missing":
            self.send(proto.res_err(r.rid, proto.NOT_FOUND, "no such image"))
        else:
            self.send(proto.res_ok(r.rid, {"present": True, "id": "sha256:abc"}))

    def op_docker_inspect(self, r: Req) -> None:
        name = str(r.args.get("name"))
        if name.startswith("err-"):
            self.send(proto.res_err(r.rid, name[4:], "scripted refusal"))
        elif name == "no-error-object":
            self.send({"type": "res", "id": r.rid, "ok": False})
        elif name == "ctl":
            self.send(proto.res_err(r.rid, proto.IO_ERROR, "disk\x07 full\nsecond line hf_abcdefghijklmnop"))
        elif name.startswith("slow-"):
            time.sleep(0.4)
            self.send(proto.res_ok(r.rid, {"info": {"name": name, "state": "running"}}))
        else:
            self.send(proto.res_ok(r.rid, {"info": {"name": name, "state": "running"}}))

    def op_docker_stop(self, r: Req) -> None:
        self.send(proto.res_ok(r.rid, {}))

    def op_models_scan(self, r: Req) -> None:               # the op the tests use to "hang"
        while not r.cancel.wait(0.05):
            pass
        self.cancelled(r)

    def op_docker_logs_follow(self, r: Req) -> None:
        total = int(r.args.get("tail") or 0)                # 0: follow until cancelled
        n = 0
        # a node that ignores cancel (scenario ignore_cancel) keeps sending events after it, which the hub must not keep
        while not (r.cancel.is_set() and not self.sc["ignore_cancel"]) and (total == 0 or n < total):
            n += 1
            self.send(proto.event(r.rid, {"line": f"line {n}"}))
            self.log("follow", id=r.rid, n=n)
            if self.sc["follow_delay_s"]:
                time.sleep(self.sc["follow_delay_s"])
        if r.cancel.is_set():
            self.cancelled(r)
        else:
            self.send(proto.res_ok(r.rid, {"lines": n}))

    def op_docker_pull(self, r: Req) -> None:
        for i in range(3):
            self.send(proto.event(r.rid, {"line": f"layer {i}"}))
        self.send({"type": "event", "id": r.rid, "event": {"unknown_field": 1}})   # an event the hub must drop
        self.send(proto.res_err(r.rid, proto.DOCKER_ERROR, "pull failed (scripted)"))

    def op_scripts_run(self, r: Req) -> None:
        for i in range(2):
            self.send(proto.event(r.rid, {"stream": "out", "line": f"step {i}"}))
        self.send(proto.res_ok(r.rid, {"rc": 0}))

    def op_http_request(self, r: Req) -> None:
        body_len = int(r.args.get("body_len") or 0)
        got = 0
        while got < body_len:
            kind, data = r.inbox.get()
            if kind == "end":
                break
            got += len(data)
        self.send(proto.event(r.rid, {"status": 200, "headers": {"content-type": "text/plain"}}))
        body = f"engine answer, request body was {got} bytes".encode()
        self.send_data(r.rid, body)
        self.send(proto.res_ok(r.rid, {"status": 200, "bytes": len(body)}))

    def op_fs_read(self, r: Req) -> None:
        length, offset, chunk = int(r.args.get("length") or 0), int(r.args.get("offset") or 0), self.sc["data_chunk"]
        sent = 0
        while sent < length and not r.cancel.is_set():
            n = min(chunk, length - sent)
            if not self.sc["ignore_credit"]:
                with r.cv:
                    if r.credit < n:
                        self.log("stalled", id=r.rid, sent=sent, credit=r.credit)
                    while r.credit < n and not r.cancel.is_set():
                        r.cv.wait(0.1)
                    r.credit -= n
            if r.cancel.is_set():
                break
            self.send_data(r.rid, pattern(offset + sent, n))
            sent += n
            self.log("sent", id=r.rid, sent=sent)
            if self.sc["crash_after_bytes"] and sent >= self.sc["crash_after_bytes"]:
                os._exit(7)
        if r.cancel.is_set():
            self.cancelled(r)
        else:
            self.send(proto.res_ok(r.rid, {"size": sent}))

    def op_fs_write(self, r: Req) -> None:
        deadline = time.monotonic() + self.sc["upload_hold_s"]
        while time.monotonic() < deadline and not r.cancel.is_set():
            time.sleep(0.02)
        digest, total, since = hashlib.sha256(), 0, 0
        while not r.cancel.is_set():
            try:
                kind, data = r.inbox.get(timeout=0.05)
            except queue.Empty:
                continue
            if kind == "end":
                self.send(proto.res_ok(r.rid, {"size": total, "sha256": digest.hexdigest()}))
                return
            if self.sc["upload_consume_delay_s"]:
                time.sleep(self.sc["upload_consume_delay_s"])
            digest.update(data)
            total += len(data)
            since += len(data)
            self.log("consumed", id=r.rid, total=total)
            if since >= self.sc["credit_batch"]:
                r.granted += since
                self.send(proto.credit(r.rid, since))
                since = 0
        self.cancelled(r)

    # ── the main loop ─────────────────────────────────────────────────────────────────────────────────────
    def dispatch(self, msg: dict[str, Any]) -> None:
        t = msg.get("type")
        rid = msg.get("id")
        if t == "ping":
            self.log("ping", id=rid)
            if not self.sc["no_pong"]:
                self.send(proto.pong(rid))
        elif t == "req":
            self.log("req", id=rid, op=msg.get("op"), args=msg.get("args"))
            r = Req(rid, msg["op"], msg.get("args") or {})
            self.reqs[rid] = r
            threading.Thread(target=self.serve, args=(r,), daemon=True).start()
        elif t == "cancel":
            self.log("cancel", id=rid)
            r = self.reqs.get(rid)
            if r is not None:
                r.cancel.set()
        elif t == "credit":
            self.log("credit", id=rid, bytes=msg.get("bytes"))
            r = self.reqs.get(rid)
            if r is not None:
                with r.cv:
                    r.credit += int(msg.get("bytes") or 0)
                    r.cv.notify_all()
        elif t == "end":
            self.log("end", id=rid)
            r = self.reqs.get(rid)
            if r is not None:
                r.inbox.put(("end", b""))
        else:
            self.log("other", type=t)

    def main(self) -> None:
        sc = self.sc
        if sc["stderr_noise"]:
            sys.stderr.write("noise line\n" * sc["stderr_noise"])
        if sc["stderr"]:
            sys.stderr.write(sc["stderr"] + "\n")
        sys.stderr.flush()
        self.log("start", pid=os.getpid(), env=sorted(os.environ))
        if sc["exit_before_hello"] is not None:
            os._exit(int(sc["exit_before_hello"]))
        if sc["junk"]:
            self.write(sc["junk"].encode())
        if sc["junk_lines"]:
            for _ in range(sc["junk_lines"]):
                self.write(b"j" * 1023 + b"\n")
            time.sleep(30)                                   # never speaks the protocol
            return
        time.sleep(sc["hello_delay_s"])
        self.write(proto.MAGIC)
        first = sc["first_frame"]
        if first == "none":
            time.sleep(30)
            return
        if first == "fatal":
            self.send({"type": "fatal", "error": "scripted fatal: no GPU support\nsecond line"})
            os._exit(1)
        self.send(self.sample(0) if first == "sample" else self.hello())
        if sc["spawn_child"]:
            child = subprocess.Popen(["sleep", "120"])
            self.log("child", pid=child.pid)
        if sc["silent_after_hello"]:
            time.sleep(3600)
        if sc["push_disks"] is not None:
            self.send({"type": "disks", "t": time.time(), **sc["push_disks"]})
        if sc["push_inventory"] is not None:
            self.send({"type": "inventory", "t": time.time(), **sc["push_inventory"]})
        if sc["samples"]:
            threading.Thread(target=self.sampler, daemon=True).start()
        if sc["pushes"]:
            threading.Thread(target=self.scheduled, daemon=True).start()
        if sc["flood"]:
            threading.Thread(target=self.flooder, daemon=True).start()
        if sc["exit_after_s"] is not None:
            threading.Thread(target=self.exiter, daemon=True).start()
        stdin = sys.stdin.buffer
        while True:
            frame = proto.read_frame_sync(stdin)
            if frame is None:
                self.log("eof")
                os._exit(0)
            if frame.kind == proto.KIND_DATA:
                r = self.reqs.get(frame.stream)
                if r is not None:
                    r.received += len(frame.data)
                    if r.received > proto.WINDOW + r.granted:
                        self.log("violation", id=r.rid, received=r.received, granted=r.granted)
                    self.log("data", id=r.rid, total=r.received)
                    r.inbox.put(("data", frame.data))
            elif frame.msg is not None:
                self.dispatch(frame.msg)


if __name__ == "__main__":
    FakeNode(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}).main()
