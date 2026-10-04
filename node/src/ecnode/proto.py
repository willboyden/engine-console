"""ecnode wire protocol, version 1: the hub (Engine Console) <-> one node (ecnode) per machine.

LOCKED CONTRACT. docs/FLEET-PROTOCOL.md is the prose version of this file. The hub imports this very module
from the installed `ecnode` package, so both ends agree byte for byte. Changes are additive only (a new op, a
new optional field, a new message type). Anything that would break an older peer bumps PROTO; peers whose
PROTO differs refuse each other at hello time, and the UI says which side to update.

Transport. The node talks on stdin/stdout: as a plain subprocess on the hub's own machine, or through ssh on
any other machine, with a key that authorized_keys restricts to exactly the node command. stderr is free-form
diagnostics; the hub keeps a short tail of it to explain a dropped link.

Preamble. A login shell may print to stdout before the node starts, so the node first writes MAGIC on a line of
its own. The hub discards up to MAX_PREAMBLE bytes looking for a line ending in MAGIC; after it only frames
follow, and the node's first frame is its `hello`.

Frames.    u32 LENGTH (big-endian: bytes that follow) | u8 KIND | BODY
  KIND_JSON  BODY is one UTF-8 JSON object of at most MAX_JSON bytes. NaN and Infinity are invalid.
  KIND_DATA  BODY is u32 STREAM (big-endian; the request id) | payload of at most MAX_DATA bytes.
A LENGTH outside those limits, an unknown KIND or a truncated frame is a protocol error: drop the link.

Requests. The hub sends {"type":"req","id":N,"op":...,"args":{...}}. The node answers with exactly one
{"type":"res","id":N,"ok":true,"result":{...}} or {"type":"res","id":N,"ok":false,"error":{"code","message"}}.
Before that res it may send any number of {"type":"event","id":N,"event":{...}} frames and DATA frames on
stream N; after the res nothing more is sent for N. {"type":"cancel","id":N} asks the node to stop early; it
still answers with one res (error code "cancelled" unless it had already finished).

Uploads (hub -> node bytes: fs.write, an http.request body) are DATA frames on stream N followed by
{"type":"end","id":N}. The node sends its res only after that end (or after a cancel).

Flow control. Every stream starts with WINDOW bytes of credit in each direction. A sender never puts more DATA
on a stream than the credit the receiver has granted; the receiver grants more with
{"type":"credit","id":N,"bytes":k} as it consumes. JSON frames are never credit-limited, so pings and small
requests are never stuck behind a bulk transfer.

Pushed by the node without a request: hello (first), sample (every SAMPLE_S), disks (every DISKS_S), inventory
(on change, at least every INVENTORY_S), metrics (one per engine scrape), models_changed, log, fatal, pong.
Liveness: the hub sends {"type":"ping","id":N} every PING_S and the node answers {"type":"pong","id":N}. The node
exits when stdin closes or nothing has arrived for HUB_SILENCE_S; the hub drops a node silent for SILENT_S.

Trust. The hub treats every node as UNTRUSTED: each node -> hub message is rebuilt from a whitelist
(sanitize_*): unknown fields dropped, numbers finite, strings and lists capped, NULs removed. A node treats the
hub as its operator but obeys its own local policy first; nothing a hub sends can widen that policy. Arguments
reach a node's handlers only through clean_args(), and policy checks happen after it.
"""
from __future__ import annotations

import asyncio
import json
import math
import struct
from dataclasses import dataclass
from typing import IO, Any

PROTO = 1
MAGIC = b"ecnode-proto 1\n"
MAX_PREAMBLE = 64 * 1024
KIND_JSON = 1
KIND_DATA = 2
MAX_JSON = 1024 * 1024
MAX_DATA = 1024 * 1024
WINDOW = 8 * 1024 * 1024
MAX_ID = 2**31 - 1
MAX_INFLIGHT = 64              # requests a node runs at once; more are refused with "busy"

SAMPLE_S = 2.0
DISKS_S = 30.0
INVENTORY_S = 30.0
PING_S = 10.0
SILENT_S = 30.0
HUB_SILENCE_S = 30.0
HELLO_TIMEOUT_S = 25.0

MAX_METRICS_TEXT = 512 * 1024  # per engine scrape, after the hub's metric-family filter
MAX_LOG_TEXT = 768 * 1024      # docker.logs result
MAX_FILE_CONTENT = 128 * 1024  # one generated config file in a run spec (at most 4 of them)

# res.error.code values. Hub services map them to HTTP problems; the message is human text from the node.
BAD_ARGS = "bad_args"
UNKNOWN_OP = "unknown_op"
MONITOR_ONLY = "monitor_only"          # the node's policy is in monitor mode
POLICY_DENIED = "policy_denied"        # allowed in principle, but not by this node's policy
NOT_FOUND = "not_found"
NOT_OWNED = "not_owned"                # not a container this console created
CONFLICT = "conflict"
IN_USE = "in_use"                      # e.g. deleting a model a running container mounts
PORT_IN_USE = "port_in_use"
NO_SPACE = "no_space"
HASH_MISMATCH = "hash_mismatch"
SIZE_MISMATCH = "size_mismatch"
IO_ERROR = "io_error"
DOCKER_ERROR = "docker_error"
HTTP_ERROR = "http_error"
TIMEOUT = "timeout"
CANCELLED = "cancelled"
BUSY = "busy"
UNSUPPORTED = "unsupported"
INTERNAL = "internal"

_HDR = struct.Struct(">IB")
_STREAM = struct.Struct(">I")


class ProtoError(ValueError):
    """The peer broke the protocol. Readers raise it; the link is dropped."""


@dataclass(frozen=True)
class Frame:
    kind: int
    msg: dict[str, Any] | None = None   # KIND_JSON
    stream: int = 0                     # KIND_DATA
    data: bytes = b""                   # KIND_DATA


def valid_id(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= MAX_ID


# ── encoding ─────────────────────────────────────────────────────────────────────────────────────────────
def encode_json(obj: dict[str, Any]) -> bytes:
    """One JSON frame. allow_nan=False: a NaN is a bug on the sending side, never something to send."""
    body = json.dumps(obj, allow_nan=False, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(body) > MAX_JSON:
        raise ProtoError(f"message of {len(body)} bytes is over MAX_JSON ({MAX_JSON})")
    return _HDR.pack(len(body) + 1, KIND_JSON) + body


def encode_data(stream: int, payload: bytes) -> bytes:
    if not valid_id(stream):
        raise ProtoError("stream id out of range")
    if len(payload) > MAX_DATA:
        raise ProtoError(f"data payload of {len(payload)} bytes is over MAX_DATA ({MAX_DATA})")
    return _HDR.pack(len(payload) + 5, KIND_DATA) + _STREAM.pack(stream) + payload


# ── decoding ─────────────────────────────────────────────────────────────────────────────────────────────
def _reject_constant(name: str) -> Any:
    raise ProtoError(f"non-finite number {name} in message")


def _finite_float(text: str) -> float:
    v = float(text)                     # "1e999" is not a NaN/Infinity constant, but parses to inf
    if not math.isfinite(v):
        raise ProtoError(f"non-finite number {text[:20]} in message")
    return v


def decode_json(body: bytes) -> dict[str, Any]:
    try:
        obj = json.loads(body.decode("utf-8"), parse_constant=_reject_constant, parse_float=_finite_float)
    except ProtoError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as e:  # RecursionError: deep nesting on 3.12/3.13
        raise ProtoError(f"not JSON: {str(e)[:120]}") from None
    if not isinstance(obj, dict):
        raise ProtoError("message must be a JSON object")
    return obj


def parse_header(hdr: bytes) -> tuple[int, int]:
    """(kind, body length) from the 5 header bytes, or ProtoError."""
    length, kind = _HDR.unpack(hdr)
    body = length - 1
    if kind == KIND_JSON:
        ok = 2 <= body <= MAX_JSON
    elif kind == KIND_DATA:
        ok = 4 <= body <= 4 + MAX_DATA
    else:
        raise ProtoError(f"unknown frame kind {kind}")
    if not ok:
        raise ProtoError(f"frame length {length} out of range for kind {kind}")
    return kind, body


def decode_body(kind: int, body: bytes) -> Frame:
    if kind == KIND_JSON:
        return Frame(KIND_JSON, msg=decode_json(body))
    (stream,) = _STREAM.unpack_from(body)
    if not valid_id(stream):
        raise ProtoError("data frame with an out-of-range stream id")
    return Frame(KIND_DATA, stream=stream, data=bytes(body[4:]))


def _read_exact_sync(f: IO[bytes], n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


def read_frame_sync(f: IO[bytes]) -> Frame | None:
    """Next frame from a blocking binary stream; None on a clean end of stream between frames."""
    hdr = _read_exact_sync(f, 5)
    if not hdr:
        return None
    if len(hdr) < 5:
        raise ProtoError("stream ended inside a frame header")
    kind, n = parse_header(hdr)
    body = _read_exact_sync(f, n)
    if len(body) < n:
        raise ProtoError("stream ended inside a frame")
    return decode_body(kind, body)


async def read_frame_async(r: asyncio.StreamReader) -> Frame | None:
    """Next frame from an asyncio stream; None on a clean end of stream between frames."""
    try:
        hdr = await r.readexactly(5)
    except asyncio.IncompleteReadError as e:
        if not e.partial:
            return None
        raise ProtoError("stream ended inside a frame header") from None
    kind, n = parse_header(hdr)
    try:
        body = await r.readexactly(n)
    except asyncio.IncompleteReadError:
        raise ProtoError("stream ended inside a frame") from None
    return decode_body(kind, body)


def find_magic_sync(f: IO[bytes]) -> int:
    """Skip shell noise up to and including the MAGIC line; returns how many noise bytes were skipped."""
    skipped = 0
    while True:
        line = f.readline(MAX_PREAMBLE + 1)
        if not line:
            raise ProtoError("the far side closed before speaking the ecnode protocol")
        if line.endswith(MAGIC):
            return skipped + len(line) - len(MAGIC)
        skipped += len(line)
        if skipped > MAX_PREAMBLE:
            raise ProtoError("the far side printed something other than the ecnode protocol")


async def find_magic_async(r: asyncio.StreamReader) -> int:
    skipped = 0
    while True:
        try:
            line = await r.readuntil(b"\n")
        except asyncio.IncompleteReadError:
            raise ProtoError("the far side closed before speaking the ecnode protocol") from None
        except asyncio.LimitOverrunError:
            raise ProtoError("the far side printed something other than the ecnode protocol") from None
        if line.endswith(MAGIC):
            return skipped + len(line) - len(MAGIC)
        skipped += len(line)
        if skipped > MAX_PREAMBLE:
            raise ProtoError("the far side printed something other than the ecnode protocol")


# ── message builders (both sides) ──────────────────────────────────────────────────────────────────────────
def req(rid: int, op: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "req", "id": rid, "op": op, "args": args or {}}


def res_ok(rid: int, result: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "res", "id": rid, "ok": True, "result": result or {}}


def res_err(rid: int, code: str, message: str) -> dict[str, Any]:
    return {"type": "res", "id": rid, "ok": False, "error": {"code": code, "message": message[:1000]}}


def event(rid: int, ev: dict[str, Any]) -> dict[str, Any]:
    return {"type": "event", "id": rid, "event": ev}


def credit(rid: int, nbytes: int) -> dict[str, Any]:
    return {"type": "credit", "id": rid, "bytes": nbytes}


def end(rid: int) -> dict[str, Any]:
    return {"type": "end", "id": rid}


def cancel(rid: int) -> dict[str, Any]:
    return {"type": "cancel", "id": rid}


def ping(rid: int) -> dict[str, Any]:
    return {"type": "ping", "id": rid}


def pong(rid: int) -> dict[str, Any]:
    return {"type": "pong", "id": rid}


# ── whitelist cleaning ───────────────────────────────────────────────────────────────────────────────────────
# A spec is INT | NUM | BOOL | JSONV | S(n) | L(spec, cap) | M(spec, cap) | {field: spec}.
# clean() never raises: a wrong type becomes None, a dict spec yields exactly its own keys (missing -> None),
# lists and maps are capped and lose their None members, strings are capped and lose NUL characters.
INT = "int"
NUM = "num"
BOOL = "bool"
JSONV = "json"                  # any JSON value, bounded (depth, items, string length): untrusted data
MAX_NUM = 1e15
JSON_MAX_DEPTH = 8
JSON_MAX_ITEMS = 512
JSON_MAX_NODES = 4000
JSON_MAX_STR = 4096


def S(n: int) -> tuple[str, int]:
    return ("str", n)


def L(spec: Any, cap: int) -> tuple[str, Any, int]:
    return ("list", spec, cap)


def M(spec: Any, cap: int) -> tuple[str, Any, int]:
    return ("map", spec, cap)


def clean_json(v: Any, _depth: int = 0, _budget: list[int] | None = None) -> Any:
    budget = _budget if _budget is not None else [JSON_MAX_NODES]
    budget[0] -= 1
    if budget[0] < 0 or _depth > JSON_MAX_DEPTH:
        return None
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v if abs(v) <= 2**53 else None
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, str):
        return v[:JSON_MAX_STR].replace("\x00", "")
    if isinstance(v, list):
        return [clean_json(x, _depth + 1, budget) for x in v[:JSON_MAX_ITEMS]]
    if isinstance(v, dict):
        out: dict[str, Any] = {}
        for k, x in list(v.items())[:JSON_MAX_ITEMS]:
            if isinstance(k, str) and 0 < len(k) <= 256:
                out[k.replace("\x00", "")] = clean_json(x, _depth + 1, budget)
        return out
    return None


def clean(v: Any, spec: Any) -> Any:
    if v is None:
        return None
    if spec == INT:
        if isinstance(v, bool) or not isinstance(v, int) or abs(v) > 2**53:
            return None
        return v
    if spec == NUM:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        if isinstance(v, int) and abs(v) > 2**53:
            return None
        return v if math.isfinite(v) and abs(v) <= MAX_NUM else None
    if spec == BOOL:
        return v if isinstance(v, bool) else None
    if spec == JSONV:
        return clean_json(v)
    if isinstance(spec, dict):
        if not isinstance(v, dict):
            return None
        return {k: clean(v.get(k), sub) for k, sub in spec.items()}
    kind = spec[0]
    if kind == "str":
        return v[: spec[1]].replace("\x00", "") if isinstance(v, str) else None
    if kind == "list":
        if not isinstance(v, list):
            return None
        items = (clean(x, spec[1]) for x in v[: spec[2]])
        return [x for x in items if x is not None]
    if kind == "map":
        if not isinstance(v, dict):
            return None
        out: dict[str, Any] = {}
        for k, x in list(v.items())[: spec[2]]:
            if isinstance(k, str) and 0 < len(k) <= 256:
                c = clean(x, spec[1])
                if c is not None:
                    out[k.replace("\x00", "")] = c
        return out
    raise AssertionError(f"bad spec {spec!r}")


# ── node -> hub: pushed messages ────────────────────────────────────────────────────────────────────────────
GPU_STATIC = {"index": INT, "uuid": S(96), "name": S(128), "total_bytes": INT, "compute_capability": S(16),
              "unified_memory": BOOL, "pci_bus_id": S(32)}
ROOT = {"id": S(32), "path": S(512), "kind": S(16), "writable": BOOL, "label": S(64)}
SCRIPT_ACTION = {"name": S(32), "label": S(64), "args": L(S(64), 32), "confirm": BOOL, "danger": BOOL}
SCRIPT = {"name": S(32), "label": S(64), "engine": S(32), "description": S(500),
          "actions": L(SCRIPT_ACTION, 16), "containers": L(S(128), 16), "status_action": S(32)}
POLICY = {"path": S(512), "mode": S(16), "error": S(500), "engine_images": L(S(256), 64),
          "gateway_image": S(256), "engine_network": S(64), "port_range": L(INT, 2), "discovery_ports": L(INT, 128),
          "model_roots": L(ROOT, 32), "scripts": L(SCRIPT, 32), "allow_pull": BOOL, "allow_delete": BOOL,
          "allow_ipc_host": BOOL, "external_control": S(16), "max_shm_gib": NUM}
HELLO = {
    "proto": INT, "version": S(32), "node_id": S(64),
    "host": {"hostname": S(253), "arch": S(32), "os": S(128), "kernel": S(128), "cpu_model": S(128),
             "cpu_cores": INT, "cpu_threads": INT, "ram_total_bytes": INT, "boot_time": NUM, "python": S(32)},
    "gpu": {"driver": S(64), "cuda": S(32), "source": S(16), "error": S(500)},
    "gpus": L(GPU_STATIC, 16),
    "docker": {"available": BOOL, "context": S(64), "rootless": BOOL, "version": S(64), "nvidia_runtime": BOOL,
               "error": S(500)},
    "addresses": L({"iface": S(32), "addr": S(64), "prefix": INT}, 64),
    "policy": POLICY,
    "forced_cmd": BOOL,     # sshd set SSH_ORIGINAL_COMMAND: the login was through a restricted key
    "via_ssh": BOOL,
}
GPU_PROC = {"pid": INT, "used_bytes": INT, "container": S(128), "name": S(64)}   # name = /proc/<pid>/comm only
GPU_SAMPLE = {"util_pct": NUM, "mem_used_bytes": INT, "mem_total_bytes": INT, "temp_c": NUM, "power_w": NUM,
              "power_limit_w": NUM, "fan_pct": NUM, "procs": L(GPU_PROC, 32)}
MEM = {"total": INT, "available": INT, "free": INT, "cached": INT, "shmem": INT, "swap_total": INT, "swap_free": INT}
CG_MEM = {"current": INT, "anon": INT, "file": INT, "shmem": INT, "kernel": INT, "source": S(8)}   # bytes
SAMPLE = {"seq": INT, "t": NUM, "gpus": M(GPU_SAMPLE, 16), "mem": MEM,
          "cpu": {"load1": NUM, "load5": NUM, "load15": NUM, "util_pct": NUM},
          "containers": M({"mem": CG_MEM, "cpu_pct": NUM}, 256)}
MOUNT = {"mount": S(512), "fstype": S(32), "source": S(512), "device": S(64), "model": S(128), "size": INT,
         "used": INT, "free": INT, "readonly": BOOL, "remote": BOOL, "remote_host": S(253), "remote_path": S(512),
         "roots": L(S(32), 32)}
DISKS = {"t": NUM, "mounts": L(MOUNT, 64)}
PORT_MAP = {"host_ip": S(64), "host_port": INT, "container_port": INT}
CONTAINER_MOUNT = {"source": S(512), "dest": S(512), "ro": BOOL, "root": S(32), "rel": S(1024)}
CONTAINER = {
    "name": S(128), "id": S(64), "image": S(256), "image_id": S(80), "state": S(16), "status": S(128),
    "created": NUM, "started_at": NUM, "pid": INT, "exit_code": INT, "labels": M(S(256), 48),
    "engine": S(32), "managed": BOOL, "role": S(16), "instance": S(64), "ports": L(PORT_MAP, 20),
    "argv": L(S(1024), 400),             # REDACTED by the node before it leaves the machine
    "params": M(S(200), 60),             # display-only `--flag value` parse of the redacted argv
    "gpus": L(S(96), 16), "all_gpus": BOOL, "mounts": L(CONTAINER_MOUNT, 32), "network_mode": S(64),
    "compose_project": S(128), "compose_service": S(128)}
ENDPOINT = {"port": INT, "container": S(128), "engine": S(32), "models": L(S(200), 20), "health_status": INT,
            "models_status": INT, "metrics_status": INT, "auth_required": BOOL}
SCRIPT_STATUS = {"name": S(32), "status": S(16), "detail": S(500), "checked_at": NUM}
INVENTORY = {"t": NUM, "containers": L(CONTAINER, 256), "endpoints": L(ENDPOINT, 64),
             "scripts": L(SCRIPT_STATUS, 32)}
METRICS = {"t": NUM, "port": INT, "container": S(128), "engine": S(32), "status": INT,
           "text": S(MAX_METRICS_TEXT)}
MODELS_CHANGED = {"generation": INT, "t": NUM}
LOG = {"level": S(8), "msg": S(1000)}
FATAL = {"error": S(1000)}
PONG = {"id": INT}
CREDIT = {"id": INT, "bytes": INT}
ERROR = {"code": S(64), "message": S(1000)}

PUSH: dict[str, Any] = {"hello": HELLO, "sample": SAMPLE, "disks": DISKS, "inventory": INVENTORY, "metrics": METRICS,
                        "models_changed": MODELS_CHANGED, "log": LOG, "fatal": FATAL, "pong": PONG, "credit": CREDIT}

# ── hub -> node: request arguments ──────────────────────────────────────────────────────────────────────────
RUN_MOUNT = {"root": S(32), "rel": S(1024), "dest": S(512), "ro": BOOL}   # host path = a policy model root + rel
RUN_VOLUME = {"name": S(64), "dest": S(512)}                               # named volumes must be called ec-*
RUN_SPEC = {
    "name": S(64), "instance": S(64), "image": S(256), "argv": L(S(4096), 512), "container_port": INT,
    "gpu_uuids": L(S(96), 16), "network_alias": S(64), "labels": M(S(256), 32), "env": M(S(4096), 128),
    "secret_env": M(S(4096), 16),        # values ride the docker child's environment, never its argv
    "mounts": L(RUN_MOUNT, 16), "volumes": L(RUN_VOLUME, 8), "shm_size": S(16), "ipc_host": BOOL,
    "files": M(S(MAX_FILE_CONTENT), 4),  # container path -> content, written by the node and mounted read-only
    "ulimits": M(S(32), 4),              # additive (v1.1): name -> "N", "-1" or "soft:hard"; the node allowlists names
}
ARGS: dict[str, Any] = {
    "node.info": {},
    "metrics.configure": {"interval_s": NUM, "families": M(L(S(64), 32), 16)},
    "docker.render": {"spec": RUN_SPEC},
    "docker.run": {"spec": RUN_SPEC},
    "docker.stop": {"name": S(128), "timeout_s": INT, "role": S(16), "instance": S(64)},
    "docker.start": {"name": S(128), "role": S(16), "instance": S(64)},
    "docker.rm": {"name": S(128), "role": S(16), "instance": S(64)},
    "docker.inspect": {"name": S(128)},
    "docker.logs": {"name": S(128), "tail": INT},
    "docker.logs_follow": {"name": S(128), "tail": INT},
    "docker.image_present": {"image": S(256)},
    "docker.images": {},
    "docker.pull": {"image": S(256)},
    "docker.external": {"name": S(128), "action": S(16)},
    "net.ensure_internal": {},
    "gateway.start": {"name": S(128), "instance": S(64), "host_port": INT, "engine_host": S(64), "engine_port": INT},
    "http.request": {"port": INT, "method": S(8), "path": S(2048), "headers": M(S(4096), 32), "body_len": INT,
                     "timeout_s": NUM},
    "fs.stat": {"root": S(32), "paths": L(S(1024), 256)},
    "fs.write": {"root": S(32), "rel": S(1024), "size": INT, "sha256": S(64), "offset": INT},
    "fs.read": {"root": S(32), "rel": S(1024), "offset": INT, "length": INT},
    "fs.symlink": {"root": S(32), "link": S(1024), "target": S(1024)},
    "fs.put_small": {"root": S(32), "rel": S(1024), "content_b64": S(128 * 1024)},
    "fs.delete": {"root": S(32), "rel": S(1024)},
    "models.scan": {"roots": L(S(32), 32), "force": BOOL},
    "models.list": {"cursor": INT, "limit": INT},
    "models.config": {"root": S(32), "rel": S(1024)},
    "scripts.run": {"name": S(32), "action": S(32), "arg": S(64)},
}
REQUIRED: dict[str, tuple[str, ...]] = {
    "metrics.configure": ("families",), "docker.render": ("spec",), "docker.run": ("spec",),
    "docker.stop": ("name",), "docker.start": ("name",), "docker.rm": ("name",), "docker.inspect": ("name",),
    "docker.logs": ("name",), "docker.logs_follow": ("name",), "docker.image_present": ("image",),
    "docker.pull": ("image",), "docker.external": ("name", "action"),
    "gateway.start": ("name", "instance", "host_port", "engine_host", "engine_port"),
    "http.request": ("port", "method", "path"), "fs.stat": ("root", "paths"),
    "fs.write": ("root", "rel", "size"), "fs.read": ("root", "rel"), "fs.symlink": ("root", "link", "target"),
    "fs.put_small": ("root", "rel", "content_b64"), "fs.delete": ("root", "rel"), "models.config": ("root", "rel"),
    "scripts.run": ("name", "action"),
}
# Allowed while the node's policy says mode "monitor": nothing here changes the machine.
MONITOR_OPS = frozenset({"node.info", "metrics.configure", "docker.render", "docker.inspect", "docker.logs",
                         "docker.logs_follow", "docker.image_present", "docker.images", "http.request", "fs.stat",
                         "fs.read", "models.scan", "models.list", "models.config"})
OPS = frozenset(ARGS)
UPLOAD_OPS = frozenset({"fs.write", "http.request"})   # may receive DATA from the hub (http only if body_len > 0)

# ── node -> hub: results and events per op ──────────────────────────────────────────────────────────────────
IMAGE = {"ref": S(256), "id": S(80), "size": INT, "created": NUM, "arch": S(16)}
MODEL_ENTRY = {"root": S(32), "rel": S(1024), "format": S(16), "name": S(256), "revision": S(64), "size": INT,
               "files": INT, "mtime": NUM, "atime": NUM, "quant": S(32), "params": INT, "arch": S(64),
               "context": INT, "shards": INT, "meta": JSONV, "warnings": L(S(300), 8)}
ROOT_SCAN = {"root": S(32), "scanned_at": NUM, "complete": BOOL, "entries": INT, "error": S(500), "duration_s": NUM}
RESULTS: dict[str, Any] = {
    "node.info": HELLO,
    "metrics.configure": {},
    "docker.render": {"argv": L(S(4096), 1024)},
    "docker.run": {"id": S(64)},
    "docker.stop": {}, "docker.start": {}, "docker.rm": {},
    "docker.inspect": {"info": CONTAINER},
    "docker.logs": {"text": S(MAX_LOG_TEXT), "truncated": BOOL},
    "docker.logs_follow": {"lines": INT},
    "docker.image_present": {"present": BOOL, "id": S(80)},
    "docker.images": {"images": L(IMAGE, 512)},
    "docker.pull": {"id": S(80)},
    "docker.external": {},
    "net.ensure_internal": {"network": S(64)},
    "gateway.start": {"id": S(64)},
    "http.request": {"status": INT, "bytes": INT},
    "fs.stat": {"items": L({"rel": S(1024), "exists": BOOL, "is_dir": BOOL, "size": INT, "partial": INT}, 256)},
    "fs.write": {"size": INT, "sha256": S(64)},
    "fs.read": {"size": INT},
    "fs.symlink": {}, "fs.put_small": {}, "fs.delete": {"freed": INT},
    "models.scan": {"generation": INT},
    "models.list": {"generation": INT, "entries": L(MODEL_ENTRY, 200), "next_cursor": INT, "roots": L(ROOT_SCAN, 32)},
    "models.config": {"config": JSONV},
    "scripts.run": {"rc": INT},
}
EVENTS: dict[str, Any] = {
    "http.request": {"status": INT, "headers": M(S(4096), 64)},   # exactly one, before any DATA
    "docker.logs_follow": {"line": S(8192)},
    "docker.pull": {"line": S(1024)},
    "scripts.run": {"stream": S(8), "line": S(8192)},              # stream: "out" | "err"
}


# ── sanitizers (hub side) and argument cleaning (node side) ────────────────────────────────────────────────
def sanitize_push(msg: dict[str, Any]) -> dict[str, Any] | None:
    """A pushed node message rebuilt from its whitelist; None for a type the hub does not know (ignore it)."""
    t = msg.get("type")
    spec = PUSH.get(t) if isinstance(t, str) else None
    if spec is None:
        return None
    out: dict[str, Any] = clean(msg, spec)
    out["type"] = t
    if t == "hello":
        if out["proto"] is None:
            raise ProtoError("hello carries no protocol version")
        out["gpus"] = [g for g in out["gpus"] or [] if g.get("uuid")]
    return out


def sanitize_res(msg: dict[str, Any], op: str) -> dict[str, Any]:
    """{"id", "ok", "result"|"error"} for the response to `op` (the hub knows the op from its own request)."""
    out: dict[str, Any] = {"id": clean(msg.get("id"), INT), "ok": msg.get("ok") is True}
    if out["ok"]:
        out["result"] = clean(msg.get("result"), RESULTS.get(op, {})) or {}
    else:
        err = clean(msg.get("error"), ERROR) or {}
        out["error"] = {"code": err.get("code") or INTERNAL,
                        "message": err.get("message") or "the node refused without saying why"}
    return out


def sanitize_event(msg: dict[str, Any], op: str) -> dict[str, Any]:
    return {"id": clean(msg.get("id"), INT), "event": clean(msg.get("event"), EVENTS.get(op, {})) or {}}


def clean_args(op: str, args: Any) -> dict[str, Any]:
    """Node side: the whitelisted arguments of a request (wrong types -> None); ProtoError for an unknown op."""
    spec = ARGS.get(op)
    if spec is None:
        raise ProtoError(f"unknown op {op!r}")
    return clean(args if isinstance(args, dict) else {}, spec) or {}


def missing_args(op: str, cleaned: dict[str, Any]) -> list[str]:
    return [k for k in REQUIRED.get(op, ()) if cleaned.get(k) in (None, "", [])]
