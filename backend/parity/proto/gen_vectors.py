"""Write backend/parity/proto/vectors.json: input -> output of the Python wire module (ecnode.proto) for a corpus of
hostile and edge inputs. crates/ec-proto/tests/test_vectors.rs asserts the Rust port gives the same output for every
vector (JSON values compared with their int/float kind).

    "$(node backend/parity/lib/python-release.mjs)/node/.venv/bin/python" -I backend/parity/proto/gen_vectors.py

Inputs are built as JSON values (they reach the cleaners from json.loads in production), so every vector is a value
both peers can hold. A vector whose input cannot be represented identically in both languages is listed in
DEVIATIONS below instead of being silently dropped.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ecnode import proto

OUT = Path(__file__).resolve().parent / "vectors.json"
P53 = 2**53

SCALARS: list[Any] = [
    None, True, False, 0, 1, -1, 7, P53, P53 + 1, -P53, -P53 - 1, 2**60, 10**15, 10**15 + 1, -(10**15),
    0.0, -0.0, 1.0, 1.5, -2.5, 1e15, 1e15 + 2.0, 1e16, 1e-300, 5e-324, 1.7976931348623157e308, 3.141592653589793,
    "", "a", "abcdefgh", "ab\x00cdefgh", "\x00\x00", "Grüße ✓", "日本語のテキスト", "e\u0301e\u0301e\u0301e\u0301",
    "🎉🎉🎉🎉🎉🎉", "x" * 70, [], [1, "2", 3], {}, {"a": 1},
]
SCALAR_SPECS = {
    "INT": proto.INT, "NUM": proto.NUM, "BOOL": proto.BOOL, "JSONV": proto.JSONV, "S5": proto.S(5), "S0": proto.S(0),
    "L_INT_3": proto.L(proto.INT, 3), "L_S2_4": proto.L(proto.S(2), 4), "M_S3_2": proto.M(proto.S(3), 2),
    "M_INT_3": proto.M(proto.INT, 3),
}
LIST_INPUTS: list[Any] = [
    [1, "2", 3, 4, 5], [None, 1, None, 2], [True, 1, 1.0, 2], ["abc", "de", "f", "\x00x", "toolong"], [[1], {"a": 1}],
]
MAP_INPUTS: list[Any] = [
    {"a": "xyzw", "b": 5, "c": "q"}, {"": "empty key", "k": "v"}, {"k" * 257: "long key", "ok": "v"},
    {"a\x00": "nul key", "b": "x"}, {"a": "first", "a\x00": "same key after NUL removal"}, {"n": None, "m": "x"},
    {"x": 1, "y": 2.0, "z": "3", "w": P53 + 1},
]


def deep(n: int) -> Any:
    v: Any = "leaf"
    for _ in range(n):
        v = [v]
    return v


JSONV_INPUTS: list[Any] = [
    deep(5), deep(8), deep(9), deep(50), {"a": deep(10)}, list(range(600)), {f"k{i}": i for i in range(600)},
    [list(range(100)) for _ in range(50)],                    # 5000 nodes: the node budget cuts in
    {"s": "x" * 5000, "n": P53 + 1, "f": 1.5, "b": True, "z": None, "": 1, "k" * 257: 2, "nul\x00": "v\x00"},
    [{"a": [{"b": [{"c": [{"d": [{"e": [{"f": "g"}]}]}]}]}]}],
]
OPS = sorted(proto.OPS)


def v(kind: str, **kw: Any) -> dict[str, Any]:
    return {"kind": kind, **kw}


def vectors() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, spec in SCALAR_SPECS.items():
        for x in [*SCALARS, *LIST_INPUTS, *MAP_INPUTS]:
            out.append(v("clean", spec=name, input=x, output=proto.clean(x, spec)))
    for x in JSONV_INPUTS:
        out.append(v("clean_json", input=x, output=proto.clean_json(x)))
    # every pushed type: a well-formed message, one with junk, and one with every field of the wrong type
    junk = {"evil": "<script>", "type": "x", "id": "1"}
    pushes: list[Any] = [
        {"type": "hello", "proto": 1, "version": "0.4.0", "node_id": "abc", "host": {"hostname": "h", "cpu_cores": 4.0,
         "boot_time": 12, "extra": 1}, "gpus": [{"uuid": "GPU-1", "index": 0, "total_bytes": 2**40},
                                               {"name": "no uuid"}, {"uuid": ""}, "junk"],
         "policy": {"mode": "manage", "port_range": [1, 2, 3], "max_shm_gib": 64, "model_roots": [{"id": "r", "x": 1}]},
         "addresses": [{"iface": "eth0", "addr": "192.0.2.1", "prefix": 24}, None], "forced_cmd": "yes"},
        {"type": "hello"}, {"type": "hello", "proto": "1"}, {"type": "hello", "proto": 1.0}, {"type": "hello", "proto": 2},
        {"type": "sample", "seq": 1, "t": 1.5, "gpus": {"GPU-1": {"util_pct": 50, "evil": 1, "procs": [
            {"pid": 1, "used_bytes": None, "container": "c" * 200, "name": "n"}]}}, "injected": "<script>",
         "mem": {"total": 1.0, "available": 5}, "cpu": {"load1": "x"}, "containers": {"c": {"mem": {"source": "cgroupv2"}}}},
        {"type": "disks", "t": 1, "mounts": [{"mount": "/", "size": -1, "roots": ["a", 5, None]}]},
        {"type": "inventory", "t": 1, "containers": [{"name": "n", "labels": {"k": "v" * 300, "n": 1}, "argv": ["a"] * 500,
                                                      "ports": [{"host_port": 8000.0}]}], "endpoints": [], "scripts": None},
        {"type": "metrics", "t": 1, "port": 8000, "text": "x" * 10, "status": True},
        {"type": "models_changed", "generation": 3, "t": 2},
        {"type": "log", "level": "warning-long", "msg": "m" * 2000},
        {"type": "fatal", "error": None}, {"type": "pong", "id": 7}, {"type": "pong", "id": True},
        {"type": "credit", "id": 1, "bytes": 1.0}, {"type": "who-knows"}, {"type": 5}, {"no": "type"},
        {**junk, "type": "sample"},
    ]
    for m in pushes:
        try:
            out.append(v("sanitize_push", input=m, output=proto.sanitize_push(m)))
        except proto.ProtoError:
            out.append(v("sanitize_push", input=m, error=True))
    reses: list[tuple[Any, str]] = [
        ({"type": "res", "id": 3, "ok": True, "result": {"present": True, "x": 1}}, "docker.image_present"),
        ({"type": "res", "id": 4, "ok": "yes"}, "docker.run"),
        ({"type": "res", "id": 4, "ok": False, "error": {"code": "", "message": ""}}, "fs.read"),
        ({"type": "res", "id": 4, "ok": False, "error": {"code": "x" * 100, "message": "m" * 2000}}, "fs.read"),
        ({"type": "res", "id": 4, "ok": False, "error": "flat"}, "fs.read"),
        ({"type": "res", "id": "4", "ok": True, "result": None}, "node.info"),
        ({"type": "res", "id": 5, "ok": True, "result": [1]}, "fs.stat"),
        ({"type": "res", "id": 6, "ok": True, "result": {"items": [{"rel": "a", "exists": 1, "size": 2.0}, "x"]}}, "fs.stat"),
        ({"type": "res", "id": 7, "ok": True, "result": {"x": 1}}, "unknown.op"),
        ({"type": "res", "id": 8, "ok": True, "result": {"config": deep(12)}}, "models.config"),
        ({"type": "res", "id": 9, "ok": True, "result": {"entries": [{"meta": {"a": [1, {"b": 2}]}, "size": 2**54}],
                                                          "next_cursor": None}}, "models.list"),
    ]
    for m, op in reses:
        out.append(v("sanitize_res", input=m, op=op, output=proto.sanitize_res(m, op)))
    for m, op in [({"id": 1, "event": {"status": 200, "headers": {"a": "b", "n": 1}}}, "http.request"),
                  ({"id": 1, "event": {"line": "l" * 9000}}, "docker.logs_follow"),
                  ({"id": 1, "event": "x"}, "docker.pull"), ({"id": 1, "event": {"stream": "out", "x": 1}}, "scripts.run"),
                  ({"id": 1, "event": {"a": 1}}, "fs.read")]:
        out.append(v("sanitize_event", input=m, op=op, output=proto.sanitize_event(m, op)))
    args_inputs: list[Any] = [{}, None, [1], "x", {"evil": True}]
    for op in OPS:
        for a in args_inputs:
            out.append(v("clean_args", op=op, input=a, output=proto.clean_args(op, a)))
    full = {"spec": {"name": "ec-x", "argv": ["a", 1, None], "labels": {"k": 1}, "secret_env": {"API_KEY": "s"},
                     "mounts": [{"root": "r", "ro": "true"}], "files": {"/a": "x" * 10}, "ulimits": {"nofile": 5}},
            "name": "n" * 200, "tail": 5.0, "timeout_s": "30", "port": True, "paths": ["a", 1, "b"], "size": 2**53,
            "offset": -1, "families": {"vllm": ["vllm:", 5, ""]}, "interval_s": 1e16, "roots": "x", "force": 0,
            "headers": {"a": "b"}, "body_len": 10, "content_b64": "YQ==", "link": "l", "target": "t", "rel": "r",
            "root": "w", "image": "i", "action": "stop", "role": "engine", "instance": "i", "host_port": 18000,
            "engine_host": "h", "engine_port": 8000, "method": "GET", "path": "/v1/models", "sha256": "0" * 64,
            "length": 3, "cursor": 0, "limit": 50, "arg": "a"}
    for op in OPS:
        out.append(v("clean_args", op=op, input=full, output=proto.clean_args(op, full)))
    for op in ["shell.exec", "", "node.info "]:
        out.append(v("clean_args", op=op, input={}, error=True))
    for op in OPS:
        for cleaned in [{}, {k: "" for k in proto.REQUIRED.get(op, ())}, {k: [] for k in proto.REQUIRED.get(op, ())},
                        {k: 0 for k in proto.REQUIRED.get(op, ())}, {k: False for k in proto.REQUIRED.get(op, ())}]:
            out.append(v("missing_args", op=op, input=cleaned, output=proto.missing_args(op, cleaned)))
    return out


# Inputs the two languages cannot hold identically, so they are not vectors (backend/PORTING.md lists them):
DEVIATIONS = [
    "integers beyond the 64-bit range (e.g. 2**64 + 5): Python keeps them as ints (clean INT/NUM/JSONV -> null); "
    "serde_json reads them as floats, so INT -> null and NUM -> null as well, but JSONV keeps the float",
    "lone UTF-16 surrogates in strings (\"\\ud800\"): Python's json accepts them, serde_json refuses the frame",
    "nesting deeper than serde_json's recursion limit (128) but below Python's (~1000): Python accepts the frame",
]


def main() -> int:
    vs = vectors()
    # round-trip check: every vector must survive JSON exactly as Python sees it
    text = json.dumps({"generated_by": "backend/parity/proto/gen_vectors.py", "count": len(vs), "deviations": DEVIATIONS,
                       "vectors": vs}, ensure_ascii=False, indent=0)
    back = json.loads(text)["vectors"]
    assert back == vs, "a vector does not survive JSON"
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(vs)} vectors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
