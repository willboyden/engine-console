"""The locked wire contract: framing, preamble, limits and the whitelist cleaners."""
from __future__ import annotations

import asyncio
import io
import json
import struct

import pytest

from ecnode import proto
from ecnode.proto import ProtoError


# ── framing ──────────────────────────────────────────────────────────────────────────────────────────────
def test_json_frame_round_trip_sync():
    msg = {"type": "req", "id": 7, "op": "docker.images", "args": {}, "unicode": "Grüße ✓"}
    f = io.BytesIO(proto.encode_json(msg))
    frame = proto.read_frame_sync(f)
    assert frame is not None and frame.kind == proto.KIND_JSON and frame.msg == msg
    assert proto.read_frame_sync(f) is None          # clean end between frames


def test_data_frame_round_trip_and_binary_payload():
    payload = bytes(range(256)) * 10
    f = io.BytesIO(proto.encode_data(42, payload))
    frame = proto.read_frame_sync(f)
    assert frame is not None and frame.kind == proto.KIND_DATA and frame.stream == 42 and frame.data == payload


def test_async_reader_reads_the_same_frames():
    blob = proto.MAGIC + proto.encode_json({"type": "hello", "proto": 1}) + proto.encode_data(3, b"abc")

    async def go():
        r = asyncio.StreamReader()
        r.feed_data(b"Welcome to the box\nmotd line\n" + blob)
        r.feed_eof()
        skipped = await proto.find_magic_async(r)
        a = await proto.read_frame_async(r)
        b = await proto.read_frame_async(r)
        c = await proto.read_frame_async(r)
        return skipped, a, b, c

    skipped, a, b, c = asyncio.run(go())
    assert skipped == len(b"Welcome to the box\nmotd line\n")
    assert a is not None and a.msg == {"type": "hello", "proto": 1}
    assert b is not None and (b.stream, b.data) == (3, b"abc")
    assert c is None


def test_magic_after_noise_without_trailing_newline():
    f = io.BytesIO(b"no newline here" + proto.MAGIC + proto.encode_json({"type": "x"}))
    assert proto.find_magic_sync(f) == len(b"no newline here")
    assert proto.read_frame_sync(f).msg == {"type": "x"}


def test_preamble_limit_and_early_close():
    with pytest.raises(ProtoError):
        proto.find_magic_sync(io.BytesIO(b"x" * 100 + b"\n"))           # closed before the magic
    noise = (b"y" * 1000 + b"\n") * 70                                  # > 64 KiB of noise
    with pytest.raises(ProtoError):
        proto.find_magic_sync(io.BytesIO(noise + proto.MAGIC))

    async def go():
        r = asyncio.StreamReader()
        r.feed_data(noise + proto.MAGIC)
        r.feed_eof()
        await proto.find_magic_async(r)

    with pytest.raises(ProtoError):
        asyncio.run(go())


@pytest.mark.parametrize("hdr", [
    struct.pack(">IB", 1, proto.KIND_JSON),                         # empty JSON body
    struct.pack(">IB", proto.MAX_JSON + 2, proto.KIND_JSON),        # over MAX_JSON
    struct.pack(">IB", 3, proto.KIND_DATA),                         # data frame without a stream id
    struct.pack(">IB", proto.MAX_DATA + 6, proto.KIND_DATA),        # over MAX_DATA
    struct.pack(">IB", 10, 9),                                      # unknown kind
])
def test_header_limits(hdr):
    with pytest.raises(ProtoError):
        proto.parse_header(hdr)


def test_truncated_frames_are_errors_not_eof():
    full = proto.encode_json({"type": "x", "pad": "z" * 50})
    with pytest.raises(ProtoError):
        proto.read_frame_sync(io.BytesIO(full[:3]))
    with pytest.raises(ProtoError):
        proto.read_frame_sync(io.BytesIO(full[:-1]))


def test_encoders_refuse_what_the_reader_would_refuse():
    with pytest.raises(ProtoError):
        proto.encode_data(0, b"x")
    with pytest.raises(ProtoError):
        proto.encode_data(1, b"x" * (proto.MAX_DATA + 1))
    with pytest.raises(ProtoError):
        proto.encode_json({"big": "x" * (proto.MAX_JSON + 1)})
    with pytest.raises(ValueError):
        proto.encode_json({"n": float("nan")})


@pytest.mark.parametrize("body", [
    b'{"n": NaN}', b'{"n": Infinity}', b'{"n": 1e999}', b"[1, 2]", b"\xff\xfe", b"{" * 20000,
])
def test_decode_json_rejects_hostile_bodies(body):
    with pytest.raises(ProtoError):
        proto.decode_json(body)


def test_data_frame_with_zero_stream_id_is_rejected():
    raw = struct.pack(">IB", 5 + 3, proto.KIND_DATA) + struct.pack(">I", 0) + b"abc"
    with pytest.raises(ProtoError):
        proto.read_frame_sync(io.BytesIO(raw))


# ── cleaning ─────────────────────────────────────────────────────────────────────────────────────────────
def test_clean_types_caps_and_nul_stripping():
    spec = {"n": proto.INT, "x": proto.NUM, "b": proto.BOOL, "s": proto.S(5), "l": proto.L(proto.INT, 3),
            "m": proto.M(proto.S(3), 2)}
    got = proto.clean({"n": True, "x": 2**60, "b": 1, "s": "ab\x00cdefgh", "l": [1, "2", 3, 4, 5],
                       "m": {"a": "xyzw", "b": 5, "c": "q"}, "extra": "dropped"}, spec)
    assert got == {"n": None, "x": None, "b": None, "s": "abcd", "l": [1, 3], "m": {"a": "xyz"}}


def test_clean_json_is_bounded():
    deep: object = "leaf"
    for _ in range(50):
        deep = [deep]
    out = proto.clean_json({"a": deep, "f": float("inf"), "big": 2**60, 5: "non-str key"})
    assert out["f"] is None and out["big"] is None and 5 not in out
    depth, node = 0, out["a"]
    while isinstance(node, list) and node:
        node, depth = node[0], depth + 1
    assert depth <= proto.JSON_MAX_DEPTH


def test_sanitize_push_whitelists_and_rejects_hello_without_proto():
    s = proto.sanitize_push({"type": "sample", "seq": 1, "t": 1.5, "gpus": {"GPU-1": {"util_pct": 50, "evil": 1}},
                             "injected": "<script>"})
    assert s is not None and s["type"] == "sample" and "injected" not in s
    assert s["gpus"]["GPU-1"]["util_pct"] == 50 and "evil" not in s["gpus"]["GPU-1"]
    assert proto.sanitize_push({"type": "who-knows"}) is None
    with pytest.raises(ProtoError):
        proto.sanitize_push({"type": "hello"})
    h = proto.sanitize_push({"type": "hello", "proto": 1, "gpus": [{"uuid": "GPU-a"}, {"name": "no uuid"}]})
    assert [g["uuid"] for g in h["gpus"]] == ["GPU-a"]


def test_sanitize_res_ok_and_error_shapes():
    ok = proto.sanitize_res({"type": "res", "id": 3, "ok": True, "result": {"present": True, "x": 1}},
                            "docker.image_present")
    assert ok == {"id": 3, "ok": True, "result": {"present": True, "id": None}}
    bad = proto.sanitize_res({"type": "res", "id": 4, "ok": "yes"}, "docker.run")
    assert bad["ok"] is False and bad["error"]["code"] == proto.INTERNAL and bad["error"]["message"]


def test_clean_args_and_required():
    a = proto.clean_args("fs.write", {"root": "hf", "rel": "x", "size": "12", "evil": True})
    assert a == {"root": "hf", "rel": "x", "size": None, "sha256": None, "offset": None}
    assert proto.missing_args("fs.write", a) == ["size"]
    with pytest.raises(ProtoError):
        proto.clean_args("shell.exec", {"cmd": "id"})


def test_every_op_has_args_and_result_specs_and_monitor_ops_exist():
    assert set(proto.RESULTS) == set(proto.ARGS)
    assert proto.MONITOR_OPS <= proto.OPS
    assert set(proto.EVENTS) <= proto.OPS
    assert set(proto.REQUIRED) <= proto.OPS
    assert proto.UPLOAD_OPS <= proto.OPS
    for op in ("docker.run", "fs.write", "fs.delete", "scripts.run", "docker.pull", "gateway.start"):
        assert op not in proto.MONITOR_OPS, op


def test_builders_produce_frames_the_reader_accepts():
    msgs = [proto.req(1, "node.info"), proto.res_ok(1, {}), proto.res_err(2, proto.BUSY, "x" * 5000),
            proto.event(3, {"line": "hi"}), proto.credit(3, proto.WINDOW), proto.end(3), proto.cancel(3),
            proto.ping(9), proto.pong(9)]
    f = io.BytesIO(b"".join(proto.encode_json(m) for m in msgs))
    got = []
    while (fr := proto.read_frame_sync(f)) is not None:
        got.append(fr.msg)
    assert got[2]["error"]["message"] == "x" * 1000
    assert [m["type"] for m in got] == ["req", "res", "res", "event", "credit", "end", "cancel", "ping", "pong"]
    json.dumps(got)  # all plain JSON
