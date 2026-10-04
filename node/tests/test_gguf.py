"""The bounded GGUF header reader: real-shaped headers, every truncation, hostile counts and lengths."""
from __future__ import annotations

import io
import random
import struct
import tracemalloc

import pytest

from ecnode import gguf

U32, STR, ARR, U64, F32, BOOL = 4, 8, 9, 10, 6, 7


def s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def kv(key: str, vtype: int, payload: bytes) -> bytes:
    return s(key) + struct.pack("<I", vtype) + payload


def build(arch: str = "llama", *, tensors: list[tuple[str, list[int]]] | None = None, extra: list[bytes] = (),
          version: int = 3, file_type: int = 15) -> bytes:
    tensors = tensors if tensors is not None else [("tok_embd", [4096, 32000]), ("blk.0.attn_q", [4096, 4096])]
    kvs = [
        kv("general.architecture", STR, s(arch)),
        kv("general.name", STR, s("Test Model 8B")),
        kv("general.file_type", U32, struct.pack("<I", file_type)),
        kv(f"{arch}.context_length", U32, struct.pack("<I", 131072)),
        kv(f"{arch}.block_count", U32, struct.pack("<I", 32)),
        kv(f"{arch}.embedding_length", U32, struct.pack("<I", 4096)),
        kv(f"{arch}.attention.head_count", U32, struct.pack("<I", 32)),
        kv(f"{arch}.attention.head_count_kv", U32, struct.pack("<I", 8)),
        kv("tokenizer.ggml.tokens", ARR, struct.pack("<IQ", STR, 3) + s("a") + s("b") + s("c")),
        kv("tokenizer.ggml.scores", ARR, struct.pack("<IQ", F32, 3) + struct.pack("<3f", 0.0, 1.0, 2.0)),
        kv("general.flag", BOOL, b"\x01"),
        *extra,
    ]
    head = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(kvs)) + b"".join(kvs)
    for name, dims in tensors:
        head += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        head += struct.pack("<IQ", 12, 0)
    return head


def test_reads_the_facts_a_library_needs():
    h = gguf.read_header(io.BytesIO(build()))
    assert h is not None and h["complete"]
    assert (h["arch"], h["name"], h["quant"], h["context"], h["block_count"]) == \
        ("llama", "Test Model 8B", "Q4_K_M", 131072, 32)
    assert (h["head_count"], h["head_count_kv"], h["embedding_length"]) == (32, 8, 4096)
    assert h["params"] == 4096 * 32000 + 4096 * 4096 and h["tensors"] == 2


def test_moe_keys_and_other_versions():
    extra = [kv("qwen3moe.expert_count", U32, struct.pack("<I", 128)),
             kv("qwen3moe.expert_used_count", U32, struct.pack("<I", 8))]
    h = gguf.read_header(io.BytesIO(build("qwen3moe", extra=extra, file_type=7)))
    assert h["expert_count"] == 128 and h["expert_used_count"] == 8 and h["quant"] == "Q8_0"
    assert gguf.read_header(io.BytesIO(b"NOPE" + b"\x00" * 40)) is None
    assert gguf.read_header(io.BytesIO(b"GGUF" + struct.pack("<I", 99) + b"\x00" * 40)) == {"version": 99,
                                                                                             "complete": False}


def test_every_truncation_is_handled_without_raising():
    blob = build()
    for n in range(len(blob)):
        h = gguf.read_header(io.BytesIO(blob[:n]))
        assert h is None or h["complete"] is False or n == len(blob)


def _hostile_headers() -> list[bytes]:
    huge = 2**63
    base = b"GGUF" + struct.pack("<I", 3)
    return [
        base + struct.pack("<QQ", 1, huge),                                  # absurd KV count
        base + struct.pack("<QQ", huge, 0),                                  # absurd tensor count
        base + struct.pack("<QQ", 0, 1) + struct.pack("<Q", huge),           # absurd key length
        base + struct.pack("<QQ", 0, 1) + s("k") + struct.pack("<I", STR) + struct.pack("<Q", huge),   # string len
        base + struct.pack("<QQ", 0, 1) + s("k") + struct.pack("<I", ARR) + struct.pack("<IQ", U64, huge),
        base + struct.pack("<QQ", 0, 1) + s("k") + struct.pack("<I", ARR) + struct.pack("<IQ", STR, huge),
        base + struct.pack("<QQ", 0, 1) + s("k") + struct.pack("<I", ARR) + struct.pack("<IQ", ARR, huge),
        base + struct.pack("<QQ", 0, 1) + s("k") + struct.pack("<I", 77),    # unknown value type
        base + struct.pack("<QQ", 1, 0) + s("t") + struct.pack("<I", 2**31) + b"\x00" * 64,   # absurd n_dims
        base + struct.pack("<QQ", 1, 0) + s("t") + struct.pack("<I", 8) + struct.pack("<8Q", *[huge] * 8)
        + struct.pack("<IQ", 0, 0),                                          # huge dims: params None, no crash
        # nested arrays deeper than the reader follows
        base + struct.pack("<QQ", 0, 1) + s("k") + struct.pack("<I", ARR)
        + struct.pack("<IQ", ARR, 1) + struct.pack("<IQ", ARR, 1) + struct.pack("<IQ", ARR, 1)
        + struct.pack("<IQ", ARR, 1) + struct.pack("<IQ", U32, 1) + b"\x00" * 4,
    ]


@pytest.mark.parametrize("blob", _hostile_headers())
def test_hostile_headers_never_raise_or_allocate_much(blob):
    tracemalloc.start()
    try:
        h = gguf.read_header(io.BytesIO(blob))
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert h is None or isinstance(h, dict)
    assert peak < 8 * 1024 * 1024


def test_random_bytes_after_the_magic():
    rng = random.Random(1234)
    for _ in range(300):
        blob = b"GGUF" + struct.pack("<I", rng.choice((1, 2, 3))) + bytes(rng.getrandbits(8)
                                                                          for _ in range(rng.randint(0, 400)))
        h = gguf.read_header(io.BytesIO(blob), budget=1 << 20)
        assert h is None or isinstance(h, dict)


def test_budget_stops_a_header_bigger_than_allowed():
    big = [kv(f"pad.{i}", STR, s("x" * 1000)) for i in range(100)]
    h = gguf.read_header(io.BytesIO(build(extra=big)), budget=20_000)
    assert h is not None and h["complete"] is False and h["arch"] == "llama"   # what was read is kept


def test_version_1_uses_32_bit_counts():
    def s1(text: str) -> bytes:
        return struct.pack("<I", len(text)) + text.encode()
    blob = (b"GGUF" + struct.pack("<III", 1, 0, 2) + s1("general.architecture") + struct.pack("<I", STR)
            + s1("gpt2") + s1("gpt2.context_length") + struct.pack("<I", U32) + struct.pack("<I", 1024))
    h = gguf.read_header(io.BytesIO(blob))
    assert h["arch"] == "gpt2" and h["context"] == 1024 and h["complete"]
