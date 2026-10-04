"""A minimal, strictly bounded GGUF header reader.

Reads only what a model library needs: general.architecture / name / file_type, a few `{arch}.*` sizes and the
total tensor element count (parameters). Every length and count in a GGUF header is untrusted: before anything
is read or skipped it is checked against a fixed byte budget, so a truncated or hostile file can neither raise
out of this module nor make it allocate more than MAX_KEEP_STR bytes per value.
"""
from __future__ import annotations

import os
import struct
from typing import IO, Any

MAGIC = b"GGUF"
MAX_HEADER_BYTES = 16 * 1024 * 1024    # bytes of header this reader will consume, at most
MAX_KV = 1_000_000
MAX_TENSORS = 1_000_000
MAX_DIMS = 8
MAX_KEY = 65_535
MAX_KEEP_STR = 4096
MAX_DEPTH = 3

# value types
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)
_FIXED = {_U8: "<B", _I8: "<b", _U16: "<H", _I16: "<h", _U32: "<I", _I32: "<i", _F32: "<f", _BOOL: "<?",
          _U64: "<Q", _I64: "<q", _F64: "<d"}
_SIZE = {t: struct.calcsize(f) for t, f in _FIXED.items()}

ARCH_KEYS = ("context_length", "block_count", "embedding_length", "attention.head_count",
             "attention.head_count_kv", "expert_count", "expert_used_count")
GENERAL_KEYS = ("general.architecture", "general.name", "general.file_type", "general.size_label",
                "general.basename", "split.count", "split.no")

# llama.cpp LLAMA_FTYPE values (general.file_type) -> the quant name people use
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
              12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
              19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
              26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
              36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE"}


class _Stop(Exception):
    """Out of budget, out of file, or malformed: stop reading and keep what was found."""


class _Reader:
    def __init__(self, f: IO[bytes], limit: int) -> None:
        self.f = f
        self.left = limit

    def take(self, n: int) -> bytes:
        if n < 0 or n > self.left:
            raise _Stop
        b = self.f.read(n)
        if len(b) != n:
            raise _Stop
        self.left -= n
        return b

    def skip(self, n: int) -> None:
        if n < 0 or n > self.left:
            raise _Stop
        self.left -= n
        while n > 0:          # read, not seek: a seek past the end would not tell us the file is truncated
            chunk = self.f.read(min(n, 1 << 20))
            if not chunk:
                raise _Stop
            n -= len(chunk)

    def unpack(self, fmt: str) -> Any:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]


def _read_str(r: _Reader, wide: bool, keep: int) -> str | None:
    n = r.unpack("<Q" if wide else "<I")
    if n > keep:
        r.skip(n)
        return None
    return r.take(n).decode("utf-8", "replace")


def _skip_value(r: _Reader, vtype: int, wide: bool, depth: int = 0) -> None:
    if vtype in _SIZE:
        r.skip(_SIZE[vtype])
    elif vtype == _STR:
        r.skip(r.unpack("<Q" if wide else "<I"))
    elif vtype == _ARR:
        if depth >= MAX_DEPTH:
            raise _Stop
        etype = r.unpack("<I")
        count = r.unpack("<Q" if wide else "<I")
        if etype in _SIZE:
            r.skip(count * _SIZE[etype])          # skip() refuses anything beyond the budget
        elif etype in (_STR, _ARR):
            if count > r.left // 4:               # every element costs at least 4 bytes
                raise _Stop
            for _ in range(count):
                _skip_value(r, etype, wide, depth + 1)
        else:
            raise _Stop
    else:
        raise _Stop


def _read_value(r: _Reader, vtype: int, wide: bool) -> Any:
    if vtype in _FIXED:
        return r.unpack(_FIXED[vtype])
    if vtype == _STR:
        return _read_str(r, wide, MAX_KEEP_STR)
    _skip_value(r, vtype, wide)
    return None


def read_header(f: IO[bytes], *, budget: int = MAX_HEADER_BYTES, tensors: bool = True) -> dict[str, Any] | None:
    """Facts from a GGUF header, or None if this is not a GGUF file. Never raises for bad input.
    Result keys: version, arch, name, file_type, quant, size_label, context, block_count, embedding_length,
    head_count, head_count_kv, expert_count, expert_used_count, split_count, tensors, params, complete."""
    r = _Reader(f, budget)
    try:
        if r.take(4) != MAGIC:
            return None
        version = r.unpack("<I")
    except (_Stop, OSError, struct.error):
        return None
    out: dict[str, Any] = {"version": version, "complete": False}
    if version not in (1, 2, 3):
        return out
    wide = version >= 2
    kv: dict[str, Any] = {}
    try:
        n_tensors = r.unpack("<Q" if wide else "<I")
        n_kv = r.unpack("<Q" if wide else "<I")
        if n_kv > MAX_KV or n_tensors > MAX_TENSORS:
            raise _Stop
        for _ in range(n_kv):
            klen = r.unpack("<Q" if wide else "<I")
            if klen > MAX_KEY:
                raise _Stop
            key = r.take(klen).decode("utf-8", "replace")
            vtype = r.unpack("<I")
            if key in GENERAL_KEYS or any(key.endswith("." + s) for s in ARCH_KEYS):
                kv[key] = _read_value(r, vtype, wide)
            else:
                _skip_value(r, vtype, wide)
        if tensors:
            total = 0
            for _ in range(n_tensors):
                _read_str(r, wide, 0)
                n_dims = r.unpack("<I")
                if n_dims > MAX_DIMS:
                    raise _Stop
                count = 1
                for _ in range(n_dims):
                    count *= r.unpack("<Q")
                r.skip(4 + 8)                 # type, offset
                total += count
            out["tensors"] = n_tensors
            out["params"] = total if total <= 2**53 else None
        out["complete"] = True
    except (_Stop, OSError, struct.error, MemoryError):
        pass
    arch = kv.get("general.architecture")
    arch = arch if isinstance(arch, str) else None
    ft = kv.get("general.file_type")
    out.update({
        "arch": arch, "name": _s(kv.get("general.name")), "size_label": _s(kv.get("general.size_label")),
        "basename": _s(kv.get("general.basename")),
        "file_type": ft if isinstance(ft, int) and not isinstance(ft, bool) else None,
        "quant": FILE_TYPES.get(ft) if isinstance(ft, int) else None,
        "split_count": _i(kv.get("split.count")),
    })
    for suffix, field in (("context_length", "context"), ("block_count", "block_count"),
                          ("embedding_length", "embedding_length"), ("attention.head_count", "head_count"),
                          ("attention.head_count_kv", "head_count_kv"), ("expert_count", "expert_count"),
                          ("expert_used_count", "expert_used_count")):
        out[field] = _i(kv.get(f"{arch}.{suffix}")) if arch else None
    return out


def read_file(path: str, fd: int | None = None, *, budget: int = MAX_HEADER_BYTES,
              tensors: bool = True) -> dict[str, Any] | None:
    """read_header on a file path (or an already opened fd, which is closed afterwards)."""
    try:
        if fd is None:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
        with os.fdopen(fd, "rb", buffering=1 << 16) as f:
            return read_header(f, budget=budget, tensors=tensors)
    except OSError:
        return None


def _s(v: object) -> str | None:
    return v[:256] if isinstance(v, str) else None


def _i(v: object) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 2**53 else None
