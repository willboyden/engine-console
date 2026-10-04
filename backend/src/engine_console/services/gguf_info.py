"""GGUF and Ollama facts -> ModelInfo (contract v2, docs/ARCHITECTURE.md 10.5: `model_info` for gguf and ollama copies).

The hub never opens a model file. A node's scanner reads a GGUF header (or an Ollama config blob) and hands the hub an
entry with a few fixed fields (`arch`, `params`, `context`, `quant`, `shards`) plus a bounded `meta` object holding
the metadata keys it found. The locked protocol leaves the shape of `meta` open (it is untrusted JSON), so this module
accepts llama.cpp's own key names (`<arch>.block_count`, `<arch>.attention.head_count_kv`, `general.file_type` ...)
either flat ("llama.block_count") or nested ({"llama": {"block_count": 32}}), and the short names of the node's header
reader (`ecnode.gguf.read_header`: block_count, head_count, head_count_kv, embedding_length, expert_count, file_type).
It reads the fixed fields first and leaves anything it cannot find as None, so the fit estimator reports low
confidence instead of a made-up number.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from engine_console.adapters.base import ModelInfo

# llama.cpp's `llama_ftype` values as stored in `general.file_type` (the same table the node's header reader carries),
# written from memory of llama.h and not checked against a real file here; a node that supplies `quant` makes it
# unnecessary, and an unknown value gives None.
FILE_TYPES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
    13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS",
    21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE",
}
_DTYPE = {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}
SHARD_SUFFIX = re.compile(r"-(\d{5})-of-(\d{5})$", re.IGNORECASE)
_SIZE_LABEL = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMBT])\s*$", re.IGNORECASE)
_UNITS = {"K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}
_RAW_KEYS = 64          # raw_config keeps a handful of scalar keys, never tokenizer vocabularies


def quant_from_file_type(value: object) -> str | None:
    return FILE_TYPES.get(value) if isinstance(value, int) and not isinstance(value, bool) else None


def normalize_quant(quant: str | None) -> str | None:
    """The spelling ModelInfo.quantization and the adapters' compatibility rules use: lower case, no vendor prefix,
    "mxfp4_moe" -> "mxfp4" (it is the same format)."""
    if not quant:
        return None
    q = quant.strip().lower()
    for prefix in ("ud-", "i1-"):
        if q.startswith(prefix):
            q = q[len(prefix):]
    return "mxfp4" if q == "mxfp4_moe" else (q or None)


def parse_size_label(label: object) -> int | None:
    """"8.0B" / "70B" / "335M" -> a parameter count (Ollama's `model_type` and GGUF's `general.size_label`)."""
    if not isinstance(label, str):
        return None
    m = _SIZE_LABEL.match(label)
    return int(float(m.group(1)) * _UNITS[m.group(2).upper()]) if m else None


def shard_names(first: str, shards: int | None = None) -> list[str]:
    """Every file of a split set from its first shard ("m-00001-of-00003.gguf" -> the three names); a plain file, or a
    name that is not the first shard, gives itself."""
    directory, _, name = first.rpartition("/")
    stem = name[:-5] if name.lower().endswith(".gguf") else name
    m = SHARD_SUFFIX.search(stem)
    if m is None:
        return [first]
    total = int(m.group(2))
    if int(m.group(1)) != 1 or total < 2 or total > 999 or (shards is not None and shards != total):
        return [first]
    head = stem[:m.start()]
    prefix = f"{directory}/" if directory else ""
    return [f"{prefix}{head}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]


# ---- reading untrusted metadata -------------------------------------------------------------------------------------
def _lookup(meta: Mapping[str, Any], parts: list[str]) -> Any:
    """meta["a.b.c"], meta["a"]["b.c"], meta["a"]["b"]["c"] ...: whichever way the node nested it."""
    for i in range(len(parts), 0, -1):
        head = ".".join(parts[:i])
        if head in meta:
            if i == len(parts):
                return meta[head]
            sub = meta[head]
            if isinstance(sub, Mapping):
                found = _lookup(sub, parts[i:])
                if found is not None:
                    return found
    return None


def _get(meta: Mapping[str, Any], key: str) -> Any:
    return _lookup(meta, key.split(".")) if key else None


def _int(v: object) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if v > 0 else None
    if isinstance(v, float) and v.is_integer() and v > 0:
        return int(v)
    return None


def _int_or_max(v: object) -> int | None:
    """Per-layer arrays exist (hybrid models: zero for a layer without attention). The largest entry is the safe
    answer for a memory estimate."""
    if isinstance(v, list):
        vals = [x for x in (_int(i) for i in v[:512]) if x is not None]
        return max(vals) if vals else None
    return _int(v)


def _str(v: object) -> str | None:
    return v.strip()[:128] if isinstance(v, str) and v.strip() else None


def _first(meta: Mapping[str, Any], keys: list[str]) -> Any:
    for k in keys:
        v = _get(meta, k)
        if v is not None:
            return v
    return None


def model_info_from_entry(name: str, entry: Mapping[str, Any], fmt: str, *, size: int | None = None) -> ModelInfo:
    """A ModelInfo for a gguf or ollama library entry. `size` is the bytes of every file of the model (weights).

    `meta` may carry llama.cpp's own keys ("llama.block_count", "general.file_type") or the short names the node's
    header reader uses ("block_count", "head_count_kv", "file_type"); both are read."""
    raw = entry.get("meta")
    meta: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    arch = (_str(entry.get("arch")) or _str(_first(meta, ["arch", "general.architecture"]))
            or (_str(_get(meta, "model_family")) if fmt == "ollama" else None))

    def keys(suffix: str, *short: str) -> list[str]:
        return ([f"{arch}.{suffix}"] if arch else []) + list(short)

    layers = _int(_first(meta, keys("block_count", "block_count", "n_layer", "num_hidden_layers")))
    heads = _int_or_max(_first(meta, keys("attention.head_count", "head_count", "attention.head_count",
                                          "n_head", "num_attention_heads")))
    kv_heads = _int_or_max(_first(meta, keys("attention.head_count_kv", "head_count_kv", "attention.head_count_kv",
                                             "n_head_kv", "num_key_value_heads")))
    if kv_heads is None and heads is not None:
        kv_heads = heads        # llama.cpp's own default when the key is absent: multi-head attention
    embedding = _int(_first(meta, keys("embedding_length", "embedding_length", "n_embd", "hidden_size")))
    key_len = _int(_first(meta, keys("attention.key_length", "key_length", "attention.key_length", "head_dim")))
    head_dim = key_len if key_len else (embedding // heads if embedding and heads else None)
    experts = _int(_first(meta, keys("expert_count", "expert_count", "num_experts")))
    context = (_int(entry.get("context"))
               or _int(_first(meta, keys("context_length", "context_length", "context", "max_position_embeddings"))))
    params = (_int(entry.get("params")) or _int(_first(meta, ["params", "general.parameter_count"]))
              or parse_size_label(_first(meta, ["general.size_label", "size_label", "model_type"])))
    file_type = _first(meta, ["general.file_type", "file_type"])
    quant = (_str(entry.get("quant")) or _str(_get(meta, "quant"))
             or quant_from_file_type(file_type) or _str(file_type))
    quantization = normalize_quant(quant)
    weight_bytes = _int(size) if size is not None else _int(entry.get("size"))
    raw_config: dict[str, Any] = {}
    for k, v in list(meta.items())[:512]:
        if len(raw_config) >= _RAW_KEYS:
            break
        if isinstance(k, str) and not k.startswith("tokenizer") and isinstance(v, (str, int, float, bool)):
            raw_config[k] = v if not isinstance(v, str) else v[:200]
    return ModelInfo(
        repo_id=name, revision=_str(entry.get("revision")), architectures=[arch] if arch else [], model_type=arch,
        num_params=params, is_moe=bool(experts and experts > 1), weight_bytes=weight_bytes,
        quantization=quantization, dtype=_DTYPE.get(quantization or ""), hidden_size=embedding, num_layers=layers,
        num_attention_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim, max_position_embeddings=context,
        sliding_window=_int(_first(meta, keys("attention.sliding_window", "sliding_window"))),
        kv_lora_rank=_int(_first(meta, keys("attention.kv_lora_rank", "kv_lora_rank"))), raw_config=raw_config)
