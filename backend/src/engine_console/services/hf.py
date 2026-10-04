"""HF service: search, model detail, the ModelInfo builder (pure) used by the fit estimator, and the GGUF quant
grouping the download picker needs (contract v2, docs/ARCHITECTURE.md 10.6)."""
from __future__ import annotations

import re
import time
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel

from engine_console.adapters.base import ModelInfo
from engine_console.domain.models import Page
from engine_console.domain.ports import HubClient, HubModel, RepoFile
from engine_console.services.store import paginate

_ST_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1,
             "I16": 2, "I32": 4, "I64": 8}
_KNOWN_QUANT_TAGS = ("nvfp4", "fp8", "mxfp4", "awq", "gptq", "int4", "int8")


class SearchHit(BaseModel):
    repo_id: str
    downloads: int | None
    likes: int | None
    pipeline_tag: str | None
    library_name: str | None
    gated: bool
    tags: list[str]
    num_params: int | None
    approx_size_gib: float | None
    quantization: str | None


class GgufGroup(BaseModel):
    """The .gguf files of one quantisation of a repo: one file, or every shard of a split set."""
    quant: str                # "Q4_K_M", "UD-Q4_K_XL", "IQ2_XXS", "BF16", "MXFP4", "mmproj-F16" ...
    files: list[str]          # repo paths, sorted: usable as `allow_patterns` of POST /downloads as they are
    total_bytes: int


class ModelDetail(BaseModel):
    info: ModelInfo
    card: str
    files: list[dict[str, Any]]
    sha: str | None
    gguf_groups: list[GgufGroup] = []     # contract v2: empty unless the repo holds .gguf files


def detect_quantization(cfg: dict[str, Any], repo_id: str = "", tags: list[str] | None = None) -> str | None:
    q = cfg.get("quantization_config")
    if isinstance(q, dict):
        method = str(q.get("quant_method", "")).lower()
        if method == "compressed-tensors":
            for grp in (q.get("config_groups") or {}).values():
                w = (grp or {}).get("weights") or {}
                bits, typ = w.get("num_bits"), str(w.get("type", "")).lower()
                if bits == 4 and typ == "float":
                    return "nvfp4"
                if bits == 8 and typ == "float":
                    return "fp8"
                if bits == 4:
                    return "int4"
                if bits == 8:
                    return "int8"
            return "compressed-tensors"
        if method in ("modelopt", "nvfp4"):
            algo = str(q.get("quant_algo", "")).lower()
            return "nvfp4" if "fp4" in algo else "fp8" if "fp8" in algo else method
        if method == "bitsandbytes":
            return "bnb4" if q.get("load_in_4bit") else "bnb8"
        if method:
            return method
    hay = " ".join([repo_id.lower(), *(t.lower() for t in (tags or []))])
    for k in _KNOWN_QUANT_TAGS:
        if k in hay:
            return k
    return None


def build_model_info(hub: HubModel, revision: str | None = None) -> ModelInfo:
    raw = hub.config or {}
    text_cfg = raw.get("text_config")
    cfg: dict[str, Any] = text_cfg if isinstance(text_cfg, dict) else raw

    def geti(*keys: str) -> int | None:
        for src in (cfg, raw):
            for k in keys:
                v = src.get(k)
                if isinstance(v, int) and not isinstance(v, bool):
                    return v
        return None

    experts = geti("num_experts", "n_routed_experts", "num_local_experts")
    hidden, heads = geti("hidden_size", "n_embd", "d_model"), geti("num_attention_heads", "n_head")
    mla = geti("kv_lora_rank")
    head_dim = geti("head_dim")
    if head_dim is None and hidden and heads and not mla:
        head_dim = hidden // heads
    st_files = [f.size for f in hub.files if f.path.endswith(".safetensors")]
    weight_bytes: int | None = sum(st_files) if st_files else None
    if weight_bytes is None and hub.safetensors_params:
        weight_bytes = sum(n * _ST_BYTES.get(dt, 2) for dt, n in hub.safetensors_params.items())
    arch = raw.get("architectures")
    dtype = cfg.get("torch_dtype") or cfg.get("dtype") or raw.get("torch_dtype") or raw.get("dtype")
    return ModelInfo(
        repo_id=hub.repo_id, revision=revision, architectures=[a for a in arch if isinstance(a, str)] if isinstance(arch, list) else [],
        model_type=raw.get("model_type"), num_params=hub.safetensors_total, is_moe=bool(experts and experts > 1),
        weight_bytes=weight_bytes, quantization=detect_quantization(raw, hub.repo_id, hub.tags),
        dtype=dtype if isinstance(dtype, str) else None, hidden_size=hidden,
        num_layers=geti("num_hidden_layers", "n_layer"), num_attention_heads=heads,
        num_kv_heads=geti("num_key_value_heads", "num_kv_heads", "multi_query_group_num"),
        head_dim=head_dim, max_position_embeddings=geti("max_position_embeddings", "n_positions"),
        sliding_window=geti("sliding_window"), kv_lora_rank=mla, gated=hub.gated, license=hub.license,
        pipeline_tag=hub.pipeline_tag, raw_config=raw)


# ---- GGUF quantisation names ----------------------------------------------------------------------------------------
# llama.cpp's quantisation names, plus the two prefixes repos put in front of them: Unsloth's dynamic quants
# ("UD-Q4_K_XL") and imatrix quants ("i1-Q4_K_M"). Matched case-insensitively (some repos use lower case) and only as a
# whole token, so "Q4_K_M" is found in "model-Q4_K_M-00001-of-00003.gguf" and in "model.q4_k_m.gguf" but not in a word.
_GGUF_SPLIT = re.compile(r"-\d{5}-of-\d{5}$", re.IGNORECASE)
_GGUF_QUANT = re.compile(
    r"(?<![A-Za-z0-9])((?:UD-|i1-)?(?:IQ\d_(?:XXS|XS|S|M|NL|XL)|TQ\d_0|Q\d_K(?:_(?:XXS|XS|S|M|L|XL))?|Q\d_[01]"
    r"|BF16|F16|F32|MXFP4(?:_MOE)?|NVFP4))(?![A-Za-z0-9])", re.IGNORECASE)


def gguf_stem(path: str) -> str:
    """The path without ".gguf" and without a split-set suffix ("-00001-of-00003")."""
    p = path[:-5] if path.lower().endswith(".gguf") else path
    return _GGUF_SPLIT.sub("", p)


def gguf_quant(path: str) -> str:
    """The quantisation token of a .gguf repo path ("unsloth/x/Q4_K_M/Model-Q4_K_M-00001-of-00002.gguf" -> "Q4_K_M").

    The last token wins (the quant is conventionally the last part of a name) and the directories count too, because
    big repos put each quant in its own folder. A multimodal projector ("mmproj-F16.gguf") is its own group, never
    mixed into the model's. A file with no recognisable token is labelled with its own name instead of guessing."""
    stem = gguf_stem(path)
    base = stem.rsplit("/", 1)[-1]
    found = list(_GGUF_QUANT.finditer(stem))
    tok: str | None = None
    if found:
        raw = found[-1].group(1)
        low = raw.lower()
        tok = ("UD-" + raw[3:].upper() if low.startswith("ud-") else
               "i1-" + raw[3:].upper() if low.startswith("i1-") else raw.upper())
    if base.lower().startswith("mmproj"):
        return f"mmproj-{tok}" if tok else "mmproj"
    return tok or base


def group_gguf(files: Iterable[tuple[str, int]]) -> list[GgufGroup]:
    """Group (path, size) pairs of .gguf files by quantisation; smallest group first (the order a picker wants)."""
    by_quant: dict[str, list[tuple[str, int]]] = {}
    for path, size in files:
        if path.lower().endswith(".gguf"):
            by_quant.setdefault(gguf_quant(path), []).append((path, size))
    groups = [GgufGroup(quant=q, files=sorted(p for p, _ in items), total_bytes=sum(s for _, s in items))
              for q, items in by_quant.items()]
    return sorted(groups, key=lambda g: (g.total_bytes, g.quant))


def _file_pairs(files: Iterable[RepoFile | Mapping[str, Any]]) -> Iterable[tuple[str, int]]:
    for f in files:
        if isinstance(f, RepoFile):
            yield f.path, f.size
        else:
            path, size = f.get("path"), f.get("size")
            if isinstance(path, str) and isinstance(size, int) and not isinstance(size, bool):
                yield path, size


class HfService:
    def __init__(self, hub: HubClient, *, cache_ttl_s: float = 300.0) -> None:
        self._hub = hub
        self._ttl = cache_ttl_s
        self._cache: dict[tuple[str, str], tuple[float, HubModel]] = {}

    async def _hub_model(self, repo_id: str, revision: str | None) -> HubModel:
        key = (repo_id, revision or "main")
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self._ttl:
            return hit[1]
        m = await self._hub.model(repo_id, revision)
        self._cache[key] = (time.monotonic(), m)
        return m

    async def model_info(self, repo_id: str, revision: str | None = None) -> ModelInfo:
        return build_model_info(await self._hub_model(repo_id, revision), revision)

    async def detail(self, repo_id: str, revision: str | None = None) -> ModelDetail:
        hub = await self._hub_model(repo_id, revision)
        card = await self._hub.card(repo_id, revision)
        return ModelDetail(info=build_model_info(hub, revision), card=card, sha=hub.sha,
                           files=[{"path": f.path, "size": f.size, "sha256": f.sha256} for f in hub.files],
                           gguf_groups=self.gguf_groups(hub.files))

    @staticmethod
    def gguf_groups(files: Iterable[RepoFile | Mapping[str, Any]]) -> list[GgufGroup]:
        """The quantisations a GGUF repo offers, split sets kept together: [{quant, files, total_bytes}]."""
        return group_gguf(_file_pairs(files))

    async def list_files(self, repo_id: str, revision: str | None) -> HubModel:
        return await self._hub_model(repo_id, revision)

    async def search(self, *, q: str = "", task: str | None = None, library: str | None = None,
                     quant: str | None = None, max_gib: float | None = None, gated: bool | None = None,
                     sort: str = "downloads", limit: int = 20, cursor: str | None = None) -> Page[SearchHit]:
        limit, offset = paginate(limit, cursor)
        # HF has no offset param: over-fetch and slice; filters that HF can't do run client-side.
        fetch = min(offset + limit + 1, 100)
        raw = await self._hub.search(q=q, task=task, library=library, quant=quant, sort=sort, limit=fetch)
        hits: list[SearchHit] = []
        for m in raw:
            size = None
            if m.safetensors_params:
                size = sum(n * _ST_BYTES.get(dt, 2) for dt, n in m.safetensors_params.items()) / 1024**3
            if max_gib is not None and size is not None and size > max_gib:
                continue
            if gated is not None and m.gated != gated:
                continue
            hits.append(SearchHit(repo_id=m.repo_id, downloads=m.downloads, likes=m.likes,
                                  pipeline_tag=m.pipeline_tag, library_name=m.library_name, gated=m.gated,
                                  tags=m.tags[:40], num_params=m.safetensors_total,
                                  approx_size_gib=round(size, 2) if size is not None else None,
                                  quantization=detect_quantization({}, m.repo_id, m.tags)))
        page = hits[offset: offset + limit]
        more = len(hits) > offset + limit
        return Page[SearchHit](items=page, next_cursor=str(offset + limit) if more else None)
