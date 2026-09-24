"""HF service: search, model detail, and the ModelInfo builder (pure) used by the fit estimator."""
from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from engine_console.adapters.base import ModelInfo
from engine_console.domain.models import Page
from engine_console.domain.ports import HubClient, HubModel
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


class ModelDetail(BaseModel):
    info: ModelInfo
    card: str
    files: list[dict[str, Any]]
    sha: str | None


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
                           files=[{"path": f.path, "size": f.size, "sha256": f.sha256} for f in hub.files])

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
