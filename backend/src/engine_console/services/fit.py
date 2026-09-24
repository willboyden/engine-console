"""Fit estimator: a pure function from (model facts, engine memory model, hardware) to a FitReport.

Everything is per GPU, in GiB. Design rules:
  * Never fabricate: a missing input degrades `confidence` and, when the KV cache cannot be
    computed at all, the verdict becomes `unknown` unless weights alone already prove `wont_fit`.
  * KV formula: 2 * layers * kv_heads * head_dim * kv_bytes * tokens (K and V), with
      - GQA/MQA via num_kv_heads (MQA = 1), split across TP ranks (replicated when tp > kv_heads),
      - MLA: layers * (kv_lora_rank + qk_rope_head_dim) * kv_bytes, one latent shared by all heads,
        replicated (not split) across TP ranks,
      - sliding-window layers hold min(ctx, window) tokens instead of ctx (assumes the engine's
        hybrid KV allocator; an engine that allocates full length for those layers needs more).
  * `kv_cache_gib` in the breakdown is the KV needed for `concurrency` sequences of `max_len`
    tokens, i.e. the smallest pool at which the configured context is actually servable.
  * Activations, CUDA graphs and overhead are heuristics (see the constants); they are engine
    knobs the adapter can override through memory_model().
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

from engine_console.adapters.base import Compat, Hardware, ModelInfo
from engine_console.domain.models import (
    Confidence,
    FitReport,
    GpuFit,
    HostMemInfo,
    HostRamFit,
    ResidentUse,
    Verdict,
)

GIB = float(1024**3)
FITS_MAX = 0.90       # total <= 90 % of budget  -> fits
TIGHT_MAX = 1.00      # 90 %..100 %              -> tight ; above -> wont_fit
DEFAULT_MEM_FRACTION = 0.90
DEFAULT_OVERHEAD_GIB = 1.0     # CUDA context + NCCL buffers, per GPU
DEFAULT_GRAPH_GIB = 1.0
DEFAULT_BATCHED_TOKENS = 8192
ACT_FACTOR = 16                # bytes-per-hidden-element multiplier for prefill activations (rough)
ACT_MIN_GIB = 0.5
ACT_DEFAULT_GIB = 1.5          # when hidden_size is unknown
DEFAULT_QK_ROPE = 64           # DeepSeek MLA decoupled-rope dim when absent from config

# bytes per parameter, only used when the on-disk safetensors size is unknown
BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4, "float32": 4, "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2,
    "fp8": 1, "int8": 1, "nvfp4": 0.5, "mxfp4": 0.5, "fp4": 0.5, "awq": 0.5, "gptq": 0.5, "int4": 0.5,
}


def _cfg(model: ModelInfo) -> dict[str, Any]:
    raw = model.raw_config
    inner = raw.get("text_config")
    return inner if isinstance(inner, dict) else raw


def _weights_bytes(model: ModelInfo) -> tuple[float | None, str | None]:
    """Return (bytes, caveat). Caveat is set when the number is derived rather than measured."""
    if model.weight_bytes:
        return float(model.weight_bytes), None
    if model.num_params:
        key = (model.quantization or model.dtype or "").lower()
        bpp = BYTES_PER_PARAM.get(key)
        if bpp is not None:
            return model.num_params * bpp, f"weights estimated from {model.num_params:,} params at {bpp} B/param"
    return None, None


def _layer_split(model: ModelInfo) -> tuple[int, int, int | None, str | None]:
    """(full_layers, sliding_layers, window, note)."""
    layers = model.num_layers or 0
    window = model.sliding_window
    cfg = _cfg(model)
    if not window or cfg.get("use_sliding_window") is False or layers <= 0:
        return layers, 0, None, None
    types = cfg.get("layer_types")
    if isinstance(types, list) and len(types) == layers:
        n_sl = sum(1 for t in types if t == "sliding_attention")
        return layers - n_sl, n_sl, window, None
    pattern = cfg.get("sliding_window_pattern")
    if isinstance(pattern, int) and pattern > 1:
        n_full = layers // pattern
        return n_full, layers - n_full, window, None
    return 0, layers, window, "sliding_window set without a layer pattern: assumed every layer is windowed"


class _Kv:
    """Per-GPU KV cache sizing for one (model, kv dtype, tp)."""

    def __init__(self, per_layer_token_bytes: float, n_full: int, n_sl: int, window: int | None) -> None:
        self.per_layer = per_layer_token_bytes
        self.n_full, self.n_sl, self.window = n_full, n_sl, window

    def bytes_for(self, ctx: int, seqs: int) -> float:
        eff_sl = min(ctx, self.window) if self.window else ctx
        return self.per_layer * (self.n_full * ctx + self.n_sl * eff_sl) * seqs

    @property
    def bytes_per_token_full(self) -> float:
        return self.per_layer * (self.n_full + self.n_sl)

    def max_ctx(self, pool_bytes: float, seqs: int) -> int | None:
        """Largest per-sequence context that fits `seqs` sequences in the pool (closed form)."""
        if pool_bytes <= 0:
            return 0
        t = pool_bytes / (self.per_layer * seqs)   # token-layer budget per sequence
        if self.n_sl == 0 or not self.window:
            return int(t / self.n_full) if self.n_full else None
        if t <= (self.n_full + self.n_sl) * self.window:
            return int(t / (self.n_full + self.n_sl))
        return int((t - self.n_sl * self.window) / self.n_full) if self.n_full else None


def _kv_model(model: ModelInfo, tp: int, kv_bytes: float) -> tuple[_Kv | None, list[str], bool]:
    """Build the KV sizing. Returns (kv|None, notes, lowered) where lowered means confidence -> low."""
    notes: list[str] = []
    layers = model.num_layers
    if not layers:
        return None, ["num_layers missing from config: KV cache cannot be computed"], True
    n_full, n_sl, window, wnote = _layer_split(model)
    if wnote:
        notes.append(wnote)
    if model.kv_lora_rank:
        rope = int(_cfg(model).get("qk_rope_head_dim", DEFAULT_QK_ROPE))
        per_layer = (model.kv_lora_rank + rope) * kv_bytes
        notes.append("MLA latent KV cache is replicated on every TP rank")
        return _Kv(per_layer, n_full, n_sl, window), notes, False
    heads = model.num_attention_heads
    kv_heads = model.num_kv_heads
    lowered = False
    if kv_heads is None:
        if not heads:
            return None, notes + ["attention head counts missing: KV cache cannot be computed"], True
        kv_heads = heads
        lowered = True
        notes.append("num_key_value_heads missing: assumed MHA (upper bound on KV size)")
    head_dim = model.head_dim or (model.hidden_size // heads if model.hidden_size and heads else None)
    if not head_dim:
        return None, notes + ["head_dim unknown: KV cache cannot be computed"], True
    per_gpu_heads = max(1, math.ceil(kv_heads / max(tp, 1)))
    if tp > kv_heads:
        notes.append(f"tp={tp} exceeds {kv_heads} KV heads: KV heads are replicated across ranks")
    return _Kv(2 * per_gpu_heads * head_dim * kv_bytes, n_full, n_sl, window), notes, lowered


def _activations_gib(model: ModelInfo, mem: dict[str, float], tp: int) -> tuple[float, bool]:
    if "activations_gib" in mem:
        return float(mem["activations_gib"]), False
    if not model.hidden_size:
        return ACT_DEFAULT_GIB, True
    tokens = mem.get("max_batched_tokens", DEFAULT_BATCHED_TOKENS)
    return max(ACT_MIN_GIB, tokens * model.hidden_size * 2 * ACT_FACTOR / tp / GIB), False


def _verdict(total: float, budget: float) -> Verdict:
    if budget <= 0:
        return "wont_fit"
    ratio = total / budget
    if ratio <= FITS_MAX:
        return "fits"
    if ratio <= TIGHT_MAX:
        return "tight"
    return "wont_fit"


_CONF: tuple[Confidence, ...] = ("high", "medium", "low")
_RANK = {"fits": 0, "tight": 1, "unknown": 2, "wont_fit": 3}


def _worst(verdicts: Sequence[Verdict]) -> Verdict:
    return max(verdicts, key=lambda v: _RANK[v]) if verdicts else "unknown"


def _tp_required(model: ModelInfo, mem: dict[str, float], hw: Hardware, weights: float | None,
                 max_len: int | None, concurrency: int, kv_bytes: float, fraction: float) -> int | None:
    if weights is None or not hw.gpu_total_gib:
        return None
    budget = min(hw.gpu_total_gib) * fraction
    for tp in (1, 2, 4, 8):
        if tp > len(hw.gpu_ids):
            break
        if model.num_attention_heads and model.num_attention_heads % tp:
            continue
        act, _ = _activations_gib(model, mem, tp)
        fixed = weights / tp / GIB + act + mem.get("cuda_graph_gib", DEFAULT_GRAPH_GIB) \
            + mem.get("overhead_gib", DEFAULT_OVERHEAD_GIB)
        kv_gib = 0.0
        if max_len:
            kv, _, _ = _kv_model(model, tp, kv_bytes)
            if kv is not None:
                kv_gib = kv.bytes_for(max_len, concurrency) / GIB
        if fixed + kv_gib <= budget:
            return tp
    return None


def estimate_fit(model: ModelInfo, mem: dict[str, float], hw: Hardware, *, concurrency: int = 1,
                 resident: Sequence[ResidentUse] = (), compat: Sequence[Compat] = (),
                 hw_all: Hardware | None = None) -> FitReport:
    notes: list[str] = []
    conf_rank = 0  # 0 high, 1 medium, 2 low

    def lower(to: int) -> None:
        nonlocal conf_rank
        conf_rank = max(conf_rank, to)

    n_gpu = len(hw.gpu_ids)
    tp = max(1, int(mem.get("tp") or n_gpu or 1))
    fraction = float(mem.get("mem_fraction", DEFAULT_MEM_FRACTION))
    if "mem_fraction" not in mem:
        notes.append(f"engine gave no mem_fraction: assumed {DEFAULT_MEM_FRACTION}")
    kv_bytes = float(mem.get("kv_bytes_per_elem", 2.0))

    weights_total, wcaveat = _weights_bytes(model)
    if wcaveat:
        notes.append(wcaveat)
        lower(1)

    max_len = int(mem["max_len"]) if mem.get("max_len") else None
    if max_len is None and model.max_position_embeddings:
        max_len = model.max_position_embeddings
        notes.append(f"max_len not set: using the model's native {max_len}")
        lower(1)
    if max_len and model.max_position_embeddings and max_len > model.max_position_embeddings:
        notes.append(f"max_len {max_len} exceeds native {model.max_position_embeddings} (needs rope scaling)")

    compat_list = list(compat)
    if n_gpu == 0:
        return FitReport(verdict="unknown", confidence="low", tp=tp, tp_required=None, concurrency=concurrency,
                         max_len=max_len, per_gpu=[], max_context_at_current_concurrency=None,
                         max_concurrency_at_current_context=None,
                         notes=notes + ["no GPU selected or detected"], compat=compat_list)

    kv, kv_notes, kv_low = _kv_model(model, tp, kv_bytes)
    notes.extend(kv_notes)
    if kv_low:
        lower(2)
    if kv is not None and any("assumed every layer" in n for n in kv_notes):
        lower(1)
    if kv is not None and max_len is None:
        kv = None
        notes.append("context length unknown: KV cache cannot be sized")
        lower(2)

    act_est, act_default = _activations_gib(model, mem, tp)
    # SGLang-style engines carve activations out of the non-static share, so the static budget
    # (weights + KV) is not charged for them; the adapter signals that with nonstatic_reserve_frac.
    charge_act = "nonstatic_reserve_frac" not in mem
    act = act_est if charge_act else 0.0
    if act_default:
        notes.append("hidden_size missing: activation memory is a default guess")
        lower(1)
    graphs = float(mem.get("cuda_graph_gib", DEFAULT_GRAPH_GIB))
    overhead = float(mem.get("overhead_gib", DEFAULT_OVERHEAD_GIB))
    weights_gpu = weights_total / tp / GIB if weights_total is not None else None
    kv_gib = kv.bytes_for(max_len, concurrency) / GIB if kv is not None and max_len else None
    fixed_kv = mem.get("kv_cache_fixed_gib")   # engine reserves an explicit KV pool (e.g. vLLM kv-cache-memory)
    if fixed_kv:
        kv_gib = float(fixed_kv)
        notes.append(f"KV pool fixed by engine settings at {fixed_kv:g} GiB per GPU")

    if tp > n_gpu:
        notes.append(f"tp={tp} needs {tp} GPUs but only {n_gpu} selected")

    used = list(range(min(tp, n_gpu)))
    per_gpu: list[GpuFit] = []
    verdicts: list[Verdict] = []
    freeable: set[str] = set()
    known_pool: list[float] = []
    if weights_gpu is None:
        notes.append("weights size unknown (no safetensors sizes or parameter count)")
        lower(2)
        verdicts.append("unknown")
    for i in used:
        total_vram = hw.gpu_total_gib[i]
        budget = total_vram * fraction
        free = hw.gpu_free_gib[i]
        if weights_gpu is None:
            per_gpu.append(GpuFit(gpu_id=hw.gpu_ids[i], uuid=hw.gpu_uuids[i], name=hw.gpu_names[i],
                                  weights_gib=0.0, kv_cache_gib=0.0, activations_gib=round(act, 3),
                                  cuda_graphs_gib=graphs, overhead_gib=overhead, total_gib=0.0,
                                  budget_gib=round(budget, 3), vram_total_gib=total_vram, free_gib=free,
                                  utilization_pct=0.0, verdict="unknown"))
            continue
        fixed = weights_gpu + act + graphs + overhead
        if kv_gib is None:
            total = fixed   # lower bound: KV unknown
            v: Verdict = "wont_fit" if fixed > budget else "unknown"
        else:
            total = fixed + kv_gib
            v = _verdict(total, budget)
            known_pool.append(budget - fixed)
        physical = total + (0.0 if charge_act else act_est)
        if not charge_act and act_est + graphs + overhead > mem["nonstatic_reserve_frac"] * total_vram:
            notes.append(f"GPU {hw.gpu_ids[i]}: estimated activations ({act_est:.1f} GiB) exceed the non-static "
                         f"reserve ({mem['nonstatic_reserve_frac'] * total_vram:.1f} GiB); lower the static fraction")
        if v in ("fits", "tight") and physical > free:
            v = "wont_fit"
            here = [r for r in resident if hw.gpu_ids[i] in r.gpu_ids]
            if here and physical <= free + sum(r.gib_per_gpu for r in here):
                freeable.update(r.name for r in here)
                notes.append(f"GPU {hw.gpu_ids[i]}: needs {physical:.1f} GiB but only {free:.1f} GiB is free "
                             "(other engines are resident)")
            else:
                notes.append(f"GPU {hw.gpu_ids[i]}: needs {physical:.1f} GiB but only {free:.1f} GiB is free")
        per_gpu.append(GpuFit(gpu_id=hw.gpu_ids[i], uuid=hw.gpu_uuids[i], name=hw.gpu_names[i],
                              weights_gib=round(weights_gpu, 3), kv_cache_gib=round(kv_gib or 0.0, 3),
                              activations_gib=round(act, 3), cuda_graphs_gib=graphs, overhead_gib=overhead,
                              total_gib=round(total, 3), budget_gib=round(budget, 3),
                              vram_total_gib=total_vram, free_gib=free,
                              utilization_pct=round(total / budget * 100, 1) if budget > 0 else 0.0, verdict=v))
        verdicts.append(v)

    verdict = _worst(verdicts)
    if tp > n_gpu:
        verdict = "wont_fit"
    conf: Confidence
    if kv_gib is None and verdict == "unknown":
        conf = "low"
    elif verdict == "wont_fit" and kv_gib is None:
        conf = "medium"   # a lower bound already exceeds the budget
    else:
        conf = _CONF[conf_rank]

    max_ctx: int | None = None
    max_conc: int | None = None
    if kv is not None and max_len and known_pool:
        pool = (float(fixed_kv) if fixed_kv else min(known_pool)) * GIB
        max_ctx = kv.max_ctx(pool, concurrency)
        if max_ctx is not None and model.max_position_embeddings:
            max_ctx = min(max_ctx, model.max_position_embeddings)
        one = kv.bytes_for(max_len, 1)
        max_conc = max(0, int(pool // one)) if one > 0 else None

    tpr = _tp_required(model, mem, hw_all or hw, weights_total, max_len, concurrency, kv_bytes, fraction)
    if any(c.level == "block" for c in compat_list) and verdict in ("fits", "tight"):
        notes.append("engine compatibility rules block this configuration")

    return FitReport(verdict=verdict, confidence=conf, tp=tp, tp_required=tpr, concurrency=concurrency,
                     max_len=max_len, kv_bytes_per_token=kv.bytes_per_token_full if kv else None,
                     per_gpu=per_gpu, max_context_at_current_concurrency=max_ctx,
                     max_concurrency_at_current_context=max_conc, fits_if_stop=sorted(freeable),
                     notes=notes, compat=compat_list)


ENGINE_PROCESS_GIB = 4.0   # heuristic: python + CUDA runtime anon memory (~3 GiB measured on a running SGLang container)
HOST_OK = 0.80


def estimate_host_ram(breakdown: dict[str, float], host: HostMemInfo | None) -> HostRamFit:
    """Host-RAM verdict from an adapter's `host_memory_gib()` plus a small engine-process baseline.

    Only unreclaimable memory (anon, shared, pinned) counts against MemAvailable, which already excludes the
    reclaimable page cache. A NaN component means the adapter could not compute it: verdict `unknown`."""
    unknown = sorted(k for k, v in breakdown.items() if v != v)
    known = {k: round(v, 3) for k, v in breakdown.items() if v == v}
    known["engine_process"] = ENGINE_PROCESS_GIB
    needed = round(sum(known.values()), 3)
    notes = [
        "Weight loading streams through the page cache; only the unreclaimable part (anon, shared memory, pinned "
        "buffers) counts against available RAM, so the model's file size is not added here.",
        f"engine_process is a {ENGINE_PROCESS_GIB:g} GiB heuristic (python + CUDA runtime), not a measurement.",
    ]
    verdict: Literal["ok", "tight", "wont_fit", "unknown"]
    if unknown:
        notes.append(f"could not be computed (model or hardware facts missing): {', '.join(unknown)}; needed is a lower bound")
    if host is None:
        notes.append("host memory could not be read")
        return HostRamFit(needed_gib=needed, available_gib=None, total_gib=None, verdict="unknown", breakdown=known, notes=notes)
    if unknown:
        verdict = "unknown"
    else:
        ratio = needed / host.available_gib if host.available_gib > 0 else float("inf")
        verdict = "ok" if ratio <= HOST_OK else "tight" if ratio <= 1.0 else "wont_fit"
    if host.shmem_gib > 0.25 * host.total_gib:
        notes.append(f"{host.shmem_gib:.1f} GiB of host RAM is already shared memory and cannot be reclaimed")
    return HostRamFit(needed_gib=needed, available_gib=round(host.available_gib, 3), total_gib=round(host.total_gib, 3),
                      verdict=verdict, breakdown=known, notes=notes)
