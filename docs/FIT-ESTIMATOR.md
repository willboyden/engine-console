# Fit estimator

Source: `backend/src/engine_console/services/fit.py` (pure function `estimate_fit`), fed by the adapter's
`memory_model()` and by `FitService` (`services/fitting.py`). Everything below is per GPU, in GiB
(1 GiB = 2^30 bytes). Unit tests: `backend/tests/core/test_fit.py`. Accuracy against real engine runs
has **not been measured**; see "Calibrating against real measurements".

## Inputs

| Input | Source |
|---|---|
| `ModelInfo`: layers, attention/KV heads, head_dim, hidden_size, `kv_lora_rank`, sliding window, native max length, weight bytes, params, dtype, quantization | `config.json` + safetensors metadata via `services/hf.py` |
| `mem` dict: `tp`, `mem_fraction`, `max_len`, `kv_bytes_per_elem`, optional `activations_gib`, `cuda_graph_gib`, `overhead_gib`, `kv_cache_fixed_gib`, `nonstatic_reserve_frac` | adapter `memory_model(model, params)` |
| `Hardware`: per-GPU total and currently free VRAM | NVML (`services/hardware.py`) |
| `concurrency` (default 1), resident instances | request body / lifecycle |

## Formulas

```
weights_gpu   = weight_bytes / tp        (if weight_bytes unknown: params * bytes_per_param[dtype|quant])
KV / token / layer (standard attention) = 2 * ceil(kv_heads / tp) * head_dim * kv_bytes
KV / token / layer (MLA)                = (kv_lora_rank + qk_rope_head_dim) * kv_bytes   (replicated on every rank)
KV(ctx, seqs) = per_layer * (n_full * ctx + n_sliding * min(ctx, window)) * seqs
kv_cache_gib  = KV(max_len, concurrency)     # smallest pool at which the configured context is servable
activations   = max(0.5, batched_tokens * hidden * 2 * 16 / tp)   (default batched_tokens 8192; 1.5 if hidden unknown)
cuda_graphs   = 1.0   (default; adapter may override)
overhead      = 1.0   (default; CUDA context + NCCL buffers)
total         = weights_gpu + activations + cuda_graphs + overhead + kv_cache_gib
budget        = gpu_total * mem_fraction      (0.90 assumed if the engine gives none)
verdict       = fits if total <= 0.90*budget ; tight if <= 1.00*budget ; else wont_fit
```

Other rules in the code:

- `max_context_at_current_concurrency` inverts the KV formula on the remaining pool (`budget - fixed`),
  capped at the model's native length; `max_concurrency_at_current_context` is `pool // KV(max_len, 1)`.
- If the adapter reports `kv_cache_fixed_gib` (an explicit KV pool), that value replaces the computed KV.
- If the adapter reports `nonstatic_reserve_frac` (SGLang-style), activations are not charged to the static
  budget but still count against physically free VRAM.
- A configuration that passes the budget check still becomes `wont_fit` if the total exceeds currently
  **free** VRAM; when stopping resident instances would make room, `fits_if_stop` lists them.
- `tp_required` is the smallest tensor-parallel degree at which the same configuration fits.
- Sliding-window layers use `layer_types` or `sliding_window_pattern`; with a window but no pattern every
  layer is assumed windowed (confidence lowered to medium).
- `tp > kv_heads`: KV heads are replicated across ranks (each rank keeps at least one).

## Confidence

`high` unless a rule lowered it. Derived weights (params x bytes/param), unknown native length, guessed
activations or the windowed-layer assumption give `medium`. Missing layers, heads or head_dim (KV cannot
be computed), or missing weights, give `low`. If KV cannot be sized and weights alone already exceed the
budget, the verdict is `wont_fit` at medium confidence (a lower bound suffices); otherwise `unknown`.
The estimator degrades rather than inventing inputs.

## Known error sources

1. Activations, CUDA graphs and overhead are heuristics (constants above), not measurements. They matter
   most for small models and large batch/context settings.
2. Sliding-window KV assumes the engine uses a hybrid allocator. An engine that allocates full length for
   those layers needs more than estimated.
3. Quantized checkpoints: weight size comes from safetensors sizes when available; unquantized layers
   (embeddings, lm_head, routers) count only as far as they are in the files.
4. MoE and multimodal: vision towers and expert-parallel layouts are not modelled beyond total weight bytes / tp.
5. Paged-KV block rounding, prefix cache and fragmentation are not modelled; real usable KV is a little below the pool.
6. Speculative-decoding draft models and LoRA adapters are not added to weights (unverified: no such
   handling found in `fit.py`).
7. Free VRAM is a snapshot. Another engine, a fine-tuning job or any other GPU process can change it between estimate and launch.
8. Only `tp` is considered; pipeline and context parallelism are not.

## Calibrating against real measurements

Nothing auto-calibrates. To check one model/engine pair by hand:

1. Launch from the console with a known `max_len`, concurrency-equivalent (`max_num_seqs`) and
   `mem_fraction`; note the estimate shown in the launch view (`POST /api/v1/fit`).
2. Once `ready`, read real usage: `nvidia-smi --query-gpu=uuid,memory.used --format=csv`, plus the
   engine's own startup lines for weight memory and KV-cache size.
3. Compare per component: weights, KV (engine KV tokens x bytes/token), and the remainder
   (`memory.used - weights - KV`) against `activations + cuda_graphs + overhead`.
4. If the remainder differs consistently, tune the adapter's `memory_model()` values
   (`cuda_graph_gib`, `overhead_gib`, `activations_gib`), not the core constants: the error is engine-specific.
   Record the measurement first and add a `test_fit.py` case with the measured numbers.
5. Repeat at two context lengths to separate KV error (scales with context) from fixed overhead.

No measured comparison table is included because none has been run for this document.
