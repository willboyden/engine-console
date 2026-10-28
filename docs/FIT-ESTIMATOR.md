# Fit estimator

Source: `backend/crates/ec-hub/src/services/fit.rs` (pure function `estimate_fit`), fed by the adapter's
`memory_model()` and by `FitService` (`services/fitting.rs`, same directory). The formulas are unchanged by the 0.5.0 Rust
rewrite, which reproduces the Python results (the Python tests of this estimator are mapped to Rust tests in
`backend/TEST-MAP.md`). Everything below is per GPU, in GiB
(1 GiB = 2^30 bytes). Unit tests: `backend/crates/ec-hub/tests/test_fit.rs`, `test_fit_offload.rs` and `test_fit_host_ram.rs`. Accuracy
against real engine runs has **not been measured**, with one exception: a single measurement of the memory a vLLM process
holds outside its configured share (see "Measured: memory held outside the pool"). See also "Calibrating against real
measurements".

## Inputs

| Input | Source |
|---|---|
| `ModelInfo`: layers, attention/KV heads, head_dim, hidden_size, `kv_lora_rank`, sliding window, native max length, weight bytes, params, dtype, quantization | `config.json` + safetensors metadata via `services/hf.rs` |
| `mem` dict: `tp`, `mem_fraction`, `max_len`, `kv_bytes_per_elem`, optional `activations_gib`, `cuda_graph_gib`, `overhead_gib`, `kv_cache_fixed_gib`, `nonstatic_reserve_frac`, the partial-offload shares `weights_gpu_fraction` and `kv_gpu_fraction`, and how the engine sizes its share: `share_of_free`, `claims_share`, `outside_share_gib` (see "How an engine sizes its share") | adapter `memory_model(model, params)` |
| `Hardware`: per-GPU total and currently free VRAM | the node's NVML or `nvidia-smi` report (`services/hardware.rs`) |
| `concurrency` (default 1), resident instances | request body / lifecycle |

## Formulas

```
weights_gpu   = weight_bytes * weights_gpu_fraction / tp   (if weight_bytes unknown: params * bytes_per_param[dtype|quant])
KV / token / layer (standard attention) = 2 * ceil(kv_heads / tp) * head_dim * kv_bytes
KV / token / layer (MLA)                = (kv_lora_rank + qk_rope_head_dim) * kv_bytes   (replicated on every rank)
KV(ctx, seqs) = per_layer * (n_full * ctx + n_sliding * min(ctx, window)) * seqs
kv_cache_gib  = KV(max_len, concurrency) * kv_gpu_fraction   # smallest pool at which the configured context is servable
activations   = max(0.5, batched_tokens * hidden * 2 * 16 / tp)   (default batched_tokens 8192; 1.5 if hidden unknown)
cuda_graphs   = 1.0   (default; adapter may override)
overhead      = 1.0   (default; CUDA context + NCCL buffers)
total         = weights_gpu + activations + cuda_graphs + overhead + kv_cache_gib
budget        = gpu_total * mem_fraction      (0.90 assumed if the engine gives none; gpu_free instead of gpu_total
                                                 when the adapter reports share_of_free)
verdict       = fits if total <= 0.90*budget ; tight if <= 1.00*budget ; else wont_fit
```

Other rules in the code:

- `max_context_at_current_concurrency` inverts the KV formula on the remaining pool (`budget - fixed`),
  capped at the model's native length; `max_concurrency_at_current_context` is `pool // KV(max_len, 1)`.
- If the adapter reports `kv_cache_fixed_gib` (an explicit KV pool), that value replaces the computed KV.
- If the adapter reports `nonstatic_reserve_frac` (SGLang-style), activations are not charged to the static
  budget but still count against physically free VRAM.
- A configuration that passes the budget check still becomes `wont_fit` if what it must find free exceeds currently
  **free** VRAM: the total (plus non-static activations), or the whole `budget` for an engine that reports
  `claims_share`, plus `outside_share_gib` either way. When stopping resident instances would make room, `fits_if_stop`
  lists them.
- `tp_required` is the smallest tensor-parallel degree at which the same configuration fits.
- Sliding-window layers use `layer_types` or `sliding_window_pattern`; with a window but no pattern every
  layer is assumed windowed (confidence lowered to medium).
- `tp > kv_heads`: KV heads are replicated across ranks (each rank keeps at least one).
- Partial GPU offload (`weights_gpu_fraction`, `kv_gpu_fraction`): see the next section.

## Partial GPU offload (`weights_gpu_fraction`, `kv_gpu_fraction`)

An engine that can leave part of a model in host RAM tells the estimator how much stays on the GPU, through two optional
keys of `memory_model()`:

| Key | Meaning | Effect in `estimate_fit` |
|---|---|---|
| `weights_gpu_fraction` | share (0 to 1) of the weights that the engine keeps on the GPU | `weights_gpu` is multiplied by it, before the division by `tp` |
| `kv_gpu_fraction` | share (0 to 1) of the KV cache that stays on the GPU | `kv_cache_gib` is multiplied by it |

Rules, from `gpu_fraction` and `estimate_fit` in `services/fit.rs`:

- A missing key means `1.0` (everything on the GPU). A value above 1 or below 0 is clamped to that range, and a value that
  is not a finite number counts as `1.0`.
- The GPU is charged only for the share it holds. `notes` says what stays in host RAM: "N% of the weights stay in host RAM
  (partial GPU offload): it fits more, but every token is slower" and "N% of the KV cache stays in host RAM".
- `max_context_at_current_concurrency` and `max_concurrency_at_current_context` use a pool stretched by `1 / kv_gpu_fraction`,
  because the GPU pool only has to hold the GPU-resident share of the cache. `tp_required` applies the same two fractions.
- The estimate does not know how fast the offloaded part is: a smaller fraction fits more and is slower, and the verdict
  does not say by how much.
- The host RAM those shares need is a separate estimate (`host_ram`, below). Only the llama.cpp adapter reports the two keys
  today (from `-ngl` below the layer count, `--no-kv-offload` and `-dev none`, assuming equal layer sizes and a full-attention
  KV cache on every layer) and reports the matching CPU-resident weights and KV in its `host_memory_gib()`. Ollama
  offloads layers when a model does not fit, but the amount is chosen by its scheduler at load time, so it is not modelled.

Status: implemented and unit-tested with fakes (`backend/crates/ec-hub/tests/test_fit_offload.rs`, `backend/crates/ec-hub/tests/llamacpp/test_memory.rs`).
Not compared with a measured llama.cpp launch.

## Host RAM (`host_ram`)

A second, separate estimate covers host memory, because CPU offload and host-side caches move GPU-sized amounts into RAM.
Status: verified with fakes only; not compared against a measured launch.

Inputs come from the adapter's `host_memory_gib()`:

| Component | vLLM | SGLang |
|---|---|---|
| `cpu_weight_offload` | `cpu_offload_gb` x TP x PP (GiB per GPU) | `cpu_offload_gb` x TP (assumed per TP rank: **unconfirmed**) |
| `kv_offload_buffer` | `kv_offloading_size` (already summed across ranks) | not applicable |
| HiCache host pool | not applicable | `hicache_size` x TP, or `hicache_ratio` (default 2.0) x device KV pool x TP (assumed per TP rank: **unconfirmed**); NaN (unknown) if the device pool cannot be computed |
| vLLM `swap_space` | **not modelled**: absent from the pinned parameter catalog | not applicable |

Then `needed = sum(components) + engine_process`, where `engine_process` is a fixed **4 GiB heuristic** (Python plus the CUDA
runtime; roughly 3 GiB was observed on one running SGLang container, so this is not a calibrated figure).

Verdict against current `MemAvailable`: `ok` if needed <= 80 % of available, `tight` if <= 100 %, else `wont_fit`.
`unknown` if any component is NaN (needed is then a lower bound) or host memory cannot be read.

Why only unreclaimable memory counts: `MemAvailable` already treats the page cache as reclaimable, and loading weights streams
through the page cache, so the model's file size is deliberately not added. What is added is memory the kernel cannot take
back under pressure: anonymous memory, shared memory (tmpfs, `/dev/shm`, CUDA IPC buffers) and pinned buffers. If shared
memory on the host is already above 25 % of total, the estimate says so, because that memory is counted as "cached" yet is
not reclaimable.

## Confidence

`high` unless a rule lowered it. Derived weights (params x bytes/param), unknown native length, guessed
activations or the windowed-layer assumption give `medium`. Missing layers, heads or head_dim (KV cannot
be computed), or missing weights, give `low`. If KV cannot be sized and weights alone already exceed the
budget, the verdict is `wont_fit` at medium confidence (a lower bound suffices); otherwise `unknown`.
The estimator degrades rather than inventing inputs.

## Known error sources

1. Activations, CUDA graphs and overhead are heuristics (constants above), not measurements, except the one vLLM
   overhead measurement described above. They matter most for small models and large batch/context settings.
2. Sliding-window KV assumes the engine uses a hybrid allocator. An engine that allocates full length for
   those layers needs more than estimated.
3. Quantized checkpoints: weight size comes from safetensors sizes when available; unquantized layers
   (embeddings, lm_head, routers) count only as far as they are in the files.
4. MoE and multimodal: vision towers and expert-parallel layouts are not modelled beyond total weight bytes / tp.
5. Paged-KV block rounding, prefix cache and fragmentation are not modelled; real usable KV is a little below the pool.
6. Speculative-decoding draft models and LoRA adapters are not added to weights (unverified: no such
   handling found in `fit.rs`).
7. Free VRAM is a snapshot. Another engine, a fine-tuning job or any other GPU process can change it between estimate and launch.
8. Only `tp` is considered; pipeline and context parallelism are not.

## How an engine sizes its share

Engines do not all read `mem_fraction` the same way. Three optional `memory_model()` keys say how, each checked in the
source of the pinned image:

| Key | Engine | What the engine does | Effect in `estimate_fit` |
|---|---|---|---|
| `claims_share` | vLLM 0.27.1 | `worker/utils.py` `request_memory`: refuses to start unless `gpu_memory_utilization x card` is free, then fills that share with weights, activations, CUDA graphs and KV. It checks the share even when `kv_cache_memory_bytes` fixes the KV pool | the free-VRAM check needs the whole `budget` free, not just the total this configuration uses |
| `outside_share_gib` | vLLM 0.27.1 | the CUDA context and NCCL buffers exist before the worker takes its free-memory snapshot (`gpu_worker.init_device`), so they are held on top of the share | never charged against `budget`; added to what must be free |
| `share_of_free` | SGLang 0.5.20 | `kv_cache_configurator._profile_available_bytes`: KV pool = free after weights - free before load x (1 - fraction), that is `fraction x free at launch - weights` | `budget = gpu_free * mem_fraction`. Another engine already on the card shrinks it, which is why start order matters; `fits_if_stop` names the engines whose memory would make it fit |

The other adapters (TensorRT-LLM, llama.cpp, Ollama) report none of these, so the estimator treats them as using only
what the configuration needs. TensorRT-LLM's own rule (KV pool = `free_gpu_memory_fraction` x (device total - peak
non-KV use), `KvCacheCreator._cal_max_memory`) is described in its adapter; the estimator budgets `fraction x total`
for it, which is pessimistic by about (1 - fraction) x weights.

## Measured: memory held outside the pool (vLLM)

One measurement exists. vLLM 0.27.1 on one GPU with `enforce_eager`, a 0.5B-parameter model and
`gpu_memory_utilization` 0.15 on a 95.6 GiB card held **16.9 GiB** (16.86 GiB) against a share of 14.34 GiB (0.15 x 95.6).
That is about **2.5 GiB outside the weights and KV pool**: the CUDA context and library workspaces, memory the engine
does not count against its own share.

What the code does with it (`adapters/vllm.rs`, `memory_model`):

- The vLLM adapter returns `outside_share_gib = 2.5 + 0.5 x (tp - 1)` and `overhead_gib = 0`. The 2.5 is this one
  measurement. The 0.5 GiB per extra tensor-parallel rank (NCCL buffers) is a guess and has not been measured. With
  `claims_share`, a launch at share 0.15 on that card needs 14.34 + 2.5 = 16.84 GiB free, which matches what it held.
- vLLM's CUDA-graph pool (0.75 GiB in the adapter, from 0.59 to 0.89 GiB measured on a Qwen3-class model) is charged
  inside the share: 0.27.1 estimates it and takes it off the KV pool (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` is on by
  default).
- The other adapters keep heuristics, not measurements: SGLang leaves `overhead_gib` at 0 (its activations and graphs
  come out of the non-static share, and its process overhead has not been measured), TensorRT-LLM uses
  `1.0 + 0.5 x (ranks - 1)`, llama.cpp uses the larger of 1 GiB and its fit margin, Ollama uses 1 GiB plus its
  `gpu_overhead`. The estimator's own default is 1 GiB.

What is **not** calibrated:

- One run of one engine version with one small model on one card type says nothing about larger models, other kernels,
  `tp > 1`, a CUDA-graph pool of another size, or any other engine.
- SGLang's memory outside its share (CUDA context, workspaces) has not been measured, so a launch that fits exactly can
  still fail by that margin.

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
   Record the measurement first and add a `test_fit.rs` case with the measured numbers.
5. Repeat at two context lengths to separate KV error (scales with context) from fixed overhead.

No measured comparison table is included because none has been run for this document; the single measurement above is the
only one on record.
