# TensorRT-LLM test fixtures

Everything here describes the pinned image `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` (linux/amd64, manifest digest
`sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9`, TensorRT-LLM 1.2.1, source commit
`376f7e1bd8ed543f75014309e3fd4b237e9b0e73` from the image's own `TRT_LLM_GIT_COMMIT`). No model was ever started to make
them, so nothing below is the output of a serving engine under load. The table says exactly what each file is.

| file | what it is | real or synthetic |
|---|---|---|
| `serve_help_1.2.1.txt` | stdout of `trtllm-serve serve --help` | captured, banner cut |
| `top_help_1.2.1.txt` | stdout of `trtllm-serve --help` (the sub-commands) | captured, banner cut |
| `llm_args_1.2.1.json` | field tables of the LLM API argument classes (`TorchLlmArgs`, `KvCacheConfig`, `CudaGraphConfig`, ...), the click options of `trtllm-serve serve`, the parser registries and the PyTorch backend's model registry | introspected inside the image (`capture/introspect.py`), descriptions dropped |
| `iter_stats_1.2.1.json` | what `GET /metrics` returns per engine step | key names and layout from the engine's own C++ serializer; every value is an arbitrary test number |
| `collector_1.2.1.prom` | what `GET /prometheus/metrics` carries with `return_perf_metrics: true` | the engine's own `MetricsCollector` after four invented requests |
| `infer_max_seq_len_1.2.1.json` | what the engine's own `infer_max_seq_len` returns for 15 model configs | computed by the engine on a stand-in config object (`capture/infer_max_seq_len.py`); no weights |
| `startup_1.2.1.log` | start-up log lines | mixed, see below |

## `startup_1.2.1.log`

* Real: the `[MM/DD/YYYY-HH:MM:SS] [TRT-LLM] [I]` line prefix (printed by the engine's own logger), the four uvicorn
  lines (`Application startup complete.` is the ready marker), and the tqdm progress bar with the description the
  weight loader uses.
* Copied from the engine's sources, not captured: the message bodies (`Use ... GB for model weights.`,
  `Estimated max memory in KV cache : ...`, `Run generation-only CUDA graph warmup for batch size=...`, and so on) with
  invented numbers. `adapters/trtllm_parse.py` names the source file of each pattern.
* Not covered by any fixture: the C++ line `[MemUsageChange] Allocated ... GiB for max tokens in paged KV cache (...)`.
  Its text comes from the compiled library's string table; the `[TensorRT-LLM][INFO]` prefix that precedes it in a real
  log is recalled, not observed, which is why the parser matches the message only. The failure texts used in
  `test_trtllm_parse.py` are likewise copied from source strings.

## Regenerating

All capture scripts run inside the image with no network, every capability dropped and exactly one GPU attached
(the package refuses to import without a CUDA driver library; the scripts allocate no GPU memory):

```
capture/run.sh introspect.py   > raw/introspect.txt
capture/run.sh iterstats.py    > raw/iterstats.txt
capture/run.sh logs_and_prom.py > raw/logs.txt 2> raw/logs.err
docker run --rm --network none --cap-drop ALL --gpus device=<uuid> nvcr.io/nvidia/tensorrt-llm/release:1.2.1 \
    trtllm-serve serve --help > raw/serve_help.txt
docker run --rm --network none --cap-drop ALL --gpus device=<uuid> nvcr.io/nvidia/tensorrt-llm/release:1.2.1 \
    trtllm-serve --help > raw/top_help.txt
python capture/assemble.py raw .
capture/run.sh infer_max_seq_len.py > raw/infer.txt     # then keep the JSON after the banner as infer_max_seq_len_1.2.1.json
```

Running without the image's entrypoint (`--entrypoint trtllm-serve`) fails: TensorRT's libraries are found only through
`LD_LIBRARY_PATH` entries that NVIDIA's `nvidia_entrypoint.sh` sets up.

`introspect.py` and `iterstats.py` are deterministic: re-running them and `assemble.py` reproduces the committed
`llm_args_1.2.1.json` and `iter_stats_1.2.1.json` byte for byte. The log and Prometheus captures differ in timestamps
and creation times only.

## How long one `/metrics` read takes

`capture/iterresult_timing.py` runs the engine's own `IterationResult` class, the object behind the `/metrics`
handler (`async for stat in llm.get_stats_async(2)`), without a server or a model. Its output in the pinned image:
a read with three statistics queued returns after 2.0 s, a second read with no new request returns at once, and after
`mark_undone()` (which every `get_stats_async` call performs) an empty queue again takes 2.0 s. So one `GET /metrics`
always takes at least two seconds, and a scraper with a read timeout of two seconds or less never gets an answer.

## Behaviors the adapter relies on, checked with the engine's own functions

`capture/semantics.py` calls `get_llm_args`, `update_llm_args_with_extra_dict`, `get_attention_backend` and
`load_chat_template` of the pinned image (no model). Its output, summarized:

* `kv_cache_config` from the options file is merged into the CLI's value: `--free_gpu_memory_fraction 0.85` plus
  `enable_block_reuse: false` yields both.
* `scheduler_config` from the options file replaces the CLI's `SchedulerConfig` object as a whole (the CLI's dynamic
  batch settings are gone), which is why the adapter restates them whenever it writes the block.
* An attention backend name outside `TRTLLM`, `FLASHINFER`, `VANILLA`, `FLASHINFER_STAR_ATTENTION` (also a lower-case
  `trtllm`) silently becomes `TrtllmAttention`. FlashInfer is installed in the image.
* A chat template is opened as a file first; text containing `{`, `}` or a newline is then used as the template, any
  other string that is not an openable file stops the engine with a `ValueError`.

## Offline model resolution from a read-only cache

The adapter mounts the host's Hugging Face cache read-only, sets `HF_HOME` and `HF_HUB_OFFLINE=1`, and passes the repo
id as the model. `capture/make_hf_cache.py` builds a one-file cache in the console's layout and `capture/download_probe.py`
calls the engine's own `download_hf_model` on it (no model is loaded):

```
python capture/make_hf_cache.py <scratch>/hfcache
docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges:true --gpus device=<uuid> \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -v <scratch>/hfcache:/hf:ro -v <capture dir>:/work:ro \
    nvcr.io/nvidia/tensorrt-llm/release:1.2.1 python3 /work/download_probe.py
```

In the pinned image the repo id resolved to the `snapshots/<sha>` folder, the cache stayed unwritable, and a repo id
that is not in the cache raised `LocalEntryNotFoundError: Cannot find an appropriate cached snapshot folder ...`
(the text `trtllm_parse.py` maps to the `model_not_in_cache` error).

## MPI inside the node's container limits

Tensor and pipeline parallelism start MPI workers inside the container. `capture/mpi_spawn.py` spawns two workers with
`mpi4py` (`MPIPoolExecutor`, as TensorRT-LLM's `MpiPoolSession` does) and moves a 16 MiB payload between them, with no
model and no GPU. In the pinned image it succeeded under `--network none --cap-drop ALL --security-opt
no-new-privileges:true --pids-limit 4096`, both with a private IPC namespace and with `--ipc=host`. This says nothing
about NCCL or CUDA IPC between two GPUs, which was not tried.

## Validating generated launches against the real image

`capture/make_cases.py` writes one launch (argv plus generated engine-options YAML) per catalog key, per scenario in
`samples.py` and per preset. `capture/validate_launch.py` replays them inside the image: the real click command parses
the argv, then the steps of `serve()` run (`get_llm_args`, the YAML merge, strict `TorchLlmArgs` validation, `Mapping`
validation) without loading a model or a tokenizer.

```
# in the Python reference tree (the last Python release: `node backend/parity/lib/python-release.mjs` prints it)
cd "$(node backend/parity/lib/python-release.mjs)/backend"
.venv/bin/python tests/trtllm/fixtures/capture/make_cases.py <scratch>/cases.json
cp tests/trtllm/fixtures/capture/validate_launch.py <scratch>/
docker run --rm --network none --cap-drop ALL --gpus device=<uuid> -v <scratch>:/work:ro \
    nvcr.io/nvidia/tensorrt-llm/release:1.2.1 python3 /work/validate_launch.py
```

The last run of this check covered 94 launches (every catalog key, every scenario, every preset) and all 94 passed.
The unit tests in `test_trtllm_launch.py` are the offline twin: they walk the same YAML against `llm_args_1.2.1.json`.
Re-run the image check after any change to the catalog or to the YAML generation.
