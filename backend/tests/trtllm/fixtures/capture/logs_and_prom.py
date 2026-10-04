"""Fixture material for the metrics and start-up parsers, produced by the image's own code:
 1. Prometheus text of `MetricsCollector` after four synthetic requests (the families `trtllm-serve` exposes with
    `return_perf_metrics: true`);
 2. lines printed through the engine's own logger (the `[date] [TRT-LLM] [I]` prefix is real; the message bodies are
    copied from the engine's sources, the numbers are arbitrary: no model was started);
 3. uvicorn start-up lines (stderr), configured as `OpenAIServer.__call__` does;
 4. a real tqdm bar with the description the weight loader uses.
Sections are separated by lines that start with '##SECTION '."""
import asyncio
import os
import sys
import time

os.environ["TLLM_LOG_LEVEL"] = "info"


def section(name):
    sys.stdout.flush()
    sys.stderr.flush()
    print(f"\n##SECTION {name}", flush=True)


import tqdm  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from prometheus_client import generate_latest  # noqa: E402
from tensorrt_llm.logger import logger  # noqa: E402
from tensorrt_llm.metrics.collector import MetricsCollector  # noqa: E402
from tensorrt_llm.metrics.enums import MetricNames as M  # noqa: E402

collector = MetricsCollector({"model_name": "undefined", "engine_type": "undefined"})
for stop, e2e, ttft, tpot, queue in (("stop", 1.2, 0.08, 0.021, 0.01), ("stop", 3.4, 0.21, 0.024, 0.4),
                                     ("length", 9.8, 0.9, 0.031, 1.7), ("stop", 0.6, 0.04, 0.019, 0.0)):
    collector.log_metrics_dict({"finished_reason": stop, M.E2E: e2e, M.TTFT: ttft, M.TPOT: tpot,
                                M.REQUEST_QUEUE_TIME: queue})
section("prometheus")
print(generate_latest().decode())

section("logger")
logger.info("Downloading model org/model-8B from HuggingFace")
logger.info("Finished downloading model org/model-8B from HuggingFace")
logger.info("Prefetching 14.96GB checkpoint files.")
logger.info("Use 14.96 GB for model weights.")
print("Model init total -- 21.34s")
logger.info("max_seq_len is not specified, using inferred value 131072")
logger.warning("Specified max_seq_len=262144 is larger than what the model can support (131072). "
               "Setting max_seq_len to 131072. ")
logger.info("Memory used after loading model weights (inside torch) in memory usage profiling: 14.96 GiB")
logger.info("Memory used after loading model weights (outside torch) in memory usage profiling: 0.61 GiB")
logger.info("Memory dynamically allocated during inference (inside torch) in memory usage profiling: 2.10 GiB")
logger.info("Memory used outside torch (e.g., NCCL and CUDA graphs) in memory usage profiling: 1.35 GiB")
logger.info("Peak memory during memory usage profiling (torch + non-torch): 19.42 GiB, available KV cache memory "
            "when calculating max tokens: 64.80 GiB, fraction is set 0.9, kv size is 131072. device total memory "
            "94.99 GiB, , tmp kv_mem 1.02 GiB")
logger.info("Estimated max memory in KV cache : 64.80 GiB")
logger.warning("Adjusted max_seq_len to 98304")
logger.info("Running autotuner warmup...")
logger.info("[Autotuner] Autotuning process starts ...")
logger.info("[Autotuner] Autotuning process ends")
logger.info("[Autotuner] Cache size after warmup is 42")
logger.info("Creating CUDA graph instances for 34 batch sizes.")
logger.info("Run generation-only CUDA graph warmup for batch size=128, draft_len=0, max_seq_len=131072")
logger.info("Run generation-only CUDA graph warmup for batch size=1, draft_len=0, max_seq_len=131072")
logger.info("Running torch.compile warmup...")
logger.info("Running piecewise CUDA graph warmup...")
logger.info("Run piecewise CUDA graph warmup for num tokens=2048")
logger.info("Run warmup with 8192 tokens, include 0 generation tokens")
logger.info("iter = 154, global_rank = 0, rank = 0, currank_total_requests = 3/9, host_step_time = 9.8ms, "
            "prev_device_step_time = 9.1ms, timestamp = 2026-10-04 09:00:00, num_scheduled_requests: 3, "
            "states = {'num_ctx_requests': 0, 'num_ctx_tokens': 0, 'num_generation_tokens': 3}")
logger.set_rank(1)
logger.info("Use 7.48 GB for model weights.")
logger.error("Executor creation failed due to insufficient GPU memory.")

section("uvicorn")


async def serve_briefly():
    server = uvicorn.Server(uvicorn.Config(FastAPI(), host="127.0.0.1", port=18999, log_level="info",
                                           timeout_keep_alive=5))
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(1.0)
    server.should_exit = True
    await task


asyncio.run(serve_briefly())

section("tqdm")
bar = tqdm.tqdm(total=4, desc="Loading safetensors weights in parallel", file=sys.stdout)
for _ in range(4):
    time.sleep(0.15)
    bar.update(1)
bar.close()
print()
