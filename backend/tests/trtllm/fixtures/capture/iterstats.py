"""Build IterationStats objects the way the PyTorch executor fills them (`py_executor._update_iter_stats`) and
serialise them with the engine's own `BaseWorker._stats_serializer`. Key names and layout are the engine's; every
VALUE is an arbitrary test number. Prints one JSON document."""
import json
import sys

from tensorrt_llm.bindings.executor import (
    InflightBatchingStats,
    IterationStats,
    KvCacheStats,
    RequestStats,
    SpecDecodingStats,
    StaticBatchingStats,
)
from tensorrt_llm.executor.base_worker import BaseWorker


def make(spec: bool, req: bool, iter_no: int):
    s = IterationStats()
    s.timestamp = "10-04-2026 09:00:00.123456"
    s.num_new_active_requests = 1
    s.num_active_requests = 3
    s.new_active_requests_queue_latency_ms = 0.25
    s.inflight_batching_stats = InflightBatchingStats()
    s.static_batching_stats = StaticBatchingStats()
    if spec:
        s.specdec_stats = SpecDecodingStats()
        s.specdec_stats.num_draft_tokens = 8
        s.specdec_stats.num_accepted_tokens = 5
        s.specdec_stats.num_requests_with_draft_tokens = 2
        s.specdec_stats.acceptance_length = 3.5
        s.specdec_stats.iter_latency_ms = 1.5
        s.specdec_stats.draft_overhead = 0.1
    s.iter_latency_ms = 12.5
    s.num_queued_requests = 2
    s.num_completed_requests = 1
    s.max_num_active_requests = 64
    s.gpu_mem_usage = 80 * 1024 ** 3
    s.cpu_mem_usage = 0
    s.pinned_mem_usage = 0
    s.iter = iter_no
    kv = KvCacheStats()
    kv.max_num_blocks = 1000
    kv.free_num_blocks = 750
    kv.used_num_blocks = 250
    kv.tokens_per_block = 32
    kv.alloc_total_blocks = 400
    kv.alloc_new_blocks = 300
    kv.reused_blocks = 100
    kv.missed_blocks = 300
    kv.cache_hit_rate = 0.25
    s.kv_cache_stats = kv
    ib = s.inflight_batching_stats
    ib.num_scheduled_requests = 3
    ib.num_context_requests = 1
    ib.num_gen_requests = 2
    ib.num_paused_requests = 0
    ib.avg_num_decoded_tokens_per_iter = 0
    ib.micro_batch_id = 0
    reqs = None
    if req:
        r = RequestStats()
        r.id = 7
        r.context_prefill_position = 128
        r.num_generated_tokens = 5
        r.avg_num_decoded_tokens_per_iter = 1.0
        r.scheduled = True
        reqs = [r]
    return s, reqs


out = {"plain": json.loads(BaseWorker._stats_serializer(make(False, False, 1))),
       "with_spec_and_requests": json.loads(BaseWorker._stats_serializer(make(True, True, 2)))}
json.dump(out, sys.stdout, indent=1, sort_keys=True)
