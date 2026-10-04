"""One valid sample value for every catalog key, plus the companion parameters a lone key needs to validate.

Shared by the tests and by the one-off validation of the generated launches inside the pinned image (that run is
documented in fixtures/README.md). `test_samples_cover_catalog` fails when a parameter is added without a sample."""
from __future__ import annotations

from typing import Any

SAMPLES: dict[str, Any] = {
    # memory
    "free_gpu_memory_fraction": 0.8, "kv_cache_dtype": "fp8", "kv_cache_max_tokens": 65536,
    "kv_cache_max_gib": 20.0, "kv_host_cache_gib": 8.0, "tokens_per_block": 64,
    "kv_max_attention_window": ["4096", "256"], "mamba_ssm_cache_dtype": "float32",
    # parallelism
    "tensor_parallel_size": 2, "pipeline_parallel_size": 2, "context_parallel_size": 2,
    "moe_expert_parallel_size": 2, "moe_tensor_parallel_size": 2, "moe_cluster_parallel_size": 2,
    "enable_attention_dp": True, "allreduce_strategy": "NCCL", "orchestrator_type": "rpc",
    "nccl_p2p_disable": True,
    # context
    "max_seq_len": 32768, "max_beam_width": 2, "media_io_kwargs": {"video": {"num_frames": 8}},
    # model loading / quantization
    "revision": "main", "tokenizer": "example-org/tokenizer-repo", "tokenizer_mode": "slow",
    "custom_tokenizer": "deepseek_v32", "trust_remote_code": True, "moe_backend": "CUTLASS",
    "moe_max_num_tokens": 4096, "nvfp4_gemm_backends": ["cutlass", "cuda_core"],
    # scheduling
    "max_batch_size": 256, "max_num_tokens": 4096, "enable_chunked_prefill": True,
    "enable_block_reuse": False, "enable_partial_reuse": False,
    "capacity_scheduler_policy": "MAX_UTILIZATION", "context_chunking_policy": "EQUAL_PROGRESS",
    "disable_overlap_scheduler": True, "stream_interval": 4, "batch_wait_timeout_ms": 5.0,
    "batch_wait_timeout_iters": 2, "batch_wait_max_tokens_ratio": 0.5, "num_postprocess_workers": 2,
    "sampler_type": "TorchSampler",
    # tools and reasoning
    "tool_parser": "qwen3_coder", "reasoning_parser": "qwen3",
    "chat_template": "{{ messages[0]['content'] }}", "guided_decoding_backend": "xgrammar",
    # performance
    "enable_cuda_graphs": False, "cuda_graph_max_batch_size": 64, "cuda_graph_batch_sizes": ["1", "2", "4", "8"],
    "cuda_graph_padding": True, "attention_backend": "FLASHINFER", "enable_torch_compile": True,
    "torch_compile_piecewise_cuda_graph": True, "torch_compile_inductor": True, "enable_autotuner": False,
    # speculative decoding
    "spec_decoding_type": "NGram", "spec_max_draft_len": 4, "spec_draft_model": "example-org/draft-model-1B",
    "spec_num_nextn_predict_layers": 2, "spec_ngram_max_matching_size": 3, "spec_eagle3_one_model": False,
    "spec_max_concurrency": 8,
    "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 2},
    # network / observability
    "log_level": "debug", "otlp_traces_endpoint": "http://otel-collector:4317", "enable_iter_perf_stats": False,
    "iter_stats_max_entries": 250, "print_iter_log": True,
    # advanced
    "extra_llm_options": {"enable_min_latency": True, "garbage_collection_gen0_threshold": 10000},
    "image": "nvcr.io/nvidia/tensorrt-llm/release:1.2.1",
}

# What a key needs next to it so the combination validates (checked by test_each_sample_validates).
COMPANIONS: dict[str, dict[str, Any]] = {
    "moe_expert_parallel_size": {"tensor_parallel_size": 2},
    "moe_tensor_parallel_size": {"tensor_parallel_size": 2},
    "moe_cluster_parallel_size": {"tensor_parallel_size": 2},
    "torch_compile_piecewise_cuda_graph": {"enable_torch_compile": True},
    "torch_compile_inductor": {"enable_torch_compile": True},
    "spec_decoding_type": {"spec_max_draft_len": 4, "spec_ngram_max_matching_size": 3},
    "spec_max_draft_len": {"spec_decoding_type": "NGram", "spec_ngram_max_matching_size": 3},
    "spec_draft_model": {"spec_decoding_type": "DraftTarget", "spec_max_draft_len": 3},
    "spec_num_nextn_predict_layers": {"spec_decoding_type": "MTP"},
    "spec_ngram_max_matching_size": {"spec_decoding_type": "NGram", "spec_max_draft_len": 4},
    "spec_eagle3_one_model": {"spec_decoding_type": "Eagle", "spec_draft_model": "example-org/eagle3-head",
                              "spec_max_draft_len": 3},
    "spec_max_concurrency": {"spec_decoding_type": "MTP"},
}

# Whole-launch scenarios (each validates). Used for the golden-launch tests and the real-image validation.
SCENARIOS: dict[str, dict[str, Any]] = {
    "empty": {},
    "mtp": {"spec_decoding_type": "MTP", "spec_num_nextn_predict_layers": 2, "spec_max_concurrency": 16},
    "eagle3": {"spec_decoding_type": "Eagle", "spec_draft_model": "example-org/eagle3-head",
               "spec_max_draft_len": 3, "spec_eagle3_one_model": True},
    "ngram": {"spec_decoding_type": "NGram", "spec_max_draft_len": 4, "spec_ngram_max_matching_size": 3},
    "draft_target": {"spec_decoding_type": "DraftTarget", "spec_draft_model": "example-org/draft-model-1B",
                     "spec_max_draft_len": 3},
    "auto_spec": {"spec_decoding_type": "AUTO"},
    "spec_json": {"speculative_config": {"decoding_type": "NGram", "max_draft_len": 2,
                                         "max_matching_ngram_size": 2}},
    "graphs_off": {"enable_cuda_graphs": False},
    "graphs_sized": {"cuda_graph_batch_sizes": ["1", "2", "4", "8", "16"], "cuda_graph_padding": True},
    "compile": {"enable_torch_compile": True, "torch_compile_piecewise_cuda_graph": True,
                "torch_compile_inductor": True},
    "scheduler": {"capacity_scheduler_policy": "MAX_UTILIZATION", "context_chunking_policy": "EQUAL_PROGRESS",
                  "enable_chunked_prefill": True},
    "kv": {"kv_cache_dtype": "fp8", "enable_block_reuse": False, "kv_cache_max_gib": 12.5,
           "kv_host_cache_gib": 4.0, "tokens_per_block": 64, "kv_max_attention_window": ["4096", "512"]},
    "moe_tp4": {"tensor_parallel_size": 4, "moe_expert_parallel_size": 2, "moe_tensor_parallel_size": 2,
                "moe_backend": "CUTLASS", "moe_max_num_tokens": 8192},
    "parsers": {"tool_parser": "qwen3", "reasoning_parser": "qwen3", "guided_decoding_backend": "llguidance",
                "chat_template": "{{ messages }}"},
    "extra": {"extra_llm_options": {"garbage_collection_gen0_threshold": 10000, "enable_min_latency": True}},
    "stats_history": {"enable_iter_perf_stats": True, "iter_stats_max_entries": 250, "print_iter_log": True},
    "everything_on_cli": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1, "max_seq_len": 65536,
                          "max_batch_size": 64, "max_num_tokens": 8192, "free_gpu_memory_fraction": 0.85,
                          "num_postprocess_workers": 2, "trust_remote_code": True, "revision": "main",
                          "tokenizer": "example-org/tokenizer-repo", "log_level": "info",
                          "otlp_traces_endpoint": "http://otel-collector:4317", "enable_chunked_prefill": True,
                          "max_beam_width": 1,
                          "media_io_kwargs": {"video": {"num_frames": 8}}, "reasoning_parser": "deepseek-r1",
                          "tool_parser": "deepseek_v3", "chat_template": "{{ messages }}"},
}


def sample_params(key: str) -> dict[str, Any]:
    """The key with its sample value and the companions it needs."""
    return {key: SAMPLES[key], **COMPANIONS.get(key, {})}
