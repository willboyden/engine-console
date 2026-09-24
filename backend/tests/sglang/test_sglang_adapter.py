"""Offline tests for the SGLang adapter (imports only adapters.*; no docker, no network)."""
from __future__ import annotations

import pytest

from engine_console.adapters.base import Hardware, ModelInfo
from engine_console.adapters.sglang import SglangAdapter
from engine_console.adapters.sglang_parsers import histogram_quantile

A = SglangAdapter()


def hw(n: int = 1, cc: str = "12.0") -> Hardware:
    return Hardware(gpu_ids=list(range(n)), gpu_uuids=[f"GPU-{i}" for i in range(n)],
                    gpu_names=["RTX PRO 6000"] * n, gpu_total_gib=[95.0] * n,
                    gpu_free_gib=[94.0] * n, compute_capability=cc, host_ram_gib=246.0)


def mi(repo: str = "Qwen/Qwen3.6-35B-A3B", **kw: object) -> ModelInfo:
    return ModelInfo(repo_id=repo, **kw)


def codes(model: ModelInfo, params: dict[str, object], h: Hardware | None = None) -> dict[str, str]:
    return {c.code: c.level for c in A.compatibility(model, params, h or hw())}


# ---- catalog ----
def test_catalog_size_and_uniqueness() -> None:
    cat = A.param_catalog()
    assert len(cat) >= 60
    assert len({p.key for p in cat}) == len(cat)
    assert len({p.flag for p in cat}) == len(cat)


def test_catalog_shape() -> None:
    for p in A.param_catalog():
        assert p.flag.startswith("--") and p.docs_url and p.help
        if p.type == "enum":
            assert p.choices
            assert p.default is None or p.default in p.choices
        if p.min is not None and p.max is not None:
            assert p.min < p.max
    keys = {p.key for p in A.param_catalog()}
    for k in ("mem_fraction_static", "tool_call_parser", "enable_metrics", "tp_size", "quantization",
              "speculative_algorithm", "enable_hierarchical_cache", "enable_lora"):
        assert k in keys


def test_catalog_returns_copies() -> None:
    A.param_catalog()[0].label = "mutated"
    assert A.param_catalog()[0].label != "mutated"


# ---- presets / validate ----
@pytest.mark.parametrize("name", ["balanced", "max-throughput", "low-latency", "long-context", "tool-agent", "reasoning"])
def test_presets_validate(name: str) -> None:
    assert A.validate(A.presets()[name]) == []


def test_validate_errors() -> None:
    assert "unknown parameter 'nope'" in A.validate({"nope": 1})
    assert A.validate({"mem_fraction_static": 1.5})
    assert A.validate({"schedule_policy": "bogus"})
    assert A.validate({"tp_size": True})
    assert A.validate({"tp_size": 1.5})
    assert A.validate({"api_key": "--evil"})
    assert A.validate({"speculative_algorithm": "EAGLE3"})
    assert A.validate({"speculative_num_steps": 3})
    assert A.validate({"enable_dp_attention": True, "tp_size": 2, "dp_size": 1})
    assert A.validate({"tp_size": 4, "ep_size": 3})
    assert A.validate({"enable_hierarchical_cache": True, "disable_radix_cache": True})
    assert A.validate({"json_model_override_args": "{bad"})
    assert A.validate({"speculative_algorithm": "NEXTN", "speculative_num_steps": 3}) == []


# ---- build_launch ----
def test_build_launch_golden() -> None:
    spec = A.build_launch("Qwen/Qwen3.6-35B-A3B", {
        "quantization": "fp8", "context_length": 262144, "mem_fraction_static": 0.85,
        "tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3",
        "disable_radix_cache": False, "enable_mixed_chunk": True, "schedule_conservativeness": 1,
    }, hw(), served_name="qwen", hf_cache_container_path="/root/.cache/huggingface")
    assert spec.image.startswith("lmsysorg/sglang:v0.5.14") and ":latest" not in spec.image
    a = spec.argv
    assert a[:12] == ["python3", "-m", "sglang.launch_server", "--model-path", "Qwen/Qwen3.6-35B-A3B",
                      "--host", "0.0.0.0", "--port", "30000", "--served-model-name", "qwen", "--enable-metrics"]
    pairs = dict(zip(a[12:], a[13:], strict=False))
    assert pairs["--quantization"] == "fp8" and pairs["--mem-fraction-static"] == "0.85"
    assert pairs["--tool-call-parser"] == "qwen3_coder" and pairs["--context-length"] == "262144"
    assert "--enable-mixed-chunk" in a and "--disable-radix-cache" not in a
    assert spec.env == {"HF_HOME": "/root/.cache/huggingface"}
    assert (spec.container_port, spec.health_path, spec.metrics_path) == (30000, "/health", "/metrics")


def test_metrics_forced_and_served_override() -> None:
    spec = A.build_launch("m", {"enable_metrics": False, "served_model_name": "mine"}, hw(),
                          served_name="x", hf_cache_container_path="/c")
    assert spec.argv.count("--enable-metrics") == 1
    assert spec.argv[spec.argv.index("--served-model-name") + 1] == "mine"


def test_build_launch_list_json_and_rejects() -> None:
    spec = A.build_launch("m", {"lora_paths": ["a=/x", "b=/y"], "json_model_override_args": {"k": 1},
                                "cuda_graph_max_bs_decode": 64}, hw(), served_name="s", hf_cache_container_path="/c")
    i = spec.argv.index("--lora-paths")
    assert spec.argv[i + 1:i + 3] == ["a=/x", "b=/y"]
    assert '{"k":1}' in spec.argv and "--cuda-graph-max-bs-decode" in spec.argv
    with pytest.raises(ValueError):
        A.build_launch("--x", {}, hw(), served_name="s", hf_cache_container_path="/c")
    with pytest.raises(ValueError):
        A.build_launch("m", {"bad": 1}, hw(), served_name="s", hf_cache_container_path="/c")


# ---- compatibility ----
def test_compat_gpus_and_mem_fraction() -> None:
    assert codes(mi(), {"tp_size": 2}, hw(1))["gpus_insufficient"] == "block"
    assert "gpus_insufficient" not in codes(mi(), {"tp_size": 2}, hw(2))
    assert codes(mi(), {"mem_fraction_static": 0.95})["mem_fraction_high"] == "warn"
    assert "mem_fraction_high" not in codes(mi(), {"mem_fraction_static": 0.85})


def test_compat_quant_rules() -> None:
    assert codes(mi(quantization="nvfp4", is_moe=True), {})["nvfp4_ok_sglang"] == "ok"
    assert codes(mi("openai/gpt-oss-120b", quantization="mxfp4", is_moe=True), {})["mxfp4_gptoss_engine"] == "warn"
    assert codes(mi("x/moe", quantization="mxfp4", is_moe=True), {})["mxfp4_moe_sm120"] == "block"
    assert codes(mi(quantization="nvfp4"), {"quantization": "fp8"})["quant_override_prequantized"] == "warn"
    assert codes(mi(), {"quantization": "modelopt_fp4"})["modelopt_unquantized"] == "warn"


def test_compat_attention_backend() -> None:
    assert codes(mi(), {"attention_backend": "fa3"})["attention_backend_hopper_only"] == "block"
    assert codes(mi(), {"attention_backend": "trtllm_mla"})["attention_backend_dc_blackwell"] == "warn"
    assert not any(c.startswith("attention_backend") for c in codes(mi(), {"attention_backend": "flashinfer"}))
    assert "attention_backend_hopper_only" not in codes(mi(), {"attention_backend": "fa3"}, hw(1, "9.0"))


def test_compat_tool_parser_dialect() -> None:
    assert codes(mi(), {"tool_call_parser": "qwen"})["tool_parser_dialect_mismatch"] == "warn"
    assert "tool_parser_dialect_mismatch" not in codes(mi(), {"tool_call_parser": "qwen3_coder"})
    assert codes(mi("Qwen/Qwen3-32B"), {"tool_call_parser": "qwen3_coder"})["tool_parser_dialect_mismatch"] == "warn"
    assert "tool_parser_dialect_mismatch" not in codes(mi("Qwen/Qwen3-32B"), {"tool_call_parser": "qwen25"})
    assert codes(mi("openai/gpt-oss-20b"), {"tool_call_parser": "hermes"})["tool_parser_dialect_mismatch"] == "warn"
    assert codes(mi(), {})["no_tool_parser"] == "warn"


def test_compat_misc() -> None:
    m = mi(is_moe=True)
    assert codes(m, {"enable_dp_attention": True, "tp_size": 2, "dp_size": 1}, hw(2))["dp_attention_dp_ne_tp"] == "block"
    assert codes(mi(), {"enable_dp_attention": True, "tp_size": 2, "dp_size": 2}, hw(2))["dp_attention_unsupported_arch"] == "warn"
    assert codes(mi(max_position_embeddings=32768), {"context_length": 65536})["context_beyond_native"] == "warn"
    assert codes(mi(gated=True), {})["gated_model"] == "warn"
    assert codes(mi(), {"enable_metrics": False})["metrics_forced"] == "warn"
    assert codes(mi(), {"api_key": "s3cret"})["api_key_in_argv"] == "warn"


# ---- memory model ----
def test_memory_model() -> None:
    m = A.memory_model(mi(max_position_embeddings=262144),
                       {"mem_fraction_static": 0.85, "kv_cache_dtype": "fp8_e4m3", "tp_size": 2,
                        "max_running_requests": 8})
    assert m["mem_fraction"] == 0.85 and m["kv_bytes_per_elem"] == 1.0 and m["tp"] == 2.0
    assert m["max_len"] == 262144 and m["max_seqs"] == 8 and m["overhead_gib"] == 0.0
    assert abs(m["nonstatic_reserve_frac"] - 0.15) < 1e-9
    d = A.memory_model(mi(), {})
    assert d["mem_fraction"] == 0.85 and d["kv_bytes_per_elem"] == 2.0 and d["mem_fraction_is_default"] == 1.0


# ---- metrics ----
PROM = '''# HELP sglang:num_running_reqs The number of running requests
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="qwen",engine_type="unified",tp_rank="0",pp_rank="0"} 3.0
sglang:num_queue_reqs{model_name="qwen",tp_rank="0"} 5.0
sglang:token_usage{model_name="qwen",tp_rank="0"} 0.42
sglang:cache_hit_rate{model_name="qwen",tp_rank="0"} 0.75
sglang:gen_throughput{model_name="qwen",tp_rank="0"} 812.5
sglang:spec_accept_length{model_name="qwen",tp_rank="0"} 2.6
sglang:spec_accept_rate{model_name="qwen",tp_rank="0"} 0.65
sglang:prompt_tokens_total{model_name="qwen"} 120000.0
sglang:generation_tokens_total{model_name="qwen"} 45000.0
sglang:num_retracted_requests_total{model_name="qwen"} 2.0
sglang:time_to_first_token_seconds_bucket{le="0.1",model_name="qwen"} 10.0
sglang:time_to_first_token_seconds_bucket{le="0.5",model_name="qwen"} 90.0
sglang:time_to_first_token_seconds_bucket{le="1.0",model_name="qwen"} 100.0
sglang:time_to_first_token_seconds_bucket{le="+Inf",model_name="qwen"} 100.0
sglang:time_to_first_token_seconds_sum{model_name="qwen"} 40.0
sglang:time_to_first_token_seconds_count{model_name="qwen"} 100.0
sglang:inter_token_latency_seconds_bucket{le="0.01",model_name="qwen"} 50.0
sglang:inter_token_latency_seconds_bucket{le="0.02",model_name="qwen"} 100.0
sglang:inter_token_latency_seconds_bucket{le="+Inf",model_name="qwen"} 100.0
sglang:e2e_request_latency_seconds_bucket{le="1.0",model_name="qwen"} 20.0
sglang:e2e_request_latency_seconds_bucket{le="5.0",model_name="qwen"} 100.0
sglang:e2e_request_latency_seconds_bucket{le="+Inf",model_name="qwen"} 100.0
'''


def test_parse_metrics() -> None:
    m = A.parse_metrics(PROM)
    assert m["requests_running"] == 3 and m["requests_waiting"] == 5
    assert m["kv_cache_usage_pct"] == pytest.approx(42.0) and m["prefix_cache_hit_pct"] == pytest.approx(75.0)
    assert m["generation_tps"] == 812.5 and m["prompt_tokens_total"] == 120000
    assert m["generation_tokens_total"] == 45000 and m["preemptions_total"] == 2
    assert m["spec_decode_accept_pct"] == pytest.approx(65.0) and m["spec_accept_length"] == 2.6
    assert m["ttft_p50_s"] == pytest.approx(0.1 + 0.4 * (50 - 10) / 80)
    assert 0.5 < m["ttft_p95_s"] < 1.0
    assert m["itl_p50_s"] == pytest.approx(0.01)
    assert m["e2e_p50_s"] == pytest.approx(1.0 + 4.0 * (50 - 20) / 80)
    assert "prompt_tps" not in m


def test_parse_metrics_tolerance() -> None:
    assert A.parse_metrics("") == {}
    assert A.parse_metrics("garbage line\nsglang:token_usage NaN\n") == {}
    alt = A.parse_metrics('sglang_num_running_reqs{tp_rank="0"} 2\nsglang_prompt_tokens 9\n')
    assert alt == {"requests_running": 2, "prompt_tokens_total": 9}
    two = A.parse_metrics('sglang:num_running_reqs{dp_rank="0"} 2\nsglang:num_running_reqs{dp_rank="1"} 3\n'
                          'sglang:token_usage{dp_rank="0"} 0.2\nsglang:token_usage{dp_rank="1"} 0.4\n')
    assert two["requests_running"] == 5 and two["kv_cache_usage_pct"] == pytest.approx(30.0)
    fb = A.parse_metrics("sglang:kv_used_tokens 500\nsglang:max_total_num_tokens 1000\n")
    assert fb["kv_cache_usage_pct"] == pytest.approx(50.0)
    assert set(A.parse_metrics('sglang:num_running_reqs 0\n')) == {"requests_running"}


def test_histogram_quantile() -> None:
    assert histogram_quantile(0.5, []) is None
    assert histogram_quantile(0.5, [(1.0, 0.0), (float("inf"), 0.0)]) is None
    assert histogram_quantile(0.99, [(1.0, 5.0), (float("inf"), 10.0)]) == 1.0


# ---- startup log ----
@pytest.mark.parametrize("line,expected", [
    ("[2026-09-23 10:00:00] Load weight begin. avail mem=93.10 GB", {"phase": "loading_weights", "pct": 0.0}),
    ("Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:10<00:10,  5.0s/it]",
     {"phase": "loading_weights", "pct": 50.0}),
    ("Multi-thread loading shards:  25%|██▌       | 1/4 [00:02<00:06]", {"phase": "loading_weights", "pct": 25.0}),
    ("Load weight end. type=Qwen3MoeForCausalLM, dtype=torch.bfloat16, avail mem=58.70 GB, mem usage=34.50 GB.",
     {"phase": "loading_weights", "pct": 100.0, "weights_gib": 34.5}),
    ("KV Cache is allocated. #tokens: 1234567, K size: 20.00 GB, V size: 20.00 GB",
     {"phase": "allocating_kv", "kv_cache_tokens": 1234567}),
    ("max_total_num_tokens=1234567, chunked_prefill_size=8192, max_prefill_tokens=16384, "
     "max_running_requests=2048, context_len=262144, available_gpu_mem=9.5 GB",
     {"phase": "allocating_kv", "kv_cache_tokens": 1234567, "max_running_requests": 2048, "context_len": 262144}),
    ("Capture cuda graph begin. This can take up to several minutes. avail mem=9.50 GB",
     {"phase": "capturing_graphs", "pct": 0.0}),
    ("Capturing batches (bs=16 avail_mem=9.10 GB):  40%|████      | 4/10 [00:03<00:04]",
     {"phase": "capturing_graphs", "batch_size": 16, "pct": 40.0}),
    ("Capture cuda graph end. Time elapsed: 12.31 s. mem usage=1.20 GB. avail mem=8.30 GB.",
     {"phase": "capturing_graphs", "pct": 100.0, "cuda_graph_gib": 1.2}),
    ("INFO:     The server is fired up and ready to roll!", {"phase": "ready", "pct": 100.0}),
    ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB", {"phase": "failed", "error": "oom"}),
])
def test_startup_log(line: str, expected: dict[str, object]) -> None:
    assert A.parse_startup_log(line) == expected


def test_startup_log_noise_and_tqdm_cr() -> None:
    assert A.parse_startup_log("") is None
    assert A.parse_startup_log("Prefill batch, #new-seq: 1, #new-token: 12") is None
    r = A.parse_startup_log("Loading safetensors checkpoint shards:   0% Completed | 0/4\rLoading safetensors "
                            "checkpoint shards:  75% Completed | 3/4 [00:1<00:1]")
    assert r == {"phase": "loading_weights", "pct": 75.0}
    assert A.parse_startup_log("\x1b[32mThe server is fired up and ready to roll!\x1b[0m") == {"phase": "ready", "pct": 100.0}
