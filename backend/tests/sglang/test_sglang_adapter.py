"""Offline tests for the SGLang adapter (imports only adapters.*; no docker, no network)."""
from __future__ import annotations

import re

import pytest

from engine_console.adapters.base import MODEL_MOUNT, Hardware, LaunchContext, ModelInfo
from engine_console.adapters.sglang import SglangAdapter
from engine_console.adapters.sglang_params import DEFAULT_IMAGE
from engine_console.adapters.sglang_parsers import histogram_quantile

A = SglangAdapter()
UNITS = {"GiB", "GB", "MiB", "bytes", "tokens", "requests", "GPUs", "s"}


def hw(n: int = 1, cc: str = "12.0", free: list[float] | None = None) -> Hardware:
    return Hardware(gpu_ids=list(range(n)), gpu_uuids=[f"GPU-{i}" for i in range(n)],
                    gpu_names=["RTX PRO 6000"] * n, gpu_total_gib=[95.0] * n,
                    gpu_free_gib=free if free is not None else [94.0] * n, compute_capability=cc, host_ram_gib=246.0)


def mi(repo: str = "Qwen/Qwen3.6-35B-A3B", **kw: object) -> ModelInfo:
    return ModelInfo(repo_id=repo, **kw)


def codes(model: ModelInfo, params: dict[str, object], h: Hardware | None = None) -> dict[str, str]:
    return {c.code: c.level for c in A.compatibility(model, params, h or hw())}


# ---- catalog ----
def test_catalog_size_and_uniqueness() -> None:
    cat = A.param_catalog()
    assert len(cat) >= 100
    assert len({p.key for p in cat}) == len(cat)
    assert len({p.flag for p in cat}) == len(cat)


def test_catalog_shape() -> None:
    for p in A.param_catalog():
        assert re.fullmatch(r"--[a-z0-9][a-z0-9-]*|env:[A-Z0-9_]+", p.flag) and p.docs_url and p.help, p.key
        if p.flag.startswith("env:"):
            assert p.docs_url == "https://docs.sglang.io/docs/references/environment_variables", p.key
        else:
            assert p.docs_url.startswith("https://docs.sglang.io/docs/advanced_features/server_arguments#"), p.key
        if p.type == "enum":
            assert p.choices
            assert p.default is None or p.default in p.choices
        if p.min is not None and p.max is not None:
            assert p.min < p.max
    keys = {p.key for p in A.param_catalog()}
    for k in ("mem_fraction_static", "tool_call_parser", "enable_metrics", "tp_size", "quantization",
              "speculative_algorithm", "enable_hierarchical_cache", "enable_lora", "fp4_gemm_backend",
              "max_mamba_cache_size", "default_chat_template_kwargs", "media_url_max_file_size_mb"):
        assert k in keys


def test_levels_and_units() -> None:
    cat = A.param_catalog()
    for p in cat:
        assert p.level in ("basic", "advanced", "expert"), p.key
        if p.level == "basic":
            assert not p.advanced, p.key
        if p.level == "expert":
            assert p.advanced, p.key
        if p.unit is not None:
            assert p.unit in UNITS, (p.key, p.unit)
    basic = {p.key for p in cat if p.level == "basic"}
    assert 8 <= len(basic) <= 14
    assert {"context_length", "mem_fraction_static", "tp_size", "max_running_requests", "quantization",
            "tool_call_parser", "reasoning_parser"} <= basic
    by = {p.key: p for p in cat}
    assert by["context_length"].unit == "tokens" and by["watchdog_timeout"].unit == "s"
    assert by["max_running_requests"].unit == "requests" and by["cpu_offload_gb"].unit == "GB"


def test_retired_and_added_keys() -> None:
    keys = {p.key for p in A.param_catalog()}
    assert "api_key" not in keys                    # no environment variable in the image: it would always be refused
    (msg,) = A.validate({"api_key": "s3cret"})
    assert "authenticating proxy" in msg and "s3cret" not in msg


def test_v2_adapter_fields() -> None:
    assert A.model_formats == frozenset({"hf", "plain"})
    assert A.images_by_arch == {} and A.image_for("aarch64") == DEFAULT_IMAGE == "lmsysorg/sglang:v0.5.20-cu130"
    assert A.metric_prefixes == ("sglang:",)
    assert A.preset_info() == {}


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
    assert A.validate({"revision": "--evil"})
    assert A.validate({"speculative_algorithm": "EAGLE3"})
    assert A.validate({"speculative_num_steps": 3})
    assert A.validate({"enable_dp_attention": True, "tp_size": 2, "dp_size": 1})
    assert A.validate({"tp_size": 4, "ep_size": 3})
    assert A.validate({"enable_hierarchical_cache": True, "disable_radix_cache": True})
    assert A.validate({"json_model_override_args": "{bad"})
    assert A.validate({"quantization": "marlin"})                 # no longer a choice in 0.5.20
    assert A.validate({"attention_backend": "cutlass_mla"})       # gone in 0.5.20
    assert A.validate({"speculative_algorithm": "NEXTN", "speculative_num_steps": 3}) == []


# ---- build_launch ----
def test_build_launch_golden() -> None:
    spec = A.build_launch("Qwen/Qwen3.6-35B-A3B", {
        "quantization": "fp8", "context_length": 262144, "mem_fraction_static": 0.85,
        "tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3",
        "disable_radix_cache": False, "enable_mixed_chunk": True, "schedule_conservativeness": 1,
    }, hw(), served_name="qwen", hf_cache_container_path="/root/.cache/huggingface")
    assert spec.image == "lmsysorg/sglang:v0.5.20-cu130" and ":latest" not in spec.image
    assert spec.argv == [
        "python3", "-m", "sglang.launch_server", "--model-path", "Qwen/Qwen3.6-35B-A3B",
        "--host", "0.0.0.0", "--port", "30000", "--served-model-name", "qwen", "--enable-metrics",
        "--mem-fraction-static", "0.85", "--context-length", "262144", "--quantization", "fp8",
        "--schedule-conservativeness", "1.0", "--enable-mixed-chunk",
        "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3"]
    assert spec.env == {"HF_HUB_DISABLE_TELEMETRY": "1", "HF_HOME": "/root/.cache/huggingface"}
    assert spec.needs_hf_cache is True
    assert (spec.container_port, spec.health_path, spec.metrics_path) == (30000, "/health", "/metrics")


def test_env_param_becomes_a_container_variable_not_an_argument() -> None:
    spec = A.build_launch("m", {"allow_longer_context_len": True, "context_length": 1048576}, hw(),
                          served_name="s", hf_cache_container_path="/c")
    assert spec.env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] == "1"
    assert not any("ALLOW_OVERWRITE" in a or a.startswith("env:") for a in spec.argv)
    assert "--context-length" in spec.argv and "1048576" in spec.argv
    off = A.build_launch("m", {"allow_longer_context_len": False}, hw(), served_name="s", hf_cache_container_path="/c")
    assert off.env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] == "0"


def test_metrics_forced_and_served_override() -> None:
    spec = A.build_launch("m", {"enable_metrics": False, "served_model_name": "mine"}, hw(),
                          served_name="x", hf_cache_container_path="/c")
    assert spec.argv.count("--enable-metrics") == 1
    assert spec.argv[spec.argv.index("--served-model-name") + 1] == "mine"


def test_build_launch_list_json_and_rejects() -> None:
    spec = A.build_launch("m", {"lora_paths": ["a=/x", "b=/y"], "json_model_override_args": {"k": 1},
                                "cuda_graph_max_bs_decode": 64, "default_chat_template_kwargs": {"enable_thinking": True}},
                          hw(), served_name="s", hf_cache_container_path="/c")
    i = spec.argv.index("--lora-paths")
    assert spec.argv[i + 1:i + 3] == ["a=/x", "b=/y"]
    assert '{"k":1}' in spec.argv and "--cuda-graph-max-bs-decode" in spec.argv
    assert '{"enable_thinking":true}' in spec.argv
    with pytest.raises(ValueError):
        A.build_launch("--x", {}, hw(), served_name="s", hf_cache_container_path="/c")
    with pytest.raises(ValueError):
        A.build_launch("m", {"bad": 1}, hw(), served_name="s", hf_cache_container_path="/c")


def test_no_secret_can_reach_the_command_line() -> None:
    with pytest.raises(ValueError, match="authenticating proxy"):
        A.build_launch("m", {"api_key": "s3cret"}, hw(), served_name="s", hf_cache_container_path="/c")
    spec = A.build_launch("m", {}, hw(), served_name="s", hf_cache_container_path="/c")
    assert not any("key" in a.lower() for a in spec.argv)


def _ctx(**kw: object) -> LaunchContext:
    base: dict[str, object] = {"model": "org/m", "model_format": "hf", "served_name": "chat",
                               "hf_cache_container_path": "/hf", "params": {}, "hw": hw()}
    return LaunchContext(**(base | kw))   # type: ignore[arg-type]


def test_context_launch_for_hf_matches_build_launch() -> None:
    params = A.presets()["tool-agent"]
    assert A.build_launch_ctx(_ctx(params=params)) == A.build_launch(
        "org/m", params, hw(), served_name="chat", hf_cache_container_path="/hf")


def test_plain_model_directory_is_launched_by_path() -> None:
    spec = A.build_launch_ctx(_ctx(model="my-model-dir", model_format="plain", model_path=MODEL_MOUNT,
                                   params={"context_length": 8192, "revision": "main"}))
    a = spec.argv
    assert a[:5] == ["python3", "-m", "sglang.launch_server", "--model-path", "/models"]
    assert a[a.index("--served-model-name") + 1] == "chat"
    assert "--revision" not in a and "--context-length" in a        # a directory has no hub revision
    assert spec.needs_hf_cache is False
    assert "HF_HOME" not in spec.env and spec.env["HF_HUB_OFFLINE"] == "1"
    assert "my-model-dir" not in " ".join(a)
    assert A.build_launch_ctx(_ctx(model_format="plain")).argv[4] == MODEL_MOUNT


@pytest.mark.parametrize("fmt", ["gguf", "ollama"])
def test_unsupported_formats_are_refused(fmt: str) -> None:
    with pytest.raises(ValueError, match="cannot load"):
        A.build_launch_ctx(_ctx(model_format=fmt, model_path="/models/x.gguf"))


# ---- compatibility ----
def test_compat_gpus_and_mem_fraction() -> None:
    assert codes(mi(), {"tp_size": 2}, hw(1))["gpus_insufficient"] == "block"
    assert "gpus_insufficient" not in codes(mi(), {"tp_size": 2}, hw(2))
    assert codes(mi(), {"mem_fraction_static": 0.95})["mem_fraction_high"] == "warn"
    assert "mem_fraction_high" not in codes(mi(), {"mem_fraction_static": 0.85})


def test_compat_unbalanced_tensor_parallel_memory() -> None:
    busy = hw(2, free=[35.0, 0.6])
    c = {x.code: x for x in A.compatibility(mi(), {"tp_size": 2}, busy)}
    assert c["tp_memory_unbalanced"].level == "block" and "unbalanced" in c["tp_memory_unbalanced"].message
    assert "tp_memory_unbalanced" not in codes(mi(), {"tp_size": 2}, hw(2, free=[90.0, 85.0]))   # within 10 %
    assert "tp_memory_unbalanced" not in codes(mi(), {}, busy)                                  # tp 1 is not affected


def test_compat_quant_rules() -> None:
    assert codes(mi(quantization="nvfp4", is_moe=True), {})["nvfp4_ok_sglang"] == "ok"
    assert codes(mi("openai/gpt-oss-120b", quantization="mxfp4", is_moe=True), {})["mxfp4_gptoss_engine"] == "warn"
    assert codes(mi("x/moe", quantization="mxfp4", is_moe=True), {})["mxfp4_moe_sm120"] == "block"
    assert "mxfp4_moe_sm120" not in codes(mi("x/moe", quantization="mxfp4", is_moe=True), {}, hw(1, "10.0"))
    assert codes(mi(quantization="nvfp4"), {"quantization": "fp8"})["quant_override_prequantized"] == "warn"
    assert codes(mi(), {"quantization": "modelopt_fp4"})["modelopt_unquantized"] == "warn"


def test_compat_attention_backend() -> None:
    c = {x.code: x for x in A.compatibility(mi(), {"attention_backend": "fa3"}, hw())}
    assert c["attention_backend_hopper_only"].level == "block"
    assert "SM>=80 and SM<=90" in c["attention_backend_hopper_only"].message          # the engine's own assertion
    assert codes(mi(), {"decode_attention_backend": "flashmla"})["attention_backend_hopper_only"] == "block"
    assert codes(mi(), {"attention_backend": "trtllm_mla"})["attention_backend_dc_blackwell"] == "warn"
    assert codes(mi(), {"attention_backend": "fa4"})["attention_backend_dc_blackwell"] == "warn"
    # trtllm_mha is explicitly allowed on SM120 by the image (SM90, SM100 and SM120 checks)
    assert not any(c.startswith("attention_backend") for c in codes(mi(), {"attention_backend": "trtllm_mha"}))
    assert not any(c.startswith("attention_backend") for c in codes(mi(), {"attention_backend": "flashinfer"}))
    assert "attention_backend_hopper_only" not in codes(mi(), {"attention_backend": "fa3"}, hw(1, "9.0"))


def test_compat_trtllm_mha_with_fp8_kv_on_sm120() -> None:
    p = {"attention_backend": "trtllm_mha", "kv_cache_dtype": "fp8_e4m3"}
    assert codes(mi(), p)["trtllm_mha_fp8_kv_sm120"] == "block"
    assert "trtllm_mha_fp8_kv_sm120" not in codes(mi(), p, hw(1, "10.0"))
    assert "trtllm_mha_fp8_kv_sm120" not in codes(mi(), {"attention_backend": "trtllm_mha"})


def test_compat_flashinfer_linear_attention_needs_a_bf16_state() -> None:
    """Image source: handle_linear_attn_backend raises for SM100+ (capability major >= 10, so sm_120 too)."""
    bad = {x.code: x for x in A.compatibility(mi(), {"linear_attn_decode_backend": "flashinfer",
                                                       "mamba_ssm_dtype": "float32"}, hw())}
    assert bad["linear_attn_flashinfer_needs_bf16_state"].level == "block"
    assert "requires --mamba-ssm-dtype bfloat16" in bad["linear_attn_flashinfer_needs_bf16_state"].message
    unset = codes(mi(), {"linear_attn_backend": "flashinfer"})
    assert unset["linear_attn_flashinfer_needs_bf16_state"] == "warn"
    assert "linear_attn_flashinfer_needs_bf16_state" not in codes(
        mi(), {"linear_attn_decode_backend": "flashinfer", "mamba_ssm_dtype": "bfloat16"})
    assert "linear_attn_flashinfer_needs_bf16_state" not in codes(mi(), {"linear_attn_decode_backend": "triton"})
    assert "linear_attn_flashinfer_needs_bf16_state" not in codes(
        mi(), {"linear_attn_decode_backend": "flashinfer", "mamba_ssm_dtype": "float32"}, hw(1, "9.0"))


def test_compat_tool_parser_dialect() -> None:
    assert codes(mi(), {"tool_call_parser": "qwen"})["tool_parser_dialect_mismatch"] == "warn"
    assert "tool_parser_dialect_mismatch" not in codes(mi(), {"tool_call_parser": "qwen3_coder"})
    assert codes(mi("Qwen/Qwen3-32B"), {"tool_call_parser": "qwen3_coder"})["tool_parser_dialect_mismatch"] == "warn"
    assert "tool_parser_dialect_mismatch" not in codes(mi("Qwen/Qwen3-32B"), {"tool_call_parser": "qwen25"})
    # a size suffix such as "-8B" must not turn Qwen3-8B into a "Qwen3.x" XML model
    assert "tool_parser_dialect_mismatch" not in codes(mi("Qwen/Qwen3-8B"), {"tool_call_parser": "hermes"})
    assert codes(mi("openai/gpt-oss-20b"), {"tool_call_parser": "hermes"})["tool_parser_dialect_mismatch"] == "warn"
    assert codes(mi(), {})["no_tool_parser"] == "warn"


def test_compat_misc() -> None:
    m = mi(is_moe=True)
    assert codes(m, {"enable_dp_attention": True, "tp_size": 2, "dp_size": 1}, hw(2))["dp_attention_dp_ne_tp"] == "block"
    assert codes(mi(), {"enable_dp_attention": True, "tp_size": 2, "dp_size": 2}, hw(2))["dp_attention_unsupported_arch"] == "warn"
    assert codes(mi(max_position_embeddings=32768), {"context_length": 65536})["context_beyond_native"] == "warn"
    refuse = {c.code: c for c in A.compatibility(mi(max_position_embeddings=32768), {"context_length": 65536}, hw())}
    assert "allow_longer_context_len" in refuse["context_beyond_native"].message
    allowed = {c.code: c for c in A.compatibility(
        mi(max_position_embeddings=32768), {"context_length": 65536, "allow_longer_context_len": True}, hw())}
    assert "incorrect model outputs" in allowed["context_beyond_native"].message
    assert codes(mi(gated=True), {})["gated_model"] == "warn"
    assert codes(mi(), {"enable_metrics": False})["metrics_forced"] == "warn"
    assert codes(mi(), {"kv_cache_dtype": "nvfp4"})["kv_fp4_experimental"] == "warn"
    assert "api_key_in_argv" not in codes(mi(), {})


def test_every_message_is_generic_text() -> None:
    """The repository is public: no site-specific wording in anything a user is shown."""
    banned = re.compile(r"\blab\b|this repo|validated here|/home/|@[a-z0-9-]+\.[a-z]{2,}", re.I)
    seen = 0
    for model in (mi(), mi("openai/gpt-oss-120b", quantization="mxfp4", is_moe=True),
                  mi("Qwen/Qwen3-8B", quantization="nvfp4", gated=True)):
        for params in ({}, {"tp_size": 2, "attention_backend": "fa3", "mem_fraction_static": 0.95,
                            "tool_call_parser": "hermes", "kv_cache_dtype": "fp8_e4m3", "enable_dp_attention": True,
                            "log_requests": True, "trust_remote_code": True, "cpu_offload_gb": 2}):
            for c in A.compatibility(model, params, hw(2, free=[90.0, 20.0])):
                seen += 1
                assert not banned.search(c.message), c
    assert seen > 10


# ---- memory model ----
def test_memory_model() -> None:
    m = A.memory_model(mi(max_position_embeddings=262144),
                       {"mem_fraction_static": 0.85, "kv_cache_dtype": "fp8_e4m3", "tp_size": 2,
                        "max_running_requests": 8})
    assert m["mem_fraction"] == 0.85 and m["kv_bytes_per_elem"] == 1.0 and m["tp"] == 2.0
    assert m["max_len"] == 262144 and m["max_seqs"] == 8 and m["overhead_gib"] == 0.0
    assert abs(m["nonstatic_reserve_frac"] - 0.15) < 1e-9
    assert m["share_of_free"] == 1.0 and "claims_share" not in m   # 0.5.20: fraction x memory free at launch
    d = A.memory_model(mi(), {})
    assert d["mem_fraction"] == 0.85 and d["kv_bytes_per_elem"] == 2.0 and d["mem_fraction_is_default"] == 1.0
    assert A.memory_model(mi(), {"kv_cache_dtype": "nvfp4"})["kv_bytes_per_elem"] == 0.5


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
sglang:num_retracted_reqs{model_name="qwen",tp_rank="0"} 7.0
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
    assert m["generation_tokens_total"] == 45000
    assert m["preemptions_total"] == 2                    # the cumulative counter, never the per-interval gauge (7)
    assert m["spec_decode_accept_pct"] == pytest.approx(65.0) and m["spec_accept_length"] == 2.6
    assert m["ttft_p50_s"] == pytest.approx(0.1 + 0.4 * (50 - 10) / 80)
    assert 0.5 < m["ttft_p95_s"] < 1.0
    assert m["itl_p50_s"] == pytest.approx(0.01)
    assert m["e2e_p50_s"] == pytest.approx(1.0 + 4.0 * (50 - 20) / 80)
    assert "prompt_tps" not in m


def test_gauge_without_the_counter_is_not_reported_as_a_total() -> None:
    assert "preemptions_total" not in A.parse_metrics('sglang:num_retracted_reqs{tp_rank="0"} 7\n')


def _fixture(name: str) -> str:
    from pathlib import Path
    return (Path(__file__).parent / "fixtures" / name).read_text()


def test_real_v0520_scrape() -> None:
    """A real SGLang 0.5.20 scrape (idle, tensor-parallel 2); see the header of the fixture for how it was captured."""
    m = A.parse_metrics(_fixture("metrics_v0.5.20.prom"))
    assert m["requests_running"] == 0 and m["requests_waiting"] == 0
    assert m["kv_cache_usage_pct"] == 0 and m["prefix_cache_hit_pct"] == 0 and m["generation_tps"] == 0
    # the token counters are split by is_streaming and summed
    assert m["prompt_tokens_total"] == 21261962 + 111648670
    assert m["generation_tokens_total"] == 720460 + 819339
    assert m["spec_decode_accept_pct"] == pytest.approx(14.1667, abs=1e-3) and m["spec_accept_length"] == 1.425
    assert 0.1 < m["ttft_p50_s"] <= m["ttft_p95_s"] < 5 and 0 < m["itl_p50_s"] < 0.05
    assert 0 < m["e2e_p50_s"] <= m["e2e_p95_s"]
    # the retraction counter is created lazily (first retraction); the per-interval gauge must not stand in for it
    assert "preemptions_total" not in m
    assert set(m) <= {"requests_running", "requests_waiting", "kv_cache_usage_pct", "prefix_cache_hit_pct",
                      "prompt_tokens_total", "generation_tokens_total", "ttft_p50_s", "ttft_p95_s", "itl_p50_s",
                      "e2e_p50_s", "e2e_p95_s", "prompt_tps", "generation_tps", "preemptions_total",
                      "spec_decode_accept_pct", "spec_accept_length"}


def test_kv_usage_fallback_under_tensor_parallelism() -> None:
    """Every rank reports the pool size but only rank 0 reports the used tokens (as in the live data)."""
    text = ('sglang:kv_used_tokens{tp_rank="0"} 500\nsglang:max_total_num_tokens{tp_rank="0"} 1000\n'
            'sglang:max_total_num_tokens{tp_rank="1"} 1000\n')
    assert A.parse_metrics(text)["kv_cache_usage_pct"] == pytest.approx(50.0)


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
# Lines below are built from the f-strings of the pinned 0.5.20 source (and, where marked, the 0.5.14 wording); none was
# captured from a live engine.
@pytest.mark.parametrize("line,expected", [
    ("[2026-09-23 10:00:00] Load weight begin. avail mem=93.10 GB", {"phase": "loading_weights", "pct": 0.0}),
    ("Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:10<00:10,  5.0s/it]",
     {"phase": "loading_weights", "pct": 50.0}),
    ("Multi-thread loading shards:  25%|██▌       | 1/4 [00:02<00:06]", {"phase": "loading_weights", "pct": 25.0}),
    ("Load weight end. elapsed=41.30 s, type=Qwen3MoeForCausalLM, dtype=torch.bfloat16, avail mem=58.70 GB, "
     "mem usage=34.50 GB.", {"phase": "loading_weights", "pct": 100.0, "weights_gib": 34.5, "load_seconds": 41.3}),
    ("Load weight end. type=Qwen3MoeForCausalLM, dtype=torch.bfloat16, avail mem=58.70 GB, mem usage=34.50 GB.",
     {"phase": "loading_weights", "pct": 100.0, "weights_gib": 34.5}),                    # 0.5.14 wording
    ("KV Cache is allocated. dtype: torch.bfloat16, #tokens: 1234567, K size: 20.00 GB, V size: 20.00 GB",
     {"phase": "allocating_kv", "kv_cache_tokens": 1234567}),
    ("Full KV Cache VA upper bound. dtype: torch.bfloat16, #tokens: 99, K size: 1.00 GB, V size: 1.00 GB",
     {"phase": "allocating_kv", "kv_cache_tokens": 99}),
    ("KV Cache is allocated. #tokens: 1234567, K size: 20.00 GB, V size: 20.00 GB",
     {"phase": "allocating_kv", "kv_cache_tokens": 1234567}),                              # 0.5.14 wording
    ("max_total_num_tokens=1234567, chunked_prefill_size=8192, max_prefill_tokens=16384, "
     "max_running_requests=2048, context_len=262144, available_gpu_mem=9.5 GB",
     {"phase": "allocating_kv", "kv_cache_tokens": 1234567, "max_running_requests": 2048, "context_len": 262144}),
    ("Capture decode CUDA graph begin. backend=full, num_tokens_per_req=1, bs=[1, 2, 4], avail mem=9.50 GB",
     {"phase": "capturing_graphs", "pct": 0.0}),
    ("Capture prefill CUDA graph begin. backend=tc_piecewise, num_tokens=4096, avail mem=9.50 GB",
     {"phase": "capturing_graphs", "pct": 0.0}),
    ("Capture cuda graph begin. This can take up to several minutes. avail mem=9.50 GB",
     {"phase": "capturing_graphs", "pct": 0.0}),                                           # 0.5.14 wording
    ("Capturing batches (bs=16 avail_mem=9.10 GB):  40%|████      | 4/10 [00:03<00:04]",
     {"phase": "capturing_graphs", "batch_size": 16, "pct": 40.0}),
    ("Capturing num tokens (num_tokens=4096 avail_mem=9.00 GB):  50%|█████     | 5/10 [00:03<00:04]",
     {"phase": "capturing_graphs", "num_tokens": 4096, "pct": 50.0}),
    ("Capture decode CUDA graph end. elapsed=12.31 s, mem usage=1.20 GB, avail mem=8.30 GB.",
     {"phase": "capturing_graphs", "pct": 100.0, "cuda_graph_gib": 1.2}),
    ("Capture cuda graph end. Time elapsed: 12.31 s. mem usage=1.20 GB. avail mem=8.30 GB.",
     {"phase": "capturing_graphs", "pct": 100.0, "cuda_graph_gib": 1.2}),                  # 0.5.14 wording
    ("INFO:     The server is fired up and ready to roll!", {"phase": "ready", "pct": 100.0}),
    ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB", {"phase": "failed", "error": "oom"}),
    ("RuntimeError: The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
     "pre_model_load_memory=10.0, local_gpu_memory=35.0",
     {"phase": "failed", "error": "memory_unbalanced"}),
    ("ValueError: Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.5. Raise "
     "--mem-fraction-static above 0.612 (minimum viable = 1 - available/pre = 0.6111).",
     {"phase": "failed", "error": "mem_fraction_too_low", "suggested_mem_fraction_static": 0.612}),
    ("AssertionError: FlashAttention v3 Backend requires SM>=80 and SM<=90. Please use `--attention-backend flashinfer`.",
     {"phase": "failed", "error": "attention_backend_unsupported"}),
    ("Capture cuda graph failed: boom\nPossible solutions: ...", {"phase": "failed", "error": "cuda_graph_capture_failed"}),
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
