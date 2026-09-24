from __future__ import annotations

import pytest

from engine_console.adapters.base import Hardware
from engine_console.adapters.vllm import VllmAdapter


def test_presets_validate(adapter: VllmAdapter) -> None:
    pre = adapter.presets()
    assert {"balanced", "max-throughput", "low-latency", "long-context", "tool-agent", "reasoning"} <= set(pre)
    for name, p in pre.items():
        assert adapter.validate(p) == [], name


def test_validate_errors(adapter: VllmAdapter) -> None:
    assert adapter.validate({"nope": 1}) == ["unknown parameter 'nope'"]
    assert adapter.validate({"gpu_memory_utilization": 1.5})
    assert adapter.validate({"gpu_memory_utilization": "0.9"})
    assert adapter.validate({"max_model_len": True})
    assert adapter.validate({"max_model_len": 1000.5})
    assert adapter.validate({"kv_cache_dtype": "fp7"})
    assert adapter.validate({"enforce_eager": 1})
    assert adapter.validate({"hf_overrides": "{not json"})
    assert adapter.validate({"cudagraph_capture_sizes": [1, 0]})
    assert adapter.validate({"image": "vllm/vllm-openai:latest"})
    assert adapter.validate({"image": "vllm/vllm-openai"})
    assert adapter.validate({"allowed_local_media_path": "/"})
    assert adapter.validate({"enable_auto_tool_choice": True})
    assert adapter.validate({"lora_modules": ["a=/x"]})
    assert adapter.validate({"speculative_config": {"method": "mtp"}, "spec_method": "mtp"})


def test_validate_accepts_valid_and_none(adapter: VllmAdapter) -> None:
    assert adapter.validate({"max_model_len": None}) == []
    assert adapter.validate({"hf_overrides": '{"a": 1}', "max_lora_rank": 16, "enable_lora": True,
                             "lora_modules": ["x=/y"], "image": "vllm/vllm-openai:v0.27.1"}) == []


def test_build_launch_golden_balanced(adapter: VllmAdapter, hw: Hardware) -> None:
    spec = adapter.build_launch("example-org/chat-model-36B", adapter.presets()["balanced"], hw,
                                served_name="chat", hf_cache_container_path="/root/.cache/huggingface")
    assert spec.image == "vllm/vllm-openai:v0.23.0"
    assert spec.argv == [
        "--model", "example-org/chat-model-36B", "--host", "0.0.0.0", "--port", "8000",
        "--served-model-name", "chat",
        "--gpu-memory-utilization", "0.9", "--max-model-len", "65536",
        "--enable-prefix-caching", "--performance-mode", "balanced"]
    assert spec.env["TORCH_CUDA_ARCH_LIST"] == "12.0"
    assert spec.env["HF_HOME"] == "/root/.cache/huggingface"
    assert "HF_TOKEN" not in spec.env
    assert (spec.container_port, spec.health_path, spec.metrics_path) == (8000, "/health", "/metrics")


def test_build_launch_golden_complex(adapter: VllmAdapter, hw: Hardware) -> None:
    params = {
        "quantization": "fp8", "enable_auto_tool_choice": True, "tool_call_parser": "hermes",
        "enable_chunked_prefill": False, "enable_prefix_caching": True, "enforce_eager": False,
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 2},
        "hf_overrides": '{"text_config": {"rope_parameters": {"factor": 4.0}}}',
        "cudagraph_capture_sizes": [1, 2, 4], "served_model_name": ["alias", "chat"],
        "attention_backend": "FLASHINFER", "api_key": "sekret", "allow_long_max_model_len": True,
        "image": "vllm/vllm-openai:v0.27.1", "kv_cache_dtype": "fp8", "moe_backend": "marlin",
    }
    spec = adapter.build_launch("org/m", params, hw, served_name="chat", hf_cache_container_path="/hf")
    assert spec.image == "vllm/vllm-openai:v0.27.1"
    assert spec.argv == [
        "--model", "org/m", "--host", "0.0.0.0", "--port", "8000", "--served-model-name", "chat", "alias",
        "--kv-cache-dtype", "fp8",
        "--hf-overrides", '{"text_config":{"rope_parameters":{"factor":4.0}}}',
        "--quantization", "fp8", "--moe-backend", "marlin",
        "--enable-prefix-caching", "--no-enable-chunked-prefill",
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
        "--no-enforce-eager", "--cudagraph-capture-sizes", "1", "2", "4", "--attention-backend", "FLASHINFER",
        "--speculative-config", '{"method":"mtp","num_speculative_tokens":2}']
    assert spec.env["VLLM_API_KEY"] == "sekret"
    assert "sekret" not in " ".join(spec.argv)
    assert spec.env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "1"
    assert not any("VLLM_ATTENTION_BACKEND" in k for k in spec.env)


def test_non_negatable_false_bool_is_omitted(adapter: VllmAdapter, hw: Hardware) -> None:
    from engine_console.adapters.base import ParamSpec  # noqa: F401  (documents the port used)
    spec = adapter.build_launch("m", {"enable_request_id_headers": False, "disable_custom_all_reduce": False},
                                hw, served_name="s", hf_cache_container_path="/hf")
    assert "--enable-request-id-headers" not in spec.argv
    assert "--no-disable-custom-all-reduce" in spec.argv


def test_build_launch_rejects_invalid(adapter: VllmAdapter, hw: Hardware) -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        adapter.build_launch("m", {"bogus": 1}, hw, served_name="s", hf_cache_container_path="/hf")


def test_memory_model(adapter: VllmAdapter, dense) -> None:  # type: ignore[no-untyped-def]
    mm = adapter.memory_model(dense, {"gpu_memory_utilization": 0.9, "kv_cache_dtype": "fp8",
                                      "tensor_parallel_size": 2, "max_num_seqs": 16})
    assert mm["mem_fraction"] == 0.9 and mm["kv_bytes_per_elem"] == 1.0 and mm["tp"] == 2
    assert mm["max_len"] == 131072 and mm["max_seqs"] == 16 and mm["overhead_gib"] == 1.5
    assert adapter.memory_model(dense, {"enforce_eager": True})["cuda_graph_gib"] == 0.0
    assert adapter.memory_model(dense, {})["kv_bytes_per_elem"] == 2.0
    fixed = adapter.memory_model(dense, {"kv_cache_memory_bytes": 7_000_000_000})
    assert round(fixed["kv_cache_fixed_gib"], 2) == 6.52
