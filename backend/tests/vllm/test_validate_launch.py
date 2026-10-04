from __future__ import annotations

import pytest

from engine_console.adapters.base import MODEL_MOUNT, Hardware, LaunchContext
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
    assert adapter.validate({"allowed_local_media_path": "/"})
    assert adapter.validate({"enable_auto_tool_choice": True})
    assert adapter.validate({"lora_modules": ["a=/x"]})
    assert adapter.validate({"speculative_config": {"method": "mtp"}, "spec_method": "mtp"})
    assert adapter.validate({"speculative_config": '{"model": "org/draft"}', "spec_model": "org/draft"})
    assert adapter.validate({"revision": "--evil"})
    assert adapter.validate({"served_model_name": ["--evil"]})
    assert adapter.validate({"quantization": "gguf"})                 # gone from the 0.27.1 registry
    assert adapter.validate({"quantization": "quark"})                # AMD only, left out of the catalog on purpose


def test_spec_shortcuts_merge_unless_a_key_is_given_twice(adapter: VllmAdapter) -> None:
    """vLLM 0.27.1 merges --spec-method/--spec-model/--spec-tokens into --speculative-config and refuses only a key
    that appears in both (engine/arg_utils.py)."""
    assert adapter.validate({"speculative_config": {"num_speculative_tokens": 2}, "spec_method": "mtp"}) == []
    assert adapter.validate({"speculative_config": '{"method": "mtp"}', "spec_tokens": 3}) == []
    (e,) = adapter.validate({"speculative_config": {"method": "mtp"}, "spec_method": "mtp"})
    assert "mutually exclusive" in e and "spec_method" in e
    (e2,) = adapter.validate({"speculative_config": {"num_speculative_tokens": 2}, "spec_tokens": 3})
    assert "num_speculative_tokens" in e2


def test_retired_keys_say_what_to_do_instead(adapter: VllmAdapter) -> None:
    (img,) = adapter.validate({"image": "vllm/vllm-openai:v0.25.1"})
    assert "pinned in Settings" in img
    (gemm,) = adapter.validate({"nvfp4_gemm_backend": "marlin"})
    assert "linear_backend" in gemm and "does not exist in vLLM 0.27.1" in gemm
    (moe,) = adapter.validate({"flashinfer_moe_backend": "latency"})
    assert "moe_backend" in moe


def test_validate_accepts_valid_and_none(adapter: VllmAdapter) -> None:
    assert adapter.validate({"max_model_len": None}) == []
    assert adapter.validate({"hf_overrides": '{"a": 1}', "max_lora_rank": 16, "enable_lora": True,
                             "lora_modules": ["x=/y"], "kv_cache_dtype": "int4_per_token_head",
                             "mamba_cache_mode": "align", "moe_backend": "flashinfer_b12x"}) == []


def test_build_launch_golden_balanced(adapter: VllmAdapter, hw: Hardware) -> None:
    spec = adapter.build_launch("example-org/chat-model-36B", adapter.presets()["balanced"], hw,
                                served_name="chat", hf_cache_container_path="/root/.cache/huggingface")
    assert spec.image == "vllm/vllm-openai:v0.27.1"
    assert spec.argv == [
        "--model", "example-org/chat-model-36B", "--host", "0.0.0.0", "--port", "8000",
        "--served-model-name", "chat",
        "--gpu-memory-utilization", "0.9", "--max-model-len", "65536",
        "--enable-prefix-caching", "--performance-mode", "balanced"]
    assert spec.env == {"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                        "TORCH_CUDA_ARCH_LIST": "12.0", "HF_HOME": "/root/.cache/huggingface"}
    assert spec.needs_hf_cache is True
    assert (spec.container_port, spec.health_path, spec.metrics_path) == (8000, "/health", "/metrics")


def test_build_launch_golden_complex(adapter: VllmAdapter, hw: Hardware) -> None:
    params = {
        "quantization": "fp8", "enable_auto_tool_choice": True, "tool_call_parser": "hermes",
        "enable_chunked_prefill": False, "enable_prefix_caching": True, "enforce_eager": False,
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 2},
        "hf_overrides": '{"text_config": {"rope_parameters": {"factor": 4.0}}}',
        "cudagraph_capture_sizes": [1, 2, 4], "served_model_name": ["alias", "chat"],
        "attention_backend": "FLASHINFER", "api_key": "sekret", "allow_long_max_model_len": True,
        "kv_cache_dtype": "fp8", "moe_backend": "marlin", "mamba_cache_mode": "align",
    }
    spec = adapter.build_launch("org/m", params, hw, served_name="chat", hf_cache_container_path="/hf")
    assert spec.image == "vllm/vllm-openai:v0.27.1"
    assert spec.argv == [
        "--model", "org/m", "--host", "0.0.0.0", "--port", "8000", "--served-model-name", "chat", "alias",
        "--kv-cache-dtype", "fp8", "--mamba-cache-mode", "align",
        "--hf-overrides", '{"text_config":{"rope_parameters":{"factor":4.0}}}',
        "--quantization", "fp8", "--moe-backend", "marlin",
        "--enable-prefix-caching", "--no-enable-chunked-prefill",
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
        "--no-enforce-eager", "--cudagraph-capture-sizes", "1", "2", "4", "--attention-backend", "FLASHINFER",
        "--speculative-config", '{"method":"mtp","num_speculative_tokens":2}']
    assert spec.env["VLLM_API_KEY"] == "sekret"
    assert "sekret" not in " ".join(spec.argv)                  # a secret never rides on the command line
    assert spec.env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "1"
    assert not any("ATTENTION_BACKEND" in k for k in spec.env)  # VLLM_ATTENTION_BACKEND no longer exists
    assert "HF_TOKEN" not in spec.env


def test_non_negatable_false_bool_is_omitted(adapter: VllmAdapter, hw: Hardware) -> None:
    spec = adapter.build_launch("m", {"enable_request_id_headers": False, "disable_custom_all_reduce": False},
                                hw, served_name="s", hf_cache_container_path="/hf")
    assert "--enable-request-id-headers" not in spec.argv
    assert "--no-disable-custom-all-reduce" in spec.argv


def test_nargs_flags_are_emitted_once_with_all_values(adapter: VllmAdapter, hw: Hardware) -> None:
    spec = adapter.build_launch("m", {"allowed_media_domains": ["a.example", "b.example"],
                                      "kv_cache_dtype_skip_layers": ["0", "sliding_window"]}, hw,
                                served_name="s", hf_cache_container_path="/hf")
    a = spec.argv
    assert a[a.index("--allowed-media-domains") + 1:a.index("--allowed-media-domains") + 3] == ["a.example", "b.example"]
    assert a.count("--allowed-media-domains") == 1 and a.count("--kv-cache-dtype-skip-layers") == 1


def test_arch_list_follows_the_hardware(adapter: VllmAdapter, hw: Hardware) -> None:
    hopper = hw.model_copy(update={"compute_capability": "9.0"})
    assert adapter.build_launch("m", {}, hopper, served_name="s", hf_cache_container_path="/h").env[
        "TORCH_CUDA_ARCH_LIST"] == "9.0"
    odd = hw.model_copy(update={"compute_capability": "unknown"})
    assert "TORCH_CUDA_ARCH_LIST" not in adapter.build_launch("m", {}, odd, served_name="s",
                                                              hf_cache_container_path="/h").env


def test_build_launch_rejects_invalid(adapter: VllmAdapter, hw: Hardware) -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        adapter.build_launch("m", {"bogus": 1}, hw, served_name="s", hf_cache_container_path="/hf")
    with pytest.raises(ValueError, match="invalid model identifier"):
        adapter.build_launch("--evil", {}, hw, served_name="s", hf_cache_container_path="/hf")


# ---- contract v2: launch from a context ------------------------------------------------------------------------------
def _ctx(hw: Hardware, **kw: object) -> LaunchContext:
    base: dict[str, object] = {"model": "org/m", "model_format": "hf", "served_name": "chat",
                               "hf_cache_container_path": "/hf", "params": {}, "hw": hw}
    return LaunchContext(**(base | kw))   # type: ignore[arg-type]


def test_context_launch_for_hf_matches_build_launch(adapter: VllmAdapter, hw: Hardware) -> None:
    params = adapter.presets()["tool-agent"]
    via_ctx = adapter.build_launch_ctx(_ctx(hw, params=params))
    direct = adapter.build_launch("org/m", params, hw, served_name="chat", hf_cache_container_path="/hf")
    assert via_ctx == direct and via_ctx.needs_hf_cache is True


def test_plain_model_directory_is_launched_by_path(adapter: VllmAdapter, hw: Hardware) -> None:
    spec = adapter.build_launch_ctx(_ctx(hw, model="my-model-dir", model_format="plain", model_path=MODEL_MOUNT,
                                         params={"max_model_len": 8192, "revision": "main",
                                                 "served_model_name": ["alias"]}))
    assert spec.argv[:2] == ["--model", "/models"]
    assert spec.argv[spec.argv.index("--served-model-name") + 1:spec.argv.index("--served-model-name") + 3] == [
        "chat", "alias"]
    assert "--revision" not in spec.argv                     # a directory has no hub revision
    assert "--max-model-len" in spec.argv
    assert spec.needs_hf_cache is False
    assert "HF_HOME" not in spec.env and spec.env["HF_HUB_OFFLINE"] == "1"
    assert "my-model-dir" not in " ".join(spec.argv)         # the display name is never used as a path


def test_plain_model_defaults_to_the_mount_when_no_path_is_given(adapter: VllmAdapter, hw: Hardware) -> None:
    assert adapter.build_launch_ctx(_ctx(hw, model_format="plain")).argv[:2] == ["--model", MODEL_MOUNT]


@pytest.mark.parametrize("fmt", ["gguf", "ollama"])
def test_unsupported_formats_are_refused(adapter: VllmAdapter, hw: Hardware, fmt: str) -> None:
    with pytest.raises(ValueError, match="cannot load"):
        adapter.build_launch_ctx(_ctx(hw, model_format=fmt, model_path="/models/x.gguf"))


def test_memory_model(adapter: VllmAdapter, dense) -> None:  # type: ignore[no-untyped-def]
    mm = adapter.memory_model(dense, {"gpu_memory_utilization": 0.9, "kv_cache_dtype": "fp8",
                                      "tensor_parallel_size": 2, "max_num_seqs": 16})
    assert mm["mem_fraction"] == 0.9 and mm["kv_bytes_per_elem"] == 1.0 and mm["tp"] == 2
    assert mm["max_len"] == 131072 and mm["max_seqs"] == 16
    assert mm["overhead_gib"] == 3.0          # 2.5 GiB held outside the share (measured) + 0.5 GiB per extra rank (guess)
    assert adapter.memory_model(dense, {"enforce_eager": True})["cuda_graph_gib"] == 0.0
    single = adapter.memory_model(dense, {"gpu_memory_utilization": 0.15, "enforce_eager": True})
    # live measurement: a 0.15 share of a 95.6 GiB card (14.34 GiB) plus the overhead matches the 16.86 GiB the driver showed
    assert round(0.15 * 95.6 + single["overhead_gib"], 1) == 16.8 and single["cuda_graph_gib"] == 0.0
    assert adapter.memory_model(dense, {})["cuda_graph_gib"] == 0.75         # graphs come on top, as before
    assert adapter.memory_model(dense, {})["kv_bytes_per_elem"] == 2.0
    assert adapter.memory_model(dense, {"kv_cache_dtype": "int4_per_token_head"})["kv_bytes_per_elem"] == 0.5
    fixed = adapter.memory_model(dense, {"kv_cache_memory_bytes": 7_000_000_000})
    assert round(fixed["kv_cache_fixed_gib"], 2) == 6.52
