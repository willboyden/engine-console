from __future__ import annotations

from engine_console.adapters.base import Hardware, ModelInfo
from engine_console.adapters.vllm import VllmAdapter


def codes(adapter: VllmAdapter, m: ModelInfo, p: dict, hw: Hardware) -> dict:  # type: ignore[type-arg]
    return {c.code: c for c in adapter.compatibility(m, p, hw)}


def test_nvfp4_moe_warns_unless_backend_chosen(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    m = ModelInfo(repo_id="nvidia/x-30B-A3B-NVFP4", quantization="nvfp4", is_moe=True)
    c = codes(adapter, m, {}, hw)["nvfp4_moe_sm120"]
    assert c.level == "warn" and "FP8" in c.message and "SGLang" in c.message
    assert "nvfp4_moe_sm120" not in codes(adapter, m, {"moe_backend": "cutlass"}, hw)


def test_nvfp4_dense_is_clean(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    m = ModelInfo(repo_id="unsloth/Qwen3.8-27B-NVFP4", quantization="nvfp4", is_moe=False)
    assert "nvfp4_moe_sm120" not in codes(adapter, m, {}, hw)


def test_mxfp4_moe_blocks_except_gpt_oss(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    bad = ModelInfo(repo_id="acme/moe-mxfp4", quantization="mxfp4", is_moe=True)
    assert codes(adapter, bad, {}, hw)["mxfp4_moe_sm120"].level == "block"
    oss = ModelInfo(repo_id="openai/gpt-oss-120b", quantization="mxfp4", is_moe=True)
    c = codes(adapter, oss, {}, hw)
    assert "mxfp4_moe_sm120" not in c and c["gpt_oss_prefer_ollama"].level == "warn"


def test_gated(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    c = codes(adapter, ModelInfo(repo_id="google/gemma-4-31B", gated=True), {}, hw)
    assert "HF_TOKEN" in c["gated_needs_hf_token"].message


def test_tp_exceeds_gpus_and_divisibility(adapter, hw, dense) -> None:  # type: ignore[no-untyped-def]
    assert codes(adapter, dense, {"tensor_parallel_size": 4}, hw)["tp_exceeds_gpus"].level == "block"
    assert codes(adapter, dense, {"tensor_parallel_size": 2}, hw)["tp_over_pcie"].level == "warn"
    odd = dense.model_copy(update={"num_attention_heads": 36})
    assert "tp_heads_indivisible" in codes(adapter, odd, {"tensor_parallel_size": 8, "data_parallel_size": 1},
                                           hw.model_copy(update={"gpu_ids": list(range(8))}))
    assert codes(adapter, dense, {"data_parallel_size": 2, "tensor_parallel_size": 2}, hw)["tp_exceeds_gpus"]


def test_kv_fp8_backend_combo(adapter, hw, dense) -> None:  # type: ignore[no-untyped-def]
    c = codes(adapter, dense, {"kv_cache_dtype": "fp8", "attention_backend": "FLASH_ATTN"}, hw)
    assert "FLASHINFER" in c["kv_fp8_flash_attn"].message
    assert "kv_fp8_flash_attn" not in codes(adapter, dense, {"kv_cache_dtype": "fp8",
                                                             "attention_backend": "FLASHINFER"}, hw)


def test_tool_and_reasoning_parser_hints(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    m = ModelInfo(repo_id="Qwen/Qwen3.6-35B-A3B", pipeline_tag="text-generation")
    c = codes(adapter, m, {}, hw)
    assert "tool_call_parser=qwen3_xml" in c["tool_parser_missing"].message
    assert "reasoning_parser=qwen3" in c["reasoning_parser_missing"].message
    c2 = codes(adapter, m, {"tool_call_parser": "qwen3_xml", "enable_auto_tool_choice": True,
                            "reasoning_parser": "qwen3"}, hw)
    assert "tool_parser_missing" not in c2 and "reasoning_parser_missing" not in c2
    assert "tool_parser_inactive" in codes(adapter, m, {"tool_call_parser": "hermes"}, hw)
    coder = ModelInfo(repo_id="Qwen/Qwen3-Coder-30B", pipeline_tag="text-generation")
    assert "reasoning_parser_missing" not in codes(adapter, coder, {}, hw)


def test_quant_override_on_prequantized_blocks(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    m = ModelInfo(repo_id="unsloth/Qwen3.8-27B-NVFP4", quantization="nvfp4")
    assert codes(adapter, m, {"quantization": "fp8"}, hw)["quant_override_prequantized"].level == "block"
    assert "quant_override_prequantized" not in codes(adapter, m, {}, hw)


def test_fp8_recommended_for_big_bf16(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    m = ModelInfo(repo_id="a/b", weight_bytes=int(90 * 1024 ** 3))
    assert "fp8_recommended" in codes(adapter, m, {}, hw)
    assert "fp8_recommended" not in codes(adapter, m, {"quantization": "fp8"}, hw)


def test_max_len_rules(adapter, hw, dense) -> None:  # type: ignore[no-untyped-def]
    assert codes(adapter, dense, {"max_model_len": 262144}, hw)["max_len_exceeds_model"].level == "block"
    ok = codes(adapter, dense, {"max_model_len": 262144, "hf_overrides": {"a": 1}}, hw)
    assert ok["yarn_short_context_quality"].level == "warn" and "max_len_exceeds_model" not in ok
    assert "max_len_exceeds_model" not in codes(adapter, dense, {"max_model_len": 262144,
                                                                 "allow_long_max_model_len": True}, hw)


def test_free_vram(adapter, hw, dense) -> None:  # type: ignore[no-untyped-def]
    busy = hw.model_copy(update={"gpu_free_gib": [9.7, 94.0]})
    c = codes(adapter, dense, {"gpu_memory_utilization": 0.9}, busy)["insufficient_free_vram"]
    assert c.level == "block" and "0.09" in c.message
    assert "insufficient_free_vram" not in codes(adapter, dense, {"gpu_memory_utilization": 0.05}, busy)
    assert "insufficient_free_vram" not in codes(adapter, dense, {"kv_cache_memory_bytes": 7_000_000_000}, busy)


def test_image_version_rules(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    laguna = ModelInfo(repo_id="poolside/Laguna-S-2.1-NVFP4")
    assert codes(adapter, laguna, {}, hw)["image_too_old_laguna"].level == "block"
    assert "image_too_old_laguna" not in codes(adapter, laguna, {"image": "vllm/vllm-openai:v0.25.1"}, hw)
    omni = ModelInfo(repo_id="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4")
    assert "image_pin_nemotron_omni" in codes(adapter, omni, {}, hw)


def test_misc_rules(adapter, hw, dense) -> None:  # type: ignore[no-untyped-def]
    c = codes(adapter, dense, {"speculative_config": {"method": "dflash"}, "enforce_eager": True,
                               "trust_remote_code": True, "attention_backend": "CUTLASS_MLA"}, hw)
    assert {"dflash_max_num_seqs", "eager_slow", "trust_remote_code", "mla_backend_arch"} <= set(c)
    assert c["mla_backend_arch"].level == "block"
    assert "ep_dense_model" in codes(adapter, dense, {"enable_expert_parallel": True}, hw)
