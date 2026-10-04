from __future__ import annotations

from engine_console.adapters.base import Hardware, ModelInfo
from engine_console.adapters.vllm import VllmAdapter


def codes(adapter: VllmAdapter, m: ModelInfo, p: dict, hw: Hardware) -> dict:  # type: ignore[type-arg]
    return {c.code: c for c in adapter.compatibility(m, p, hw)}


def on(hw: Hardware, cc: str) -> Hardware:
    return hw.model_copy(update={"compute_capability": cc})


def test_nvfp4_moe_warns_unless_backend_chosen(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    m = ModelInfo(repo_id="nvidia/x-30B-A3B-NVFP4", quantization="nvfp4", is_moe=True)
    c = codes(adapter, m, {}, hw)["nvfp4_moe_sm120"]
    assert c.level == "warn" and "FP8" in c.message and "SGLang" in c.message and "flashinfer_b12x" in c.message
    assert "nvfp4_moe_sm120" not in codes(adapter, m, {"moe_backend": "cutlass"}, hw)
    assert "nvfp4_moe_sm120" not in codes(adapter, m, {}, on(hw, "10.0"))       # the premise is sm_120-specific


def test_nvfp4_dense_is_clean(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    m = ModelInfo(repo_id="unsloth/Qwen3.8-27B-NVFP4", quantization="nvfp4", is_moe=False)
    assert "nvfp4_moe_sm120" not in codes(adapter, m, {}, hw)


def test_mxfp4_moe_blocks_except_gpt_oss(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    bad = ModelInfo(repo_id="acme/moe-mxfp4", quantization="mxfp4", is_moe=True)
    assert codes(adapter, bad, {}, hw)["mxfp4_moe_sm120"].level == "block"
    assert "mxfp4_moe_sm120" not in codes(adapter, bad, {}, on(hw, "10.0"))
    oss = ModelInfo(repo_id="openai/gpt-oss-120b", quantization="mxfp4", is_moe=True)
    c = codes(adapter, oss, {}, hw)
    assert "mxfp4_moe_sm120" not in c and c["gpt_oss_prefer_ollama"].level == "warn"
    assert "llama.cpp" in c["gpt_oss_prefer_ollama"].message


def test_gated(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    c = codes(adapter, ModelInfo(repo_id="google/gemma-4-31B", gated=True), {}, hw)
    assert "HF_TOKEN" in c["gated_needs_hf_token"].message


def test_tp_exceeds_gpus_and_divisibility(adapter, hw, dense) -> None:  # type: ignore[no-untyped-def]
    assert codes(adapter, dense, {"tensor_parallel_size": 4}, hw)["tp_exceeds_gpus"].level == "block"
    c = codes(adapter, dense, {"tensor_parallel_size": 2}, hw)["tp_over_pcie"]
    assert c.level == "warn" and "NVLink" in c.message
    odd = dense.model_copy(update={"num_attention_heads": 36})
    assert "tp_heads_indivisible" in codes(adapter, odd, {"tensor_parallel_size": 8, "data_parallel_size": 1},
                                           hw.model_copy(update={"gpu_ids": list(range(8))}))
    assert codes(adapter, dense, {"data_parallel_size": 2, "tensor_parallel_size": 2}, hw)["tp_exceeds_gpus"]


def test_fp8_kv_with_flash_attn_is_refused_on_sm120(adapter, hw, dense) -> None:  # type: ignore[no-untyped-def]
    c = codes(adapter, dense, {"kv_cache_dtype": "fp8", "attention_backend": "FLASH_ATTN"}, hw)["kv_fp8_flash_attn"]
    assert c.level == "block" and "FLASHINFER" in c.message and "FlashAttention 2" in c.message
    assert "kv_fp8_flash_attn" not in codes(adapter, dense, {"kv_cache_dtype": "fp8",
                                                             "attention_backend": "FLASHINFER"}, hw)
    assert "kv_fp8_flash_attn" not in codes(adapter, dense, {"kv_cache_dtype": "fp8", "attention_backend": "FLASH_ATTN"},
                                            on(hw, "9.0"))          # FA3 on Hopper handles fp8 KV


def test_kv_dtype_must_be_supported_by_the_chosen_backend(adapter, hw, dense) -> None:  # type: ignore[no-untyped-def]
    c = codes(adapter, dense, {"kv_cache_dtype": "fp8_per_token_head", "attention_backend": "FLASHINFER"}, hw)
    assert c["kv_dtype_backend_mismatch"].level == "block" and "fp8_e5m2" in c["kv_dtype_backend_mismatch"].message
    assert "kv_dtype_backend_mismatch" not in codes(
        adapter, dense, {"kv_cache_dtype": "fp8_per_token_head", "attention_backend": "TRITON_ATTN"}, hw)
    # without an explicit backend vLLM picks one that fits, so nothing is said
    assert "kv_dtype_backend_mismatch" not in codes(adapter, dense, {"kv_cache_dtype": "fp8_per_token_head"}, hw)


def test_fp8_kv_accuracy_note_names_the_engine_warning(adapter, hw, dense) -> None:  # type: ignore[no-untyped-def]
    assert "uncalibrated" in codes(adapter, dense, {"kv_cache_dtype": "fp8"}, hw)["kv_fp8_accuracy"].message


def test_mla_backends_checked_against_sm120(adapter, hw, dense) -> None:  # type: ignore[no-untyped-def]
    for b in ("CUTLASS_MLA", "FLASHMLA", "FLASHINFER_MLA", "FLASH_ATTN_MLA"):
        c = codes(adapter, dense, {"attention_backend": b}, hw)["mla_backend_arch"]
        assert c.level == "block" and "TRITON_MLA" in c.message
        assert "mla_backend_arch" not in codes(adapter, dense, {"attention_backend": b}, on(hw, "10.0"))
    assert "mla_backend_arch" not in codes(adapter, dense, {"attention_backend": "TRITON_MLA"}, hw)
    assert "mla_backend_arch" not in codes(adapter, dense, {"attention_backend": "FLASHINFER_MLA_SPARSE_SM120"}, hw)


def test_kernel_backends_for_other_hardware_are_blocked(adapter, hw, dense) -> None:  # type: ignore[no-untyped-def]
    for p in ({"moe_backend": "aiter"}, {"linear_backend": "xpu"}, {"moe_backend": "hpc"}):
        assert codes(adapter, dense, p, hw)["kernel_backend_not_for_gpu"].level == "block"
    assert "kernel_backend_not_for_gpu" not in codes(adapter, dense, {"moe_backend": "hpc"}, on(hw, "9.0"))
    assert "kernel_backend_not_for_gpu" not in codes(adapter, dense, {"moe_backend": "flashinfer_b12x"}, hw)


def test_deep_gemm_needs_fp8_block_weights(adapter, hw, dense) -> None:  # type: ignore[no-untyped-def]
    assert codes(adapter, dense, {"moe_backend": "deep_gemm"}, hw)["deep_gemm_fp8_only"].level == "warn"
    fp8 = dense.model_copy(update={"quantization": "fp8"})
    assert "deep_gemm_fp8_only" not in codes(adapter, fp8, {"moe_backend": "deep_gemm"}, hw)
    assert "deep_gemm_sm120" not in codes(adapter, dense, {"moe_backend": "deep_gemm"}, hw)   # DeepGEMM does run on 12.x


def test_xxhash_prefix_hash_needs_a_package_the_image_lacks(adapter, hw, dense) -> None:  # type: ignore[no-untyped-def]
    for algo in ("xxhash", "xxhash_cbor"):
        c = codes(adapter, dense, {"prefix_caching_hash_algo": algo}, hw)["prefix_hash_xxhash_missing"]
        assert c.level == "warn" and "sha256" in c.message
    assert "prefix_hash_xxhash_missing" not in codes(adapter, dense, {"prefix_caching_hash_algo": "sha256_cbor"}, hw)


def test_tool_and_reasoning_parser_hints(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    m = ModelInfo(repo_id="Qwen/Qwen3.6-35B-A3B", pipeline_tag="text-generation")
    c = codes(adapter, m, {}, hw)
    assert "tool_call_parser=qwen3_xml" in c["tool_parser_missing"].message
    assert "reasoning_parser=qwen3" in c["reasoning_parser_missing"].message
    c2 = codes(adapter, m, {"tool_call_parser": "qwen3_xml", "enable_auto_tool_choice": True,
                            "reasoning_parser": "qwen3"}, hw)
    assert "tool_parser_missing" not in c2 and "reasoning_parser_missing" not in c2
    assert "tool_parser_dialect_mismatch" not in c2
    assert "tool_parser_inactive" in codes(adapter, m, {"tool_call_parser": "qwen3_xml"}, hw)
    coder = ModelInfo(repo_id="Qwen/Qwen3-Coder-30B", pipeline_tag="text-generation")
    assert "reasoning_parser_missing" not in codes(adapter, coder, {}, hw)


def test_tool_parser_dialect_mismatch(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    xml = ModelInfo(repo_id="Qwen/Qwen3.8-27B", pipeline_tag="text-generation")
    bad = codes(adapter, xml, {"tool_call_parser": "hermes", "enable_auto_tool_choice": True}, hw)
    assert bad["tool_parser_dialect_mismatch"].level == "warn" and "qwen3_xml" in bad["tool_parser_dialect_mismatch"].message
    for ok in ("qwen3_xml", "qwen3_coder"):                    # one parser class in the pinned image
        assert "tool_parser_dialect_mismatch" not in codes(
            adapter, xml, {"tool_call_parser": ok, "enable_auto_tool_choice": True}, hw)
    # Qwen3-8B is the older JSON dialect: the "3.x" regex must not swallow a size suffix such as "-8B"
    old = ModelInfo(repo_id="Qwen/Qwen3-8B", pipeline_tag="text-generation")
    assert "tool_parser_dialect_mismatch" not in codes(
        adapter, old, {"tool_call_parser": "hermes", "enable_auto_tool_choice": True}, hw)
    assert "tool_parser_dialect_mismatch" in codes(
        adapter, old, {"tool_call_parser": "qwen3_xml", "enable_auto_tool_choice": True}, hw)
    oss = ModelInfo(repo_id="openai/gpt-oss-20b", pipeline_tag="text-generation")
    assert "tool_parser_dialect_mismatch" not in codes(
        adapter, oss, {"tool_call_parser": "openai", "enable_auto_tool_choice": True}, hw)


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
    assert codes(adapter, dense, {"gpu_memory_utilization": 0.97}, hw)["util_very_high"].level == "warn"


def test_hybrid_models_need_an_explicit_max_num_seqs(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    """Image source (config/compilation.py): with full CUDA graphs, max_num_seqs above the Mamba cache blocks stops
    start-up, whatever the memory share."""
    hybrid = ModelInfo(repo_id="example-org/hybrid-moe-30B-A3B-NVFP4", architectures=["NemotronHForCausalLM"],
                       is_moe=True, quantization="nvfp4")
    for share in (0.3, 0.9):
        c = codes(adapter, hybrid, {"gpu_memory_utilization": share}, hw)["hybrid_max_num_seqs"]
        assert c.level == "warn" and "max_num_seqs" in c.message and "Mamba cache blocks" in c.message
    assert "hybrid_max_num_seqs" not in codes(adapter, hybrid, {"max_num_seqs": 256}, hw)
    assert "hybrid_max_num_seqs" not in codes(adapter, hybrid, {"enforce_eager": True}, hw)   # no full graphs, no check
    dense = ModelInfo(repo_id="a/dense", architectures=["LlamaForCausalLM"])
    assert "hybrid_max_num_seqs" not in codes(adapter, dense, {"gpu_memory_utilization": 0.3}, hw)
    qwen_next = ModelInfo(repo_id="Qwen/Qwen3-Next-80B-A3B", architectures=["Qwen3NextForCausalLM"], is_moe=True)
    assert "hybrid_max_num_seqs" in codes(adapter, qwen_next, {}, hw)


def test_image_family_note(adapter, hw) -> None:  # type: ignore[no-untyped-def]
    omni = ModelInfo(repo_id="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4")
    c = codes(adapter, omni, {}, hw)["image_pin_nemotron_omni"]
    assert c.level == "warn" and "0.20.0" in c.message and "0.27.1" in c.message
    laguna = ModelInfo(repo_id="poolside/Laguna-S-2.1-NVFP4")
    assert not {"image_too_old_laguna", "image_old_qwen38"} & set(codes(adapter, laguna, {}, hw))


def test_misc_rules(adapter, hw, dense) -> None:  # type: ignore[no-untyped-def]
    c = codes(adapter, dense, {"speculative_config": {"method": "dflash"}, "enforce_eager": True,
                               "trust_remote_code": True, "enable_sleep_mode": True, "cpu_offload_gb": 4}, hw)
    assert {"eager_slow", "trust_remote_code", "sleep_mode_dev_router", "cpu_offload_slow"} <= set(c)
    assert "dflash_max_num_seqs" not in c                       # the rule had no evidence behind it and was dropped
    assert "VLLM_SERVER_DEV_MODE" in c["sleep_mode_dev_router"].message
    assert "ep_dense_model" in codes(adapter, dense, {"enable_expert_parallel": True}, hw)


def test_every_message_is_generic_text(adapter, hw, dense) -> None:  # type: ignore[no-untyped-def]
    """The repository is public: no site-specific wording in anything a user is shown."""
    import re
    models = [dense, ModelInfo(repo_id="nvidia/x-NVFP4", quantization="nvfp4", is_moe=True, gated=True),
              ModelInfo(repo_id="openai/gpt-oss-120b", quantization="mxfp4", is_moe=True),
              ModelInfo(repo_id="nvidia/Nemotron-3-Nano-Omni-30B", architectures=["NemotronHForCausalLM"])]
    params = [{}, {"tensor_parallel_size": 2, "kv_cache_dtype": "fp8", "attention_backend": "FLASH_ATTN",
                   "gpu_memory_utilization": 0.3, "moe_backend": "deep_gemm", "enable_sleep_mode": True,
                   "speculative_config": {"method": "dflash"}, "tool_call_parser": "hermes", "max_model_len": 10 ** 7}]
    banned = re.compile(r"\blab\b|this repo|lightning|/home/|@[a-z0-9-]+\.[a-z]{2,}", re.I)
    seen = 0
    for m in models:
        for p in params:
            for c in adapter.compatibility(m, p, hw):
                seen += 1
                assert not banned.search(c.message), c
    assert seen > 10
