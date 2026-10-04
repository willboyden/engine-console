"""Compatibility rules: every code has a positive and a negative case, and every message carries a fix."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from engine_console.adapters import trtllm as trtllm_module
from engine_console.adapters.base import Compat, Hardware, ModelInfo
from engine_console.adapters.trtllm import TrtllmAdapter, _inferred_max_seq_len

from .conftest import read_json


def codes(adapter: TrtllmAdapter, m: ModelInfo, p: dict[str, Any], hw: Hardware) -> dict[str, Compat]:
    return {c.code: c for c in adapter.compatibility(m, p, hw)}


def test_clean_launch_has_no_findings(adapter: TrtllmAdapter, hw1: Hardware, dense: ModelInfo) -> None:
    assert codes(adapter, dense, {}, hw1) == {}


def test_unsupported_architecture_blocks(adapter: TrtllmAdapter, hw1: Hardware) -> None:
    new = ModelInfo(repo_id="example-org/newfam-30B", architectures=["Qwen3_5MoeForConditionalGeneration"])
    c = codes(adapter, new, {}, hw1)["unsupported_architecture"]
    assert c.level == "block" and "vLLM or SGLang" in c.message and "Qwen3_5MoeForConditionalGeneration" in c.message
    assert "unsupported_architecture" not in codes(adapter, ModelInfo(repo_id="a/b", architectures=["Qwen3ForCausalLM"]), {}, hw1)
    assert "unsupported_architecture" not in codes(adapter, ModelInfo(repo_id="a/b"), {}, hw1)   # unknown: no claim
    # only the FIRST architecture counts (AutoModelForCausalLM.from_config reads architectures[0])
    assert "unsupported_architecture" in codes(
        adapter, ModelInfo(repo_id="a/b", architectures=["WeirdModel", "LlamaForCausalLM"]), {}, hw1)


def test_world_size_rules(adapter: TrtllmAdapter, hw: Hardware, hw1: Hardware, dense: ModelInfo) -> None:
    assert codes(adapter, dense, {"tensor_parallel_size": 4}, hw)["world_exceeds_gpus"].level == "block"
    assert codes(adapter, dense, {"tensor_parallel_size": 2, "pipeline_parallel_size": 2}, hw)["world_exceeds_gpus"]
    assert "world_exceeds_gpus" not in codes(adapter, dense, {"tensor_parallel_size": 2}, hw)
    unused = codes(adapter, dense, {}, hw)["gpus_unused"]
    assert unused.level == "warn" and "2 GPUs" in unused.message
    assert "gpus_unused" not in codes(adapter, dense, {"pipeline_parallel_size": 2}, hw)
    assert "gpus_unused" not in codes(adapter, dense, {}, hw1)


def test_tensor_parallel_over_pcie_prefers_pipeline(adapter: TrtllmAdapter, hw: Hardware, dense: ModelInfo) -> None:
    c = codes(adapter, dense, {"tensor_parallel_size": 2}, hw)["tp_over_pcie"]
    assert c.level == "warn" and "pipeline" in c.message.lower() and "RTX PRO 6000" in c.message
    assert "tp_over_pcie" not in codes(adapter, dense, {"pipeline_parallel_size": 2}, hw)


def test_heads_must_divide(adapter: TrtllmAdapter, dense: ModelInfo) -> None:
    hw8 = Hardware(gpu_ids=list(range(8)), gpu_uuids=[f"g{i}" for i in range(8)], gpu_names=["x"] * 8,
                   gpu_total_gib=[95.6] * 8, gpu_free_gib=[94.0] * 8, compute_capability="12.0", host_ram_gib=512.0)
    odd = dense.model_copy(update={"num_attention_heads": 36})
    assert codes(adapter, odd, {"tensor_parallel_size": 8}, hw8)["tp_heads_indivisible"].level == "block"
    assert "tp_heads_indivisible" not in codes(adapter, odd, {"tensor_parallel_size": 4}, hw8)
    assert "tp_heads_indivisible" not in codes(adapter, odd, {"tensor_parallel_size": 8, "enable_attention_dp": True}, hw8)


def test_speculation_conflicts(adapter: TrtllmAdapter, hw: Hardware, dense: ModelInfo) -> None:
    pp = {"pipeline_parallel_size": 2, "spec_decoding_type": "MTP"}
    assert codes(adapter, dense, pp, hw)["pp_with_speculation"].level == "block"
    assert "pp_with_speculation" in codes(adapter, dense, {"pipeline_parallel_size": 2,
                                                           "speculative_config": {"decoding_type": "Eagle"}}, hw)
    assert "pp_with_speculation" not in codes(adapter, dense, {"pipeline_parallel_size": 2,
                                                               "spec_decoding_type": "AUTO"}, hw)
    clash = codes(adapter, dense, {"sampler_type": "TRTLLMSampler", "spec_decoding_type": "MTP"}, hw)
    assert clash["trtllm_sampler_conflict"].level == "block" and "auto" in clash["trtllm_sampler_conflict"].message
    assert "trtllm_sampler_conflict" not in codes(adapter, dense, {"sampler_type": "TRTLLMSampler"}, hw)


def test_moe_backend_rules(adapter: TrtllmAdapter, hw1: Hardware, dense: ModelInfo) -> None:
    c = codes(adapter, dense, {"moe_backend": "TRTLLM"}, hw1)["moe_backend_trtllm_sm120"]
    assert c.level == "block" and "CUTLASS" in c.message and "SM120" in c.message
    for backend in ("DEEPGEMM", "CUTEDSL", "WIDEEP", "TRITON"):
        assert codes(adapter, dense, {"moe_backend": backend}, hw1)["moe_backend_datacenter_only"].level == "warn"
    triton = codes(adapter, dense, {"moe_backend": "TRITON", "tensor_parallel_size": 2, "moe_expert_parallel_size": 2}, hw1)
    assert triton["moe_backend_triton_ep"].level == "block" and "Hopper" in triton["moe_backend_triton_ep"].message
    assert "moe_backend_triton_ep" not in codes(adapter, dense, {"moe_backend": "TRITON"}, hw1)
    hopper_ep = hw1.model_copy(update={"compute_capability": "9.0"})
    assert not codes(adapter, dense, {"moe_backend": "TRITON", "moe_expert_parallel_size": 2}, hopper_ep).keys() & {
        "moe_backend_triton_ep", "moe_backend_datacenter_only"}
    assert not codes(adapter, dense, {"moe_backend": "CUTLASS"}, hw1)
    hopper = hw1.model_copy(update={"compute_capability": "9.0"})
    assert "moe_backend_trtllm_sm120" not in codes(adapter, dense, {"moe_backend": "TRTLLM"}, hopper)
    gb10 = hw1.model_copy(update={"compute_capability": "12.1"})                    # sm_121 also raises in the engine
    assert "moe_backend_trtllm_sm120" in codes(adapter, dense, {"moe_backend": "TRTLLM"}, gb10)


def test_quantization_rules(adapter: TrtllmAdapter, hw1: Hardware) -> None:
    block = ModelInfo(repo_id="a/b-FP8", quantization="fp8", raw_config={"quantization_config": {
        "quant_method": "fp8", "weight_block_size": [128, 128]}})
    assert codes(adapter, block, {}, hw1)["fp8_block_scaling_sm120"].level == "warn"
    per_tensor = ModelInfo(repo_id="a/b-FP8", quantization="fp8", raw_config={"quantization_config": {"quant_method": "fp8"}})
    assert "fp8_block_scaling_sm120" not in codes(adapter, per_tensor, {}, hw1)
    assert "awq_gptq_sm120" in codes(adapter, ModelInfo(repo_id="a/b", quantization="awq"), {}, hw1)
    assert "awq_gptq_sm120" in codes(adapter, ModelInfo(repo_id="a/b", quantization="gptq"), {}, hw1)
    assert "awq_gptq_sm120" not in codes(adapter, ModelInfo(repo_id="a/b", quantization="awq"),
                                         {}, hw1.model_copy(update={"compute_capability": "9.0"}))
    plain = ModelInfo(repo_id="a/b")
    assert codes(adapter, plain, {"kv_cache_dtype": "nvfp4"}, hw1)["nvfp4_kv_sm120"].level == "warn"
    assert "nvfp4_kv_sm120" not in codes(adapter, plain, {"kv_cache_dtype": "fp8"}, hw1)
    assert "quant_format_unlisted" in codes(adapter, ModelInfo(repo_id="a/b", quantization="bitsandbytes"), {}, hw1)
    assert "quant_format_unlisted" not in codes(adapter, ModelInfo(repo_id="a/b", quantization="nvfp4"), {}, hw1)
    assert "mxfp4_family_unlisted" in codes(adapter, ModelInfo(repo_id="acme/moe-mxfp4", quantization="mxfp4"), {}, hw1)
    assert "mxfp4_family_unlisted" not in codes(adapter, ModelInfo(repo_id="openai/gpt-oss-120b", quantization="mxfp4"),
                                                {}, hw1)
    # NVFP4 itself is listed for sm_120: dense and MoE alike, so no finding
    assert not codes(adapter, ModelInfo(repo_id="a/b-NVFP4", quantization="nvfp4", is_moe=True), {}, hw1)


def test_attention_backend_rules(adapter: TrtllmAdapter, hw1: Hardware, dense: ModelInfo) -> None:
    assert codes(adapter, dense, {"attention_backend": "VANILLA"}, hw1)["attention_backend_vanilla"].level == "warn"
    c = codes(adapter, dense, {"attention_backend": "FLASHINFER"}, hw1)["attention_backend_flashinfer_reuse"]
    assert "enable_block_reuse=false" in c.message
    assert "attention_backend_flashinfer_reuse" not in codes(
        adapter, dense, {"attention_backend": "FLASHINFER", "enable_block_reuse": False}, hw1)
    assert not codes(adapter, dense, {"attention_backend": "TRTLLM"}, hw1)


def test_max_seq_len_is_capped_by_the_engine(adapter: TrtllmAdapter, hw1: Hardware, dense: ModelInfo) -> None:
    c = codes(adapter, dense, {"max_seq_len": 262144}, hw1)["max_seq_len_capped"]
    assert c.level == "warn" and "131072" in c.message and "extend" in c.message
    assert "max_seq_len_capped" not in codes(adapter, dense, {"max_seq_len": 131072}, hw1)
    assert "max_seq_len_capped" not in codes(adapter, dense, {}, hw1)


def test_inferred_length_matches_the_engines_own_function(adapter: TrtllmAdapter) -> None:
    """The table in fixtures/infer_max_seq_len_1.2.1.json is the output of the engine's `infer_max_seq_len` for each
    config; the adapter's re-implementation must agree with every row."""
    rows = read_json("infer_max_seq_len_1.2.1.json")["cases"]
    assert len(rows) >= 15
    for row in rows:
        model = ModelInfo(repo_id="a/b", raw_config=row["config"])
        if not row["config"]:
            assert _inferred_max_seq_len(model) is None            # no config at all: the adapter claims nothing
            continue
        assert _inferred_max_seq_len(model) == row["inferred"], row["config"]


def test_inferred_length_drives_the_capped_warning(adapter: TrtllmAdapter, hw1: Hardware, dense: ModelInfo) -> None:
    linear = dense.model_copy(update={"raw_config": {"max_position_embeddings": 131072,
                                                     "rope_scaling": {"type": "linear", "factor": 2.0}}})
    assert "max_seq_len_capped" not in codes(adapter, linear, {"max_seq_len": 262144}, hw1)
    assert "max_seq_len_capped" in codes(adapter, linear, {"max_seq_len": 262145}, hw1)
    yarn = dense.model_copy(update={"raw_config": {"max_position_embeddings": 131072,
                                                   "rope_scaling": {"rope_type": "yarn", "factor": 4.0}}})
    assert "max_seq_len_capped" in codes(adapter, yarn, {"max_seq_len": 262144}, hw1)
    bare = ModelInfo(repo_id="a/b", raw_config={"hidden_size": 64})        # the engine falls back to 2048
    assert "2048" in codes(adapter, bare, {"max_seq_len": 4096}, hw1)["max_seq_len_capped"].message
    assert "max_seq_len_capped" not in codes(adapter, ModelInfo(repo_id="a/b"), {"max_seq_len": 10 ** 6}, hw1)


def test_memory_rules(adapter: TrtllmAdapter, hw1: Hardware, dense: ModelInfo) -> None:
    assert codes(adapter, dense, {"free_gpu_memory_fraction": 0.98}, hw1)["kv_cache_fraction_high"].level == "warn"
    assert "kv_cache_fraction_high" not in codes(adapter, dense, {"free_gpu_memory_fraction": 0.9}, hw1)
    huge = dense.model_copy(update={"weight_bytes": 140 * 1024 ** 3})
    c = codes(adapter, huge, {}, hw1)["weights_exceed_gpu"]
    assert c.level == "block" and "FP8" in c.message and "loads a checkpoint as it is" in c.message
    two = Hardware(gpu_ids=[0, 1], gpu_uuids=["a", "b"], gpu_names=["x"] * 2, gpu_total_gib=[95.6] * 2,
                   gpu_free_gib=[94.0] * 2, compute_capability="12.0", host_ram_gib=246.0)
    assert "weights_exceed_gpu" not in codes(adapter, huge, {"tensor_parallel_size": 2}, two)
    busy = hw1.model_copy(update={"gpu_free_gib": [30.0]})
    free = codes(adapter, dense, {}, busy)["insufficient_free_vram"]
    assert free.level == "block" and "30.0 GiB" in free.message and "Stop" in free.message
    assert "insufficient_free_vram" not in codes(adapter, dense, {}, hw1)


def test_parser_rules(adapter: TrtllmAdapter, hw1: Hardware) -> None:
    qwen = ModelInfo(repo_id="Qwen/Qwen3-32B", model_type="qwen3", pipeline_tag="text-generation")
    missing = codes(adapter, qwen, {}, hw1)
    assert "qwen3_coder" in missing["tool_parser_mismatch"].message and "qwen3" in missing["tool_parser_mismatch"].message
    assert "reasoning_parser=qwen3" in missing["reasoning_parser_mismatch"].message
    ok = codes(adapter, qwen, {"tool_parser": "qwen3_coder", "reasoning_parser": "qwen3"}, hw1)
    assert not ok
    wrong = codes(adapter, qwen, {"tool_parser": "kimi_k2", "reasoning_parser": "deepseek-r1"}, hw1)
    assert "does not match" in wrong["tool_parser_mismatch"].message
    assert "use qwen3" in wrong["reasoning_parser_mismatch"].message
    coder = ModelInfo(repo_id="Qwen/Qwen3-Coder-30B", model_type="qwen3_moe", pipeline_tag="text-generation")
    c = codes(adapter, coder, {}, hw1)
    assert "tool_parser=qwen3_coder." in c["tool_parser_mismatch"].message and "reasoning_parser_mismatch" not in c
    kimi = ModelInfo(repo_id="moonshotai/Kimi-K2-Instruct", pipeline_tag="text-generation")
    assert "tool_parser=kimi_k2" in codes(adapter, kimi, {}, hw1)["tool_parser_mismatch"].message
    r1 = ModelInfo(repo_id="deepseek-ai/DeepSeek-R1-0528", architectures=["DeepseekV3ForCausalLM"])
    both = codes(adapter, r1, {}, hw1)
    assert "deepseek_v3" in both["tool_parser_mismatch"].message and "deepseek-r1" in both["reasoning_parser_mismatch"].message
    llama = ModelInfo(repo_id="meta-llama/Llama-3.3-70B", pipeline_tag="text-generation")
    assert not codes(adapter, llama, {}, hw1)                              # no family claim without certainty
    embed = ModelInfo(repo_id="Qwen/Qwen3-Embedding-8B", pipeline_tag="feature-extraction")
    assert "tool_parser_mismatch" not in codes(adapter, embed, {}, hw1)


def test_checkpoint_read_ahead_needs_host_ram(adapter: TrtllmAdapter, hw1: Hardware, dense: ModelInfo) -> None:
    """The weight loader reads files into host memory before loading (weight_loader.prefetch_files)."""
    big = dense.model_copy(update={"weight_bytes": 120 * 1024 ** 3})
    c = codes(adapter, big, {}, hw1)["weights_prefetch_host_ram"]          # 120 GiB of 246 GiB
    assert c.level == "warn" and "120 GiB" in c.message and "246 GiB" in c.message and "90 %" in c.message
    assert "weights_prefetch_host_ram" not in codes(adapter, dense, {}, hw1)                  # 36 GiB
    assert "weights_prefetch_host_ram" not in codes(adapter, dense.model_copy(update={"weight_bytes": None}), {}, hw1)
    small_host = hw1.model_copy(update={"host_ram_gib": 64.0})
    assert "weights_prefetch_host_ram" in codes(adapter, dense, {}, small_host)               # 36 > 0.4 x 64


def test_misc_rules(adapter: TrtllmAdapter, hw1: Hardware, dense: ModelInfo) -> None:
    c = codes(adapter, dense, {"trust_remote_code": True, "enable_iter_perf_stats": False,
                               "enable_cuda_graphs": False}, hw1)
    assert {"trust_remote_code", "iter_stats_off", "cuda_graphs_off"} <= set(c)
    assert "iter_stats_off" not in codes(adapter, dense, {}, hw1)             # the console default is on
    arm = hw1.model_copy(update={"arch": "aarch64"})
    assert "image_arch_unverified" in codes(adapter, dense, {}, arm)
    other = {"image": "nvcr.io/nvidia/tensorrt-llm/release:1.2.1-arm64"}
    assert "image_arch_unverified" not in codes(adapter, dense, other, arm)
    c = codes(adapter, dense, {"image": "nvcr.io/nvidia/tensorrt-llm/release:1.3.0"}, hw1)["image_version_unverified"]
    assert c.level == "warn" and "1.2.1" in c.message and "1.3.0" in c.message
    assert "image_version_unverified" not in codes(adapter, dense, {"image": adapter.default_image}, hw1)
    vl = ModelInfo(repo_id="Qwen/Qwen2.5-VL-7B", architectures=["Qwen2_5_VLForConditionalGeneration"],
                   pipeline_tag="image-text-to-text")
    assert "multimodal_block_reuse" in codes(adapter, vl, {}, hw1)
    assert "multimodal_block_reuse" not in codes(adapter, vl, {"enable_block_reuse": False}, hw1)


def test_every_message_is_actionable_and_every_code_is_found_by_the_learn_scan(adapter: TrtllmAdapter) -> None:
    """The learn gate finds `add("<level>", "<code>", ...)` literals with ast; keep that true for this module."""
    src = Path(trtllm_module.__file__).read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "add" and len(node.args) >= 3:
            level, code = node.args[0], node.args[1]
            assert isinstance(level, ast.Constant) and level.value in ("ok", "warn", "block")
            assert isinstance(code, ast.Constant) and isinstance(code.value, str)
            found[code.value] = level.value
    assert len(found) >= 25
    assert all(c == c.lower() and " " not in c for c in found)
