"""compatibility() of the llama.cpp adapter: every rule, its level, and the fix it names."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import engine_console.adapters.llamacpp as module
from engine_console.adapters.base import Compat, Hardware, ModelInfo
from engine_console.adapters.llamacpp import LlamaCppAdapter

# every code the adapter can emit (the learn gate needs a `check.<code>` concept for each; the report lists them)
EXPECTED_CODES = {
    "blackwell_cuda_build", "fit_auto_adjust", "kv_quant_needs_flash_attn", "tensor_split_mismatch",
    "main_gpu_out_of_range", "split_mode_none_multi_gpu", "tp_over_pcie", "split_mode_tensor_experimental",
    "cpu_offload_slow", "context_beyond_native", "ctx_per_slot_small", "kv_exceeds_vram", "api_key_blocks_metrics",
    "jinja_disabled", "spec_type_needs_model", "moe_experts_on_cpu"}


def codes(adapter: LlamaCppAdapter, m: ModelInfo, p: dict[str, Any], hw: Hardware) -> dict[str, Compat]:
    found = adapter.compatibility(m, p, hw)
    assert len({c.code for c in found}) == len(found), "a code was emitted twice"
    return {c.code: c for c in found}


def test_the_emitted_codes_are_exactly_the_documented_ones() -> None:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    emitted = {n.args[1].value for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "add"
               and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str)}
    assert emitted == EXPECTED_CODES


def test_blackwell_note_only_on_compute_capability_12(adapter: LlamaCppAdapter, hw: Hardware, dense: ModelInfo) -> None:
    c = codes(adapter, dense, {}, hw)["blackwell_cuda_build"]
    assert c.level == "ok" and "sm_120a" in c.message and "CUDA 12.8" in c.message
    older = hw.model_copy(update={"compute_capability": "8.9"})
    assert "blackwell_cuda_build" not in codes(adapter, dense, {}, older)


def test_engine_fit_is_explained_when_something_is_left_unset(adapter: LlamaCppAdapter, hw: Hardware, dense: ModelInfo) -> None:
    c = codes(adapter, dense, {}, hw)["fit_auto_adjust"]
    assert c.level == "ok" and "ctx_size and n_gpu_layers" in c.message and "1024 MiB" in c.message
    only_layers = codes(adapter, dense, {"ctx_size": 8192}, hw)["fit_auto_adjust"]
    assert "n_gpu_layers" in only_layers.message and "ctx_size" not in only_layers.message
    assert "fit_auto_adjust" not in codes(adapter, dense, {"ctx_size": 8192, "n_gpu_layers": "all"}, hw)
    assert "2048 MiB" in codes(adapter, dense, {"fit_target": 2048}, hw)["fit_auto_adjust"].message


def test_a_missing_fit_with_no_context_is_a_warning_with_the_cost(adapter: LlamaCppAdapter, hw: Hardware, dense: ModelInfo) -> None:
    off = codes(adapter, dense, {"fit": "off"}, hw)["fit_auto_adjust"]
    assert off.level == "warn" and "fit is off" in off.message and "10.0 GiB" in off.message and "ctx_size" in off.message
    for params, needle in (({"tensor_split": "3,1"}, "tensor_split"), ({"override_tensor": "exps=CPU"}, "override_tensor"),
                           ({"split_mode": "row"}, "split_mode=row"), ({"split_mode": "tensor"}, "split_mode=tensor")):
        c = codes(adapter, dense, params, hw)["fit_auto_adjust"]
        assert c.level == "warn" and needle in c.message and "turns the automatic fit off" in c.message
    assert "fit_auto_adjust" not in codes(adapter, dense, {"fit": "off", "ctx_size": 8192}, hw)
    no_facts = ModelInfo(repo_id="x")
    assert "GiB" not in codes(adapter, no_facts, {"fit": "off"}, hw)["fit_auto_adjust"].message


def test_quantized_value_cache_needs_flash_attention(adapter: LlamaCppAdapter, hw: Hardware, dense: ModelInfo) -> None:
    c = codes(adapter, dense, {"cache_type_v": "q8_0", "flash_attn": "off"}, hw)["kv_quant_needs_flash_attn"]
    assert c.level == "block" and "V cache quantization requires flash_attn" in c.message and "flash_attn to on" in c.message
    assert "kv_quant_needs_flash_attn" not in codes(adapter, dense, {"cache_type_v": "q8_0", "flash_attn": "auto"}, hw)
    assert "kv_quant_needs_flash_attn" not in codes(adapter, dense, {"cache_type_v": "q8_0"}, hw)
    assert "kv_quant_needs_flash_attn" not in codes(adapter, dense, {"cache_type_k": "q4_0", "flash_attn": "off"}, hw)
    assert "kv_quant_needs_flash_attn" not in codes(adapter, dense, {"cache_type_v": "bf16", "flash_attn": "off"}, hw)


def test_gpu_placement_rules(adapter: LlamaCppAdapter, hw: Hardware, hw2: Hardware, dense: ModelInfo) -> None:
    assert codes(adapter, dense, {"tensor_split": "3,1"}, hw)["tensor_split_mismatch"].level == "warn"
    assert "tensor_split_mismatch" not in codes(adapter, dense, {"tensor_split": "3,1"}, hw2)
    assert "tensor_split_mismatch" not in codes(adapter, dense, {"tensor_split": "1,0"}, hw)
    assert codes(adapter, dense, {"main_gpu": 1}, hw)["main_gpu_out_of_range"].level == "block"
    assert "main_gpu_out_of_range" not in codes(adapter, dense, {"main_gpu": 1}, hw2)
    assert "main_gpu_out_of_range" not in codes(adapter, dense, {"main_gpu": 0}, hw)
    idle = codes(adapter, dense, {"split_mode": "none"}, hw2)["split_mode_none_multi_gpu"]
    assert idle.level == "warn" and "2 are selected" in idle.message
    assert "split_mode_none_multi_gpu" not in codes(adapter, dense, {"split_mode": "none"}, hw)


def test_row_and_tensor_splits_warn_about_pcie(adapter: LlamaCppAdapter, hw: Hardware, hw2: Hardware, dense: ModelInfo) -> None:
    for mode in ("row", "tensor"):
        c = codes(adapter, dense, {"split_mode": mode}, hw2)["tp_over_pcie"]
        assert c.level == "warn" and "NVLink" in c.message and "PCIe" in c.message
        assert "tp_over_pcie" not in codes(adapter, dense, {"split_mode": mode}, hw)
    assert "tp_over_pcie" not in codes(adapter, dense, {"split_mode": "layer"}, hw2)
    assert "tp_over_pcie" not in codes(adapter, dense, {}, hw2)


def test_tensor_split_mode_is_experimental_and_needs_flash_attention(adapter: LlamaCppAdapter, hw2: Hardware, dense: ModelInfo) -> None:
    c = codes(adapter, dense, {"split_mode": "tensor"}, hw2)["split_mode_tensor_experimental"]
    assert c.level == "warn" and "EXPERIMENTAL" in c.message
    blocked = codes(adapter, dense, {"split_mode": "tensor", "flash_attn": "off"}, hw2)["split_mode_tensor_experimental"]
    assert blocked.level == "block" and "flash_attn to on or auto" in blocked.message
    assert "split_mode_tensor_experimental" not in codes(adapter, dense, {"split_mode": "row"}, hw2)


def test_cpu_offload_is_called_slow(adapter: LlamaCppAdapter, hw: Hardware, dense: ModelInfo) -> None:
    part = codes(adapter, dense, {"n_gpu_layers": 20}, hw)["cpu_offload_slow"]
    assert part.level == "warn" and "20 of 65 layers" in part.message
    for params in ({"n_gpu_layers": 0}, {"device": "none"}, {"n_gpu_layers": "0"}):
        assert "whole model runs on the CPU" in codes(adapter, dense, params, hw)["cpu_offload_slow"].message
    for params in ({}, {"n_gpu_layers": "all"}, {"n_gpu_layers": "auto"}, {"n_gpu_layers": 65}, {"n_gpu_layers": 999}):
        assert "cpu_offload_slow" not in codes(adapter, dense, params, hw), params
    assert "cpu_offload_slow" not in codes(adapter, ModelInfo(repo_id="x"), {"n_gpu_layers": 20}, hw)   # layer count unknown


def test_context_past_the_trained_length(adapter: LlamaCppAdapter, hw: Hardware, dense: ModelInfo) -> None:
    c = codes(adapter, dense, {"ctx_size": 131072, "n_gpu_layers": "all"}, hw)["context_beyond_native"]
    assert c.level == "warn" and "40960" in c.message
    assert "rope_scaling=yarn" in c.message and "rope_scale=4" in c.message and "yarn_orig_ctx=40960" in c.message
    scaled = codes(adapter, dense, {"ctx_size": 131072, "rope_scaling": "yarn", "rope_scale": 4}, hw)["context_beyond_native"]
    assert "stretches" in scaled.message and "rope_scaling=yarn with" not in scaled.message
    assert "context_beyond_native" not in codes(adapter, dense, {"ctx_size": 40960}, hw)
    assert "context_beyond_native" not in codes(adapter, dense, {}, hw)
    # slots divide the context unless they share one buffer
    assert "context_beyond_native" not in codes(adapter, dense, {"ctx_size": 131072, "parallel": 4}, hw)
    assert "context_beyond_native" in codes(adapter, dense, {"ctx_size": 131072, "parallel": 4, "kv_unified": True}, hw)
    assert "context_beyond_native" in codes(adapter, dense, {"ctx_size": 131072, "parallel": -1}, hw)


def test_context_per_slot(adapter: LlamaCppAdapter, hw: Hardware, dense: ModelInfo) -> None:
    c = codes(adapter, dense, {"ctx_size": 8192, "parallel": 8}, hw)["ctx_per_slot_small"]
    assert c.level == "warn" and "1024 tokens" in c.message and "kv_unified" in c.message
    assert "ctx_per_slot_small" not in codes(adapter, dense, {"ctx_size": 8192, "parallel": 8, "kv_unified": True}, hw)
    assert "ctx_per_slot_small" not in codes(adapter, dense, {"ctx_size": 65536, "parallel": 8}, hw)
    assert "ctx_per_slot_small" not in codes(adapter, dense, {"ctx_size": 8192}, hw)
    assert "ctx_per_slot_small" not in codes(adapter, dense, {"ctx_size": 8192, "parallel": 1}, hw)


def test_kv_cache_against_free_vram(adapter: LlamaCppAdapter, hw: Hardware, dense: ModelInfo) -> None:
    c = codes(adapter, dense, {"ctx_size": 400_000, "n_gpu_layers": "all"}, hw)["kv_exceeds_vram"]
    assert c.level == "warn" and "97.7 GiB" in c.message and "90.0 GiB" in c.message and "q8_0" in c.message
    assert "kv_exceeds_vram" not in codes(adapter, dense, {"ctx_size": 400_000, "cache_type_k": "q8_0", "cache_type_v": "q8_0"}, hw)
    assert "kv_exceeds_vram" not in codes(adapter, dense, {"ctx_size": 400_000, "kv_offload": False}, hw)
    assert "kv_exceeds_vram" not in codes(adapter, dense, {"ctx_size": 32768}, hw)
    assert "kv_exceeds_vram" not in codes(adapter, dense, {}, hw)                    # context unset: the engine's fit decides
    small = hw.model_copy(update={"gpu_free_gib": [10.0]})
    assert "kv_exceeds_vram" in codes(adapter, dense, {"ctx_size": 65536}, small)    # 16 GiB of KV into 10 GiB free


def test_kv_check_uses_the_main_gpu_when_not_splitting(adapter: LlamaCppAdapter, hw2: Hardware, dense: ModelInfo) -> None:
    both = codes(adapter, dense, {"ctx_size": 400_000}, hw2)          # 97.7 GiB into 130 GiB free over two GPUs
    assert "kv_exceeds_vram" not in both
    one = codes(adapter, dense, {"ctx_size": 400_000, "split_mode": "none"}, hw2)
    assert "kv_exceeds_vram" in one and "90.0 GiB" in one["kv_exceeds_vram"].message


def test_api_key_warns_that_metrics_are_lost(adapter: LlamaCppAdapter, hw: Hardware, dense: ModelInfo) -> None:
    c = codes(adapter, dense, {"api_key": "distinctive-key-value"}, hw)["api_key_blocks_metrics"]
    assert c.level == "warn" and "401" in c.message and "/metrics" in c.message
    assert "distinctive-key-value" not in c.message                                   # the value itself never appears
    assert "api_key_blocks_metrics" not in codes(adapter, dense, {}, hw)


def test_jinja_disabled(adapter: LlamaCppAdapter, hw: Hardware, dense: ModelInfo) -> None:
    assert codes(adapter, dense, {"jinja": False}, hw)["jinja_disabled"].level == "warn"
    assert "jinja_disabled" not in codes(adapter, dense, {"jinja": True}, hw)
    assert "jinja_disabled" not in codes(adapter, dense, {}, hw)


def test_speculative_types_need_their_model(adapter: LlamaCppAdapter, hw: Hardware, dense: ModelInfo) -> None:
    for t in ("draft-simple", "draft-eagle3", "draft-dflash"):
        c = codes(adapter, dense, {"spec_type": t}, hw)["spec_type_needs_model"]
        assert c.level == "warn" and "spec_draft_model" in c.message
        assert "spec_type_needs_model" not in codes(adapter, dense, {"spec_type": t, "spec_draft_model": "/models/d.gguf"}, hw)
    mtp = codes(adapter, dense, {"spec_type": "draft-mtp"}, hw)["spec_type_needs_model"]
    assert "MTP layers" in mtp.message
    for t in ("none", "ngram-mod", "ngram-simple"):
        assert "spec_type_needs_model" not in codes(adapter, dense, {"spec_type": t}, hw)


def test_moe_experts_on_cpu(adapter: LlamaCppAdapter, hw: Hardware, dense: ModelInfo, moe: ModelInfo) -> None:
    c = codes(adapter, moe, {"cpu_moe": True}, hw)["moe_experts_on_cpu"]
    assert c.level == "ok" and "host RAM" in c.message and "load_mode none" in c.message
    assert "load_mode none" not in codes(adapter, moe, {"cpu_moe": True, "load_mode": "none"}, hw)["moe_experts_on_cpu"].message
    for params in ({"n_cpu_moe": 20}, {"override_tensor": "exps=CPU"}, {"override_tensor": r"blk\.0\.=CPU"}):
        assert codes(adapter, moe, params, hw)["moe_experts_on_cpu"].level == "ok", params
    wrong = codes(adapter, dense, {"cpu_moe": True}, hw)["moe_experts_on_cpu"]
    assert wrong.level == "warn" and "not one" in wrong.message
    assert "moe_experts_on_cpu" not in codes(adapter, moe, {}, hw)
    assert "moe_experts_on_cpu" not in codes(adapter, moe, {"n_cpu_moe": 0}, hw)
    assert "moe_experts_on_cpu" not in codes(adapter, moe, {"override_tensor": "blk.0=CUDA0"}, hw)


def test_a_clean_launch_has_only_notes(adapter: LlamaCppAdapter, hw: Hardware, dense: ModelInfo) -> None:
    found = adapter.compatibility(dense, {"ctx_size": 16384, "n_gpu_layers": "all", "flash_attn": "on"}, hw)
    assert [c.code for c in found] == ["blackwell_cuda_build"] and found[0].level == "ok"
    assert all(c.level in ("ok", "warn", "block") and c.message for c in found)
