"""Model fixtures for the fit estimator.

APPROXIMATIONS: field values are written from memory of each model family's public config.json and
published checkpoint sizes; they are close enough to exercise the estimator's formulas (GQA, MoE, MLA,
sliding window) but are NOT byte-exact copies of the Hub files. Do not use them as ground truth.
"""
from __future__ import annotations

from typing import Any

from engine_console.adapters.base import Hardware, ModelInfo

GIB = 1024**3


def qwen3_32b_dense(**kw: Any) -> ModelInfo:
    """Qwen3-32B-style dense GQA: 64 layers, 64 heads / 8 KV heads, head_dim 128 (~65.5 GB in bf16)."""
    base: dict[str, Any] = dict(
        repo_id="Qwen/Qwen3-32B", architectures=["Qwen3ForCausalLM"], model_type="qwen3", num_params=32_762_000_000,
        weight_bytes=65_530_000_000, dtype="bfloat16", hidden_size=5120, num_layers=64, num_attention_heads=64,
        num_kv_heads=8, head_dim=128, max_position_embeddings=40960,
        raw_config={"model_type": "qwen3", "num_hidden_layers": 64, "use_sliding_window": False, "sliding_window": None})
    return ModelInfo(**{**base, **kw})


def qwen3_30b_a3b_moe(**kw: Any) -> ModelInfo:
    """Qwen3-30B-A3B-style MoE: 48 layers, 32 heads / 4 KV heads, 128 experts (~61 GB bf16, all experts resident)."""
    base: dict[str, Any] = dict(
        repo_id="Qwen/Qwen3-30B-A3B", architectures=["Qwen3MoeForCausalLM"], model_type="qwen3_moe",
        num_params=30_500_000_000, num_active_params=3_300_000_000, is_moe=True, weight_bytes=61_000_000_000,
        dtype="bfloat16", hidden_size=2048, num_layers=48, num_attention_heads=32, num_kv_heads=4, head_dim=128,
        max_position_embeddings=40960, raw_config={"model_type": "qwen3_moe", "num_experts": 128})
    return ModelInfo(**{**base, **kw})


def llama33_70b(**kw: Any) -> ModelInfo:
    """Llama-3.3-70B-Instruct-style: 80 layers, 64 heads / 8 KV heads, head_dim 128 (~141 GB bf16)."""
    base: dict[str, Any] = dict(
        repo_id="meta-llama/Llama-3.3-70B-Instruct", architectures=["LlamaForCausalLM"], model_type="llama",
        num_params=70_553_000_000, weight_bytes=141_107_000_000, dtype="bfloat16", hidden_size=8192, num_layers=80,
        num_attention_heads=64, num_kv_heads=8, head_dim=128, max_position_embeddings=131072, gated=True,
        raw_config={"model_type": "llama"})
    return ModelInfo(**{**base, **kw})


def gpt_oss_120b(**kw: Any) -> ModelInfo:
    """gpt-oss-120b-style: 36 layers alternating sliding(128)/full, 64 heads / 8 KV heads, head_dim 64, MXFP4 (~65 GB)."""
    layer_types = ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(36)]
    base: dict[str, Any] = dict(
        repo_id="openai/gpt-oss-120b", architectures=["GptOssForCausalLM"], model_type="gpt_oss",
        num_params=116_800_000_000, num_active_params=5_100_000_000, is_moe=True, weight_bytes=65_000_000_000,
        quantization="mxfp4", hidden_size=2880, num_layers=36, num_attention_heads=64, num_kv_heads=8, head_dim=64,
        max_position_embeddings=131072, sliding_window=128,
        raw_config={"model_type": "gpt_oss", "layer_types": layer_types, "num_local_experts": 128})
    return ModelInfo(**{**base, **kw})


def deepseek_v2_lite_mla(**kw: Any) -> ModelInfo:
    """DeepSeek-V2-Lite-style MLA + MoE: 27 layers, kv_lora_rank 512, qk_rope_head_dim 64 (~31 GB bf16)."""
    base: dict[str, Any] = dict(
        repo_id="deepseek-ai/DeepSeek-V2-Lite", architectures=["DeepseekV2ForCausalLM"], model_type="deepseek_v2",
        num_params=15_700_000_000, is_moe=True, weight_bytes=31_400_000_000, dtype="bfloat16", hidden_size=2048,
        num_layers=27, num_attention_heads=16, num_kv_heads=16, head_dim=None, max_position_embeddings=163840,
        kv_lora_rank=512, raw_config={"model_type": "deepseek_v2", "kv_lora_rank": 512, "qk_rope_head_dim": 64,
                                       "n_routed_experts": 64})
    return ModelInfo(**{**base, **kw})


def hw(n: int = 1, total: float = 96.0, free: float | list[float] | None = None) -> Hardware:
    frees = [free] * n if isinstance(free, int | float) else (free or [total] * n)
    return Hardware(gpu_ids=list(range(n)), gpu_uuids=[f"GPU-{i}" for i in range(n)],
                    gpu_names=["RTX PRO 6000 Blackwell"] * n, gpu_total_gib=[total] * n, gpu_free_gib=list(frees),
                    compute_capability="12.0", host_ram_gib=246.0)
