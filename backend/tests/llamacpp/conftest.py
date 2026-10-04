"""Shared fixtures. Offline: only `adapters.base` and the llama.cpp adapter are imported; no core code, no docker, no GPU."""
from __future__ import annotations

import pytest

from engine_console.adapters.base import Hardware, ModelInfo
from engine_console.adapters.llamacpp import LlamaCppAdapter


@pytest.fixture
def adapter() -> LlamaCppAdapter:
    return LlamaCppAdapter()


@pytest.fixture
def hw() -> Hardware:
    """One Blackwell card with 90 GiB free."""
    return Hardware(gpu_ids=[0], gpu_uuids=["GPU-aaaa"], gpu_names=["RTX PRO 6000 Blackwell"], gpu_total_gib=[95.6],
                    gpu_free_gib=[90.0], compute_capability="12.0", host_ram_gib=246.0)


@pytest.fixture
def hw2() -> Hardware:
    return Hardware(gpu_ids=[0, 1], gpu_uuids=["GPU-aaaa", "GPU-bbbb"], gpu_names=["RTX PRO 6000 Blackwell"] * 2,
                    gpu_total_gib=[95.6, 95.6], gpu_free_gib=[90.0, 40.0], compute_capability="12.0", host_ram_gib=246.0)


@pytest.fixture
def dense() -> ModelInfo:
    """A 32B-class dense GGUF (Q4_K_M, 18.6 GiB): 64 layers, 8 KV heads of 128 -> 256 KiB of f16 KV per token."""
    return ModelInfo(repo_id="example-org/chat-model-32B-GGUF", architectures=["qwen3"], model_type="qwen3",
                     num_params=32_000_000_000, weight_bytes=int(18.6 * 1024 ** 3), quantization="q4_k_m",
                     hidden_size=5120, num_layers=64, num_attention_heads=64, num_kv_heads=8, head_dim=128,
                     max_position_embeddings=40960)


@pytest.fixture
def moe() -> ModelInfo:
    """A 30B-A3B-class mixture-of-experts GGUF (17.5 GiB): 48 layers, 4 KV heads of 128, 8 of 128 experts per token."""
    return ModelInfo(repo_id="example-org/moe-30B-A3B-GGUF", architectures=["qwen3moe"], model_type="qwen3moe",
                     num_params=30_500_000_000, num_active_params=3_300_000_000, is_moe=True,
                     weight_bytes=int(17.5 * 1024 ** 3), quantization="q4_k_m", hidden_size=2048, num_layers=48,
                     num_attention_heads=32, num_kv_heads=4, head_dim=128, max_position_embeddings=262144,
                     raw_config={"qwen3moe.expert_count": 128, "qwen3moe.expert_used_count": 8})
