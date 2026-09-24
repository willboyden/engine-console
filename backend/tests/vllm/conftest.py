"""Shared fixtures. Offline: only `adapters.base` and the vLLM adapter are imported; no core code."""
from __future__ import annotations

from pathlib import Path

import pytest

from engine_console.adapters.base import Hardware, ModelInfo
from engine_console.adapters.vllm import VllmAdapter

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def adapter() -> VllmAdapter:
    return VllmAdapter()


@pytest.fixture
def hw() -> Hardware:
    return Hardware(gpu_ids=[0, 1], gpu_uuids=["GPU-aaaa", "GPU-bbbb"],
                    gpu_names=["RTX PRO 6000 Blackwell"] * 2, gpu_total_gib=[95.6, 95.6],
                    gpu_free_gib=[94.0, 94.0], compute_capability="12.0", host_ram_gib=246.0)


@pytest.fixture
def dense() -> ModelInfo:
    return ModelInfo(repo_id="NousResearch/Hermes-4.3-36B", architectures=["LlamaForCausalLM"],
                     num_params=36_000_000_000, weight_bytes=72 * 1024 ** 3, num_attention_heads=64,
                     num_kv_heads=8, max_position_embeddings=131072, pipeline_tag="text-generation")


def read(name: str) -> str:
    return (FIX / name).read_text()
