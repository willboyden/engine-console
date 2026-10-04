"""Shared fixtures. Offline: only `adapters.base` and the Ollama adapter are imported; no core code, no docker, no GPU."""
from __future__ import annotations

import pytest

from engine_console.adapters.base import Hardware, ModelInfo
from engine_console.adapters.ollama import OllamaAdapter


@pytest.fixture
def adapter() -> OllamaAdapter:
    return OllamaAdapter()


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
def small_gpu() -> Hardware:
    """A 16 GiB card with 14 GiB free (Ada, compute capability 8.9)."""
    return Hardware(gpu_ids=[0], gpu_uuids=["GPU-cccc"], gpu_names=["RTX 4080"], gpu_total_gib=[16.0],
                    gpu_free_gib=[14.0], compute_capability="8.9", host_ram_gib=64.0)


@pytest.fixture
def llama8b() -> ModelInfo:
    """An 8B-class dense model from an Ollama store: 32 layers, 8 KV heads of 128 -> 128 KiB of f16 KV per token."""
    return ModelInfo(repo_id="llama3.1:8b", architectures=["llama"], model_type="llama", num_params=8_000_000_000,
                     weight_bytes=int(4.6 * 1024 ** 3), quantization="q4_k_m", hidden_size=4096, num_layers=32,
                     num_attention_heads=32, num_kv_heads=8, head_dim=128, max_position_embeddings=131072)


@pytest.fixture
def bare() -> ModelInfo:
    """What the library knows about an Ollama model whose config blob carries no layer or head counts."""
    return ModelInfo(repo_id="mystery:latest", weight_bytes=int(8 * 1024 ** 3))
