"""Shared fixtures. Offline: only `adapters.base` and the TensorRT-LLM adapter are imported; no core code, no docker."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from engine_console.adapters.base import Hardware, ModelInfo
from engine_console.adapters.trtllm import TrtllmAdapter

FIX = Path(__file__).parent / "fixtures"


def read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def read_json(name: str) -> Any:
    return json.loads(read(name))


@pytest.fixture
def adapter() -> TrtllmAdapter:
    return TrtllmAdapter()


@pytest.fixture
def hw() -> Hardware:
    return Hardware(gpu_ids=[0, 1], gpu_uuids=["GPU-aaaa", "GPU-bbbb"],
                    gpu_names=["RTX PRO 6000 Blackwell"] * 2, gpu_total_gib=[95.6, 95.6],
                    gpu_free_gib=[94.0, 94.0], compute_capability="12.0", host_ram_gib=246.0)


@pytest.fixture
def hw1(hw: Hardware) -> Hardware:
    """One selected GPU."""
    return hw.model_copy(update={"gpu_ids": [0], "gpu_uuids": ["GPU-aaaa"], "gpu_names": hw.gpu_names[:1],
                                 "gpu_total_gib": [95.6], "gpu_free_gib": [94.0]})


@pytest.fixture
def dense() -> ModelInfo:
    """A Llama-family dense model with FP8-sized weights: 36 GiB."""
    return ModelInfo(repo_id="example-org/chat-model-36B", architectures=["LlamaForCausalLM"],
                     model_type="llama", num_params=36_000_000_000, weight_bytes=36 * 1024 ** 3,
                     num_attention_heads=64, num_kv_heads=8, max_position_embeddings=131072,
                     pipeline_tag="text-generation")
