"""model_info for library copies: hf and plain from config.json (through the node), gguf and ollama from the scanned header;
and the pure GGUF/Ollama helpers behind it."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from engine_console.domain.refs import ModelRef
from engine_console.services.gguf_info import (
    model_info_from_entry,
    normalize_quant,
    parse_size_label,
    quant_from_file_type,
    shard_names,
)

from .conftest import World
from .fakes import COMMIT, FakeNode, RootCfg, entry, gguf_entry, hf_entry
from .helpers import nfs_alpha as alpha
from .helpers import nfs_beta as beta

QWEN: dict[str, Any] = {
    "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": 5120, "num_hidden_layers": 64,
    "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128, "max_position_embeddings": 40960,
    "torch_dtype": "bfloat16"}


def node(tmp: Path, **kw: Any) -> FakeNode:
    return FakeNode(tmp, "alpha", scan_disk=False, roots=[RootCfg("hf", "hf-home"), RootCfg("gguf", "gguf"),
                                                          RootCfg("ro", "plain", writable=False), RootCfg("ol", "ollama")], **kw)


# ---- hf and plain: config.json through the node --------------------------------------------------------------------------
async def test_hf_info_comes_from_the_nodes_config_and_the_copys_size(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "Qwen/Qwen3-32B", size=65_000_000_000, params=32_000_000_000)]
    a.configs[("hf", "hub/models--Qwen--Qwen3-32B")] = QWEN
    w = await World(tmp_path, [a]).start()
    info = await w.library.model_info("alpha", ModelRef(format="hf", repo_id="Qwen/Qwen3-32B"))
    assert info is not None
    assert (info.repo_id, info.revision, info.num_layers, info.num_kv_heads, info.head_dim) == ("Qwen/Qwen3-32B", COMMIT, 64, 8, 128)
    assert info.max_position_embeddings == 40960 and info.model_type == "qwen3" and info.dtype == "bfloat16"
    assert info.weight_bytes == 65_000_000_000 and info.num_params == 32_000_000_000
    assert a.ops("models.config") == [{"root": "hf", "rel": "hub/models--Qwen--Qwen3-32B"}]
    await w.stop()


async def test_info_is_cached_until_the_library_changes(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m")]
    a.configs[("hf", "hub/models--org--m")] = QWEN
    w = await World(tmp_path, [a]).start()
    ref = ModelRef(format="hf", repo_id="org/m")
    first = await w.library.model_info("alpha", ref)
    first.num_layers = 1                          # type: ignore[union-attr]  # a caller scribbling on its copy ...
    second = await w.library.model_info("alpha", ref)
    assert second.num_layers == 64 and len(a.ops("models.config")) == 1     # type: ignore[union-attr]  # ... changes nothing
    a.set_entries([hf_entry("hf", "org/m", size=5)])
    await w.library.refresh("alpha")
    assert (await w.library.model_info("alpha", ref)).weight_bytes == 5      # type: ignore[union-attr]
    assert len(a.ops("models.config")) == 2
    await w.stop()


async def test_a_missing_config_gives_only_what_is_known(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m", size=777, quant="fp8")]
    w = await World(tmp_path, [a]).start()
    info = await w.library.model_info("alpha", ModelRef(format="hf", repo_id="org/m"))
    assert info is not None and info.weight_bytes == 777 and info.quantization == "fp8"
    assert info.num_layers is None and info.num_kv_heads is None and info.head_dim is None      # never made up
    await w.stop()


async def test_plain_info_reads_the_directorys_config(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [entry("ro", "mydir", "plain", "mydir", size=4242)]
    a.configs[("ro", "mydir")] = {**QWEN, "quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}}
    w = await World(tmp_path, [a]).start()
    info = await w.library.model_info("alpha", ModelRef(format="plain", root_id="ro", rel="mydir"))
    assert info is not None and info.repo_id == "mydir" and info.quantization == "fp8" and info.weight_bytes == 4242
    await w.stop()


async def test_info_falls_back_to_a_machine_that_shares_the_storage_when_the_owner_is_down(tmp_path: Path) -> None:
    a, b = alpha(tmp_path), beta(tmp_path)
    a.entries = [hf_entry("hf", "org/m")]
    b.entries = [hf_entry("shared", "org/m")]
    b.configs[("shared", "hub/models--org--m")] = QWEN        # only beta can read it right now
    w = await World(tmp_path, [a, b]).start()
    a.state = "down"
    info = await w.library.model_info("beta", ModelRef(format="hf", repo_id="org/m"))
    assert info is not None and info.num_layers == 64 and not a.ops("models.config")
    await w.stop()


# ---- gguf and ollama: the header the node read ----------------------------------------------------------------------------------
async def test_gguf_info_from_the_nodes_header_facts(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [gguf_entry("gguf", "m/Qwen3-8B-Q4_K_M.gguf", name="Qwen3-8B", size=5_000_000_000, params=8_190_000_000,
                            arch="qwen3", context=40960, shards=None,
                            meta={"block_count": 36, "embedding_length": 4096, "head_count": 32, "head_count_kv": 8,
                                  "expert_count": None, "file_type": 15, "complete": True})]
    w = await World(tmp_path, [a]).start()
    info = await w.library.model_info("alpha", ModelRef(format="gguf", root_id="gguf", rel="m/Qwen3-8B-Q4_K_M.gguf"))
    assert info is not None and info.repo_id == "Qwen3-8B" and info.architectures == ["qwen3"] and info.model_type == "qwen3"
    assert (info.num_layers, info.num_attention_heads, info.num_kv_heads, info.hidden_size, info.head_dim) == (36, 32, 8, 4096, 128)
    assert info.max_position_embeddings == 40960 and info.num_params == 8_190_000_000 and info.weight_bytes == 5_000_000_000
    assert info.quantization == "q4_k_m" and info.is_moe is False and not a.ops("models.config")     # no node call needed
    await w.stop()


def test_gguf_info_reads_llama_cpp_key_names_flat_and_nested() -> None:
    flat = {"general.architecture": "llama", "llama.block_count": 32, "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8, "llama.embedding_length": 4096, "llama.context_length": 131072,
            "llama.attention.key_length": 128, "llama.expert_count": 8, "general.file_type": 18, "general.size_label": "8B",
            "tokenizer.ggml.tokens": ["a"] * 3, "general.name": "x"}
    nested = {"general": {"architecture": "llama", "file_type": 18, "size_label": "8B"},
              "llama": {"block_count": 32, "attention": {"head_count": 32, "head_count_kv": 8, "key_length": 128},
                        "embedding_length": 4096, "context_length": 131072, "expert_count": 8}}
    for meta in (flat, nested):
        info = model_info_from_entry("m", {"meta": meta, "size": 123}, "gguf")
        assert (info.num_layers, info.num_attention_heads, info.num_kv_heads, info.head_dim) == (32, 32, 8, 128), meta
        assert info.max_position_embeddings == 131072 and info.is_moe is True and info.quantization == "q6_k"
        assert info.num_params == 8_000_000_000 and info.weight_bytes == 123 and info.architectures == ["llama"]
    assert "tokenizer.ggml.tokens" not in model_info_from_entry("m", {"meta": flat}, "gguf").raw_config


def test_gguf_info_handles_per_layer_arrays_and_missing_keys() -> None:
    info = model_info_from_entry("m", {"arch": "hybrid", "meta": {"hybrid.block_count": 4, "hybrid.attention.head_count": [0, 8, 0, 8],
                                                                  "hybrid.attention.head_count_kv": [0, 2, 0, 4]}}, "gguf")
    assert info.num_attention_heads == 8 and info.num_kv_heads == 4 and info.head_dim is None      # the safe (largest) entry
    bare = model_info_from_entry("m", {}, "gguf")
    assert bare.num_layers is None and bare.num_kv_heads is None and bare.quantization is None and bare.architectures == []
    assert model_info_from_entry("m", {"meta": "not a dict", "size": -5}, "gguf").weight_bytes is None
    mha = model_info_from_entry("m", {"arch": "a", "meta": {"a.attention.head_count": 16, "a.embedding_length": 2048}}, "gguf")
    assert mha.num_kv_heads == 16 and mha.head_dim == 128                   # kv heads default to the head count, as in llama.cpp


def test_quant_helpers() -> None:
    assert quant_from_file_type(15) == "Q4_K_M" and quant_from_file_type(32) == "BF16" and quant_from_file_type(999) is None
    assert quant_from_file_type(True) is None and quant_from_file_type("15") is None
    assert normalize_quant("Q4_K_M") == "q4_k_m" and normalize_quant("UD-Q4_K_XL") == "q4_k_xl"
    assert normalize_quant("MXFP4_MOE") == "mxfp4" and normalize_quant("i1-Q4_K_M") == "q4_k_m" and normalize_quant("") is None
    assert parse_size_label("8.0B") == 8_000_000_000 and parse_size_label("335M") == 335_000_000 and parse_size_label("x") is None
    assert parse_size_label(None) is None


def test_shard_names() -> None:
    assert shard_names("m/x-00001-of-00003.gguf") == ["m/x-00001-of-00003.gguf", "m/x-00002-of-00003.gguf", "m/x-00003-of-00003.gguf"]
    assert shard_names("x-00001-of-00002.gguf", 2) == ["x-00001-of-00002.gguf", "x-00002-of-00002.gguf"]
    assert shard_names("x-00001-of-00002.gguf", 5) == ["x-00001-of-00002.gguf"]          # the entry disagrees: trust nothing
    assert shard_names("x-00002-of-00003.gguf") == ["x-00002-of-00003.gguf"]              # not the first shard
    assert shard_names("plain.gguf") == ["plain.gguf"] and shard_names("x-00001-of-00001.gguf") == ["x-00001-of-00001.gguf"]


async def test_ollama_info_from_the_config_blob_facts(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [entry("ol", "models/manifests/registry.ollama.ai/library/llama3/8b", "ollama", "llama3:8b", size=4_700_000_000,
                       meta={"model_format": "gguf", "model_family": "llama", "model_type": "8.0B", "file_type": "Q4_0"})]
    w = await World(tmp_path, [a]).start()
    info = await w.library.model_info("alpha", ModelRef(format="ollama", name="llama3:8b"))
    assert info is not None and info.repo_id == "llama3:8b" and info.model_type == "llama" and info.num_params == 8_000_000_000
    assert info.quantization == "q4_0" and info.weight_bytes == 4_700_000_000 and info.num_layers is None
    await w.stop()


async def test_no_info_without_a_copy_or_for_other_formats(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [entry("ro", "stuff", "other", "stuff")]
    w = await World(tmp_path, [a]).start()
    assert await w.library.model_info("alpha", ModelRef(format="hf", repo_id="org/none")) is None
    assert await w.library.model_info("alpha", ModelRef(format="plain", root_id="ro", rel="stuff")) is None
    assert await w.library.model_info("nobody", ModelRef(format="hf", repo_id="org/none")) is None
    await w.stop()


@pytest.mark.parametrize("fmt", ["hf", "plain"])
async def test_a_node_that_cannot_answer_gives_a_minimal_info_not_an_error(tmp_path: Path, fmt: str) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m", size=9)] if fmt == "hf" else [entry("ro", "d", "plain", "d", size=9)]
    w = await World(tmp_path, [a]).start()
    a.state = "down"
    ref = ModelRef(format="hf", repo_id="org/m") if fmt == "hf" else ModelRef(format="plain", root_id="ro", rel="d")
    info = await w.library.model_info("alpha", ref)
    assert info is not None and info.weight_bytes == 9 and info.num_layers is None
    await w.stop()
