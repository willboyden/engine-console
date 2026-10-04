"""GGUF-only HF repos: the node lists one entry per repo (rel = the repo directory, quants in meta.gguf_files); the library
lists one model per quant, reads that quant's header through the node, and the launch mounts the whole repo directory so
the snapshot links into blobs/ resolve."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from engine_console.adapters.base import MODEL_MOUNT
from engine_console.domain.refs import ModelRef
from engine_console.services.library import expand_gguf_repo
from engine_console.services.runspec import model_mounts

from .conftest import World
from .fakes import COMMIT, FakeNode, RootCfg

REPO_REL = "models--unsloth--Big-GGUF"
Q4 = f"snapshots/{COMMIT}/Q4_K_M/Big-Q4_K_M-00001-of-00002.gguf"
Q8 = f"snapshots/{COMMIT}/Big-Q8_0.gguf"


def repo_entry(**kw: Any) -> dict[str, Any]:
    files = [{"rel": Q4, "quant": "Q4_K_M", "size": 300, "files": 2, "shards": 2, "role": None},
             {"rel": Q8, "quant": "Q8_0", "size": 500, "files": 1, "shards": None, "role": None},
             {"rel": f"snapshots/{COMMIT}/mmproj-F16.gguf", "quant": "F16", "size": 9, "files": 1, "shards": None,
              "role": "mmproj"}]
    return {"root": "gguf", "rel": REPO_REL, "format": "gguf", "name": "unsloth/Big-GGUF", "revision": COMMIT,
            "size": 809, "files": 4, "mtime": 1.0, "atime": 2.0, "quant": None, "params": None, "arch": "qwen3",
            "context": 32768, "shards": None, "meta": {"hf_repo": "unsloth/Big-GGUF", "gguf_files": files},
            "warnings": [], **kw}


def test_expand_lists_one_entry_per_quant_and_skips_the_projector() -> None:
    out = expand_gguf_repo(repo_entry())
    assert [e["rel"] for e in out] == [f"{REPO_REL}/{Q4}", f"{REPO_REL}/{Q8}"]
    assert [(e["quant"], e["size"], e["shards"]) for e in out] == [("Q4_K_M", 300, 2), ("Q8_0", 500, None)]
    assert all(e["meta"] == {"hf_repo": "unsloth/Big-GGUF", "repo_rel": REPO_REL} for e in out)
    assert all(any("mmproj" in w for w in e["warnings"]) for e in out)


def test_expand_leaves_everything_else_alone() -> None:
    loose = {"root": "gguf", "rel": "dir/m-Q4_K_M.gguf", "format": "gguf", "meta": {"block_count": 2}}
    hf = {"root": "hf", "rel": "hub/models--a--b", "format": "hf", "meta": {"gguf_files": [{"rel": "x.gguf"}]}}
    assert expand_gguf_repo(loose) == [loose] and expand_gguf_repo(hf) == [hf]
    assert expand_gguf_repo(repo_entry(meta={"gguf_files": []}))[0]["rel"] == REPO_REL


async def test_library_lists_each_quant_and_reads_its_header_through_the_node(tmp_path: Path) -> None:
    a = FakeNode(tmp_path, "alpha", scan_disk=False, roots=[RootCfg("gguf", "gguf")])
    a.entries = [repo_entry()]
    a.configs[("gguf", f"{REPO_REL}/{Q4}")] = {"gguf": {"arch": "qwen3", "context": 32768, "params": 30_000_000_000,
                                                         "block_count": 48, "embedding_length": 4096, "head_count": 32,
                                                         "head_count_kv": 4}}
    w = await World(tmp_path, [a]).start()
    keys = sorted(m.key for m in w.library.models(format="gguf"))
    assert keys == ["gguf:unsloth/Big-GGUF:Q4_K_M", "gguf:unsloth/Big-GGUF:Q8_0"]
    ref = ModelRef(format="gguf", root_id="gguf", rel=f"{REPO_REL}/{Q4}")
    found = w.library.find("alpha", ref)
    assert found is not None and found.size_bytes == 300 and found.rel == f"{REPO_REL}/{Q4}"
    info = await w.library.model_info("alpha", ref)
    assert info is not None and (info.num_layers, info.num_kv_heads, info.num_params) == (48, 4, 30_000_000_000)
    assert info.weight_bytes == 300 and info.quantization == "q4_k_m"
    assert a.ops("models.config") == [{"root": "gguf", "rel": f"{REPO_REL}/{Q4}"}]
    await w.stop()


def test_a_quant_inside_an_hf_repo_mounts_the_repo_directory() -> None:
    mounts, path = model_mounts(ModelRef(format="gguf", root_id="gguf", rel=f"{REPO_REL}/{Q4}"), copy_rel=None,
                                ollama_root=None)
    assert mounts == [{"root": "gguf", "rel": REPO_REL, "dest": MODEL_MOUNT, "ro": True}]
    assert path == f"{MODEL_MOUNT}/{Q4}"
    home = f"hub/{REPO_REL}/{Q8}"
    mounts, path = model_mounts(ModelRef(format="gguf", root_id="hf", rel=home), copy_rel=None, ollama_root=None)
    assert mounts[0]["rel"] == f"hub/{REPO_REL}" and path == f"{MODEL_MOUNT}/{Q8}"


def test_a_loose_gguf_file_still_mounts_its_own_directory() -> None:
    mounts, path = model_mounts(ModelRef(format="gguf", root_id="gguf", rel="dir/m-Q4_K_M.gguf"), copy_rel=None,
                                ollama_root=None)
    assert mounts == [{"root": "gguf", "rel": "dir", "dest": MODEL_MOUNT, "ro": True}]
    assert path == f"{MODEL_MOUNT}/m-Q4_K_M.gguf"
