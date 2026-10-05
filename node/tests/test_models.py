"""The model scanner: every root kind, blob sizes counted once, bounded metadata, budgets, generations."""
from __future__ import annotations

import json
import os
import struct
import threading
import time

import pytest

from ecnode import models, proto
from ecnode.errors import OpError
from ecnode.models import Budget, Scanner, model_config, scan_root
from ecnode.policy import Root, parse

from .test_gguf import build

REV = "a" * 40


def safetensors(shapes: dict[str, list[int]]) -> bytes:
    hdr = json.dumps({"__metadata__": {"format": "pt"},
                      **{k: {"dtype": "BF16", "shape": v, "data_offsets": [0, 0]} for k, v in shapes.items()}}).encode()
    return struct.pack("<Q", len(hdr)) + hdr


def hf_repo(hub, repo_id, files: dict[str, bytes], *, rev=REV, extra_snapshot=False):
    repo = hub / ("models--" + repo_id.replace("/", "--"))
    (repo / "blobs").mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(rev)
    snap = repo / "snapshots" / rev
    snap.mkdir(parents=True)
    for name, content in files.items():
        blob = f"blob{abs(hash(content)) % 10**12}"
        (repo / "blobs" / blob).write_bytes(content)
        target = snap / name
        target.parent.mkdir(parents=True, exist_ok=True)
        depth = name.count("/")
        target.symlink_to("../" * (2 + depth) + "blobs/" + blob)
    if extra_snapshot:                       # an older revision sharing a blob: counted once
        old = repo / "snapshots" / ("b" * 40)
        old.mkdir()
        for name in files:
            (old / name.split("/")[-1]).symlink_to(os.readlink(snap / name))
    return repo


def root(path, kind="auto", rid="r"):
    return Root(id=rid, path=str(path), kind=kind, writable=True, label=rid)


def entries_of(r):
    found, err = scan_root(r, Budget())
    assert err is None
    return {e["rel"]: e for e in found}


def test_hf_home_repo_sizes_blobs_once_and_reads_bounded_metadata(tmp_path):
    cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "max_position_embeddings": 131072,
           "num_hidden_layers": 32, "torch_dtype": "bfloat16", "unrelated_big_field": "x" * 10000,
           "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128], "noise": 1}}
    weights = safetensors({"a": [4096, 4096], "b": [10]})
    hf_repo(tmp_path / "hf" / "hub", "org/model-a", {"config.json": json.dumps(cfg).encode(),
                                                     "model.safetensors": weights, "tokenizer.json": b"{}"},
            extra_snapshot=True)
    e = entries_of(root(tmp_path / "hf", "hf-home"))["hub/models--org--model-a"]
    assert (e["format"], e["name"], e["revision"], e["quant"], e["arch"], e["context"]) == \
        ("hf", "org/model-a", REV, "fp8", "LlamaForCausalLM", 131072)
    assert e["size"] == len(json.dumps(cfg)) + len(weights) + 2 and e["files"] == 3
    assert e["params"] == 4096 * 4096 + 10 and e["shards"] == 1
    assert "unrelated_big_field" not in e["meta"] and e["meta"]["quantization_config"] == \
        {"quant_method": "fp8", "weight_block_size": [128, 128]}
    assert any("2 revisions" in w for w in e["warnings"])


def test_hf_hub_layout_auto_detected_and_quant_from_name_and_downloads_in_progress(tmp_path):
    repo = hf_repo(tmp_path / "hub", "org/Model-NVFP4", {"config.json": b'{"model_type": "x"}',
                                                          "model.safetensors": b"w" * 10})
    (repo / "blobs" / "zzz.incomplete").write_bytes(b"partial")
    e = entries_of(root(tmp_path / "hub", "auto"))["models--org--Model-NVFP4"]
    assert e["quant"] == "nvfp4" and e["params"] is None            # packed 4-bit: no element count
    assert any("still downloading" in w for w in e["warnings"])


def test_gguf_only_hf_repo_is_one_entry_at_the_repo_dir_listing_its_quants(tmp_path):
    hub = tmp_path / "hf" / "hub"
    hf_repo(hub, "org/Model-GGUF", {"Model-Q4_K_M.gguf": build(), "Model-Q8_0.gguf": build(file_type=7)})
    hf_repo(hub, "org/Solo-GGUF", {"Solo-Q5_K_M.gguf": build(file_type=17)})
    found = entries_of(root(tmp_path / "hf", "hf-home"))
    assert set(found) == {"hub/models--org--Model-GGUF", "hub/models--org--Solo-GGUF"}   # rel = the repo dir
    e = found["hub/models--org--Model-GGUF"]
    assert (e["format"], e["name"], e["revision"], e["quant"], e["arch"]) == ("gguf", "org/Model-GGUF", REV, None,
                                                                              "llama")
    files = {f["quant"]: f for f in e["meta"]["gguf_files"]}
    assert set(files) == {"Q4_K_M", "Q8_0"} and e["meta"]["hf_repo"] == "org/Model-GGUF"
    assert files["Q8_0"]["rel"] == f"snapshots/{REV}/Model-Q8_0.gguf" and files["Q8_0"]["files"] == 1
    solo = found["hub/models--org--Solo-GGUF"]
    assert solo["quant"] == "Q5_K_M" and solo["params"] == 4096 * 32000 + 4096 * 4096
    hub_root = entries_of(root(hub, "hf-hub"))
    assert set(hub_root) == {"models--org--Model-GGUF", "models--org--Solo-GGUF"}         # hf-hub: no hub/


def test_gguf_root_groups_split_sets(tmp_path):
    g = tmp_path / "g"
    (g / "deep" / "er").mkdir(parents=True)
    (g / "Qwen3-8B-Q4_K_M.gguf").write_bytes(build())
    (g / "deep" / "er" / "Big-UD-Q2_K_XL-00001-of-00003.gguf").write_bytes(build())
    (g / "deep" / "er" / "Big-UD-Q2_K_XL-00002-of-00003.gguf").write_bytes(build(tensors=[("x", [10])]))
    (g / ".hidden").mkdir()
    (g / ".hidden" / "skip.gguf").write_bytes(build())
    found = entries_of(root(g, "gguf"))
    single = found["Qwen3-8B-Q4_K_M.gguf"]
    assert (single["name"], single["quant"], single["context"], single["arch"]) == ("Qwen3-8B", "Q4_K_M", 131072,
                                                                                   "llama")
    split = found["deep/er/Big-UD-Q2_K_XL-00001-of-00003.gguf"]
    assert split["shards"] == 3 and split["files"] == 2 and any("2 of 3" in w for w in split["warnings"])
    assert split["params"] == 4096 * 32000 + 4096 * 4096 + 10
    assert split["meta"]["files"] == ["Big-UD-Q2_K_XL-00001-of-00003.gguf", "Big-UD-Q2_K_XL-00002-of-00003.gguf"]
    assert not any(".hidden" in k for k in found)


def test_plain_root_and_secret_files_are_skipped(tmp_path):
    p = tmp_path / "p"
    m = p / "team" / "llama-ft"
    m.mkdir(parents=True)
    (m / "config.json").write_text('{"architectures": ["X"], "max_position_embeddings": 4096}')
    (m / "model.safetensors").write_bytes(safetensors({"w": [2, 3]}))
    (m / "credentials.json").write_text("never read")
    (p / "notes").mkdir()
    (p / "notes" / "config.json").write_text("{}")      # config without weights: not a model
    e = entries_of(root(p, "plain"))["team/llama-ft"]
    assert (e["format"], e["name"], e["params"], e["context"]) == ("plain", "llama-ft", 6, 4096)
    assert e["files"] == 2                              # credentials.json is not even counted


def test_plain_dir_with_weights_in_an_unknown_format_is_listed_with_a_warning(tmp_path):
    p = tmp_path / "p"
    m = p / "Custom-Model"
    m.mkdir(parents=True)
    (m / "config.json").write_text('{"architectures": ["X"]}')
    with open(m / "custom-00000.ftw", "wb") as f:      # sparse: large on paper, nothing written
        f.truncate(models.BIG_FILE + 1)
    (m / "small-00001.ftw").write_bytes(b"x" * 10)      # small unknown files alone would not make it a model
    e = entries_of(root(p, "plain"))["Custom-Model"]
    assert e["format"] == "plain" and e["size"] >= models.BIG_FILE
    assert any(".ftw" in w and "do not load" in w for w in e["warnings"])
    only_small = p / "Tiny"
    only_small.mkdir()
    (only_small / "config.json").write_text("{}")
    (only_small / "x.ftw").write_bytes(b"x")
    assert "Tiny" not in entries_of(root(p, "plain"))


def test_ollama_home_never_touches_its_keys(tmp_path, monkeypatch):
    home = tmp_path / ".ollama"
    blobs = home / "models" / "blobs"
    blobs.mkdir(parents=True)
    (home / "id_ed25519").write_text("PRIVATE KEY")
    (home / "history").write_text("old prompts")
    model = build()
    cfg = json.dumps({"model_format": "gguf", "model_family": "llama", "model_type": "8.0B",
                      "file_type": "Q4_K_M"}).encode()
    dm, dc = "1" * 64, "2" * 64
    (blobs / f"sha256-{dm}").write_bytes(model)
    (blobs / f"sha256-{dc}").write_bytes(cfg)
    man = home / "models" / "manifests" / "registry.ollama.ai" / "library" / "llama3"
    man.mkdir(parents=True)
    (man / "8b").write_text(json.dumps({"config": {"digest": f"sha256:{dc}", "size": len(cfg)},
                                        "layers": [{"mediaType": "application/vnd.ollama.image.model",
                                                    "digest": f"sha256:{dm}", "size": len(model)},
                                                   {"mediaType": "application/vnd.ollama.image.license",
                                                    "digest": "sha256:" + "3" * 64, "size": 10}]}))
    hf = home / "models" / "manifests" / "hf.co" / "org" / "repo"
    hf.mkdir(parents=True)
    (hf / "Q8_0").write_text(json.dumps({"layers": []}))
    opened: list[str] = []
    real_open = os.open
    monkeypatch.setattr(os, "open", lambda p, *a, **k: (opened.append(str(p)), real_open(p, *a, **k))[1])
    found = entries_of(root(home, "ollama"))
    by_models_dir = entries_of(root(home / "models", "auto"))        # a root at the models/ dir itself
    assert set(by_models_dir) == {"manifests/registry.ollama.ai/library/llama3/8b", "manifests/hf.co/org/repo/Q8_0"}
    e = found["models/manifests/registry.ollama.ai/library/llama3/8b"]
    assert (e["format"], e["name"], e["quant"], e["arch"], e["context"]) == ("ollama", "llama3:8b", "Q4_K_M",
                                                                             "llama", 131072)
    assert e["size"] == len(model) + 10 and any("missing" in w for w in e["warnings"])
    assert found["models/manifests/hf.co/org/repo/Q8_0"]["name"] == "hf.co/org/repo:Q8_0"
    assert not any(p.endswith(("id_ed25519", "history")) for p in opened), opened


def test_other_root_and_auto_fallback(tmp_path):
    o = tmp_path / "o"
    (o / "a" / "b").mkdir(parents=True)
    (o / "a" / "b" / "f").write_bytes(b"x" * 100)
    (o / "a" / "token").write_text("secret")
    e = entries_of(root(o, "other"))["a"]
    assert (e["format"], e["size"], e["files"]) == ("other", 100, 1)
    assert entries_of(root(o, "auto")) == {}           # no models found by plain + gguf


def test_budget_exhaustion_keeps_what_was_found(tmp_path):
    g = tmp_path / "g"
    (g / "a").mkdir(parents=True)
    (g / "b").mkdir()
    for i in range(3):
        (g / "a" / f"m{i}-Q4_0.gguf").write_bytes(build())
    for i in range(30):
        (g / "b" / f"m{i:02d}-Q4_0.gguf").write_bytes(build())
    found, err = scan_root(root(g, "gguf"), Budget(entries=10))
    assert err and "10 directory entries" in err
    assert sorted(e["rel"] for e in found) == [f"a/m{i}-Q4_0.gguf" for i in range(3)]
    found2, err2 = scan_root(root(g, "gguf"), Budget(entries=1000, seconds=0.0))
    assert err2 and "s (scan budget)" in err2


def test_symlinked_files_leaving_the_root_are_ignored(tmp_path):
    out = tmp_path / "outside.gguf"
    out.write_bytes(build())
    g = tmp_path / "g"
    g.mkdir()
    (g / "escape.gguf").symlink_to(out)
    assert entries_of(root(g, "gguf")) == {}


# ── models.config ────────────────────────────────────────────────────────────────────────────────────────
def test_model_config_for_hf_plain_gguf_and_refusals(tmp_path):
    hf_repo(tmp_path / "hf" / "hub", "org/m", {"config.json": b'{"model_type": "llama", "n": 1}'})
    (tmp_path / "pl" / "m").mkdir(parents=True)
    (tmp_path / "pl" / "m" / "config.json").write_text('{"a": 1}')
    (tmp_path / "pl" / "f.gguf").write_bytes(build())
    (tmp_path / "secret.json").write_text('{"leak": true}')
    (tmp_path / "pl" / "esc").mkdir()
    (tmp_path / "pl" / "esc" / "config.json").symlink_to(tmp_path / "secret.json")
    p = parse({"model_roots": [{"id": "hf", "path": str(tmp_path / "hf"), "kind": "hf-home"},
                               {"id": "pl", "path": str(tmp_path / "pl"), "kind": "plain"}]}, None)
    assert model_config(p, "hf", "hub/models--org--m")["config"] == {"model_type": "llama", "n": 1}
    assert model_config(p, "pl", "m")["config"] == {"a": 1}
    assert model_config(p, "pl", "f.gguf")["config"]["gguf"]["arch"] == "llama"
    for rel in ("esc", "nope"):
        with pytest.raises(OpError) as e:
            model_config(p, "pl", rel)
        assert e.value.code == proto.NOT_FOUND
    with pytest.raises(OpError):
        model_config(p, "pl", "../hf")


# ── the background scanner ───────────────────────────────────────────────────────────────────────────────
def test_scanner_generations_paging_and_publishing(tmp_path):
    g = tmp_path / "g"
    g.mkdir()
    for i in range(5):
        (g / f"m{i}-Q4_0.gguf").write_bytes(build())
    pol = parse({"model_roots": [{"id": "g", "path": str(g), "kind": "gguf"},
                                 {"id": "missing", "path": str(tmp_path / "nope"), "kind": "gguf"}]}, None)
    changed: list[int] = []
    sc = Scanner(lambda: pol, changed.append)
    assert sc.page(None, None)["generation"] == 0
    gen = sc.request(None, force=True)
    assert sc.run_once() == gen and changed == [gen]
    p1 = sc.page(0, 2)
    assert p1["generation"] == gen and len(p1["entries"]) == 2 and p1["next_cursor"] == 2
    p3 = sc.page(4, 2)
    assert len(p3["entries"]) == 1 and p3["next_cursor"] is None
    scans = {r["root"]: r for r in p1["roots"]}
    assert scans["g"]["complete"] and scans["g"]["entries"] == 5
    assert "does not exist" in scans["missing"]["error"]
    # an automatic rescan that finds nothing new publishes nothing ...
    with sc._cv:
        sc._pending, sc._pending_all = True, True
    assert sc.run_once() is None and changed == [gen]
    # ... a requested one always publishes the generation it promised
    promised = sc.request(None, force=True)
    assert promised > gen and sc.run_once() == promised and changed[-1] == promised
    # a change is published
    (g / "m9-Q8_0.gguf").write_bytes(build(file_type=7))
    with sc._cv:
        sc._pending, sc._pending_all = True, True
    newer = sc.run_once()
    assert newer and sc.page(0, 200)["generation"] == newer and len(sc.page(0, 200)["entries"]) == 6


def test_scanner_answers_a_fresh_request_with_the_current_generation(tmp_path):
    pol = parse({"model_roots": []}, None)
    sc = Scanner(lambda: pol, lambda g: None)
    sc.request(None, force=True)
    gen = sc.run_once()
    assert sc.request(None, force=False) == gen          # fresh: no new scan
    assert sc.request(None, force=True) == gen + 1


def test_hung_root_is_reported_and_the_previous_result_kept(tmp_path, monkeypatch):
    g = tmp_path / "g"
    g.mkdir()
    (g / "a-Q4_0.gguf").write_bytes(build())
    pol = parse({"model_roots": [{"id": "g", "path": str(g), "kind": "gguf"}]}, None)
    sc = Scanner(lambda: pol, lambda gen: None, seconds=0.1)
    monkeypatch.setattr(models, "HUNG_GRACE_S", 0.2)
    sc.request(None, force=True)
    sc.run_once()
    gate = threading.Event()
    real = models.scan_root
    monkeypatch.setattr(models, "scan_root", lambda r, b: (gate.wait(5), real(r, b))[1])
    sc.request(None, force=True)
    t0 = time.monotonic()
    sc.run_once()
    gate.set()
    assert time.monotonic() - t0 < 3
    page = sc.page(0, 10)
    assert len(page["entries"]) == 1 and "did not answer" in page["roots"][0]["error"]


def test_m3_a_hostile_safetensors_header_is_refused_without_burning_cpu(tmp_path):
    """security review 2026-10-04 M3: `count *= dim` over ~5 million dims costs minutes of big-int CPU per file."""
    import time

    from ecnode.models import _safetensors_params

    def one(shapes: dict[str, list[int]]) -> int | None:
        f = tmp_path / "w.safetensors"
        f.write_bytes(safetensors(shapes))
        return _safetensors_params(str(tmp_path), [str(f)], None)

    assert one({"a": [4096, 4096]}) == 4096 * 4096
    assert one({"a": [1] * 100_000}) is None                    # far too many dims: refused, whatever they multiply to
    assert one({"a": [2**40, 2**40]}) is None                   # past 2**53 parameters
    t0 = time.monotonic()
    assert one({"a": [3] * 2_000_000}) is None                  # unbounded, this one is minutes of CPU
    assert time.monotonic() - t0 < 5.0
