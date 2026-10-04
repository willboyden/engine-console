"""Copies between machines and drives: a library copy's files go source node -> hub -> target node, with the HF layout
preserved where both ends are HF caches."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub import scan_cache_dir

from engine_console.domain.errors import Conflict, NotFound, ProblemError, UpstreamError
from engine_console.domain.fleet import CopyRequest
from engine_console.domain.models import Download, DownloadRequest
from engine_console.domain.refs import ModelRef
from engine_console.services.downloads import GgufSource, HfSource, PlainSource, classify_copy_source

from .conftest import World
from .fakes import COMMIT, FakeNode, RootCfg, entry, gguf_entry, hf_entry, seed_hf_cache
from .helpers import PAYLOAD, SHA, TOTAL, nfs_alpha, nfs_beta, partial_of, stop_midway, until

FILES = {"config.json": b"{}", "w.safetensors": PAYLOAD, "notes.txt": b"n"}
HF = ModelRef(format="hf", repo_id="org/m")


def machines(tmp: Path) -> tuple[FakeNode, FakeNode]:
    src = FakeNode(tmp, "alpha", label="Alpha", addresses=["10.0.0.1"], roots=[
        RootCfg("hf", "hf-home"), RootCfg("hf2", "hf-hub"), RootCfg("gguf", "gguf"), RootCfg("pl", "plain"), RootCfg("ro", "hf-home", writable=False)])
    dst = FakeNode(tmp, "beta", label="Beta", addresses=["10.0.0.2"], roots=[
        RootCfg("hub", "hf-hub"), RootCfg("home", "hf-home"), RootCfg("gguf", "gguf"), RootCfg("pl", "plain"), RootCfg("ro", "hf-hub", writable=False)])
    return src, dst


async def world(tmp: Path, *, seed: bool = True, files: dict[str, bytes] | None = None) -> tuple[World, FakeNode, FakeNode]:
    src, dst = machines(tmp)
    if seed:
        seed_hf_cache(src.roots["hf"].path, "org/m", files or FILES)           # type: ignore[arg-type]
        src.scan()
    w = World(tmp, [src, dst])
    w.hub.add("org/m", files or FILES)
    await w.start()
    return w, src, dst


def source_copy(w: World, host: str = "alpha", ref: ModelRef = HF) -> str:
    c = w.library.find(host, ref)
    assert c is not None
    return c.id


async def copy(w: World, cid: str, host: str = "beta", root: str | None = None) -> Download:
    d = await w.downloads.copy(cid, CopyRequest(host_id=host, root_id=root))
    await w.downloads.wait(d.id)
    return w.downloads.get(d.id)


# ---- Hugging Face cache directories -------------------------------------------------------------------------------------------------------
async def test_copy_between_machines_keeps_the_hf_layout_and_moves_the_big_files_machine_to_machine(tmp_path: Path) -> None:
    w, src, dst = await world(tmp_path)
    cid = source_copy(w)
    d = await copy(w, cid, "beta", "hub")
    assert d.state == "completed" and d.kind == "copy" and d.source_copy_id == cid
    assert (d.host_id, d.root_id, d.repo_id, d.commit_sha) == ("beta", "hub", "org/m", COMMIT)
    assert (d.total_bytes, d.done_bytes, d.files_total, d.files_done) == (TOTAL, TOTAL, 3, 3)
    repo = dst.roots["hub"].path / "models--org--m"                                   # type: ignore[operator]
    assert (repo / "blobs" / SHA).read_bytes() == PAYLOAD
    link = repo / "snapshots" / COMMIT / "w.safetensors"
    assert link.is_symlink() and os.readlink(link) == f"../../blobs/{SHA}"
    assert (repo / "snapshots" / COMMIT / "config.json").read_bytes() == b"{}"
    assert (repo / "snapshots" / COMMIT / "notes.txt").read_bytes() == b"n"
    assert (repo / "refs" / "main").read_text() == COMMIT
    assert [r.repo_id for r in scan_cache_dir(dst.roots["hub"].path).repos] == ["org/m"]      # type: ignore[arg-type]
    # the large file came from the source machine, was hashed by the target's node, and never touched Hugging Face
    assert src.reads == [{"rel": f"hub/models--org--m/blobs/{SHA}", "offset": 0}]
    assert not any(o[2] == "w.safetensors" for o in w.hub.opens) and {o[2] for o in w.hub.opens} == {"config.json", "notes.txt"}
    assert next(x for x in dst.writes if x["rel"].endswith(SHA))["sha256"] == SHA
    # both machines now list the model
    await w.settle()
    assert [c.host_id for c in w.library.models()[0].copies] == ["alpha", "beta"]
    await w.stop()


async def test_copy_from_an_hf_home_into_an_hf_home_and_the_default_root(tmp_path: Path) -> None:
    w, _, dst = await world(tmp_path)
    d = await w.downloads.copy(source_copy(w), CopyRequest(host_id="beta"))
    assert d.root_id == "hub"                 # beta's first writable HF root, in policy order (the read-only one is skipped)
    await w.downloads.wait(d.id)
    d2 = await copy(w, source_copy(w), "beta", "home")
    assert d2.state == "completed" and (dst.roots["home"].path / "hub" / "models--org--m" / "refs" / "main").is_file()   # type: ignore[operator]
    await w.stop()


async def test_copy_to_another_drive_of_the_same_machine(tmp_path: Path) -> None:
    w, src, _ = await world(tmp_path)
    d = await copy(w, source_copy(w), "alpha", "hf2")
    assert d.state == "completed" and d.host_id == "alpha" and d.root_id == "hf2"
    assert (src.roots["hf2"].path / "models--org--m" / "blobs" / SHA).read_bytes() == PAYLOAD      # type: ignore[operator]
    await w.settle()
    assert sorted(c.root_id for c in w.library.copies("alpha")) == ["hf", "hf2"]
    await w.stop()


async def test_copy_resumes_from_the_partial_on_the_target(tmp_path: Path) -> None:
    w, src, dst = await world(tmp_path)
    rel = f"models--org--m/blobs/{SHA}"
    part = partial_of(dst, "hub", rel)
    part.parent.mkdir(parents=True)
    part.write_bytes(PAYLOAD[:20])
    d = await copy(w, source_copy(w), "beta", "hub")
    assert d.state == "completed"
    assert src.reads == [{"rel": f"hub/models--org--m/blobs/{SHA}", "offset": 20}]         # read from where the target stopped
    assert next(x for x in dst.writes if x["rel"] == rel)["offset"] == 20
    assert (dst.roots["hub"].path / rel).read_bytes() == PAYLOAD                          # type: ignore[operator]
    await w.stop()


async def test_a_copy_whose_bytes_change_in_transit_fails_the_nodes_check_and_can_be_retried(tmp_path: Path) -> None:
    w, src, dst = await world(tmp_path)
    blob = src.roots["hf"].path / "hub" / "models--org--m" / "blobs" / SHA              # type: ignore[operator]
    blob.write_bytes(b"X" * 50)                                                          # bit rot on the source disk
    d = await copy(w, source_copy(w), "beta", "hub")
    assert d.state == "failed" and d.error_code == "hash_mismatch" and (d.error or "").startswith("Beta: ")
    assert not partial_of(dst, "hub", f"models--org--m/blobs/{SHA}").exists()            # the node threw the bad partial away
    blob.write_bytes(PAYLOAD)
    await w.downloads.resume(d.id)
    await w.downloads.wait(d.id)
    assert w.downloads.get(d.id).state == "completed"
    await w.stop()


class DeadStream:
    """An fs.read that fails at its first await, as a node whose disk is failing would."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def __aiter__(self) -> Any:
        return self._it()

    async def _it(self) -> Any:
        raise self.error
        yield  # pragma: no cover

    async def result(self) -> dict[str, Any]:
        raise self.error

    async def cancel(self) -> None:
        return None


async def test_a_read_error_on_the_source_machine_fails_the_copy_naming_that_machine(tmp_path: Path) -> None:
    w, src, _ = await world(tmp_path)
    cid = source_copy(w)
    real = src.open
    src.open = lambda op, args=None: DeadStream(src._err("io_error", "the disk is failing")) if op == "fs.read" else real(op, args)  # type: ignore[method-assign,assignment]
    d = await copy(w, cid, "beta", "hub")
    assert d.state == "failed" and d.error_code == "io_error" and (d.error or "").startswith("Alpha: ")
    await w.stop()


async def test_a_source_blob_that_is_missing_means_the_copy_is_not_of_that_revision(tmp_path: Path) -> None:
    w, src, dst = await world(tmp_path)
    (src.roots["hf"].path / "hub" / "models--org--m" / "blobs" / SHA).unlink()          # type: ignore[operator]
    d = await copy(w, source_copy(w), "beta", "hub")
    assert d.state == "failed" and d.error_code == "copy_source_mismatch" and not dst.writes
    await w.stop()


async def test_a_copy_needs_the_hubs_listing_and_says_so_when_hugging_face_is_out_of_reach(tmp_path: Path) -> None:
    w, _, dst = await world(tmp_path)
    cid = source_copy(w)

    async def down(repo_id: str, revision: str | None) -> Any:
        raise UpstreamError("cannot reach Hugging Face: ConnectError")

    w.hub.model = down                                                  # type: ignore[method-assign]
    d = await copy(w, cid, "beta", "hub")
    assert d.state == "failed" and d.error_code == "upstream_error" and not dst.writes
    await w.stop()


async def test_a_copy_of_a_model_with_only_some_files_stays_partial(tmp_path: Path) -> None:
    w, src, dst = await world(tmp_path)
    extra = b"E" * 30
    w.hub.add("org/m", {**FILES, "extra.safetensors": extra})            # the repo has one more large file than the source holds
    d = await copy(w, source_copy(w), "beta", "hub")
    assert d.state == "completed" and d.files_total == 3
    snaps = dst.roots["hub"].path / "models--org--m" / "snapshots" / COMMIT              # type: ignore[operator]
    assert sorted(p.name for p in snaps.iterdir()) == ["config.json", "notes.txt", "w.safetensors"]     # not extra.safetensors
    await w.stop()


async def test_copy_into_a_gguf_root_flattens_the_snapshot(tmp_path: Path) -> None:
    w, _, dst = await world(tmp_path)
    d = await copy(w, source_copy(w), "beta", "gguf")
    assert d.state == "completed"
    target = dst.roots["gguf"].path / "org--m"                                           # type: ignore[operator]
    assert sorted(p.name for p in target.iterdir()) == ["config.json", "notes.txt", "w.safetensors"]
    assert (target / "w.safetensors").read_bytes() == PAYLOAD and not (target / "w.safetensors").is_symlink()
    await w.stop()


# ---- one quant of a GGUF repo, and loose .gguf files -------------------------------------------------------------------------------------------
async def test_a_gguf_quant_in_an_hf_cache_copies_only_its_own_files(tmp_path: Path) -> None:
    q4, q8 = b"q4" * 20, b"q8" * 30
    files = {"m-Q4_K_M.gguf": q4, "m-Q8_0.gguf": q8, "README.md": b"hi"}
    src, dst = machines(tmp_path)
    seed_hf_cache(src.roots["hf"].path, "org/g", files)                               # type: ignore[arg-type]
    src.scan_disk_enabled = False
    src.entries = [{**gguf_entry("hf", f"hub/models--org--g/snapshots/{COMMIT}/m-Q4_K_M.gguf", name="org/g", quant="Q4_K_M", size=len(q4))}]
    w = World(tmp_path, [src, dst])
    w.hub.add("org/g", files)
    await w.start()
    cid = w.library.models()[0].copies[0].id
    d = await copy(w, cid, "beta", "hub")
    assert d.state == "completed" and d.files_total == 1 and d.total_bytes == len(q4)
    snaps = dst.roots["hub"].path / "models--org--g" / "snapshots" / COMMIT              # type: ignore[operator]
    assert [p.name for p in snaps.iterdir()] == ["m-Q4_K_M.gguf"] and (snaps / "m-Q4_K_M.gguf").read_bytes() == q4
    await w.stop()


async def test_a_loose_gguf_file_and_every_shard_of_a_split_set_are_copied_as_they_are(tmp_path: Path) -> None:
    src, dst = machines(tmp_path)
    shards = {f"models/big-Q4_K_M-0000{i}-of-00003.gguf": bytes([i]) * (40 + i) for i in (1, 2, 3)}
    single = b"s" * 25
    for rel, data in {**shards, "models/one-Q8_0.gguf": single}.items():
        p = src.roots["gguf"].path / rel                                                # type: ignore[operator]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    src.scan_disk_enabled = False
    first = "models/big-Q4_K_M-00001-of-00003.gguf"
    src.entries = [gguf_entry("gguf", first, name="big", quant="Q4_K_M", size=sum(map(len, shards.values())), shards=3),
                   gguf_entry("gguf", "models/one-Q8_0.gguf", name="one", quant="Q8_0", size=len(single))]
    w = World(tmp_path, [src, dst])
    await w.start()
    big = w.library.find("alpha", ModelRef(format="gguf", root_id="gguf", rel=first))
    one = w.library.find("alpha", ModelRef(format="gguf", root_id="gguf", rel="models/one-Q8_0.gguf"))
    assert big is not None and one is not None
    d = await copy(w, big.id, "beta", "gguf")
    assert d.state == "completed" and d.files_total == 3 and d.total_bytes == sum(map(len, shards.values()))
    for rel, data in shards.items():
        assert (dst.roots["gguf"].path / rel).read_bytes() == data                       # type: ignore[operator]
    assert [x["rel"] for x in src.reads] == list(shards)
    assert all(x["sha256"] is None for x in dst.writes)                                   # no hash to check, size still is
    d2 = await copy(w, one.id, "beta")                                                    # default: beta's gguf root
    assert d2.state == "completed" and d2.root_id == "gguf" and (dst.roots["gguf"].path / "models/one-Q8_0.gguf").read_bytes() == single  # type: ignore[operator]
    assert w.hub.listings == 0                                                            # no Hugging Face involved at all
    await w.stop()


async def test_a_split_set_with_a_missing_shard_is_not_copied_half(tmp_path: Path) -> None:
    src, dst = machines(tmp_path)
    p = src.roots["gguf"].path / "x-Q4_K_M-00001-of-00002.gguf"                          # type: ignore[operator]
    p.write_bytes(b"1" * 10)                                                              # shard 2 is not there
    src.scan_disk_enabled = False
    src.entries = [gguf_entry("gguf", "x-Q4_K_M-00001-of-00002.gguf", name="x", size=10, shards=2)]
    w = World(tmp_path, [src, dst])
    await w.start()
    d = await copy(w, w.library.models()[0].copies[0].id, "beta", "gguf")
    assert d.state == "failed" and d.error_code == "copy_source_mismatch" and "shard" in (d.error or "")
    assert not dst.writes
    await w.stop()


# ---- refusals --------------------------------------------------------------------------------------------------------------------------------------
async def test_what_cannot_be_copied_says_why(tmp_path: Path) -> None:
    src, dst = machines(tmp_path)
    src.scan_disk_enabled = False
    src.entries = [entry("pl", "x/manifests", "ollama", "q:1"), entry("pl", "other", "other", "o"),
                   entry("pl", "uncounted", "plain", "uncounted", files=None),                       # no exact file count
                   entry("pl", "bounded", "plain", "bounded", files=5, warnings=["size is a lower bound: the scan budget ran out"]),
                   gguf_entry("hf", "hub/models--org--bad--name/snapshots/x/y.gguf"), hf_entry("hf", "org/ok")]
    w = World(tmp_path, [src, dst])
    await w.start()
    refused = {c.rel for c in w.library.copies("alpha")} - {"hub/models--org--ok"}
    assert refused == {"x/manifests", "other", "uncounted", "bounded", "hub/models--org--bad--name/snapshots/x/y.gguf"}
    for c in w.library.copies("alpha"):
        if c.rel in refused:
            with pytest.raises(ProblemError) as ei:
                await w.downloads.copy(c.id, CopyRequest(host_id="beta"))
            assert ei.value.status == 501 and ei.value.code == "copy_unsupported" and "directory listing" in ei.value.detail
    assert w.store.one("SELECT COUNT(*) AS n FROM downloads")["n"] == 0                 # type: ignore[index]
    await w.stop()


# ---- plain model directories: enumerated by name, then proved complete -----------------------------------------------------------------
SHARDS = {"model-00001-of-00002.safetensors": b"a" * 70, "model-00002-of-00002.safetensors": b"b" * 55}
PLAIN = {"config.json": b'{"model_type": "x"}', "tokenizer.json": b"{}", "README.md": b"# hi", **SHARDS,
         "model.safetensors.index.json": json.dumps({"metadata": {}, "weight_map": {"a": "model-00001-of-00002.safetensors",
                                                                                      "b": "model-00002-of-00002.safetensors",
                                                                                      "c": "model-00001-of-00002.safetensors"}}).encode()}


def plain_world(tmp: Path, files: dict[str, bytes] | None = None, rel: str = "mymodel", **entry_kw: Any) -> tuple[World, FakeNode, FakeNode]:
    src, dst = machines(tmp)
    files = files if files is not None else PLAIN
    for name, data in files.items():
        p = src.roots["pl"].path / rel / name                                              # type: ignore[operator]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    src.scan_disk_enabled = False
    src.entries = [entry("pl", rel, "plain", rel.rsplit("/", 1)[-1], size=sum(map(len, files.values())), files=len(files), **entry_kw)]
    return World(tmp, [src, dst]), src, dst


async def test_a_plain_directory_is_copied_when_every_file_is_accounted_for(tmp_path: Path) -> None:
    w, src, dst = plain_world(tmp_path)
    await w.start()
    d = await copy(w, w.library.models()[0].copies[0].id, "beta", "pl")
    assert d.state == "completed", (d.error_code, d.error)
    assert d.files_total == 6 and d.total_bytes == sum(map(len, PLAIN.values())) and d.root_id == "pl"
    for name, data in PLAIN.items():
        assert (dst.roots["pl"].path / "mymodel" / name).read_bytes() == data              # type: ignore[operator]
    read = {x["rel"] for x in src.reads}
    assert read == {f"mymodel/{n}" for n in PLAIN} | {"mymodel/model.safetensors.index.json"}   # the index is read twice: names, then bytes
    assert w.hub.listings == 0 and not any(op == "models.config" for op, _ in dst.calls)    # no Hugging Face, no config probing
    await w.stop()


async def test_a_plain_directory_with_a_file_the_probe_cannot_name_is_refused_with_the_numbers(tmp_path: Path) -> None:
    w, _, dst = plain_world(tmp_path, {**PLAIN, "weird-name.dat": b"?" * 9})
    await w.start()
    d = await copy(w, w.library.models()[0].copies[0].id, "beta", "pl")
    assert d.state == "failed" and d.error_code == "copy_listing_incomplete"
    assert "found 6 of the directory's 7 files" in (d.error or "") and "305 of 314 bytes" in (d.error or "")
    assert not dst.writes                                                                  # nothing was half copied
    await w.stop()


async def test_a_plain_directory_with_a_subfolder_is_refused(tmp_path: Path) -> None:
    w, _, dst = plain_world(tmp_path, {**PLAIN, "1_Pooling/config.json": b"{}"})
    await w.start()
    d = await copy(w, w.library.models()[0].copies[0].id, "beta", "pl")
    assert d.state == "failed" and d.error_code == "copy_listing_incomplete" and not dst.writes
    await w.stop()


async def test_a_directory_that_changed_since_the_scan_is_refused(tmp_path: Path) -> None:
    w, src, dst = plain_world(tmp_path)
    await w.start()
    (src.roots["pl"].path / "mymodel" / "README.md").write_bytes(b"# a much longer readme than the scan saw")   # type: ignore[operator]
    d = await copy(w, w.library.models()[0].copies[0].id, "beta", "pl")
    assert d.state == "failed" and d.error_code == "copy_listing_incomplete" and not dst.writes
    await w.stop()


async def test_hostile_names_in_a_weight_index_are_never_sent_to_a_node(tmp_path: Path) -> None:
    index = json.dumps({"weight_map": {"a": "../evil.bin", "b": "/abs/path", "c": 5, "d": "a\\b", "e": "ok.safetensors", "f": "x/./y"}}).encode()
    files = {"config.json": b"{}", "ok.safetensors": b"o" * 33, "model.safetensors.index.json": index}
    w, src, _ = plain_world(tmp_path, files)
    await w.start()
    d = await copy(w, w.library.models()[0].copies[0].id, "beta", "pl")
    assert d.state == "completed"
    statted = [p for op, a in src.calls if op == "fs.stat" for p in a["paths"]]
    assert "mymodel/ok.safetensors" in statted
    assert not any(".." in p or p.startswith("/") or "\\" in p or "/./" in p for p in statted)
    await w.stop()


async def test_an_oversized_or_broken_weight_index_is_handled(tmp_path: Path) -> None:
    big = b'{"weight_map": {"a": "x.safetensors"}, "pad": "' + b"p" * (5 * 1024 * 1024) + b'"}'
    w, _, dst = plain_world(tmp_path, {"config.json": b"{}", "model.safetensors.index.json": big})
    await w.start()
    d = await copy(w, w.library.models()[0].copies[0].id, "beta", "pl")
    assert d.state == "failed" and d.error_code == "copy_listing_incomplete" and "4 MiB" in (d.error or "") and not dst.writes
    await w.stop()
    w2, _, _ = plain_world(tmp_path / "x", {"config.json": b"{}", "model.safetensors.index.json": b"not json at all"})
    await w2.start()
    d2 = await copy(w2, w2.library.models()[0].copies[0].id, "beta", "pl")
    assert d2.state == "completed"                                       # nothing to learn from it; the totals still add up
    await w2.stop()


async def test_a_directory_that_is_itself_the_model_root_and_names_with_odd_characters(tmp_path: Path) -> None:
    src, dst = machines(tmp_path)
    src.roots["whole"] = RootCfg("whole", "plain", path=src.base / "whole")
    src.roots["whole"].path.mkdir()                                                       # type: ignore[union-attr]
    for name, data in {"config.json": b"{}", "pytorch_model.bin": b"w" * 7}.items():
        (src.roots["whole"].path / name).write_bytes(data)                                 # type: ignore[union-attr,operator]
    odd = src.roots["pl"].path / "a b" / "c d"                                            # type: ignore[operator]
    odd.mkdir(parents=True)
    (odd / "config.json").write_bytes(b"")
    src.scan_disk_enabled = False
    src.entries = [entry("whole", "", "plain", "root model!", size=9, files=2), entry("pl", "a b/c d", "plain", "c d", size=0, files=1)]
    w = World(tmp_path, [src, dst])
    await w.start()
    whole = next(c for c in w.library.copies("alpha") if c.rel == "")
    spaced = next(c for c in w.library.copies("alpha") if c.rel == "a b/c d")
    assert whole.path == str(src.roots["whole"].path)                                     # the root itself is the model
    d = await copy(w, whole.id, "beta", "pl")
    assert d.state == "completed", (d.error_code, d.error)
    assert (dst.roots["pl"].path / "root_model_" / "config.json").read_bytes() == b"{}"  # type: ignore[operator]
    d2 = await copy(w, spaced.id, "beta", "pl")
    assert d2.state == "completed" and (dst.roots["pl"].path / "c_d" / "config.json").is_file()          # type: ignore[operator]
    await w.stop()


async def test_plain_directories_go_only_into_plain_or_auto_roots_and_only_those_take_them(tmp_path: Path) -> None:
    w, src, dst = plain_world(tmp_path)
    dst.roots["auto"] = RootCfg("auto", "auto", path=dst.base / "auto")
    dst.roots["auto"].path.mkdir()                                                          # type: ignore[union-attr]
    await w.start()
    cid = w.library.models()[0].copies[0].id
    for root in ("hub", "home", "gguf", "ro"):
        with pytest.raises(ProblemError) as ei:
            await w.downloads.copy(cid, CopyRequest(host_id="beta", root_id=root))
        assert ei.value.code in ("root_kind_unsupported", "root_not_writable"), root
    d = await w.downloads.copy(cid, CopyRequest(host_id="beta"))                           # default: beta's first plain root
    assert d.root_id == "pl"
    await w.downloads.wait(d.id)
    d2 = await w.downloads.copy(cid, CopyRequest(host_id="beta", root_id="auto"))          # an empty auto root takes it as files
    await w.downloads.wait(d2.id)
    assert w.downloads.get(d2.id).state == "completed" and (dst.roots["auto"].path / "mymodel" / "config.json").is_file()  # type: ignore[union-attr,operator]
    # and a repository download or a loose gguf never lands in a plain root
    with pytest.raises(ProblemError) as ei2:
        await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="beta", root_id="pl"))
    assert ei2.value.code == "root_kind_unsupported"
    await w.stop()


async def test_a_loose_gguf_can_go_into_an_empty_auto_root(tmp_path: Path) -> None:
    src, dst = machines(tmp_path)
    (src.roots["gguf"].path / "m-Q4_K_M.gguf").write_bytes(b"1" * 10)                  # type: ignore[operator]
    src.scan()
    dst.roots["auto"] = RootCfg("auto", "auto", path=dst.base / "auto")
    dst.roots["auto"].path.mkdir()                                                      # type: ignore[union-attr]
    w = World(tmp_path, [src, dst])
    await w.start()
    d = await copy(w, w.library.models()[0].copies[0].id, "beta", "auto")
    assert d.state == "completed" and (dst.roots["auto"].path / "m-Q4_K_M.gguf").read_bytes() == b"1" * 10  # type: ignore[union-attr,operator]
    await w.stop()


async def test_copy_refusals_at_create(tmp_path: Path) -> None:
    w, src, dst = await world(tmp_path)
    cid = source_copy(w)

    async def refused(host: str = "beta", root: str | None = None, copy_id: str | None = None) -> ProblemError:
        with pytest.raises((ProblemError, NotFound)) as ei:
            await w.downloads.copy(copy_id or cid, CopyRequest(host_id=host, root_id=root))
        return ei.value                                                                  # type: ignore[return-value]

    e = await refused(copy_id="f" * 16)
    assert (e.status, e.code) == (404, "copy_not_found")
    e = await refused(host="alpha", root="hf")
    assert (e.status, e.code) == (409, "copy_same_location")                              # onto itself
    e = await refused(host="beta", root="ro")
    assert (e.status, e.code) == (409, "root_not_writable")
    e = await refused(host="beta", root="pl")
    assert (e.status, e.code) == (422, "root_kind_unsupported")
    e = await refused(host="nobody")
    assert (e.status, e.code) == (404, "unknown_host")
    dst.mode = "monitor"
    e = await refused()
    assert (e.status, e.code) == (409, "monitor_only")
    dst.mode = "manage"
    dst.state = "down"
    e = await refused()
    assert (e.status, e.code) == (503, "node_unavailable") and e.detail.startswith("Beta: ")
    dst.state = "up"
    src.state = "down"
    e = await refused()
    assert (e.status, e.code) == (503, "node_unavailable") and e.detail.startswith("Alpha: ")      # the source must be reachable too
    src.state = "up"
    assert w.store.one("SELECT COUNT(*) AS n FROM downloads")["n"] == 0                   # type: ignore[index]
    await w.stop()


async def test_a_loose_gguf_file_only_goes_into_a_gguf_root(tmp_path: Path) -> None:
    src, dst = machines(tmp_path)
    (src.roots["gguf"].path / "m-Q4_K_M.gguf").write_bytes(b"1" * 10)                  # type: ignore[operator]
    src.scan()
    w = World(tmp_path, [src, dst])
    await w.start()
    cid = w.library.models()[0].copies[0].id
    with pytest.raises(ProblemError) as ei:
        await w.downloads.copy(cid, CopyRequest(host_id="beta", root_id="hub"))
    assert (ei.value.status, ei.value.code) == (422, "root_kind_unsupported")
    await w.stop()


async def test_a_machine_that_already_sees_the_copy_over_nfs_can_still_copy_it_to_its_own_drive(tmp_path: Path) -> None:
    a = nfs_alpha(tmp_path)
    b = nfs_beta(tmp_path)
    a.entries = [hf_entry("hf", "org/m")]
    b.entries = [hf_entry("shared", "org/m")]
    w = World(tmp_path, [a, b])
    w.hub.add("org/m", FILES)
    await w.start()
    cid = w.library.models()[0].copies[0].id
    assert w.library.models()[0].copies[0].visible_on == ["alpha", "beta"]
    with pytest.raises(ProblemError) as ei:
        await w.downloads.copy(cid, CopyRequest(host_id="beta", root_id="shared"))        # it is that storage already
    assert ei.value.code == "copy_same_location"
    src = w.library.source(cid)
    assert src is not None and sorted(src.locations) == [("alpha", "hf"), ("beta", "shared")]
    # beta's own drive is another place: the request is accepted (these roots are literal paths, not directories, so the
    # row is cancelled before its task takes a first step)
    d = await w.downloads.copy(cid, CopyRequest(host_id="beta", root_id="own"))
    assert d.root_id == "own" and d.host_id == "beta" and d.kind == "copy" and d.state == "queued"
    await w.downloads.cancel(d.id)
    assert not any(op.startswith("fs.") for n in (a, b) for op, _ in n.calls)
    await w.stop()


async def test_a_copy_row_cannot_be_started_twice(tmp_path: Path) -> None:
    w, _, _ = await world(tmp_path)
    stop_midway(w, 1)
    cid = source_copy(w)
    d = await w.downloads.copy(cid, CopyRequest(host_id="beta", root_id="hub"))
    with pytest.raises(Conflict) as ei:
        await w.downloads.copy(cid, CopyRequest(host_id="beta", root_id="hub"))
    assert ei.value.code == "download_exists"
    await w.downloads.copy(cid, CopyRequest(host_id="beta", root_id="home"))              # another root is another copy
    await w.downloads.cancel(d.id)
    await w.stop()


async def test_a_copy_whose_source_vanished_fails_clearly_when_resumed(tmp_path: Path) -> None:
    w, _, _ = await world(tmp_path)
    w.store.execute("INSERT INTO downloads(id,repo_id,state,created_at,updated_at,host_id,root_id,kind,source_copy_id)"
                    " VALUES('dl_x','org/m','paused',1,1,'beta','hub','copy',?)", ("0" * 16,))
    await w.downloads.resume("dl_x")
    await w.downloads.wait("dl_x")
    d = w.downloads.get("dl_x")
    assert d.state == "failed" and d.error_code == "copy_source_gone"
    await w.stop()


# ---- pause, cancel and progress ------------------------------------------------------------------------------------------------------------------------
async def test_a_copy_can_be_paused_and_resumed_midway_and_cancel_cleans_the_partial(tmp_path: Path) -> None:
    w, src, dst = await world(tmp_path)
    src.read_chunk = 16
    cid = source_copy(w)
    rel = f"models--org--m/blobs/{SHA}"
    # the small files come from the hub: let them through, hold the large one on the source side
    gate = asyncio.Event()
    real_iter = src.open

    class Slow:
        def __init__(self, inner: Any) -> None:
            self.inner = inner
            self.count = 0

        def __aiter__(self) -> Any:
            return self._it()

        async def _it(self) -> Any:
            async for item in self.inner:
                if self.count >= 2:
                    await gate.wait()
                self.count += 1
                yield item

        async def result(self) -> dict[str, Any]:
            return await self.inner.result()

        async def cancel(self) -> None:
            await self.inner.cancel()

    src.open = lambda op, args=None: Slow(real_iter(op, args)) if op == "fs.read" else real_iter(op, args)    # type: ignore[method-assign,assignment]
    d = await w.downloads.copy(cid, CopyRequest(host_id="beta", root_id="hub"))
    await until(lambda: partial_of(dst, "hub", rel).exists() and partial_of(dst, "hub", rel).stat().st_size == 32, "32 bytes")
    assert (await w.downloads.pause(d.id)).state == "paused"
    assert partial_of(dst, "hub", rel).stat().st_size == 32
    gate.set()
    await w.downloads.resume(d.id)
    await w.downloads.wait(d.id)
    assert w.downloads.get(d.id).state == "completed"
    assert {"rel": f"hub/models--org--m/blobs/{SHA}", "offset": 32} in src.reads
    assert (dst.roots["hub"].path / rel).read_bytes() == PAYLOAD                          # type: ignore[operator]
    await w.stop()


async def test_copy_progress_events_say_it_is_a_copy(tmp_path: Path) -> None:
    w, _, _ = await world(tmp_path)
    events: list[dict[str, Any]] = []

    async def listen() -> None:
        async for ev in w.bus.subscribe("downloads"):
            events.append(ev)

    t = asyncio.create_task(listen())
    await asyncio.sleep(0)
    cid = source_copy(w)
    d = await copy(w, cid, "beta", "hub")
    await asyncio.sleep(0.01)
    t.cancel()
    assert d.state == "completed" and events and all(e["kind"] == "copy" and e["source_copy_id"] == cid for e in events)
    assert events[-1]["state"] == "completed" and events[-1]["done_bytes"] == TOTAL
    await w.stop()


# ---- classification ------------------------------------------------------------------------------------------------------------------------------------
def test_classify_copy_source() -> None:
    from engine_console.domain.fleet import LibraryCopy
    from engine_console.services.library import CopySource

    def src(rel: str, fmt: str = "hf", revision: str | None = None, shards: int | None = None, files: int | None = None,
            warnings: list[str] | None = None) -> CopySource:
        copy = LibraryCopy(id="0" * 16, host_id="h", host_label="H", root_id="r", rel=rel, path="/x/" + rel, size_bytes=1, files=files)
        return CopySource(copy=copy, format=fmt, name="n", revision=revision, quant=None, shards=shards, locations=[("h", "r")],
                          warnings=warnings or [])

    assert classify_copy_source(src("hub/models--org--m", revision=COMMIT)) == HfSource("org/m", "hub/models--org--m", COMMIT, None)
    assert classify_copy_source(src("models--org--m", revision="main")) == HfSource("org/m", "models--org--m", None, None)
    snap = classify_copy_source(src(f"hub/models--org--m/snapshots/{COMMIT}/q/m-00001-of-00002.gguf", "gguf", shards=2))
    assert snap == HfSource("org/m", "hub/models--org--m", COMMIT, ("q/m-00001-of-00002.gguf", "q/m-00002-of-00002.gguf"))
    assert classify_copy_source(src(f"models--org--m/snapshots/{COMMIT}/model.safetensors")) is None      # a subset that is not gguf
    assert classify_copy_source(src("models--a--b--c")) is None                                           # not a repo id we accept
    assert classify_copy_source(src("sub/m-Q4_K_M.gguf", "gguf")) == GgufSource(("sub/m-Q4_K_M.gguf",))
    assert classify_copy_source(src("m-00001-of-00002.gguf", "gguf", shards=2)) == GgufSource(
        ("m-00001-of-00002.gguf", "m-00002-of-00002.gguf"))
    assert classify_copy_source(src("sub/x.gguf", "plain")) is None            # not a counted directory: no exact totals
    assert classify_copy_source(src("dir", "plain", files=3)) == PlainSource("dir", "n", 3, 1)
    assert classify_copy_source(src("", "plain", files=3)) == PlainSource("", "n", 3, 1)
    assert classify_copy_source(src("dir", "plain", files=3, warnings=["size is a lower bound: x"])) is None
    assert classify_copy_source(src("../dir", "plain", files=3)) is None and classify_copy_source(src("dir", "other", files=3)) is None
    assert classify_copy_source(src("../x.gguf", "gguf")) is None


def test_the_hash_helper_agrees_with_the_nodes() -> None:
    assert hashlib.sha256(PAYLOAD).hexdigest() == SHA
