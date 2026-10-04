"""Downloads into any machine's model roots: layouts, resume, verification, refusals, pause/resume/cancel, progress."""
from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub import scan_cache_dir

from engine_console.domain.errors import BadRequest, Conflict, NotFound, ProblemError, Unprocessable
from engine_console.domain.models import Download, DownloadRequest
from engine_console.domain.ports import HubModel, RepoFile
from engine_console.domain.refs import ModelRef
from engine_console.services.downloads import (
    blob_name,
    files_dir,
    hf_repo_rel,
    layout_of,
    snapshot_link,
)
from engine_console.services.library import RootSpec, RootUsage

from .conftest import World
from .fakes import COMMIT, FakeNode, RootCfg, entry, gguf_entry
from .helpers import FILES, PAYLOAD, SHA, TOTAL, partial_of, release, stop_midway, until


def spark(tmp: Path, **kw: Any) -> FakeNode:
    roots = kw.pop("roots", None) or [RootCfg("hf", "hf-home"), RootCfg("hub", "hf-hub"), RootCfg("gguf", "gguf"),
                                      RootCfg("ro", "hf-home", writable=False), RootCfg("pl", "plain"), RootCfg("ol", "ollama")]
    kw.setdefault("addresses", ["10.0.0.5"])
    return FakeNode(tmp, "spark", label="Spark", roots=roots, **kw)


async def make(tmp: Path, node: FakeNode | None = None, **kw: Any) -> tuple[World, FakeNode]:
    node = node or spark(tmp)
    w = World(tmp, [node], **kw)
    w.hub.add("org/m", FILES, config={"model_type": "x"})
    await w.start()
    return w, node


async def run(w: World, repo: str = "org/m", **req: Any) -> Download:
    d = await w.downloads.create(DownloadRequest(repo_id=repo, **req))
    await w.downloads.wait(d.id)
    return w.downloads.get(d.id)


# ---- layouts (pure) --------------------------------------------------------------------------------------------------------------
def test_layout_helpers() -> None:
    assert hf_repo_rel("hf-home", "org/m") == "hub/models--org--m" and hf_repo_rel("hf-hub", "org/m") == "models--org--m"
    assert files_dir("org/m") == "org--m"
    assert snapshot_link("hub/models--o--m", COMMIT, "w.bin", "abc") == (f"hub/models--o--m/snapshots/{COMMIT}/w.bin", "../../blobs/abc")
    assert snapshot_link("models--o--m", COMMIT, "onnx/m.onnx", "abc")[1] == "../../../blobs/abc"      # one folder deeper
    from engine_console.domain.ports import RepoFile
    assert blob_name(RepoFile("w", 1, "f" * 64), COMMIT) == "f" * 64                  # LFS: content addressed
    small = blob_name(RepoFile("config.json", 2), COMMIT)
    assert small.startswith("nolfs-") and len(small) == 46
    assert blob_name(RepoFile("config.json", 2), "b" * 40) != small                    # a new commit is a new blob
    assert blob_name(RepoFile("other.json", 2), COMMIT) != small


@pytest.mark.parametrize(("kind", "usage", "layout"), [
    ("hf-home", RootUsage(), "hf-home"), ("hf-hub", RootUsage(), "hf-hub"), ("gguf", RootUsage(), "files"),
    ("plain", RootUsage(), "files"), ("ollama", RootUsage(), None), ("other", RootUsage(), None),
    ("auto", RootUsage(), "hf-home"),                                                  # empty: start an HF cache
    ("auto", RootUsage(entries=1, hf_home=True), "hf-home"), ("auto", RootUsage(entries=1, hf_hub=True), "hf-hub"),
    ("auto", RootUsage(entries=2, loose=True), "files"),
    ("auto", RootUsage(entries=2, loose=True, other=True), None),                       # would change how the node scans it
    ("auto", RootUsage(entries=1, ollama=True), None), ("auto", RootUsage(entries=1, other=True), None),
])
def test_layout_of_a_root(kind: str, usage: RootUsage, layout: str | None) -> None:
    assert layout_of(RootSpec("r", "/x", kind, True, None), usage) == layout


def test_an_empty_auto_root_takes_what_the_caller_is_writing() -> None:
    auto = RootSpec("r", "/x", "auto", True, None)
    assert layout_of(auto, RootUsage()) == "hf-home" and layout_of(auto, RootUsage(), "files") == "files"
    assert layout_of(auto, RootUsage(entries=1, hf_hub=True), "files") == "hf-hub"       # a root with an HF cache stays one


# ---- hf-home, hf-hub and gguf roots ------------------------------------------------------------------------------------------------
async def test_download_into_an_hf_home_root_writes_a_real_hf_cache(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "completed" and d.host_id == "spark" and d.root_id == "hf" and d.kind == "hf"
    assert (d.total_bytes, d.done_bytes, d.files_total, d.files_done, d.commit_sha) == (TOTAL, TOTAL, 3, 3, COMMIT)
    assert d.error is None and d.current_file is None
    repo = node.roots["hf"].path / "hub" / "models--org--m"          # type: ignore[operator]
    assert (repo / "blobs" / SHA).read_bytes() == PAYLOAD
    link = repo / "snapshots" / COMMIT / "w.safetensors"
    assert link.is_symlink() and os.readlink(link) == f"../../blobs/{SHA}" and link.read_bytes() == PAYLOAD
    assert (repo / "snapshots" / COMMIT / "config.json").read_bytes() == b"{}"
    assert (repo / "refs" / "main").read_text() == COMMIT
    assert not list(repo.rglob("*.incomplete"))
    # the oracle: the Hugging Face client library reads it as a cache
    info = scan_cache_dir(node.roots["hf"].path / "hub")             # type: ignore[operator]
    [cached] = [r for r in info.repos if r.repo_id == "org/m"]
    assert [rev.commit_hash for rev in cached.revisions] == [COMMIT] and len(next(iter(cached.revisions)).files) == 3
    # and the library sees the new model once the node has scanned
    await w.settle()
    assert node.scans == [{"force": True}]
    found = w.library.find("spark", ModelRef(format="hf", repo_id="org/m"))
    assert found is not None and found.size_bytes == TOTAL and found.root_id == "hf"
    await w.stop()


async def test_download_into_an_hf_hub_root_has_no_hub_directory(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    d = await run(w, host_id="spark", root_id="hub")
    assert d.state == "completed"
    root = node.roots["hub"].path
    assert (root / "models--org--m" / "blobs" / SHA).is_file() and (root / "models--org--m" / "refs" / "main").is_file()  # type: ignore[operator]
    assert not (root / "hub").exists()                                                                                   # type: ignore[operator]
    assert [r.repo_id for r in scan_cache_dir(root).repos] == ["org/m"]                                                  # type: ignore[arg-type]
    await w.stop()


async def test_a_revision_names_the_ref(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    d = await run(w, host_id="spark", root_id="hf", revision="v1.0")
    assert d.state == "completed" and d.revision == "v1.0"
    assert (node.roots["hf"].path / "hub" / "models--org--m" / "refs" / "v1.0").read_text() == COMMIT     # type: ignore[operator]
    await w.stop()


async def test_a_gguf_root_gets_the_selected_files_under_the_repo_name(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    q4, q8 = b"q4" * 20, b"q8" * 30
    w.hub.add("unsloth/Qwen3-GGUF", {"Qwen3-Q4_K_M.gguf": q4, "Qwen3-Q8_0.gguf": q8, "README.md": b"hi", "sub/extra.gguf": b"zz"})
    d = await run(w, "unsloth/Qwen3-GGUF", host_id="spark", root_id="gguf", allow_patterns=["Qwen3-Q4_K_M.gguf"])
    assert d.state == "completed" and d.total_bytes == len(q4) and d.files_total == 1
    target = node.roots["gguf"].path / "unsloth--Qwen3-GGUF"          # type: ignore[operator]
    assert [p.name for p in target.rglob("*") if p.is_file()] == ["Qwen3-Q4_K_M.gguf"]
    assert (target / "Qwen3-Q4_K_M.gguf").read_bytes() == q4
    assert node.writes[0]["sha256"] == hashlib.sha256(q4).hexdigest()                      # verified by the node
    assert not list(target.rglob("*.incomplete")) and not (target / "blobs").exists()      # no blobs, no links
    await w.settle()
    m = w.library.models(format="gguf")
    assert [x.key for x in m] == ["gguf:Qwen3-Q4_K_M:Q4_K_M"]
    await w.stop()


async def test_a_gguf_root_refuses_a_selection_with_no_gguf_file(tmp_path: Path) -> None:
    w, _ = await make(tmp_path)
    d = await run(w, host_id="spark", root_id="gguf")                  # FILES holds no .gguf
    assert d.state == "failed" and d.error_code == "no_gguf_files"
    await w.stop()


async def test_a_gguf_root_replaces_an_older_file_of_the_same_size(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    new = b"N" * 40
    w.hub.add("org/g", {"m-Q4_K_M.gguf": new})
    old = node.roots["gguf"].path / "org--g" / "m-Q4_K_M.gguf"       # type: ignore[operator]
    old.parent.mkdir(parents=True)
    old.write_bytes(b"O" * 40)                                         # same size, other bytes: the size proves nothing
    d = await run(w, "org/g", host_id="spark", root_id="gguf")
    assert d.state == "completed" and old.read_bytes() == new
    await w.stop()


async def test_a_small_file_is_a_new_blob_at_a_new_commit(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    await run(w, host_id="spark", root_id="hf")
    w.hub.add("org/m", {**FILES, "config.json": b'{"v": 2}'}, sha="d" * 40, config={"model_type": "x"})
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "completed"
    blobs = sorted(p.name for p in (node.roots["hf"].path / "hub" / "models--org--m" / "blobs").iterdir())      # type: ignore[operator]
    assert len([b for b in blobs if b.startswith("nolfs-")]) == 4 and blobs.count(SHA) == 1    # 2 small x 2 commits; the weights once
    assert (node.roots["hf"].path / "hub" / "models--org--m" / "snapshots" / ("d" * 40) / "config.json").read_bytes() == b'{"v": 2}'  # type: ignore[operator]
    await w.stop()


async def test_two_files_with_the_same_content_share_a_blob(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    w.hub.add("org/twin", {"a.safetensors": PAYLOAD, "b.safetensors": PAYLOAD})
    d = await run(w, "org/twin", host_id="spark", root_id="hf")
    assert d.state == "completed" and d.done_bytes == d.total_bytes == 100 and d.files_done == 2
    snaps = node.roots["hf"].path / "hub" / "models--org--twin"        # type: ignore[operator]
    assert len(list((snaps / "blobs").iterdir())) == 1
    assert (snaps / "snapshots" / COMMIT / "a.safetensors").read_bytes() == (snaps / "snapshots" / COMMIT / "b.safetensors").read_bytes()
    assert len([w_ for w_ in node.writes if w_["rel"].endswith(SHA)]) == 1         # sent once
    await w.stop()


async def test_a_zero_byte_file_needs_no_download_request(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    w.hub.add("org/e", {"empty.bin": b"", "w.safetensors": PAYLOAD})
    d = await run(w, "org/e", host_id="spark", root_id="hf")
    assert d.state == "completed" and d.total_bytes == 50
    assert [o[2] for o in w.hub.opens] == ["w.safetensors"]                           # nothing to ask Hugging Face for
    assert (node.roots["hf"].path / "hub" / "models--org--e" / "snapshots" / COMMIT / "empty.bin").read_bytes() == b""  # type: ignore[operator]
    await w.stop()


async def test_allow_patterns_select_files(tmp_path: Path) -> None:
    w, _ = await make(tmp_path)
    assert (await run(w, host_id="spark", root_id="hf", allow_patterns=["*.json"])).files_total == 1
    d = await run(w, "org/m", host_id="spark", root_id="hub", allow_patterns=["nothing*"])
    assert d.state == "failed" and d.error_code == "no_files"
    await w.stop()


# ---- default roots and auto roots ----------------------------------------------------------------------------------------------------------
async def test_the_default_root_is_the_first_writable_hf_root(tmp_path: Path) -> None:
    node = spark(tmp_path, roots=[RootCfg("ro", "hf-home", writable=False), RootCfg("g", "gguf"), RootCfg("pl", "plain"),
                                  RootCfg("hub", "hf-hub"), RootCfg("hf", "hf-home")])
    w, _ = await make(tmp_path, node)
    d = await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="spark"))
    assert d.root_id == "hub"            # not the read-only one, not the gguf or plain roots: the first writable HF cache
    await w.downloads.wait(d.id)
    assert w.downloads.get(d.id).state == "completed"
    await w.stop()


async def test_an_empty_auto_root_starts_an_hf_cache_and_one_with_models_keeps_its_kind(tmp_path: Path) -> None:
    node = spark(tmp_path, roots=[RootCfg("a1", "auto"), RootCfg("a2", "auto"), RootCfg("a3", "auto"), RootCfg("a4", "auto")])
    w = World(tmp_path, [node])
    w.hub.add("org/m", FILES)
    node.entries = []
    node.scan_disk_enabled = False
    node.entries = [entry("a2", "models--x--y", "hf", "x/y"), gguf_entry("a3", "loose.gguf"), entry("a4", "manifests/q", "ollama", "q:1")]
    await w.start()
    d1 = await run(w, host_id="spark", root_id="a1")
    assert d1.state == "completed" and (node.roots["a1"].path / "hub" / "models--org--m").is_dir()              # type: ignore[operator]
    d2 = await run(w, host_id="spark", root_id="a2")
    assert d2.state == "completed" and (node.roots["a2"].path / "models--org--m").is_dir()                      # type: ignore[operator]
    d3 = await run(w, host_id="spark", root_id="a3", allow_patterns=["config.json"])
    assert d3.state == "completed" and (node.roots["a3"].path / "org--m" / "config.json").is_file()            # type: ignore[operator]
    with pytest.raises(Unprocessable) as ei:
        await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="spark", root_id="a4"))
    assert ei.value.code == "root_kind_unsupported"
    await w.stop()


# ---- resume and verification ---------------------------------------------------------------------------------------------------------
async def test_a_partial_on_the_node_is_resumed_not_restarted(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    rel = f"hub/models--org--m/blobs/{SHA}"
    part = partial_of(node, "hf", rel)
    part.parent.mkdir(parents=True)
    part.write_bytes(PAYLOAD[:20])
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "completed" and d.done_bytes == TOTAL
    assert ("org/m", COMMIT, "w.safetensors", 20) in w.hub.opens                       # the Range started where the partial ends
    assert next(x for x in node.writes if x["rel"] == rel) == {"rel": rel, "offset": 20, "size": 50, "sha256": SHA}
    assert (node.roots["hf"].path / rel).read_bytes() == PAYLOAD                      # type: ignore[operator]
    assert not part.exists()
    await w.stop()


async def test_a_server_that_ignores_range_restarts_the_file_cleanly(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    w.hub.ignore_range = True
    rel = f"hub/models--org--m/blobs/{SHA}"
    part = partial_of(node, "hf", rel)
    part.parent.mkdir(parents=True)
    part.write_bytes(PAYLOAD[:20])
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "completed" and d.done_bytes == d.total_bytes == TOTAL
    assert next(x for x in node.writes if x["rel"] == rel)["offset"] == 0                # offset 0 truncates the old partial
    assert (node.roots["hf"].path / rel).read_bytes() == PAYLOAD                      # type: ignore[operator]
    await w.stop()


async def test_a_partial_that_is_already_whole_only_needs_the_nodes_verification(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    rel = f"hub/models--org--m/blobs/{SHA}"
    part = partial_of(node, "hf", rel)
    part.parent.mkdir(parents=True)
    part.write_bytes(PAYLOAD)
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "completed" and "w.safetensors" not in [o[2] for o in w.hub.opens]
    await w.stop()


async def test_a_partial_longer_than_the_file_starts_over(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    rel = f"hub/models--org--m/blobs/{SHA}"
    part = partial_of(node, "hf", rel)
    part.parent.mkdir(parents=True)
    part.write_bytes(b"x" * 500)
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "completed" and next(x for x in node.writes if x["rel"] == rel)["offset"] == 0
    await w.stop()


async def test_a_hash_mismatch_discards_the_partial_and_a_resume_refetches(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    w.hub.corrupt.add("w.safetensors")
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "failed" and d.error_code == "hash_mismatch" and d.error is not None and d.error.startswith("Spark: ")
    rel = f"hub/models--org--m/blobs/{SHA}"
    assert not partial_of(node, "hf", rel).exists() and not (node.roots["hf"].path / rel).exists()           # type: ignore[operator]
    w.hub.corrupt.clear()
    await w.downloads.resume(d.id)
    await w.downloads.wait(d.id)
    d2 = w.downloads.get(d.id)
    assert d2.state == "completed" and d2.error is None and d2.error_code is None
    assert [o for o in w.hub.opens if o[2] == "w.safetensors"][-1][3] == 0                  # refetched from the start
    assert (node.roots["hf"].path / rel).read_bytes() == PAYLOAD                          # type: ignore[operator]
    await w.stop()


async def test_a_node_io_error_midway_keeps_the_partial_and_resume_continues_from_it(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    node.fail_after_bytes = 30
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "failed" and d.error_code == "io_error" and d.error is not None and d.error.startswith("Spark: ")
    rel = f"hub/models--org--m/blobs/{SHA}"
    assert partial_of(node, "hf", rel).stat().st_size == 30
    await w.downloads.resume(d.id)
    await w.downloads.wait(d.id)
    assert w.downloads.get(d.id).state == "completed"
    assert [x for x in node.writes if x["rel"] == rel][-1]["offset"] == 30
    assert ("org/m", COMMIT, "w.safetensors", 30) in w.hub.opens
    await w.stop()


async def test_a_source_that_ends_early_keeps_the_partial_and_reports_a_size_mismatch(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    data, short = w.hub.repos["org/m"][1], PAYLOAD[:35]
    w.hub.repos["org/m"][1]["w.safetensors"] = short                      # the server sends 35 of the 50 bytes it listed
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "failed" and d.error_code == "size_mismatch"
    assert partial_of(node, "hf", f"hub/models--org--m/blobs/{SHA}").stat().st_size == 35        # no end(): nothing discarded
    data["w.safetensors"] = PAYLOAD
    await w.downloads.resume(d.id)
    await w.downloads.wait(d.id)
    assert w.downloads.get(d.id).state == "completed"
    await w.stop()


async def test_a_source_that_sends_too_much_is_stopped(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    w.hub.repos["org/m"][0].files[1].size = 30                              # listed as 30 bytes, served as 50
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "failed" and d.error_code == "size_mismatch" and "more than the declared 30" in (d.error or "")
    assert partial_of(node, "hf", f"hub/models--org--m/blobs/{SHA}").stat().st_size <= 30
    await w.stop()


def _add_file(path: str, size: int = 1, sha: str | None = None) -> Callable[[HubModel], None]:
    return lambda hm: hm.files.append(RepoFile(path, size, sha))


def _set_sha(value: str | None) -> Callable[[HubModel], None]:
    return lambda hm: setattr(hm, "sha", value)


@pytest.mark.parametrize(("mutate", "code"), [
    (_add_file("../evil"), "hf_bad_path"), (_add_file("/abs"), "hf_bad_path"), (_add_file("a\\b"), "hf_bad_path"),
    (_add_file("a/./b"), "hf_bad_path"), (_add_file("x.bin", sha="NOT-A-HASH"), "hf_bad_hash"),
    (_add_file("x.bin", size=-1), "hf_bad_size"), (_set_sha("short"), "hf_bad_sha"), (_set_sha(None), "hf_bad_sha"),
])
async def test_a_hostile_hub_listing_fails_the_download_before_anything_is_written(
        tmp_path: Path, mutate: Callable[[HubModel], None], code: str) -> None:
    w, node = await make(tmp_path)
    mutate(w.hub.repos["org/m"][0])
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "failed" and d.error_code == code
    assert not node.writes and not [c for op, c in node.calls if op in ("fs.symlink", "fs.put_small")]
    await w.stop()


async def test_a_gated_repo_fails_with_the_hubs_message(tmp_path: Path) -> None:
    w, _ = await make(tmp_path)
    w.hub.gated.add("org/m")
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "failed" and d.error_code == "hf_gated" and "gated" in (d.error or "")
    await w.stop()


async def test_an_egress_proxy_that_is_down_fails_the_download_closed_and_writes_nothing(tmp_path: Path) -> None:
    from engine_console.domain.errors import EgressProxyUnavailable
    w, node = await make(tmp_path)

    async def down(repo_id: str, revision: str | None) -> Any:
        raise EgressProxyUnavailable("egress proxy unreachable: refusing to connect directly")

    w.hub.model = down                                                  # type: ignore[method-assign]
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "failed" and d.error_code == "egress_proxy_unavailable" and "directly" in (d.error or "")
    assert not node.writes and w.hub.opens == []                        # never fell back to anything else
    await w.stop()


async def test_the_proxy_going_away_midway_keeps_the_partial_for_a_resume(tmp_path: Path) -> None:
    from engine_console.domain.errors import EgressProxyUnavailable
    w, node = await make(tmp_path)
    real = w.hub.open_file
    calls = {"n": 0}

    async def flaky(repo_id: str, revision: str, path: str, start: int) -> Any:
        calls["n"] += 1
        if path == "w.safetensors" and calls["n"] < 3:
            stream = await real(repo_id, revision, path, start)
            inner = stream.chunks

            async def chunks() -> Any:
                n = 0
                async for c in inner():
                    n += 1
                    if n == 2:
                        raise EgressProxyUnavailable("egress proxy unreachable: refusing to connect directly")
                    yield c

            stream.chunks = chunks                                       # type: ignore[method-assign]
            return stream
        return await real(repo_id, revision, path, start)

    w.hub.open_file = flaky                                              # type: ignore[method-assign]
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "failed" and d.error_code == "egress_proxy_unavailable"
    rel = f"hub/models--org--m/blobs/{SHA}"
    assert partial_of(node, "hf", rel).stat().st_size == 16               # the one chunk that arrived is on the node
    await w.downloads.resume(d.id)
    await w.downloads.wait(d.id)
    assert w.downloads.get(d.id).state == "completed" and (node.roots["hf"].path / rel).read_bytes() == PAYLOAD   # type: ignore[operator]
    await w.stop()


# ---- space ---------------------------------------------------------------------------------------------------------------------------------
async def test_the_disk_precheck_uses_the_free_space_of_the_target_mount(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    node.free = 10
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "failed" and d.error_code == "insufficient_disk" and "Spark" in (d.error or "") and "GB free" in (d.error or "")
    assert not node.writes
    node.free = 10**12
    await w.downloads.resume(d.id)
    await w.downloads.wait(d.id)
    assert w.downloads.get(d.id).state == "completed"
    await w.stop()


async def test_bytes_already_on_the_node_do_not_count_against_the_free_space(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    await run(w, host_id="spark", root_id="hf")
    node.free = 5                                              # no room for anything new, but nothing is needed
    w.downloads._db.execute("DELETE FROM downloads")
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "completed"
    await w.stop()


async def test_a_node_that_runs_out_of_space_reads_as_insufficient_disk(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    node.space_left = 20
    d = await run(w, host_id="spark", root_id="hf")
    assert d.state == "failed" and d.error_code == "insufficient_disk" and (d.error or "").startswith("Spark: ")
    await w.stop()


# ---- refusals at create ----------------------------------------------------------------------------------------------------------------
async def test_refusals_say_why_and_leave_no_row(tmp_path: Path) -> None:
    w, node = await make(tmp_path)

    async def refused(**req: Any) -> ProblemError:
        with pytest.raises(ProblemError) as ei:
            await w.downloads.create(DownloadRequest(repo_id="org/m", **req))
        return ei.value

    e = await refused(host_id="nobody")
    assert (e.status, e.code) == (404, "unknown_host")
    e = await refused(host_id="spark", root_id="ro")
    assert (e.status, e.code) == (409, "root_not_writable") and "read-only" in e.detail and e.detail.startswith("Spark: ")
    e = await refused(host_id="spark", root_id="nope")
    assert (e.status, e.code) == (422, "unknown_root")
    for kind_root in ("pl", "ol"):
        e = await refused(host_id="spark", root_id=kind_root)
        assert (e.status, e.code) == (422, "root_kind_unsupported")
    node.state = "down"
    e = await refused(host_id="spark")
    assert (e.status, e.code) == (503, "node_unavailable") and e.detail.startswith("Spark: ")
    node.state = "up"
    node.mode = "monitor"
    e = await refused(host_id="spark")
    assert (e.status, e.code) == (409, "monitor_only") and "monitor" in e.detail
    node.mode = "manage"
    assert w.store.one("SELECT COUNT(*) AS n FROM downloads")["n"] == 0            # type: ignore[index]
    await w.stop()


async def test_no_writable_root_that_can_hold_a_repo(tmp_path: Path) -> None:
    w, _ = await make(tmp_path, spark(tmp_path, roots=[RootCfg("ro", "hf-home", writable=False), RootCfg("pl", "plain")]))
    with pytest.raises(Conflict) as ei:
        await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="spark"))
    assert ei.value.code == "no_writable_root"
    await w.stop()


async def test_bad_requests(tmp_path: Path) -> None:
    w, _ = await make(tmp_path)
    for bad in ("../etc", "a--b/c", "x"):
        with pytest.raises(BadRequest) as ei:
            await w.downloads.create(DownloadRequest.model_construct(repo_id=bad, host_id="spark"))
        assert ei.value.code == "invalid_repo_id"
    for pats in (["x"] * 65, ["a" * 300], [""]):
        with pytest.raises(BadRequest) as ei:
            await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="spark", allow_patterns=pats))
        assert ei.value.code == "invalid_params"
    await w.stop()


async def test_a_run_time_refusal_is_a_failure_that_can_be_retried(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    w.hub.corrupt.add("w.safetensors")
    d = await run(w, host_id="spark", root_id="hf")
    assert d.error_code == "hash_mismatch"
    node.mode = "monitor"                                              # the operator changed the node's policy meanwhile
    await w.downloads.resume(d.id)
    await w.downloads.wait(d.id)
    d2 = w.downloads.get(d.id)
    assert d2.state == "failed" and d2.error_code == "monitor_only"
    node.mode = "manage"
    w.hub.corrupt.clear()
    await w.downloads.resume(d.id)
    await w.downloads.wait(d.id)
    assert w.downloads.get(d.id).state == "completed"
    await w.stop()


# ---- queue ---------------------------------------------------------------------------------------------------------------------------------
async def test_the_queue_is_capped_and_a_repeat_of_a_running_download_is_refused(tmp_path: Path) -> None:
    w, _ = await make(tmp_path, max_queue=2)
    w.hub.gate = asyncio.Event()                                       # nothing finishes
    w.hub.add("org/two", FILES)
    w.hub.add("org/three", FILES)
    d1 = await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="spark", root_id="hf"))
    with pytest.raises(Conflict) as ei:
        await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="spark", root_id="hf"))
    assert ei.value.code == "download_exists"
    await w.downloads.create(DownloadRequest(repo_id="org/two", host_id="spark", root_id="hf"))
    with pytest.raises(Conflict) as ei2:
        await w.downloads.create(DownloadRequest(repo_id="org/three", host_id="spark", root_id="hf"))
    assert ei2.value.code == "download_queue_full"
    await w.downloads.cancel(d1.id)
    await w.stop()


async def test_the_same_repo_may_go_to_two_different_roots(tmp_path: Path) -> None:
    w, _ = await make(tmp_path)
    a = await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="spark", root_id="hf"))
    b = await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="spark", root_id="hub"))
    await w.downloads.wait(a.id)
    await w.downloads.wait(b.id)
    assert w.downloads.get(a.id).state == w.downloads.get(b.id).state == "completed"
    await w.stop()


async def test_recover_parks_running_rows(tmp_path: Path) -> None:
    w, _ = await make(tmp_path)
    w.store.execute("INSERT INTO downloads(id,repo_id,state,created_at,updated_at) VALUES('d','a/b','running',1,1)")
    w.store.execute("INSERT INTO downloads(id,repo_id,state,created_at,updated_at,kind,host_id) VALUES('c','a/b','queued',1,1,'copy','x')")
    w.downloads.recover()
    assert w.downloads.get("d").state == "paused" and w.downloads.get("c").state == "paused"
    assert w.downloads.get("d").host_id == "local" and w.downloads.get("d").kind == "hf" and w.downloads.get("d").root_id is None
    await w.stop()


async def test_a_v1_database_keeps_its_downloads_across_the_migration(tmp_path: Path) -> None:
    import sqlite3
    from importlib import resources

    from engine_console.services.store import Store, _split_sql
    db = tmp_path / "v1.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, applied_at REAL)")
    for name in ("0001_init.sql", "0002_container_port.sql"):             # the schema as v1 shipped it
        for stmt in _split_sql((resources.files("engine_console.migrations") / name).read_text()):
            con.execute(stmt)
        con.execute("INSERT INTO schema_migrations VALUES (?, 1)", (name,))
    con.execute("INSERT INTO downloads(id,repo_id,revision,state,total_bytes,done_bytes,created_at,updated_at)"
                " VALUES('dl_old','org/old','main','completed',10,10,1,2)")
    con.commit()
    con.close()
    store = Store(db)
    assert "0004_library_and_transfers.sql" in {r["name"] for r in store.all("SELECT name FROM schema_migrations")}
    w = World(tmp_path, [spark(tmp_path)], store=store)
    d = w.downloads.get("dl_old")
    assert (d.state, d.repo_id, d.revision, d.host_id, d.root_id, d.kind, d.source_copy_id) == (
        "completed", "org/old", "main", "local", None, "hf", None)
    assert store.one("SELECT COUNT(*) AS n FROM library_snapshots")["n"] == 0                  # type: ignore[index]
    await w.stop()


async def test_list_and_get(tmp_path: Path) -> None:
    w, _ = await make(tmp_path)
    d = await run(w, host_id="spark", root_id="hf")
    page = w.downloads.list_page(10)
    assert [x.id for x in page.items] == [d.id] and page.next_cursor is None
    with pytest.raises(NotFound):
        w.downloads.get("nope")
    await w.stop()


# ---- pause, resume, cancel ---------------------------------------------------------------------------------------------------------------------
async def test_pause_keeps_the_partial_and_resume_continues_from_it(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    stop_midway(w)
    rel = f"hub/models--org--m/blobs/{SHA}"
    d = await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="spark", root_id="hf"))
    await until(lambda: partial_of(node, "hf", rel).exists() and partial_of(node, "hf", rel).stat().st_size == 32, "32 bytes")
    paused = await w.downloads.pause(d.id)
    assert paused.state == "paused" and paused.current_file == "w.safetensors"
    assert partial_of(node, "hf", rel).stat().st_size == 32                        # the partial stays on the node
    release(w)
    await w.downloads.resume(d.id)
    await w.downloads.wait(d.id)
    done = w.downloads.get(d.id)
    assert done.state == "completed" and done.done_bytes == TOTAL
    assert ("org/m", COMMIT, "w.safetensors", 32) in w.hub.opens                   # continued where it stopped
    assert next(x for x in reversed(node.writes) if x["rel"] == rel)["offset"] == 32
    assert (node.roots["hf"].path / rel).read_bytes() == PAYLOAD                  # type: ignore[operator]
    await w.stop()


async def test_cancel_removes_the_partial_when_the_nodes_policy_allows(tmp_path: Path) -> None:
    for allow, left in ((True, False), (False, True)):
        sub = tmp_path / str(allow)
        w, node = await make(sub, spark(sub, allow_delete=allow))
        stop_midway(w)
        rel = f"hub/models--org--m/blobs/{SHA}"
        d = await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="spark", root_id="hf"))
        await until(lambda n=node, r=rel: partial_of(n, "hf", r).exists() and partial_of(n, "hf", r).stat().st_size == 32, "32 bytes")  # type: ignore[misc]
        assert (await w.downloads.cancel(d.id)).state == "cancelled"
        assert partial_of(node, "hf", rel).exists() is left                         # removed only where the node allows deletes
        with pytest.raises(Conflict):
            await w.downloads.cancel(d.id)
        await w.stop()


async def test_cancelling_a_paused_download_cleans_up_too(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    stop_midway(w)
    rel = f"hub/models--org--m/blobs/{SHA}"
    d = await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="spark", root_id="hf"))
    await until(lambda: partial_of(node, "hf", rel).exists() and partial_of(node, "hf", rel).stat().st_size == 32, "32 bytes")
    await w.downloads.pause(d.id)
    assert partial_of(node, "hf", rel).exists()
    await w.downloads.cancel(d.id)
    assert not partial_of(node, "hf", rel).exists() and w.downloads.get(d.id).state == "cancelled"
    await w.stop()


async def test_cancelling_a_failed_download_removes_the_partial_it_left(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    node.fail_after_bytes = 30
    d = await run(w, host_id="spark", root_id="hf")
    rel = f"hub/models--org--m/blobs/{SHA}"
    assert d.state == "failed" and partial_of(node, "hf", rel).stat().st_size == 30
    assert (await w.downloads.cancel(d.id)).state == "cancelled"
    assert not partial_of(node, "hf", rel).exists()
    await w.stop()


async def test_deleting_a_running_download_stops_it_and_removes_the_partial(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    stop_midway(w)
    rel = f"hub/models--org--m/blobs/{SHA}"
    d = await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="spark", root_id="hf"))
    await until(lambda: partial_of(node, "hf", rel).exists() and partial_of(node, "hf", rel).stat().st_size == 32, "32 bytes")
    await w.downloads.delete(d.id)
    assert not partial_of(node, "hf", rel).exists()
    with pytest.raises(NotFound):
        w.downloads.get(d.id)
    await w.stop()


async def test_pause_and_cancel_right_after_create_are_honoured_before_the_first_step(tmp_path: Path) -> None:
    w, _ = await make(tmp_path)
    d = await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="spark", root_id="hf"))
    assert (await w.downloads.pause(d.id)).state == "paused"
    with pytest.raises(Conflict):
        await w.downloads.pause(d.id)
    d2 = await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="spark", root_id="hub"))
    assert (await w.downloads.cancel(d2.id)).state == "cancelled"
    with pytest.raises(Conflict):
        await w.downloads.resume(d2.id)
    await w.stop()


async def test_shutdown_parks_running_downloads(tmp_path: Path) -> None:
    w, _ = await make(tmp_path)
    w.hub.gate = asyncio.Event()
    d = await w.downloads.create(DownloadRequest(repo_id="org/m", host_id="spark", root_id="hf"))
    await asyncio.sleep(0.02)
    await w.downloads.shutdown()
    assert w.downloads.get(d.id).state == "paused"
    await w.stop()


# ---- progress ------------------------------------------------------------------------------------------------------------------------------------
class Tick:
    """A clock that advances 0.3 s every time it is read, so progress is published on every chunk."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.3
        return self.t


async def test_progress_is_published_on_the_bus_with_speed_and_eta(tmp_path: Path) -> None:
    w, _ = await make(tmp_path, clock=Tick())
    events: list[dict[str, Any]] = []

    async def listen() -> None:
        async for ev in w.bus.subscribe("downloads"):
            events.append(ev)

    t = asyncio.create_task(listen())
    await asyncio.sleep(0)
    d = await run(w, host_id="spark", root_id="hf")
    await asyncio.sleep(0.01)
    t.cancel()
    assert d.state == "completed" and events
    assert all(e["host_id"] == "spark" and e["root_id"] == "hf" and e["kind"] == "hf" and e["id"] == d.id for e in events)
    running = [e for e in events if e["state"] == "running" and e["done_bytes"] > 0]
    assert running and [e["done_bytes"] for e in running] == sorted(e["done_bytes"] for e in running)
    assert any(e["speed_bps"] > 0 and e["eta_s"] is not None for e in running)
    assert max(e["done_bytes"] for e in events) <= TOTAL and events[-1]["state"] == "completed" and events[-1]["done_bytes"] == TOTAL
    assert any(e["current_file"] == "w.safetensors" for e in running)
    await w.stop()


async def test_progress_counts_a_resumed_partial_as_done(tmp_path: Path) -> None:
    w, node = await make(tmp_path, clock=Tick())
    part = partial_of(node, "hf", f"hub/models--org--m/blobs/{SHA}")
    part.parent.mkdir(parents=True)
    part.write_bytes(PAYLOAD[:20])
    events: list[dict[str, Any]] = []

    async def listen() -> None:
        async for ev in w.bus.subscribe("downloads"):
            events.append(ev)

    t = asyncio.create_task(listen())
    await asyncio.sleep(0)
    await run(w, host_id="spark", root_id="hf")
    await asyncio.sleep(0.01)
    t.cancel()
    first = next(e for e in events if e["state"] == "running")
    assert first["done_bytes"] == 20 and first["total_bytes"] == TOTAL           # not 0: the node already has those bytes
    await w.stop()


# ---- other machines --------------------------------------------------------------------------------------------------------------------------------
async def test_downloads_go_to_whichever_machine_is_named(tmp_path: Path) -> None:
    a = spark(tmp_path)
    b = FakeNode(tmp_path, "gpu2", label="GPU box", roots=[RootCfg("models", "hf-hub")])
    w = World(tmp_path, [a, b])
    w.hub.add("org/m", FILES)
    await w.start()
    d1 = await run(w, host_id="gpu2")
    assert d1.state == "completed" and d1.root_id == "models" and d1.host_id == "gpu2"
    assert (b.roots["models"].path / "models--org--m" / "refs" / "main").is_file()          # type: ignore[operator]
    assert not (a.base / "hf").exists() or not list((a.base / "hf").rglob("blobs"))        # nothing leaked to the other machine
    assert [x[0] for x in a.calls if x[0].startswith("fs.")] == []
    await w.settle()
    assert w.library.find("gpu2", ModelRef(format="hf", repo_id="org/m")) is not None
    assert w.library.find("spark", ModelRef(format="hf", repo_id="org/m")) is None
    await w.stop()


async def test_a_node_is_only_ever_asked_for_file_and_library_operations(tmp_path: Path) -> None:
    w, node = await make(tmp_path)
    await run(w, host_id="spark", root_id="hf")
    assert {op for op, _ in node.calls} <= {"models.list", "models.scan", "fs.stat", "fs.write", "fs.symlink", "fs.put_small"}
    text = repr(node.calls)
    assert "huggingface" not in text and "http" not in text and "Authorization" not in text and "Bearer" not in text
    await w.stop()
