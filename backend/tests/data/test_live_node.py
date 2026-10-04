"""Live check against the REAL node: two `ecnode --stdio` subprocesses, started the way the hub starts its local node and
reached through the hub's own `NodeLink`, with the library and the download service on top.

Opt in with ENGINE_CONSOLE_LIVE_NODE=1. It touches only temp directories: the nodes run with a PATH that has no docker and
no nvidia-smi, their policies name temp roots, and Hugging Face is the in-memory fake hub (no network). It exists because
the in-process fake node is only as faithful as its author's reading of docs/FLEET-PROTOCOL.md; this is the proof that the
real `fs.stat` / `fs.write` / `fs.read` / `fs.symlink` / `fs.put_small` / `fs.delete` / `models.*` behave as the data side
assumes (partials, resume offsets, hash checks, relative links, the scanner's entry shapes).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from engine_console.domain.fleet import CopyRequest, HostConfig
from engine_console.domain.models import DownloadRequest
from engine_console.domain.refs import ModelRef
from engine_console.services.common import EventBus
from engine_console.services.downloads import DownloadService
from engine_console.services.fleet import NodeLink
from engine_console.services.hf import HfService
from engine_console.services.library import LibraryService
from engine_console.services.settings import SettingsService
from engine_console.services.store import Store

from .conftest import make_adapters, make_settings
from .fakes import COMMIT, FakeHub

pytestmark = pytest.mark.skipif(os.environ.get("ENGINE_CONSOLE_LIVE_NODE") != "1",
                                reason="live node check: set ENGINE_CONSOLE_LIVE_NODE=1 to run it")


def tiny_gguf(path: Path) -> None:
    """A minimal valid GGUF v3 file (header + two tensor descriptors, no tensor data): enough for the node's reader."""
    import struct

    def string(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    kvs = [("general.architecture", 8, string("llama")), ("general.name", 8, string("Tiny")),
           ("general.file_type", 4, struct.pack("<I", 15)), ("llama.context_length", 4, struct.pack("<I", 4096)),
           ("llama.block_count", 4, struct.pack("<I", 2)), ("llama.embedding_length", 4, struct.pack("<I", 64)),
           ("llama.attention.head_count", 4, struct.pack("<I", 4)), ("llama.attention.head_count_kv", 4, struct.pack("<I", 2))]
    body = b"".join(string(k) + struct.pack("<I", t) + v for k, t, v in kvs)
    tensors = b"".join(string(n) + struct.pack("<I", 2) + struct.pack("<QQ", 64, 16) + struct.pack("<I", 0) + struct.pack("<Q", 0)
                       for n in ("a.weight", "b.weight"))
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 2, len(kvs)) + body + tensors)


class Fleet:
    def __init__(self, links: dict[str, NodeLink]) -> None:
        self.links = links

    def hosts(self) -> list[HostConfig]:
        return [x.host for x in self.links.values()]

    def node(self, host_id: str) -> NodeLink:
        return self.links[host_id]


def write_policy(path: Path, roots: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({
        "version": 1, "mode": "manage", "docker": {"binary": "/nonexistent/docker", "context": None},
        "engine_images": ["vllm/vllm-openai:*"], "model_roots": roots, "scripts": [], "allow_pull": False,
        "allow_delete": True, "allow_ipc_host": False, "external_control": "none", "max_shm_gib": 8}))
    path.chmod(0o644)


def start_link(bus: EventBus, host_id: str, label: str, policy: Path) -> NodeLink:
    argv = [sys.executable, "-I", "-m", "ecnode", "--stdio", "--policy", str(policy)]
    env = {"HOME": str(policy.parent), "PATH": "/nonexistent", "LANG": "C.UTF-8"}
    return NodeLink(HostConfig(id=host_id, label=label), argv, env, bus)


async def wait_for(cond: Callable[[], bool | Awaitable[bool]], what: str, timeout: float = 20.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        r = cond()
        if (await r) if asyncio.iscoroutine(r) else r:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


async def test_real_nodes_download_copy_resume_verify_delete_and_cancel(tmp_path: Path) -> None:
    roots: dict[str, list[dict[str, Any]]] = {
        "alpha": [{"id": "hf", "path": str(tmp_path / "a" / "hf"), "kind": "hf-home", "writable": True, "label": "HF cache"},
                  {"id": "gguf", "path": str(tmp_path / "a" / "gguf"), "kind": "gguf", "writable": True},
                  {"id": "pl", "path": str(tmp_path / "a" / "pl"), "kind": "plain", "writable": True}],
        "beta": [{"id": "hub", "path": str(tmp_path / "b" / "hub"), "kind": "hf-hub", "writable": True},
                 {"id": "gguf", "path": str(tmp_path / "b" / "gguf"), "kind": "gguf", "writable": True},
                 {"id": "ro", "path": str(tmp_path / "b" / "ro"), "kind": "hf-home", "writable": False},
                 {"id": "pl", "path": str(tmp_path / "b" / "pl"), "kind": "plain", "writable": True}]}
    bus = EventBus()
    links: dict[str, NodeLink] = {}
    for host, rs in roots.items():
        for r in rs:
            Path(r["path"]).mkdir(parents=True)
        policy = tmp_path / host / "policy.json"
        policy.parent.mkdir(exist_ok=True)
        write_policy(policy, rs)
        links[host] = start_link(bus, host, host.title(), policy)
    fleet = Fleet(links)
    store = Store(":memory:")
    lib = LibraryService(store, fleet, make_adapters(), bus, in_use=lambda h, p: [], retry_delays=(0.2,))
    hub = FakeHub()
    dl = DownloadService(store, hub, HfService(hub, cache_ttl_s=0), SettingsService(store, make_settings(tmp_path)), bus,
                         fleet, lib, concurrency=2)  # type: ignore[arg-type]
    try:
        for lk in links.values():
            await lk.start()
        await wait_for(lambda: all(lk.link().state == "up" for lk in links.values()), "both nodes to come up")
        await lib.start()
        await lib.wait_idle()

        # a download through the real node, spanning several DATA frames
        payload = os.urandom(3 * 1024 * 1024 + 123)
        small = b'{"model_type": "x", "num_hidden_layers": 2}'
        hub.add("org/m", {"config.json": small, "model.safetensors": payload, "tok.txt": b"tok"})
        d = await dl.create(DownloadRequest(repo_id="org/m", host_id="alpha", root_id="hf"))
        await dl.wait(d.id)
        d = dl.get(d.id)
        assert d.state == "completed", (d.error_code, d.error)
        repo = tmp_path / "a" / "hf" / "hub" / "models--org--m"
        sha = hashlib.sha256(payload).hexdigest()
        assert (repo / "blobs" / sha).read_bytes() == payload
        assert os.readlink(repo / "snapshots" / COMMIT / "model.safetensors") == f"../../blobs/{sha}"
        assert (repo / "refs" / "main").read_text() == COMMIT
        ref = ModelRef(format="hf", repo_id="org/m")
        await wait_for(lambda: lib.find("alpha", ref) is not None, "the real scanner to list the download")
        copy = lib.find("alpha", ref)
        assert copy is not None and copy.rel == "hub/models--org--m" and copy.size_bytes == len(payload) + len(small) + 3
        info = await lib.model_info("alpha", ref)                            # the real models.config
        assert info is not None and info.num_layers == 2 and info.weight_bytes == copy.size_bytes

        # a copy to the other node: the large blob machine to machine (fs.read -> fs.write), the rest from the hub
        cp = await dl.copy(copy.id, CopyRequest(host_id="beta", root_id="hub"))
        await dl.wait(cp.id)
        cp = dl.get(cp.id)
        assert cp.state == "completed", (cp.error_code, cp.error)
        other = tmp_path / "b" / "hub" / "models--org--m"
        assert (other / "blobs" / sha).read_bytes() == payload and (other / "snapshots" / COMMIT / "tok.txt").read_bytes() == b"tok"
        assert not any(o[2] == "model.safetensors" for o in hub.opens[-2:])
        await wait_for(lambda: bool(lib.models(q="org/m")) and len(lib.models(q="org/m")[0].copies) == 2, "both machines list it")

        # resume from a real partial, then a corrupted download that the node refuses and a retry that works
        g = os.urandom(2 * 1024 * 1024 + 7)
        hub.add("org/g", {"m-Q4_K_M.gguf": g, "README.md": b"hi"})
        part = tmp_path / "a" / "gguf" / "org--g"
        part.mkdir()
        (part / "m-Q4_K_M.gguf.incomplete").write_bytes(g[:1_000_000])
        d = await dl.create(DownloadRequest(repo_id="org/g", host_id="alpha", root_id="gguf", allow_patterns=["m-Q4_K_M.gguf"]))
        await dl.wait(d.id)
        assert dl.get(d.id).state == "completed" and (part / "m-Q4_K_M.gguf").read_bytes() == g
        assert [o[3] for o in hub.opens if o[2] == "m-Q4_K_M.gguf"] == [1_000_000]
        hub.corrupt.add("m-Q4_K_M.gguf")
        d = await dl.create(DownloadRequest(repo_id="org/g", host_id="beta", root_id="gguf", allow_patterns=["m-Q4_K_M.gguf"]))
        await dl.wait(d.id)
        bad = dl.get(d.id)
        assert bad.state == "failed" and bad.error_code == "hash_mismatch" and (bad.error or "").startswith("Beta: ")
        assert not list((tmp_path / "b" / "gguf").rglob("*.incomplete"))
        hub.corrupt.clear()
        await dl.resume(d.id)
        await dl.wait(d.id)
        assert dl.get(d.id).state == "completed"

        # a plain model directory: no listing op exists, so the copy probes names and PROVES it found every file against the
        # byte total and file count the real scanner reported
        shard_a, shard_b = os.urandom(300_000), os.urandom(120_000)
        index = json.dumps({"weight_map": {"x": "model-00001-of-00002.safetensors", "y": "model-00002-of-00002.safetensors"}}).encode()
        plain = tmp_path / "a" / "pl" / "mymodel"
        plain.mkdir()
        for name, data in {"config.json": b'{"model_type": "x"}', "tokenizer.json": b"{}", "model.safetensors.index.json": index,
                           "model-00001-of-00002.safetensors": shard_a, "model-00002-of-00002.safetensors": shard_b}.items():
            (plain / name).write_bytes(data)
        await lib.rescan("alpha")
        await wait_for(lambda: lib.find("alpha", ModelRef(format="plain", root_id="pl", rel="mymodel")) is not None, "the plain directory")
        pcopy = lib.find("alpha", ModelRef(format="plain", root_id="pl", rel="mymodel"))
        assert pcopy is not None and pcopy.files == 5
        pc = await dl.copy(pcopy.id, CopyRequest(host_id="beta", root_id="pl"))
        await dl.wait(pc.id)
        pc = dl.get(pc.id)
        assert pc.state == "completed", (pc.error_code, pc.error)
        copied = tmp_path / "b" / "pl" / "mymodel"
        assert (copied / "model-00002-of-00002.safetensors").read_bytes() == shard_b and (copied / "config.json").is_file()
        (plain / "surprise.dat").write_bytes(b"?")                              # a file no probe would name: the proof refuses
        await lib.rescan("alpha")
        await wait_for(lambda: (lib.find("alpha", ModelRef(format="plain", root_id="pl", rel="mymodel")) or pcopy).files == 6, "rescan")
        pc2 = await dl.copy(pcopy.id, CopyRequest(host_id="beta", root_id="pl"))
        await dl.wait(pc2.id)
        refused = dl.get(pc2.id)
        assert refused.state == "failed" and refused.error_code == "copy_listing_incomplete"

        # a GGUF file the real scanner read: its header facts become a ModelInfo (layers, heads, kv heads, head size, quant)
        tiny_gguf(tmp_path / "a" / "gguf" / "tiny-Q4_K_M.gguf")
        await lib.rescan("alpha")
        gref = ModelRef(format="gguf", root_id="gguf", rel="tiny-Q4_K_M.gguf")
        await wait_for(lambda: lib.find("alpha", gref) is not None, "the real scanner to read the gguf")
        ginfo = await lib.model_info("alpha", gref)
        assert ginfo is not None and (ginfo.num_layers, ginfo.num_attention_heads, ginfo.num_kv_heads, ginfo.head_dim) == (2, 4, 2, 16)
        assert ginfo.hidden_size == 64 and ginfo.max_position_embeddings == 4096 and ginfo.quantization == "q4_k_m"
        assert ginfo.architectures == ["llama"] and ginfo.num_params == 2 * 64 * 16

        # the node's own refusals reach the caller with their codes
        with pytest.raises(Exception) as ei:                                 # noqa: PT011 - a ProblemError, code checked below
            await dl.create(DownloadRequest(repo_id="org/m", host_id="beta", root_id="ro"))
        assert getattr(ei.value, "code", None) == "root_not_writable"

        # delete one copy through the library: the real fs.delete
        await wait_for(lambda: any(c.host_id == "beta" for m in lib.models(q="org/m") for c in m.copies), "beta's copy to be listed")
        beta_copy = next(c for c in lib.models(q="org/m")[0].copies if c.host_id == "beta")
        await lib.delete_copy(beta_copy.id)
        assert not other.exists() and [c.host_id for c in lib.models(q="org/m")[0].copies] == ["alpha"]

        # a cancel in the middle of a file removes the partial from the real disk
        hub.pass_chunks = 2
        hub.gate = asyncio.Event()
        hub.add("org/big", {"b.safetensors": os.urandom(4 * 1024 * 1024)})
        d = await dl.create(DownloadRequest(repo_id="org/big", host_id="alpha", root_id="hf"))
        blobs = tmp_path / "a" / "hf" / "hub" / "models--org--big" / "blobs"
        await wait_for(lambda: blobs.exists() and any(p.name.endswith(".incomplete") and p.stat().st_size > 0 for p in blobs.iterdir()),
                       "a partial on the real disk")
        await dl.cancel(d.id)
        assert dl.get(d.id).state == "cancelled" and not list(blobs.glob("*.incomplete"))
    finally:
        hub.pass_chunks = None
        if hub.gate is not None:
            hub.gate.set()
        await lib.stop()
        await dl.shutdown()
        for lk in links.values():
            await lk.stop()
        store.close()
