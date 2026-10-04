"""Small helpers shared by the data tests: the NFS two-machine setup, the sample repository, and the knobs that stop a
transfer in the middle of a file."""
from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from pathlib import Path

from .conftest import World
from .fakes import FakeNode, RootCfg, mount_dict

PAYLOAD = b"0123456789" * 5
FILES = {"config.json": b"{}", "w.safetensors": PAYLOAD, "notes.txt": b"n"}
SHA = hashlib.sha256(PAYLOAD).hexdigest()
TOTAL = 2 + 50 + 1


def nfs_alpha(tmp: Path, **kw: object) -> FakeNode:
    """The machine that owns the disk: 10.0.0.1, an HF cache at /data/hf and a GGUF dir at /data/gguf."""
    kw.setdefault("addresses", ["10.0.0.1", "127.0.0.1"])
    kw.setdefault("hostname", "alpha.lan")
    kw.setdefault("mounts", [mount_dict("/", roots=["hf", "gguf"])])
    return FakeNode(tmp, "alpha", roots=[RootCfg("hf", "hf-home", path=Path("/data/hf")),
                                         RootCfg("gguf", "gguf", path=Path("/data/gguf"))], scan_disk=False, **kw)  # type: ignore[arg-type]


def nfs_mount(remote_host: str = "10.0.0.1", remote_path: str = "/data", mount: str = "/mnt/alpha",
              roots: list[str] | None = None) -> dict[str, object]:
    return mount_dict(mount, fstype="nfs4", source=f"{remote_host}:{remote_path}", model=None, remote=True,
                      remote_host=remote_host, remote_path=remote_path, roots=roots or [])


def nfs_beta(tmp: Path, mount: dict[str, object] | None = None, **kw: object) -> FakeNode:
    """The machine that mounts it: /mnt/alpha, read-only roots `shared` (HF cache) and `shared_g` (GGUF)."""
    mount = mount or nfs_mount(roots=["shared", "shared_g"])
    return FakeNode(tmp, "beta", roots=[RootCfg("shared", "hf-home", writable=False, path=Path("/mnt/alpha/hf")),
                                        RootCfg("shared_g", "gguf", writable=False, path=Path("/mnt/alpha/gguf")),
                                        RootCfg("own", "hf-hub", path=Path("/srv/hub"))],
                    addresses=["10.0.0.2"], hostname="beta", mounts=[mount_dict("/", roots=["own"]), mount], scan_disk=False,
                    **kw)  # type: ignore[arg-type]


async def until(cond: Callable[[], bool], what: str = "condition", timeout: float = 3.0) -> None:
    for _ in range(int(timeout / 0.002)):
        if cond():
            return
        await asyncio.sleep(0.002)
    raise AssertionError(f"timed out waiting for {what}")


def stop_midway(w: World, chunks: int = 3) -> asyncio.Event:
    """Let `chunks` chunks through (config.json is one, then two 16-byte chunks of the weights), then hold the rest."""
    w.hub.pass_chunks = chunks
    w.hub.gate = asyncio.Event()
    return w.hub.gate


def release(w: World) -> None:
    w.hub.pass_chunks = None
    assert w.hub.gate is not None
    w.hub.gate.set()


def partial_of(node: FakeNode, root: str, rel: str) -> Path:
    return node.roots[root].path / (rel + ".incomplete")      # type: ignore[operator]
