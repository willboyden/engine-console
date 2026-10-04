"""Offline fakes for the data side: an in-process fleet whose nodes do real file work in a temp directory, and a fake
Hugging Face hub.

`FakeNode` implements `domain.fleet.NodeClient` closely enough that the library, downloads and copies run end to end
with no docker, no ssh and no network:

* every request's arguments go through `ecnode.proto.clean_args` and must come out unchanged (a misspelt or over-long
  argument fails the test instead of being dropped silently, as a real node would drop it), and `REQUIRED` is checked;
* every result and every pushed message is rebuilt with the protocol's own cleaners, so the hub sees what a real link
  would give it (missing fields are None);
* monitor mode refuses every op outside `proto.MONITOR_OPS`; a node whose link is down raises `NodeUnavailable`;
* `fs.*` follows docs/FLEET-PROTOCOL.md: path rules, writable roots only, a `.incomplete` partial that must be resumed
  at its own size (or restarted at 0), size and sha256 checked at `end` (a mismatch deletes the partial), relative
  symlinks inside the root, `allow_delete`, `in_use`.

Test hooks live on the node: `entries` (what `models.list` returns), `fail_after_bytes` (an `io_error` in the middle
of the next upload), `space_left` (a `no_space` once that many bytes were written), `in_use_paths`, `calls`, `writes`.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import shutil
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ecnode import proto

from engine_console.adapters.base import (
    Compat,
    EngineAdapter,
    Hardware,
    LaunchSpec,
    ModelInfo,
    ParamSpec,
)
from engine_console.domain.errors import GatedModel, NotFound
from engine_console.domain.fleet import (
    HostConfig,
    LinkStatus,
    NodeError,
    NodeKind,
    NodeUnavailable,
    StreamItem,
)
from engine_console.domain.ports import HubModel, RepoFile
from engine_console.services.common import EventBus
from engine_console.services.hf import gguf_quant

COMMIT = "c0ffee00000000000000000000000000000000ff"
LFS_SUFFIXES = (".safetensors", ".gguf", ".bin")


# ---- adapters ---------------------------------------------------------------------------------------------------------
class StubAdapter(EngineAdapter):
    """Just enough of an engine for the library to ask which formats it loads."""

    def __init__(self, id: str, formats: frozenset[str]) -> None:
        self.id = id
        self.display_name = id
        self.default_image = f"fake/{id}:1"
        self.model_formats = formats

    def param_catalog(self) -> list[ParamSpec]:
        return []

    def presets(self) -> dict[str, dict[str, Any]]:
        return {}

    def validate(self, params: dict[str, Any]) -> list[str]:
        return []

    def build_launch(self, model: str, params: dict[str, Any], hw: Hardware, *, served_name: str,
                     hf_cache_container_path: str) -> LaunchSpec:
        raise NotImplementedError

    def compatibility(self, model: ModelInfo, params: dict[str, Any], hw: Hardware) -> list[Compat]:
        return []

    def memory_model(self, model: ModelInfo, params: dict[str, Any]) -> dict[str, float]:
        return {}

    def parse_metrics(self, prometheus_text: str) -> dict[str, float]:
        return {}

    def parse_startup_log(self, line: str) -> dict[str, Any] | None:
        return None


# ---- the node ---------------------------------------------------------------------------------------------------------
@dataclass
class RootCfg:
    id: str
    kind: str
    writable: bool = True
    path: Path | None = None           # default: <node dir>/<id>, created
    label: str | None = None


def mount_dict(mount: str, *, fstype: str = "ext4", source: str | None = None, model: str | None = "Test SSD",
               size: int = 10**12, used: int = 10**11, free: int = 9 * 10**11, remote: bool = False,
               remote_host: str | None = None, remote_path: str | None = None, roots: list[str] | None = None,
               readonly: bool = False) -> dict[str, Any]:
    return {"mount": mount, "fstype": fstype, "source": source, "device": None if remote else "sda1", "model": model,
            "size": size, "used": used, "free": free, "readonly": readonly, "remote": remote,
            "remote_host": remote_host, "remote_path": remote_path, "roots": roots or []}


class FakeNode:
    """One machine's node. See the module docstring for what it checks."""

    def __init__(self, base: Path, host_id: str = "local", *, label: str | None = None,
                 roots: Sequence[RootCfg] = (), mode: str = "manage", allow_delete: bool = True,
                 addresses: Sequence[str] = (), hostname: str | None = None, mounts: list[dict[str, Any]] | None = None,
                 state: str = "up", scan_disk: bool = True, ssh: str | None = None, bus: EventBus | None = None) -> None:
        self.host = HostConfig(id=host_id, label=label or host_id.title(), ssh=ssh)
        self.base = base / host_id
        self.base.mkdir(parents=True, exist_ok=True)
        self.roots: dict[str, RootCfg] = {}
        for r in roots:
            if r.path is None:
                r.path = self.base / r.id
                r.path.mkdir(parents=True, exist_ok=True)
            self.roots[r.id] = r
        self.mode = mode
        self.allow_delete = allow_delete
        self.addresses = list(addresses)
        self.hostname = hostname or host_id
        self._mounts = mounts
        self.state = state
        self.scan_disk_enabled = scan_disk
        self.bus = bus
        self.forgotten = False
        self.free = 10**12
        # test hooks and records
        self.entries: list[dict[str, Any]] = []
        self.generation = 1
        self.scans: list[dict[str, Any]] = []
        self.configs: dict[tuple[str, str], dict[str, Any]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.writes: list[dict[str, Any]] = []
        self.reads: list[dict[str, Any]] = []
        self.fail_after_bytes: int | None = None
        self.space_left: int | None = None
        self.in_use_paths: set[str] = set()
        self.symlink_conflicts = False
        self.read_chunk = 1024
        self.list_generation_bump_once = False
        if scan_disk:
            self.scan()

    # ---- the NodeClient surface ----------------------------------------------------------------------------------------
    def link(self) -> LinkStatus:
        return LinkStatus(state=self.state)  # type: ignore[arg-type]

    def hello(self) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "type": "hello", "proto": proto.PROTO, "version": "0.4.0", "node_id": "0123456789abcdef",
            "host": {"hostname": self.hostname, "arch": "x86_64", "os": "Linux", "kernel": "7", "cpu_model": "cpu",
                     "cpu_cores": 8, "cpu_threads": 16, "ram_total_bytes": 64 * 2**30, "boot_time": 1.0, "python": "3.12"},
            "gpu": {"driver": "1", "cuda": "13", "source": "none"}, "gpus": [],
            "docker": {"available": False}, "addresses": [{"iface": "eth0", "addr": a, "prefix": 24} for a in self.addresses],
            "policy": {"path": "/etc/ecnode/policy.json", "mode": self.mode, "allow_delete": self.allow_delete,
                       "model_roots": [{"id": r.id, "path": str(r.path), "kind": r.kind, "writable": r.writable,
                                        "label": r.label or r.id} for r in self.roots.values()]},
            "forced_cmd": True, "via_ssh": False}
        out = proto.sanitize_push(raw)
        assert out is not None
        return out

    def disks(self) -> dict[str, Any]:
        mounts = self._mounts if self._mounts is not None else [
            mount_dict("/", free=self.free, roots=sorted(self.roots))]
        if self._mounts is not None and self.free != 10**12:
            mounts = [{**m, "free": self.free} if m["mount"] == "/" else m for m in mounts]
        out = proto.sanitize_push({"type": "disks", "t": 1.0, "mounts": mounts})
        assert out is not None
        return out

    def latest(self, kind: NodeKind) -> dict[str, Any] | None:
        if self.forgotten:
            return None
        if kind == "hello":
            return self.hello()
        if kind == "disks":
            return self.disks()
        return None

    def forget(self) -> None:
        """What the hub knows of a machine it has not heard from since it started: no hello, no disks."""
        self.forgotten = True

    def ops(self, name: str) -> list[dict[str, Any]]:
        return [a for op, a in self.calls if op == name]

    def announce_models_changed(self) -> None:
        assert self.bus is not None
        self.bus.publish("node.models_changed", {"host_id": self.host.id, "msg": {"type": "models_changed",
                                                                                   "generation": self.generation}})

    def set_entries(self, entries: list[dict[str, Any]]) -> None:
        self.entries = entries
        self.generation += 1

    # ---- requests ------------------------------------------------------------------------------------------------------
    def _err(self, code: str, message: str) -> NodeError:
        return NodeError(self.host.id, code, message)

    def _gate(self, op: str, args: dict[str, Any]) -> None:
        if op not in proto.OPS:
            raise self._err(proto.UNKNOWN_OP, f"{op!r} is not an operation of the node protocol")
        if self.state != "up":
            raise NodeUnavailable(self.host.id, f"{self.host.label} is not connected ({self.state})")
        cleaned = proto.clean_args(op, args)
        seen = {k: v for k, v in cleaned.items() if v is not None}      # clean() lists every key of the spec
        if seen != args:
            raise AssertionError(f"{op}: the node would drop or change arguments: sent {args!r}, node sees {seen!r}")
        missing = proto.missing_args(op, cleaned)
        if missing:
            raise self._err(proto.BAD_ARGS, f"{op}: missing {missing}")
        if self.mode != "manage" and op not in proto.MONITOR_OPS:
            raise self._err(proto.MONITOR_ONLY, f"{op} needs the node policy to be in manage mode")

    async def call(self, op: str, args: dict[str, Any] | None = None, *, timeout: float = 30.0) -> dict[str, Any]:
        a = dict(args or {})
        self.calls.append((op, a))
        self._gate(op, a)
        handler = getattr(self, "_op_" + op.replace(".", "_"), None)
        if handler is None:
            raise self._err(proto.UNSUPPORTED, f"the fake node does not implement {op}")
        await asyncio.sleep(0)
        result = handler(a)
        out: dict[str, Any] = proto.sanitize_res({"id": 1, "ok": True, "result": result}, op)["result"]
        return out

    def open(self, op: str, args: dict[str, Any] | None = None) -> Any:
        a = dict(args or {})
        self.calls.append((op, a))
        if op == "fs.write":
            return _WriteStream(self, a)
        if op == "fs.read":
            return _ReadStream(self, a)
        raise AssertionError(f"the fake node cannot stream {op}")

    # ---- paths ---------------------------------------------------------------------------------------------------------
    def _resolve(self, root_id: Any, rel: Any, *, write: bool = False) -> Path:
        root = self.roots.get(root_id)
        if root is None or root.path is None:
            raise self._err(proto.NOT_FOUND, f"no model root {root_id!r}")
        # Roots given as literal paths ("/data/hf") exist only to test library arithmetic. A test that does file work
        # must use a root under the node's own temp directory, never a path that could be a real model directory.
        assert self.base in (root.path, *root.path.parents), f"test would touch a real directory: {root.path}"
        if write and not root.writable:
            raise self._err(proto.POLICY_DENIED, f"root {root.id!r} is read-only")
        if not isinstance(rel, str) or not rel or rel.startswith("/") or "\\" in rel or "\x00" in rel or any(
                seg in ("", ".", "..") for seg in rel.split("/")):
            raise self._err(proto.BAD_ARGS, f"bad path {rel!r}")
        path = root.path / rel
        parent = Path(os.path.realpath(path.parent))
        base = Path(os.path.realpath(root.path))
        if parent != base and base not in parent.parents:
            raise self._err(proto.POLICY_DENIED, "the path leaves the root")
        return path

    # ---- operations ----------------------------------------------------------------------------------------------------
    def _op_fs_stat(self, a: dict[str, Any]) -> dict[str, Any]:
        items = []
        for rel in a["paths"]:
            p = self._resolve(a["root"], rel)
            part = p.with_name(p.name + ".incomplete")
            exists = p.exists()
            items.append({"rel": rel, "exists": exists, "is_dir": p.is_dir(),
                          "size": p.stat().st_size if exists and p.is_file() else 0,
                          "partial": part.stat().st_size if part.is_file() else 0})
        return {"items": items}

    def _op_fs_symlink(self, a: dict[str, Any]) -> dict[str, Any]:
        link = self._resolve(a["root"], a["link"], write=True)
        target = a["target"]
        if target.startswith("/"):
            raise self._err(proto.BAD_ARGS, "the target must be relative")
        resolved = Path(os.path.realpath(link.parent / target))
        base = Path(os.path.realpath(self.roots[a["root"]].path or "/"))
        if resolved != base and base not in resolved.parents:
            raise self._err(proto.POLICY_DENIED, "the link target leaves the root")
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            if self.symlink_conflicts:
                raise self._err(proto.CONFLICT, "the link exists")
            link.unlink()
        link.symlink_to(target)
        return {}

    def _op_fs_put_small(self, a: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(a["root"], a["rel"], write=True)
        data = base64.b64decode(a["content_b64"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return {}

    def _op_fs_delete(self, a: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(a["root"], a["rel"], write=True)
        if not self.allow_delete:
            raise self._err(proto.POLICY_DENIED, "this node's policy does not allow deletes")
        target = str(path)
        if any(u == target or u.startswith(target + "/") or target.startswith(u + "/") for u in self.in_use_paths):
            raise self._err(proto.IN_USE, "a running container mounts that path")
        if not os.path.lexists(path):
            raise self._err(proto.NOT_FOUND, "no such path")
        freed = 0
        if path.is_dir() and not path.is_symlink():
            freed = sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())
            shutil.rmtree(path)
        else:
            freed = path.stat().st_size if path.is_file() and not path.is_symlink() else 0
            path.unlink()
        rel = a["rel"]
        kept = [e for e in self.entries if not (e.get("root") == a["root"] and (e.get("rel") == rel or str(e.get("rel")).startswith(rel + "/")))]
        if len(kept) != len(self.entries):          # what the node's next scan will report
            self.entries = kept
            self.generation += 1
        return {"freed": freed}

    def _op_models_scan(self, a: dict[str, Any]) -> dict[str, Any]:
        self.scans.append(a)
        if self.scan_disk_enabled:
            self.scan()
        return {"generation": self.generation}

    def _op_models_list(self, a: dict[str, Any]) -> dict[str, Any]:
        limit = min(max(a.get("limit") or 200, 1), 200)
        cursor = a.get("cursor") or 0
        generation = self.generation
        if self.list_generation_bump_once and cursor:
            self.list_generation_bump_once = False
            self.generation += 1        # a scan finished between two pages
            generation = self.generation
        page = self.entries[cursor:cursor + limit]
        out: dict[str, Any] = {
            "generation": generation, "entries": page,
            "roots": [{"root": r.id, "scanned_at": 1.0, "complete": True,
                       "entries": sum(1 for e in self.entries if e.get("root") == r.id)} for r in self.roots.values()]}
        if cursor + limit < len(self.entries):
            out["next_cursor"] = cursor + limit
        return out

    def _op_models_config(self, a: dict[str, Any]) -> dict[str, Any]:
        cfg = self.configs.get((a["root"], a["rel"]))
        if cfg is None:
            raise self._err(proto.NOT_FOUND, "no config.json")
        return {"config": cfg}

    # ---- a tiny scanner (hf-home, hf-hub and gguf roots) ----------------------------------------------------------------
    def scan(self) -> None:
        entries: list[dict[str, Any]] = []
        for r in self.roots.values():
            assert r.path is not None and self.base in (r.path, *r.path.parents), f"test would scan a real directory: {r.path}"
            if r.kind in ("hf-home", "hf-hub"):
                hub = r.path / "hub" if r.kind == "hf-home" else r.path
                prefix = "hub/" if r.kind == "hf-home" else ""
                for d in sorted(hub.glob("models--*")) if hub.is_dir() else []:
                    ref = d / "refs" / "main"
                    if not ref.is_file() or not (d / "snapshots" / ref.read_text().strip()).is_dir():
                        continue
                    org, _, name = d.name[len("models--"):].partition("--")
                    blobs = [f for f in (d / "blobs").glob("*") if f.is_file() and not f.name.endswith(".incomplete")]
                    entries.append({"root": r.id, "rel": prefix + d.name, "format": "hf", "name": f"{org}/{name}",
                                    "revision": ref.read_text().strip(), "size": sum(f.stat().st_size for f in blobs),
                                    "files": len(blobs), "mtime": 1.0, "atime": 2.0})
            elif r.kind == "gguf":
                for f in sorted(r.path.rglob("*.gguf")):
                    rel = f.relative_to(r.path).as_posix()
                    entries.append({"root": r.id, "rel": rel, "format": "gguf", "name": f.stem, "quant": gguf_quant(rel),
                                    "size": f.stat().st_size, "files": 1, "mtime": 1.0, "atime": 2.0})
        if entries != self.entries:
            self.entries = entries
            self.generation += 1


class _WriteStream:
    """An upload into `fs.write`: the node side of docs/FLEET-PROTOCOL.md for one file."""

    def __init__(self, node: FakeNode, args: dict[str, Any]) -> None:
        self._node = node
        self._error: NodeError | None = None
        self._result: dict[str, Any] | None = None
        self._ended = False
        self._sent = 0
        self.cancelled = False
        try:
            node._gate("fs.write", args)
            self._final = node._resolve(args["root"], args["rel"], write=True)
            self._part = self._final.with_name(self._final.name + ".incomplete")
            self._size: int = args["size"]
            self._sha: str | None = args.get("sha256")
            offset = args.get("offset") or 0
            if offset == 0:
                self._final.parent.mkdir(parents=True, exist_ok=True)
                self._part.write_bytes(b"")
            else:
                have = self._part.stat().st_size if self._part.is_file() else -1
                if have != offset:
                    raise node._err(proto.CONFLICT, f"offset {offset} is not the size of the partial ({have})")
            node.writes.append({"rel": args["rel"], "offset": offset, "size": self._size, "sha256": self._sha})
        except NodeError as e:
            self._error = e

    def __aiter__(self) -> AsyncIterator[StreamItem]:
        return self._empty()

    async def _empty(self) -> AsyncIterator[StreamItem]:
        if self._error:
            raise self._error
        return
        yield  # pragma: no cover

    async def write(self, data: bytes) -> None:
        if self._error:
            raise self._error
        node = self._node
        if len(data) > proto.MAX_DATA:
            raise AssertionError("a DATA frame over MAX_DATA")
        if node.space_left is not None and len(data) > node.space_left:
            self._error = node._err(proto.NO_SPACE, "the filesystem would drop below 1 GiB free")
            raise self._error
        if node.fail_after_bytes is not None and self._sent + len(data) > node.fail_after_bytes:
            keep = max(0, node.fail_after_bytes - self._sent)
            with self._part.open("ab") as fh:
                fh.write(data[:keep])
            node.fail_after_bytes = None
            self._error = node._err(proto.IO_ERROR, "read error on the source disk")
            raise self._error
        with self._part.open("ab") as fh:
            fh.write(data)
        self._sent += len(data)
        if node.space_left is not None:
            node.space_left -= len(data)
        await asyncio.sleep(0)

    async def end(self) -> None:
        if self._error:
            raise self._error
        self._ended = True
        data = self._part.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != self._size:
            self._part.unlink()
            self._error = self._node._err(proto.SIZE_MISMATCH, f"got {len(data)} bytes, expected {self._size}")
        elif self._sha and digest != self._sha:
            self._part.unlink()
            self._error = self._node._err(proto.HASH_MISMATCH, "the sha256 does not match")
        else:
            os.replace(self._part, self._final)
            self._result = {"size": len(data), "sha256": digest}

    async def result(self) -> dict[str, Any]:
        if self._error:
            raise self._error
        assert self._ended, "result() before end()"
        return self._result or {}

    async def cancel(self) -> None:
        self.cancelled = True


class _ReadStream:
    def __init__(self, node: FakeNode, args: dict[str, Any]) -> None:
        self._node = node
        self._error: NodeError | None = None
        self.cancelled = False
        self._data = b""
        try:
            node._gate("fs.read", args)
            path = node._resolve(args["root"], args["rel"])
            if not path.is_file() or path.is_symlink():
                raise node._err(proto.NOT_FOUND, "no such file")
            self._data = path.read_bytes()[args.get("offset") or 0:]
            node.reads.append({"rel": args["rel"], "offset": args.get("offset") or 0})
        except NodeError as e:
            self._error = e

    def __aiter__(self) -> AsyncIterator[StreamItem]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[StreamItem]:
        if self._error:
            raise self._error
        step = self._node.read_chunk
        for i in range(0, len(self._data), step):
            yield StreamItem("data", data=self._data[i:i + step])
            await asyncio.sleep(0)

    async def write(self, data: bytes) -> None:
        raise AssertionError("fs.read takes no upload")

    async def end(self) -> None:
        raise AssertionError("fs.read takes no upload")

    async def result(self) -> dict[str, Any]:
        if self._error:
            raise self._error
        return {"size": len(self._data)}

    async def cancel(self) -> None:
        self.cancelled = True


class FakeFleet:
    """`FleetPort` over fake nodes. `set_state` also publishes the `fleet` bus topic like the real link does."""

    def __init__(self, bus: EventBus, nodes: Sequence[FakeNode]) -> None:
        self.bus = bus
        self.nodes = {n.host.id: n for n in nodes}
        for n in nodes:
            n.bus = bus

    def hosts(self) -> list[HostConfig]:
        return [n.host for n in self.nodes.values()]

    def node(self, host_id: str) -> FakeNode:
        n = self.nodes.get(host_id)
        if n is None:
            raise NotFound(f"no host {host_id[:40]!r} is configured", code="unknown_host")
        return n

    def set_state(self, host_id: str, state: str) -> None:
        self.node(host_id).state = state
        self.bus.publish("fleet", {"id": host_id, "link": {"state": state}})


# ---- the Hugging Face hub -----------------------------------------------------------------------------------------------
@dataclass
class FakeStream:
    data: bytes
    offset: int = 0
    total: int | None = None
    chunk: int = 16
    hub: FakeHub | None = None
    closed: bool = field(default=False)

    async def chunks(self) -> AsyncIterator[bytes]:
        for i in range(0, len(self.data), self.chunk):
            if self.hub is not None:
                await self.hub.turnstile()
            yield self.data[i:i + self.chunk]
            await asyncio.sleep(0)

    async def aclose(self) -> None:
        self.closed = True


class FakeHub:
    """In-memory hub: repo_id -> (HubModel, {path: bytes}). Large-looking files (weights) carry their sha256 like LFS."""

    def __init__(self) -> None:
        self.repos: dict[str, tuple[HubModel, dict[str, bytes]]] = {}
        self.gated: set[str] = set()
        self.corrupt: set[str] = set()
        self.ignore_range = False
        self.opens: list[tuple[str, str, str, int]] = []
        self.gate: asyncio.Event | None = None       # while it exists and is not set, every chunk waits for it ...
        self.pass_chunks: int | None = None          # ... after this many chunks have gone through (None: none are let through free)
        self.listings = 0

    async def turnstile(self) -> None:
        """Called before every chunk of every stream: how a test stops a transfer in the middle of a file."""
        if self.pass_chunks is not None and self.pass_chunks > 0:
            self.pass_chunks -= 1
            return
        if self.gate is not None:
            await self.gate.wait()

    def add(self, repo_id: str, files: dict[str, bytes], *, sha: str = COMMIT, config: dict[str, Any] | None = None) -> None:
        rf = [RepoFile(path=p, size=len(b), sha256=hashlib.sha256(b).hexdigest() if p.endswith(LFS_SUFFIXES) else None)
              for p, b in files.items()]
        hm = HubModel(repo_id=repo_id, sha=sha, gated=False, license=None, pipeline_tag="text-generation",
                      library_name=None, tags=[], files=rf, config=config)
        self.repos[repo_id] = (hm, files)

    async def search(self, *, q: str, task: str | None, library: str | None, quant: str | None, sort: str, limit: int,
                     token_ok: bool = True) -> list[HubModel]:
        return [m for m, _ in self.repos.values() if q.lower() in m.repo_id.lower()][:limit]

    async def model(self, repo_id: str, revision: str | None) -> HubModel:
        self.listings += 1
        if repo_id in self.gated:
            raise GatedModel(f"gated: {repo_id}")
        if repo_id not in self.repos:
            raise NotFound(f"not found on Hugging Face: {repo_id}")
        return self.repos[repo_id][0]

    async def card(self, repo_id: str, revision: str | None) -> str:
        return f"# {repo_id}"

    async def open_file(self, repo_id: str, revision: str, path: str, start: int) -> FakeStream:
        self.opens.append((repo_id, revision, path, start))
        data = self.repos[repo_id][1][path]
        if path in self.corrupt:
            data = b"X" * len(data)
        if self.ignore_range and start:
            return FakeStream(data, 0, len(data), hub=self)
        return FakeStream(data[start:], start, len(data), hub=self)


# ---- real HF cache directories, made without the code under test ---------------------------------------------------------
def seed_hf_cache(root: Path, repo_id: str, files: dict[str, bytes], *, layout: str = "hf-home", commit: str = COMMIT,
                  refs: str = "main") -> Path:
    """Write a Hugging Face cache directory the way huggingface_hub does: LFS blobs named by sha256, small files by
    their git object id (so a copy cannot guess their names), relative snapshot links, refs/<name>."""
    hub = root / "hub" if layout == "hf-home" else root
    repo = hub / ("models--" + repo_id.replace("/", "--"))
    for path, data in files.items():
        blob = hashlib.sha256(data).hexdigest() if path.endswith(LFS_SUFFIXES) else hashlib.sha1(  # noqa: S324
            b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()
        (repo / "blobs").mkdir(parents=True, exist_ok=True)
        (repo / "blobs" / blob).write_bytes(data)
        link = repo / "snapshots" / commit / path
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(repo / "blobs" / blob, link.parent))
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / refs).write_text(commit)
    return repo


# ---- entries as a node's scanner reports them -----------------------------------------------------------------------------
def hf_entry(root: str, repo: str, *, size: int = 1000, revision: str = COMMIT, home: bool = True, **kw: Any) -> dict[str, Any]:
    folder = "models--" + repo.replace("/", "--")
    return {"root": root, "rel": ("hub/" if home else "") + folder, "format": "hf", "name": repo, "revision": revision,
            "size": size, "files": 3, "mtime": 10.0, "atime": 20.0, **kw}


def gguf_entry(root: str, rel: str, *, quant: str = "Q4_K_M", size: int = 4000, name: str | None = None,
               **kw: Any) -> dict[str, Any]:
    stem = rel.rsplit("/", 1)[-1][:-5]
    return {"root": root, "rel": rel, "format": "gguf", "name": name or stem, "quant": quant, "size": size, "files": 1,
            "mtime": 10.0, "atime": 20.0, **kw}


def entry(root: str, rel: str, fmt: str, name: str | None = None, *, size: int = 100, **kw: Any) -> dict[str, Any]:
    return {"root": root, "rel": rel, "format": fmt, "name": name, "size": size, "files": 1, "mtime": 10.0, **kw}
