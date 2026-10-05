"""The cross-machine model library (contract v2, docs/ARCHITECTURE.md 10.5).

Every host's node scans its own model roots; the hub pages through each node's `models.list` and merges the answers:

* One model on one host's root is a **copy** (`LibraryCopy`); copies of the same model, on any machine, group under
  one `LibraryModel` key.
* **Shared storage** is folded in: when host B's root sits on an NFS mount of an export of host A's disk, B's copy and
  A's are the same bytes, so the model lists A's copy once and says B can load it (`visible_on`).
* The last good snapshot of every host lives in SQLite, so a machine that is down still shows its last-known models;
  they are marked stale in `warnings` and cannot be deleted from here.
* `in_use_by` and `deletable` are not cached: they are computed on every read from the `in_use` callback and from the
  node's current policy, because an instance can start or stop at any moment without anyone telling the library.
* `GET /library` shows each copy once, on the machine that owns the disk. `find(host, ref)` and `visible_copies(host)`
  show the same copy in `host`'s own terms (its root id, its path, the NFS mount it goes through), because that is what
  an engine started on `host` has to mount; the copy's `id` is unchanged, so deleting or copying it still acts on the
  one real set of files.

The hub treats every node as untrusted. Entries have already been through `ecnode.proto.sanitize_res` and are cleaned
again when read back from SQLite; node strings are used as keys and display text only, never to build a hub-side path,
a SQL statement or HTML.

Which configured host owns an NFS server address, and how a mount is named, follow `fleet_status` (the host detail
page), so the library and `GET /hosts/{id}` never disagree about who shares what.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import json
import logging
import posixpath
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from ecnode import proto

from engine_console.adapters import AdapterRegistry
from engine_console.adapters.base import ModelInfo
from engine_console.domain.errors import Conflict, NotFound
from engine_console.domain.fleet import (
    FleetPort,
    LibraryCopy,
    LibraryFormat,
    LibraryModel,
    MountInfo,
    NodeClient,
    NodeError,
    RootInfo,
)
from engine_console.domain.models import LocalModel
from engine_console.domain.ports import HubModel
from engine_console.domain.refs import ModelRef
from engine_console.services.common import EventBus, now
from engine_console.services.gguf_info import model_info_from_entry, normalize_quant
from engine_console.services.hf import build_model_info, gguf_quant, gguf_stem
from engine_console.services.nodeerr import problem_from_node
from engine_console.services.store import Store

log = logging.getLogger(__name__)

FORMATS: tuple[str, ...] = ("hf", "gguf", "ollama", "plain", "other")
LIBRARY_TOPIC = "library"           # EventBus: {"host_id", "generation", "models"} after a host's snapshot changed
MAX_ENTRIES = 20_000                # per host: the node scans at most 50 000 directory entries per root
MAX_WARNINGS = 10
_COPY_ID = re.compile(r"[0-9a-f]{16}")
_HF_DIR = re.compile(r"(?:^|/)models--([^/]+?)--([^/]+?)(?:/|$)")
# Root kinds that naturally hold each format: when one machine reaches a copy through two of its roots, the natural one is
# the root an engine should mount.
_NATURAL_KINDS: dict[str, tuple[str, ...]] = {"hf": ("hf-home", "hf-hub"), "gguf": ("gguf",), "plain": ("plain",),
                                              "ollama": ("ollama",)}


# ---- small helpers on untrusted values ------------------------------------------------------------------------------
def _s(v: object, cap: int = 1024) -> str | None:
    return v[:cap] if isinstance(v, str) and v else None


def _i(v: object) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _f(v: object) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _norm(path: str) -> str:
    n = posixpath.normpath(path) if path else ""
    return "/" + n.lstrip("/") if n.startswith("//") else n    # POSIX keeps a leading "//" as it is; both mean "/"


def _under(path: str, base: str) -> bool:
    """`path` is `base` or inside it, by whole path components ("/data/hf2" is not under "/data/hf")."""
    return base == "/" or path == base or path.startswith(base + "/")


def _overlap(a: str, b: str) -> bool:
    return _under(a, b) or _under(b, a)


def _rel_ok(rel: str) -> bool:
    """The node's own path rules for a relative path (docs/FLEET-PROTOCOL.md, "Path rules")."""
    if not rel or rel.startswith("/") or "\\" in rel or "\x00" in rel:
        return False
    return all(seg not in ("", ".", "..") for seg in rel.split("/"))


def copy_id(host_id: str, root_id: str, rel: str) -> str:
    """Stable id of one model on one host's root (docs/ARCHITECTURE.md 10.5)."""
    return hashlib.sha1(f"{host_id}|{root_id}|{rel}".encode(), usedforsecurity=False).hexdigest()[:16]


def _canon_addr(text: str) -> str:
    t = text.strip().strip("[]").split("%", 1)[0].lower()
    try:
        return str(ipaddress.ip_address(t))
    except ValueError:
        return t


def _is_loopback(addr: str) -> bool:
    try:
        return ipaddress.ip_address(addr).is_loopback
    except ValueError:
        return addr in ("localhost", "localhost.localdomain")


# ---- what the hub knows about one host's storage ---------------------------------------------------------------------
@dataclass(frozen=True)
class RootSpec:
    id: str
    path: str
    kind: str
    writable: bool
    label: str | None


@dataclass(frozen=True)
class Mount:
    mount: str
    fstype: str | None = None
    source: str | None = None
    device: str | None = None
    model: str | None = None
    size: int | None = None
    used: int | None = None
    free: int | None = None
    readonly: bool = False
    remote: bool = False
    remote_host: str | None = None
    remote_path: str | None = None
    roots: tuple[str, ...] = ()

    @property
    def nfs(self) -> bool:
        return bool(self.fstype and self.fstype.lower().startswith("nfs") and self.remote_host and self.remote_path)

    @staticmethod
    def parse(d: object) -> Mount | None:
        if not isinstance(d, dict):
            return None
        mount = _s(d.get("mount"), 512)
        if mount is None or not mount.startswith("/"):
            return None
        roots = d.get("roots")
        remote_path = _s(d.get("remote_path"), 512)
        return Mount(
            mount=_norm(mount), fstype=_s(d.get("fstype"), 32), source=_s(d.get("source"), 512),
            device=_s(d.get("device"), 64), model=_s(d.get("model"), 128), size=_i(d.get("size")),
            used=_i(d.get("used")), free=_i(d.get("free")), readonly=d.get("readonly") is True,
            remote=d.get("remote") is True, remote_host=_s(d.get("remote_host"), 253),
            remote_path=_norm(remote_path) if remote_path else None,
            roots=tuple(r for r in roots if isinstance(r, str))[:32] if isinstance(roots, list) else ())


def mount_label(m: Mount, owner_label: str | None = None) -> str:
    """The name a person recognises a drive by: "tank (zfs)", "NFS from gpu-box-2", "/ (ext4, Samsung 990)"."""
    ft = (m.fstype or "").lower()
    if ft == "zfs":
        pool = (m.source or "").split("/", 1)[0].split("@", 1)[0]
        return f"{pool or m.mount} (zfs)"
    if ft.startswith("nfs") or m.remote:
        kind = "NFS" if ft.startswith("nfs") else ("CIFS" if ft in ("cifs", "smb3", "smbfs") else (ft.upper() or "Network"))
        return f"{kind} from {owner_label or m.remote_host or m.source or 'another machine'}"
    if m.model:
        return f"{m.mount} ({m.fstype}, {m.model})" if m.fstype else f"{m.mount} ({m.model})"
    return f"{m.mount} ({m.fstype})" if m.fstype else m.mount


def mount_for(mounts: Sequence[Mount], path: str) -> Mount | None:
    """The mount a path lives on: the one with the longest mount point that contains it."""
    best: Mount | None = None
    for m in mounts:
        if _under(path, m.mount) and (best is None or len(m.mount) > len(best.mount)):
            best = m
    return best


def _mount_of_root(spec: RootSpec, mounts: Sequence[Mount]) -> Mount | None:
    """The mount a model root lives on: the one the node says holds it, else the longest mount path above it."""
    claimed = [m for m in mounts if spec.id in m.roots]
    return claimed[0] if len(claimed) == 1 else mount_for(mounts, spec.path)


@dataclass
class HostFacts:
    """What the hub needs from a node's `hello` and `disks` to build that host's copies. Kept in SQLite beside the
    entries, so a host that is down can still be shown."""
    roots: list[RootSpec] = field(default_factory=list)
    mode: str = "monitor"
    allow_delete: bool = False
    addresses: list[str] = field(default_factory=list)
    hostname: str | None = None
    mounts: list[Mount] = field(default_factory=list)

    def root(self, root_id: str) -> RootSpec | None:
        return next((r for r in self.roots if r.id == root_id), None)

    def signature(self) -> tuple[Any, ...]:
        """What a copy's place in the library depends on: change it and the merged view must be rebuilt. Free space
        is not part of it."""
        mounts = sorted(((m.mount, m.fstype, m.remote_host, m.remote_path, m.roots) for m in self.mounts), key=repr)
        return (tuple((r.id, r.path, r.kind) for r in self.roots), tuple(sorted(self.addresses)), self.hostname,
                tuple(mounts))

    def to_json(self) -> dict[str, Any]:
        return {"mode": self.mode, "allow_delete": self.allow_delete, "addresses": self.addresses,
                "hostname": self.hostname, "mounts": [_mount_json(m) for m in self.mounts]}

    def roots_json(self) -> list[dict[str, Any]]:
        return [{"id": r.id, "path": r.path, "kind": r.kind, "writable": r.writable, "label": r.label}
                for r in self.roots]

    @staticmethod
    def from_json(roots: object, facts: object) -> HostFacts:
        out = HostFacts(roots=_parse_roots(roots))
        if isinstance(facts, dict):
            out.mode = "manage" if facts.get("mode") == "manage" else "monitor"
            out.allow_delete = facts.get("allow_delete") is True
            addrs = facts.get("addresses")
            out.addresses = [a for a in addrs if isinstance(a, str)][:64] if isinstance(addrs, list) else []
            out.hostname = _s(facts.get("hostname"), 253)
            mounts = facts.get("mounts")
            out.mounts = [m for m in (Mount.parse(x) for x in (mounts if isinstance(mounts, list) else [])[:64])
                          if m is not None]
        return out


def _mount_json(m: Mount) -> dict[str, Any]:
    return {"mount": m.mount, "fstype": m.fstype, "source": m.source, "device": m.device, "model": m.model,
            "size": m.size, "used": m.used, "free": m.free, "readonly": m.readonly, "remote": m.remote,
            "remote_host": m.remote_host, "remote_path": m.remote_path, "roots": list(m.roots)}


def _parse_roots(raw: object) -> list[RootSpec]:
    roots: list[RootSpec] = []
    for r in (raw if isinstance(raw, list) else [])[:32]:
        if not isinstance(r, dict):
            continue
        rid, path = _s(r.get("id"), 32), _s(r.get("path"), 512)
        if rid is None or path is None or not path.startswith("/") or any(x.id == rid for x in roots):
            continue
        roots.append(RootSpec(id=rid, path=_norm(path), kind=_s(r.get("kind"), 16) or "auto",
                              writable=r.get("writable") is True, label=_s(r.get("label"), 64)))
    return roots


def parse_facts(hello: dict[str, Any] | None, disks: dict[str, Any] | None,
                fallback: HostFacts | None) -> HostFacts | None:
    """Facts from a node's latest `hello` and `disks`; the stored ones when the node has not spoken since the hub
    started. None when there is neither."""
    if hello is None:
        return fallback
    pol = hello.get("policy")
    pol = pol if isinstance(pol, dict) else {}
    host = hello.get("host")
    addrs = [a.get("addr") for a in (hello.get("addresses") or []) if isinstance(a, dict)]
    mounts_raw = disks.get("mounts") if disks is not None else None
    if isinstance(mounts_raw, list):
        mounts = [m for m in (Mount.parse(x) for x in mounts_raw[:64]) if m is not None]
    else:
        mounts = list(fallback.mounts) if fallback else []
    return HostFacts(
        roots=_parse_roots(pol.get("model_roots")), mode="manage" if pol.get("mode") == "manage" else "monitor",
        allow_delete=pol.get("allow_delete") is True,
        addresses=[a for a in addrs if isinstance(a, str) and a][:64],
        hostname=_s(host.get("hostname"), 253) if isinstance(host, dict) else None, mounts=mounts)


@dataclass
class HostSnapshot:
    """One host's last good `models.list`, plus the facts that were true when it was taken."""
    generation: int | None
    entries: list[dict[str, Any]]
    scans: list[dict[str, Any]]
    facts: HostFacts
    updated_at: float


@dataclass(frozen=True)
class RootUsage:
    """What the last scan found in one root: enough to tell whether an `auto` root holds an HF cache."""
    entries: int = 0
    hf_home: bool = False      # an hf entry under hub/
    hf_hub: bool = False       # an hf entry directly in the root (models--org--name)
    loose: bool = False        # gguf files or plain model directories
    ollama: bool = False       # an Ollama store
    other: bool = False        # anything else


@dataclass
class CopySource:
    """What a copy of a library copy needs to know about its source (the entry as the node scanned it)."""
    copy: LibraryCopy
    format: str
    name: str
    revision: str | None
    quant: str | None
    shards: int | None
    locations: list[tuple[str, str]]     # every (host id, root id) that reaches these bytes: the owner and NFS aliases
    warnings: list[str]                  # the node's own warnings for this entry (e.g. "size is a lower bound")


# ---- who owns an address ---------------------------------------------------------------------------------------------
@dataclass
class _HostView:
    id: str
    label: str
    state: str
    facts: HostFacts


class _AddressBook:
    """Which configured host owns an address or a name. An address that two hosts both report (a docker bridge such
    as 172.17.0.1 exists on every machine with docker) identifies nobody, so it is left out."""

    def __init__(self, views: Mapping[str, _HostView]) -> None:
        by_addr: dict[str, set[str]] = {}
        by_name: dict[str, set[str]] = {}
        for hid, v in views.items():
            for a in v.facts.addresses:
                n = _canon_addr(a)
                if n and not _is_loopback(n):
                    by_addr.setdefault(n, set()).add(hid)
            name = (v.facts.hostname or "").lower()
            if name and not _is_loopback(name):
                by_name.setdefault(name, set()).add(hid)
                by_name.setdefault(name.split(".", 1)[0], set()).add(hid)
        self._addr = {k: next(iter(v)) for k, v in by_addr.items() if len(v) == 1}
        self._name = {k: next(iter(v)) for k, v in by_name.items() if len(v) == 1}

    def find(self, server: str | None, exclude: str) -> str | None:
        """The configured host an NFS server address or name belongs to (never `exclude`: a machine mounting its own
        export is not borrowing someone else's disk)."""
        if not server:
            return None
        key = _canon_addr(server)
        if not key or _is_loopback(key):
            return None
        hid = self._addr.get(key) or self._name.get(key)
        return hid if hid is not None and hid != exclude else None


@dataclass(frozen=True)
class _RootLoc:
    host_id: str
    root_id: str
    owner: str        # the host whose disk this really is
    path: str         # the root's path expressed on the owner (a root that is an NFS mount is translated)
    imported: bool


def _root_locations(views: Mapping[str, _HostView], book: _AddressBook) -> list[_RootLoc]:
    locs: list[_RootLoc] = []
    for hid, v in views.items():
        for spec in v.facts.roots:
            m = _mount_of_root(spec, v.facts.mounts)
            owner, path, imported = hid, spec.path, False
            if m is not None and m.nfs and m.remote_path and _under(spec.path, m.mount):
                server = book.find(m.remote_host, hid)
                if server is not None:
                    rel = posixpath.relpath(spec.path, m.mount)
                    owner, imported = server, True
                    path = m.remote_path if rel == "." else posixpath.normpath(posixpath.join(m.remote_path, rel))
            locs.append(_RootLoc(hid, spec.id, owner, path, imported))
    return locs


def _shares(views: Mapping[str, _HostView], book: _AddressBook, order: Mapping[str, int]
            ) -> dict[tuple[str, str], tuple[str | None, list[str]]]:
    """(host_id, root_id) -> (shared_from, visible_on). Two roots are the same storage when they resolve to the same
    owner host and their paths on it overlap. `visible_on` lists this host first, then the owner, then the rest in
    configuration order; `shared_from` is set only for a root that is another configured host's disk AND overlaps one
    of that host's own roots (otherwise that host's library would not list what this one sees)."""
    locs = _root_locations(views, book)
    out: dict[tuple[str, str], tuple[str | None, list[str]]] = {}
    for loc in locs:
        peers = [o for o in locs if o.owner == loc.owner and _overlap(o.path, loc.path)]
        seen = {o.host_id for o in peers} | {loc.host_id}
        visible = [loc.host_id]
        if loc.owner in seen and loc.owner != loc.host_id:
            visible.append(loc.owner)
        visible += sorted((h for h in seen if h not in visible), key=lambda h: order.get(h, 1 << 30))
        shared = loc.owner if loc.imported and any(o.host_id == loc.owner and not o.imported for o in peers) else None
        out[(loc.host_id, loc.root_id)] = (shared, visible)
    return out


# ---- the merged view -----------------------------------------------------------------------------------------------
@dataclass
class _Alias:
    host_id: str
    root_id: str
    rel: str
    path: str


@dataclass
class _Rec:
    id: str
    host_id: str
    root_id: str
    rel: str
    path: str
    fmt: str
    name: str
    key: str
    entry: dict[str, Any]
    size: int
    mount: Mount | None
    base: LibraryCopy
    visible_on: list[str]
    aliases: list[_Alias]


@dataclass
class _Snapshot:
    recs: dict[str, _Rec] = field(default_factory=dict)
    models: dict[str, LibraryModel] = field(default_factory=dict)
    members: dict[str, list[str]] = field(default_factory=dict)
    alias: dict[tuple[str, str, str], str] = field(default_factory=dict)
    by_host: dict[str, list[str]] = field(default_factory=dict)
    hf_by_name: dict[str, list[str]] = field(default_factory=dict)
    ollama_by_name: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class _ReadCtx:
    views: dict[str, _HostView]
    order: dict[str, int]
    book: _AddressBook
    drives: dict[tuple[str, str], MountInfo | None] = field(default_factory=dict)


def _loadable_hf(entry: dict[str, Any]) -> bool:
    return entry.get("revision") is not None or (_i(entry.get("files")) or 0) > 0


def _fmt_of(entry: dict[str, Any]) -> str:
    f = _s(entry.get("format"), 16)
    return f if f in FORMATS else "other"


def _derive_name(fmt: str, entry: dict[str, Any], rel: str) -> str:
    name = _s(entry.get("name"), 256)
    last = rel.rsplit("/", 1)[-1] if rel else ""
    if name:
        return name
    if fmt == "hf":
        m = _HF_DIR.search(rel)
        return f"{m.group(1)}/{m.group(2)}" if m else (last or "model")
    if fmt == "gguf":
        return gguf_stem(last) or "model"
    return last or "model"


def _entry_key(fmt: str, name: str, entry: dict[str, Any], rel: str) -> tuple[str, str | None]:
    """(key, quant). hf: the repo id; gguf: repo id or file stem plus ":<quant>"; ollama: name:tag; plain: the
    directory name; anything else "other:<dir>"."""
    quant = _s(entry.get("quant"), 32)
    if fmt == "gguf":
        quant = quant or gguf_quant(rel)
        base = name
        if quant and not base.lower().endswith(":" + quant.lower()):
            base = f"{base}:{quant.upper()}"
        return f"gguf:{base}", quant
    if fmt == "other":
        return f"other:{name}", quant
    return f"{fmt}:{name}", quant


GGUF_HEADER_KEYS = ("block_count", "embedding_length", "head_count", "head_count_kv", "expert_count",
                    "expert_used_count", "file_type", "size_label", "version")


def expand_gguf_repo(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """A GGUF-only HF repo arrives from the node as ONE entry whose rel is the repo directory (its snapshot files are
    links into `blobs/`, so only a mount of the whole repo directory resolves them), with every quant (or split set)
    in meta.gguf_files. The library lists each quant as a model of its own: rel = <repo dir>/<path of its first file>,
    which is what a launch names and what the mount logic turns back into "mount the repo directory"."""
    meta = entry.get("meta")
    files = meta.get("gguf_files") if isinstance(meta, dict) else None
    base_rel = entry.get("rel")
    if (_fmt_of(entry) != "gguf" or not isinstance(files, list) or not files or not isinstance(base_rel, str)
            or not base_rel or base_rel.lower().endswith(".gguf")):
        return [entry]
    assert isinstance(meta, dict)
    projector = any(isinstance(f, dict) and f.get("role") == "mmproj" for f in files)
    warnings = [w for w in entry.get("warnings") or [] if isinstance(w, str)][:7]
    if projector:
        warnings.append("the repo also has a vision projector (mmproj) file; image input is not wired up for it yet")
    out: list[dict[str, Any]] = []
    for f in files:
        if not isinstance(f, dict) or f.get("role") is not None or not isinstance(f.get("rel"), str) or not f["rel"]:
            continue
        out.append({**entry, "rel": f"{base_rel}/{f['rel']}", "size": f.get("size"), "files": f.get("files"),
                    "shards": f.get("shards"), "quant": f.get("quant"), "params": None, "warnings": warnings,
                    "meta": {"hf_repo": meta.get("hf_repo"), "repo_rel": base_rel}})
    return out or [entry]


def _mount_info(m: Mount, owner: str | None, views: Mapping[str, _HostView]) -> MountInfo:
    owner_label = views[owner].label if owner and owner in views else None
    return MountInfo(
        mount=m.mount, label=mount_label(m, owner_label), fstype=m.fstype, source=m.source, device=m.device,
        model=m.model, size_bytes=m.size, used_bytes=m.used, free_bytes=m.free, readonly=m.readonly, remote=m.remote,
        remote_host=m.remote_host, remote_host_id=owner, remote_path=m.remote_path, roots=list(m.roots))


class LibraryService:
    """The model library. See the module docstring. All methods run on the event loop; none blocks on a node except
    the `async` ones."""

    def __init__(self, store: Store, fleet: FleetPort, adapters: AdapterRegistry, bus: EventBus, *,
                 in_use: Callable[[str, str], list[str]], page_limit: int = 200,
                 retry_delays: Sequence[float] = (5.0, 15.0, 60.0), call_timeout: float = 30.0) -> None:
        self._db = store
        self._fleet = fleet
        self._adapters = adapters
        self._bus = bus
        self._in_use = in_use
        self._page_limit = max(1, min(page_limit, 200))
        self._retry_delays = tuple(retry_delays) or (5.0,)
        self._call_timeout = call_timeout
        self._snaps: dict[str, HostSnapshot] = {}
        self._view = _Snapshot()
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._dirty: set[str] = set()
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._consumers: list[asyncio.Task[None]] = []
        self._up: set[str] = set()
        self._errors: dict[str, str] = {}
        self._sigs: dict[str, tuple[Any, ...]] = {}
        self._info_cache: dict[tuple[str, int | None, str], ModelInfo] = {}

    # ---- lifecycle ----------------------------------------------------------------------------------------------------
    async def start(self) -> None:
        """Load the stored snapshots, subscribe to the node and fleet topics, and fetch every host that is up. The
        fetches run in the background: a slow or dead machine must not hold the console's start-up."""
        if self._consumers:
            return
        self._load({h.id for h in self._fleet.hosts()})
        self._rebuild()
        for topic, handler in (("node.models_changed", self._on_changed), ("fleet", self._on_fleet),
                               ("node.disks", self._on_facts), ("node.hello", self._on_facts)):
            self._consumers.append(asyncio.create_task(self._consume(topic, handler), name=f"library-{topic}"))
        # Each consumer registers its subscription when it first runs. Yield so that an event published right after
        # start() returns is not lost.
        for _ in range(2):
            await asyncio.sleep(0)
        for h in self._fleet.hosts():
            if self._state(h.id) == "up":
                self._up.add(h.id)
                self._schedule(h.id)

    async def stop(self) -> None:
        tasks = [*self._consumers, *self._workers.values()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._consumers.clear()
        self._workers.clear()

    async def wait_idle(self, timeout: float = 10.0) -> None:
        """Test and ops helper: block until no host is being fetched."""
        pending = [t for t in self._workers.values() if not t.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout)

    # ---- events -------------------------------------------------------------------------------------------------------
    async def _consume(self, topic: str, handler: Callable[[Any], None]) -> None:
        async for ev in self._bus.subscribe(topic):
            try:
                handler(ev)
            except Exception:  # noqa: BLE001 - one malformed event must not end the subscription
                log.exception("library: handling a %s event failed", topic)

    def _known(self, host_id: object) -> bool:
        return isinstance(host_id, str) and any(h.id == host_id for h in self._fleet.hosts())

    def _on_changed(self, ev: Any) -> None:
        host_id = ev.get("host_id") if isinstance(ev, dict) else None
        if self._known(host_id) and self._state(str(host_id)) == "up":
            self._schedule(str(host_id))

    def _on_fleet(self, ev: Any) -> None:
        if not isinstance(ev, dict):
            return
        host_id = ev.get("id")
        link = ev.get("link")
        state = link.get("state") if isinstance(link, dict) else None
        if not self._known(host_id):
            return
        host = str(host_id)
        if state == "up":
            if host not in self._up:
                self._up.add(host)
                self._schedule(host)
        else:
            self._up.discard(host)   # the library is shown stale at read time; nothing to rebuild

    def _on_facts(self, ev: Any) -> None:
        """A host's `hello` (policy: roots, addresses) or `disks` (mounts) changed in a way that moves copies between
        drives or changes who shares them: rebuild. Free-space changes do not need it."""
        host_id = ev.get("host_id") if isinstance(ev, dict) else None
        if not self._known(host_id) or str(host_id) not in self._snaps:
            return
        view = self._views().get(str(host_id))
        if view is not None and self._sigs.get(str(host_id)) != view.facts.signature():
            self._rebuild()

    # ---- fetching -----------------------------------------------------------------------------------------------------
    def _state(self, host_id: str) -> str:
        try:
            return self._fleet.node(host_id).link().state
        except NotFound:
            return "disabled"

    def _schedule(self, host_id: str) -> None:
        """Ask for a refetch of one host. Requests that arrive while a fetch runs collapse into one more fetch."""
        self._dirty.add(host_id)
        t = self._workers.get(host_id)
        if t is None or t.done():
            self._workers[host_id] = asyncio.create_task(self._worker(host_id), name=f"library-fetch-{host_id}")

    async def _worker(self, host_id: str) -> None:
        failures = 0
        while host_id in self._dirty:
            self._dirty.discard(host_id)
            try:
                await self._fetch(host_id)
                failures = 0
                continue
            except asyncio.CancelledError:
                raise
            except NodeError as e:
                self._errors[host_id] = f"{e.code}: {e.message}"[:300]
                log.warning("library: fetching %s failed: %s", host_id, e.code)
            except Exception:  # noqa: BLE001 - a bad answer from one node must not stop the others
                self._errors[host_id] = "unexpected error while reading the node's library"
                log.exception("library: fetching %s failed", host_id)
            failures += 1
            if self._state(host_id) != "up":
                return          # fetched again when the link comes back up
            await asyncio.sleep(self._retry_delays[min(failures - 1, len(self._retry_delays) - 1)])
            self._dirty.add(host_id)

    async def refresh(self, host_id: str) -> None:
        """Fetch one host's library now (errors propagate as NodeError)."""
        await self._fetch(host_id)

    async def _fetch(self, host_id: str) -> None:
        node = self._fleet.node(host_id)
        async with self._locks[host_id]:
            entries, scans, generation = await self._list_all(node, host_id)
            prev = self._snaps.get(host_id)
            facts = parse_facts(node.latest("hello"), node.latest("disks"), prev.facts if prev else None)
            snap = HostSnapshot(generation=generation, entries=entries, scans=scans, facts=facts or HostFacts(),
                                updated_at=now())
            self._snaps[host_id] = snap
            self._errors.pop(host_id, None)
            for k in [k for k in self._info_cache if k[0].startswith(host_id + "|")]:
                del self._info_cache[k]
            await asyncio.to_thread(self._persist, host_id, snap)
            self._rebuild()
        self._bus.publish(LIBRARY_TOPIC, {"host_id": host_id, "generation": generation, "models": len(entries)})

    async def _list_all(self, node: NodeClient, host_id: str
                        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int | None]:
        """Every page of `models.list`. A scan finishing between two pages would mix two generations, so a pass that
        sees the generation change starts over (twice at most, then the mix is accepted)."""
        for attempt in range(3):
            out = await self._one_pass(node, host_id, accept_change=attempt == 2)
            if out is not None:
                return out
        raise AssertionError("unreachable")   # the last pass accepts a change

    async def _one_pass(self, node: NodeClient, host_id: str, *, accept_change: bool
                        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int | None] | None:
        entries: list[dict[str, Any]] = []
        scans: list[dict[str, Any]] = []
        generation: int | None = None
        cursor = 0
        for _page in range(MAX_ENTRIES // self._page_limit + 2):
            args: dict[str, Any] = {"limit": self._page_limit}
            if cursor:
                args["cursor"] = cursor
            res = await node.call("models.list", args, timeout=self._call_timeout)
            gen = _i(res.get("generation"))
            if generation is None:
                generation = gen
            elif gen is not None and gen != generation and not accept_change:
                return None
            entries.extend(e for e in (res.get("entries") or []) if isinstance(e, dict))
            if not scans:
                scans = [s for s in (res.get("roots") or []) if isinstance(s, dict)]
            nxt = _i(res.get("next_cursor"))
            if res.get("next_cursor") is None:
                return entries, scans, generation
            if nxt is None or nxt <= cursor:
                log.warning("library: %s sent a page cursor that does not advance; keeping what it sent", host_id)
                return entries, scans, generation
            cursor = nxt
            if len(entries) >= MAX_ENTRIES:
                break
        log.warning("library: %s has more than %d entries; the rest are not listed", host_id, MAX_ENTRIES)
        return entries[:MAX_ENTRIES], scans, generation

    # ---- persistence --------------------------------------------------------------------------------------------------
    def _persist(self, host_id: str, snap: HostSnapshot) -> None:
        self._db.execute(
            "INSERT INTO library_snapshots(host_id,generation,roots_json,entries_json,scans_json,facts_json,updated_at)"
            " VALUES(?,?,?,?,?,?,?) ON CONFLICT(host_id) DO UPDATE SET generation=excluded.generation,"
            " roots_json=excluded.roots_json, entries_json=excluded.entries_json, scans_json=excluded.scans_json,"
            " facts_json=excluded.facts_json, updated_at=excluded.updated_at",
            (host_id, snap.generation, json.dumps(snap.facts.roots_json()), json.dumps(snap.entries),
             json.dumps(snap.scans), json.dumps(snap.facts.to_json()), snap.updated_at))

    def _load(self, known: set[str]) -> None:
        for r in self._db.all("SELECT * FROM library_snapshots"):
            host_id = r["host_id"]
            if host_id not in known:
                self._db.execute("DELETE FROM library_snapshots WHERE host_id=?", (host_id,))
                continue
            try:
                entries = proto.clean(json.loads(r["entries_json"]), proto.L(proto.MODEL_ENTRY, MAX_ENTRIES)) or []
                scans = proto.clean(json.loads(r["scans_json"]), proto.L(proto.ROOT_SCAN, 32)) or []
                facts = HostFacts.from_json(json.loads(r["roots_json"]), json.loads(r["facts_json"]))
            except (ValueError, TypeError):
                log.warning("library: dropping an unreadable stored snapshot of %s", host_id)
                self._db.execute("DELETE FROM library_snapshots WHERE host_id=?", (host_id,))
                continue
            self._snaps[host_id] = HostSnapshot(generation=_i(r["generation"]), entries=entries, scans=scans,
                                                facts=facts, updated_at=float(r["updated_at"]))

    # ---- building the merged view -------------------------------------------------------------------------------------
    def _views(self) -> dict[str, _HostView]:
        views: dict[str, _HostView] = {}
        for h in self._fleet.hosts():
            node = self._fleet.node(h.id)
            snap = self._snaps.get(h.id)
            facts = parse_facts(node.latest("hello"), node.latest("disks"), snap.facts if snap else None)
            if facts is None:
                continue
            views[h.id] = _HostView(id=h.id, label=h.label, state=node.link().state, facts=facts)
        return views

    def _make_rec(self, view: _HostView, entry: dict[str, Any]) -> _Rec | None:
        root_id, rel = _s(entry.get("root"), 32), entry.get("rel")
        spec = view.facts.root(root_id) if root_id else None
        if root_id is None or spec is None or not isinstance(rel, str):
            return None
        rel = "" if rel in ("", ".") else rel
        if rel and not _rel_ok(rel):
            return None            # a node-supplied path that breaks the node's own rules is not listed
        fmt = _fmt_of(entry)
        name = _derive_name(fmt, entry, rel)
        key, _quant = _entry_key(fmt, name, entry, rel)
        path = _norm(spec.path + "/" + rel) if rel else spec.path
        size = max(0, _i(entry.get("size")) or 0)
        cid = copy_id(view.id, root_id, rel)
        base = LibraryCopy(
            id=cid, host_id=view.id, host_label=view.label, root_id=root_id, root_label=spec.label, rel=rel, path=path,
            size_bytes=size, files=_i(entry.get("files")), mtime=_f(entry.get("mtime")), atime=_f(entry.get("atime")),
            visible_on=[view.id], writable=spec.writable)
        return _Rec(id=cid, host_id=view.id, root_id=root_id, rel=rel, path=path, fmt=fmt, name=name, key=key,
                    entry=entry, size=size, mount=mount_for(view.facts.mounts, path), base=base,
                    visible_on=[view.id], aliases=[_Alias(view.id, root_id, rel, path)])

    @staticmethod
    def _merge_shared(recs: list[_Rec], book: _AddressBook) -> list[_Rec]:
        """Fold copies on an NFS mount into the copy on the disk that exports it (docs/ARCHITECTURE.md 10.5): the NFS
        server must be one of a configured host's addresses, and the copy's path on that host (export path plus the
        copy's path under the mount) must be a copy that host lists. Anything else stays a copy of its own."""
        owners: dict[tuple[str, str], _Rec] = {}
        for r in recs:
            if r.mount is None or not r.mount.nfs:
                owners[(r.host_id, _norm(r.path))] = r
        kept: list[_Rec] = []
        for r in recs:
            if r.mount is None or not r.mount.nfs:
                kept.append(r)
                continue
            owner = book.find(r.mount.remote_host, r.host_id)
            target: _Rec | None = None
            if owner and r.mount.remote_path and _under(r.path, r.mount.mount):
                sub = posixpath.relpath(r.path, r.mount.mount)
                equivalent = r.mount.remote_path if sub == "." else posixpath.normpath(
                    posixpath.join(r.mount.remote_path, sub))
                target = owners.get((owner, equivalent))
            if target is None:
                kept.append(r)       # the owner is not a configured host (or does not list it): B's copy stays
                continue
            if r.host_id not in target.visible_on:
                target.visible_on.append(r.host_id)
            target.aliases.append(_Alias(r.host_id, r.root_id, r.rel, r.path))
        return kept

    def _rebuild(self) -> None:
        views = self._views()
        order = {h.id: n for n, h in enumerate(self._fleet.hosts())}
        book = _AddressBook(views)
        recs: list[_Rec] = []
        for host_id, snap in self._snaps.items():
            view = views.get(host_id)
            if view is None:
                continue
            self._sigs[host_id] = view.facts.signature()
            for raw in snap.entries:
                for entry in expand_gguf_repo(raw):
                    rec = self._make_rec(view, entry)
                    if rec is not None:
                        recs.append(rec)
        recs.sort(key=lambda r: (order.get(r.host_id, 1 << 30), r.root_id, r.rel))
        kept = self._merge_shared(recs, book)

        merged = _Snapshot()
        by_key: dict[str, list[_Rec]] = {}
        for r in kept:
            merged.recs[r.id] = r
            merged.by_host.setdefault(r.host_id, []).append(r.id)
            for a in r.aliases:
                merged.alias[(a.host_id, a.root_id, a.rel)] = r.id
            if r.fmt == "hf":
                merged.hf_by_name.setdefault(r.name, []).append(r.id)
            elif r.fmt == "ollama":
                merged.ollama_by_name.setdefault(r.name, []).append(r.id)
            by_key.setdefault(r.key, []).append(r)
        for key, members in by_key.items():
            merged.models[key] = self._model_of(key, members, views)
            merged.members[key] = [m.id for m in members]
        self._view = merged

    def _engines_for(self, fmt: str) -> list[str]:
        if fmt not in ("hf", "gguf", "ollama", "plain"):
            return []
        return [a.id for a in self._adapters.list() if fmt in a.model_formats]

    def _model_of(self, key: str, members: list[_Rec], views: Mapping[str, _HostView]) -> LibraryModel:
        first = members[0]

        def pick(field_name: str, kind: type) -> Any:
            for r in members:
                v = r.entry.get(field_name)
                if isinstance(v, kind) and not isinstance(v, bool) and v not in ("", 0):
                    return v
            return None

        warnings: list[str] = []
        for r in members:
            for w in r.entry.get("warnings") or []:
                if isinstance(w, str) and w and w not in warnings:
                    warnings.append(w)
        revisions = {str(r.entry.get("revision")) for r in members if r.entry.get("revision")}
        if len(revisions) > 1:
            warnings.append("copies are at different revisions: " + ", ".join(sorted(v[:12] for v in revisions)))
        for r in members:
            for scan in self._snaps[r.host_id].scans:
                if scan.get("root") == r.root_id and (scan.get("complete") is False or scan.get("error")):
                    label = views[r.host_id].label if r.host_id in views else r.host_id
                    why = f": {scan.get('error')}" if scan.get("error") else ""
                    w = f"{label}: the scan of root {r.root_id} did not finish{why}"
                    if w not in warnings:
                        warnings.append(w)
        fmt = cast(LibraryFormat, first.fmt)    # _fmt_of() only ever returns a member of FORMATS
        return LibraryModel(
            key=key, name=first.name, format=fmt, revision=pick("revision", str),
            quant=_entry_key(first.fmt, first.name, first.entry, first.rel)[1], params=pick("params", int),
            arch=pick("arch", str), context=pick("context", int), shards=pick("shards", int),
            size_bytes=max(r.size for r in members), engines=self._engines_for(first.fmt), copies=[],
            warnings=warnings[:MAX_WARNINGS])

    # ---- reading (live overlay) ---------------------------------------------------------------------------------------
    def _ctx(self) -> _ReadCtx:
        views = self._views()
        return _ReadCtx(views=views, order={h.id: n for n, h in enumerate(self._fleet.hosts())},
                        book=_AddressBook(views))

    def _users(self, rec: _Rec) -> list[str]:
        """Instances using this copy on any machine that can see it (the owner's path and every NFS alias)."""
        out: list[str] = []
        for a in rec.aliases:
            try:
                ids = self._in_use(a.host_id, a.path)
            except Exception:  # noqa: BLE001 - a bug in the callback must not blank the library
                log.exception("library: the in_use callback failed for %s", a.host_id)
                ids = []
            out.extend(i for i in ids if i not in out)
        return out

    def _drive(self, rec: _Rec, ctx: _ReadCtx) -> MountInfo | None:
        return self._drive_of(rec.host_id, rec.mount, ctx)

    @staticmethod
    def _drive_of(host_id: str, mount: Mount | None, ctx: _ReadCtx) -> MountInfo | None:
        """The drive a path lives on, with the free space the node last reported (the mount found when the view was
        built is only the fallback: free space moves, mounts rarely do)."""
        if mount is None:
            return None
        cache_key = (host_id, mount.mount)
        if cache_key not in ctx.drives:
            view = ctx.views.get(host_id)
            live = next((m for m in view.facts.mounts if m.mount == mount.mount), None) if view else None
            m = live or mount
            owner = ctx.book.find(m.remote_host, host_id) if m.remote else None
            ctx.drives[cache_key] = _mount_info(m, owner, ctx.views)
        return ctx.drives[cache_key]

    def _copy_view(self, rec: _Rec, ctx: _ReadCtx, alias: _Alias | None = None, *, with_users: bool = True) -> LibraryCopy:
        """The copy as the API shows it. With `alias` (a machine that reaches the copy over shared storage) the same copy
        is expressed in THAT machine's terms: its root, its path, the NFS mount it goes through. The id stays the owner's,
        so every action on it still acts on the one real copy."""
        view = ctx.views.get(rec.host_id)
        spec = view.facts.root(rec.root_id) if view else None
        # with_users=False is for the lifecycle, which only needs where the copy is: asking it who uses the copy from
        # there would call back into the lifecycle (instances_using -> locate -> ...) without end.
        users = self._users(rec) if with_users else []
        writable = bool(spec and spec.writable)
        deletable = bool(view and spec and view.state == "up" and writable and view.facts.allow_delete
                         and view.facts.mode == "manage" and not users and rec.rel)
        out = rec.base.model_copy(update={
            "writable": writable, "deletable": deletable, "in_use_by": users, "drive": self._drive(rec, ctx),
            "visible_on": list(rec.visible_on)})
        if alias is None or alias.host_id == rec.host_id:
            return out
        there = ctx.views.get(alias.host_id)
        there_spec = there.facts.root(alias.root_id) if there else None
        there_mount = mount_for(there.facts.mounts, alias.path) if there else None
        return out.model_copy(update={
            "host_id": alias.host_id, "host_label": there.label if there else alias.host_id, "root_id": alias.root_id,
            "root_label": there_spec.label if there_spec else None, "rel": alias.rel, "path": alias.path,
            "writable": bool(there_spec and there_spec.writable), "drive": self._drive_of(alias.host_id, there_mount, ctx)})

    @staticmethod
    def _pick_alias(rec: _Rec, host_id: str, ctx: _ReadCtx) -> _Alias:
        """The way `host_id` reaches this copy. Normally one; when it has two roots on the same storage, the root whose
        kind naturally holds this format (an HF cache for an HF model) wins, then an `auto` root, then the first."""
        mine = [a for a in rec.aliases if a.host_id == host_id]
        if len(mine) <= 1:
            return mine[0] if mine else rec.aliases[0]
        view = ctx.views.get(host_id)

        def rank(a: _Alias) -> int:
            spec = view.facts.root(a.root_id) if view else None
            kind = spec.kind if spec else ""
            return 0 if kind in _NATURAL_KINDS.get(rec.fmt, ()) else 1 if kind == "auto" else 2

        return min(mine, key=rank)

    def _model_view(self, key: str, ctx: _ReadCtx) -> LibraryModel:
        snap = self._view
        copies = [self._copy_view(snap.recs[i], ctx) for i in snap.members[key]]
        warnings = list(snap.models[key].warnings)
        for host_id in sorted({c.host_id for c in copies}, key=lambda h: ctx.order.get(h, 1 << 30)):
            v = ctx.views.get(host_id)
            if v is not None and v.state not in ("up", "stale"):
                warnings.append(f"{v.label} is {v.state}: showing its last-known library, which cannot be changed "
                                "until it reconnects")
        return snap.models[key].model_copy(update={"copies": copies, "warnings": warnings[:MAX_WARNINGS]})

    def models(self, host: str | None = None, format: str | None = None, q: str | None = None) -> list[LibraryModel]:
        """The library, by name. `host` keeps the models that host holds or can load from shared storage (every copy
        of such a model is still listed: the page wants to show where else it lives). `q` is matched word by word,
        case-insensitively, against the name, key, quantisation, architecture and format."""
        ctx = self._ctx()
        snap = self._view
        words = (q or "").lower().split()
        out: list[LibraryModel] = []
        for key, m in snap.models.items():
            if format and m.format != format:
                continue
            if host and not any(host in snap.recs[i].visible_on for i in snap.members[key]):
                continue
            if words:
                hay = " ".join(x for x in (m.name, m.key, m.quant, m.arch, m.format) if x).lower()
                if not all(w in hay for w in words):
                    continue
            out.append(self._model_view(key, ctx))
        return sorted(out, key=lambda m: (m.name.lower(), m.key))

    def model(self, key: str) -> LibraryModel | None:
        return self._model_view(key, self._ctx()) if key in self._view.models else None

    def copies(self, host_id: str) -> list[LibraryCopy]:
        """The copies physically on `host_id`'s disks (a shared copy belongs to the machine that owns the disk)."""
        ctx = self._ctx()
        return [self._copy_view(self._view.recs[i], ctx) for i in self._view.by_host.get(host_id, [])]

    def visible_copies(self, host_id: str) -> list[LibraryCopy]:
        """Every copy `host_id` can load: its own plus those it reaches over shared storage, each expressed in
        `host_id`'s own terms (see `find`)."""
        ctx = self._ctx()
        return [self._copy_view(r, ctx, self._pick_alias(r, host_id, ctx))
                for r in self._view.recs.values() if host_id in r.visible_on]

    def copy(self, copy_id: str) -> LibraryCopy | None:
        rec = self._rec(copy_id)
        return self._copy_view(rec, self._ctx()) if rec else None

    def _rec(self, copy_id: str) -> _Rec | None:
        return self._view.recs.get(copy_id) if _COPY_ID.fullmatch(copy_id) else None

    def _find_rec(self, host_id: str, ref: ModelRef, *, own: bool = False) -> _Rec | None:
        """The copy `host_id` would load for `ref`. `own` keeps only copies on `host_id`'s own disks."""
        snap = self._view
        ids: list[str] = []
        if ref.format == "hf" and ref.repo_id:
            # a repo directory with no snapshot (a download in progress, or one that was cut off) is listed, with the
            # node's warning, but it is not a model an engine can load
            ids = [i for i in snap.hf_by_name.get(ref.repo_id, []) if _loadable_hf(snap.recs[i].entry)]
        elif ref.format == "ollama" and ref.name:
            ids = snap.ollama_by_name.get(ref.name, [])
        elif ref.format in ("gguf", "plain") and ref.root_id and ref.rel is not None:
            rid = snap.alias.get((host_id, ref.root_id, ref.rel))
            if rid is not None:
                ids = [rid]
            else:   # the owner's coordinates, for a host that reaches the copy over shared storage
                ids = [i for i, r in snap.recs.items() if (r.root_id, r.rel) == (ref.root_id, ref.rel)]
            ids = [i for i in ids if snap.recs[i].fmt == ref.format]
        cands = [snap.recs[i] for i in ids
                 if host_id in snap.recs[i].visible_on and (not own or snap.recs[i].host_id == host_id)]
        own_disk = [r for r in cands if r.host_id == host_id]
        return own_disk[0] if own_disk else (cands[0] if cands else None)

    def _alias_for(self, rec: _Rec, host_id: str, ref: ModelRef, ctx: _ReadCtx) -> _Alias:
        """The way `host_id` reaches `rec`: the root a gguf or plain ref names, else the natural one."""
        if ref.format in ("gguf", "plain") and ref.root_id and ref.rel is not None:
            for a in rec.aliases:
                if (a.host_id, a.root_id, a.rel) == (host_id, ref.root_id, ref.rel):
                    return a
        return self._pick_alias(rec, host_id, ctx)

    def find(self, host_id: str, ref: ModelRef, *, own: bool = False) -> LibraryCopy | None:
        """The copy `host_id` would load for `ref`: one on its own disks, else one it reaches over shared storage. hf by
        repo id, gguf and plain by root + path, ollama by name:tag.

        A copy on shared storage is returned in `host_id`'s own terms: its root id, path and the NFS mount it goes
        through, because that is what an engine started on `host_id` mounts. `id` and `visible_on` stay the real copy's,
        so a delete or a copy of it still acts on the one set of files. `own=True` keeps only copies on the machine's own
        disks (what a delete from that machine may remove)."""
        rec = self._find_rec(host_id, ref, own=own)
        if rec is None:
            return None
        ctx = self._ctx()
        return self._copy_view(rec, ctx, self._alias_for(rec, host_id, ref, ctx))

    def locate(self, host_id: str, ref: ModelRef) -> LibraryCopy | None:
        """`find` without `in_use_by` (empty) and with `deletable` not to be trusted: where the copy is and how
        `host_id` reaches it, for the lifecycle's launches and locations."""
        rec = self._find_rec(host_id, ref, own=False)
        if rec is None:
            return None
        ctx = self._ctx()
        return self._copy_view(rec, ctx, self._alias_for(rec, host_id, ref, ctx), with_users=False)

    def source(self, copy_id: str) -> CopySource | None:
        rec = self._rec(copy_id)
        if rec is None:
            return None
        return CopySource(copy=self._copy_view(rec, self._ctx()), format=rec.fmt, name=rec.name,
                          revision=_s(rec.entry.get("revision"), 64), quant=_s(rec.entry.get("quant"), 32),
                          shards=_i(rec.entry.get("shards")), locations=[(a.host_id, a.root_id) for a in rec.aliases],
                          warnings=[w for w in (rec.entry.get("warnings") or []) if isinstance(w, str)])

    def local_models(self) -> list[LocalModel]:
        """The v1 `GET /models` shape: the Hugging Face models of every host, each with its host id."""
        out = [LocalModel(repo_id=r.name, size_bytes=r.size,
                          revisions=[str(r.entry["revision"])] if r.entry.get("revision") else [],
                          last_used=_f(r.entry.get("atime")), path=r.path, host_id=r.host_id)
               for r in self._view.recs.values() if r.fmt == "hf"]
        return sorted(out, key=lambda m: (m.repo_id, m.host_id))

    def status(self) -> dict[str, dict[str, Any]]:
        """Per host: when its library was last read, which generation, whether it is stale, the last error."""
        out: dict[str, dict[str, Any]] = {}
        for h in self._fleet.hosts():
            snap = self._snaps.get(h.id)
            out[h.id] = {"generation": snap.generation if snap else None,
                         "updated_at": snap.updated_at if snap else None,
                         "entries": len(snap.entries) if snap else 0, "stale": self._state(h.id) != "up",
                         "error": self._errors.get(h.id)}
        return out

    # ---- roots --------------------------------------------------------------------------------------------------------
    def host_facts(self, host_id: str) -> HostFacts | None:
        """The host's policy roots, delete switch, mode and mounts: live when its node has spoken, else stored."""
        try:
            node = self._fleet.node(host_id)
        except NotFound:
            return None
        snap = self._snaps.get(host_id)
        return parse_facts(node.latest("hello"), node.latest("disks"), snap.facts if snap else None)

    def host_label(self, host_id: str) -> str:
        try:
            return self._fleet.node(host_id).host.label
        except NotFound:
            return host_id

    def root_usage(self, host_id: str, root_id: str) -> RootUsage:
        snap = self._snaps.get(host_id)
        n = hf_home = hf_hub = loose = ollama = other = 0
        for e in (snap.entries if snap else []):
            if e.get("root") != root_id:
                continue
            n += 1
            fmt, rel = _fmt_of(e), e.get("rel") or ""
            if fmt == "hf":
                hf_home += rel.startswith("hub/")
                hf_hub += rel.startswith("models--")
            elif fmt in ("gguf", "plain"):
                loose += 1
            elif fmt == "ollama":
                ollama += 1
            else:
                other += 1
        return RootUsage(entries=n, hf_home=bool(hf_home), hf_hub=bool(hf_hub), loose=bool(loose),
                         ollama=bool(ollama), other=bool(other))

    def free_bytes(self, host_id: str, path: str) -> int | None:
        """Free bytes on the drive holding `path`, from the node's `disks` (None when it has not reported)."""
        facts = self.host_facts(host_id)
        m = mount_for(facts.mounts, _norm(path)) if facts else None
        return m.free if m else None

    def roots(self, host_id: str) -> list[RootInfo]:
        ctx = self._ctx()
        view = ctx.views.get(host_id)
        if view is None:
            return []
        shares = _shares(ctx.views, ctx.book, ctx.order)
        out: list[RootInfo] = []
        for spec in view.facts.roots:
            m = _mount_of_root(spec, view.facts.mounts)
            shared_from, visible = shares.get((host_id, spec.id), (None, [host_id]))
            owner = ctx.book.find(m.remote_host, host_id) if m is not None and m.remote else None
            out.append(RootInfo(id=spec.id, path=spec.path, kind=spec.kind, writable=spec.writable, label=spec.label,
                                mount=_mount_info(m, owner, ctx.views) if m is not None else None,
                                shared_from=shared_from, visible_on=visible))
        return out

    def root(self, host_id: str, root_id: str) -> RootInfo | None:
        return next((r for r in self.roots(host_id) if r.id == root_id), None)

    def hf_root(self, host_id: str) -> RootInfo | None:
        """The host's first Hugging Face cache root: kind hf-home or hf-hub, or an `auto` root whose scan found one."""
        for r in self.roots(host_id):
            if r.kind in ("hf-home", "hf-hub"):
                return r
            if r.kind == "auto":
                u = self.root_usage(host_id, r.id)
                if u.hf_home or u.hf_hub:
                    return r
        return None

    def ollama_root(self, host_id: str) -> RootInfo | None:
        for r in self.roots(host_id):
            if r.kind == "ollama":
                return r
            if r.kind == "auto" and self.root_usage(host_id, r.id).ollama:
                return r
        return None

    # ---- model facts --------------------------------------------------------------------------------------------------
    async def model_info(self, host_id: str, ref: ModelRef) -> ModelInfo | None:
        """The facts the fit estimator needs, from the copy `host_id` would load: hf and plain from the model's
        config.json (asked of a node that can read it), gguf and ollama from the scanned header. None when the host
        has no copy; a ModelInfo with only what is known when a config is missing."""
        rec = self._find_rec(host_id, ref)
        if rec is None or rec.fmt == "other":
            return None
        snap = self._snaps.get(rec.host_id)
        ck = (f"{rec.host_id}|{rec.id}", snap.generation if snap else None, rec.fmt)
        cached = self._info_cache.get(ck)
        if cached is not None:
            return cached.model_copy()
        revision = _s(rec.entry.get("revision"), 64)
        if rec.fmt in ("gguf", "ollama"):
            entry = rec.entry
            raw_meta = entry.get("meta")
            meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
            if rec.fmt == "gguf" and "block_count" not in meta:
                # A quant expanded from an HF repo (expand_gguf_repo) carries no header facts of its own: read that
                # file's header through the node (models.config on a .gguf answers {"gguf": {...}}).
                cfg = await self._config_of(rec)
                header = cfg.get("gguf") if isinstance(cfg, dict) else None
                if isinstance(header, dict):
                    entry = {**entry, "arch": entry.get("arch") or header.get("arch"),
                             "context": entry.get("context") or header.get("context"),
                             "params": entry.get("params") or header.get("params"),
                             "meta": {**meta, **{k: header[k] for k in GGUF_HEADER_KEYS if header.get(k) is not None}}}
            info = model_info_from_entry(rec.name, entry, rec.fmt, size=rec.size)
        else:
            cfg = await self._config_of(rec)
            hub = HubModel(repo_id=rec.name, sha=revision, gated=False, license=None, pipeline_tag=None,
                           library_name=None, tags=[], files=[], safetensors_total=_i(rec.entry.get("params")),
                           config=cfg)
            info = build_model_info(hub, revision)
            fill: dict[str, Any] = {}
            if info.weight_bytes is None and rec.size:
                fill["weight_bytes"] = rec.size
            if info.quantization is None and (q := normalize_quant(_s(rec.entry.get("quant"), 32))):
                fill["quantization"] = q
            info = info.model_copy(update=fill)
        self._info_cache[ck] = info
        return info.model_copy()

    async def _config_of(self, rec: _Rec) -> dict[str, Any] | None:
        """models.config from the owner's node, else from a machine that reaches the copy over shared storage."""
        for a in rec.aliases:
            try:
                node = self._fleet.node(a.host_id)
                if node.link().state != "up":
                    continue
                res = await node.call("models.config", {"root": a.root_id, "rel": a.rel}, timeout=self._call_timeout)
            except NodeError as e:
                if e.code != "not_found":
                    log.info("library: models.config on %s answered %s", a.host_id, e.code)
                continue
            except NotFound:
                continue
            cfg = res.get("config")
            if isinstance(cfg, dict):
                return cfg
        return None

    # ---- commands -----------------------------------------------------------------------------------------------------
    async def rescan(self, host_id: str | None = None) -> list[str]:
        """Ask nodes to rescan their model roots now (`models.scan` force) and re-read what they already have. Returns
        the hosts asked. For one named host a refusal is an error; for all hosts, unreachable ones are skipped."""
        if host_id is not None:
            self._fleet.node(host_id)           # NotFound(unknown_host) for a host that is not configured
            targets = [host_id]
        else:
            targets = [h.id for h in self._fleet.hosts() if self._state(h.id) == "up"]
        done: list[str] = []
        for hid in targets:
            node = self._fleet.node(hid)
            try:
                await node.call("models.scan", {"force": True}, timeout=self._call_timeout)
            except NodeError as e:
                if host_id is not None:
                    raise problem_from_node(e, node.host.label) from None
                log.info("library: rescan skipped %s (%s)", hid, e.code)
                continue
            done.append(hid)
            self._schedule(hid)
        return done

    async def delete_copy(self, copy_id: str) -> None:
        """Delete one copy from its machine's disk. 404 unknown copy; 503 host not up; 403 the node's policy forbids
        deletes (or the root is read-only); 409 in use (here, or refused by the node)."""
        rec = self._rec(copy_id)
        if rec is None:
            raise NotFound("no such model copy", code="copy_not_found")
        view = self._ctx().views.get(rec.host_id)
        label = view.label if view else rec.host_id
        if view is None or view.state != "up":
            raise problem_from_node(NodeError(rec.host_id, "node_unavailable",
                                         f"the machine is not connected ({view.state if view else 'unknown'})"), label)
        spec = view.facts.root(rec.root_id)
        if not view.facts.allow_delete:
            raise problem_from_node(NodeError(rec.host_id, "policy_denied",
                                         "this machine's policy does not allow deleting models"), label)
        if view.facts.mode != "manage":
            raise problem_from_node(NodeError(rec.host_id, "monitor_only", "this machine's policy is monitor-only"), label)
        if spec is None or not spec.writable:
            raise problem_from_node(NodeError(rec.host_id, "policy_denied",
                                         "that model root is read-only in this machine's policy"), label)
        users = self._users(rec)
        if users:
            raise Conflict(f"the model is used by {', '.join(users)}; stop it first", code="model_in_use")
        if not rec.rel:
            raise Conflict("this entry is a whole model root; the node never deletes a root", code="root_not_deletable")
        try:
            await self._fleet.node(rec.host_id).call("fs.delete", {"root": rec.root_id, "rel": rec.rel}, timeout=300.0)
        except NodeError as e:
            raise problem_from_node(e, label) from None
        await self._forget(rec)
        # Everyone who could see those bytes has to look again: the owner's node and the machines mounting its disk.
        # No refetch is queued here. A scan runs in the background, so a `models.list` right now could still list the
        # deleted copy and bring it back for a moment; the node pushes `models_changed` when its scan sees the change.
        configured = {h.id for h in self._fleet.hosts()}
        for hid in rec.visible_on:
            if hid in configured:
                with contextlib.suppress(NodeError, NotFound):
                    await self._fleet.node(hid).call("models.scan", {"force": True}, timeout=self._call_timeout)

    async def _forget(self, rec: _Rec) -> None:
        """Drop a deleted copy from every host snapshot that listed it, so the library shows the deletion at once
        instead of after the nodes' next scan."""
        touched: list[tuple[str, HostSnapshot]] = []
        for a in rec.aliases:
            snap = self._snaps.get(a.host_id)
            if snap is None:
                continue
            kept = [e for e in snap.entries
                    if not (e.get("root") == a.root_id and ("" if e.get("rel") in (None, ".") else e.get("rel")) == a.rel)]
            if len(kept) != len(snap.entries):
                snap.entries = kept
                touched.append((a.host_id, snap))
        if touched:
            self._rebuild()
            for host_id, snap in touched:
                await asyncio.to_thread(self._persist, host_id, snap)
