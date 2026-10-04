"""Pure builders: what a host's node last said, plus its link state, become the API shapes HostSummary / HostDetail.

Nothing here talks to a node, reads a file or keeps state. Every input is untrusted data that has already been
through `ecnode.proto.sanitize_*` (unknown fields dropped, strings capped, wrong types turned into None), so any
value can still be missing or None: every accessor tolerates that and degrades to "unknown" instead of raising.

Some facts only exist across hosts. Whether host B's model root is really host A's disk (an NFS mount of A) can
only be told by comparing B's mounts with A's addresses, so `build_detail` takes every host's data."""
from __future__ import annotations

import ipaddress
import logging
import posixpath
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from engine_console.domain.fleet import (
    CpuFacts,
    DockerFacts,
    GpuLive,
    GpuProc,
    HostConfig,
    HostCounts,
    HostDetail,
    HostSummary,
    ImageInfo,
    LinkStatus,
    MountInfo,
    NodeFacts,
    PolicyFacts,
    RootInfo,
    ScriptAction,
    ScriptInfo,
)
from engine_console.domain.models import HostMemInfo
from engine_console.services.hostmem import alerts_for

log = logging.getLogger(__name__)
GIB = float(1024**3)
LIVE_STATES = ("up", "stale")           # link states in which a sample still describes the machine
SCRIPT_STATUSES = frozenset({"up", "down", "unknown", "error"})

InstanceResolver = Callable[[str, str], str | None]   # (host_id, container_name) -> console instance id
CountsProvider = Callable[[str], HostCounts]


@dataclass(frozen=True)
class HostData:
    """Everything the hub knows about one host at one moment."""
    host: HostConfig
    link: LinkStatus
    hello: dict[str, Any] | None = None
    sample: dict[str, Any] | None = None
    disks: dict[str, Any] | None = None
    inventory: dict[str, Any] | None = None


# ── tolerant accessors (every one of them takes untrusted, possibly missing data) ───────────────────────────
def _d(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _l(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def _s(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None


def _i(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _f(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _b(v: Any) -> bool | None:
    return v if isinstance(v, bool) else None


def _gib(v: Any) -> float | None:
    n = _i(v)
    return None if n is None else n / GIB


def _abs(v: Any) -> str | None:
    """A normalised absolute POSIX path, or None. Used only to compare paths, never to open anything."""
    s = _s(v)
    return posixpath.normpath(s) if s is not None and s.startswith("/") else None


# ── one host ─────────────────────────────────────────────────────────────────────────────────────────────
def node_facts(hello: dict[str, Any] | None) -> NodeFacts | None:
    if not hello:
        return None
    host, gpu = _d(hello.get("host")), _d(hello.get("gpu"))
    return NodeFacts(
        version=_s(hello.get("version")), hostname=_s(host.get("hostname")), arch=_s(host.get("arch")),
        os=_s(host.get("os")), kernel=_s(host.get("kernel")), cpu_model=_s(host.get("cpu_model")),
        cpu_cores=_i(host.get("cpu_cores")), cpu_threads=_i(host.get("cpu_threads")), python=_s(host.get("python")),
        driver=_s(gpu.get("driver")), cuda=_s(gpu.get("cuda")), gpu_source=_s(gpu.get("source")),
        forced_cmd=_b(hello.get("forced_cmd")), via_ssh=_b(hello.get("via_ssh")))


def docker_facts(hello: dict[str, Any] | None) -> DockerFacts | None:
    if not hello:
        return None
    d = _d(hello.get("docker"))
    return DockerFacts(available=d.get("available") is True, context=_s(d.get("context")),
                       rootless=_b(d.get("rootless")), version=_s(d.get("version")),
                       nvidia_runtime=_b(d.get("nvidia_runtime")), error=_s(d.get("error")))


def policy_facts(hello: dict[str, Any] | None) -> PolicyFacts | None:
    if not hello:
        return None
    p = _d(hello.get("policy"))
    pr = [x for x in _l(p.get("port_range")) if _i(x) is not None]
    ipc = _b(p.get("allow_ipc_host"))
    ext = _s(p.get("external_control"))
    return PolicyFacts(
        path=_s(p.get("path")),
        mode="manage" if p.get("mode") == "manage" else "monitor",   # anything unexpected means read-only
        error=_s(p.get("error")), engine_images=[x for x in _l(p.get("engine_images")) if isinstance(x, str)],
        gateway_image=_s(p.get("gateway_image")), engine_network=_s(p.get("engine_network")),
        port_range=pr if len(pr) == 2 else [], allow_pull=p.get("allow_pull") is True,
        allow_delete=p.get("allow_delete") is True, allow_ipc_host=True if ipc is None else ipc,
        external_control="stop_start" if ext == "stop_start" else "none", max_shm_gib=_f(p.get("max_shm_gib")))


def cpu_facts(sample: dict[str, Any] | None) -> CpuFacts | None:
    if not sample:
        return None
    c = _d(sample.get("cpu"))
    return CpuFacts(load1=_f(c.get("load1")), load5=_f(c.get("load5")), load15=_f(c.get("load15")),
                    util_pct=_f(c.get("util_pct")))


def host_memory(sample: dict[str, Any] | None, link: LinkStatus) -> HostMemInfo | None:
    """HostMemInfo from the sample's /proc/meminfo numbers (bytes). `used` is total - available, the same rule as
    the local /hardware report, so alerts mean the same thing on every machine."""
    if not sample:
        return None
    m = _d(sample.get("mem"))
    total, avail = _i(m.get("total")), _i(m.get("available"))
    if not total or total <= 0 or avail is None:
        return None
    swap_total = _i(m.get("swap_total")) or 0
    swap_free = _i(m.get("swap_free")) or 0
    t = _f(sample.get("t"))
    # the node's clock minus the measured skew is the hub's clock, which is what a browser compares against
    updated = t - (link.skew_s or 0.0) if t is not None else (link.last_seen or 0.0)
    return HostMemInfo(
        total_gib=total / GIB, used_gib=max(total - avail, 0) / GIB, available_gib=avail / GIB,
        free_gib=(_i(m.get("free")) or 0) / GIB, cached_gib=(_i(m.get("cached")) or 0) / GIB,
        shmem_gib=(_i(m.get("shmem")) or 0) / GIB, swap_total_gib=swap_total / GIB,
        swap_used_gib=max(swap_total - swap_free, 0) / GIB, updated_at=updated)


def _instance_of(resolver: InstanceResolver | None, host_id: str, container: str | None) -> str | None:
    if resolver is None or container is None:
        return None
    try:
        return resolver(host_id, container)
    except Exception:   # noqa: BLE001 - a resolver bug must not take the host list down
        log.warning("instance resolver failed for host %s", host_id, exc_info=True)
        return None


def _gpu_order(g: dict[str, Any]) -> int:
    idx = _i(g.get("index"))
    return idx if idx is not None else 10**6          # a GPU with no index goes last


def build_gpus(d: HostData, sample: dict[str, Any] | None, resolver: InstanceResolver | None) -> list[GpuLive]:
    """Static facts from hello, live numbers from the sample (keyed by UUID), bytes -> GiB. Without a hello there
    is nothing to list: the hub only ever shows GPUs the node announced."""
    live = _d(_d(sample).get("gpus"))
    out: list[GpuLive] = []
    for g in sorted(map(_d, _l(_d(d.hello).get("gpus"))), key=_gpu_order):
        uuid = _s(g.get("uuid"))
        if uuid is None:
            continue
        s = _d(live.get(uuid))
        total = _gib(g.get("total_bytes")) or _gib(s.get("mem_total_bytes")) or 0.0
        used = _gib(s.get("mem_used_bytes"))
        procs = [GpuProc(pid=pid, name=_s(p.get("name")), container=_s(p.get("container")),
                         instance_id=_instance_of(resolver, d.host.id, _s(p.get("container"))),
                         used_gib=_gib(p.get("used_bytes")))
                 for p in map(_d, _l(s.get("procs"))) if (pid := _i(p.get("pid"))) is not None]
        out.append(GpuLive(
            index=_i(g.get("index")) or 0, uuid=uuid, name=_s(g.get("name")) or "GPU", total_gib=total, used_gib=used,
            free_gib=None if used is None else max(total - used, 0.0), util_pct=_f(s.get("util_pct")),
            temp_c=_f(s.get("temp_c")), power_w=_f(s.get("power_w")), power_limit_w=_f(s.get("power_limit_w")),
            fan_pct=_f(s.get("fan_pct")), compute_capability=_s(g.get("compute_capability")),
            unified_memory=g.get("unified_memory") is True, procs=procs))
    return out


def _counts(provider: CountsProvider | None, host_id: str) -> HostCounts:
    if provider is None:
        return HostCounts()
    try:
        return provider(host_id)
    except Exception:   # noqa: BLE001 - same reason as the resolver
        log.warning("counts provider failed for host %s", host_id, exc_info=True)
        return HostCounts()


def build_summary(d: HostData, *, resolver: InstanceResolver | None = None,
                  counts: CountsProvider | None = None) -> HostSummary:
    # A sample from before a link drop describes a machine we can no longer see: show it as unknown, not as live.
    sample = d.sample if d.link.state in LIVE_STATES else None
    mem = host_memory(sample, d.link)
    return HostSummary(
        id=d.host.id, label=d.host.label, remote=d.host.ssh is not None, link=d.link, node=node_facts(d.hello),
        gpus=build_gpus(d, sample, resolver), memory=mem, cpu=cpu_facts(sample), docker=docker_facts(d.hello),
        policy=policy_facts(d.hello), counts=_counts(counts, d.host.id), alerts=alerts_for(mem) if mem else [])


# ── disks, roots and storage shared between hosts ────────────────────────────────────────────────────────
def _under(path: str, base: str) -> bool:
    """`path` is `base` or inside it, compared by whole path components ("/data/hf2" is not under "/data/hf")."""
    if base == "/":
        return True
    return path == base or path.startswith(base + "/")


def _overlap(a: str, b: str) -> bool:
    return _under(a, b) or _under(b, a)


def _norm_addr(v: Any) -> str | None:
    s = _s(v)
    if s is None:
        return None
    s = s.strip().strip("[]").split("%", 1)[0].lower()
    try:
        return str(ipaddress.ip_address(s))
    except ValueError:
        return s


def _is_loopback(addr: str) -> bool:
    try:
        return ipaddress.ip_address(addr).is_loopback
    except ValueError:
        return addr in ("localhost", "localhost.localdomain")


class AddressBook:
    """Which configured host owns an address or name. An address that two hosts both report (a docker bridge such
    as 172.17.0.1 exists on every machine with docker) identifies nobody, so it is left out."""

    def __init__(self, fleet: Mapping[str, HostData]) -> None:
        by_addr: dict[str, set[str]] = {}
        by_name: dict[str, set[str]] = {}
        for hid, d in fleet.items():
            for a in map(_d, _l(_d(d.hello).get("addresses"))):
                addr = _norm_addr(a.get("addr"))
                if addr and not _is_loopback(addr):
                    by_addr.setdefault(addr, set()).add(hid)
            name = (_s(_d(_d(d.hello).get("host")).get("hostname")) or "").lower()
            if name and not _is_loopback(name):
                by_name.setdefault(name, set()).add(hid)
                by_name.setdefault(name.split(".", 1)[0], set()).add(hid)
        self._addr = {k: next(iter(v)) for k, v in by_addr.items() if len(v) == 1}
        self._name = {k: next(iter(v)) for k, v in by_name.items() if len(v) == 1}

    def find(self, server: str | None, exclude: str) -> str | None:
        """The configured host an NFS server address or name belongs to (never `exclude`: a machine mounting its
        own export is not borrowing someone else's disk)."""
        key = _norm_addr(server)
        if not key or _is_loopback(key):
            return None
        hid = self._addr.get(key) or self._name.get(key)
        return hid if hid is not None and hid != exclude else None


def mount_label(mount: str, fstype: str | None, source: str | None, model: str | None, remote: bool,
                server: str | None, server_label: str | None) -> str:
    """The human name of a filesystem: "tank (zfs)", "NFS from gpu-box-2", "/ (ext4, Samsung 990)"."""
    ft = (fstype or "").lower()
    if ft == "zfs":
        pool = (source or "").split("/", 1)[0].split("@", 1)[0]
        return f"{pool or mount} (zfs)"
    if ft.startswith("nfs") or remote:
        kind = "NFS" if ft.startswith("nfs") else ("CIFS" if ft in ("cifs", "smb3", "smbfs") else (ft.upper() or "Network"))
        return f"{kind} from {server_label or server or source or 'another machine'}"
    if model:
        return f"{mount} ({fstype}, {model})" if fstype else f"{mount} ({model})"
    return f"{mount} ({fstype})" if fstype else mount


def _mount_info(m: dict[str, Any], d: HostData, book: AddressBook, labels: Mapping[str, str],
                roots: list[str]) -> MountInfo:
    mount = _s(m.get("mount")) or "?"
    fstype, source, model = _s(m.get("fstype")), _s(m.get("source")), _s(m.get("model"))
    remote = m.get("remote") is True
    server = _s(m.get("remote_host"))
    owner = book.find(server, d.host.id) if remote else None
    return MountInfo(
        mount=mount, label=mount_label(mount, fstype, source, model, remote, server, labels.get(owner or "")),
        fstype=fstype, source=source, device=_s(m.get("device")), model=model, size_bytes=_i(m.get("size")),
        used_bytes=_i(m.get("used")), free_bytes=_i(m.get("free")), readonly=m.get("readonly") is True,
        remote=remote, remote_host=server, remote_host_id=owner, remote_path=_s(m.get("remote_path")), roots=roots)


def _mount_of_root(root_id: str, root_path: str, mounts: Sequence[dict[str, Any]]) -> int | None:
    """Index of the mount a model root lives on: the one the node says holds it, else the longest mount path that
    is a prefix of the root's path."""
    claimed = [i for i, m in enumerate(mounts) if root_id in _l(m.get("roots"))]
    if len(claimed) == 1:
        return claimed[0]
    best, best_len = None, -1
    for i, m in enumerate(mounts):
        mp = _abs(m.get("mount"))
        if mp is not None and _under(root_path, mp):
            n = 0 if mp == "/" else mp.count("/")
            if n > best_len:
                best, best_len = i, n
    return best


@dataclass(frozen=True)
class _RootLoc:
    host_id: str
    root_id: str
    owner: str        # the host whose disk this really is
    path: str         # the root's path expressed on the owner (a root that is an NFS mount is translated)
    imported: bool


def _root_locations(fleet: Mapping[str, HostData], book: AddressBook) -> list[_RootLoc]:
    locs: list[_RootLoc] = []
    for hid, d in fleet.items():
        mounts = [m for m in map(_d, _l(_d(d.disks).get("mounts")))]
        for root in map(_d, _l(_d(_d(d.hello).get("policy")).get("model_roots"))):
            rid, rpath = _s(root.get("id")), _abs(root.get("path"))
            if rid is None or rpath is None:
                continue
            owner, path, imported = hid, rpath, False
            mi = _mount_of_root(rid, rpath, mounts)
            m = mounts[mi] if mi is not None else {}
            mp, export = _abs(m.get("mount")), _abs(m.get("remote_path"))
            server = book.find(_s(m.get("remote_host")), hid) if (_s(m.get("fstype")) or "").lower().startswith("nfs") else None
            if server is not None and mp is not None and export is not None and _under(rpath, mp):
                rel = posixpath.relpath(rpath, mp)
                owner, imported = server, True
                path = export if rel == "." else posixpath.normpath(posixpath.join(export, rel))
            locs.append(_RootLoc(hid, rid, owner, path, imported))
    return locs


def _shares(fleet: Mapping[str, HostData], book: AddressBook) -> dict[tuple[str, str], tuple[str | None, list[str]]]:
    """(host_id, root_id) -> (shared_from, visible_on). Two roots are the same storage when they resolve to the
    same owner host and their paths on it overlap. `visible_on` lists this host first, then the owner, then the
    rest in configuration order; `shared_from` is set only for a root that is another configured host's disk AND
    overlaps one of that host's own roots (otherwise that host's library would not list what this one sees)."""
    order = list(fleet)
    locs = _root_locations(fleet, book)
    out: dict[tuple[str, str], tuple[str | None, list[str]]] = {}
    for loc in locs:
        peers = [o for o in locs if o.owner == loc.owner and _overlap(o.path, loc.path)]
        hosts_seen = {o.host_id for o in peers}
        hosts_seen.add(loc.host_id)
        visible = [loc.host_id]
        if loc.owner in hosts_seen and loc.owner != loc.host_id:
            visible.append(loc.owner)
        visible += [h for h in order if h in hosts_seen and h not in visible]
        shared = loc.owner if loc.imported and any(o.host_id == loc.owner and not o.imported for o in peers) else None
        out[(loc.host_id, loc.root_id)] = (shared, visible)
    return out


# ── scripts, images, addresses ───────────────────────────────────────────────────────────────────────────
def build_scripts(d: HostData) -> list[ScriptInfo]:
    """Scripts from the node's policy (names, labels, allowed args: never the argv) with the last status the node
    reported for each in its inventory."""
    status = {n: st for st in map(_d, _l(_d(d.inventory).get("scripts"))) if (n := _s(st.get("name"))) is not None}
    out: list[ScriptInfo] = []
    for s in map(_d, _l(_d(_d(d.hello).get("policy")).get("scripts"))):
        name = _s(s.get("name"))
        if name is None:
            continue
        st = status.get(name, {})
        raw_status = _s(st.get("status"))
        actions = [ScriptAction(name=an, label=_s(a.get("label")) or an,
                                args=[x for x in _l(a.get("args")) if isinstance(x, str)],
                                confirm=a.get("confirm") is True, danger=a.get("danger") is True)
                   for a in map(_d, _l(s.get("actions"))) if (an := _s(a.get("name"))) is not None]
        out.append(ScriptInfo(
            host_id=d.host.id, name=name, label=_s(s.get("label")) or name, engine=_s(s.get("engine")),
            description=_s(s.get("description")) or "", status=raw_status if raw_status in SCRIPT_STATUSES else "unknown",
            detail=_s(st.get("detail")), checked_at=_f(st.get("checked_at")), actions=actions,
            containers=[x for x in _l(s.get("containers")) if isinstance(x, str)]))
    return out


def images_from_result(result: Mapping[str, Any]) -> list[ImageInfo]:
    """ImageInfo list from a sanitized `docker.images` result."""
    return [ImageInfo(ref=ref, id=_s(i.get("id")), size_bytes=_i(i.get("size")), created=_f(i.get("created")),
                      arch=_s(i.get("arch")))
            for i in map(_d, _l(result.get("images"))) if (ref := _s(i.get("ref"))) is not None]


def addresses_of(hello: dict[str, Any] | None) -> list[str]:
    """"10.0.0.5/24 (eth0)" for every non-loopback interface address the node reported."""
    out: list[str] = []
    for a in map(_d, _l(_d(hello).get("addresses"))):
        addr, prefix, iface = _s(a.get("addr")), _i(a.get("prefix")), _s(a.get("iface"))
        norm = _norm_addr(addr)
        if addr is None or norm is None or _is_loopback(norm):
            continue
        out.append(f"{addr}{f'/{prefix}' if prefix is not None else ''}{f' ({iface})' if iface else ''}")
    return out


# ── the detail view ──────────────────────────────────────────────────────────────────────────────────────
def build_detail(host_id: str, fleet: Mapping[str, HostData], *, images: Sequence[ImageInfo] = (),
                 resolver: InstanceResolver | None = None, counts: CountsProvider | None = None) -> HostDetail:
    """The detail of `host_id`; `fleet` holds every configured host's data (including this one), which is how a
    mount is matched to the host that exports it. `images` come from a separate (cached) `docker.images` call:
    they are not part of the node's pushed inventory."""
    d = fleet[host_id]
    summary = build_summary(d, resolver=resolver, counts=counts)
    book = AddressBook(fleet)
    labels = {hid: x.host.label for hid, x in fleet.items()}
    raw_mounts = [m for m in map(_d, _l(_d(d.disks).get("mounts")))]
    root_specs = [r for r in map(_d, _l(_d(_d(d.hello).get("policy")).get("model_roots")))
                  if _s(r.get("id")) and _abs(r.get("path"))]
    root_mount: dict[str, int | None] = {str(r["id"]): _mount_of_root(str(r["id"]), str(_abs(r.get("path"))), raw_mounts)
                                         for r in root_specs}
    mounts = [_mount_info(m, d, book, labels, [rid for rid, mi in root_mount.items() if mi == i])
              for i, m in enumerate(raw_mounts)]
    shares = _shares(fleet, book)
    roots: list[RootInfo] = []
    for r in root_specs:
        rid = str(r["id"])
        mi = root_mount[rid]
        shared_from, visible = shares.get((host_id, rid), (None, [host_id]))
        roots.append(RootInfo(id=rid, path=str(r.get("path")), kind=_s(r.get("kind")) or "auto",
                              writable=r.get("writable") is True, label=_s(r.get("label")),
                              mount=mounts[mi] if mi is not None else None, shared_from=shared_from,
                              visible_on=visible))
    return HostDetail(
        **{name: getattr(summary, name) for name in HostSummary.model_fields},
        disks=mounts, roots=roots, images=list(images), scripts=build_scripts(d), addresses=addresses_of(d.hello))
