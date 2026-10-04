"""Instance lifecycle across the fleet: preflight -> node `docker.run` + `gateway.start` -> a supervised state machine
stopped -> starting -> loading -> ready -> stopping | failed.

Contract v2 (docs/ARCHITECTURE.md §10): every docker action is an operation on the node of the host the instance
runs on; the hub never runs docker itself. Three classes of instance are served from here:
  * managed  - containers the console created (rows in the `instances` table, full control);
  * script   - an engine started by a script in the node's policy (derived from the inventory; start/stop run the
               script's `up`/`down`);
  * external - any other engine container (derived; monitor-only unless the node policy allows plain stop/start).

`tick()` is the supervisor step (called every couple of seconds by a background task, or directly in tests). It
reads container state from the host's pushed inventory and only asks the node (`docker.inspect`) when a container
is missing from an inventory that may not have caught up yet."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from engine_console.adapters import AdapterRegistry
from engine_console.adapters.base import EngineAdapter, LaunchContext, LaunchSpec, ModelInfo
from engine_console.config import Settings
from engine_console.domain.errors import BadRequest, Conflict, NotFound, ProblemError, Unprocessable
from engine_console.domain.fleet import FleetPort, HostCounts, LibraryCopy, NodeClient, NodeError
from engine_console.domain.models import (
    CommandSnippets,
    FitReport,
    HostMemory,
    Instance,
    InstanceCreate,
    InstancePatch,
    InstanceState,
    Page,
    PreflightCheck,
    PreflightReport,
    ResidentUse,
)
from engine_console.domain.refs import InstanceControls, InstanceVram, ModelLocation, ModelRef
from engine_console.domain.repo_id import check_repo_id
from engine_console.services import nodeview as nv
from engine_console.services.common import EventBus, new_id, now, scrub
from engine_console.services.discovery import EXT, SCRIPT
from engine_console.services.docker import command_snippets, gateway_name
from engine_console.services.externals import (
    HostCtx,
    build_host_instances,
    external_controls,
    external_instance_id,
    matches_script,
    memory_for,
    script_instance_id,
    vram_for,
)
from engine_console.services.fitting import FitService
from engine_console.services.hardware import HardwareService
from engine_console.services.libport import LibraryPort, NullLibrary
from engine_console.services.nodeerr import NodeCaller, shielded
from engine_console.services.nodehttp import NodeHttp
from engine_console.services.runspec import (
    build_run_spec,
    check_ref,
    display_of,
    engine_labels,
    hf_mounts,
    model_mounts,
    rel_ok,
    split_env,
    validate_run_spec,
)
from engine_console.services.scripts import ScriptService
from engine_console.services.secrets_store import MARKER, SECRET_KEY, SecretStore, split_secrets
from engine_console.services.settings import SettingsService
from engine_console.services.store import Store, paginate

log = logging.getLogger(__name__)
# compile/JIT caches survive container recreation; named volumes need no host path. An adapter may declare its own
# with a `named_volumes` attribute ([(volume name, container path)]).
NAMED_VOLUMES: dict[str, list[tuple[str, str]]] = {
    "vllm": [("ec-vllm-cache", "/root/.cache/vllm"), ("ec-flashinfer-cache", "/root/.cache/flashinfer")],
}
ACTIVE: tuple[str, ...] = ("starting", "loading", "ready", "stopping")
RUNNING: tuple[str, ...] = ("starting", "loading", "ready", "auth_required")
UNAVAILABLE_CODES = ("node_unavailable", "timeout", "busy")   # the node could not act: the instance's state did not change
TOPIC = "instances"
INVENTORY_LAG_S = 40.0     # the node pushes its inventory on change and at least every 30 s: a younger gap is not proof
GATEWAY_ATTEMPTS = 8       # ports tried when gateway.start answers port_in_use
GONE_TTL_S = 30.0          # a vanished external engine shows `stopped` this long
_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.\-]{0,63}")
_ROOT_RE = re.compile(r"[A-Za-z0-9_.\-]{1,32}")
_LABEL_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.:/@\-]{0,254}")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "model"


def _safe_name(s: str) -> str:
    """A derived instance name that satisfies InstanceName (it ends up in labels and is re-read at adoption)."""
    out = re.sub(r"[^A-Za-z0-9 _.\-]+", "-", s).strip(" .-_")[:64].strip(" .-_")
    return out if _NAME_RE.fullmatch(out) else "model"


def _needs_copy(ref: ModelRef) -> bool:
    """An Ollama server with no model named is a bare server; everything else must exist in the host's library."""
    return ref.format != "ollama" or bool(ref.name)


def _mask(text: str, secrets: list[str]) -> str:
    for sv in secrets:
        text = text.replace(sv, "[redacted]")
    return text


# Parsed command-line keys (inventory `params`, display-only parse of the redacted argv) that name weights an engine
# loads: the model itself, a draft model for speculative decoding, a separate tokenizer or vision projector.
_MODEL_KEYS = ("model", "model_path", "model_id", "model_name_or_path", "speculative_model", "speculative_draft_model_path",
               "draft_model", "tokenizer", "tokenizer_path", "mmproj")
_HF_DIR = re.compile(r"models--([^/]+?)--([^/]+)")


def _loads_model(container: dict[str, Any], path: str) -> bool:
    """Does a container that mounts a folder holding `path` load that model? True when it cannot be told (no model
    argument parsed), so a delete is refused rather than pulled from under a running engine."""
    params = container.get("params") or {}
    values = [str(params[k]).strip().rstrip("/").lower() for k in _MODEL_KEYS if params.get(k)]
    label = (container.get("labels") or {}).get("engine-console.model")
    if label:
        values.append(str(label).strip().rstrip("/").lower())
    if not values:
        return True
    target = path.rstrip("/").lower()
    hf = _HF_DIR.search(target)
    repo, folder = (f"{hf.group(1)}/{hf.group(2)}", hf.group(0)) if hf else (None, None)
    base = target.rsplit("/", 1)[-1]
    for v in values:
        if v == target or v.startswith(target + "/"):
            return True                              # the same host path (mounted at the same place)
        if repo and (v == repo or (folder and folder in v)):
            return True                              # the repo id, or a path into that repo's cache folder
        if not repo and v.rsplit("/", 1)[-1] == base:
            return True                              # a plain folder mounted somewhere else in the container
    return False


@dataclass
class Plan:
    """Everything that identifies one launch of a managed instance (secrets resolved: memory only, never persisted)."""
    iid: str
    name: str
    engine: str
    host_id: str
    ref: ModelRef
    params: dict[str, Any]
    gpu_ids: list[int]
    port: int
    container: str
    image_req: str | None = None

    @property
    def display(self) -> str:
        return display_of(self.ref)


@dataclass
class Built:
    run: dict[str, Any]                     # the docker.run spec WITHOUT secret values
    secret_env: dict[str, str]              # name -> value; sent only inside the docker.run request
    launch: LaunchSpec
    image: str
    gpu_ids: list[int]
    gpu_uuids: list[str]
    location: ModelLocation | None
    secret_values: list[str] = field(default_factory=list)


class LifecycleService:
    def __init__(self, store: Store, adapters: AdapterRegistry, fleet: FleetPort, hardware: HardwareService,
                 settings: SettingsService, fit: FitService, http: NodeHttp, cfg: Settings, bus: EventBus, *,
                 library: LibraryPort | None = None, scripts: ScriptService | None = None,
                 secrets: SecretStore | None = None) -> None:
        self._db = store
        self._adapters = adapters
        self._fleet = fleet
        self._nodes = NodeCaller(fleet)
        self._hw = hardware
        self._settings = settings
        self._fit = fit
        self._http = http
        self._cfg = cfg
        self._bus = bus
        self._library: LibraryPort = library or NullLibrary()
        self._scripts = scripts
        self._secrets = secrets or SecretStore(cfg.data_dir / "secrets")
        self._locks: dict[str, asyncio.Lock] = {}
        self._built: dict[str, Built] = {}                       # launch info per managed instance (in memory)
        # derived (script / external) instances, all keyed by host so one host's summary only rebuilds that host
        self._first_seen: dict[str, dict[str, float]] = {}       # when each derived id first appeared
        self._prev_ext: dict[str, dict[str, Instance]] = {}      # external engines seen running at the last look
        self._gone: dict[str, dict[str, tuple[float, Instance]]] = {}   # vanished ones, shown stopped for a while
        self._stopped_ext: dict[str, dict[str, Instance]] = {}   # externals the hub itself stopped (so they can be started)
        self._ext_by_container: dict[str, dict[str, str]] = {}   # container name -> derived instance id
        self._links: dict[str, tuple[bool, float | None]] = {}
        fit.set_resident_provider(self.resident)
        fit.set_library(self._library)

    def set_library(self, library: LibraryPort) -> None:
        self._library = library
        self._fit.set_library(library)

    def set_scripts(self, scripts: ScriptService) -> None:
        self._scripts = scripts

    # -- hosts ---------------------------------------------------------------------------------------------------
    def _lock(self, host_id: str) -> asyncio.Lock:
        return self._locks.setdefault(host_id, asyncio.Lock())

    def _others(self) -> dict[str, tuple[str, frozenset[str]]]:
        """Every host's label and addresses, to tell which configured machine an NFS export comes from."""
        return {h.id: (h.label, frozenset(nv.addresses(self._fleet.node(h.id)))) for h in self._fleet.hosts()}

    def _ctx(self, host_id: str, others: dict[str, tuple[str, frozenset[str]]] | None = None) -> HostCtx:
        node = self._nodes.node(host_id)
        return HostCtx(host_id=host_id, label=node.host.label, node=node, others=others if others is not None else self._others())

    def _label(self, host_id: str) -> str:
        return self._nodes.label(host_id)

    # -- persistence ---------------------------------------------------------------------------------------------
    @staticmethod
    def _ref_of(r: Any) -> ModelRef:
        if r["model"]:
            with contextlib.suppress(ValueError):
                return ModelRef.model_validate_json(r["model"])
        return ModelRef(format="hf", repo_id=r["repo_id"])

    def _managed_controls(self, host_id: str) -> InstanceControls:
        try:
            node = self._nodes.node(host_id)
        except NotFound:
            return InstanceControls(reason=f"host {host_id} is not configured any more")
        label = node.host.label
        if not nv.link_up(node):
            return InstanceControls(reason=f"{label} is not connected: {node.link().error or node.link().state}")
        if nv.mode(node) != "manage":
            return InstanceControls(logs=True, reason=f"{label}'s node is in monitor mode: engines there can be watched but "
                                                      "not changed (its operator switches the policy mode to manage)")
        return InstanceControls(start=True, stop=True, restart=True, remove=True, logs=True, configure=True)

    def _to_model(self, r: Any) -> Instance:
        host_id = r["host_id"]
        state: InstanceState = r["state"]
        reason: str | None = None
        names = {r["container_name"]}
        vram: list[InstanceVram] = []
        mem: HostMemory | None = None
        label = host_id
        try:
            node = self._nodes.node(host_id)
            label = node.host.label
            if state in ACTIVE:
                if not nv.link_up(node):
                    state, reason = "unreachable", f"{label} is not connected: {node.link().error or node.link().state}"
                else:
                    vram, mem = vram_for(node, names), memory_for(node, names)
        except NotFound:
            if state in ACTIVE:
                state, reason = "unreachable", f"host {host_id} is not configured any more"
        started = r["started_at"]
        up = now() - started if started and r["state"] in ("starting", "loading", "ready") else None
        ref = self._ref_of(r)
        return Instance(
            id=r["id"], name=r["name"], engine=r["engine"], repo_id=r["repo_id"], params=json.loads(r["params"]),
            gpu_ids=json.loads(r["gpu_ids"]), gpu_uuids=json.loads(r["gpu_uuids"]), port=r["port"],
            container_name=r["container_name"], container_port=r["container_port"],
            internal_endpoint=(f"http://{r['container_name']}:{r['container_port']}" if r["container_port"] else None),
            image=r["image"], state=state, phase=r["phase"], progress_pct=r["progress_pct"], pinned=bool(r["pinned"]),
            ttl_idle_s=r["ttl_idle_s"], profile_id=r["profile_id"], error=r["error"], last_logs=r["last_logs"],
            fit=json.loads(r["fit"]) if r["fit"] else None, created_at=r["created_at"], started_at=started,
            last_request_at=r["last_request_at"], uptime_s=up, state_reason=reason,
            endpoint=f"http://127.0.0.1:{r['port']}" if r["port"] else None, host_memory=mem, host_id=host_id,
            host_label=label, management="managed", vram=vram, model=ref, model_location=self._location_of(host_id, ref),
            controls=self._managed_controls(host_id))

    def _location_of(self, host_id: str, ref: ModelRef) -> ModelLocation | None:
        copy = self._library.find(host_id, ref) if _needs_copy(ref) else None
        return self._location_from_copy(copy) if copy else None

    @staticmethod
    def _location_from_copy(copy: LibraryCopy) -> ModelLocation:
        d = copy.drive
        return ModelLocation(host_id=copy.host_id, root_id=copy.root_id, root_label=copy.root_label, path=copy.path,
                             drive=d.label if d else None, fstype=d.fstype if d else None, remote=bool(d and d.remote),
                             shared_from=(d.remote_host_id or d.remote_host) if d and d.remote else None)

    def _row(self, iid: str) -> Any:
        return self._db.one("SELECT * FROM instances WHERE id=?", (iid,))

    # -- derived (script / external) instances --------------------------------------------------------------------
    def _busy(self, host_id: str) -> dict[str, str]:
        return self._scripts.busy(host_id) if self._scripts else {}

    def externals(self, host_id: str | None = None) -> list[Instance]:
        """Script and external instances of every host (or one), from the pushed inventories. A vanished external engine
        stays listed as `stopped` for a short while, and one the hub itself stopped stays so it can be started again."""
        t = now()
        others = self._others()
        out: list[Instance] = []
        for h in self._fleet.hosts():
            if host_id is None or h.id == host_id:
                out += self._externals_of(self._ctx(h.id, others), t)
        return out

    def _externals_of(self, ctx: HostCtx, t: float) -> list[Instance]:
        hid = ctx.host_id
        seen = self._first_seen.setdefault(hid, {})
        current: dict[str, Instance] = {}
        by_container: dict[str, str] = {}
        for inst in build_host_instances(ctx, seen, t, self._busy(hid)):
            current[inst.id] = self._overlay_link(inst, ctx.node)
            if inst.container_name:
                by_container[inst.container_name] = inst.id
        containers = nv.containers(ctx.node)
        for s in nv.scripts(ctx.node):   # every container of a script's engine maps to the script's instance
            for c in containers:
                if matches_script(str(c["name"]), s):
                    by_container[str(c["name"])] = script_instance_id(hid, str(s["name"]))
        self._ext_by_container[hid] = by_container
        gone = self._gone.setdefault(hid, {})
        stopped = self._stopped_ext.setdefault(hid, {})
        prev = self._prev_ext.get(hid, {})
        if nv.link_up(ctx.node):
            for iid, inst in prev.items():   # an external engine that was running and is not any more
                if iid not in current and inst.state != "stopped":
                    gone[iid] = (t + GONE_TTL_S, self._as_stopped(inst, ctx))
        self._prev_ext[hid] = {i: v for i, v in current.items() if v.management == "external"}
        for iid in [i for i, (until, _) in gone.items() if until < t or i in current]:
            del gone[iid]
        for iid in [i for i in stopped if i in current]:
            del stopped[iid]    # the container is running again
        for iid in [i for i in seen if i not in current and i not in gone and i not in stopped]:
            del seen[iid]
        extra = [inst for _, inst in gone.values()] + [inst for iid, inst in stopped.items() if iid not in gone]
        return [*current.values(), *extra]

    def _connected(self, host_id: str) -> bool:
        try:
            return nv.link_up(self._nodes.node(host_id))
        except NotFound:
            return False

    @staticmethod
    def _overlay_link(inst: Instance, node: NodeClient) -> Instance:
        """What was last pushed by a node we cannot reach any more is not live: show it as unreachable."""
        if nv.link_up(node) or inst.state in ("stopped", "unreachable"):
            return inst
        link = node.link()
        return inst.model_copy(update={"state": "unreachable",
                                       "state_reason": f"{inst.host_label} is not connected: {link.error or link.state}"})

    def _as_stopped(self, inst: Instance, ctx: HostCtx | None = None) -> Instance:
        return inst.model_copy(update={"state": "stopped", "state_reason": "container is no longer running", "vram": [],
                                       "host_memory": None,
                                       "controls": external_controls(ctx or self._ctx(inst.host_id), "stopped")})

    # -- queries -------------------------------------------------------------------------------------------------
    def get(self, iid: str) -> Instance:
        if iid.startswith((EXT, SCRIPT)):
            inst = next((i for i in self.externals() if i.id == iid), None)
            if inst is None:
                raise NotFound(f"no such instance: {iid}")
            return inst
        r = self._row(iid)
        if r is None:
            raise NotFound(f"no such instance: {iid}")
        return self._to_model(r)

    def _instances_on(self, host_id: str) -> list[Instance]:
        rows = self._db.all("SELECT * FROM instances WHERE host_id=?", (host_id,))
        return [self._to_model(r) for r in rows] + self.externals(host_id)

    def list_page(self, limit: int = 100, cursor: str | None = None, host_id: str | None = None) -> Page[Instance]:
        limit, off = paginate(limit, cursor)
        where, args = ("WHERE host_id=?", [host_id]) if host_id else ("", [])
        rows = self._db.all(f"SELECT * FROM instances {where} ORDER BY created_at DESC", args)  # noqa: S608
        items = [self._to_model(r) for r in rows] + self.externals(host_id)
        return Page[Instance](items=items[off: off + limit], next_cursor=str(off + limit) if len(items) > off + limit else None)

    def all_instances(self) -> list[Instance]:
        """Every instance of every host (managed, script, external)."""
        return [self._to_model(r) for r in self._db.all("SELECT * FROM instances")] + self.externals()

    def all_active(self) -> list[Instance]:
        return [self._to_model(r) for r in self._db.all(
            "SELECT * FROM instances WHERE state IN ('starting','loading','ready','stopping')")]

    def lookup(self, host_id: str, *, port: int | None = None, container: str | None = None) -> Instance | None:
        """The instance of a host that owns a gateway/published port or a container name (metrics routing)."""
        for r in self._db.all("SELECT * FROM instances WHERE host_id=? AND state IN ('starting','loading','ready')", (host_id,)):
            if (port is not None and r["port"] == port) or (container and container in (r["container_name"], gateway_name(r["container_name"]))):
                return self._to_model(r)
        for i in self.externals(host_id):
            if (port is not None and i.port == port) or (container and i.container_name == container):
                return i
        return None

    def instance_for_container(self, host_id: str, container: str) -> str | None:
        """The console instance id behind a container name on a host (to label GPU processes); None if unknown."""
        r = self._db.one("SELECT id FROM instances WHERE host_id=? AND container_name=?", (host_id, container))
        if r is not None:
            return str(r["id"])
        known = self._ext_by_container.get(host_id, {}).get(container)
        if known:
            return known
        node = self._nodes.node(host_id)
        s = next((s for s in nv.scripts(node) if matches_script(container, s)), None)
        if s is not None:
            return script_instance_id(host_id, str(s["name"]))
        c = nv.container_named(node, container)
        return external_instance_id(host_id, container) if c and c.get("engine") and not c.get("managed") else None

    def counts(self, host_id: str) -> HostCounts:
        insts = self._instances_on(host_id)
        copies = self._library.copies(host_id)
        return HostCounts(engines_running=sum(1 for i in insts if i.state in RUNNING), engines_total=len(insts),
                          models=len(copies), models_bytes=sum(c.size_bytes for c in copies))

    def require_managed(self, iid: str) -> Instance:
        inst = self.get(iid)
        if inst.management != "managed":
            what = ("a script's engine: use its start/stop/restart (they run the script)" if inst.management == "script"
                    else "an external engine the console does not own (monitor-only)")
            raise Conflict(f"this is {what}", code="instance_not_managed")
        return inst

    @staticmethod
    def _cn(inst: Instance) -> str:
        if inst.container_name is None:
            raise Conflict("this instance has no container", code="instance_not_managed")
        return inst.container_name

    def touch(self, iid: str) -> None:
        self._db.execute("UPDATE instances SET last_request_at=? WHERE id=?", (now(), iid))

    def _set(self, iid: str, **cols: Any) -> None:
        for k in ("params", "gpu_ids", "gpu_uuids", "fit", "model"):
            if k in cols and not isinstance(cols[k], str) and cols[k] is not None:
                cols[k] = json.dumps(cols[k])
        sets = ", ".join(f"{k}=?" for k in cols)   # column names are literals from this module
        self._db.execute(f"UPDATE instances SET {sets} WHERE id=?", [*cols.values(), iid])  # noqa: S608
        self._bus.publish(TOPIC, self.get(iid).model_dump())

    # -- resident engines (for "fits if you stop X") and mounts in use ----------------------------------------------
    def resident(self, host_id: str) -> list[ResidentUse]:
        out: list[ResidentUse] = []
        for i in self._instances_on(host_id):
            if i.state in ("stopped", "failed", "stopping", "unreachable"):
                continue
            known = [v for v in i.vram if v.used_gib is not None]
            if known:
                out.append(ResidentUse(name=i.name, gpu_ids=[v.gpu_index for v in known],
                                       gib_per_gpu=sum(v.used_gib or 0.0 for v in known) / len(known)))
            elif i.management == "managed":
                per = (i.fit or {}).get("per_gpu") or []
                out.append(ResidentUse(name=i.name, gpu_ids=i.gpu_ids, gib_per_gpu=float(per[0]["total_gib"]) if per else 0.0))
        return out

    def instances_using(self, host_id: str, path: str) -> list[str]:
        """Ids of the instances of a host using the model at `path`: their container mounts the model itself (or a
        part of it), or mounts a folder that holds it AND loads that very model. Engines usually mount a whole
        shared model cache; counting every model in it as in use would make none of them deletable."""
        node = self._nodes.node(host_id)
        norm = path.rstrip("/")
        ids: list[str] = []
        for c in nv.containers(node):
            if c.get("state") not in ("running", "restarting", "paused", "created"):
                continue
            for m in c.get("mounts") or []:
                src = str(m.get("source") or "").rstrip("/")
                if not src:
                    continue
                own = norm == src or src.startswith(norm + "/")
                holder = norm.startswith(src + "/")
                if own or (holder and _loads_model(c, norm)):
                    iid = self.instance_for_container(host_id, str(c["name"]))
                    if iid and iid not in ids:
                        ids.append(iid)
                    break
        for r in self._db.all("SELECT * FROM instances WHERE host_id=? AND state IN ('starting','loading','ready','stopping')",
                              (host_id,)):   # a launch the inventory has not caught up with yet
            loc = self._location_of(host_id, self._ref_of(r))
            if loc is None or r["id"] in ids:
                continue
            base = loc.path.rstrip("/")
            if norm == base or norm.startswith(base + "/") or base.startswith(norm + "/"):
                ids.append(str(r["id"]))
        return ids

    # -- build the launch ------------------------------------------------------------------------------------------
    def _real_params(self, inst: Instance) -> dict[str, Any]:
        """Stored params with MARKER placeholders replaced by the real secret values (memory only)."""
        stored = self._secrets.get(inst.id)
        return {k: stored.get(k, v) if v == MARKER else v for k, v in inst.params.items()}

    def _param_errors(self, adapter: EngineAdapter, params: dict[str, Any]) -> list[str]:
        errs = [f"'{k}' is a per-instance image override; choose the image in the launch form (or settings pins)"
                for k in (s.key for s in adapter.param_catalog() if s.flag == "@image") if k in params]
        errs += [f"'{k}' still holds the [set] placeholder: supply the real value" for k, v in params.items() if v == MARKER]
        real = {k: v for k, v in params.items() if v != MARKER}
        errs += adapter.validate(real)
        secret_values = [v for k, v in params.items() if SECRET_KEY.search(k) and isinstance(v, str) and v and v != MARKER]
        return [_mask(e, secret_values) for e in errs]

    def _pick_image(self, adapter: EngineAdapter, node: NodeClient, host_id: str, requested: str | None) -> str:
        """InstanceCreate.image, else the settings pin for (host, engine) or engine, else the adapter's pinned image for
        the host's CPU architecture."""
        return requested or self._settings.image_pin(adapter.id, host_id) or adapter.image_for(nv.arch(node))

    @staticmethod
    def _volumes(adapter: EngineAdapter) -> list[tuple[str, str]]:
        declared = getattr(adapter, "named_volumes", None)
        return list(declared) if declared else NAMED_VOLUMES.get(adapter.id, [])

    def _build(self, plan: Plan) -> Built:
        """The docker.run spec for a plan, validated the way the node will validate it. Raises ProblemError."""
        adapter = self._adapters.get(plan.engine)
        node = self._nodes.node(plan.host_id)
        ref = check_ref(plan.ref)
        if ref.format not in adapter.model_formats:
            raise Unprocessable(f"{adapter.display_name} cannot load {ref.format} models "
                                f"(it loads: {', '.join(sorted(adapter.model_formats))})", code="format_unsupported")
        hw = self._hw.hardware(plan.host_id, plan.gpu_ids or None)
        copy = self._library.find(plan.host_id, ref) if _needs_copy(ref) else None
        if copy is None and _needs_copy(ref):
            raise Unprocessable(f"{plan.display} is not in {self._label(plan.host_id)}'s model library", code="model_missing")
        mounts: list[dict[str, Any]] = []
        model_path: str | None = None
        model_arg = plan.display
        if ref.format == "hf":
            model_arg = ref.repo_id or plan.display
        elif ref.format == "ollama":
            root = self._library.ollama_root(plan.host_id)
            mounts, model_path = model_mounts(ref, copy_rel=copy.rel if copy else None,
                                              ollama_root=(root.id, root.path) if root else None)
            model_arg = ref.name or ""
        else:
            mounts, model_path = model_mounts(ref, copy_rel=copy.rel if copy else None, ollama_root=None)
        ctx = LaunchContext(model=model_arg, model_format=ref.format, model_path=model_path, served_name=plan.display,
                            hf_cache_container_path=self._cfg.hf_cache_container_path, params=plan.params, hw=hw)
        try:
            launch = adapter.build_launch_ctx(ctx)
        except ValueError as e:
            raise Unprocessable(str(e), code="invalid_params") from None
        if launch.needs_hf_cache:
            mounts += self._hf_mounts(node, plan.host_id, ref, copy)
        pol = nv.policy(node)
        image = self._pick_image(adapter, node, plan.host_id, plan.image_req)
        if not nv.image_allowed(image, nv.engine_images(node)):
            raise Unprocessable(f"image {image} is not allowed by {self._label(plan.host_id)}'s policy "
                                "(engine_images)", code="image_not_allowed")
        secret_values = [v for k, v in plan.params.items() if SECRET_KEY.search(k) and isinstance(v, str) and v and v != MARKER]
        if any(sv in arg for sv in secret_values for arg in launch.argv):
            raise Unprocessable("this engine adapter would put a secret parameter (e.g. api_key) on the container "
                                "command line, where any local user could read it; unset it or use an engine that "
                                "takes it from an environment variable", code="secret_in_argv")
        env, secret_env = split_env(launch.env, offline=self._cfg.engine_hf_offline)
        ipc_host = launch.ipc_host and len(hw.gpu_uuids) > 1 and pol.get("allow_ipc_host") is not False
        labels = engine_labels(instance=plan.iid, engine=adapter.id, display=plan.display, ref=ref, port=plan.port,
                               name=plan.name, gpu_ids=hw.gpu_ids, gpu_uuids=hw.gpu_uuids)
        run = build_run_spec(name=plan.container, instance=plan.iid, image=image, launch=launch, gpu_uuids=hw.gpu_uuids,
                             labels=labels, env=env, secret_env={k: "x" for k in secret_env}, mounts=mounts,
                             volumes=self._volumes(adapter), ipc_host=ipc_host)
        errs = validate_run_spec(run, max_shm_gib=nv.num(pol.get("max_shm_gib")))
        if errs:
            raise Unprocessable("; ".join(errs), code="invalid_launch")
        run.pop("secret_env")   # names only were there for validation; the values travel separately, in docker.run only
        secret_values += [v for v in secret_env.values() if v]
        return Built(run=run, secret_env=secret_env, launch=launch, image=image, gpu_ids=hw.gpu_ids, gpu_uuids=hw.gpu_uuids,
                     location=self._location_from_copy(copy) if copy else None, secret_values=secret_values)

    def _hf_mounts(self, node: NodeClient, host_id: str, ref: ModelRef, copy: LibraryCopy | None) -> list[dict[str, Any]]:
        if ref.format == "hf" and copy is not None:
            root = nv.model_roots(node).get(copy.root_id, {})
            return hf_mounts(copy.root_id, root.get("kind"), copy.rel, self._cfg.hf_cache_container_path,
                             self._cfg.hf_cache_readonly or not root.get("writable"))
        hf = self._library.hf_root(host_id)   # another format whose adapter still wants the HF cache next to it
        return hf_mounts(hf.id, hf.kind, "", self._cfg.hf_cache_container_path, True) if hf is not None else []

    def _plan_of(self, inst: Instance, port: int | None = None) -> Plan:
        return Plan(iid=inst.id, name=inst.name, engine=inst.engine, host_id=inst.host_id,
                    ref=inst.model or ModelRef(format="hf", repo_id=inst.repo_id), params=self._real_params(inst),
                    gpu_ids=inst.gpu_ids, port=port or inst.port or self._cfg.port_range_start, container=self._cn(inst),
                    image_req=self._row_image_req(inst.id))

    def _row_image_req(self, iid: str) -> str | None:
        r = self._row(iid)
        return r["image_req"] if r is not None else None

    def launch_for(self, inst: Instance) -> LaunchSpec | None:
        """The adapter's LaunchSpec for a managed instance (health/metrics paths); None when it cannot be built now."""
        if (b := self._built.get(inst.id)) is not None:
            return b.launch
        try:
            built = self._build(self._plan_of(inst))
        except (ProblemError, NodeError, KeyError, ValueError):
            return None
        self._built[inst.id] = built
        return built.launch

    # -- ports -----------------------------------------------------------------------------------------------------
    def _candidate_ports(self, host_id: str, exclude_id: str | None) -> list[int]:
        """Free ports of the host's gateway range: not used by another active instance there, not published by any
        container in its inventory (a stopped instance's own old gateway does not count against it)."""
        node = self._nodes.node(host_id)
        lo, hi = nv.port_range(node, (self._cfg.port_range_start, self._cfg.port_range_end))
        used = {r["port"] for r in self._db.all(
            "SELECT port FROM instances WHERE host_id=? AND state IN ('starting','loading','ready','stopping') AND id != ?",
            (host_id, exclude_id or "")) if r["port"]}
        mine: set[str] = set()
        for c in nv.containers(node):
            if exclude_id and c.get("instance") == exclude_id:
                mine.add(str(c["name"]))
                continue
            used |= set(nv.host_ports(c))
        used |= {int(e["port"]) for e in nv.endpoints(node) if e.get("container") not in mine}
        return [p for p in range(lo, hi + 1) if p not in used]

    def _alloc_port(self, host_id: str, exclude_id: str | None) -> int | None:
        free = self._candidate_ports(host_id, exclude_id)
        return free[0] if free else None

    # -- preflight ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _ref_for(req: InstanceCreate) -> ModelRef:
        ref = req.model or ModelRef(format="hf", repo_id=req.repo_id)
        if ref.format == "hf" and not ref.repo_id and req.repo_id:
            ref = ref.model_copy(update={"repo_id": req.repo_id})
        return ref

    async def preflight(self, req: InstanceCreate, *, exclude_id: str | None = None,
                        params: dict[str, Any] | None = None) -> tuple[PreflightReport, int | None]:
        host_id = req.host_id
        node = self._nodes.node(host_id)   # 404 unknown_host
        label = node.host.label
        adapter = self._adapters.get(req.engine)
        p = params if params is not None else req.params
        ref = check_ref(self._ref_for(req))
        display = display_of(ref)
        checks: list[PreflightCheck] = []
        fit: FitReport | None = None

        if not nv.link_up(node):
            link = node.link()
            checks.append(PreflightCheck(code="host_down", level="block",
                                         message=f"{label} is not connected ({link.error or link.state}): nothing can be "
                                                 "started there until its link is up"))
            return PreflightReport(ok=False, gpu_ids=req.gpu_ids, checks=checks), None
        if nv.mode(node) != "manage":
            where = nv.policy(node).get("path") or "its policy file"
            checks.append(PreflightCheck(code="monitor_only", level="block",
                                         message=f"{label}'s node is in monitor mode ({where}): it can be watched but not "
                                                 "changed. Its operator must switch \"mode\" to \"manage\" in that policy file."))
        errs = self._param_errors(adapter, p)
        if errs:
            checks.append(PreflightCheck(code="invalid_params", level="block", message="; ".join(errs)))
        format_ok = ref.format in adapter.model_formats
        if not format_ok:
            checks.append(PreflightCheck(code="format_unsupported", level="block",
                                         message=f"{adapter.display_name} cannot load {ref.format} models "
                                                 f"(it loads: {', '.join(sorted(adapter.model_formats))})"))
        copy = self._library.find(host_id, ref) if _needs_copy(ref) else None
        if copy is None and _needs_copy(ref):
            checks.append(PreflightCheck(code="model_missing", level="block",
                                         message=f"{display} is not in {label}'s model library: download or copy it to "
                                                 "that machine first"))
        model_ok = format_ok and (copy is not None or not _needs_copy(ref))
        info: ModelInfo | None = None
        if model_ok and not errs:
            try:
                info = await self._fit.info_for(host_id, ref, allow_hub=False)
            except ProblemError as e:
                checks.append(PreflightCheck(code="fit_unavailable", level="warn", message=f"no fit estimate: {e.detail}"))
        gpu_ids = self._resolve_gpus(adapter, req, p, info, display)
        if info is not None and not errs:
            fit = self._fit.assess_info(req.engine, info, p, gpu_ids, host_id=host_id)
            level = {"fits": "ok", "tight": "warn", "unknown": "warn", "wont_fit": "block"}[fit.verdict]
            msg = f"fit verdict: {fit.verdict} ({fit.confidence} confidence)"
            if fit.fits_if_stop:
                msg += f"; fits if you stop {', '.join(fit.fits_if_stop)}"
            checks.append(PreflightCheck(code="fit", level=level, message=msg))
            for c in fit.compat:
                if c.level != "ok":
                    checks.append(PreflightCheck(code=c.code, level=c.level, message=c.message))
        trusted = [k for k, v in p.items() if "trust_remote" in k and v not in (None, False, "", MARKER)]
        if trusted:
            checks.append(PreflightCheck(code="trust_remote_code", level="warn",
                                         message=f"{', '.join(trusted)} runs Python code from the model repo inside the "
                                                 "container; it is off unless you set it explicitly"))
        port = self._alloc_port(host_id, exclude_id)
        if port is None:
            lo, hi = nv.port_range(node, (self._cfg.port_range_start, self._cfg.port_range_end))
            checks.append(PreflightCheck(code="no_free_port", level="block", message=f"no free port in {lo}-{hi} on {label}"))
        else:
            checks.append(PreflightCheck(code="port", level="ok", message=f"host port {port}"))
        self._gpu_checks(checks, node, host_id, label, req, gpu_ids, exclude_id)
        image = self._pick_image(adapter, node, host_id, req.image)
        if not nv.image_allowed(image, nv.engine_images(node)):
            checks.append(PreflightCheck(code="image_not_allowed", level="block",
                                         message=f"image {image} is not allowed by {label}'s policy (engine_images)"))
        else:
            try:
                present = await self._nodes.call(host_id, "docker.image_present", {"image": image}, timeout=30)
            except ProblemError as e:
                checks.append(PreflightCheck(code="image_check_failed", level="warn", message=e.detail))
            else:
                if not present.get("present"):
                    checks.append(PreflightCheck(code="image_missing", level="block",
                                                 message=f"image {image} is not present on {label}: pull it on that machine "
                                                         "(the console never pulls)"))
        if not nv.policy(node).get("gateway_image"):
            checks.append(PreflightCheck(code="gateway_missing", level="block",
                                         message=f"{label}'s policy names no gateway_image, so engines there cannot get "
                                                 "their loopback gateway"))
        if model_ok and not errs and not any(c.level == "block" and c.code in ("image_not_allowed", "no_gpu", "gpu_unknown")
                                             for c in checks):
            try:   # the same build a real launch does: catches a secret on argv, a bad mount or an invalid spec
                self._build(Plan(iid="preflight", name="preflight", engine=req.engine, host_id=host_id, ref=ref, params=p,
                                 gpu_ids=gpu_ids, port=port or 0, container="ec-preflight", image_req=req.image))
            except ProblemError as e:
                checks.append(PreflightCheck(code=e.code, level="block", message=e.detail))
        return PreflightReport(ok=not any(c.level == "block" for c in checks), gpu_ids=gpu_ids, checks=checks, fit=fit), port

    def _resolve_gpus(self, adapter: EngineAdapter, req: InstanceCreate, params: dict[str, Any], info: ModelInfo | None,
                      display: str) -> list[int]:
        try:
            return self._fit.resolve_gpus(adapter.id, info or ModelInfo(repo_id=display), params, req.gpu_ids, req.host_id)
        except Exception:   # noqa: BLE001 - an adapter's memory model may not cope with an unknown model: use the request
            return list(req.gpu_ids)

    def _gpu_checks(self, checks: list[PreflightCheck], node: NodeClient, host_id: str, label: str, req: InstanceCreate,
                    gpu_ids: list[int], exclude_id: str | None) -> None:
        have = {g["index"] for g in nv.static_gpus(node) if isinstance(g.get("index"), int)}
        if not have:
            checks.append(PreflightCheck(code="no_gpu", level="block", message=f"{label} reports no GPU"))
            return
        missing = [i for i in req.gpu_ids if i not in have]
        if missing:
            checks.append(PreflightCheck(code="gpu_unknown", level="block",
                                         message=f"{label} has no GPU {', '.join(map(str, missing))} (it has "
                                                 f"{', '.join(map(str, sorted(have)))})"))
            return
        chosen = set(gpu_ids) or have
        clash = [i.name for i in self._instances_on(host_id) if i.id != exclude_id and i.state in RUNNING
                 and chosen & (set(i.gpu_ids) | {v.gpu_index for v in i.vram})]
        if clash:
            checks.append(PreflightCheck(code="gpu_shared", level="warn",
                                         message=f"GPU already used by {', '.join(clash)}; relying on free VRAM"))

    # -- commands --------------------------------------------------------------------------------------------------
    async def create(self, req: InstanceCreate, profile_params: dict[str, Any] | None = None) -> Instance:
        params = {**(profile_params or {}), **req.params}
        public, secret_values = split_secrets(params)
        async with self._lock(req.host_id):
            report, port = await self.preflight(req, params=params)
            if not report.ok or port is None:
                raise Conflict("preflight failed", code="preflight_failed", preflight=report.model_dump())
            ref = check_ref(self._ref_for(req))
            iid = new_id("inst_")
            display = display_of(ref)
            name = req.name or _safe_name(display.split("/")[-1])
            plan = Plan(iid=iid, name=name, engine=req.engine, host_id=req.host_id, ref=ref, params=params,
                        gpu_ids=report.gpu_ids, port=port, container=f"ec-{_slug(name)}-{iid[-6:]}", image_req=req.image)
            built = self._build(plan)
            self._db.execute(
                "INSERT INTO instances(id,name,engine,repo_id,params,gpu_ids,gpu_uuids,port,container_name,image,state,"
                "profile_id,ttl_idle_s,fit,created_at,host_id,model,image_req) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (iid, name, req.engine, display, json.dumps(public), json.dumps(built.gpu_ids), json.dumps(built.gpu_uuids),
                 port, plan.container, built.image, "starting", req.profile_id, req.ttl_idle_s,
                 report.fit.model_dump_json() if report.fit else None, now(), req.host_id, ref.model_dump_json(), req.image))
            self._secrets.put(iid, secret_values)
            await self._launch(plan, built)
        return self.get(iid)

    async def _call_ignoring_missing(self, host_id: str, op: str, args: dict[str, Any], *, timeout: float = 60.0) -> None:
        """Stop/remove are idempotent: a container that is already gone is the outcome we wanted."""
        try:
            await self._nodes.call(host_id, op, args, timeout=timeout)
        except ProblemError as e:
            if e.code != "not_found":
                raise

    async def _remove_containers(self, plan: Plan) -> None:
        await self._call_ignoring_missing(plan.host_id, "docker.rm", {"name": gateway_name(plan.container), "role": "gateway",
                                                                       "instance": plan.iid})
        await self._call_ignoring_missing(plan.host_id, "docker.rm", {"name": plan.container, "instance": plan.iid})

    async def _gateway(self, plan: Plan, container_port: int) -> int:
        """Start the loopback gateway; when its port is taken, try the next free one of the host's range."""
        order = [plan.port, *[p for p in self._candidate_ports(plan.host_id, plan.iid) if p != plan.port]]
        last: ProblemError | None = None
        for port in order[:GATEWAY_ATTEMPTS]:
            try:
                await self._nodes.call(plan.host_id, "gateway.start",
                                       {"name": gateway_name(plan.container), "instance": plan.iid, "host_port": port,
                                        "engine_host": plan.container, "engine_port": container_port}, timeout=60)
                return port
            except ProblemError as e:
                if e.code != "port_in_use":
                    raise
                last = e
        raise last or ProblemError("no free port for the gateway", code="no_free_port", status=409)

    async def _launch(self, plan: Plan, built: Built) -> None:
        try:
            await self._nodes.call(plan.host_id, "net.ensure_internal", timeout=30)
            await self._remove_containers(plan)   # clear stale containers of the same name (the node verifies ownership)
            await self._nodes.call(plan.host_id, "docker.run", {"spec": {**built.run, "secret_env": built.secret_env}},
                                   timeout=180)
            try:
                port = await self._gateway(plan, built.launch.container_port)
            except ProblemError:
                with contextlib.suppress(ProblemError):   # never leave an engine running without its front door
                    await self._nodes.call(plan.host_id, "docker.rm", {"name": plan.container, "instance": plan.iid})
                raise
            self._built[plan.iid] = built
            self._set(plan.iid, state="starting", started_at=now(), error=None, phase=None, progress_pct=None,
                      last_logs=None, container_port=built.launch.container_port, port=port)
        except ProblemError as e:
            self._set(plan.iid, state="failed", error=e.detail)
            raise

    async def start(self, iid: str) -> Instance:
        inst = self.get(iid)
        if inst.management == "script":
            return await self._script_action(inst, "up")
        if inst.management == "external":
            return await self._external_action(inst, "start")
        if inst.state in ("starting", "loading", "ready"):
            raise Conflict(f"instance is already {inst.state}")
        async with self._lock(inst.host_id):
            inst = self.get(iid)   # another request may have started it while this one waited for the host's lock
            if inst.state in ("starting", "loading", "ready"):
                raise Conflict(f"instance is already {inst.state}")
            req = InstanceCreate(engine=inst.engine, model=inst.model, params=inst.params, gpu_ids=inst.gpu_ids,
                                 host_id=inst.host_id, image=self._row_image_req(iid))   # a restart keeps its GPUs
            real = self._real_params(inst)
            report, port = await self.preflight(req, exclude_id=iid, params=real)
            if not report.ok or port is None:
                raise Conflict("preflight failed", code="preflight_failed", preflight=report.model_dump())
            plan = self._plan_of(inst, port)
            built = self._build(plan)
            self._set(iid, port=port, gpu_uuids=built.gpu_uuids, image=built.image, state="starting",
                      fit=report.fit.model_dump_json() if report.fit else None)
            await self._launch(plan, built)
        return self.get(iid)

    async def stop(self, iid: str) -> Instance:
        inst = self.get(iid)
        if inst.management == "script":
            return await self._script_action(inst, "down")
        if inst.management == "external":
            return await self._external_action(inst, "stop")
        if inst.state in ("stopped",):
            return inst
        row = self._row(iid)
        before = row["state"] if row is not None else inst.state
        cn = self._cn(inst)
        self._set(iid, state="stopping")
        try:
            await self._call_ignoring_missing(inst.host_id, "docker.stop", {"name": gateway_name(cn), "role": "gateway",
                                                                             "instance": iid, "timeout_s": 30}, timeout=90)
            await self._call_ignoring_missing(inst.host_id, "docker.stop", {"name": cn, "instance": iid, "timeout_s": 30},
                                              timeout=90)
        except ProblemError as e:
            # a node that could not even be asked changed nothing: do not claim the engine failed
            self._set(iid, **({"state": before} if e.code in UNAVAILABLE_CODES else {"state": "failed", "error": e.detail}))
            raise
        self._set(iid, state="stopped", phase=None, progress_pct=None)
        return self.get(iid)

    async def restart(self, iid: str) -> Instance:
        inst = self.get(iid)
        if inst.management == "script":
            if self._scripts is None:
                raise Conflict("scripts are not available", code="instance_not_managed")
            if not inst.controls.restart:
                raise Conflict(f"cannot restart this engine: {inst.controls.reason}", code="instance_not_managed")
            await self._scripts.restart(inst.host_id, inst.script or "")
            return self.get(iid)
        if inst.management == "external":
            raise Conflict("an external engine can only be stopped or started, not restarted", code="instance_not_managed")
        await self.stop(iid)
        if self._cfg.restart_settle_s > 0:
            # the fit check reads the host's last pushed sample, which may still show the VRAM this very engine just freed
            await asyncio.sleep(self._cfg.restart_settle_s)
        return await self.start(iid)

    async def _script_action(self, inst: Instance, action: str) -> Instance:
        if self._scripts is None:
            raise Conflict("scripts are not available", code="instance_not_managed")
        if not (inst.controls.start if action == "up" else inst.controls.stop):
            raise Conflict(f"cannot {'start' if action == 'up' else 'stop'} this engine: {inst.controls.reason}",
                           code="instance_not_managed")
        await self._scripts.run(inst.host_id, inst.script or "", action)
        return self.get(inst.id)

    async def _external_action(self, inst: Instance, action: str) -> Instance:
        if not (inst.controls.start if action == "start" else inst.controls.stop) or not inst.container_name:
            raise Conflict(f"cannot {action} this engine: {inst.controls.reason}", code="instance_not_managed")
        await self._nodes.call(inst.host_id, "docker.external", {"name": inst.container_name, "action": action}, timeout=90)
        if action == "stop":
            stopped = self._as_stopped(inst)
            self._stopped_ext.setdefault(inst.host_id, {})[inst.id] = stopped
            self._prev_ext.get(inst.host_id, {}).pop(inst.id, None)
            return stopped
        self._stopped_ext.get(inst.host_id, {}).pop(inst.id, None)
        self._gone.get(inst.host_id, {}).pop(inst.id, None)
        return inst.model_copy(update={"state": "starting", "state_reason": "start requested"})

    async def remove(self, iid: str) -> None:
        inst = self.require_managed(iid)
        cn = self._cn(inst)
        await self._call_ignoring_missing(inst.host_id, "docker.rm", {"name": gateway_name(cn), "role": "gateway", "instance": iid})
        await self._call_ignoring_missing(inst.host_id, "docker.rm", {"name": cn, "instance": iid})
        self._secrets.delete(iid)
        self._built.pop(iid, None)
        self._db.execute("DELETE FROM instances WHERE id=?", (iid,))

    def patch(self, iid: str, p: InstancePatch) -> Instance:
        self.require_managed(iid)
        cols: dict[str, Any] = {}
        if p.pinned is not None:
            cols["pinned"] = int(p.pinned)
        if p.ttl_idle_s is not None:
            cols["ttl_idle_s"] = p.ttl_idle_s
        if p.name is not None:
            if not p.name.strip():
                raise BadRequest("name must not be empty")
            cols["name"] = p.name
        if cols:
            self._set(iid, **cols)
        return self.get(iid)

    # -- logs / command ----------------------------------------------------------------------------------------------
    def _scrub(self, inst: Instance, text: str) -> str:
        """Engines may echo their own args (api keys) into logs: mask known secret values, then token shapes."""
        values = list(self._secrets.get(inst.id).values())
        built = self._built.get(inst.id)
        if built is None:
            with contextlib.suppress(ProblemError, NodeError, KeyError, ValueError):
                built = self._build(self._plan_of(inst))
        if built is not None:
            values += built.secret_values
        for v in values:
            if len(v) >= 6:
                text = text.replace(v, "[redacted]")
        return scrub(text)

    async def logs(self, iid: str, tail: int = 200) -> str:
        inst = self.require_managed(iid)
        res = await self._nodes.call(inst.host_id, "docker.logs", {"name": self._cn(inst), "tail": max(1, min(tail, 5000))},
                                     timeout=30)
        return self._scrub(inst, str(res.get("text") or ""))

    async def follow_logs(self, iid: str, tail: int = 100) -> AsyncIterator[str]:
        inst = self.require_managed(iid)
        stream = self._nodes.open(inst.host_id, "docker.logs_follow", {"name": self._cn(inst), "tail": max(1, min(tail, 5000))})
        try:
            async for item in stream:
                ev = item.event if item.kind == "event" else None
                if ev and isinstance(ev.get("line"), str):
                    yield self._scrub(inst, ev["line"])
        except NodeError as e:
            raise self._nodes.problem(e) from None
        finally:
            await shielded(stream.cancel())

    async def command(self, iid: str) -> CommandSnippets:
        """"Copy as command": the node renders the exact `docker run` it would use; the compose file and the bare engine
        command are built here from the same spec. Secret values are never part of it: names only."""
        inst = self.require_managed(iid)
        built = self._build(self._plan_of(inst))
        names = {k: "[set]" for k in built.secret_env}
        rendered = await self._nodes.call(inst.host_id, "docker.render", {"spec": {**built.run, "secret_env": names}}, timeout=30)
        node = self._nodes.node(inst.host_id)
        roots = {rid: str(r["path"]) for rid, r in nv.model_roots(node).items() if r.get("path")}
        return CommandSnippets(**command_snippets({**built.run, "secret_env": names}, [str(a) for a in rendered.get("argv") or []],
                                                  host_paths=roots,
                                                  network=nv.policy(node).get("engine_network") or self._cfg.engine_network))

    # -- supervisor ---------------------------------------------------------------------------------------------------
    async def sync_links(self) -> None:
        """Adopt each host's console-owned containers when its link comes up (and once at start-up)."""
        for h in self._fleet.hosts():
            node = self._fleet.node(h.id)
            key = (nv.link_up(node), node.link().connected_since)
            if key[0] and key != self._links.get(h.id):
                with contextlib.suppress(Exception):
                    await self.adopt(h.id)
            self._links[h.id] = key

    async def tick(self) -> None:
        try:
            await self.sync_links()
        except Exception:   # noqa: BLE001 - never let adoption break supervision
            log.exception("link sync failed")
        for inst in self.all_active():
            try:
                await self._supervise(inst)
            except Exception:   # noqa: BLE001 - one bad instance must not stop supervision of the rest
                log.exception("supervisor error for %s", inst.id)
        with contextlib.suppress(Exception):
            self.externals()    # keeps the vanished-engine bookkeeping moving

    async def _container(self, host_id: str, name: str, since: float | None) -> dict[str, Any] | None:
        """A container's facts: from the pushed inventory, asking the node only when it is not there and the inventory
        may simply not have caught up with something we just did."""
        node = self._nodes.node(host_id)
        c = nv.container_named(node, name)
        if c is not None:
            return c
        recent = since is not None and now() - since < INVENTORY_LAG_S
        if nv.inventory(node) and not recent:
            return None
        try:
            res = await self._nodes.call(host_id, "docker.inspect", {"name": name}, timeout=20)
        except ProblemError as e:
            if e.code == "not_found":
                return None
            raise
        info = res.get("info")
        return info if isinstance(info, dict) and info.get("name") else None

    async def _logs_text(self, host_id: str, name: str, tail: int) -> str:
        with contextlib.suppress(ProblemError):
            return str((await self._nodes.call(host_id, "docker.logs", {"name": name, "tail": tail}, timeout=30)).get("text") or "")
        return ""

    async def _supervise(self, inst: Instance) -> None:
        if inst.state == "unreachable" or inst.engine not in self._adapters:
            return
        try:
            node = self._nodes.node(inst.host_id)
        except NotFound:
            return
        if not nv.link_up(node):
            return   # cannot see that machine: keep the last known state, do not invent a failure
        cn = self._cn(inst)
        if inst.state == "stopping":
            # a stop whose request went away half way (a browser closing it) must not leave the instance stopping forever
            gone = await self._container(inst.host_id, cn, None)
            if gone is None or gone.get("state") != "running":
                self._set(inst.id, state="stopped", phase=None, progress_pct=None)
            return
        info = await self._container(inst.host_id, cn, inst.started_at)
        if info is None or info.get("state") != "running":
            logs = await self._logs_text(inst.host_id, cn, 200) if info is not None else ""
            code = f" with code {info.get('exit_code')}" if info is not None else ""
            self._set(inst.id, state="failed", error=f"container exited{code}" if info else "container disappeared",
                      last_logs=self._scrub(inst, logs)[-20000:] or None, phase=None)
            with contextlib.suppress(ProblemError):
                await self._nodes.call(inst.host_id, "docker.stop", {"name": gateway_name(cn), "role": "gateway", "instance": inst.id})
            return
        if info.get("role") == "engine":   # legacy (pre-gateway) containers publish their own port
            gw = await self._container(inst.host_id, gateway_name(cn), inst.started_at)
            if gw is None or gw.get("role") != "gateway" or gw.get("instance") != inst.id or gw.get("state") != "running":
                self._set(inst.id, state="failed", phase=None, error="gateway container missing or stopped; restart the instance")
                return
        adapter = self._adapters.get(inst.engine)
        if inst.state in ("starting", "loading"):
            await self._progress(inst, adapter)
        elif inst.state == "ready":
            ttl = inst.ttl_idle_s if inst.ttl_idle_s is not None else self._settings.idle_ttl_s()
            last = inst.last_request_at or inst.started_at or 0.0
            if ttl and not inst.pinned and now() - last > ttl:
                log.info("stopping idle instance %s after %ss", inst.id, ttl)
                await self.stop(inst.id)

    async def _progress(self, inst: Instance, adapter: EngineAdapter) -> None:
        state: InstanceState = inst.state
        phase, pct = inst.phase, inst.progress_pct
        cn = self._cn(inst)
        for line in (await self._logs_text(inst.host_id, cn, 40)).splitlines():
            parsed = adapter.parse_startup_log(line)
            if parsed:
                phase = str(parsed.get("phase", phase)) if parsed.get("phase") else phase
                if parsed.get("pct") is not None:
                    pct = float(parsed["pct"])
        if phase and phase != "ready" and state == "starting":
            state = "loading"
        if await self._probe(inst):
            state, phase, pct = "ready", "ready", 100.0
        elif inst.started_at and now() - inst.started_at > self._cfg.startup_timeout_s:
            logs = await self._logs_text(inst.host_id, cn, 200)
            self._set(inst.id, state="failed", error="startup timed out", last_logs=self._scrub(inst, logs)[-20000:] or None)
            return
        if (state, phase, pct) != (inst.state, inst.phase, inst.progress_pct):
            self._set(inst.id, state=state, phase=phase, progress_pct=pct)

    async def _probe(self, inst: Instance) -> bool:
        """Readiness: GET the engine's health path through the host's node (the gateway's loopback port)."""
        if inst.port is None:
            return False
        launch = self.launch_for(inst)
        try:
            r = await self._http.request(inst.host_id, inst.port, "GET", launch.health_path if launch else "/health",
                                         timeout_s=3.0, max_bytes=64 * 1024)
        except (ProblemError, NodeError):
            return False
        return r.status == 200

    # -- adoption ------------------------------------------------------------------------------------------------------
    async def adopt(self, host_id: str | None = None) -> int:
        """Re-attach to console-owned engine containers that outlived the console, on one host or every host."""
        ids = [host_id] if host_id else [h.id for h in self._fleet.hosts()]
        return sum(self._adopt_host(h) for h in ids)

    def _adopt_host(self, host_id: str) -> int:
        node = self._nodes.node(host_id)
        n = 0
        lo, hi = nv.port_range(node, (self._cfg.port_range_start, self._cfg.port_range_end))
        idx = nv.uuid_index(node)
        for c in nv.containers(node):
            if not c.get("managed"):
                continue
            lab: dict[str, str] = {str(k): str(v) for k, v in (c.get("labels") or {}).items()}
            if lab.get("engine-console.role") == "gateway" or c.get("role") == "gateway":
                continue   # gateways are only ever handled as part of their own instance
            cname = str(c["name"])
            running = c.get("state") == "running"
            row = self._db.one("SELECT id, state FROM instances WHERE container_name=? AND host_id=?", (cname, host_id))
            if row is not None:
                if running:
                    self._set(row["id"], state="loading")   # the supervisor promotes it to ready via the health probe
                elif row["state"] in ACTIVE and c.get("exit_code") in (None, 0):
                    self._set(row["id"], state="stopped")   # a clean exit: someone stopped it while we were away
                # any other exit is a crash: the supervisor reports it with the exit code and the engine's last logs
                continue
            iid = lab.get("engine-console.instance")
            engine, model = lab.get("engine-console.engine"), lab.get("engine-console.model")
            if not (iid and engine and model) or engine not in self._adapters:
                continue
            try:   # labels are attacker-influenced strings: validate before they reach SQL rows, paths or UI
                ref = self._ref_from_labels(lab.get("engine-console.format", "hf"), model, lab)
                # our ids are `inst_...`: anything else could shadow a derived instance (`ext-...`, `scr-...`) in the API
                if not (iid.startswith("inst_") and _ID_RE.fullmatch(iid) and _ID_RE.fullmatch(engine)
                        and _NAME_RE.fullmatch(lab.get("engine-console.name", cname))):
                    raise ValueError("bad label")
            except ValueError:
                log.warning("not adopting %s: invalid engine-console.* labels", cname)
                continue
            port = self._adopt_port(node, cname, iid, lab)
            if port is None or not lo <= port <= hi:
                log.warning("not adopting %s: label port outside the console range", cname)
                continue
            uuids = [u for u in lab.get("engine-console.gpus", "").split(",") if u in idx]
            gpu_ids = [int(x) for x in lab.get("engine-console.gpu_ids", "").split(",") if x.isdigit()] or [idx[u] for u in uuids]
            image = str(c.get("image") or self._adapters.get(engine).image_for(nv.arch(node)))
            try:
                self._db.execute(
                    "INSERT INTO instances(id,name,engine,repo_id,params,gpu_ids,gpu_uuids,port,container_name,image,state,created_at,"
                    "started_at,host_id,model) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (iid, lab.get("engine-console.name", cname), engine, model, "{}", json.dumps(gpu_ids), json.dumps(uuids), port,
                     cname, image, "loading" if running else "stopped", now(), now() if running else None, host_id,
                     ref.model_dump_json()))
            except sqlite3.IntegrityError:   # a container claiming the id of an instance that already exists: not ours to adopt
                log.warning("not adopting %s: its instance id is already taken", cname)
                continue
            n += 1
        return n

    @staticmethod
    def _ref_from_labels(fmt: str, model: str, lab: dict[str, str]) -> ModelRef:
        if fmt == "hf":
            check_repo_id(model)
            return ModelRef(format="hf", repo_id=model)
        if not _LABEL_MODEL.fullmatch(model):
            raise ValueError("bad model label")
        if fmt == "ollama":
            return ModelRef(format="ollama", name=model)
        root, rel = lab.get("engine-console.root"), lab.get("engine-console.rel")
        if fmt in ("gguf", "plain") and root and _ROOT_RE.fullmatch(root) and (not rel or rel_ok(rel)):
            return ModelRef.model_validate({"format": fmt, "root_id": root, "rel": rel})
        raise ValueError("bad model location labels")

    @staticmethod
    def _adopt_port(node: NodeClient, cname: str, iid: str, lab: dict[str, str]) -> int | None:
        """The gateway's real published port (a retry may have moved it off the label's), else the label's."""
        gw = nv.container_named(node, gateway_name(cname))
        if gw is not None and gw.get("instance") in (iid, None):
            ports = nv.host_ports(gw)
            if ports:
                return ports[0]
        try:
            return int(lab.get("engine-console.port", ""))
        except ValueError:
            return None
