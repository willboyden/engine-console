"""Engines the console did not create, read from each host's pushed inventory (contract v2, §10.4).

Two classes, both derived (never stored):
  * `script` - containers matched by a script registered in the node's policy become ONE instance per script
    (id `scr-<host>-<script>`, each '-' of the host doubled); start/stop/restart map to that script's `up`/`down` actions.
  * `external` - any other running engine container (id `ext-<host>-<slug>`, same rule), monitor-only unless the node policy
    says `external_control: "stop_start"`.

Pure functions over a `NodeClient`'s latest messages: no I/O, no clocks (the caller passes `now`)."""
from __future__ import annotations

import fnmatch
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from engine_console.domain.fleet import NodeClient
from engine_console.domain.models import HostMemory, Instance, InstanceState
from engine_console.domain.refs import InstanceControls, InstanceVram, ModelLocation
from engine_console.services import nodeview as nv
from engine_console.services.common import scrub
from engine_console.services.discovery import EXT, SCRIPT, clean_text, display_name, slug
from engine_console.services.hostmem import container_memory, sum_memory

_SCRIPT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,31}")
GONE_CAP = 64


@dataclass(frozen=True)
class HostCtx:
    """What the builders need to know about one host, plus how to recognise the other hosts (shared storage)."""
    host_id: str
    label: str
    node: NodeClient
    others: Mapping[str, tuple[str, frozenset[str]]] = field(default_factory=dict)   # host id -> (label, addresses)


def _host_part(host_id: str) -> str:
    """Host ids may hold '-', and so may the slug after them, so `ext-gpu-2-vllm` could be host `gpu` running `2-vllm`
    or host `gpu-2` running `vllm`; an action on one row would reach the other host (security review M1). Every '-'
    of the host id is doubled. A slug never holds '--' (slug() collapses runs) and never starts with '-', so the first
    single '-' after the host part is the separator and two hosts can never produce the same id."""
    return host_id.replace("-", "--")


def script_instance_id(host_id: str, name: str) -> str:
    return f"{SCRIPT}{_host_part(host_id)}-{slug(name)}"


def external_instance_id(host_id: str, key: str) -> str:
    return f"{EXT}{_host_part(host_id)}-{slug(key)}"


# ---- measurements -------------------------------------------------------------------------------------------------
def vram_for(node: NodeClient, names: set[str]) -> list[InstanceVram]:
    """Per GPU, the VRAM the driver attributes to processes of these containers. None where the driver cannot say."""
    idx = nv.uuid_index(node)
    live = nv.sample(node).get("gpus")
    out: list[InstanceVram] = []
    for uuid, g in (live.items() if isinstance(live, dict) else []):
        if uuid not in idx or not isinstance(g, dict):
            continue
        procs = [p for p in g.get("procs") or [] if isinstance(p, dict) and p.get("container") in names]
        if not procs:
            continue
        known = [p["used_bytes"] for p in procs if isinstance(p.get("used_bytes"), int)]
        out.append(InstanceVram(gpu_index=idx[uuid], gpu_uuid=str(uuid),
                                used_gib=sum(known) / nv.GIB if known else None))
    return sorted(out, key=lambda v: v.gpu_index)


def memory_for(node: NodeClient, names: set[str]) -> HostMemory | None:
    per = nv.sample(node).get("containers")
    if not isinstance(per, dict):
        return None
    parts = [m for n in sorted(names) if isinstance(per.get(n), dict) and (m := container_memory(per[n].get("mem")))]
    return sum_memory(parts)


def drive_label(m: dict[str, Any], ctx: HostCtx) -> str:
    """Human label of a filesystem: "tank (zfs)", "NFS from gpu-box-2", "/ (ext4, Samsung 990)"."""
    if m.get("remote") and m.get("remote_host"):
        owner = _owner_of(str(m["remote_host"]), ctx)
        return f"NFS from {ctx.others[owner][0] if owner else m['remote_host']}"
    kind = ", ".join(x for x in (m.get("fstype"), m.get("model")) if x)
    return f"{m.get('mount', '?')} ({kind})" if kind else str(m.get("mount", "?"))


def _owner_of(address: str, ctx: HostCtx) -> str | None:
    """The configured host (other than this one) that owns an NFS server address, if any."""
    return next((h for h, (_, addrs) in ctx.others.items() if h != ctx.host_id and address in addrs), None)


def location_for(ctx: HostCtx, mounts: list[dict[str, Any]]) -> ModelLocation | None:
    """Where an engine's model lives, from its container mounts that fall inside a policy model root (the node
    reports `root`/`rel` per mount) joined with that root's filesystem from the node's `disks`."""
    cands = [m for m in mounts if isinstance(m, dict) and m.get("root") and m.get("source")]
    if not cands:
        return None
    pick = next((m for m in cands if str(m.get("dest", "")).startswith("/models")), cands[0])
    root_id = str(pick["root"])
    root = nv.model_roots(ctx.node).get(root_id, {})
    disks = nv.disk_mounts(ctx.node)
    drive = next((d for d in disks if root_id in (d.get("roots") or [])), None)
    owner = _owner_of(str(drive.get("remote_host")), ctx) if drive and drive.get("remote_host") else None
    return ModelLocation(
        host_id=ctx.host_id, root_id=root_id, root_label=root.get("label") or None, path=str(pick["source"]),
        drive=drive_label(drive, ctx) if drive else None, fstype=(drive or {}).get("fstype") or None,
        remote=bool(drive and drive.get("remote")),
        shared_from=(owner or (drive.get("remote_host") if drive and drive.get("remote") else None)) or None)


# ---- endpoint reading ---------------------------------------------------------------------------------------------
def endpoint_of(c: dict[str, Any], eps: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The probed endpoint that belongs to a container: by name, else by one of its published ports."""
    named = [e for e in eps if e.get("container") == c.get("name")]
    ports = set(nv.host_ports(c))
    cands = named or [e for e in eps if e.get("port") in ports]
    # prefer the one that reports healthy, then any that answers at all
    return min(cands, key=lambda e: (not (nv.ok_status(e.get("health_status")) or nv.ok_status(e.get("models_status"))),
                                     not _answered(e)), default=None) if cands else None


def _answered(e: dict[str, Any]) -> bool:
    return any(nv.ok_status(e.get(k)) or nv.auth_status(e.get(k)) for k in ("health_status", "models_status", "metrics_status"))


def endpoint_state(ep: dict[str, Any] | None, *, has_ports: bool) -> tuple[InstanceState, str | None, int | None, list[str]]:
    """(state, reason, port, served models) for a running engine from its probed endpoint."""
    if ep is None:
        if not has_ports:
            return ("unreachable", "no published host port: the console reaches engines on their machine's loopback "
                                   "(publish a port on 127.0.0.1 to monitor it)", None, [])
        return "unreachable", "its published port(s) were not probed or did not answer yet", None, []
    port = int(ep["port"])
    models = [clean_text(m) for m in (ep.get("models") or [])[:20] if isinstance(m, str)]
    if nv.ok_status(ep.get("health_status")) or nv.ok_status(ep.get("models_status")):
        reason = None
        if nv.auth_status(ep.get("models_status")) or ep.get("auth_required"):
            reason = "/v1 endpoints need an API key (HTTP 401/403): metrics work, chat does not"
        return "ready", reason, port, models
    if _answered(ep) or ep.get("auth_required"):
        return ("auth_required", "the engine answered HTTP 401/403: an API key is required, so metrics and chat are "
                                 "unavailable", port, models)
    if any(ep.get(k) == 503 for k in ("health_status", "models_status")):   # up, but not ready to serve yet
        return "loading", "the engine answers HTTP 503: it is still loading its model", port, models
    return "unreachable", f"port {port} did not answer /v1/models, /health or /metrics", port, models


# ---- instances ----------------------------------------------------------------------------------------------------
def _gpu_uuids(ctx: HostCtx, c: dict[str, Any]) -> tuple[list[int], list[str]]:
    idx = nv.uuid_index(ctx.node)
    if c.get("all_gpus"):
        uuids = sorted(idx, key=lambda u: idx[u])
    else:
        uuids = [u for u in c.get("gpus") or [] if isinstance(u, str) and u in idx]
    return [idx[u] for u in uuids], uuids


def _params_of(c: dict[str, Any]) -> dict[str, Any]:
    """The node's display-only parse of an engine's (already redacted) command line. Node text is untrusted, so it is
    scrubbed again here: a token-shaped value never reaches the API, even if a node forgot to redact it."""
    raw = c.get("params")
    if not isinstance(raw, dict):
        return {}
    return {clean_text(k, 64): scrub(clean_text(v)) for k, v in raw.items()}


def _started(ctx: HostCtx, c: dict[str, Any]) -> float | None:
    t = nv.num(c.get("started_at"))
    return None if t is None else t - (ctx.node.link().skew_s or 0.0)


def external_controls(ctx: HostCtx, state: InstanceState) -> InstanceControls:
    policy_ok = nv.external_control(ctx.node) == "stop_start"
    manage = nv.mode(ctx.node) == "manage" and nv.link_up(ctx.node)
    if policy_ok and manage:
        running = state != "stopped"
        return InstanceControls(start=not running, stop=running, restart=False, remove=False, logs=False, configure=True,
                                reason="not created by the console: only stop and start are allowed on this machine")
    why = ("this machine is not connected" if not nv.link_up(ctx.node)
           else "this machine's node is in monitor mode" if nv.mode(ctx.node) != "manage"
           else "this machine's policy does not allow controlling engines the console did not create")
    return InstanceControls(configure=True, reason=f"monitor-only: {why}. Clone it as a managed instance to control it.")


def _ext_instance(ctx: HostCtx, c: dict[str, Any], ep: dict[str, Any] | None, iid: str, first_seen: float,
                  now: float) -> Instance:
    state, reason, port, models = endpoint_state(ep, has_ports=bool(nv.host_ports(c)))
    name = str(c["name"])
    gpu_ids, uuids = _gpu_uuids(ctx, c)
    started = _started(ctx, c)
    params = _params_of(c)
    return Instance(
        id=iid, name=display_name(name), engine=str(c.get("engine") or "openai_compatible"),
        repo_id=models[0] if models else display_name(name), params=params, gpu_ids=gpu_ids, gpu_uuids=uuids, port=port,
        container_name=name, image=str(c["image"]) if c.get("image") else None, state=state, created_at=first_seen,
        started_at=started, uptime_s=max(now - started, 0.0) if started else None, managed=False, source="external",
        endpoint=f"http://127.0.0.1:{port}" if port else None, served_models=models, state_reason=reason,
        history_since=first_seen, host_id=ctx.host_id, host_label=ctx.label, management="external",
        vram=vram_for(ctx.node, {name}), host_memory=memory_for(ctx.node, {name}),
        model_location=location_for(ctx, [m for m in c.get("mounts") or [] if isinstance(m, dict)]),
        controls=external_controls(ctx, state))


def _bare_instance(ctx: HostCtx, ep: dict[str, Any], iid: str, first_seen: float) -> Instance:
    state, reason, port, models = endpoint_state(ep, has_ports=True)
    engine = str(ep.get("engine") or "openai_compatible")
    name = f"127.0.0.1:{ep['port']}"
    return Instance(
        id=iid, name=display_name(name), engine=engine, repo_id=models[0] if models else display_name(name), params={},
        gpu_ids=[], gpu_uuids=[], port=port, container_name=None, image=None, state=state, created_at=first_seen,
        managed=False, source="external", endpoint=f"http://127.0.0.1:{port}" if port else None, served_models=models,
        state_reason=reason, history_since=first_seen, host_id=ctx.host_id, host_label=ctx.label, management="external",
        controls=external_controls(ctx, state))


def script_controls(ctx: HostCtx, script: dict[str, Any], running: bool, busy: str | None) -> InstanceControls:
    label = script.get("label") or script.get("name")
    actions = {a.get("name") for a in script.get("actions") or [] if isinstance(a, dict)}
    reason = f"managed by the script '{label}' registered in this machine's policy"
    if not nv.link_up(ctx.node):
        return InstanceControls(reason=f"{reason}; this machine is not connected")
    if nv.mode(ctx.node) != "manage":
        return InstanceControls(reason=f"{reason}; this machine's node is in monitor mode, so scripts cannot run")
    if busy:
        return InstanceControls(reason=f"{reason}; its '{busy}' action is running")
    up, down = "up" in actions, "down" in actions
    return InstanceControls(start=up and not running, stop=down and running, restart=up and down, reason=reason)


def _script_instance(ctx: HostCtx, script: dict[str, Any], matched: list[dict[str, Any]], eps: list[dict[str, Any]],
                     first_seen: float, now: float, busy: str | None) -> Instance:
    name = str(script["name"])
    main = next((c for c in matched if c.get("engine")), matched[0] if matched else None)
    ep = endpoint_of(main, eps) if main else None
    running = bool(matched)
    models: list[str] = []
    port: int | None = None
    reason: str | None = None
    state: InstanceState = "stopped"
    if running:
        state, reason, port, models = endpoint_state(ep, has_ports=bool(main and nv.host_ports(main)))
        if state == "unreachable":   # the container is up and simply not answering yet: it is loading
            state, reason = "loading", "the engine's container is running but does not answer yet"
    if busy == "down" and running:
        state = "stopping"
    elif busy == "up" and not running:
        state = "starting"
    names = {str(c["name"]) for c in matched}
    gpu_ids: list[int] = []
    uuids: list[str] = []
    for c in matched:
        g, u = _gpu_uuids(ctx, c)
        gpu_ids += [x for x in g if x not in gpu_ids]
        uuids += [x for x in u if x not in uuids]
    started = min((s for c in matched if (s := _started(ctx, c)) is not None), default=None)
    label = clean_text(script.get("label") or name, 64).strip()   # a human label: only control characters and length are cleaned
    params = _params_of(main) if main else {}
    return Instance(
        id=script_instance_id(ctx.host_id, name), name=label or name,
        engine=str(script.get("engine") or (main or {}).get("engine") or "script"),
        repo_id=models[0] if models else label or name, params=params, gpu_ids=sorted(gpu_ids), gpu_uuids=uuids,
        port=port, container_name=str(main["name"]) if main else None, image=str(main["image"]) if main and main.get("image") else None,
        state=state, phase=f"script: {busy} is running" if busy else None, created_at=first_seen, started_at=started,
        uptime_s=max(now - started, 0.0) if started and running else None, managed=False, source="external",
        endpoint=f"http://127.0.0.1:{port}" if port else None, served_models=models, state_reason=reason,
        history_since=first_seen, host_id=ctx.host_id, host_label=ctx.label, management="script", script=name,
        vram=vram_for(ctx.node, names), host_memory=memory_for(ctx.node, names),
        model_location=location_for(ctx, [m for c in matched for m in c.get("mounts") or [] if isinstance(m, dict)]),
        controls=script_controls(ctx, script, running, busy))


def matches_script(name: str, script: dict[str, Any]) -> bool:
    return any(isinstance(p, str) and fnmatch.fnmatchcase(name, p) for p in script.get("containers") or [])


def build_host_instances(ctx: HostCtx, first_seen: dict[str, float], now: float,
                         busy: Mapping[str, str] | None = None) -> list[Instance]:
    """Script instances (one per policy script) and external instances of one host, from its latest inventory.

    `first_seen` is updated in place with the first time each id was seen (the history of its metrics starts there).
    `busy` maps a script name to the action currently running for it."""
    node = ctx.node
    scripts = [s for s in nv.scripts(node) if _SCRIPT_NAME.fullmatch(str(s["name"]))]
    eps = nv.endpoints(node)
    conts = [c for c in nv.containers(node) if c.get("state") == "running"]
    owned_ports = {p for c in nv.containers(node) if c.get("managed") for p in nv.host_ports(c)}
    out: list[Instance] = []
    claimed: set[str] = set()
    for s in scripts:
        matched = [c for c in conts if not c.get("managed") and matches_script(str(c["name"]), s)]
        claimed |= {str(c["name"]) for c in matched}
        iid = script_instance_id(ctx.host_id, str(s["name"]))
        out.append(_script_instance(ctx, s, matched, eps, first_seen.setdefault(iid, now), now, (busy or {}).get(str(s["name"]))))
    taken: set[str] = {i.id for i in out}
    covered_ports: set[int] = set(owned_ports)
    for c in sorted(conts, key=lambda x: str(x["name"])):
        name = str(c["name"])
        covered_ports |= set(nv.host_ports(c))
        if c.get("managed") or name in claimed or not c.get("engine"):
            continue
        iid = external_instance_id(ctx.host_id, name)
        if iid in taken:   # slug collision: disambiguate deterministically from the raw name
            iid += "-" + hashlib.sha1(name.encode(), usedforsecurity=False).hexdigest()[:6]
        taken.add(iid)
        out.append(_ext_instance(ctx, c, endpoint_of(c, eps), iid, first_seen.setdefault(iid, now), now))
    for ep in sorted(eps, key=lambda e: int(e["port"])):   # bare endpoints: an engine with no container of its own
        if ep.get("container") or int(ep["port"]) in covered_ports:
            continue
        if not (ep.get("engine") or (_answered(ep) and nv.ok_status(ep.get("models_status"))
                                     and (nv.ok_status(ep.get("metrics_status")) or nv.ok_status(ep.get("health_status"))))):
            continue   # not an engine we can recognise: never listed
        iid = external_instance_id(ctx.host_id, f"127.0.0.1:{ep['port']}")
        if iid in taken:
            iid += "-" + hashlib.sha1(str(ep["port"]).encode(), usedforsecurity=False).hexdigest()[:6]
        taken.add(iid)
        out.append(_bare_instance(ctx, ep, iid, first_seen.setdefault(iid, now)))
    return out
