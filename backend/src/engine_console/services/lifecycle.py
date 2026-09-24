"""Instance lifecycle: preflight -> docker run -> supervised state machine
stopped -> starting -> loading -> ready -> stopping | failed.

`tick()` is the supervisor step (called every couple of seconds by a background task, or directly in tests).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import socket
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from engine_console.adapters import AdapterRegistry
from engine_console.adapters.base import EngineAdapter, LaunchSpec
from engine_console.config import Settings
from engine_console.domain.errors import BadRequest, Conflict, NotFound, ProblemError, Unprocessable
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
from engine_console.domain.repo_id import check_repo_id
from engine_console.services.common import EventBus, new_id, now, scrub
from engine_console.services.discovery import EXT, DiscoveryService, ScrapeTarget
from engine_console.services.docker import (
    INSTANCE_LABEL,
    LABEL,
    ROLE_LABEL,
    ContainerSpec,
    DockerCli,
    GatewaySpec,
    command_snippets,
    ensure_internal_network,
    start_gateway,
)
from engine_console.services.downloads import DownloadService
from engine_console.services.fitting import FitService
from engine_console.services.hardware import HardwareService
from engine_console.services.hf import build_model_info
from engine_console.services.hostmem import HostMemService
from engine_console.services.secrets_store import MARKER, SECRET_KEY, SecretStore, split_secrets
from engine_console.services.settings import SettingsService, image_allowed
from engine_console.services.store import Store, paginate

log = logging.getLogger(__name__)
SECRET_ENV = re.compile(r"(key|token|secret|password)", re.I)
# compile/JIT caches survive container recreation; named volumes need no host path
NAMED_VOLUMES: dict[str, list[tuple[str, str]]] = {
    "vllm": [("ec-vllm-cache", "/root/.cache/vllm"), ("ec-flashinfer-cache", "/root/.cache/flashinfer")],
}
ACTIVE: tuple[str, ...] = ("starting", "loading", "ready", "stopping")
TOPIC = "instances"


def _bind_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "model"


_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.\-]{0,63}")


def gateway_name(container_name: str) -> str:
    return f"{container_name}-gw"


def _mask(text: str, secrets: list[str]) -> str:
    for sv in secrets:
        text = text.replace(sv, "[redacted]")
    return text


class LifecycleService:
    discovery: DiscoveryService | None = None   # set by the composition root
    hostmem: HostMemService | None = None       # set by the composition root

    @staticmethod
    def _cn(inst: Instance) -> str:
        """Container name of a console-owned instance; external engines are monitor-only and never reach docker."""
        if not inst.managed or inst.container_name is None:
            raise Conflict("this is an external engine the console does not own (monitor-only)", code="instance_not_managed")
        return inst.container_name

    def __init__(self, store: Store, adapters: AdapterRegistry, docker: DockerCli, hardware: HardwareService,
                 settings: SettingsService, downloads: DownloadService, fit: FitService,
                 http: httpx.AsyncClient, cfg: Settings, bus: EventBus,
                 port_free: Callable[[int], bool] = _bind_free, secrets: SecretStore | None = None) -> None:
        self._db = store
        self._adapters = adapters
        self._docker = docker
        self._hw = hardware
        self._settings = settings
        self._downloads = downloads
        self._fit = fit
        self._http = http
        self._cfg = cfg
        self._bus = bus
        self._port_free = port_free
        self._pids: dict[str, int] = {}
        self._secrets = secrets or SecretStore(cfg.data_dir / "secrets")
        self._lock = asyncio.Lock()
        fit.set_resident_provider(self.resident)

    # -- persistence -------------------------------------------------------------------------------
    def host_memory(self, iid: str) -> HostMemory | None:
        if self.hostmem is None:
            return None
        pid = self.discovery.pid(iid) if iid.startswith(EXT) and self.discovery else self._pids.get(iid, 0)
        return self.hostmem.container(pid)

    def _to_model(self, r: Any) -> Instance:
        started = r["started_at"]
        up = now() - started if started and r["state"] in ("starting", "loading", "ready") else None
        return Instance(id=r["id"], name=r["name"], engine=r["engine"], repo_id=r["repo_id"], params=json.loads(r["params"]),
                        gpu_ids=json.loads(r["gpu_ids"]), gpu_uuids=json.loads(r["gpu_uuids"]), port=r["port"],
                        container_name=r["container_name"], container_port=r["container_port"],
                        internal_endpoint=(f"http://{r['container_name']}:{r['container_port']}"
                                           if r["container_port"] else None), image=r["image"], state=r["state"], phase=r["phase"],
                        progress_pct=r["progress_pct"], pinned=bool(r["pinned"]), ttl_idle_s=r["ttl_idle_s"],
                        profile_id=r["profile_id"], error=r["error"], last_logs=r["last_logs"],
                        fit=json.loads(r["fit"]) if r["fit"] else None, created_at=r["created_at"], started_at=started,
                        last_request_at=r["last_request_at"], uptime_s=up,
                        endpoint=f"http://127.0.0.1:{r['port']}" if r["port"] else None,
                        host_memory=self.host_memory(r["id"]) if r["state"] in ACTIVE else None)

    def get(self, iid: str) -> Instance:
        if iid.startswith(EXT):
            ext = self.discovery.get(iid) if self.discovery else None
            if ext is None:
                raise NotFound(f"no such instance: {iid}")
            return self._with_mem(ext)
        r = self._db.one("SELECT * FROM instances WHERE id=?", (iid,))
        if r is None:
            raise NotFound(f"no such instance: {iid}")
        return self._to_model(r)

    def list_page(self, limit: int = 100, cursor: str | None = None) -> Page[Instance]:
        limit, off = paginate(limit, cursor)
        rows = self._db.all("SELECT * FROM instances ORDER BY created_at DESC")
        # console-owned first, then discovered externals; a discovered engine never duplicates an owned one
        items = [self._to_model(r) for r in rows] + [self._with_mem(i) for i in (self.discovery.snapshot() if self.discovery else [])]
        return Page[Instance](items=items[off: off + limit], next_cursor=str(off + limit) if len(items) > off + limit else None)

    def _with_mem(self, inst: Instance) -> Instance:
        return inst.model_copy(update={"host_memory": self.host_memory(inst.id)}) if inst.state != "stopped" else inst

    def require_managed(self, iid: str) -> Instance:
        inst = self.get(iid)
        self._cn(inst)
        return inst

    def owned_ports(self) -> set[int]:
        return {r["port"] for r in self._db.all("SELECT port FROM instances WHERE port IS NOT NULL")}

    def scrape_targets(self) -> list[ScrapeTarget]:
        out = [ScrapeTarget(i.id, i.engine, i.port, self.launch_for(i).metrics_path)
               for i in self.all_active() if i.state == "ready" and i.port]
        return out + (self.discovery.targets() if self.discovery else [])

    def all_active(self) -> list[Instance]:
        return [self._to_model(r) for r in self._db.all(
            "SELECT * FROM instances WHERE state IN ('starting','loading','ready','stopping')")]

    def _set(self, iid: str, **cols: Any) -> None:
        for k in ("params", "gpu_ids", "gpu_uuids", "fit"):
            if k in cols and not isinstance(cols[k], str) and cols[k] is not None:
                cols[k] = json.dumps(cols[k])
        sets = ", ".join(f"{k}=?" for k in cols)   # column names are literals from this module
        self._db.execute(f"UPDATE instances SET {sets} WHERE id=?", [*cols.values(), iid])  # noqa: S608
        self._bus.publish(TOPIC, self.get(iid).model_dump())

    def resident(self) -> list[ResidentUse]:
        out: list[ResidentUse] = []
        for i in self.all_active():
            if i.state == "stopping":
                continue
            per = (i.fit or {}).get("per_gpu") or []
            gib = float(per[0]["total_gib"]) if per else 0.0
            out.append(ResidentUse(name=i.name, gpu_ids=i.gpu_ids, gib_per_gpu=gib))
        return out

    def uses_repo(self, repo_id: str) -> bool:
        return self._db.one("SELECT 1 FROM instances WHERE repo_id=? AND state IN "
                            "('starting','loading','ready','stopping')", (repo_id,)) is not None

    def touch(self, iid: str) -> None:
        self._db.execute("UPDATE instances SET last_request_at=? WHERE id=?", (now(), iid))

    # -- container spec ----------------------------------------------------------------------------
    def _spec(self, adapter: EngineAdapter, inst_id: str, name: str, repo_id: str, params: dict[str, Any],
              gpu_ids: list[int], port: int, container: str
              ) -> tuple[ContainerSpec, LaunchSpec, list[str], str, dict[str, str]]:
        hw = self._hw.hardware(gpu_ids)
        launch = adapter.build_launch(repo_id, params, hw, served_name=repo_id,
                                      hf_cache_container_path=self._cfg.hf_cache_container_path)
        # Only the settings pin or the adapter default: a per-instance `image` param is rejected in validation
        # and launch.image is deliberately ignored, so a request can never choose what runs on the GPU.
        image = self._settings.image_pin(adapter.id) or adapter.default_image
        if not self._settings.engine_image_allowed(image):
            raise Unprocessable(f"image {image} is not an allowed engine image (ENGINE_IMAGE_ALLOWLIST / adapter pins)",
                                code="image_not_allowed")
        secret_values = [v for k, v in params.items() if SECRET_KEY.search(k) and isinstance(v, str) and v and v != MARKER]
        if any(sv in arg for sv in secret_values for arg in launch.argv):
            raise Unprocessable("this engine adapter would put a secret parameter (e.g. api_key) on the container "
                                "command line, where any local user could read it; unset it or use an engine that "
                                "takes it from an environment variable", code="secret_in_argv")
        env: dict[str, str] = {}
        secret_env: dict[str, str] = {}
        passthrough: list[str] = []
        for k, v in launch.env.items():
            if SECRET_ENV.search(k):
                secret_env[k] = v        # e.g. VLLM_API_KEY: value-less `-e NAME`, value rides the child env only
                passthrough.append(k)
            else:
                env[k] = v
        if self._cfg.engine_hf_offline:
            env.setdefault("HF_HUB_OFFLINE", "1")
        elif self._settings.hf_token():
            passthrough.append("HF_TOKEN")   # by name only: the value rides the child env, never argv
            if (tok := self._settings.hf_token()):
                secret_env["HF_TOKEN"] = tok
        spec = ContainerSpec(
            name=container, image=image, argv=list(launch.argv), host_port=None, container_port=launch.container_port,
            gpu_uuids=list(hw.gpu_uuids), network=self._cfg.engine_network, network_alias=container,
            labels={ROLE_LABEL: "engine", f"{LABEL}.instance": inst_id, f"{LABEL}.engine": adapter.id, f"{LABEL}.model": repo_id,
                    f"{LABEL}.port": str(port), f"{LABEL}.name": name,
                    f"{LABEL}.gpus": ",".join(hw.gpu_uuids)},
            env=env, env_passthrough=passthrough,
            mounts=[(str(self._settings.hf_cache_dir()), self._cfg.hf_cache_container_path, self._cfg.hf_cache_readonly),
                    *[(vol, path, False) for vol, path in NAMED_VOLUMES.get(adapter.id, [])]],
            # host IPC shares the host's SysV/POSIX IPC namespace: only worth it for multi-GPU NCCL; a single-GPU
            # engine gets a private namespace plus the large /dev/shm from shm_size.
            shm_size=launch.shm_size, ipc_host=launch.ipc_host and len(hw.gpu_uuids) > 1)
        return spec, launch, hw.gpu_uuids, image, secret_env

    def _real_params(self, inst: Instance) -> dict[str, Any]:
        """Stored params with MARKER placeholders replaced by the real secret values (memory only)."""
        stored = self._secrets.get(inst.id)
        return {k: stored.get(k, v) if v == MARKER else v for k, v in inst.params.items()}

    def _param_errors(self, adapter: EngineAdapter, params: dict[str, Any]) -> list[str]:
        errs = [f"'{k}' is a per-instance image override; images come from settings pins only"
                for k in (s.key for s in adapter.param_catalog() if s.flag == "@image") if k in params]
        errs += [f"'{k}' still holds the [set] placeholder: supply the real value" for k, v in params.items() if v == MARKER]
        real = {k: v for k, v in params.items() if v != MARKER}
        errs += adapter.validate(real)
        secret_values = [v for k, v in params.items() if SECRET_KEY.search(k) and isinstance(v, str) and v and v != MARKER]
        return [_mask(e, secret_values) for e in errs]

    def _alloc_port(self, exclude_id: str | None, preferred: int | None = None) -> int | None:
        used = {r["port"] for r in self._db.all(
            "SELECT port FROM instances WHERE state IN ('starting','loading','ready','stopping') AND id != ?",
            (exclude_id or "",)) if r["port"]}
        cands = ([preferred] if preferred else []) + list(range(self._cfg.port_range_start, self._cfg.port_range_end + 1))
        for p in cands:
            if p not in used and self._port_free(p):
                return p
        return None

    # -- preflight ---------------------------------------------------------------------------------
    async def preflight(self, req: InstanceCreate, *, exclude_id: str | None = None,
                        params: dict[str, Any] | None = None) -> tuple[PreflightReport, int | None]:
        adapter = self._adapters.get(req.engine)
        p = params if params is not None else req.params
        checks: list[PreflightCheck] = []
        fit: FitReport | None = None
        port = self._alloc_port(exclude_id)

        errs = self._param_errors(adapter, p)
        if errs:
            checks.append(PreflightCheck(code="invalid_params", level="block", message="; ".join(errs)))
        hub = self._downloads.local_hub_model(req.repo_id)
        gpu_ids = req.gpu_ids or self._settings.default_gpu_ids()
        if hub is not None and not errs:
            gpu_ids = self._fit.resolve_gpus(req.engine, build_model_info(hub), p, req.gpu_ids)
        if hub is None:
            checks.append(PreflightCheck(code="model_not_cached", level="block",
                                         message=f"{req.repo_id} is not in the local cache: download it first"))
        elif not errs:
            fit = self._fit.assess_info(req.engine, build_model_info(hub), p, gpu_ids)
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
        if port is None:
            checks.append(PreflightCheck(code="no_free_port", level="block",
                                         message=f"no free port in {self._cfg.port_range_start}-{self._cfg.port_range_end}"))
        else:
            checks.append(PreflightCheck(code="port", level="ok", message=f"host port {port}"))
        hw = self._hw.hardware(gpu_ids or None)
        if not hw.gpu_ids:
            checks.append(PreflightCheck(code="no_gpu", level="block", message="no GPU available (NVML and nvidia-smi found none)"))
        else:
            clash = [i.name for i in self.all_active() if i.id != exclude_id and set(i.gpu_ids) & set(hw.gpu_ids)]
            if clash:
                checks.append(PreflightCheck(code="gpu_shared", level="warn",
                                             message=f"GPU already used by {', '.join(clash)}; relying on free VRAM"))
        image = self._settings.image_pin(req.engine) or adapter.default_image
        if not self._settings.engine_image_allowed(image):
            checks.append(PreflightCheck(code="image_not_allowed", level="block",
                                         message=f"image {image} is not an allowed engine image"))
        elif not errs and hub is not None:
            try:
                self._spec(adapter, "preflight", "preflight", req.repo_id, p, gpu_ids, port or 0, "preflight")
            except ProblemError as e:
                checks.append(PreflightCheck(code=e.code, level="block", message=e.detail))
        if not await self._docker.image_present(image):
            checks.append(PreflightCheck(code="image_missing", level="block",
                                         message=f"image {image} is not present locally: docker pull it once (the console never pulls)"))
        return PreflightReport(ok=not any(c.level == "block" for c in checks), gpu_ids=gpu_ids, checks=checks, fit=fit), port

    # -- commands ------------------------------------------------------------------------------------
    async def create(self, req: InstanceCreate, profile_params: dict[str, Any] | None = None) -> Instance:
        params = {**(profile_params or {}), **req.params}
        public, secret_values = split_secrets(params)
        async with self._lock:
            report, port = await self.preflight(req, params=params)
            if not report.ok or port is None:
                raise Conflict("preflight failed", code="preflight_failed", preflight=report.model_dump())
            adapter = self._adapters.get(req.engine)
            iid = new_id("inst_")
            name = req.name or req.repo_id.split("/")[-1]
            container = f"ec-{_slug(name)}-{iid[-6:]}"
            spec, _, uuids, image, secrets_env = self._spec(adapter, iid, name, req.repo_id, params, report.gpu_ids, port,
                                                            container)
            hw = self._hw.hardware(report.gpu_ids)
            self._db.execute(
                "INSERT INTO instances(id,name,engine,repo_id,params,gpu_ids,gpu_uuids,port,container_name,image,state,"
                "profile_id,ttl_idle_s,fit,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (iid, name, req.engine, req.repo_id, json.dumps(public), json.dumps(hw.gpu_ids), json.dumps(uuids), port,
                 container, image, "starting", req.profile_id, req.ttl_idle_s,
                 report.fit.model_dump_json() if report.fit else None, now()))
            self._secrets.put(iid, secret_values)
            await self._launch(iid, spec, secrets_env)
        return self.get(iid)

    def _gateway_spec(self, iid: str, container: str, host_port: int, engine_port: int) -> GatewaySpec:
        image = self._cfg.gateway_image
        if not image_allowed(image, self._cfg.gateway_image_allowlist):
            raise Unprocessable(f"gateway image {image} is not on GATEWAY_IMAGE_ALLOWLIST", code="image_not_allowed")
        return GatewaySpec(name=gateway_name(container), image=image, host_port=host_port, engine_host=container,
                           engine_port=engine_port, internal_network=self._cfg.engine_network, instance_id=iid)

    async def _remove_containers(self, iid: str, container: str) -> None:
        await self._docker.remove(gateway_name(container), role="gateway", instance=iid)
        await self._docker.remove(container)

    async def _launch(self, iid: str, spec: ContainerSpec, secrets_env: dict[str, str], host_port: int | None = None) -> None:
        port = host_port if host_port is not None else self.get(iid).port
        try:
            await ensure_internal_network(self._docker, self._cfg.engine_network)
            await self._remove_containers(iid, spec.name)   # clear stale containers of the same name (label-verified)
            await self._docker.run_container(spec, secrets_env or None)
            if port is None:
                raise ProblemError("no host port allocated for the gateway", code="no_free_port", status=409)
            try:
                await start_gateway(self._docker, self._gateway_spec(iid, spec.name, port, spec.container_port))
            except ProblemError:
                await self._docker.remove(spec.name)   # never leave an engine running without its front door
                raise
            self._set(iid, state="starting", started_at=now(), error=None, phase=None, progress_pct=None,
                      last_logs=None, container_port=spec.container_port)
        except ProblemError as e:
            self._set(iid, state="failed", error=e.detail)
            raise

    async def start(self, iid: str) -> Instance:
        inst = self.require_managed(iid)
        if inst.state in ("starting", "loading", "ready"):
            raise Conflict(f"instance is already {inst.state}")
        async with self._lock:
            req = InstanceCreate(engine=inst.engine, repo_id=inst.repo_id, params=inst.params, gpu_ids=inst.gpu_ids,
                                 name=inst.name)   # a restart keeps the GPUs it was created on
            real = self._real_params(inst)
            report, port = await self.preflight(req, exclude_id=iid, params=real)
            if not report.ok or port is None:
                raise Conflict("preflight failed", code="preflight_failed", preflight=report.model_dump())
            adapter = self._adapters.get(inst.engine)
            spec, _, uuids, image, secrets_env = self._spec(adapter, iid, inst.name, inst.repo_id, real, inst.gpu_ids,
                                                            port, self._cn(inst))
            self._set(iid, port=port, gpu_uuids=uuids, image=image, state="starting",
                      fit=report.fit.model_dump_json() if report.fit else None)
            await self._launch(iid, spec, secrets_env)
        return self.get(iid)

    async def stop(self, iid: str) -> Instance:
        inst = self.require_managed(iid)
        if inst.state in ("stopped",):
            return inst
        self._set(iid, state="stopping")
        try:
            await self._docker.stop(gateway_name(self._cn(inst)), role="gateway", instance=iid)
            await self._docker.stop(self._cn(inst))
        except ProblemError as e:
            self._set(iid, state="failed", error=e.detail)
            raise
        self._set(iid, state="stopped", phase=None, progress_pct=None)
        return self.get(iid)

    async def restart(self, iid: str) -> Instance:
        await self.stop(iid)
        return await self.start(iid)

    async def remove(self, iid: str) -> None:
        inst = self.require_managed(iid)
        await self._remove_containers(iid, self._cn(inst))
        self._secrets.delete(iid)
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

    def _scrub(self, inst: Instance, text: str) -> str:
        """Engines may echo their own args (api keys) into logs: mask known secret values, then token shapes."""
        real = self._real_params(inst)
        values = list(self._secrets.get(inst.id).values())
        with contextlib.suppress(ProblemError):   # e.g. secret_in_argv: the stored values above are still masked
            values += list(self._spec(self._adapters.get(inst.engine), inst.id, inst.name, inst.repo_id, real,
                                      inst.gpu_ids, inst.port or self._cfg.port_range_start,
                                      self._cn(inst))[4].values())
        for v in values:
            if len(v) >= 6:
                text = text.replace(v, "[redacted]")
        return scrub(text)

    async def logs(self, iid: str, tail: int = 200) -> str:
        inst = self.require_managed(iid)
        return self._scrub(inst, await self._docker.logs(self._cn(inst), max(1, min(tail, 5000))))

    async def follow_logs(self, iid: str, tail: int = 100) -> AsyncIterator[str]:
        inst = self.require_managed(iid)
        async for line in self._docker.follow_logs(self._cn(inst), tail):
            yield self._scrub(inst, line)

    def command(self, iid: str) -> CommandSnippets:
        inst = self.require_managed(iid)
        adapter = self._adapters.get(inst.engine)
        spec, _, _, _, _ = self._spec(adapter, inst.id, inst.name, inst.repo_id, inst.params, inst.gpu_ids,
                                      inst.port or self._cfg.port_range_start, self._cn(inst))
        return CommandSnippets(**command_snippets(spec, self._cfg.docker_context))

    # -- supervisor ------------------------------------------------------------------------------------
    async def tick(self) -> None:
        if self.discovery is not None:
            try:
                await self.discovery.maybe_refresh()   # re-discover externals every discovery_interval_s
            except Exception:  # noqa: BLE001 - never let discovery break supervision
                log.exception("discovery refresh failed")
        for inst in self.all_active():
            try:
                await self._supervise(inst)
            except Exception:  # noqa: BLE001 - one bad instance must not stop supervision of the rest
                log.exception("supervisor error for %s", inst.id)

    async def _supervise(self, inst: Instance) -> None:
        if inst.state == "stopping":
            return
        info = await self._docker.inspect(self._cn(inst))
        if info is not None and info.running and info.pid:
            self._pids[inst.id] = info.pid          # cached for cgroup memory reads (no extra docker calls)
        if info is None or not info.running:
            logs = await self._docker.logs(self._cn(inst), 200) if info is not None else ""
            code = f" with code {info.exit_code}" if info is not None else ""
            self._set(inst.id, state="failed", error=f"container exited{code}" if info else "container disappeared",
                      last_logs=self._scrub(inst, logs)[-20000:] or None, phase=None)
            with contextlib.suppress(ProblemError):
                await self._docker.stop(gateway_name(self._cn(inst)), role="gateway", instance=inst.id)
            return
        if info.labels.get(ROLE_LABEL) == "engine":   # legacy (pre-gateway) containers publish their own port
            gw = await self._docker.inspect(gateway_name(self._cn(inst)))
            own = gw is not None and gw.labels.get(ROLE_LABEL) == "gateway" and gw.labels.get(INSTANCE_LABEL) == inst.id
            if gw is None or not own or not gw.running:
                self._set(inst.id, state="failed", phase=None,
                          error="gateway container missing or stopped; restart the instance")
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
        for line in (await self._docker.logs(self._cn(inst), 40)).splitlines():
            parsed = adapter.parse_startup_log(line)
            if parsed:
                phase = str(parsed.get("phase", phase)) if parsed.get("phase") else phase
                if parsed.get("pct") is not None:
                    pct = float(parsed["pct"])
        if phase and phase != "ready" and state == "starting":
            state = "loading"
        if await self._probe(inst, adapter):
            state, phase, pct = "ready", "ready", 100.0
        elif inst.started_at and now() - inst.started_at > self._cfg.startup_timeout_s:
            logs = await self._docker.logs(self._cn(inst), 200)
            self._set(inst.id, state="failed", error="startup timed out", last_logs=self._scrub(inst, logs)[-20000:] or None)
            return
        if (state, phase, pct) != (inst.state, inst.phase, inst.progress_pct):
            self._set(inst.id, state=state, phase=phase, progress_pct=pct)

    def launch_for(self, inst: Instance) -> LaunchSpec:
        """The adapter's LaunchSpec for an instance (health/metrics paths, ports)."""
        _, launch, _, _, _ = self._spec(self._adapters.get(inst.engine), inst.id, inst.name, inst.repo_id, inst.params,
                                     inst.gpu_ids, inst.port or self._cfg.port_range_start, self._cn(inst))
        return launch

    async def _probe(self, inst: Instance, adapter: EngineAdapter) -> bool:
        if inst.port is None:
            return False
        launch = self.launch_for(inst)
        try:
            r = await self._http.get(f"http://127.0.0.1:{inst.port}{launch.health_path}", timeout=3.0)
        except httpx.HTTPError:
            return False
        return r.status_code == 200

    async def adopt(self) -> int:
        """On start-up: re-attach to labelled containers that outlived the console."""
        n = 0
        for cname in await self._docker.list_console_containers():
            info = await self._docker.inspect(cname)
            if info is None or info.labels.get(ROLE_LABEL) == "gateway":
                continue   # gateways are only ever handled as part of their own instance
            row = self._db.one("SELECT id FROM instances WHERE container_name=?", (cname,))
            if row is not None:
                if info.running:
                    self._set(row["id"], state="loading")   # supervisor promotes to ready via the health probe
                elif self.get(row["id"]).state in ACTIVE:
                    self._set(row["id"], state="stopped")
                continue
            lab = info.labels
            iid = lab.get(f"{LABEL}.instance")
            engine, model = lab.get(f"{LABEL}.engine"), lab.get(f"{LABEL}.model")
            if not (iid and engine and model) or engine not in self._adapters:
                continue
            try:   # labels are attacker-influenced strings: validate before they reach SQL rows, paths or UI
                check_repo_id(model)
                if not (_ID_RE.fullmatch(iid) and _ID_RE.fullmatch(engine)
                        and _NAME_RE.fullmatch(lab.get(f"{LABEL}.name", cname))):
                    raise ValueError("bad label")
            except ValueError:
                log.warning("not adopting %s: invalid engine-console.* labels", cname)
                continue
            try:
                port = int(lab.get(f"{LABEL}.port", 0))
            except ValueError:
                continue
            if not self._cfg.port_range_start <= port <= self._cfg.port_range_end:
                log.warning("not adopting %s: label port outside the console range", cname)
                continue
            hw_all = self._hw.report()
            uuid_to_idx = {g.uuid: g.index for g in hw_all.gpus}
            uuids = [u for u in lab.get(f"{LABEL}.gpus", "").split(",") if u]
            self._db.execute(
                "INSERT INTO instances(id,name,engine,repo_id,params,gpu_ids,gpu_uuids,port,container_name,image,state,created_at,started_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (iid, lab.get(f"{LABEL}.name", cname), engine, model, "{}",
                 json.dumps([uuid_to_idx[u] for u in uuids if u in uuid_to_idx]), json.dumps(uuids), port, cname, self._adapters.get(engine).default_image,
                 "loading" if info.running else "stopped", now(), now() if info.running else None))
            n += 1
        return n
