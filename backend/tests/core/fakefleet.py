"""An in-process fleet for tests: `FakeNode` implements `domain.fleet.NodeClient`, `FakeFleet` implements `FleetPort`,
`FakeLibrary` the slice of the model library the runtime services use.

The fake node behaves like a real `ecnode` closely enough that the hub's runtime services are exercised end to end
without docker: `docker.run` creates a container that then shows up in the pushed `inventory`, `gateway.start`
publishes a port (or answers `port_in_use`), `docker.stop/rm` enforce ownership, `http.request` obeys the node's
port and path rules, `scripts.run` streams lines. Fidelity is enforced where it matters:

* every request's arguments go through `ecnode.proto.clean_args` and must come out unchanged (a misspelt or
  over-long argument fails the test instead of being silently dropped, as the real node would drop it);
* monitor mode refuses every op outside `proto.MONITOR_OPS`;
* every pushed message and every result is rebuilt with `proto.sanitize_push` / `proto.clean`, like the hub's client.

Put test data on the node (`add_container`, `add_endpoint`, `set_gpu_free`, ...) and read what the hub asked for from
`node.calls` / `node.ops(name)`."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from ecnode import proto

from engine_console.adapters.base import ModelInfo
from engine_console.domain.errors import NotFound
from engine_console.domain.fleet import (
    HostConfig,
    LibraryCopy,
    LinkStatus,
    MountInfo,
    NodeError,
    NodeKind,
    NodeUnavailable,
    RootInfo,
    StreamItem,
)
from engine_console.domain.ports import HubModel, RepoFile
from engine_console.domain.refs import ModelRef
from engine_console.services.hf import build_model_info

GIB = 1024**3
HTTP_PREFIXES = ("/v1/", "/health", "/metrics", "/version", "/api/", "/get_model_info", "/get_server_info", "/server_info")
DEFAULT_ROOTS: list[dict[str, Any]] = [
    {"id": "hf", "path": "/data/hf", "kind": "hf-home", "writable": True, "label": "HF cache"},
    {"id": "gguf", "path": "/data/gguf", "kind": "gguf", "writable": True, "label": "GGUF files"},
    {"id": "ollama", "path": "/data/ollama", "kind": "ollama", "writable": False, "label": "Ollama"},
]


def make_hello(*, hostname: str = "testbox", arch: str = "x86_64", gpus: int = 2, gpu_gib: float = 96.0, cc: str = "12.0",
               unified: bool = False, mode: str = "manage", engine_images: tuple[str, ...] = ("fake/engine:*",),
               gateway_image: str | None = "fake/gateway:1", engine_network: str = "engine-console-engines",
               port_range: tuple[int, int] = (18000, 18099), roots: list[dict[str, Any]] | None = None,
               scripts: list[dict[str, Any]] | None = None, external_control: str = "none", allow_ipc_host: bool = True,
               max_shm_gib: float | None = 64.0, addresses: tuple[str, ...] = ("10.0.0.5",),
               policy_path: str = "/etc/ecnode/policy.json", ram_gib: float = 246.0, uuid_prefix: str = "GPU-uuid-",
               source: str = "nvml") -> dict[str, Any]:
    msg = {
        "type": "hello", "proto": proto.PROTO, "version": "0.1.0", "node_id": "n" * 16,
        "host": {"hostname": hostname, "arch": arch, "os": "Linux", "kernel": "6.x", "cpu_model": "cpu", "cpu_cores": 16,
                 "cpu_threads": 32, "ram_total_bytes": int(ram_gib * GIB), "boot_time": 1.0, "python": "3.12.0"},
        "gpu": {"driver": "595.91", "cuda": "13.0", "source": source},
        "gpus": [{"index": i, "uuid": f"{uuid_prefix}{i}", "name": "RTX PRO 6000 Blackwell", "total_bytes": int(gpu_gib * GIB),
                  "compute_capability": cc, "unified_memory": unified} for i in range(gpus)],
        "docker": {"available": True, "context": "rootless", "rootless": True, "version": "29.6", "nvidia_runtime": True},
        "addresses": [{"iface": "eth0", "addr": a, "prefix": 24} for a in addresses],
        "policy": {"path": policy_path, "mode": mode, "engine_images": list(engine_images), "gateway_image": gateway_image,
                   "engine_network": engine_network, "port_range": list(port_range), "model_roots": roots if roots is not None
                   else copy.deepcopy(DEFAULT_ROOTS), "scripts": scripts or [], "allow_pull": False, "allow_delete": False,
                   "allow_ipc_host": allow_ipc_host, "external_control": external_control, "max_shm_gib": max_shm_gib},
        "forced_cmd": None, "via_ssh": False}
    out = proto.sanitize_push(msg)
    assert out is not None
    return out


BIG_MODEL_SCRIPT: dict[str, Any] = {
    "name": "big-model", "label": "Big model (2 GPUs)", "engine": "sglang", "description": "Starts before the small one.",
    "containers": ["myproj-sglang-*"], "status_action": "status",
    "actions": [{"name": "up", "label": "Start", "confirm": True}, {"name": "down", "label": "Stop", "danger": True},
                {"name": "status", "label": "Check status"},
                {"name": "switch", "label": "Switch model", "args": ["model-a", "model-b"], "confirm": True}]}


def cgroup(current: float = 68.6, anon: float = 3.3, file: float = 64.5, shmem: float = 64.0, kernel: float = 0.6) -> dict[str, Any]:
    """A container's cgroup memory block in bytes (numbers measured on a real SGLang container by the v1 tests)."""
    return {"mem": {"current": int(current * GIB), "anon": int(anon * GIB), "file": int(file * GIB),
                    "shmem": int(shmem * GIB), "kernel": int(kernel * GIB), "source": "cgroup"}, "cpu_pct": 1.0}


def ports(host_port: int, container_port: int = 8000, ip: str = "0.0.0.0") -> list[dict[str, Any]]:  # noqa: S104
    return [{"host_ip": ip, "host_port": host_port, "container_port": container_port}]


class FailedStream:
    """What a real client returns from open() when the link is down: the error shows up when the stream is used."""

    def __init__(self, error: NodeError) -> None:
        self._error = error

    def __aiter__(self) -> AsyncIterator[StreamItem]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[StreamItem]:
        raise self._error
        yield   # pragma: no cover

    async def write(self, data: bytes) -> None:
        raise self._error

    async def end(self) -> None:
        raise self._error

    async def result(self) -> dict[str, Any]:
        raise self._error

    async def cancel(self) -> None:
        return None


class FakeStream:
    """One streaming request. `producer(stream)` yields the events and DATA the node would send; `finish()` gives the
    res result (or raises the NodeError the node would answer with)."""

    def __init__(self, producer: Callable[[FakeStream], AsyncIterator[StreamItem]], finish: Callable[[], dict[str, Any]]) -> None:
        self._producer = producer
        self._finish = finish
        self.uploaded = bytearray()
        self.ended = asyncio.Event()
        self.cancelled = False
        self.cancel_event = asyncio.Event()
        self.done = False

    def __aiter__(self) -> AsyncIterator[StreamItem]:
        return self._run()

    async def _run(self) -> AsyncIterator[StreamItem]:
        try:
            async for item in self._producer(self):
                yield item
        finally:
            self.done = True

    async def write(self, data: bytes) -> None:
        self.uploaded += data

    async def end(self) -> None:
        self.ended.set()

    async def result(self) -> dict[str, Any]:
        return self._finish()

    async def cancel(self) -> None:
        self.cancelled = True
        self.cancel_event.set()


@dataclass
class ScriptRun:
    """What `scripts.run` of one (script, action) does on the fake node."""
    lines: list[tuple[str, str]] = field(default_factory=list)      # (stream, line)
    rc: int = 0
    gate: asyncio.Event | None = None                                # when set, the run waits here after its first line
    effect: Callable[[FakeNode], None] | None = None                 # e.g. add the containers the script would start
    error: NodeError | None = None


def _lost(sent: Any, kept: Any, path: str = "") -> list[str]:
    """Where `kept` (after proto.clean) differs from what the hub `sent`: those are fields the real node would drop."""
    if sent is None:
        return []
    if isinstance(sent, dict):
        if not isinstance(kept, dict):
            return [path or "<root>"]
        return [x for k, v in sent.items() for x in _lost(v, kept.get(k), f"{path}.{k}" if path else k)]
    if isinstance(sent, list):
        if not isinstance(kept, list) or len(kept) != len(sent):
            return [path]
        return [x for i, (a, b) in enumerate(zip(sent, kept, strict=True)) for x in _lost(a, b, f"{path}[{i}]")]
    return [] if sent == kept else [path]


class FakeNode:
    """One machine's node. Defaults: manage mode, two 96 GiB GPUs, an idle host, the `fake/engine:*` images allowed."""

    def __init__(self, host: HostConfig, *, hello: dict[str, Any] | None = None, **hello_kw: Any) -> None:
        self.host = host
        self.hello = hello or make_hello(hostname=host.label, **hello_kw)
        self.state = "up"
        self.link_error: str | None = None
        self.connected_since = 1000.0
        self.skew_s = 0.0
        # live numbers
        self.gpu_free_gib: float | list[float] | None = None      # None = idle (all free)
        self.gpu_procs: dict[int, list[dict[str, Any]]] = {}       # gpu index -> [{pid, container, used_bytes, name}]
        self.mem = {"total": 246.0, "available": 54.0, "free": 3.0, "cached": 90.0, "shmem": 66.0, "swap_total": 32.0,
                    "swap_free": 8.0}
        self.cg: dict[str, dict[str, Any]] = {}                    # container name -> cgroup block
        self.seq = 1
        # inventory
        self.containers: dict[str, dict[str, Any]] = {}
        self.endpoints: list[dict[str, Any]] = []
        self.script_status: dict[str, dict[str, Any]] = {}
        self.disks: list[dict[str, Any]] = []
        # behaviour
        self.logs: dict[str, str] = {}
        self.missing_images: set[str] = set()
        self.port_taken: set[int] = set()
        self.network_internal = True
        self.fail: dict[str, NodeError] = {}
        self.http_routes: dict[tuple[int | None, str], Callable[[httpx.Request], httpx.Response]] = {}
        self.http_requests: list[httpx.Request] = []
        self.http_chunk = 64
        self.script_runs: dict[tuple[str, str], ScriptRun] = {}
        self.running_scripts: set[str] = set()
        self.script_calls: list[tuple[str, str, str | None]] = []
        # records
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.run_specs: list[dict[str, Any]] = []
        self.metrics_config: dict[str, Any] | None = None
        self.streams: list[FakeStream] = []

    # -- NodeClient ------------------------------------------------------------------------------------------------
    def link(self) -> LinkStatus:
        up = self.state in ("up", "stale")
        return LinkStatus(state=self.state, error=self.link_error, connected_since=self.connected_since if up else None,  # type: ignore[arg-type]
                          skew_s=self.skew_s if up else None, last_seen=1.0)

    def latest(self, kind: NodeKind) -> dict[str, Any] | None:
        if kind == "hello":
            return self.hello
        if kind == "sample":
            return self._sample()
        if kind == "inventory":
            return self._inventory()
        if kind == "disks":
            return proto.sanitize_push({"type": "disks", "t": 1.0, "mounts": copy.deepcopy(self.disks)})
        return None

    async def call(self, op: str, args: dict[str, Any] | None = None, *, timeout: float = 30.0) -> dict[str, Any]:
        args = copy.deepcopy(args or {})
        self.calls.append((op, args))
        self._gate(op, args)
        if op in self.fail:
            raise self.fail[op]
        handler = getattr(self, "_op_" + op.replace(".", "_"), None)
        if handler is None:
            raise NodeError(self.host.id, "unknown_op", f"the fake node does not implement {op}")
        out = handler(args)
        if asyncio.iscoroutine(out):
            out = await out
        return proto.clean(out, proto.RESULTS[op]) or {}

    def open(self, op: str, args: dict[str, Any] | None = None) -> FakeStream | FailedStream:
        args = copy.deepcopy(args or {})
        self.calls.append((op, args))
        try:
            self._gate(op, args)
            if op in self.fail:
                raise self.fail[op]
            stream = getattr(self, "_open_" + op.replace(".", "_"))(args)
        except NodeError as e:
            return FailedStream(e)
        self.streams.append(stream)
        return stream

    # -- the pushed messages ----------------------------------------------------------------------------------------
    def _sample(self) -> dict[str, Any] | None:
        if self.state not in ("up", "stale"):
            return self._stale_sample
        gpus: dict[str, Any] = {}
        for g in self.hello["gpus"]:
            total = g["total_bytes"]
            free = self.gpu_free_gib
            free_gib = (free[g["index"]] if isinstance(free, list) else free) if free is not None else total / GIB
            used = max(total - int(free_gib * GIB), 0)
            gpus[g["uuid"]] = {"util_pct": 5.0, "mem_used_bytes": None if g.get("unified_memory") else used,   # the driver
                               "mem_total_bytes": None if g.get("unified_memory") else total,                   # cannot say
                               "temp_c": 40.0, "power_w": 90.0, "power_limit_w": 300.0, "fan_pct": 30.0,
                               "procs": copy.deepcopy(self.gpu_procs.get(g["index"], []))}
        raw = {"type": "sample", "seq": self.seq, "t": 1.0, "gpus": gpus,
               "mem": {k: int(v * GIB) for k, v in self.mem.items()}, "cpu": {"load1": 0.5, "util_pct": 3.0},
               "containers": copy.deepcopy(self.cg)}
        out = proto.sanitize_push(raw)
        self._stale_sample = out
        return out

    _stale_sample: dict[str, Any] | None = None

    def _inventory(self) -> dict[str, Any] | None:
        raw = {"type": "inventory", "t": 1.0, "containers": copy.deepcopy(list(self.containers.values())),
               "endpoints": copy.deepcopy(self.endpoints), "scripts": copy.deepcopy(list(self.script_status.values()))}
        return proto.sanitize_push(raw)

    def tick_sample(self) -> None:
        """The node pushed a newer sample (the hub's metrics service records one system point per new `seq`)."""
        self.seq += 1

    def set_gpu_free(self, gib: float | list[float] | None) -> None:
        self.gpu_free_gib = gib

    # -- test data ------------------------------------------------------------------------------------------------------
    def add_container(self, name: str, *, image: str = "vllm/vllm-openai:v0.23.0", engine: str | None = None,
                      state: str = "running", managed: bool = False, labels: dict[str, str] | None = None,
                      host_ports: list[dict[str, Any]] | None = None, argv: list[str] | None = None,
                      params: dict[str, str] | None = None, mounts: list[dict[str, Any]] | None = None,
                      gpus: list[str] | None = None, all_gpus: bool = False, started_at: float | None = 100.0,
                      exit_code: int | None = None) -> dict[str, Any]:
        labels = dict(labels or {})
        if managed:
            labels.setdefault("engine-console", "1")
        c: dict[str, Any] = {
            "name": name, "id": hashlib.sha1(name.encode(), usedforsecurity=False).hexdigest()[:12], "image": image,
            "state": state, "status": state, "created": 50.0, "started_at": started_at, "labels": labels,
            "managed": managed, "role": labels.get("engine-console.role"), "instance": labels.get("engine-console.instance"),
            "ports": host_ports or [], "argv": argv or [], "params": params or {}, "gpus": gpus or [], "all_gpus": all_gpus,
            "mounts": mounts or [], "exit_code": exit_code}
        if engine:
            c["engine"] = engine
        self.containers[name] = c
        return c

    def add_endpoint(self, port: int, *, container: str | None = None, engine: str | None = None,
                     models: tuple[str, ...] = (), health: int | None = 200, models_status: int | None = 200,
                     metrics: int | None = None, auth_required: bool = False) -> dict[str, Any]:
        e: dict[str, Any] = {"port": port, "container": container, "engine": engine, "models": list(models),
                             "health_status": health, "models_status": models_status, "metrics_status": metrics,
                             "auth_required": auth_required}
        self.endpoints = [x for x in self.endpoints if x["port"] != port] + [e]
        return e

    def ops(self, op: str) -> list[dict[str, Any]]:
        return [a for o, a in self.calls if o == op]

    def serve(self, port: int | None, path: str, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        """Answer GET/POST `path` on `port` (None: any port) like an engine would."""
        self.http_routes[(port, path)] = handler

    # -- request checks ---------------------------------------------------------------------------------------------------
    def _gate(self, op: str, args: dict[str, Any]) -> None:
        if self.state != "up":
            raise NodeUnavailable(self.host.id, self.link_error or f"the node is {self.state}")
        if op not in proto.OPS:
            raise NodeError(self.host.id, "unknown_op", op)
        cleaned = proto.clean_args(op, args)
        lost = _lost(args, cleaned)
        if lost:   # the real node would silently drop these: a hub bug (misspelt, over-long or wrongly typed argument)
            raise NodeError(self.host.id, "bad_args", f"{op}: the node would drop {', '.join(lost)}")
        missing = proto.missing_args(op, cleaned)
        if missing:
            raise NodeError(self.host.id, "bad_args", f"{op}: missing {', '.join(missing)}")
        if self.mode != "manage" and op not in proto.MONITOR_OPS:
            raise NodeError(self.host.id, "monitor_only", f"{op} needs the policy mode manage; this node is in monitor mode")

    @property
    def mode(self) -> str:
        return str(self.hello["policy"]["mode"])

    def set_mode(self, mode: str) -> None:
        self.hello["policy"]["mode"] = mode

    @property
    def policy(self) -> dict[str, Any]:
        out: dict[str, Any] = self.hello["policy"]
        return out

    def root_path(self, root_id: str) -> str:
        for r in self.policy["model_roots"]:
            if r["id"] == root_id:
                return str(r["path"])
        raise NodeError(self.host.id, "bad_args", f"{root_id} is not a model root of this policy")

    def _owned(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        c = self.containers.get(name)
        if c is None:
            raise NodeError(self.host.id, "not_found", f"no container {name}")
        if not c.get("managed") or c["labels"].get("engine-console") != "1":
            raise NodeError(self.host.id, "not_owned", f"{name} is not a container this console created")
        if args.get("role") and c["labels"].get("engine-console.role") != args["role"]:
            raise NodeError(self.host.id, "not_owned", f"{name} does not have role {args['role']}")
        if args.get("instance") and c["labels"].get("engine-console.instance") != args["instance"]:
            raise NodeError(self.host.id, "not_owned", f"{name} belongs to another instance")
        return c

    # -- node ops (request/response) -----------------------------------------------------------------------------------------
    def _op_node_info(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.hello

    def _op_metrics_configure(self, args: dict[str, Any]) -> dict[str, Any]:
        self.metrics_config = args
        return {}

    def _op_net_ensure_internal(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self.network_internal:
            raise NodeError(self.host.id, "conflict", "the engine network exists but is not --internal; refusing to use it")
        return {"network": self.policy["engine_network"]}

    def _op_docker_image_present(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"present": args["image"] not in self.missing_images, "id": "sha256:x"}

    def _op_docker_images(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"images": []}

    def _check_spec(self, spec: dict[str, Any]) -> None:
        if not proto_name_ok(spec.get("name", "")):
            raise NodeError(self.host.id, "bad_args", "bad container name")
        from engine_console.services import nodeview as nv
        if not nv.image_allowed(spec["image"], self.policy["engine_images"]):
            raise NodeError(self.host.id, "policy_denied", f"image {spec['image']} is not allowed by this node's policy")
        known = {g["uuid"] for g in self.hello["gpus"]}
        if any(u not in known for u in spec.get("gpu_uuids") or []):
            raise NodeError(self.host.id, "bad_args", "an unknown GPU uuid")
        if any(not str(k).startswith("engine-console.") for k in (spec.get("labels") or {})):
            raise NodeError(self.host.id, "bad_args", "labels must start with engine-console.")
        if spec.get("ipc_host") and self.policy.get("allow_ipc_host") is False:
            raise NodeError(self.host.id, "policy_denied", "host IPC is not allowed by this node's policy")
        for m in spec.get("mounts") or []:
            self.root_path(m["root"])   # raises bad_args for a root that is not in the policy

    def _op_docker_render(self, args: dict[str, Any]) -> dict[str, Any]:
        spec = args["spec"]
        self._check_spec(spec)
        argv = ["docker", "run", "-d", "--name", spec["name"], "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
                "--pids-limit", "4096", "--network", self.policy["engine_network"]]
        for k, v in (spec.get("env") or {}).items():
            argv += ["-e", f"{k}={v}"]
        for k in (spec.get("secret_env") or {}):
            argv += ["-e", k]                      # a name only: the node never renders a secret value
        argv += [spec["image"], *(spec.get("argv") or [])]
        return {"argv": argv}

    def _op_docker_run(self, args: dict[str, Any]) -> dict[str, Any]:
        spec = args["spec"]
        self._check_spec(spec)
        self.run_specs.append(copy.deepcopy(spec))
        labels = {"engine-console": "1", **(spec.get("labels") or {})}
        mounts = []
        for m in spec.get("mounts") or []:
            base = self.root_path(m["root"])
            mounts.append({"source": base + ("/" + m["rel"] if m.get("rel") else ""), "dest": m["dest"],
                           "ro": m.get("ro", True), "root": m["root"], "rel": m.get("rel") or ""})
        if spec["name"] in self.containers and self.containers[spec["name"]]["state"] == "running":
            raise NodeError(self.host.id, "conflict", f"a container named {spec['name']} is already running")
        self.add_container(spec["name"], image=spec["image"], engine=labels.get("engine-console.engine"), managed=True,
                           labels=labels, argv=list(spec.get("argv") or []), mounts=mounts,
                           gpus=list(spec.get("gpu_uuids") or []), started_at=1.0)
        return {"id": hashlib.sha1(spec["name"].encode(), usedforsecurity=False).hexdigest()[:12]}

    def _op_gateway_start(self, args: dict[str, Any]) -> dict[str, Any]:
        lo, hi = self.policy["port_range"]
        if not lo <= args["host_port"] <= hi:
            raise NodeError(self.host.id, "bad_args", "host_port is outside the policy's port range")
        if args["host_port"] in self.port_taken:
            raise NodeError(self.host.id, "port_in_use", f"port {args['host_port']} is already in use")
        engine = self.containers.get(args["engine_host"])
        if engine is None or not engine.get("managed"):
            raise NodeError(self.host.id, "not_found", f"{args['engine_host']} is not a container this console created")
        if not self.policy.get("gateway_image"):
            raise NodeError(self.host.id, "policy_denied", "the policy names no gateway_image")
        self.add_container(args["name"], image=self.policy["gateway_image"], managed=True, started_at=1.0,
                           labels={"engine-console.role": "gateway", "engine-console.instance": args["instance"]},
                           host_ports=ports(args["host_port"], 8080, "127.0.0.1"))
        self.port_taken.add(args["host_port"])
        return {"id": "gw" + args["name"][-6:]}

    def _op_docker_stop(self, args: dict[str, Any]) -> dict[str, Any]:
        c = self._owned(args["name"], args)
        c["state"], c["status"], c["exit_code"] = "exited", "exited", 0
        for p in c.get("ports") or []:
            self.port_taken.discard(p["host_port"])
        return {}

    def _op_docker_rm(self, args: dict[str, Any]) -> dict[str, Any]:
        c = self._owned(args["name"], args)
        for p in c.get("ports") or []:
            self.port_taken.discard(p["host_port"])
        del self.containers[args["name"]]
        return {}

    def _op_docker_inspect(self, args: dict[str, Any]) -> dict[str, Any]:
        c = self.containers.get(args["name"])
        if c is None:
            raise NodeError(self.host.id, "not_found", f"no container {args['name']}")
        return {"info": c}

    def _op_docker_logs(self, args: dict[str, Any]) -> dict[str, Any]:
        self._owned(args["name"], {})   # logs of containers the console did not create can hold secrets: refused
        return {"text": self.logs.get(args["name"], ""), "truncated": False}

    def _op_docker_external(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.policy.get("external_control") != "stop_start":
            raise NodeError(self.host.id, "policy_denied", "this node's policy does not allow controlling other engines")
        c = self.containers.get(args["name"])
        if c is None:
            raise NodeError(self.host.id, "not_found", f"no container {args['name']}")
        if c.get("managed"):
            raise NodeError(self.host.id, "not_owned", "that container is the console's own: use the instance routes")
        from engine_console.services.externals import matches_script
        if any(matches_script(args["name"], s) for s in self.policy["scripts"]):
            raise NodeError(self.host.id, "policy_denied", "that engine is managed by a script: run the script's action")
        if args["action"] not in ("stop", "start"):
            raise NodeError(self.host.id, "bad_args", "action must be stop or start")
        c["state"] = "exited" if args["action"] == "stop" else "running"
        return {}

    # -- node ops (streams) --------------------------------------------------------------------------------------------------------
    def _open_docker_logs_follow(self, args: dict[str, Any]) -> FakeStream:
        self._owned(args["name"], {})
        lines = self.logs.get(args["name"], "").splitlines()

        async def producer(s: FakeStream) -> AsyncIterator[StreamItem]:
            for ln in lines:
                if s.cancelled:
                    return
                yield StreamItem("event", event={"line": ln})
                await asyncio.sleep(0)

        return FakeStream(producer, lambda: {"lines": len(lines)})

    def _open_http_request(self, args: dict[str, Any]) -> FakeStream:
        port, method, path = args["port"], args["method"], args["path"]
        from engine_console.services import nodeview as nv
        allowed = {p for c in self.containers.values() if c.get("managed") for p in nv.host_ports(c)}
        allowed |= {e["port"] for e in self.endpoints}
        if port not in allowed:
            raise NodeError(self.host.id, "policy_denied", f"port {port} is neither a gateway nor a known engine endpoint")
        if method not in ("GET", "POST"):
            raise NodeError(self.host.id, "policy_denied", "only GET and POST")
        if not path.startswith(HTTP_PREFIXES):
            raise NodeError(self.host.id, "policy_denied", f"path {path[:40]} is not allowed")
        outcome: dict[str, Any] = {}

        async def producer(s: FakeStream) -> AsyncIterator[StreamItem]:
            if args.get("body_len"):
                await s.ended.wait()
                if len(s.uploaded) != args["body_len"]:
                    outcome["error"] = NodeError(self.host.id, "bad_args", "uploaded body does not match body_len")
                    return
            req = httpx.Request(method, f"http://127.0.0.1:{port}{path}", headers=args.get("headers") or {},
                                content=bytes(s.uploaded))
            self.http_requests.append(req)
            route = self.http_routes.get((port, path)) or self.http_routes.get((None, path))
            try:
                resp = route(req) if route else httpx.Response(503, text="not up")
            except NodeError as e:
                outcome["error"] = e
                return
            body = resp.content
            yield StreamItem("event", event={"status": resp.status_code, "headers": dict(resp.headers)})
            for i in range(0, len(body), self.http_chunk):
                if s.cancelled:
                    outcome["cancelled"] = True
                    return
                yield StreamItem("data", data=body[i:i + self.http_chunk])
                await asyncio.sleep(0)
            outcome["result"] = {"status": resp.status_code, "bytes": len(body)}

        def finish() -> dict[str, Any]:
            if "error" in outcome:
                raise outcome["error"]
            if outcome.get("cancelled"):
                raise NodeError(self.host.id, "cancelled", "cancelled")
            return dict(outcome.get("result") or {})

        return FakeStream(producer, finish)

    def _open_scripts_run(self, args: dict[str, Any]) -> FakeStream:
        name, action, arg = args["name"], args["action"], args.get("arg")
        script = next((s for s in self.policy["scripts"] if s["name"] == name), None)
        if script is None:
            raise NodeError(self.host.id, "not_found", f"no script {name}")
        act = next((a for a in script["actions"] if a["name"] == action), None)
        if act is None:
            raise NodeError(self.host.id, "not_found", f"script {name} has no action {action}")
        if (act.get("args") and arg not in act["args"]) or (arg is not None and not act.get("args")):
            raise NodeError(self.host.id, "bad_args", "argument not allowed for this action")
        if name in self.running_scripts:
            raise NodeError(self.host.id, "busy", f"script {name} is already running")
        self.script_calls.append((name, action, arg))
        run = self.script_runs.get((name, action), ScriptRun())
        self.running_scripts.add(name)
        outcome: dict[str, Any] = {}

        async def producer(s: FakeStream) -> AsyncIterator[StreamItem]:
            try:
                for i, (stream, line) in enumerate(run.lines):
                    if s.cancelled:
                        outcome["cancelled"] = True
                        return
                    yield StreamItem("event", event={"stream": stream, "line": line})
                    if i == 0 and run.gate is not None:
                        waiter = asyncio.ensure_future(run.gate.wait())
                        canceller = asyncio.ensure_future(s.cancel_event.wait())
                        await asyncio.wait({waiter, canceller}, return_when=asyncio.FIRST_COMPLETED)
                        waiter.cancel()
                        canceller.cancel()
                    await asyncio.sleep(0)
                if run.error is not None:
                    outcome["error"] = run.error
                    return
                if run.effect is not None and run.rc == 0:
                    run.effect(self)
                outcome["result"] = {"rc": run.rc}
            finally:
                self.running_scripts.discard(name)

        def finish() -> dict[str, Any]:
            if "error" in outcome:
                raise outcome["error"]
            if outcome.get("cancelled"):
                raise NodeError(self.host.id, "cancelled", "cancelled")
            return dict(outcome.get("result") or {})

        return FakeStream(producer, finish)


def proto_name_ok(name: str) -> bool:
    return re.fullmatch(r"ec-[a-z0-9][a-z0-9-]{0,58}", name) is not None


class FakeFleet:
    """A FleetPort over FakeNodes. `start`/`stop` and the `set_*` hooks exist because the app container drives them."""

    def __init__(self, nodes: list[FakeNode]) -> None:
        self._nodes = {n.host.id: n for n in nodes}
        self.resolver: Callable[[str, str], str | None] | None = None
        self.counts: Callable[[str], Any] | None = None
        self.started = False

    def add(self, node: FakeNode) -> FakeNode:
        self._nodes[node.host.id] = node
        return node

    def hosts(self) -> list[HostConfig]:
        return [n.host for n in self._nodes.values()]

    def node(self, host_id: str) -> FakeNode:
        n = self._nodes.get(host_id)
        if n is None:
            raise NotFound(f"no host {host_id[:40]!r} is configured", code="unknown_host")
        return n

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    def set_instance_resolver(self, fn: Callable[[str, str], str | None] | None) -> None:
        self.resolver = fn

    def set_counts_provider(self, fn: Callable[[str], Any] | None) -> None:
        self.counts = fn


class FakeLibrary:
    """The model library as the runtime services see it: copies registered by the test, roots read from each node's policy."""

    def __init__(self, fleet: FakeFleet) -> None:
        self._fleet = fleet
        self._copies: dict[tuple[str, tuple[str, ...]], LibraryCopy] = {}
        self._infos: dict[tuple[str, tuple[str, ...]], ModelInfo] = {}
        self.started = False

    @staticmethod
    def _key(ref: ModelRef) -> tuple[str, ...]:
        if ref.format == "hf":
            return ("hf", ref.repo_id or "")
        if ref.format == "ollama":
            return ("ollama", ref.name or "")
        return (ref.format, ref.root_id or "", ref.rel or "")

    def add(self, host_id: str, ref: ModelRef, *, root_id: str, rel: str, size_bytes: int = 65_530_000_000,
            info: ModelInfo | None = None, drive: MountInfo | None = None, root_label: str | None = None) -> LibraryCopy:
        node = self._fleet.node(host_id)
        base = node.root_path(root_id)
        c = LibraryCopy(id=hashlib.sha1(f"{host_id}|{root_id}|{rel}".encode(), usedforsecurity=False).hexdigest()[:16],
                        host_id=host_id, host_label=node.host.label, root_id=root_id, root_label=root_label, rel=rel,
                        path=base + ("/" + rel if rel else ""), size_bytes=size_bytes, drive=drive, visible_on=[host_id])
        self._copies[(host_id, self._key(ref))] = c
        if info is not None:
            self._infos[(host_id, self._key(ref))] = info
        return c

    def add_hf(self, host_id: str, repo_id: str, cfg: dict[str, Any], size: int = 65_530_000_000, *, root_id: str = "hf",
               rel: str | None = None, drive: MountInfo | None = None) -> LibraryCopy:
        hub = HubModel(repo_id=repo_id, sha="c0ffee", gated=False, license=None, pipeline_tag=None, library_name=None,
                       tags=[], files=[RepoFile("model.safetensors", size)], config=cfg)
        return self.add(host_id, ModelRef(format="hf", repo_id=repo_id), root_id=root_id, size_bytes=size,
                        rel=rel if rel is not None else f"hub/models--{repo_id.replace('/', '--')}",
                        info=build_model_info(hub), drive=drive)

    def remove(self, host_id: str, ref: ModelRef) -> None:
        self._copies.pop((host_id, self._key(ref)), None)

    def find(self, host_id: str, ref: ModelRef) -> LibraryCopy | None:
        return self._copies.get((host_id, self._key(ref)))

    async def model_info(self, host_id: str, ref: ModelRef) -> ModelInfo | None:
        return self._infos.get((host_id, self._key(ref)))

    def _root(self, host_id: str, kinds: tuple[str, ...]) -> RootInfo | None:
        for r in self._fleet.node(host_id).policy["model_roots"]:
            if r["kind"] in kinds:
                return RootInfo(id=r["id"], path=r["path"], kind=r["kind"], writable=bool(r.get("writable")), label=r.get("label"))
        return None

    def hf_root(self, host_id: str) -> RootInfo | None:
        return self._root(host_id, ("hf-home", "hf-hub"))

    def ollama_root(self, host_id: str) -> RootInfo | None:
        return self._root(host_id, ("ollama",))

    def copies(self, host_id: str) -> list[LibraryCopy]:
        return [c for (h, _), c in self._copies.items() if h == host_id]

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False


async def until(condition: Callable[[], bool], timeout: float = 5.0, what: str = "the condition") -> None:
    """Poll `condition` until it holds. Tests wait on state, never on a guessed number of milliseconds: a loaded machine
    (other suites running next to this one) must not turn a correct test into a flaky one."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.005)
