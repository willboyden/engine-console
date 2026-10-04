"""The node's own view of its machine, independent of the wire: hello, sample, disks, inventory, metrics.

The stdio server (server.py) pushes these on their schedules and calls the op handlers (ops.py) with this agent.
Every collector degrades instead of failing: no GPUs, no docker or a dead mount give empty or None fields and a
human `error`, never an exception out of a push loop.
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import __version__, proto
from .discovery import PortProbe, View, build_endpoints, container_view, http_get, probe_port, script_status
from .disks import Disks
from .docker import Docker, effective_context
from .errors import OpError
from .fsops import WriteLocks
from .gpus import Gpus
from .hostinfo import Paths, addresses, host_facts, node_id
from .memory import Mem
from .models import Scanner
from .policy import Policy, PolicyStore
from .scripts import Scripts
from .util import Runner, json_size, run

METRICS_RAW_MAX = 8 * 1024 * 1024
INVENTORY_BUDGET = 900 * 1024       # bytes of JSON: one inventory must fit one frame (MAX_JSON is 1 MiB)


@dataclass
class Inventory:
    msg: dict[str, Any]                                    # proto.INVENTORY without "type"
    views: list[View] = field(default_factory=list)
    names_by_id: dict[str, str] = field(default_factory=dict)
    engines: list[tuple[str, int]] = field(default_factory=list)   # (name, init pid) of running engine containers
    endpoint_ports: set[int] = field(default_factory=set)
    gateway_ports: set[int] = field(default_factory=set)
    error: str | None = None

    @classmethod
    def empty(cls) -> Inventory:
        return cls(msg={"t": None, "containers": [], "endpoints": [], "scripts": []})


@dataclass
class MetricsConf:
    interval_s: float | None = None                        # None: not configured, nothing is scraped
    families: dict[str, list[str]] = field(default_factory=dict)


def filter_metrics(text: str, prefixes: tuple[str, ...], cap: int = proto.MAX_METRICS_TEXT) -> str:
    """Keep the samples (and the # HELP / # TYPE lines) of metric families whose name starts with a prefix."""
    keep: list[str] = []
    size = 0
    for line in text.splitlines():
        if line.startswith(("# HELP ", "# TYPE ")):
            parts = line.split(" ", 3)
            name = parts[2] if len(parts) > 2 else ""
        elif not line.strip() or line.startswith("#"):
            continue
        else:
            name = line.split("{", 1)[0].split(" ", 1)[0]
        if name.startswith(prefixes):
            if size + len(line) + 1 > cap:
                break
            keep.append(line)
            size += len(line) + 1
    return "\n".join(keep) + ("\n" if keep else "")


def shrink_inventory(msg: dict[str, Any], budget: int = INVENTORY_BUDGET) -> dict[str, Any]:
    """An inventory that fits one frame: on a machine with very many or very verbose containers, first the argv
    and labels of containers that are not engines are trimmed, then those containers are left out, then every
    argv is cut short. Engines and console containers stay listed."""
    if json_size(msg) <= budget:
        return msg
    others = [c for c in msg["containers"] if not c.get("engine") and not c.get("managed")]
    for c in others:
        c["argv"], c["params"], c["labels"] = c["argv"][:8], {}, {}
    if json_size(msg) > budget:
        msg["containers"] = [c for c in msg["containers"] if c.get("engine") or c.get("managed")]
    for cap in (100, 20):
        if json_size(msg) <= budget:
            break
        for c in msg["containers"]:
            c["argv"] = [a[:256] for a in c["argv"][:cap]]
    return msg


class Agent:
    def __init__(self, policies: PolicyStore, *, paths: Paths | None = None, gpus: Gpus | None = None,
                 runner: Runner = run, prober: Callable[[int], PortProbe] = probe_port,
                 metrics_get: Callable[..., Any] = http_get) -> None:
        self.policies = policies
        self.paths = paths or Paths()
        self.gpus = gpus if gpus is not None else Gpus()
        self.mem = Mem(self.paths)
        self.disk = Disks(self.paths)
        self.runner = runner
        self.prober = prober
        self.metrics_get = metrics_get
        self.scripts = Scripts()
        self.write_locks = WriteLocks()
        self.scanner: Scanner | None = None
        self.metrics_conf = MetricsConf()
        self._gpu_static: list[dict[str, Any]] = []
        self._gpu_info: dict[str, Any] = {"driver": None, "cuda": None, "source": "none", "error": None}
        self._inventory = Inventory.empty()
        self._inv_lock = threading.Lock()
        self.first_inventory = threading.Event()     # set once the first inventory (container names) is known
        self._cpu_prev: dict[str, tuple[float, int]] = {}
        self._contexts: dict[tuple[str, str | None], str | None] = {}

    @property
    def policy(self) -> Policy:
        return self.policies.current()

    def docker(self, policy: Policy | None = None) -> Docker:
        policy = policy or self.policy
        key = (policy.docker_binary, policy.docker_context)
        if key not in self._contexts:
            self._contexts[key] = effective_context(*key, runner=self.runner)
        return Docker(policy.docker_binary, self._contexts[key], runner=self.runner)

    @property
    def gpu_uuids(self) -> set[str]:
        return {g["uuid"] for g in self._gpu_static if g.get("uuid")}

    def gpu_by_index(self) -> dict[str, str]:
        return {str(g["index"]): g["uuid"] for g in self._gpu_static if g.get("index") is not None and g.get("uuid")}

    @property
    def inventory(self) -> Inventory:
        return self._inventory

    # ── hello ────────────────────────────────────────────────────────────────────────────────────────────
    def hello(self, *, reload: bool = False) -> dict[str, Any]:
        policy = self.policies.reload() if reload else self.policy
        with ThreadPoolExecutor(max_workers=4) as ex:
            f_gpus = ex.submit(self.gpus.static)
            f_docker = ex.submit(self.docker(policy).facts)
            f_addr = ex.submit(addresses, self.runner, self.paths)
            f_host = ex.submit(host_facts, self.paths)
            self._gpu_static, self._gpu_info = f_gpus.result()
            docker = f_docker.result()
            addrs = f_addr.result()
            host = f_host.result()
        return {"type": "hello", "proto": proto.PROTO, "version": __version__, "node_id": node_id(self.paths),
                "host": host, "gpu": self._gpu_info, "gpus": self._gpu_static, "docker": docker,
                "addresses": addrs, "policy": policy.summary(),
                "forced_cmd": "SSH_ORIGINAL_COMMAND" in os.environ, "via_ssh": "SSH_CONNECTION" in os.environ}

    # ── sample ───────────────────────────────────────────────────────────────────────────────────────────
    def sample(self, seq: int) -> dict[str, Any]:
        stats, procs = self.gpus.sample()
        inv = self._inventory
        gpus: dict[str, dict[str, Any]] = {uuid: {**s, "procs": []} for uuid, s in stats.items()}
        for uuid, pid, used in procs:
            g = gpus.get(uuid)
            if g is None or len(g["procs"]) >= 32:
                continue
            cid = self.mem.container_id_of(pid)
            g["procs"].append({"pid": pid, "used_bytes": used, "name": self.mem.comm(pid),
                               "container": inv.names_by_id.get(cid) if cid else None})
        for g in gpus.values():
            g["procs"].sort(key=lambda p: -(p["used_bytes"] or 0))
        load1, load5, load15 = self.mem.loadavg()
        containers: dict[str, Any] = {}
        now_m = time.monotonic()
        for name, pid in inv.engines[:256]:
            usage = self.mem.container_cpu_usec(pid)
            prev = self._cpu_prev.get(name)
            cpu_pct = None
            if usage is not None and prev is not None and now_m > prev[0] and usage >= prev[1]:
                cpu_pct = round((usage - prev[1]) / 1e6 / (now_m - prev[0]) * 100.0, 1)
            if usage is not None:
                self._cpu_prev[name] = (now_m, usage)
            containers[name] = {"mem": self.mem.container_mem(pid), "cpu_pct": cpu_pct}
        self._cpu_prev = {k: v for k, v in self._cpu_prev.items() if k in containers}
        return {"type": "sample", "seq": seq, "t": round(time.time(), 3), "gpus": gpus, "mem": self.mem.meminfo(),
                "cpu": {"load1": load1, "load5": load5, "load15": load15, "util_pct": self.mem.cpu_util()},
                "containers": containers}

    # ── disks ────────────────────────────────────────────────────────────────────────────────────────────
    def disks(self) -> dict[str, Any]:
        return {"type": "disks", **self.disk.report(self.policy.model_roots)}

    # ── inventory ────────────────────────────────────────────────────────────────────────────────────────
    def refresh_inventory(self) -> Inventory:
        policy = self.policy
        d = self.docker(policy)
        views: list[View] = []
        error: str | None = None
        try:
            rows = {r["id"]: r for r in [*d.ps(), *d.ps(all_=True, label="engine-console=1")]}
            roots = [(r, self.disk.resolve(r.path)) for r in policy.model_roots]
            by_index = self.gpu_by_index()
            for doc in d.inspect(list(rows)[:256]):
                row = rows.get(str(doc.get("Id") or ""), {})
                v = container_view(doc, roots=roots, gpu_by_index=by_index, status=row.get("status"))
                if v is not None:
                    views.append(v)
        except OpError as e:
            error = e.message
        endpoints = build_endpoints(views, policy.discovery_ports, self.prober)
        now = round(time.time(), 3)
        running = [v.info["name"] for v in views if v.running]
        msg = shrink_inventory({"t": now, "containers": [dict(v.info) for v in views], "endpoints": endpoints,
                                "scripts": script_status(policy, running, now)})
        gateway_ports = {v.published.get(8080) or (v.loopback_ports[0] if v.loopback_ports else 0)
                         for v in views if v.running and v.info["managed"] and v.info["role"] == "gateway"}
        # cgroup memory is sampled for every running engine-classified or console-owned container
        inv = Inventory(msg=msg, views=views, error=error,
                        names_by_id={v.info["id"]: v.info["name"] for v in views if v.info["id"]},
                        engines=[(v.info["name"], v.pid) for v in views
                                 if v.running and v.pid and (v.info["engine"] or v.info["managed"])],
                        endpoint_ports={int(e["port"]) for e in endpoints}, gateway_ports=gateway_ports - {0})
        with self._inv_lock:
            self._inventory = inv
        self.first_inventory.set()
        return inv

    def allowed_ports(self, refresh: bool) -> set[int]:
        inv = self.refresh_inventory() if refresh else self._inventory
        return inv.endpoint_ports | inv.gateway_ports

    def running_mounts(self) -> list[str]:
        """Host paths that running containers mount, read fresh from docker (for fs.delete's in_use check)."""
        d = self.docker()
        res = d.call("ps", "-q", "--no-trunc", timeout=20)
        if res.rc == 127:
            return []                 # no docker CLI on this machine: nothing can be running in a container
        if not res.ok:
            raise OpError(proto.DOCKER_ERROR, "cannot ask docker which containers are running, so cannot tell "
                                              "whether the model is in use; try again")
        out: list[str] = []
        for doc in d.inspect(res.out.split()[:512]):
            for m in doc.get("Mounts") or []:
                if isinstance(m, dict) and isinstance(m.get("Source"), str) and m["Source"].startswith("/"):
                    out.append(m["Source"])
        return out

    # ── metrics ──────────────────────────────────────────────────────────────────────────────────────────
    def configure_metrics(self, interval_s: float | None, families: dict[str, list[str]]) -> None:
        interval = 10.0 if interval_s is None else max(1.0, min(float(interval_s), 60.0))
        self.metrics_conf = MetricsConf(interval_s=interval,
                                        families={k: [p for p in v if p] for k, v in families.items()})

    def scrape(self) -> list[dict[str, Any]]:
        conf = self.metrics_conf
        if conf.interval_s is None:
            return []
        out: list[dict[str, Any]] = []
        for ep in self._inventory.msg["endpoints"]:
            prefixes = conf.families.get(ep.get("engine") or "") or conf.families.get("*")
            if not prefixes or ep.get("metrics_status") != 200:
                continue
            probe = self.metrics_get(int(ep["port"]), "/metrics", max_body=METRICS_RAW_MAX)
            text = probe.body.decode("utf-8", "replace") if probe.ok else ""
            out.append({"type": "metrics", "t": round(time.time(), 3), "port": ep["port"],
                        "container": ep.get("container"), "engine": ep.get("engine"), "status": probe.status,
                        "text": filter_metrics(text, tuple(prefixes))})
        return out
