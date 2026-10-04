"""Metrics (contract v2, §10.9). The hub sends each node `metrics.configure` (the metric-name prefixes of every engine
it knows); the node scrapes its engine endpoints and PUSHES `metrics` messages, which the fleet re-publishes on the
EventBus as topic `node.metrics` with payload {"host_id", "msg"}. Each one is mapped to an instance (by host + port or
container), parsed with its adapter (or the generic map for engines without one) and kept in a ring buffer (15 min)
plus SQLite rollups (1 h / 1 d). Per-host system series (RAM, shared memory, swap) come from the node's samples.

A managed engine whose node pushed nothing for a while is also read directly through `http.request` (the pull
fallback), so its charts do not depend on which endpoints a node chose to probe. Points are stamped with the hub's
clock, never the node's: a node's clock may be minutes off and the windows must still make sense."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from engine_console.adapters import AdapterRegistry
from engine_console.domain.errors import BadRequest, ProblemError
from engine_console.domain.fleet import LOCAL, FleetPort, NodeError
from engine_console.domain.models import Instance, MetricPoint
from engine_console.services import nodeview as nv
from engine_console.services.common import EventBus
from engine_console.services.discovery import parse_generic_metrics
from engine_console.services.hostmem import meminfo_from_sample
from engine_console.services.lifecycle import LifecycleService
from engine_console.services.nodeerr import NodeCaller
from engine_console.services.nodehttp import NodeHttp
from engine_console.services.store import Store

log = logging.getLogger(__name__)
RING_SECONDS = 15 * 60
TOPIC = "metrics"
SYSTEM_KEY = "system"            # the console's own machine keeps its v1 key, so its history survives the upgrade
PUSH_TOPIC = "node.metrics"
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
# Metric-name prefixes per engine: what the node keeps from a scrape. An adapter may declare its own with a
# `metric_prefixes` attribute; engines without an adapter that the node can still classify are listed here too.
PREFIXES: dict[str, list[str]] = {
    "vllm": ["vllm:"], "sglang": ["sglang:"], "llamacpp": ["llamacpp:"], "trtllm": ["trtllm", "tensorrt_llm"],
    "dynamo": ["dynamo_"],
}


def parse_window(w: str) -> int:
    if len(w) < 2 or w[-1] not in _UNITS or not w[:-1].isdigit():
        raise BadRequest("window must look like 15m, 1h, 6h, 1d, 7d", code="invalid_window")
    return int(w[:-1]) * _UNITS[w[-1]]


def system_key(host_id: str) -> str:
    return SYSTEM_KEY if host_id == LOCAL else f"{SYSTEM_KEY}:{host_id}"


class MetricsService:
    def __init__(self, store: Store, lifecycle: LifecycleService, adapters: AdapterRegistry, fleet: FleetPort,
                 http: NodeHttp, bus: EventBus, *, interval_s: float = 2.0,
                 clock: Callable[[], float] = time.time) -> None:
        self._db = store
        self._life = lifecycle
        self._adapters = adapters
        self._fleet = fleet
        self._nodes = NodeCaller(fleet)
        self._http = http
        self._bus = bus
        self._interval = interval_s
        self._clock = clock
        self._ring: dict[str, deque[MetricPoint]] = defaultdict(
            lambda: deque(maxlen=int(RING_SECONDS / max(interval_s, 0.5)) + 1))
        self._last_counters: dict[str, tuple[float, dict[str, float]]] = {}
        self._last_push: dict[str, float] = {}      # instance id -> when its engine metrics last arrived
        self._last_pull: dict[str, float] = {}
        self._last_seq: dict[str, Any] = {}         # host id -> the sample `seq` already recorded
        self._configured: dict[str, tuple[bool, float | None]] = {}
        self._warned: set[tuple[str, float | None]] = set()

    # -- node configuration -------------------------------------------------------------------------------------------
    def families(self) -> dict[str, list[str]]:
        """engine id -> metric-name prefixes, for `metrics.configure` (at most 16 engines, 32 prefixes of 64 chars)."""
        out: dict[str, list[str]] = {k: list(v) for k, v in PREFIXES.items()}
        for a in self._adapters.list():
            declared = getattr(a, "metric_prefixes", None)
            out[a.id] = [str(p) for p in declared] if declared else out.get(a.id, [f"{a.id}:"])
        return {e: [p[:64] for p in prefixes[:32]] for e, prefixes in list(out.items())[:16] if prefixes}

    async def configure(self, host_id: str) -> None:
        await self._nodes.call(host_id, "metrics.configure",
                               {"interval_s": float(self._interval), "families": self.families()}, timeout=20)

    async def sync_hosts(self) -> None:
        """Send `metrics.configure` to every host whose link just came up (or came back)."""
        for h in self._fleet.hosts():
            node = self._fleet.node(h.id)
            key = (nv.link_up(node), node.link().connected_since)
            if key[0] and key != self._configured.get(h.id):
                try:
                    await self.configure(h.id)
                except (ProblemError, NodeError) as e:
                    if (h.id, key[1]) not in self._warned:   # once per connection, not every round
                        self._warned.add((h.id, key[1]))
                        log.warning("metrics.configure failed on %s: %s", h.id, type(e).__name__)
                    continue   # try again next round
            self._configured[h.id] = key

    # -- routing a push to an instance ------------------------------------------------------------------------------
    def _parse(self, engine: str, text: str) -> dict[str, float]:
        if engine in self._adapters:   # the adapter knows its engine's metric names
            return {k: float(v) for k, v in self._adapters.get(engine).parse_metrics(text).items() if isinstance(v, int | float)}
        return parse_generic_metrics(engine, text)

    def ingest(self, host_id: str, msg: dict[str, Any]) -> None:
        """One pushed `metrics` message (already sanitised by the fleet): find its instance, parse, record."""
        port = msg.get("port") if isinstance(msg.get("port"), int) and not isinstance(msg.get("port"), bool) else None
        container = msg.get("container") if isinstance(msg.get("container"), str) and msg["container"] else None
        status = msg.get("status")
        if isinstance(status, int) and not 200 <= status < 300:
            return
        inst = self._life.lookup(host_id, port=port, container=container)
        if inst is None:
            return
        text = str(msg.get("text") or "")[:2_000_000]
        engine = inst.engine if inst.engine in self._adapters or inst.engine in PREFIXES else str(msg.get("engine") or inst.engine)
        self._record(inst, self._parse(engine, text), self._clock(), pushed=True)

    # -- recording ----------------------------------------------------------------------------------------------------
    def _record(self, inst: Instance, vals: dict[str, float], t: float, *, pushed: bool) -> None:
        mem = inst.host_memory
        if mem is not None:     # cgroup-based host RAM, in the same series and ring buffer
            vals.update(host_ram_gib=mem.total_gib, host_ram_anon_gib=mem.anon_gib, host_ram_cache_gib=mem.cache_gib,
                        host_ram_shmem_gib=mem.shmem_gib)
        if not vals:
            return
        if pushed:
            self._last_push[inst.id] = t
        self._derive_rates(inst.id, t, vals)
        if vals.get("requests_running", 0) > 0:
            self._life.touch(inst.id)
        pt = MetricPoint(t=t, values=vals)
        self._ring[inst.id].append(pt)
        self._rollup(inst.id, pt)
        self._bus.publish(TOPIC, {"instance_id": inst.id, **pt.model_dump()})

    def _derive_rates(self, iid: str, t: float, vals: dict[str, float]) -> None:
        """Adapters report lifetime counters; derive windowed tok/s from consecutive scrapes when they didn't."""
        prev = self._last_counters.get(iid)
        cur: dict[str, float] = {k: vals[k] for k in ("prompt_tokens_total", "generation_tokens_total") if k in vals}
        if prev and cur and t > prev[0]:
            dt = t - prev[0]
            for total, rate in (("prompt_tokens_total", "prompt_tps"), ("generation_tokens_total", "generation_tps")):
                if rate not in vals and total in cur and total in prev[1] and cur[total] >= prev[1][total]:
                    vals[rate] = (cur[total] - prev[1][total]) / dt
            if cur.get("prompt_tokens_total", 0) > prev[1].get("prompt_tokens_total", 0):
                self._life.touch(iid)
        self._last_counters[iid] = (t, cur)

    def _rollup(self, key: str, pt: MetricPoint) -> None:
        rows = []
        for res, step in (("h", 3600), ("d", 86400)):
            bucket = int(pt.t // step * step)
            rows += [(key, res, bucket, k, v, 1, v) for k, v in pt.values.items()]
        self._db.executemany(
            "INSERT INTO metric_rollups(instance_id,res,bucket_ts,key,sum,count,max) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(instance_id,res,bucket_ts,key) DO UPDATE SET sum=sum+excluded.sum, count=count+1, "
            "max=MAX(max, excluded.max)", rows)

    # -- sampling loop --------------------------------------------------------------------------------------------------
    async def scrape_once(self) -> None:
        """One round: system series per host, host RAM of every instance, and the pull fallback for managed engines."""
        t = self._clock()
        for h in self._fleet.hosts():
            if self._fleet.node(h.id).link().state in ("up", "stale"):
                self._sample_system(h.id, t)
        pulls = []
        for inst in self._life.all_instances():
            if inst.state in ("stopped", "failed", "unreachable") or not self._connected(inst.host_id):
                continue
            silent = t - self._last_push.get(inst.id, 0.0)
            if inst.management == "managed" and inst.state == "ready" and inst.port \
                    and silent > 3 * self._interval and t - self._last_pull.get(inst.id, 0.0) >= self._interval:
                self._last_pull[inst.id] = t
                pulls.append(self._pull(inst, t))
            elif silent > 2 * self._interval and inst.host_memory is not None:
                self._record(inst, {}, t, pushed=False)   # no engine metrics: host RAM is still worth a series
        await asyncio.gather(*pulls, return_exceptions=True)

    def _connected(self, host_id: str) -> bool:
        with contextlib.suppress(ProblemError):
            return nv.link_up(self._fleet.node(host_id))
        return False

    async def _pull(self, inst: Instance, t: float) -> None:
        launch = self._life.launch_for(inst)
        try:
            r = await self._http.request(inst.host_id, inst.port or 0, "GET", launch.metrics_path if launch else "/metrics",
                                         timeout_s=3.0, max_bytes=2_000_000)
        except (ProblemError, NodeError):
            return
        if r.status == 200:
            self._record(inst, self._parse(inst.engine, r.body.decode("utf-8", "replace")), t, pushed=False)

    def _sample_system(self, host_id: str, t: float) -> None:
        s = nv.sample(self._fleet.node(host_id))
        m = meminfo_from_sample(s.get("mem"), nv.num(s.get("t")))
        seq = s.get("seq")
        if m is None or (seq is not None and seq == self._last_seq.get(host_id)):
            return   # nothing new from that node since the last round
        self._last_seq[host_id] = seq
        key = system_key(host_id)
        pt = MetricPoint(t=t, values={"ram_used_gib": m.used_gib, "ram_available_gib": m.available_gib,
                                      "shmem_gib": m.shmem_gib, "swap_used_gib": m.swap_used_gib})
        self._ring[key].append(pt)
        self._rollup(key, pt)
        self._bus.publish(TOPIC, {"instance_id": SYSTEM_KEY, "host_id": host_id, **pt.model_dump()})

    # -- queries --------------------------------------------------------------------------------------------------------
    def series(self, iid: str, window: str = "15m") -> dict[str, object]:
        inst = self._life.get(iid)  # 404 for unknown instances
        extra: dict[str, object] = {"history_since": inst.history_since} if not inst.managed else {}
        return self._series(iid, window, extra)

    def system_series(self, window: str = "15m", host_id: str = LOCAL) -> dict[str, object]:
        """Host RAM/swap timeseries of one machine (same ring buffer and window semantics as per-instance metrics)."""
        self._nodes.node(host_id)   # 404 unknown_host
        return {**self._series(system_key(host_id), window, {}), "host_id": host_id}

    def _series(self, key: str, window: str, extra: dict[str, object]) -> dict[str, object]:
        secs = parse_window(window)
        since = self._clock() - secs
        if secs <= RING_SECONDS:
            pts = [p for p in self._ring.get(key, ()) if p.t >= since]
            return {"instance_id": key, "window": window, "resolution": "raw", "points": [p.model_dump() for p in pts], **extra}
        res, step = ("h", 3600) if secs <= 3 * 86400 else ("d", 86400)
        rows = self._db.all("SELECT bucket_ts, key, sum, count, max FROM metric_rollups WHERE instance_id=? AND res=? "
                            "AND bucket_ts >= ? ORDER BY bucket_ts", (key, res, int(since // step * step)))
        by_bucket: dict[int, dict[str, float]] = defaultdict(dict)
        for r in rows:
            by_bucket[r["bucket_ts"]][r["key"]] = r["sum"] / r["count"]
        return {"instance_id": key, "window": window, "resolution": "1h" if res == "h" else "1d",
                "points": [{"t": float(b), "values": v} for b, v in sorted(by_bucket.items())], **extra}

    def latest(self, iid: str) -> MetricPoint | None:
        ring = self._ring.get(iid)
        return ring[-1] if ring else None

    # -- running ------------------------------------------------------------------------------------------------------------
    async def consume(self) -> None:
        """Route every pushed `node.metrics` message (until cancelled)."""
        async for ev in self._bus.subscribe(PUSH_TOPIC):
            try:
                self.ingest(str(ev["host_id"]), ev["msg"])
            except Exception:   # noqa: BLE001 - one malformed push must not stop the stream
                log.exception("could not process a pushed metrics message")

    async def run(self) -> None:
        consumer = asyncio.create_task(self.consume(), name="metrics-consumer")
        try:
            while True:
                for step in (self.sync_hosts, self.scrape_once):
                    try:
                        await step()
                    except Exception:   # noqa: BLE001 - a bad round must not end the loop
                        log.exception("metrics round failed (%s)", step.__name__)
                await asyncio.sleep(self._interval)
        finally:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
