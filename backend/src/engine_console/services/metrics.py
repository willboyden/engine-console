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
import math
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
# Review M7: state keyed by names a node chooses (instance ids derived from container names) is bounded per host, so one
# bad node cannot crowd out the others, and expires. Generous for a real machine (dozens of engines).
MAX_KEYS_PER_HOST = 256        # distinct series per host; past it the least recently updated one is evicted
KEY_TTL_S = 3600.0             # in-memory state of a series nobody updated for this long is dropped (the ring holds 15 min)
HOURLY_KEEP_S = 14 * 86400     # rollup rows older than this are deleted (the 1 h series serves windows up to 3 d)
DAILY_KEEP_S = 400 * 86400
PRUNE_EVERY_S = 300.0          # a prune pass is O(series) in memory and one indexed delete per resolution
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
        self._keys: dict[str, dict[str, float]] = {}   # host id -> series key -> when it was last updated (hub clock)
        self._next_prune = 0.0
        self._adopted = False

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
            vals = {k: float(v) for k, v in self._adapters.get(engine).parse_metrics(text).items() if isinstance(v, int | float)}
        else:
            vals = parse_generic_metrics(engine, text)
        # Prometheus text allows +Inf/NaN; they would reach SSE as `Infinity`/`NaN`, which is not JSON (review L3)
        return {k: v for k, v in vals.items() if math.isfinite(v)}

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
        self._claim(inst.host_id, inst.id, t)
        self._ring[inst.id].append(pt)
        self._rollup(inst.id, pt, inst.host_id)
        self._bus.publish(TOPIC, {"instance_id": inst.id, **pt.model_dump()})

    # -- bounded state (review M7) ----------------------------------------------------------------------------------
    def _claim(self, host_id: str, key: str, t: float) -> None:
        """Note that `key` was updated; a host that already has MAX_KEYS_PER_HOST series loses its stalest one first."""
        keys = self._keys.setdefault(host_id, {})
        if key not in keys and len(keys) >= MAX_KEYS_PER_HOST:
            keep = system_key(host_id)
            victim = min((k for k in keys if k != keep), key=lambda k: keys[k], default=None)
            if victim is not None:
                self._forget(host_id, victim, rollups=True)
        keys[key] = t

    def _forget(self, host_id: str, key: str, *, rollups: bool) -> None:
        self._keys.get(host_id, {}).pop(key, None)
        self._ring.pop(key, None)
        self._last_counters.pop(key, None)
        self._last_push.pop(key, None)
        self._last_pull.pop(key, None)
        if rollups:
            self._db.execute("DELETE FROM metric_rollups WHERE instance_id=? AND host_id IS ?", (key, host_id))

    def _adopt(self) -> None:
        """After a restart, count the series already in SQLite against their host's cap (once)."""
        self._adopted = True
        for r in self._db.all("SELECT host_id, instance_id, MAX(bucket_ts) AS last FROM metric_rollups "
                              "WHERE host_id IS NOT NULL GROUP BY host_id, instance_id"):
            self._keys.setdefault(r["host_id"], {}).setdefault(r["instance_id"], float(r["last"]))
        for host_id, keys in self._keys.items():     # an over-full host (cap lowered, or a past flood) is cut back now
            while len(keys) > MAX_KEYS_PER_HOST:
                self._forget(host_id, min(keys, key=lambda k: keys[k]), rollups=True)

    def prune(self, t: float | None = None, *, force: bool = False) -> None:
        """Drop series nobody updated for KEY_TTL_S (a host removed from hosts.json or gone for good follows the same
        rule, so a short link blip loses nothing), forget per-host bookkeeping of unknown hosts and delete rollup rows
        past their retention. Cheap and self-throttled: it runs at most once per PRUNE_EVERY_S."""
        t = self._clock() if t is None else t
        if not force and t < self._next_prune:
            return
        self._next_prune = t + PRUNE_EVERY_S
        if not self._adopted:
            self._adopt()
        for host_id in list(self._keys):
            keys = self._keys[host_id]
            for key in [k for k, last in keys.items() if t - last > KEY_TTL_S]:
                self._forget(host_id, key, rollups=False)   # the rows stay until their retention runs out
            if not keys:
                del self._keys[host_id]
        known = {h.id for h in self._fleet.hosts()}
        for book in (self._last_seq, self._configured):
            for host_id in [h for h in book if h not in known]:
                del book[host_id]
        self._warned = {w for w in self._warned    # one warning per connection: older ones are dead weight
                        if w[0] in known and self._fleet.node(w[0]).link().connected_since == w[1]}
        for res, keep in (("h", HOURLY_KEEP_S), ("d", DAILY_KEEP_S)):
            self._db.execute("DELETE FROM metric_rollups WHERE res=? AND bucket_ts < ?", (res, int(t - keep)))

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

    def _rollup(self, key: str, pt: MetricPoint, host_id: str) -> None:
        rows = []
        for res, step in (("h", 3600), ("d", 86400)):
            bucket = int(pt.t // step * step)
            rows += [(key, res, bucket, k, v, 1, v, host_id) for k, v in pt.values.items()]
        self._db.executemany(
            "INSERT INTO metric_rollups(instance_id,res,bucket_ts,key,sum,count,max,host_id) VALUES(?,?,?,?,?,?,?,?) "
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
        self._claim(host_id, key, t)
        self._ring[key].append(pt)
        self._rollup(key, pt, host_id)
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

    async def _prune_round(self) -> None:
        self.prune()

    async def run(self) -> None:
        consumer = asyncio.create_task(self.consume(), name="metrics-consumer")
        try:
            while True:
                for step in (self.sync_hosts, self.scrape_once, self._prune_round):
                    try:
                        await step()
                    except Exception:   # noqa: BLE001 - a bad round must not end the loop
                        log.exception("metrics round failed (%s)", step.__name__)
                await asyncio.sleep(self._interval)
        finally:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
