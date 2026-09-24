"""Metrics scraping: every ready instance's Prometheus endpoint -> ring buffer (15 min) + SQLite rollups (1 h / 1 d)."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable

import httpx

from engine_console.adapters import AdapterRegistry
from engine_console.domain.errors import BadRequest
from engine_console.domain.models import MetricPoint
from engine_console.services.common import EventBus
from engine_console.services.discovery import parse_generic_metrics
from engine_console.services.lifecycle import LifecycleService
from engine_console.services.store import Store

log = logging.getLogger(__name__)
RING_SECONDS = 15 * 60
TOPIC = "metrics"
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_window(w: str) -> int:
    if len(w) < 2 or w[-1] not in _UNITS or not w[:-1].isdigit():
        raise BadRequest("window must look like 15m, 1h, 6h, 1d, 7d", code="invalid_window")
    return int(w[:-1]) * _UNITS[w[-1]]


class MetricsService:
    def __init__(self, store: Store, lifecycle: LifecycleService, adapters: AdapterRegistry, http: httpx.AsyncClient,
                 bus: EventBus, *, interval_s: float = 2.0, clock: Callable[[], float] = time.time) -> None:
        self._db = store
        self._life = lifecycle
        self._adapters = adapters
        self._http = http
        self._bus = bus
        self._interval = interval_s
        self._clock = clock
        self._ring: dict[str, deque[MetricPoint]] = defaultdict(lambda: deque(maxlen=int(RING_SECONDS / max(interval_s, 0.5)) + 1))
        self._last_counters: dict[str, tuple[float, dict[str, float]]] = {}

    async def scrape_once(self) -> None:
        await asyncio.gather(*(self._scrape(t.iid, t.engine, t.port, t.path) for t in self._life.scrape_targets()),
                             return_exceptions=True)

    def _parse(self, engine: str, text: str) -> dict[str, float]:
        if engine in self._adapters:   # vllm / sglang: their adapters know the metric names
            return {k: float(v) for k, v in self._adapters.get(engine).parse_metrics(text).items() if isinstance(v, int | float)}
        return parse_generic_metrics(engine, text)

    async def _scrape(self, iid: str, engine: str, port: int, path: str) -> None:
        try:
            r = await self._http.get(f"http://127.0.0.1:{port}{path}", timeout=3.0)
            r.raise_for_status()
        except httpx.HTTPError:
            return
        vals = self._parse(engine, r.text[:2_000_000])
        t = self._clock()
        self._derive_rates(iid, t, vals)
        if vals.get("requests_running", 0) > 0:
            self._life.touch(iid)
        pt = MetricPoint(t=t, values=vals)
        self._ring[iid].append(pt)
        self._rollup(iid, pt)
        self._bus.publish(TOPIC, {"instance_id": iid, **pt.model_dump()})

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

    def _rollup(self, iid: str, pt: MetricPoint) -> None:
        rows = []
        for res, step in (("h", 3600), ("d", 86400)):
            bucket = int(pt.t // step * step)
            rows += [(iid, res, bucket, k, v, 1, v) for k, v in pt.values.items()]
        self._db.executemany(
            "INSERT INTO metric_rollups(instance_id,res,bucket_ts,key,sum,count,max) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(instance_id,res,bucket_ts,key) DO UPDATE SET sum=sum+excluded.sum, count=count+1, "
            "max=MAX(max, excluded.max)", rows)

    def series(self, iid: str, window: str = "15m") -> dict[str, object]:
        inst = self._life.get(iid)  # 404 for unknown instances
        extra = {"history_since": inst.history_since} if not inst.managed else {}
        secs = parse_window(window)
        since = self._clock() - secs
        if secs <= RING_SECONDS:
            pts = [p for p in self._ring.get(iid, ()) if p.t >= since]
            return {"instance_id": iid, "window": window, "resolution": "raw", "points": [p.model_dump() for p in pts], **extra}
        res, step = ("h", 3600) if secs <= 3 * 86400 else ("d", 86400)
        rows = self._db.all("SELECT bucket_ts, key, sum, count, max FROM metric_rollups WHERE instance_id=? AND res=? "
                            "AND bucket_ts >= ? ORDER BY bucket_ts", (iid, res, int(since // step * step)))
        by_bucket: dict[int, dict[str, float]] = defaultdict(dict)
        for r in rows:
            by_bucket[r["bucket_ts"]][r["key"]] = r["sum"] / r["count"]
        return {"instance_id": iid, "window": window, "resolution": "1h" if res == "h" else "1d",
                "points": [{"t": float(b), "values": v} for b, v in sorted(by_bucket.items())], **extra}

    def latest(self, iid: str) -> MetricPoint | None:
        ring = self._ring.get(iid)
        return ring[-1] if ring else None

    async def run(self) -> None:
        while True:
            await self.scrape_once()
            await asyncio.sleep(self._interval)
