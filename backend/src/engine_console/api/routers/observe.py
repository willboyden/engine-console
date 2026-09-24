"""Metrics timeseries, benchmarks, usage, audit and the console's own /metrics."""
from __future__ import annotations

from collections import Counter as _Counter
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Response

from engine_console.api.deps import Admin, C, sse_response
from engine_console.domain.models import BenchRequest, BenchRun, Page, UsageRow
from engine_console.services.common import sse

router = APIRouter()
root_router = APIRouter()


@router.get("/metrics/instances/{iid}")
async def instance_metrics(iid: str, c: C, window: str = "15m") -> dict[str, object]:
    return c.metrics.series(iid, window)


@router.get("/metrics/stream")
async def metrics_stream(c: C, once: bool = False) -> Any:
    async def gen() -> AsyncIterator[str]:
        for i in c.lifecycle.all_active():
            pt = c.metrics.latest(i.id)
            if pt:
                yield sse({"instance_id": i.id, **pt.model_dump()}, "metrics")
        if once:
            return
        async for ev in c.bus.subscribe("metrics"):
            yield sse(ev, "metrics")
    return sse_response(gen(), c)


@router.post("/bench", status_code=202)
async def start_bench(body: BenchRequest, c: C) -> BenchRun:
    c.telemetry.bench_runs.inc()
    return await c.bench.start(body.instance_id, body.suite)


@router.get("/bench")
async def list_bench(c: C, limit: int = 50, cursor: str | None = None, instance_id: str | None = None) -> Page[BenchRun]:
    return c.bench.list_page(limit, cursor, instance_id)


@router.get("/bench/{bid}")
async def get_bench(bid: str, c: C) -> BenchRun:
    return c.bench.get(bid)


@router.get("/bench/{bid}/stream")
async def bench_stream(bid: str, c: C, once: bool = False) -> Any:
    run = c.bench.get(bid)

    async def gen() -> AsyncIterator[str]:
        yield sse(run.model_dump(), "snapshot")
        if once or run.state != "running":
            return
        async for ev in c.bus.subscribe(f"bench:{bid}"):
            yield sse(ev, "progress")
            if ev.get("event") == "done":
                return
    return sse_response(gen(), c)


@router.get("/usage")
async def usage(c: C, group_by: str = "model", since: float | None = None, until: float | None = None) -> list[UsageRow]:
    return c.usage.query(group_by, since, until)


@router.get("/usage/export.csv")
async def usage_csv(c: C, group_by: str = "model") -> Response:
    return Response(c.usage.export_csv(group_by), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="usage.csv"'})


@router.get("/audit")
async def audit(c: C, _: Admin, limit: int = 100, cursor: str | None = None) -> Page[dict[str, Any]]:
    return c.audit.list_page(limit, cursor)


@root_router.get("/metrics")
async def prometheus(c: C) -> Response:
    counts = _Counter(i.state for i in c.lifecycle.list_page(500).items)
    c.telemetry.set_instance_states({str(k): v for k, v in counts.items()})
    return Response(c.telemetry.render(), media_type="text/plain; version=0.0.4; charset=utf-8")


