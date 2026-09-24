"""One-click benchmark against a ready instance, talking to its OpenAI endpoint directly.
Measures TTFT / ITL / e2e from a streaming client, decode and prefill tok/s from usage counts, throughput per
concurrency level, and an optional prefix-cache (cold vs warm TTFT) probe."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from engine_console.domain.errors import BadRequest, Conflict, NotFound
from engine_console.domain.models import BenchRun, Page
from engine_console.services.common import EventBus, new_id, now, percentile
from engine_console.services.lifecycle import LifecycleService
from engine_console.services.store import Store, paginate

log = logging.getLogger(__name__)
SUITES: dict[str, dict[str, Any]] = {
    "quick": {"concurrency": [1, 4], "prompt_tokens": 512, "max_tokens": 128, "rounds": 1, "prefix_cache": False},
    "standard": {"concurrency": [1, 4, 16, 64], "prompt_tokens": 1024, "max_tokens": 256, "rounds": 2, "prefix_cache": True},
}
MAX_CONCURRENCY = 64   # hard ceilings: a bench suite must not become a self-inflicted DoS
MAX_LEVELS = 8
_FILLER = "The quick brown fox jumps over the lazy dog while counting to ten. "   # roughly 14 tokens


@dataclass
class ReqStat:
    ttft: float | None = None
    e2e: float = 0.0
    itl: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


def make_prompt(approx_tokens: int, nonce: str | None) -> str:
    body = _FILLER * max(1, approx_tokens // 14)
    return f"[{nonce}] {body}\nSummarise the text above in one sentence." if nonce else f"{body}\nSummarise the text above in one sentence."


async def one_request(http: httpx.AsyncClient, base: str, model: str, prompt: str, max_tokens: int) -> ReqStat:
    st = ReqStat()
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0.7}
    t0 = time.perf_counter()
    last = t0
    try:
        async with http.stream("POST", f"{base}/v1/chat/completions", json=body,
                               timeout=httpx.Timeout(10.0, read=300.0)) as r:
            if r.status_code != 200:
                st.error = f"HTTP {r.status_code}"
                return st
            async for line in r.aiter_lines():
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                try:
                    ev = json.loads(line[5:])
                except ValueError:
                    continue
                if ev.get("usage"):
                    st.prompt_tokens = int(ev["usage"].get("prompt_tokens", 0))
                    st.completion_tokens = int(ev["usage"].get("completion_tokens", 0))
                ch = (ev.get("choices") or [{}])[0].get("delta", {})
                if ch.get("content") or ch.get("reasoning_content") or ch.get("reasoning"):
                    t = time.perf_counter()
                    if st.ttft is None:
                        st.ttft = t - t0
                    else:
                        st.itl.append(t - last)
                    last = t
    except httpx.HTTPError as e:
        st.error = type(e).__name__
    st.e2e = time.perf_counter() - t0
    return st


def summarize(stats: list[ReqStat]) -> dict[str, Any]:
    ok = [s for s in stats if s.error is None and s.ttft is not None]
    ttft = [s.ttft for s in ok if s.ttft is not None]
    e2e = [s.e2e for s in ok]
    itl = [x for s in ok for x in s.itl]
    dec = [s.completion_tokens / (s.e2e - s.ttft) for s in ok if s.ttft is not None and s.e2e > s.ttft and s.completion_tokens]
    pre = [s.prompt_tokens / s.ttft for s in ok if s.ttft and s.prompt_tokens]
    return {"requests": len(stats), "errors": len(stats) - len(ok), "ttft_p50_s": percentile(ttft, 50),
            "ttft_p95_s": percentile(ttft, 95), "itl_p50_s": percentile(itl, 50), "itl_p95_s": percentile(itl, 95),
            "e2e_p50_s": percentile(e2e, 50), "e2e_p95_s": percentile(e2e, 95),
            "decode_tps": sum(dec) / len(dec) if dec else None, "prefill_tps": sum(pre) / len(pre) if pre else None}


class BenchService:
    def __init__(self, store: Store, lifecycle: LifecycleService, http: httpx.AsyncClient, bus: EventBus) -> None:
        self._db = store
        self._life = lifecycle
        self._http = http
        self._bus = bus
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @staticmethod
    def resolve_suite(suite: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(suite, str):
            if suite not in SUITES:
                raise BadRequest(f"unknown suite '{suite}'", code="unknown_suite", available=sorted(SUITES))
            return dict(SUITES[suite])
        merged = {**SUITES["quick"], **suite}
        conc = merged["concurrency"]
        unknown = set(suite) - set(SUITES["quick"]) if isinstance(suite, dict) else set()
        if unknown:
            raise BadRequest(f"unknown suite fields: {sorted(unknown)}", code="invalid_suite")
        if not (isinstance(conc, list) and 1 <= len(conc) <= MAX_LEVELS
                and all(isinstance(c, int) and not isinstance(c, bool) and 1 <= c <= MAX_CONCURRENCY for c in conc)):
            raise BadRequest(f"concurrency must be 1..{MAX_LEVELS} ints in 1..{MAX_CONCURRENCY}", code="invalid_suite")
        for k, lo, hi in (("prompt_tokens", 1, 32_768), ("max_tokens", 1, 4_096), ("rounds", 1, 5)):
            if not (isinstance(merged[k], int) and lo <= merged[k] <= hi):
                raise BadRequest(f"{k} must be an int in {lo}..{hi}", code="invalid_suite")
        return merged

    def _row(self, r: Any) -> BenchRun:
        return BenchRun(id=r["id"], instance_id=r["instance_id"], engine=r["engine"], repo_id=r["repo_id"],
                        profile_id=r["profile_id"], params=json.loads(r["params"]), suite=json.loads(r["suite"]),
                        state=r["state"], results=json.loads(r["results"]), error=r["error"],
                        created_at=r["created_at"], finished_at=r["finished_at"])

    def get(self, bid: str) -> BenchRun:
        r = self._db.one("SELECT * FROM bench_runs WHERE id=?", (bid,))
        if r is None:
            raise NotFound(f"no such benchmark: {bid}")
        return self._row(r)

    def list_page(self, limit: int = 50, cursor: str | None = None, instance_id: str | None = None) -> Page[BenchRun]:
        limit, off = paginate(limit, cursor)
        where, args = ("WHERE instance_id=?", [instance_id]) if instance_id else ("", [])
        rows = self._db.all(f"SELECT * FROM bench_runs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",  # noqa: S608
                            [*args, limit + 1, off])
        return Page[BenchRun](items=[self._row(r) for r in rows[:limit]],
                              next_cursor=str(off + limit) if len(rows) > limit else None)

    async def start(self, instance_id: str, suite: str | dict[str, Any]) -> BenchRun:
        inst = self._life.get(instance_id)
        if inst.state != "ready" or not inst.port:
            raise Conflict(f"instance is {inst.state}; benchmarks need a ready instance", code="instance_not_ready")
        spec = self.resolve_suite(suite)
        bid = new_id("bench_")
        self._db.execute("INSERT INTO bench_runs(id,instance_id,engine,repo_id,profile_id,params,suite,state,results,created_at)"
                         " VALUES(?,?,?,?,?,?,?,?,?,?)",
                         (bid, instance_id, inst.engine, inst.repo_id, inst.profile_id, json.dumps(inst.params),
                          json.dumps(spec), "running", "{}", now()))
        self._tasks[bid] = asyncio.create_task(self._run(bid, inst.port, inst.repo_id, spec), name=f"bench-{bid}")
        return self.get(bid)

    async def wait(self, bid: str) -> None:
        t = self._tasks.get(bid)
        if t:
            await asyncio.gather(t, return_exceptions=True)

    async def shutdown(self) -> None:
        for t in self._tasks.values():
            t.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    def _emit(self, bid: str, **ev: Any) -> None:
        self._bus.publish(f"bench:{bid}", {"id": bid, **ev})

    async def _run(self, bid: str, port: int, model: str, spec: dict[str, Any]) -> None:
        base = f"http://127.0.0.1:{port}"
        results: dict[str, Any] = {"concurrency": []}
        try:
            for c in spec["concurrency"]:
                self._emit(bid, event="level_start", concurrency=c)
                stats: list[ReqStat] = []
                t0 = time.perf_counter()
                for _ in range(spec["rounds"]):
                    batch = await asyncio.gather(*(one_request(
                        self._http, base, model, make_prompt(spec["prompt_tokens"], uuid.uuid4().hex[:8]),
                        spec["max_tokens"]) for _ in range(c)))
                    stats.extend(batch)
                wall = time.perf_counter() - t0
                level = {"concurrency": c, **summarize(stats),
                         "throughput_tps": sum(s.completion_tokens for s in stats) / wall if wall > 0 else None}
                results["concurrency"].append(level)
                if c == 1:
                    results["single"] = {k: v for k, v in level.items() if k != "concurrency"}
                self._emit(bid, event="level_done", level=level)
            if spec.get("prefix_cache"):
                prompt = make_prompt(spec["prompt_tokens"], uuid.uuid4().hex[:8])
                cold = await one_request(self._http, base, model, prompt, 8)
                warm = await one_request(self._http, base, model, prompt, 8)
                if cold.ttft and warm.ttft:
                    results["prefix_cache"] = {"cold_ttft_s": cold.ttft, "warm_ttft_s": warm.ttft,
                                               "speedup": cold.ttft / warm.ttft}
            self._db.execute("UPDATE bench_runs SET state='completed', results=?, finished_at=? WHERE id=?",
                             (json.dumps(results), now(), bid))
        except asyncio.CancelledError:
            self._db.execute("UPDATE bench_runs SET state='failed', error='cancelled', finished_at=? WHERE id=?", (now(), bid))
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("bench %s failed", bid)
            self._db.execute("UPDATE bench_runs SET state='failed', error=?, results=?, finished_at=? WHERE id=?",
                             (f"{type(e).__name__}: {e}", json.dumps(results), now(), bid))
        self._emit(bid, event="done", state=self.get(bid).state)
