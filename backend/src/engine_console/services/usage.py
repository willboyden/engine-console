"""Per-model / per-day / per-hour usage analytics from usage_events rows written by chat/arena/bench."""
from __future__ import annotations

import csv
import io

from engine_console.domain.errors import BadRequest
from engine_console.domain.models import UsageRow
from engine_console.services.common import now
from engine_console.services.store import Store

_GROUPS = {
    "model": "model",
    "day": "strftime('%Y-%m-%d', ts, 'unixepoch')",
    "hour": "strftime('%Y-%m-%dT%H:00Z', ts, 'unixepoch')",
}


class UsageService:
    def __init__(self, store: Store) -> None:
        self._db = store

    def record(self, *, model: str, instance_id: str | None, prompt_tokens: int, completion_tokens: int,
               latency_ms: float, source: str) -> None:
        self._db.execute("INSERT INTO usage_events(ts,model,instance_id,prompt_tokens,completion_tokens,latency_ms,source)"
                         " VALUES(?,?,?,?,?,?,?)",
                         (now(), model, instance_id, prompt_tokens, completion_tokens, latency_ms, source))

    def query(self, group_by: str = "model", since: float | None = None, until: float | None = None) -> list[UsageRow]:
        expr = _GROUPS.get(group_by)
        if expr is None:
            raise BadRequest(f"group_by must be one of {sorted(_GROUPS)}", code="invalid_group_by")
        # expr comes from the fixed table above, never from user input
        rows = self._db.all(
            f"SELECT {expr} AS k, COUNT(*) AS n, SUM(prompt_tokens) AS p, SUM(completion_tokens) AS c, "  # noqa: S608
            "AVG(latency_ms) AS l FROM usage_events WHERE ts >= ? AND ts <= ? GROUP BY k ORDER BY k",
            (since or 0.0, until or now() + 1))
        return [UsageRow(key=r["k"], requests=r["n"], prompt_tokens=r["p"] or 0, completion_tokens=r["c"] or 0,
                         avg_latency_ms=round(r["l"] or 0.0, 2)) for r in rows]

    def export_csv(self, group_by: str = "model") -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["key", "requests", "prompt_tokens", "completion_tokens", "avg_latency_ms"])
        for r in self.query(group_by):
            w.writerow([_csv_safe(r.key), r.requests, r.prompt_tokens, r.completion_tokens, r.avg_latency_ms])
        return buf.getvalue()


def _csv_safe(v: str) -> str:
    # model names are user-influenced: neutralise spreadsheet formula injection
    return "'" + v if v[:1] in ("=", "+", "-", "@") else v
