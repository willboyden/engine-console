"""Append-only audit trail (the DB itself rejects UPDATE/DELETE, see migration 0001)."""
from __future__ import annotations

import json
from typing import Any

from engine_console.domain.models import Page
from engine_console.services.common import now, redact
from engine_console.services.store import Store, paginate


class AuditService:
    def __init__(self, store: Store) -> None:
        self._db = store

    def record(self, *, actor: str, role: str, method: str, path: str, status: int, params: Any) -> None:
        self._db.execute("INSERT INTO audit(ts,actor,role,method,path,status,params) VALUES(?,?,?,?,?,?,?)",
                         (now(), actor, role, method, path, status, json.dumps(redact(params), default=str)))

    def list_page(self, limit: int, cursor: str | None) -> Page[dict[str, Any]]:
        limit, off = paginate(limit, cursor)
        rows = self._db.all("SELECT * FROM audit ORDER BY id DESC LIMIT ? OFFSET ?", (limit + 1, off))
        items = [{**{k: r[k] for k in ("id", "ts", "actor", "role", "method", "path", "status")},
                  "params": json.loads(r["params"])} for r in rows[:limit]]
        return Page[dict[str, Any]](items=items, next_cursor=str(off + limit) if len(rows) > limit else None)
