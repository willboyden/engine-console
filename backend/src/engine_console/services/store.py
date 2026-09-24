"""SQLite (WAL) store with ordered SQL migrations. One connection guarded by a lock: single node,
short queries, so simplicity beats a pool."""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

Row = sqlite3.Row


class Store:
    def __init__(self, path: Path | str) -> None:
        self._lock = threading.RLock()
        if str(path) != ":memory:":
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        if str(path) != ":memory:":
            Path(path).chmod(0o600)   # prompts/chats can be private
        self.migrate()

    # -- migrations ---------------------------------------------------------------------------
    def migrate(self) -> list[str]:
        applied: list[str] = []
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, applied_at REAL)")
            done = {r["name"] for r in self._conn.execute("SELECT name FROM schema_migrations")}
            files = sorted((f for f in resources.files("engine_console.migrations").iterdir()
                            if f.name.endswith(".sql")), key=lambda f: f.name)
            for f in files:
                if f.name in done:
                    continue
                script = f.read_text(encoding="utf-8")
                try:
                    self._conn.execute("BEGIN")
                    for stmt in _split_sql(script):
                        self._conn.execute(stmt)
                    self._conn.execute("INSERT INTO schema_migrations VALUES (?, strftime('%s','now'))", (f.name,))
                    self._conn.execute("COMMIT")
                except Exception:
                    self._conn.execute("ROLLBACK")
                    raise
                applied.append(f.name)
        return applied

    # -- helpers ------------------------------------------------------------------------------
    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, rows)

    def one(self, sql: str, params: Sequence[Any] = ()) -> Row | None:
        with self._lock:
            row: Row | None = self._conn.execute(sql, params).fetchone()
            return row

    def all(self, sql: str, params: Sequence[Any] = ()) -> list[Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _split_sql(script: str) -> list[str]:
    """Split on ';' at statement ends, keeping CREATE TRIGGER ... BEGIN ... END; bodies intact."""
    out: list[str] = []
    buf: list[str] = []
    in_trigger = False
    for line in script.splitlines():
        s = line.strip()
        if not s or s.startswith("--"):
            continue
        buf.append(line)
        if s.upper().startswith("CREATE TRIGGER"):
            in_trigger = True
        if in_trigger:
            if s.upper().rstrip(";").endswith("END"):
                out.append("\n".join(buf))
                buf, in_trigger = [], False
        elif s.endswith(";"):
            out.append("\n".join(buf))
            buf = []
    if buf:
        out.append("\n".join(buf))
    return out


def paginate(limit: int, cursor: str | None) -> tuple[int, int]:
    """Opaque cursor == integer offset. Returns (limit, offset); callers fetch limit+1 to detect a next page."""
    try:
        offset = max(0, int(cursor)) if cursor else 0
    except ValueError:
        offset = 0
    return max(1, min(limit, 500)), offset
