"""Secret engine parameters (api keys etc.): never in the DB, API responses, profiles or audit.

The DB/API only ever carry MARKER. The real value lives in a 0600 file per instance, is read only at
container launch, and reaches the container as an env var *name* (value in the child process env)."""
from __future__ import annotations

import contextlib
import json
import os
import re
from pathlib import Path
from typing import Any

SECRET_KEY = re.compile(r"(key|token|secret|password)", re.I)
MARKER = "[set]"


def split_secrets(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """(params with real secret values replaced by MARKER, {key: real value})."""
    public: dict[str, Any] = {}
    secrets: dict[str, str] = {}
    for k, v in params.items():
        if SECRET_KEY.search(k) and isinstance(v, str) and v and v != MARKER:
            secrets[k] = v
            public[k] = MARKER
        else:
            public[k] = v
    return public, secrets


class SecretStore:
    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def _path(self, iid: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_]+", iid):
            raise ValueError("bad instance id")
        return self._dir / f"{iid}.json"

    def put(self, iid: str, secrets: dict[str, str]) -> None:
        if not secrets:
            return
        self._dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._dir.chmod(0o700)
        fd = os.open(self._path(iid), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)   # O_CREAT's mode is ignored for a pre-existing, possibly looser file
        with os.fdopen(fd, "w") as fh:
            json.dump(secrets, fh)

    def get(self, iid: str) -> dict[str, str]:
        try:
            data = json.loads(self._path(iid).read_text())
        except (OSError, ValueError):
            return {}
        return {k: v for k, v in data.items() if isinstance(v, str)} if isinstance(data, dict) else {}

    def delete(self, iid: str) -> None:
        with contextlib.suppress(OSError):
            self._path(iid).unlink()
