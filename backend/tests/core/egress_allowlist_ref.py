"""A tiny reference implementation of a glob allowlist (one host glob per line, '#' comments, default deny).

This mirrors how typical forward-proxy allowlists match hosts, so tests can check that the hosts the console
needs are covered by an entry list and that lookalike hosts are not."""
from __future__ import annotations

import fnmatch


def load_patterns(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def allowed(host: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(host, p) for p in patterns)
