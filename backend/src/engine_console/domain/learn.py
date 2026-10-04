"""Learn mode: plain-language explanations for every parameter, metric, state and term in the UI.

LOCKED CONTRACT v2 (docs/ARCHITECTURE.md §10.7). Content lives in `engine_console/learn/*.json` (format and
style guide: learn/README.md) and reaches the browser as one bundle from GET /api/v1/learn.
"""
from __future__ import annotations

from pydantic import BaseModel

from engine_console.adapters.base import Explain


class Term(BaseModel):
    """One glossary entry, e.g. key "kv_cache"."""
    title: str                 # display name: "KV cache"
    short: str                 # <= 160 chars, one plain sentence
    long: str = ""             # <= 1500 chars: the full explanation, analogies welcome
    see_also: list[str] = []   # other glossary keys


class LearnBundle(BaseModel):
    version: int = 1
    glossary: dict[str, Term] = {}                 # term key -> Term
    metrics: dict[str, Explain] = {}               # canonical metric key (ARCHITECTURE §5) -> Explain
    concepts: dict[str, Explain] = {}              # UI concept key ("state.loading", "verdict.tight", ...) -> Explain
    params: dict[str, dict[str, Explain]] = {}     # engine id -> param key -> Explain
    presets: dict[str, dict[str, Explain]] = {}    # engine id -> preset name -> Explain
