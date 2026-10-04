"""Load the learn-mode content (README.md in this directory) into a validated LearnBundle.

Read once per process. A malformed file fails loudly at start-up (and in tests) instead of showing a half-empty
help popover later.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from engine_console.adapters.base import Explain
from engine_console.domain.learn import LearnBundle, Term

LEARN_DIR = Path(__file__).resolve().parent


def _read(path: Path, section: str) -> dict[str, Any]:
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get(section), dict):
        raise ValueError(f"{path.name}: expected {{\"version\": 1, \"{section}\": {{...}}}}")
    out: dict[str, Any] = doc[section]
    return out


def _per_engine(folder: Path, section: str) -> dict[str, dict[str, Explain]]:
    if not folder.is_dir():
        return {}
    return {p.stem: {k: Explain.model_validate(v) for k, v in _read(p, section).items()}
            for p in sorted(folder.glob("*.json"))}


def load_bundle(root: Path = LEARN_DIR) -> LearnBundle:
    return LearnBundle(
        glossary={k: Term.model_validate(v) for k, v in _read(root / "glossary.json", "terms").items()},
        metrics={k: Explain.model_validate(v) for k, v in _read(root / "metrics.json", "metrics").items()},
        concepts={k: Explain.model_validate(v) for k, v in _read(root / "concepts.json", "concepts").items()},
        params=_per_engine(root / "params", "params"),
        presets=_per_engine(root / "presets", "presets"),
    )


@lru_cache(maxsize=1)
def bundle() -> LearnBundle:
    """The packaged content, loaded once."""
    return load_bundle()
