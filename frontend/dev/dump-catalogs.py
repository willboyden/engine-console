#!/usr/bin/env python3
"""Dump what GET /engines and GET /engines/{id}/params serve, from the real adapters and learn content (library code only: no
service is started), for `node dev/mock-server.mjs --catalogs-file <out.json>`. That lets the Launch form and the wizard be checked
on the production parameters and wording instead of the mock's samples.

Usage (from the repository root, with the backend's own virtualenv):
    backend/.venv/bin/python frontend/dev/dump-catalogs.py /tmp/catalogs.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend" / "src"))

from engine_console.adapters import load_default_adapters  # noqa: E402
from engine_console.learn import bundle  # noqa: E402


def main(out_path: str) -> None:
    learn = bundle()
    out: dict[str, object] = {"engines": [], "params": {}}
    for a in load_default_adapters().list():
        out["engines"].append({  # type: ignore[union-attr]
            "id": a.id, "display_name": a.display_name, "default_image": a.default_image,
            "presets": [{"name": n, "params": p} for n, p in a.presets().items()],
            "presets_info": {n: e.model_dump(mode="json") for n, e in (learn.presets.get(a.id) or a.preset_info()).items()},
            "model_formats": sorted(a.model_formats), "images_by_arch": dict(a.images_by_arch), "multi_model": a.multi_model})
        explained = learn.params.get(a.id, {})
        out["params"][a.id] = [  # type: ignore[index]
            p.model_copy(update={"explain": explained.get(p.key), "level": p.level or ("advanced" if p.advanced else "basic")}).model_dump(mode="json")
            for p in a.param_catalog()]
    Path(out_path).write_text(json.dumps(out), encoding="utf-8")
    print({e: len(ps) for e, ps in out["params"].items()})  # type: ignore[attr-defined]


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "catalogs.json")
