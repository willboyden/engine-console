"""Write backend/parity/routes.python.json: the live Python app's route table (method, path, params, body, status).

Run from the repository root with the Python reference (gen/pyref.py; it imports the Python app):
    "$(node backend/parity/lib/python-release.mjs)/backend/.venv/bin/python" -I backend/parity/gen/routes_python.py
The JSON is committed, so the parity harness keeps working after the Python backend is removed.
Everything runs against a throwaway data dir with the fleet off: no node is started, nothing is written elsewhere.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi.routing import APIRoute, _EffectiveRouteContext, _IncludedRouter

from engine_console.config import Settings
from engine_console.main import create_app

OUT = Path(__file__).resolve().parents[1] / "routes.python.json"


def _param(p: Any) -> dict[str, Any]:
    info = p.field_info
    required = info.is_required()
    out: dict[str, Any] = {"name": p.alias or p.name, "required": required}
    out["type"] = str(info.annotation).replace("typing.", "")
    if not required:
        d = info.default
        out["default"] = d if isinstance(d, (str, int, float, bool)) or d is None else repr(d)
    for m in getattr(info, "metadata", []) or []:
        pat = getattr(m, "pattern", None)
        if pat:
            out["pattern"] = pat
        ml = getattr(m, "max_length", None)
        if ml is not None:
            out["max_length"] = ml
    return out


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Settings(data_dir=Path(tmp) / "data", fleet_enabled=False, background_tasks=False, otlp_enabled=False,
                       hosts_file=Path(tmp) / "hosts.json", hf_cache_dir=Path(tmp) / "hf",
                       # this checkout's frontend: the SPA fallback route exists only when the directory does
                       frontend_dir=Path(__file__).resolve().parents[3] / "frontend")
        app = create_app(cfg)
        routes: list[dict[str, Any]] = []

        def walk(items: list[Any]) -> None:
            for r in items:
                if isinstance(r, _IncludedRouter):
                    walk(r.effective_candidates())
                    walk(r.effective_low_priority_routes())
                    continue
                ctx = r if isinstance(r, _EffectiveRouteContext) else None
                route = r if isinstance(r, APIRoute) else None
                if ctx is None and route is None:
                    continue
                src = ctx if ctx is not None else route
                d = src.dependant
                body = None
                if d is not None and d.body_params:
                    bp = d.body_params[0]
                    ann = bp.field_info.annotation
                    body = {"name": bp.name, "type": str(getattr(ann, "__name__", ann)).replace("typing.", ""),
                            "required": bp.field_info.is_required()}
                for m in sorted(src.methods or []):
                    routes.append({
                        "order": len(routes), "method": m, "path": src.path, "name": src.name,
                        "module": src.endpoint.__module__, "status_code": src.status_code,
                        "path_params": [_param(p) for p in (d.path_params if d else [])],
                        "query_params": [_param(p) for p in (d.query_params if d else [])],
                        "body": body, "include_in_schema": src.include_in_schema,
                    })

        walk(list(app.router.routes))
        doc = {"generated_by": "backend/parity/gen/routes_python.py", "count": len(routes), "routes": routes}
        OUT.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {OUT} ({len(routes)} routes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
