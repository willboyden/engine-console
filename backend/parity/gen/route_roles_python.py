"""Write backend/crates/ec-hub/tests/data/route_roles.json: the role each route of the live Python app requires.

Run from the repository root with the Python reference (gen/pyref.py):
    "$(node backend/parity/lib/python-release.mjs)/backend/.venv/bin/python" -I backend/parity/gen/route_roles_python.py
The Rust hub declares a role per route (api/routes/mod.rs) and enforces it in the request pipeline (app.rs); its test
`tests/test_route_roles.rs` compares that table with this file, so a route whose Python dependencies change (an `_:
Admin` added or removed, a router included without `enforce_roles`) fails the Rust tests until the Rust table follows.

Roles, from the route's dependencies in FastAPI's resolution order (depth first, router dependencies first):
  "admin"   the chain holds `require_admin` (`_: Admin`): 403 "admin role required" for anything but an admin;
  "caller"  the chain holds `enforce_roles` (every API router's dependency) and no `require_admin`;
  "open"    no role dependency at all (the OpenAPI document, /metrics, the SPA).
`guarded` is the SecurityMiddleware's own rule (`_guarded(path)`): whether a request needs a principal at all.
Everything runs against a throwaway data dir with the fleet off: no node is started, nothing is written elsewhere.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi.routing import APIRoute, _EffectiveRouteContext, _IncludedRouter
from starlette.routing import Route

from engine_console.api.security import _guarded
from engine_console.config import Settings
from engine_console.main import create_app

OUT = Path(__file__).resolve().parents[2] / "crates" / "ec-hub" / "tests" / "data" / "route_roles.json"
ROLE_DEPS = ("enforce_roles", "require_admin")


def _chain(dependant: Any) -> list[str]:
    """Every dependency callable's name, sub-dependencies first (the order FastAPI's solve_dependencies runs them)."""
    out: list[str] = []
    for sub in dependant.dependencies:
        out.extend(_chain(sub))
        call = getattr(sub, "call", None)
        if call is not None:
            out.append(getattr(call, "__name__", repr(call)))
    return out


def _role(deps: list[str]) -> str:
    if "require_admin" in deps:
        return "admin"
    if "enforce_roles" in deps:
        return "caller"
    return "open"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        front = Path(tmp) / "frontend"
        front.mkdir()   # the SPA route exists only when the frontend dir does
        cfg = Settings(data_dir=Path(tmp) / "data", fleet_enabled=False, background_tasks=False, otlp_enabled=False,
                       hosts_file=Path(tmp) / "hosts.json", hf_cache_dir=Path(tmp) / "hf", frontend_dir=front)
        app = create_app(cfg)
        rows: list[dict[str, Any]] = []

        def add(methods: Any, path: str, deps: list[str]) -> None:
            role_deps = [d for d in deps if d in ROLE_DEPS]
            for m in sorted(methods or []):
                rows.append({"method": m, "path": path, "role": _role(role_deps), "dependencies": role_deps,
                             "guarded": _guarded(path)})

        def walk(items: list[Any]) -> None:
            for r in items:
                if isinstance(r, _IncludedRouter):
                    walk(r.effective_candidates())
                    walk(r.effective_low_priority_routes())
                elif isinstance(r, (_EffectiveRouteContext, APIRoute)):
                    add(r.methods, r.path, _chain(r.dependant) if r.dependant is not None else [])
                elif isinstance(r, Route):
                    add(r.methods, r.path, [])   # a plain starlette route (the OpenAPI JSON): no dependencies

        walk(list(app.router.routes))
        doc = {"generated_by": "backend/parity/gen/route_roles_python.py", "count": len(rows), "routes": rows}
        OUT.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {OUT} ({len(rows)} routes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
