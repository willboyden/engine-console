from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from engine_console import __version__
from engine_console.adapters.base import ParamSpec
from engine_console.api.deps import C
from engine_console.domain.models import FitReport, FitRequest, HardwareReport, Page
from engine_console.domain.repo_id import RepoId, Revision
from engine_console.services.hf import ModelDetail, SearchHit

router = APIRouter()


@router.get("/health")
async def health(c: C) -> dict[str, Any]:
    proxied = bool(c.cfg.egress_proxy)
    warnings = [] if proxied else [
        "console egress is DIRECT: Hugging Face traffic bypasses the mitmproxy chokepoint (only the in-app allowlist "
        "applies). Set EGRESS_PROXY (and run security/egress/mitmproxy/run.sh) to enforce it."]
    if c.cfg.trust_loopback:
        warnings.append("trust_loopback is ON: any local process or browser-reachable loopback client is treated as admin. "
                        "Keep the CSRF header/Host checks intact, and set trust_loopback=false on shared machines.")
    return {"status": "ok", "version": __version__, "engines": [a.id for a in c.adapters.list()],
            "docker_context": c.cfg.docker_context, "egress_mode": "proxied" if proxied else "direct",
            "warnings": warnings}


@router.get("/hardware")
async def hardware(c: C) -> HardwareReport:
    return c.hardware.report()


@router.get("/engines")
async def engines(c: C) -> list[dict[str, Any]]:
    return [{"id": a.id, "display_name": a.display_name, "default_image": c.settings.image_pin(a.id) or a.default_image,
             "presets": [{"name": n, "params": p} for n, p in a.presets().items()]} for a in c.adapters.list()]


@router.get("/engines/{engine}/params")
async def engine_params(engine: str, c: C) -> list[ParamSpec]:
    return c.adapters.get(engine).param_catalog()


@router.get("/hf/search")
async def hf_search(c: C, q: str = "", task: str | None = None, library: str | None = None, quant: str | None = None,
                    max_gib: float | None = None, gated: bool | None = None, sort: str = "downloads",
                    limit: int = 20, cursor: str | None = None) -> Page[SearchHit]:
    return await c.hf.search(q=q, task=task, library=library, quant=quant, max_gib=max_gib, gated=gated, sort=sort,
                             limit=limit, cursor=cursor)


@router.get("/hf/models/{repo_id:path}")
async def hf_model(repo_id: RepoId, c: C, revision: Revision | None = None) -> ModelDetail:
    return await c.hf.detail(repo_id, revision)


@router.post("/fit")
async def fit(body: FitRequest, c: C) -> FitReport:
    return await c.fit.assess(body)
