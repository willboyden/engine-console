from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from engine_console import __version__
from engine_console.adapters.base import ParamSpec
from engine_console.api.deps import C
from engine_console.domain.fleet import LOCAL
from engine_console.domain.learn import LearnBundle
from engine_console.domain.models import FitReport, FitRequest, HardwareReport, HostId, Page
from engine_console.domain.repo_id import RepoId, Revision
from engine_console.learn import bundle as learn_bundle
from engine_console.services.hf import ModelDetail, SearchHit

router = APIRouter()


@router.get("/health")
async def health(c: C) -> dict[str, Any]:
    proxied = bool(c.cfg.egress_proxy)
    warnings = [] if proxied else [
        "console egress is DIRECT: Hugging Face traffic bypasses any egress proxy (only the in-app allowlist "
        "applies). Set EGRESS_PROXY to route it through your proxy."]
    if proxied and c.cfg.egress_ca_bundle is None:
        warnings.append("EGRESS_PROXY is set but EGRESS_CA_BUNDLE is not: an intercepting proxy's CA is not in the system "
                        "trust store, so Hugging Face requests will fail TLS verification. Set EGRESS_CA_BUNDLE.")
    if c.cfg.trust_loopback:
        warnings.append("trust_loopback is ON: any local process or browser-reachable loopback client is treated as admin, "
                        "and this console can now start containers on every machine in the fleet. Keep the CSRF header and "
                        "Host checks intact, and set TRUST_LOOPBACK=false (the default) on shared machines.")
    return {"status": "ok", "version": __version__, "engines": [a.id for a in c.adapters.list()],
            "docker_context": c.cfg.docker_context, "egress_mode": "proxied" if proxied else "direct",
            "warnings": warnings}


@router.get("/hardware")
async def hardware(c: C, host: HostId = LOCAL) -> HardwareReport:
    """One machine's GPUs and host RAM, from what its node last pushed (default: the console's own machine)."""
    return c.hardware.report(host)


@router.get("/learn")
async def learn(c: C) -> LearnBundle:
    """Plain-language help for every parameter, metric, state and term (learn mode), for the registered engines."""
    b = learn_bundle()
    ids = {a.id for a in c.adapters.list()}
    return b.model_copy(update={"params": {e: p for e, p in b.params.items() if e in ids},
                                "presets": {e: p for e, p in b.presets.items() if e in ids}})


@router.get("/engines")
async def engines(c: C) -> list[dict[str, Any]]:
    b = learn_bundle()
    return [{"id": a.id, "display_name": a.display_name, "default_image": c.settings.image_pin(a.id) or a.default_image,
             "presets": [{"name": n, "params": p} for n, p in a.presets().items()],
             "model_formats": sorted(a.model_formats), "images_by_arch": dict(a.images_by_arch),
             "multi_model": a.multi_model,
             "presets_info": {n: e.model_dump() for n, e in (b.presets.get(a.id) or a.preset_info()).items()}}
            for a in c.adapters.list()]


@router.get("/engines/{engine}/params")
async def engine_params(engine: str, c: C) -> list[ParamSpec]:
    """The parameter catalogue with each parameter's plain-language `explain` and its `level` filled in."""
    explained = learn_bundle().params.get(engine, {})
    return [p.model_copy(update={"explain": explained.get(p.key), "level": p.level or ("advanced" if p.advanced else "basic")})
            for p in c.adapters.get(engine).param_catalog()]


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
