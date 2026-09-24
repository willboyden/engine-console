"""FastAPI application factory + `engine-console` entry point (127.0.0.1:8791 by default)."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse, Response

from engine_console import __version__
from engine_console.api.deps import enforce_roles
from engine_console.api.errors import install_error_handlers, problem
from engine_console.api.routers import admin, chat, downloads, instances, observe, profiles, system
from engine_console.api.security import SecurityMiddleware
from engine_console.config import Settings
from engine_console.container import Container, build_container
from engine_console.services.telemetry import configure_logging, setup_tracing

log = logging.getLogger("engine_console")
API = "/api/v1"
STATIC_DIRS = ("css", "js", "i18n")


def default_frontend_dir() -> Path:
    # src/engine_console/main.py -> clients/engine-console/frontend
    return Path(__file__).resolve().parents[3] / "frontend"


def create_app(cfg: Settings | None = None, container: Container | None = None) -> FastAPI:
    cfg = cfg or Settings()
    c = container or build_container(cfg)
    provider = setup_tracing(cfg)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        key_file = c.settings.ensure_bootstrap_key(cfg.data_dir)
        if key_file:
            log.info("first start: admin API key written to %s (0600)", key_file)   # path only, never the key
        existing = cfg.data_dir / "bootstrap-admin.key"
        if existing.exists():
            log.warning("bootstrap admin key file %s exists: copy the key somewhere safe (or mint a new one via "
                        "POST /keys) and DELETE this file after first use", existing)
        c.downloads.recover()
        with contextlib.suppress(Exception):
            await c.discovery.refresh()   # prefill Instances/Metrics with already-running external engines
        tasks: list[asyncio.Task[None]] = []
        if cfg.background_tasks:
            with contextlib.suppress(Exception):
                await c.lifecycle.adopt()
            tasks.append(asyncio.create_task(_loop(c.lifecycle.tick, cfg.supervisor_interval_s), name="supervisor"))
            tasks.append(asyncio.create_task(c.metrics.run(), name="metrics-scraper"))
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await c.downloads.shutdown()
            await c.bench.shutdown()
            await c.http.aclose()
            if provider is not None:
                provider.shutdown()

    # No interactive docs page: Swagger UI loads scripts from a CDN, which our CSP (and egress policy) forbids.
    # The OpenAPI JSON stays, behind auth.
    app = FastAPI(title="Engine Console", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=f"{API}/openapi.json")
    app.state.container = c
    install_error_handlers(app)
    for mod in (system, downloads, instances, profiles, observe, chat, admin):
        app.include_router(mod.router, prefix=API, dependencies=[Depends(enforce_roles)])
    app.include_router(observe.root_router)
    app.add_middleware(SecurityMiddleware, container=c)
    _mount_frontend(app, cfg.frontend_dir or default_frontend_dir())
    return app


async def _loop(fn, interval: float) -> None:  # type: ignore[no-untyped-def]
    while True:
        try:
            await fn()
        except Exception:  # noqa: BLE001
            log.exception("background loop error")
        await asyncio.sleep(interval)


def _mount_frontend(app: FastAPI, root: Path) -> None:
    """Static files at `/` with SPA fallback to index.html (hash router, but deep links still work)."""
    if not root.is_dir():
        return
    root = root.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> Response:
        # Serve ONLY the app shell and its asset dirs; dev/, tests/, package.json etc. are not the browser's business.
        if path in ("", "index.html"):
            index = root / "index.html"
            if index.is_file():
                return FileResponse(index)
            return problem(404, "not_found", "not found", "frontend not built")
        top = path.split("/", 1)[0]
        if top in STATIC_DIRS and "/" in path:
            base = (root / top).resolve()
            target = (root / path).resolve()
            if target.is_file() and base in target.parents:   # inside the asset dir: defeats ../ and symlink escapes
                return FileResponse(target)
        return problem(404, "not_found", "not found", "no such file")


def run() -> None:
    cfg = Settings()
    configure_logging()
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_config=None)


if __name__ == "__main__":
    run()
