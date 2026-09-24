from __future__ import annotations

import json

from fastapi import APIRouter, Request, Response

from engine_console.api.deps import C
from engine_console.domain.errors import BadRequest
from engine_console.domain.models import Page, Profile, ProfileDiff, ProfileIn

router = APIRouter()


@router.get("/profiles")
async def list_profiles(c: C, limit: int = 100, cursor: str | None = None, engine: str | None = None) -> Page[Profile]:
    return c.profiles.list_page(limit, cursor, engine)


@router.post("/profiles", status_code=201)
async def create_profile(body: ProfileIn, c: C) -> Profile:
    return c.profiles.create(body)


@router.post("/profiles/import", status_code=201)
async def import_profile(request: Request, c: C) -> Profile:
    """Body is YAML text (yaml content-type) or JSON {"yaml": "..."}; size-capped by the security middleware."""
    raw = (await request.body()).decode("utf-8", errors="replace")
    if "json" in request.headers.get("content-type", ""):
        try:
            doc = json.loads(raw)
        except ValueError:
            raise BadRequest("invalid JSON body", code="invalid_body") from None
        raw = doc.get("yaml", "") if isinstance(doc, dict) else ""
    return c.profiles.import_yaml(raw)


@router.get("/profiles/{a}/diff/{b}")
async def diff(a: str, b: str, c: C) -> ProfileDiff:
    return c.profiles.diff(a, b)


@router.get("/profiles/{pid}/export")
async def export(pid: str, c: C) -> Response:
    return Response(c.profiles.export_yaml(pid), media_type="application/yaml")


@router.put("/profiles/{pid}")
async def update_profile(pid: str, body: ProfileIn, c: C) -> Profile:
    return c.profiles.update(pid, body)


@router.delete("/profiles/{pid}", status_code=204)
async def delete_profile(pid: str, c: C) -> Response:
    c.profiles.delete(pid)
    return Response(status_code=204)
