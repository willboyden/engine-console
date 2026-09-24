from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response

from engine_console.api.deps import Admin, C
from engine_console.domain.models import KeyIn, KeyInfo, SettingsPatch

router = APIRouter()


@router.get("/settings")
async def get_settings(c: C) -> dict[str, Any]:
    return c.settings.effective()


@router.put("/settings")
async def put_settings(body: SettingsPatch, c: C) -> dict[str, Any]:
    return c.settings.update(body)


@router.get("/keys")
async def list_keys(c: C, _: Admin) -> list[KeyInfo]:
    return c.settings.list_keys()


@router.post("/keys", status_code=201)
async def create_key(body: KeyIn, c: C, _: Admin) -> KeyInfo:
    """The plaintext `secret` is returned exactly once, here."""
    return c.settings.create_key(body.name, body.role)


@router.delete("/keys/{kid}", status_code=204)
async def revoke_key(kid: str, c: C, _: Admin) -> Response:
    c.settings.revoke_key(kid)
    return Response(status_code=204)
