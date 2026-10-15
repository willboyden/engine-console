from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response

from engine_console.api.deps import C, slot_stream, take_stream_slot
from engine_console.domain.models import (
    ArenaMatchIn,
    ArenaVote,
    Conversation,
    ConversationIn,
    Page,
    Prompt,
    PromptIn,
)

router = APIRouter()


@router.post("/chat/completions", response_model=None)
async def chat_completions(body: dict[str, Any], c: C) -> Any:
    """OpenAI-shaped body plus `instance_id` (required) and optional `conversation_id`. SSE passthrough when stream=true.
    Each streamed reply holds an engine request open: at most MAX_CHAT_STREAMS at once, MAX_CHAT_STREAMS_PER_KEY of them
    for one key (429 too_many_chat_streams)."""
    if body.get("stream"):
        release = take_stream_slot(c.chat_streams, "streamed chat replies", code="too_many_chat_streams")
        try:
            stream = await c.chat.open_stream(body)
        except BaseException:
            release()   # refused or failed before streaming: the slot was never used
            raise
        return slot_stream(stream.aiter(), release, close=stream.resp.aclose)
    return await c.chat.complete(body)


@router.get("/conversations")
async def list_conversations(c: C, limit: int = 50, cursor: str | None = None) -> Page[Conversation]:
    return c.chat.list_conversations(limit, cursor)


@router.post("/conversations", status_code=201)
async def create_conversation(body: ConversationIn, c: C) -> Conversation:
    return c.chat.create_conversation(body)


@router.get("/conversations/{cid}")
async def get_conversation(cid: str, c: C) -> Conversation:
    return c.chat.get_conversation(cid, with_messages=True)


@router.delete("/conversations/{cid}", status_code=204)
async def delete_conversation(cid: str, c: C) -> Response:
    c.chat.delete_conversation(cid)
    return Response(status_code=204)


@router.get("/prompts")
async def list_prompts(c: C, limit: int = 100, cursor: str | None = None) -> Page[Prompt]:
    return c.chat.list_prompts(limit, cursor)


@router.post("/prompts", status_code=201)
async def create_prompt(body: PromptIn, c: C) -> Prompt:
    return c.chat.create_prompt(body)


@router.put("/prompts/{pid}")
async def update_prompt(pid: str, body: PromptIn, c: C) -> Prompt:
    return c.chat.update_prompt(pid, body)


@router.delete("/prompts/{pid}", status_code=204)
async def delete_prompt(pid: str, c: C) -> Response:
    c.chat.delete_prompt(pid)
    return Response(status_code=204)


@router.post("/arena/matches", status_code=201)
async def create_match(body: ArenaMatchIn, c: C) -> dict[str, Any]:
    return await c.chat.create_match(body)


@router.post("/arena/matches/{mid}/vote")
async def vote(mid: str, body: ArenaVote, c: C) -> dict[str, Any]:
    return c.chat.vote(mid, body.winner)


@router.get("/arena/leaderboard")
async def leaderboard(c: C) -> list[dict[str, Any]]:
    return c.chat.leaderboard()
