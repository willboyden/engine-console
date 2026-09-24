"""Chat proxy (SSE passthrough with persistence + usage), conversations, prompt library, and the arena (Elo)."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from engine_console.domain.errors import BadRequest, Conflict, NotFound, UpstreamError
from engine_console.domain.models import (
    ArenaMatchIn,
    Conversation,
    ConversationIn,
    Instance,
    Page,
    Prompt,
    PromptIn,
)
from engine_console.services.common import new_id, now
from engine_console.services.lifecycle import LifecycleService
from engine_console.services.store import Store, paginate
from engine_console.services.usage import UsageService

log = logging.getLogger(__name__)
ELO_K = 32.0
ELO_START = 1000.0


@dataclass
class ChatStream:
    """An opened upstream stream. `aiter()` relays raw SSE lines and persists the result when it ends."""
    resp: httpx.Response
    on_done: Any   # callable(text, reasoning, usage, latency_ms)
    text: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)

    async def aiter(self) -> AsyncIterator[str]:
        try:
            async for line in self.resp.aiter_lines():
                self._observe(line)
                yield line + "\n"
        finally:
            await self.resp.aclose()
            self.on_done("".join(self.text), "".join(self.reasoning), self.usage,
                         (time.perf_counter() - self.started) * 1000)

    def _observe(self, line: str) -> None:
        if not line.startswith("data:") or line.strip() == "data: [DONE]":
            return
        try:
            ev = json.loads(line[5:])
        except ValueError:
            return
        if ev.get("usage"):
            self.usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                self.text.append(d["content"])
            if d.get("reasoning_content") or d.get("reasoning"):
                self.reasoning.append(str(d.get("reasoning_content") or d.get("reasoning")))


class ChatService:
    def __init__(self, store: Store, lifecycle: LifecycleService, http: httpx.AsyncClient, usage: UsageService) -> None:
        self._db = store
        self._life = lifecycle
        self._http = http
        self._usage = usage

    # -- proxy -------------------------------------------------------------------------------------
    def _ready(self, iid: str) -> Instance:
        inst = self._life.get(iid)
        if inst.state != "ready" or not inst.port:
            raise Conflict(f"instance {inst.name} is {inst.state}", code="instance_not_ready")
        return inst

    def _prepare(self, body: dict[str, Any]) -> tuple[Instance, dict[str, Any], str | None]:
        body = dict(body)
        iid = body.pop("instance_id", None)
        conv = body.pop("conversation_id", None)
        if not iid:
            raise BadRequest("instance_id is required", code="missing_instance_id")
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            raise BadRequest("messages must be a non-empty list", code="invalid_messages")
        inst = self._ready(iid)
        if conv:
            self.get_conversation(conv)
        body["model"] = inst.repo_id   # engines are launched with served_name = repo_id
        return inst, body, conv

    def _persist(self, conv: str | None, sent: list[dict[str, Any]], inst: Instance, text: str, reasoning: str,
                 usage: dict[str, Any], latency_ms: float) -> None:
        pt, ct = int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)
        self._usage.record(model=inst.repo_id, instance_id=inst.id, prompt_tokens=pt, completion_tokens=ct,
                           latency_ms=latency_ms, source="chat")
        if conv:
            last_user = next((m for m in reversed(sent) if m.get("role") == "user"), None)
            if last_user is not None:
                c = last_user.get("content")
                self._add_message(conv, "user", c if isinstance(c, str) else json.dumps(c), None, None)
            self._add_message(conv, "assistant", text, reasoning or None, usage or None)

    async def open_stream(self, body: dict[str, Any]) -> ChatStream:
        inst, payload, conv = self._prepare(body)
        payload["stream"] = True
        payload.setdefault("stream_options", {})["include_usage"] = True
        self._life.touch(inst.id)
        req = self._http.build_request("POST", f"http://127.0.0.1:{inst.port}/v1/chat/completions", json=payload,
                                       timeout=httpx.Timeout(10.0, read=None))
        try:
            resp = await self._http.send(req, stream=True)
        except httpx.HTTPError as e:
            raise UpstreamError(f"engine unreachable: {type(e).__name__}") from e
        if resp.status_code >= 400:
            detail = (await resp.aread()).decode(errors="replace")[:500]
            await resp.aclose()
            raise UpstreamError(f"engine returned HTTP {resp.status_code}: {detail}", status=resp.status_code
                                if resp.status_code in (400, 404, 422) else 502)
        return ChatStream(resp, lambda t, r, u, ms: self._persist(conv, payload["messages"], inst, t, r, u, ms))

    async def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        inst, payload, conv = self._prepare(body)
        payload["stream"] = False
        self._life.touch(inst.id)
        t0 = time.perf_counter()
        try:
            r = await self._http.post(f"http://127.0.0.1:{inst.port}/v1/chat/completions", json=payload,
                                      timeout=httpx.Timeout(10.0, read=600.0))
        except httpx.HTTPError as e:
            raise UpstreamError(f"engine unreachable: {type(e).__name__}") from e
        if r.status_code >= 400:
            raise UpstreamError(f"engine returned HTTP {r.status_code}: {r.text[:500]}")
        data: dict[str, Any] = r.json()
        msg = ((data.get("choices") or [{}])[0].get("message")) or {}
        self._persist(conv, payload["messages"], inst, msg.get("content") or "",
                      msg.get("reasoning_content") or msg.get("reasoning") or "", data.get("usage") or {},
                      (time.perf_counter() - t0) * 1000)
        return data

    # -- conversations -----------------------------------------------------------------------------
    def _add_message(self, conv: str, role: str, content: str, reasoning: str | None, usage: dict[str, Any] | None) -> None:
        self._db.execute("INSERT INTO messages(conversation_id,role,content,reasoning,usage,ts) VALUES(?,?,?,?,?,?)",
                         (conv, role, content, reasoning, json.dumps(usage) if usage else None, now()))
        self._db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now(), conv))

    def create_conversation(self, c: ConversationIn) -> Conversation:
        cid, ts = new_id("conv_"), now()
        self._db.execute("INSERT INTO conversations VALUES(?,?,?,?,?,?)",
                         (cid, c.title, c.system_prompt, c.instance_id, ts, ts))
        return self.get_conversation(cid)

    def get_conversation(self, cid: str, with_messages: bool = False) -> Conversation:
        r = self._db.one("SELECT * FROM conversations WHERE id=?", (cid,))
        if r is None:
            raise NotFound(f"no such conversation: {cid}")
        msgs = None
        if with_messages:
            msgs = [{"role": m["role"], "content": m["content"], "reasoning": m["reasoning"], "ts": m["ts"],
                     "usage": json.loads(m["usage"]) if m["usage"] else None}
                    for m in self._db.all("SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (cid,))]
        return Conversation(id=r["id"], title=r["title"], system_prompt=r["system_prompt"], instance_id=r["instance_id"],
                            created_at=r["created_at"], updated_at=r["updated_at"], messages=msgs)

    def list_conversations(self, limit: int, cursor: str | None) -> Page[Conversation]:
        limit, off = paginate(limit, cursor)
        rows = self._db.all("SELECT id FROM conversations ORDER BY updated_at DESC LIMIT ? OFFSET ?", (limit + 1, off))
        return Page[Conversation](items=[self.get_conversation(r["id"]) for r in rows[:limit]],
                                  next_cursor=str(off + limit) if len(rows) > limit else None)

    def delete_conversation(self, cid: str) -> None:
        self.get_conversation(cid)
        self._db.execute("DELETE FROM conversations WHERE id=?", (cid,))

    # -- prompts -----------------------------------------------------------------------------------
    @staticmethod
    def _prompt(r: Any) -> Prompt:
        return Prompt(id=r["id"], title=r["title"], content=r["content"], tags=json.loads(r["tags"]),
                      created_at=r["created_at"], updated_at=r["updated_at"])

    def create_prompt(self, p: PromptIn) -> Prompt:
        pid, ts = new_id("prm_"), now()
        self._db.execute("INSERT INTO prompts VALUES(?,?,?,?,?,?)", (pid, p.title, p.content, json.dumps(p.tags), ts, ts))
        return self.get_prompt(pid)

    def get_prompt(self, pid: str) -> Prompt:
        r = self._db.one("SELECT * FROM prompts WHERE id=?", (pid,))
        if r is None:
            raise NotFound(f"no such prompt: {pid}")
        return self._prompt(r)

    def list_prompts(self, limit: int, cursor: str | None) -> Page[Prompt]:
        limit, off = paginate(limit, cursor)
        rows = self._db.all("SELECT * FROM prompts ORDER BY title LIMIT ? OFFSET ?", (limit + 1, off))
        return Page[Prompt](items=[self._prompt(r) for r in rows[:limit]],
                            next_cursor=str(off + limit) if len(rows) > limit else None)

    def update_prompt(self, pid: str, p: PromptIn) -> Prompt:
        self.get_prompt(pid)
        self._db.execute("UPDATE prompts SET title=?, content=?, tags=?, updated_at=? WHERE id=?",
                         (p.title, p.content, json.dumps(p.tags), now(), pid))
        return self.get_prompt(pid)

    def delete_prompt(self, pid: str) -> None:
        self.get_prompt(pid)
        self._db.execute("DELETE FROM prompts WHERE id=?", (pid,))

    # -- arena -------------------------------------------------------------------------------------
    async def create_match(self, m: ArenaMatchIn) -> dict[str, Any]:
        if m.instance_a == m.instance_b:
            raise BadRequest("pick two different instances", code="same_instance")
        ia, ib = self._ready(m.instance_a), self._ready(m.instance_b)
        msgs: list[dict[str, Any]] = ([{"role": "system", "content": m.system_prompt}] if m.system_prompt else [])
        msgs.append({"role": "user", "content": m.prompt})

        async def ask(inst: Instance) -> str:
            data = await self.complete({"instance_id": inst.id, "messages": msgs, "max_tokens": m.max_tokens})
            return str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")

        ra, rb = await asyncio.gather(ask(ia), ask(ib))
        mid, ts = new_id("match_"), now()
        self._db.execute("INSERT INTO arena_matches(id,prompt,system_prompt,blind,model_a,model_b,response_a,response_b,created_at)"
                         " VALUES(?,?,?,?,?,?,?,?,?)",
                         (mid, m.prompt, m.system_prompt, int(m.blind), ia.repo_id, ib.repo_id, ra, rb, ts))
        return self._match(mid)

    def _match(self, mid: str) -> dict[str, Any]:
        r = self._db.one("SELECT * FROM arena_matches WHERE id=?", (mid,))
        if r is None:
            raise NotFound(f"no such match: {mid}")
        reveal = not r["blind"] or r["winner"] is not None
        return {"id": r["id"], "prompt": r["prompt"], "blind": bool(r["blind"]), "response_a": r["response_a"],
                "response_b": r["response_b"], "winner": r["winner"],
                "model_a": r["model_a"] if reveal else None, "model_b": r["model_b"] if reveal else None}

    def vote(self, mid: str, winner: str) -> dict[str, Any]:
        r = self._db.one("SELECT * FROM arena_matches WHERE id=?", (mid,))
        if r is None:
            raise NotFound(f"no such match: {mid}")
        if r["winner"] is not None:
            raise Conflict("this match already has a vote", code="already_voted")
        a, b = r["model_a"], r["model_b"]
        self._db.execute("UPDATE arena_matches SET winner=?, voted_at=? WHERE id=?", (winner, now(), mid))
        if a == b:   # two instances of one model: a vote says nothing about relative strength
            return {**self._match(mid), "ratings": {a: round(self._rating(a), 1)}}
        ra, rb = self._rating(a), self._rating(b)
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        sa = {"a": 1.0, "b": 0.0, "tie": 0.5}[winner]
        na, nb = ra + ELO_K * (sa - ea), rb + ELO_K * ((1 - sa) - (1 - ea))
        for model, new, score in ((a, na, sa), (b, nb, 1 - sa)):
            self._db.execute(
                "INSERT INTO arena_ratings(model,rating,games,wins,losses,ties) VALUES(?,?,1,?,?,?) ON CONFLICT(model) DO UPDATE SET "
                "rating=excluded.rating, games=games+1, wins=wins+excluded.wins, losses=losses+excluded.losses, ties=ties+excluded.ties",
                (model, new, int(score == 1.0), int(score == 0.0), int(score == 0.5)))
        return {**self._match(mid), "ratings": {a: round(na, 1), b: round(nb, 1)}}

    def _rating(self, model: str) -> float:
        r = self._db.one("SELECT rating FROM arena_ratings WHERE model=?", (model,))
        return float(r["rating"]) if r else ELO_START

    def leaderboard(self) -> list[dict[str, Any]]:
        return [{"model": r["model"], "rating": round(r["rating"], 1), "games": r["games"], "wins": r["wins"],
                 "losses": r["losses"], "ties": r["ties"]}
                for r in self._db.all("SELECT * FROM arena_ratings ORDER BY rating DESC")]
