"""One pure-ASGI middleware for: Host allowlist (DNS rebinding), CSRF defences, authentication (ADR 9),
RBAC-lite, request body caps, audit of every mutating call, Prometheus metrics and a tracing span.
Pure ASGI so streaming (SSE) bodies are never buffered."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from engine_console.api.errors import problem
from engine_console.container import Container
from engine_console.services.common import redact
from engine_console.services.settings import Principal

log = logging.getLogger(__name__)
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
OPEN_PATHS = {"/api/v1/health"}   # everything else, including the OpenAPI JSON, needs auth
SAFE = {"GET", "HEAD", "OPTIONS"}
MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
# The ONLY mutating (method, path) pairs a `viewer` may perform: chat, arena, starting a conversation, and the
# read-only fit estimate. Matched exactly (not by prefix), so e.g. DELETE /conversations/{id} needs admin.
VIEWER_ALLOWED = re.compile(r"/api/v1/(chat/completions|fit|conversations|arena/matches|arena/matches/[^/]+/vote)")
KNOWN_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; font-src 'self'; "
       "connect-src 'self'; base-uri 'none'; form-action 'none'; object-src 'none'; frame-ancestors 'none'")
SECURITY_HEADERS = [
    (b"content-security-policy", CSP.encode()),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-frame-options", b"DENY"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
]


def viewer_may(method: str, path: str) -> bool:
    return method == "POST" and VIEWER_ALLOWED.fullmatch(path) is not None
# audit records metadata only for these: prompts and conversations are private content
CONTENT_PATHS = ("/api/v1/chat/completions", "/api/v1/arena", "/api/v1/conversations", "/api/v1/prompts")
CHAT_PATH = "/api/v1/chat/completions"
IMPORT_PATH = "/api/v1/profiles/import"
IMPORT_TYPES = ("application/json", "application/yaml", "application/x-yaml", "text/yaml", "text/x-yaml")
CSRF_HEADER = "x-engine-console"
BODY_CAP_AUDIT = 64 * 1024


class BodyTooLarge(HTTPException):
    """An HTTPException so FastAPI's body parsing re-raises it as-is instead of turning it into a 400."""

    def __init__(self) -> None:
        super().__init__(413, "request body too large")


def _guarded(path: str) -> bool:
    return (path.startswith("/api/") or path == "/metrics") and path not in OPEN_PATHS


class SecurityMiddleware:
    def __init__(self, app: ASGIApp, container: Container) -> None:
        self.app = app
        self.c = container
        cfg = container.cfg
        hosts = {f"127.0.0.1:{cfg.port}", f"localhost:{cfg.port}", f"[::1]:{cfg.port}", f"{cfg.host}:{cfg.port}"}
        self.hosts = {h.lower() for h in hosts} | {h.lower() for h in cfg.allowed_hosts}

    def _principal(self, scope: Scope, headers: dict[str, str]) -> Principal | None:
        auth = headers.get("authorization", "")
        if auth:
            scheme, _, tok = auth.partition(" ")
            return self.c.settings.authenticate(tok.strip()) if scheme.lower() == "bearer" and tok.strip() else None
        client = scope.get("client")
        host = client[0] if client else ""
        if self.c.cfg.trust_loopback and host in LOOPBACK and "x-forwarded-for" not in headers:
            return Principal("loopback", "admin")
        return None

    def _cap(self, path: str) -> int:
        cfg = self.c.cfg
        if path == CHAT_PATH:
            return cfg.chat_body_cap_bytes
        return cfg.import_body_cap_bytes if path == IMPORT_PATH else cfg.body_cap_bytes

    def _precheck(self, scope: Scope, headers: dict[str, str]) -> tuple[int, str, str] | None:
        """Host / CSRF / size gates that apply before authentication. Returns (status, code, detail)."""
        path, method = scope["path"], scope["method"]
        if headers.get("host", "").lower() not in self.hosts:
            return 421, "misdirected_request", "unrecognised Host header"
        if method not in SAFE:
            origin = headers.get("origin")
            if origin is not None and (origin == "null" or urlsplit(origin).netloc.lower() != headers.get("host", "").lower()):
                return 403, "cross_origin", "cross-origin requests are not allowed"
            # A custom header cannot be sent cross-site without a CORS preflight (which we never grant).
            # Bearer-key clients are not ambient-credential browsers, so they are exempt.
            if "authorization" not in headers and headers.get(CSRF_HEADER) != "1":
                return 403, "csrf_header_required", "send the X-Engine-Console: 1 header on non-GET requests"
            if path == IMPORT_PATH and headers.get("content-type", "").split(";")[0].strip().lower() not in IMPORT_TYPES:
                return 415, "unsupported_media_type", "profile import needs a JSON or YAML content-type"
            cl = headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > self._cap(path):
                return 413, "payload_too_large", "request body too large"
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path, method = scope["path"], scope["method"]
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        principal = Principal("public", "public")
        deny = self._precheck(scope, headers)
        if deny is None and _guarded(path):
            p = self._principal(scope, headers)
            if p is None:
                deny = (401, "unauthorized", "a valid bearer API key is required")
            elif p.role != "admin" and method in MUTATING and not viewer_may(method, path):
                deny = (403, "forbidden", "this key has the viewer role (read-only)")
                principal = p
            else:
                principal = p
        scope.setdefault("state", {})["principal"] = principal

        status = 500
        body_chunks: list[bytes] = []
        captured = 0
        total = 0
        cap = self._cap(path)
        started = False

        async def recv() -> Message:
            nonlocal captured, total
            msg = await receive()
            if msg["type"] == "http.request":
                chunk = msg.get("body", b"")
                total += len(chunk)
                if total > cap:
                    raise BodyTooLarge
                if captured < BODY_CAP_AUDIT:
                    part = chunk[: BODY_CAP_AUDIT - captured]
                    captured += len(part)
                    body_chunks.append(part)
            return msg

        async def snd(message: Message) -> None:
            nonlocal status, started
            if message["type"] == "http.response.start":
                status, started = message["status"], True
                names = {n for n, _ in SECURITY_HEADERS} | ({b"cache-control"} if path.startswith("/api/") else set())
                hdrs = [(k, v) for k, v in message.get("headers", []) if k.lower() not in names]
                hdrs += SECURITY_HEADERS
                if path.startswith("/api/"):
                    hdrs.append((b"cache-control", b"no-store"))
                message = {**message, "headers": hdrs}
            await send(message)

        t0 = time.perf_counter()
        span = self.c.telemetry.tracer.start_span(f"HTTP {method if method in KNOWN_METHODS else 'OTHER'}")
        try:
            if deny is not None:
                await problem(deny[0], deny[1], deny[1].replace("_", " "), deny[2])(scope, recv, snd)
            else:
                await self.app(scope, recv, snd)
        except BodyTooLarge:
            if not started:
                status = 413
                await problem(413, "payload_too_large", "payload too large", "request body too large")(scope, receive, send)
        finally:
            route = getattr(scope.get("route"), "path", None) or ("unmatched" if _guarded(path) else "static")
            dur = time.perf_counter() - t0
            mlabel = method if method in KNOWN_METHODS else "OTHER"   # bounded label cardinality
            self.c.telemetry.requests.labels(mlabel, route, str(status)).inc()
            self.c.telemetry.latency.labels(mlabel, route).observe(dur)
            span.set_attribute("http.method", mlabel)
            span.set_attribute("http.route", route)
            span.set_attribute("http.status_code", status)
            span.end()
            if _guarded(path) and method in MUTATING:
                self._audit(principal, method, path, status, scope, headers, b"".join(body_chunks), total)

    def _audit(self, principal: Principal, method: str, path: str, status: int, scope: Scope, headers: dict[str, str],
               raw: bytes, size: int) -> None:
        params: dict[str, Any] = {}
        qs = scope.get("query_string", b"").decode("latin-1")
        if qs:
            params["query"] = redact(dict(parse_qsl(qs, keep_blank_values=True)))
        if path.startswith(CONTENT_PATHS):
            # metadata only: never store prompt/chat text
            meta: dict[str, Any] = {"bytes": size}
            try:
                doc = json.loads(raw) if raw else None
                if isinstance(doc, dict):
                    meta["fields"] = sorted(doc)[:30]
                    if isinstance(doc.get("messages"), list):
                        meta["messages"] = len(doc["messages"])
            except ValueError:
                pass
            params["body"] = meta
        elif raw and "json" in headers.get("content-type", ""):
            try:
                params["body"] = redact(json.loads(raw))
            except ValueError:
                params["body"] = "[unparseable json]"
        elif raw:
            params["body"] = f"[{size} bytes, non-json]"
        try:
            self.c.audit.record(actor=principal.key_id or principal.name, role=principal.role, method=method,
                                path=path, status=status, params=params)
        except Exception:  # noqa: BLE001 - never fail a request because audit storage hiccuped
            log.error("audit write failed")
