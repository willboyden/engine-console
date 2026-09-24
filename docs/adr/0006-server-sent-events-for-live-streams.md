# ADR-0006: Server-Sent Events for live streams

Status: accepted (contract v1, see `docs/ARCHITECTURE.md` section 1); amended 2026-09-24

## Context

Logs, download progress, metrics and benchmark progress are all one-way server-to-browser.

## Decision

SSE endpoints (`/downloads/stream`, `/instances/{id}/logs/stream`, `/metrics/stream`, `/bench/{id}/stream`) fed by an in-process `EventBus`. Every SSE endpoint accepts `?once=true` for tests. The security middleware is pure ASGI so streams are never buffered.

## Consequences

- Plain HTTP, works through proxies, auto-reconnect in `EventSource`-style clients.
- Cost: one-way only; browsers cap concurrent HTTP/1.1 connections per origin (about six), so many open streams compete; the in-process bus does not span multiple processes.
- `EventSource` cannot set an Authorization header, so remote browsers rely on a fetch-based reader (see `frontend/js/sse.js`).

## Alternatives considered

- WebSockets: rejected, bidirectional state not needed.
- Polling: rejected for logs and progress, wasteful and laggy.

## Amendment (2026-09-24)

Streams are now capped: at most 8 concurrent SSE connections (`MAX_SSE_CONNECTIONS`) and a 300 s idle timeout (`SSE_IDLE_TIMEOUT_S`), which bounds the connection-flood risk named above. Not load-tested.
