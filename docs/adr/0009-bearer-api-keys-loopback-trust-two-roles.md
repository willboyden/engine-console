# ADR-0009: Bearer API keys, loopback trust, two roles

Status: accepted (contract v1, see `docs/ARCHITECTURE.md` section 1); amended 2026-09-24 after the hardening pass

## Context

oMLX and Open WebUI both gate the UI/API. The console is loopback-bound by default but may be proxied.

## Decision

Bearer keys are generated at first start (admin key written to `<data_dir>/bootstrap-admin.key`), stored hashed. Roles: `admin` (everything) and `viewer` (read-only, plus POST to chat, arena, conversations and fit). Loopback clients are trusted as admin when `trust_loopback` is true, unless an `X-Forwarded-For` header is present. Open paths: `/api/v1/health`, `/api/docs`, the OpenAPI JSON. `/metrics` is guarded.

## Consequences

- Zero-friction local use, safe-ish behind a proxy.
- Cost: any local process (any user on the host, any browser page that can reach 127.0.0.1) is admin. There is no CSRF/Origin check in the middleware (unverified: none found), so loopback trust is a weaker guarantee than it looks.
- Trust hinges on the proxy setting `X-Forwarded-For`; a proxy that does not is a full bypass.
- Only two roles: not full RBAC.

## Alternatives considered

- Always require a key: stricter, rejected for local ergonomics; can be had by `TRUST_LOOPBACK=false`.
- OIDC: rejected for a single-user box.

## Amendment (2026-09-24)

Loopback trust is now bounded by the Host allowlist, the Origin check and the CSRF header (ADR-0002 amendment).
A web page in the operator's browser can no longer make a plain cross-site POST as loopback admin, and a DNS-rebound
hostname is refused with 421. Any local *process* (not a browser) can still call the API as admin, since it can
set the header itself: `TRUST_LOOPBACK=false` remains the answer on a shared host. Audit now records the key id
rather than a label, and stores metadata only (byte count, field names, message count) for chat, arena,
conversation and prompt bodies. Bearer-key clients are exempt from the CSRF header by design.
