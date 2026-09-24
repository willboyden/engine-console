# ADR-0002: Host-run FastAPI process bound to 127.0.0.1:8791

Status: accepted (contract v1, see `docs/ARCHITECTURE.md` section 1); amended 2026-09-24 after the hardening pass

## Context

Starting and stopping engine containers requires the Docker socket. The target deployment uses rootless Docker; the socket belongs to the user.

## Decision

Run the console as a host process (systemd user unit), listening on loopback only (`Settings.host = 127.0.0.1`, port 8791). It uses the user's own rootless socket through the `rootless` docker context. It is not containerised.

## Consequences

- No container ever holds the socket. Mounting it into a network-facing container would give that service the user's full container-control reach.
- Loopback bind means remote access needs an explicit reverse proxy plus API keys (see ADR-0009).
- Cost: host Python environment to maintain (uv, Python 3.13); no image-level isolation of the console itself. Mitigated only partly by the systemd hardening in `deploy/engine-console.service`.
- A compromise of the console process is a compromise of the user's container control. This is the accepted risk; see SECURITY.md.

## Alternatives considered

- Containerised console with the socket mounted: rejected (root-equivalent reach behind a web port).
- Docker socket proxy container filtering API calls: considered viable and stricter; not built, extra moving part for a single-user box.
- Talk to the engines over systemd units instead of Docker: rejected, engines are commonly compose-based containers.

## Amendment (2026-09-24)

The console process is still the one place holding Docker reach, so the API in front of it was hardened:
a Host-header allowlist (421 on mismatch, DNS-rebinding guard), a same-origin `Origin` check on non-GET
requests (403 `cross_origin`), a required `X-Engine-Console: 1` header on non-GET requests that carry no
bearer key, request-body caps (1 MiB default, 32 MiB chat, 256 KiB profile import) and an SSE connection cap
(8) with an idle timeout. This closes the browser-driven gap the original decision left open. It does not
change the core trade-off: a compromised console process is still the user's container control. Whether the
checks hold against a real browser or rebinding attempt has not been exercised (unit tests only).
