# ADR-0012: Console egress fails closed through the host proxy

Status: accepted 2026-09-24. **Not verified:** real mitmproxy interception with the CA, and real Hugging Face
downloads through the proxy. The unit tests use fakes.

## Context

ADR-0008 relied on an in-app host allowlist. That protects only against the console's own mistakes and redirects;
the repo already has a host-level chokepoint (`security/egress/mitmproxy`), which logs every request and applies
its own allowlist.

## Decision

- `EGRESS_PROXY` (e.g. `http://127.0.0.1:8082`) routes all Hugging Face traffic through the host mitmproxy.
  `EGRESS_CA_BUNDLE` (default: the repo's `security/egress/mitmproxy/ca/mitmproxy-ca-cert.pem` if present) is the
  certificate only, never the CA key; the client is built with `trust_env=False` so `NO_PROXY` and friends cannot bypass it.
- **Fail closed.** If the proxy is configured but the CA is missing/unloadable, or the proxy is unreachable, the request
  is refused with HTTP 503 `egress_proxy_unavailable`; there is no direct fallback. With `REQUIRE_EGRESS_PROXY=true`
  and no `EGRESS_PROXY`, all HF traffic is refused. A proxy 403 "egress blocked" maps to an egress-denied error.
- The in-app allowlist stays as a second, independent layer and also checks every redirect hop.
- `/api/v1/health` and settings report `egress_mode: direct|proxied`; `direct` adds a warning to health.

## Consequences

- Default is still **direct** (`REQUIRE_EGRESS_PROXY=false`), so out of the box the mitmproxy is not enforced;
  operators must set both variables to get the fail-closed behaviour.
- The proxy must be running before downloads/search work; if it is down, the console is degraded for HF, not open.
- The proxy sees plaintext of HF traffic including the bearer token: the proxy's flow log (`flows.mitm`) is sensitive.
- Cost: TLS interception adds a CA trust step for this one client, and the allowlist now exists in two places
  (`HF_ALLOWED_HOSTS` and `allowlist.txt`) that must be kept in step.
- Unchanged: OTLP to `localhost:4317` is not proxied.

## Alternatives considered

- nftables per-uid egress rules: stronger against a compromised console, but host-level and opt-in per AGENTS.md rule 4; still recommended in addition.
- Fall back to direct when the proxy is down: rejected, that is fail-open.
- Trust `HTTPS_PROXY` from the environment: rejected, ambient env is easy to bypass or mis-set.
