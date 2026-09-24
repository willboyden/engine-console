# ADR-0008: Deny-by-default egress with an in-app allowlist

Status: accepted (contract v1, see `docs/ARCHITECTURE.md` section 1); amended 2026-09-24 after the hardening pass

## Context

Project principle: deny by default. The console needs Hugging Face and nothing else.

## Decision

`HfHttpClient` rejects any request whose host is not in `hf_allowed_hosts` (`huggingface.co`, `cdn-lfs*.huggingface.co`, `cas-bridge.xethub.hf.co`, `*.hf.co`). The check runs as a request hook, so it also applies to every redirect hop. Engines default to `HF_HUB_OFFLINE=1`. Network-level enforcement is left to the operator (an egress proxy or firewall rules).

## Consequences

- Redirect to an unlisted CDN fails loudly instead of leaking a token.
- Cost: the in-app allowlist protects only the HF client. The separate `httpx.AsyncClient` used for engine traffic and the OTLP exporter are not covered by it, so it is defence in depth, not a sandbox. A compromised process can open any socket; only network-layer rules stop that.
- HF hostnames change (Xet CDN); a new host needs an allowlist edit.

## Alternatives considered

- Trust the OS firewall alone: rejected as sole control, but recommended in addition.
- Proxy all egress via an allowlisting proxy: supported and later adopted (ADR-0012), not required by default.

## Amendment (2026-09-24): superseded in part by ADR-0012

The in-app allowlist remains as a second layer, but console egress is now routed through an allowlisting HTTP(S) proxy
when `EGRESS_PROXY` is set, and can fail closed (`REQUIRE_EGRESS_PROXY`). Engines have no route out at all
(ADR-0011), which is stronger than the previous `HF_HUB_OFFLINE=1` convention. The residual gap named above
(engine-traffic client and OTLP exporter not host-restricted) is smaller: engine traffic is loopback to gateway
ports only; OTLP still targets `localhost:4317` directly.
