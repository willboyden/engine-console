# ADR-0010: Console observability: OTLP, Prometheus, JSON logs

Status: accepted (contract v1, see `docs/ARCHITECTURE.md` section 1)

## Context

Project principle: observable from the first request, with OTLP to a local collector at `localhost:4317`.

## Decision

`Telemetry` exposes `engine_console_http_requests_total`, `engine_console_http_request_seconds`, `engine_console_bench_runs_total` and `engine_console_instances{state}` at `/metrics`. Spans (`service.name=engine-console`) go to `localhost:4317` over gRPC without TLS (`insecure=True`); export failures are dropped. Logs are JSON on stderr. Engine metrics are scraped every 2 s into a 15-minute ring plus SQLite rollups.

## Consequences

- Fits the existing Grafana/OTel stack (`deploy/grafana-engine-console.json`).
- Cost: request spans are minimal (method, route, status); no child spans for docker or HF calls. OTLP is plaintext, acceptable only to a loopback collector. `/metrics` needs a bearer key from a non-loopback scraper.
- Per-route labels use the matched route template, keeping cardinality bounded; unmatched paths collapse to `unmatched`.

## Alternatives considered

- Push metrics via OTLP only: rejected, Prometheus scrape is the repo norm.
- Reuse an LLM gateway's metrics: not applicable, the console is not on the LLM request path.
