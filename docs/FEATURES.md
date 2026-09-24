# Feature matrix

Compared against **oMLX** and **Open WebUI** as described in the project brief (their public feature
lists as summarised there; not re-checked against their current releases). "Implemented" means the
route/service and a UI view exist in this repository (grep of `backend/src` and `frontend/js`). It does
**not** mean tested against live engines. Per the frontend agent's report, the **Downloads, Arena, Benchmark compare, Instance detail and Chat image upload** views were exercised only against the mock server (`frontend/dev/mock-server.mjs`), never against the real backend. On the backend side only the internal network, engine isolation and gateway were live-tested (rootless Docker); GPU engine launches, real HF downloads through the proxy and mitmproxy interception were not.
Otherwise only offline unit tests exist and the author has not run the suite in this task. "Partial" states what is missing. "Not planned" means out of scope by design.

Engine Console differs in kind: it controls **vLLM and SGLang containers on NVIDIA GPUs**, whereas oMLX
serves models itself on Apple silicon and Open WebUI is a chat front end for any OpenAI-compatible endpoint.

## Versus oMLX

| oMLX capability | Engine Console | Status | Evidence / gap |
|---|---|---|---|
| Model downloader | HF search with filters, model card, resumable queued downloads, pause/resume/cancel, sha256 check, disk pre-check | Implemented (Downloads view mock-only; real HF download unverified) | `services/hf.py`, `services/downloads.py`, `views/models.js`, `views/downloads.js` |
| Per-model settings | Profiles may carry a `repo_id`, and the launch form applies profile params; no separate persistent per-model settings page that auto-applies on every launch | Partial | `services/profiles.py`, `views/launch.js` |
| Profiles | Named param bundles, engine presets, YAML import/export, diff, copy-as-command | Implemented | `services/profiles.py`, `/profiles/*` |
| Benchmark | `quick` and `standard` suites: TTFT, decode throughput at concurrency 1/4/16/64, optional prefix-cache probe, stored and comparable | Implemented (run/compare view mock-only; no real benchmark run verified) | `services/bench.py`, `views/bench.js` |
| Hot/cold KV cache (tiered) | Only reads engine KV usage and prefix-hit metrics; no cache tiering | Not planned | Engine-side feature; the console does not manage KV storage |
| Multi-model LRU / pinning / TTL | Several concurrent instances on distinct GPUs and ports, pin, idle TTL auto-stop; **no LRU eviction** or on-demand model swap | Partial | `services/lifecycle.py` (`pinned`, `ttl_idle_s`) |
| Usage heatmaps | Hourly heatmap, per-model and per-day totals, CSV export | Implemented | `services/usage.py`, `views/usage.js` |
| Chat | Streaming chat through an instance, history, prompt library, image upload, reasoning fold, export | Implemented (image upload mock-only) | `services/chat.py`, `views/chat.js` |
| API keys | Create/revoke, hashed at rest, admin and viewer roles | Implemented | `services/settings.py` |
| 8 languages | i18n mechanism (flat keys, plurals) but only `en` ships | Partial | `frontend/i18n/en.json` only |
| Vendored offline dependencies | Frontend has zero third-party JS and no CDN references; backend deps are pinned by `uv.lock` but installed from PyPI, not vendored | Partial | `frontend/` has no `node_modules` requirement; `backend/uv.lock` |

## Versus Open WebUI

| Open WebUI capability | Engine Console | Status | Evidence / gap |
|---|---|---|---|
| Admin panel | Settings (cache dir, TTL, image pins, token status), keys, audit log | Partial | `views/settings.js`; shows egress mode; cache dir is env-only and read-only in the UI; no user management beyond keys |
| Custom model presets | Engine presets and saved profiles change *engine launch* parameters; there are no chat-level "custom models" (system prompt + params bundle bound to a model name) | Partial | `adapters/*.presets`, `services/profiles.py` |
| Arena / Elo evaluation | Side-by-side matches, blind mode, votes, Elo leaderboard | Implemented (view mock-only) | `/arena/*`, `views/arena.js` |
| Prompt library | CRUD prompts, picker in chat | Implemented | `/prompts`, `views/chat.js` |
| RAG / knowledge bases | None | Not planned | Out of scope; use Open WebUI in front of the router |
| Tools / function-calling UI, MCP | None | Not planned | AGENTS.md routes tool use through Open WebUI + mcpo |
| Web search | None (and it would need new egress) | Not planned | Conflicts with deny-by-default egress |
| RBAC | Two roles (`admin`, `viewer`); no groups or per-model permissions | Partial | `api/security.py` |

## Console-specific capabilities (not in either product)

| Capability | Status | Where |
|---|---|---|
| Fit estimator with per-GPU breakdown, verdict, confidence, "fits if you stop X" | Implemented (accuracy uncalibrated) | `services/fit.py`, `docs/FIT-ESTIMATOR.md` |
| Schema-driven launch form from adapter param catalogs | Implemented | `adapters/*_params.py`, `frontend/js/param-form.js` |
| Adapters for vLLM and SGLang | Implemented | `adapters/vllm.py`, `adapters/sglang.py` |
| Preflight (fit, compat, free VRAM, port, image present) | Implemented (live-tested only for network isolation and gateway; GPU launch unverified) | `services/lifecycle.py` |
| Adopt already-running labelled containers | Implemented | `lifecycle.adopt` |
| Audit log of mutating calls (metadata only for chat content) | Implemented | `services/audit.py`, `api/security.py` |
| Engine network isolation: internal network, no published ports, per-instance gateway | Implemented; live-verified for isolation and gateway forwarding, not for GPU engines | ADR-0011, `services/docker.py` |
| Console egress through host mitmproxy, fail-closed 503, `egress_mode` shown | Implemented; default is direct; interception unverified | ADR-0012, `services/hf_http.py` |
| Host/Origin/CSRF-header checks, body and SSE caps | Implemented (unit-tested only) | `api/security.py` |
| Router access to an engine (`internal_endpoint`) | Implemented as opt-in; operator edits router networks; not exercised | OPERATIONS.md |
| Secret engine params kept out of DB and API | Implemented | `services/secrets_store.py` |
| Console self-observability (OTLP, `/metrics`, JSON logs) | Implemented | `services/telemetry.py`, `deploy/grafana-engine-console.json` |
| Other engines (llama.cpp, TensorRT-LLM, Dynamo, Ollama) | Not implemented; the port allows it | ADR-0001 |
| Browser end-to-end tests | Not implemented | only `node --test` on pure modules |
