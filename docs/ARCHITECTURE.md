# Engine Console — architecture & contract

A single-host control plane and observability UI for **vLLM** and **SGLang**, in the spirit of
oMLX's admin dashboard (model downloader, per-model settings, profiles, benchmark, live monitoring,
built-in chat) and Open WebUI's admin panel (model management, chat, arena/evaluation, prompts,
usage analytics, API keys, RBAC-lite). One engine-agnostic core, one adapter per engine.

Status: contract v1. Contributors implementing a slice must not change this contract silently; propose changes in a pull request.

## 1. Decisions (ADR summary — full text in `docs/adr/`)

| # | Decision | Why |
|---|---|---|
| 1 | **Hexagonal core + `EngineAdapter` port** (`adapters/base.py`) | New engine = one adapter; UI forms are generated from `param_catalog()`. |
| 2 | **Host-run FastAPI process, bound to 127.0.0.1:8791**, not a container | Controlling engines needs the rootless Docker socket; mounting it into a container would hand a network-facing service root-equivalent reach. Host process uses the user's own rootless socket. |
| 3 | **Engines run as sibling containers** started via the `docker` CLI (argv lists, never a shell), carrying an ownership label, on an internal network (ADR-0011), pinned to GPUs by **UUID** | The console only touches containers it labelled. |
| 4 | **Zero-build, zero-npm frontend** (ES modules + web components, vendored, no CDN) | No supply-chain surface, offline-capable (as oMLX vendors deps), trivially auditable. |
| 5 | **SQLite (WAL) for state** (profiles, downloads, benchmarks, chat, usage, audit) | Single node; zero ops; migrations via ordered SQL files. |
| 6 | **SSE for live streams** (logs, download progress, metrics, benchmark) | One-way, proxy-friendly, no WebSocket state. |
| 7 | **Secrets**: HF token read from env / a 0600 file, never logged, never returned by the API (`[set, N chars]` only) | Principle: secrets never touch logs, URLs or API output. |
| 8 | **Deny-by-default egress**: the only outbound hosts are `huggingface.co`, `cdn-lfs*.huggingface.co`, `cas-bridge.xethub.hf.co`, `*.hf.co`; enforced by an in-app allowlist on the HF client, plus an optional allowlisting HTTP(S) proxy (ADR-0012) | Principle: deny by default. |
| 9 | **Auth**: bearer API key (generated at first start, stored hashed) required for anything non-loopback; roles `admin` / `viewer` | oMLX + Open WebUI parity. |
| 10 | **Observability**: OTLP traces to `localhost:4317`, Prometheus `/metrics` for the console itself, structured JSON logs | Principle: observable from the first request. |

## 2. Layout

```
clients/engine-console/
  README.md, Makefile, docs/
  deploy/                               (opt-in templates: systemd user unit, Grafana dashboard, example egress proxy addon)
  backend/pyproject.toml
  backend/src/engine_console/
    adapters/base.py                    (LOCKED — the port)
    adapters/__init__.py                (registry)
    adapters/vllm.py
    adapters/sglang.py
    domain/  services/  api/  main.py
  backend/tests/                        tests/core/ · tests/vllm/ · tests/sglang/
  frontend/
```

## 3. Backend services

* `hardware` — NVML (`nvidia-ml-py`): per-GPU UUID/name/total/free/util/temp/power/fan; CC; host RAM. Falls back to `nvidia-smi --query-gpu` CSV.
* `hf` — HF Hub client via `huggingface_hub` + httpx allowlist: search (filters: task, library, quant tag, size, gated), model card (README render as text), file listing with sizes, `ModelInfo` builder from `config.json` + safetensors metadata (parameters/dtype breakdown) + `quantization_config`.
* `fit` — **fit estimator** (pure function, unit-tested). Inputs: `ModelInfo`, params, `Hardware`, adapter `memory_model()`. Output `FitReport`:
  `verdict: fits|tight|wont_fit|unknown`, `confidence: high|medium|low`, per-GPU breakdown in GiB
  (`weights`, `kv_cache`, `activations`, `cuda_graphs`, `overhead`, `total`, `budget = total*mem_fraction`), `max_context_at_current_concurrency`,
  `max_concurrency_at_current_context`, `tp_required`, `notes[]`, plus adapter `Compat[]`. KV formula: `2 * layers * kv_heads * head_dim * kv_bytes * tokens`; handle GQA/MQA, MLA (`kv_lora_rank`), sliding-window layers, FP8 KV, and degrade to `confidence=low` (never fabricate) when fields are missing. Verdict thresholds: `fits` ≤ 90 % of budget, `tight` 90–100 %, else `wont_fit`.
  It also reads **currently free** VRAM (other engines may be resident) and reports "fits if you stop X".
* `downloads` — resumable HF snapshot download into `HF_CACHE_DIR` (default `~/.cache/huggingface`), file-level progress/speed/ETA, pause/resume/cancel, disk-space pre-check, queue with concurrency 2, hash verification, SSE progress. Gated-model detection with clear "needs HF_TOKEN / accept license" message.
* `discovery` — (ADR-0013, `services/discovery.py`) read-only detection of engines started outside the console: running containers plus `127.0.0.1` port probes, listed as monitor-only instances (`managed=false`).
* `lifecycle` — instance state machine `stopped → starting → loading → ready → stopping | failed`. Start/stop/restart/remove containers via `docker` CLI; multi-instance (each on distinct GPU set + host port from a range `18000–18099`, bound 127.0.0.1); **preflight** (fit + compat + free VRAM + port + image present); log tail + SSE with adapter `parse_startup_log` progress; readiness probe; crash detection with last-200-lines capture; TTL auto-stop on idle; pin; adopt-existing (detect already-running labelled containers on start-up).
* `profiles` — named parameter bundles per model (oMLX "profiles"), presets from adapter, import/export YAML, diff between two profiles, generated equivalent `docker run` / `docker compose` snippet + raw engine CLI ("copy as command").
* `metrics` — scrape each ready instance's Prometheus endpoint every 2 s → ring buffer (15 min) + SQLite rollups (1 h/1 d); canonical keys below; SSE stream.
* `bench` — one-click benchmark against an instance: prefill tok/s, decode tok/s, TTFT p50/p95, ITL, e2e latency, throughput at concurrency {1,4,16,64}, optional prefix-cache-hit test; results stored, comparable across profiles.
* `chat` — thin authenticated proxy to instance `/v1/chat/completions` (stream), conversation persistence, system prompts/prompt library, multi-model side-by-side **arena** with blind mode + Elo, image upload for VLMs, reasoning-content rendering, per-request usage recorded.
* `usage` — per-model/day tokens, requests, latency, hourly heatmap; CSV export.
* `audit` — append-only record of every mutating API call (who, what, params redacted).
* `settings` — global settings (HF cache dir, default GPUs, idle TTL, image pins, token status), API keys (create/revoke/hash), roles.

## 4. HTTP API (all JSON unless noted; prefix `/api/v1`; OpenAPI at `/api/v1/openapi.json`, docs at `/api/docs`)

```
GET  /health                                     liveness
GET  /hardware                                   Hardware + live GPU stats
GET  /engines                                    [{id, display_name, default_image, presets[]}]
GET  /engines/{engine}/params                    ParamSpec[]  (drives the form)
GET  /hf/search?q=&task=&quant=&max_gib=&gated=&sort=&limit=
GET  /hf/models/{repo_id:path}                   ModelInfo + card text + files[]
POST /fit                                        {engine, repo_id, revision?, params, gpu_ids} -> FitReport
GET  /downloads                                  list          POST /downloads {repo_id, revision?, allow_patterns?}
POST /downloads/{id}/(pause|resume|cancel)      DELETE /downloads/{id}
GET  /downloads/stream                           SSE
GET  /models                                     locally cached models (size, last used, engines that fit)
DELETE /models/{repo_id:path}                    remove from cache
GET  /instances                                  list        POST /instances {engine, repo_id, params, gpu_ids, profile_id?, name?}
GET  /instances/{id}                             detail (state, spec, fit, port, uptime)
POST /instances/{id}/(stop|start|restart)       DELETE /instances/{id}
GET  /instances/{id}/logs?tail=  GET /instances/{id}/logs/stream (SSE)
GET  /instances/{id}/command                     {docker_run, compose_yaml, engine_cli}
GET  /profiles  POST /profiles  PUT|DELETE /profiles/{id}  GET /profiles/{a}/diff/{b}  POST /profiles/import  GET /profiles/{id}/export
GET  /metrics/instances/{id}?window=             timeseries        GET /metrics/stream (SSE)
POST /bench {instance_id, suite}  GET /bench  GET /bench/{id}  GET /bench/{id}/stream (SSE)
POST /chat/completions (SSE passthrough)  GET|POST|DELETE /conversations  GET|POST|PUT|DELETE /prompts
POST /arena/matches  POST /arena/matches/{id}/vote  GET /arena/leaderboard
GET  /usage?group_by=model|day|hour   GET /usage/export.csv
GET  /audit    GET|PUT /settings    GET|POST|DELETE /keys
GET  /metrics  (root, not /api/v1: Prometheus for the console itself)
```
Errors: RFC 7807 `application/problem+json` with `code` field. Every list endpoint paginates (`limit`, `cursor`).

## 5. Canonical metric keys (`EngineAdapter.parse_metrics` output)

`requests_running, requests_waiting, kv_cache_usage_pct (0-100), prefix_cache_hit_pct, prompt_tokens_total,
generation_tokens_total, ttft_p50_s, ttft_p95_s, itl_p50_s, e2e_p50_s, e2e_p95_s, prompt_tps, generation_tps,
preemptions_total, spec_decode_accept_pct`. Missing keys are omitted, not zero.

## 6. Frontend — single-page app, hash-router, web components

Views: **Dashboard** (GPU cards with VRAM/util/temp/power sparklines, instance tiles with state + live tok/s + KV-cache bar, alert strip) ·
**Models** (HF search with filters, model-card drawer, *fit badge per GPU/engine*, Download button; local library with sizes) ·
**Downloads** (progress, speed, ETA, pause/resume/cancel) ·
**Launch / Instance settings** (engine switcher vLLM/SGLang; schema-driven grouped param form with search, presets, advanced toggle;
**live fit meter** — stacked bar weights/KV/activations/graphs vs budget per GPU, debounced re-estimate on every `affects_memory` change; compat warnings; "copy as command"; save as profile) ·
**Instances** (state machine timeline, startup progress, log viewer with follow/filter/download, stop/restart/pin/TTL) ·
**Metrics** (time-range charts: throughput, TTFT/ITL, queue, KV usage, prefix hit; per-instance) ·
**Benchmark** (run, compare runs, chart) · **Chat** (streaming, markdown+code, reasoning fold, image upload, params drawer, history, prompt library, export) ·
**Arena** (side-by-side, blind vote, leaderboard) · **Usage** (heatmap, per-model totals) · **Settings** (HF token status, cache dir, keys, theme, language en + i18n scaffold, audit log).
Cross-cutting: dark/light, keyboard palette (⌘/Ctrl-K), toasts, empty/error/loading states everywhere, responsive ≥ 1024, WCAG AA contrast, all copy in `i18n/en.json`.
Zero runtime dependencies. Charts are hand-rolled SVG. Pure logic modules (`fit-format`, `store`, `sse`, `router`, `param-form`) are unit-tested with `node --test`.

## 7. Quality bar

* Python 3.13 via `uv`, `ruff` + `mypy --strict` clean, `pytest` green **offline** (mock HF and docker; no network, no GPU in unit tests). Coverage ≥ 85 % on `fit`, `adapters`, `lifecycle`.
* No `shell=True`; no secret ever in logs/URLs/responses; every subprocess has a timeout.
* Comments explain *why*. Do not write "verified <date>" unless actually run (claims must be backed by a run).
* Engine image tags pinned (the adapters carry the pinned versions); param catalogs must be sourced from the **pinned version's** official docs/`--help`, with `docs_url` per param.
* Blackwell (sm_120) guidance encoded in the adapters: prefer FP8; NVFP4 for dense models only; NVFP4-MoE falls back to a slow kernel on vLLM; MXFP4-MoE only for gpt-oss.

## 8. Contract addendum v1.1

* All list endpoints return `{items: [...], next_cursor: string|null}` (not bare arrays).
* `POST /fit` accepts optional `concurrency` (default 1).
* Added: `GET /downloads/{id}`, `POST /instances/preflight`, `PATCH /instances/{id}` (pin, ttl, name), `GET /conversations/{id}`.
* Every SSE endpoint accepts `?once=true` (send first snapshot then close) — useful for tests.
* Loopback clients are trusted; first start writes an admin key to `<data_dir>/bootstrap-admin.key` (0600). Any `X-Forwarded-For` header disables loopback trust.
* Secret env vars (`HF_TOKEN`, `VLLM_API_KEY`) are passed by name only; values are redacted in every API response.
* `LaunchSpec` gained `shm_size` and `ipc_host`.
* Param `flag` values `env:NAME` (env var) and `@image` (image override) are non-CLI params; UI treats them as ordinary fields.
* Network isolation: engines run on an `--internal` network (`ENGINE_NETWORK` setting), publish no ports and are on no other network. A per-instance nginx `stream{}` gateway sidecar (`<container>-gw`, digest-pinned local image `gateway_image`, read-only, cap-drop ALL) is the only container on the default bridge and publishes `127.0.0.1:<18000-18099>:8080`. Both carry the console ownership label; the gateway also carries a role label and its instance id.
* `GET /instances/{id}` gains `container_port` and `internal_endpoint` (`http://<alias>:<port>`). Access from another container (for example an OpenAI-compatible gateway or router) is opt-in: the operator joins that container to the engine network; the console never modifies other services.
* Console egress: `EGRESS_PROXY` (any allowlisting HTTP(S) proxy, e.g. `http://127.0.0.1:8082`) plus `EGRESS_CA_BUNDLE` route all HF traffic through the chokepoint and fail closed with 503 `egress_proxy_unavailable` (never direct). `REQUIRE_EGRESS_PROXY=true` refuses HF traffic without a proxy. `/health` and `/settings` report `egress_mode` (`direct`|`proxied`); direct mode adds a warning to `/health`.
* External engine discovery (monitor-only): at start and every `discovery_interval_s` (10 s), plus `POST /instances/discover` (admin, returns `{items,next_cursor}`), the console lists engines it did not create. Sources: running containers (`docker ps` + `inspect`, skipping ones with the `engine-console=1` ownership label; classified by image/command signatures, never routers or UIs) and loopback ports from `discovery_ports` (GET `/v1/models`, `/health`, `/metrics`; 1 s connect / 2 s read, 1 MB cap, no redirects). `discovery_enabled=false` turns it off.
* `Instance` gains `managed` (false for external), `source` (`console`|`external`), `endpoint`, `served_models`, `state_reason`, `history_since`, and `container_name`/`image` may be null. `state` adds `auth_required` and `unreachable`; a vanished container shows `stopped` for one cycle. External ids are `ext-<slug>`. `params` are display-only, parsed from a redacted argv; container env values are never read or returned. `history_since` is epoch seconds (float): metrics history starts at discovery and is never back-filled.
* External instances are read-only: stop/start/restart/DELETE/PATCH/logs/command return 409 `instance_not_managed`. Chat works when `ready`. `POST /bench` on an external instance needs `confirm_external: true`, otherwise 400 `confirm_external_required`.

## 9. Implementation status

"Implemented" = code exists in the repo (grep); it does not imply live-engine
verification, which has not been done. Details in `docs/FEATURES.md`.

| Area | Status | Note |
|---|---|---|
| Adapters: vLLM, SGLang | Implemented | Registry tolerates a broken module |
| Fit estimator | Implemented | Heuristic constants uncalibrated (`docs/FIT-ESTIMATOR.md`) |
| Downloads, lifecycle, profiles, metrics, bench, chat, arena, usage, audit, settings | Implemented | All routes in section 4 present in `api/routers/` |
| Auth, roles, audit, Host/Origin/CSRF, caps | Implemented | Loopback still admin for local processes (`docs/SECURITY.md`) |
| Engine isolation (internal network, gateway) and egress proxy | Implemented; isolation live-verified, GPU engines, proxy interception and real HF downloads not | ADR-0011, ADR-0012; supersedes the published-port model in section 1 items 3 and 8 |
| Frontend views (11) | Implemented | Downloads, Arena, Bench compare, Instance detail, Chat image upload exercised only against the mock server; only `en` locale |
| External engine discovery | Implemented; live-verified 2026-09-24 against one SGLang and two vLLM containers; other states and engine classes tested with fakes only | ADR-0013, `tests/core/test_discovery.py` |
| LRU model eviction, KV tiering, RAG/tools/web search, extra locales | Not implemented | See FEATURES.md |
| systemd unit, Grafana dashboard | Templates, opt-in | `deploy/`, never applied automatically |
