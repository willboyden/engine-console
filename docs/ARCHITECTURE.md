# Engine Console — architecture & contract

A single-host control plane and observability UI for **vLLM** and **SGLang**, in the spirit of
oMLX's admin dashboard (model downloader, per-model settings, profiles, benchmark, live monitoring,
built-in chat) and Open WebUI's admin panel (model management, chat, arena/evaluation, prompts,
usage analytics, API keys, RBAC-lite). One engine-agnostic core, one adapter per engine.

Status: contract v1 (sections 1-9) plus **contract v2, the fleet** (section 10, which wins where they differ). Contributors implementing a slice must not change this contract silently; propose changes in a pull request.

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
* Host RAM (read-only, from `/proc/meminfo` and the container's cgroup v2 `memory.current`/`memory.stat`, never RSS unless the cgroup is unreadable): `GET /hardware` gains `memory` (`total/used/available/free/cached/shmem/swap_total/swap_used` in GiB + `updated_at`) and `alerts[]` (`host_ram_low` warn <10 % / crit <5 % available, `swap_heavy` >50 % swap, `shmem_high` >25 % shared memory). `Instance.host_memory` (nullable, managed and external) is `{total_gib, anon_gib, cache_gib (= file - shmem), shmem_gib, kernel_gib, source: cgroup|rss}`.
* Metrics: per-instance keys `host_ram_gib`, `host_ram_anon_gib`, `host_ram_cache_gib`, `host_ram_shmem_gib` in the existing series; new `GET /metrics/system?window=` with `ram_used_gib`, `ram_available_gib`, `shmem_gib`, `swap_used_gib`. Console `/metrics` adds `engine_console_host_*_gib` gauges and `engine_console_instance_host_ram_gib{instance}`.
* `FitReport.host_ram` = `{needed_gib, available_gib, total_gib, verdict: ok|tight|wont_fit|unknown, breakdown, notes}` (ok <= 80 % of available, tight <= 100 %). Adapters may implement the optional `EngineAdapter.host_memory_gib()` (default `{}`; NaN = not computable, verdict `unknown`). Only unreclaimable memory counts; weight loading streams through the page cache.

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

## 10. Contract v2: the fleet (several machines, every engine, the model library, learn mode)

v2 turns the single-host console into one page that manages engines and models on every machine of a local
cluster, for every common engine, and explains every setting in plain language. Sections 1-9 still hold except
where this section replaces them. The locked code twins of this section are `node/src/ecnode/proto.py` (wire
protocol, see `docs/FLEET-PROTOCOL.md`), `backend/src/engine_console/domain/fleet.py`, `domain/refs.py`,
`domain/learn.py` and the v2 additions in `adapters/base.py` and `domain/models.py`.

### 10.1 Decisions

| # | Decision | Why |
|---|---|---|
| 14 | **Every machine is a host reached through a node.** An `ecnode --stdio` process on each machine (a subprocess for host `local`) does all docker, `/proc`, `/sys`, NVML and file work; the hub never does it itself. | One code path for local and remote; a hub bug cannot reach past what a node allows; a slow docker call never blocks the hub's event loop. |
| 15 | **Remote nodes over ssh, with the hub's own key, restricted in the remote `authorized_keys` to the node command** (`restrict,from=…,command=…`). `ssh -F none` by default (only that key is ever offered), `BatchMode`, `StrictHostKeyChecking=yes` with known_hosts read-only, no agent, no passwords, no forwarding. | Nothing new listens on the network; a stolen hub key can only start a node, from the hub's address. Proven by GPU Tuner's fleet mode on the same kind of machines. |
| 16 | **A node-local policy file is the hard boundary** (`/etc/ecnode/policy.json`, root-owned): monitor vs manage mode, allowed engine images, model roots, port range, scripts, pull/delete/external-control switches. | Whoever controls the hub controls at most what each machine's own administrator allowed, and the hub's key cannot edit that file. |
| 17 | **Engines stay isolated as in v1** (internal network, per-instance loopback gateway) **on every host**, and the hub reaches an engine's HTTP API through its node (`http.request`). | Remote engines need no network exposure; chat, benchmarks and health probes work the same everywhere. |
| 18 | **Downloads and copies stream through the hub**: HF → hub (allowlist, optional egress proxy, sha256) → node `fs.write`; machine → machine as node `fs.read` → hub → node `fs.write`. | One egress chokepoint; nodes need no internet; integrity is checked on the machine that stores the file. |
| 19 | **Three management classes** (`Instance.management`): `managed` (containers the console created: full control), `script` (an engine the operator starts with their own script, registered in that node's policy: start/stop only through those script actions), `external` (anything else: monitor-only unless the node policy allows plain stop/start). | Labs already start engines with scripts that encode ordering rules (start A before B); the console must use those scripts, never bypass them. |
| 20 | **Learn mode**: every parameter, metric, state, check and term has a plain-language explanation in `engine_console/learn/*.json`, served as one bundle; a test fails the build when an engine parameter has none. | The UI must be usable by someone who has never run an inference engine. |
| 21 | **Loopback is no longer trusted by default** (`ENGINE_CONSOLE_TRUST_LOOPBACK=false`; the bare `TRUST_LOOPBACK` name was retired in 0.4, review L5). The UI asks for an API key (its sign-in form is shaped so the browser's password manager can save it). | The console can now start containers on several machines; any local process could otherwise drive it. |

### 10.2 Layout and ownership (one owner per path during the v2 build)

```
node/                                  NEW package `ecnode` (stdlib only, Python >= 3.10)
  pyproject.toml, src/ecnode/__init__.py, py.typed     orchestrator
  src/ecnode/proto.py, tests/test_proto.py             LOCKED (orchestrator)
  src/ecnode/** (everything else), tests/** (other files), install.sh, README.md     node agent engineer
backend/src/engine_console/
  domain/fleet.py, domain/refs.py, domain/learn.py, adapters/base.py   LOCKED (orchestrator)
  domain/models.py                     v2 additions LOCKED; further additive fields: hub core
  services/**, api/**, main.py, container.py, config.py, cli.py, migrations/**   hub core engineer
  adapters/vllm*.py, adapters/sglang*.py                engine adapter specialist (catalog refresh)
  adapters/<new engine>*.py            one engine adapter specialist each (llama.cpp, TensorRT-LLM, Ollama)
  adapters/__init__.py                 orchestrator (registers adapters as they land)
  learn/glossary.json, metrics.json, concepts.json, README.md      learn-content writer
  learn/params/<engine>.json, learn/presets/<engine>.json          that engine's adapter specialist
backend/tests/core/**, tests/fleet/**  hub core engineer
backend/tests/<engine>/**              that engine's specialist
backend/tests/learn/**                 learn-content writer
frontend/**                            frontend engineer (its mock server follows this section)
docs/** (except below)                 docs writer; ARCHITECTURE.md section 10 and FLEET-PROTOCOL.md: orchestrator
```

### 10.3 Hosts and links

* `~/.config/engine-console/hosts.json` (override with `HOSTS_FILE`): `{"hosts": [HostConfig, ...]}`, at most 32
  hosts. Read at start-up only, with `O_NOFOLLOW`; refused (the console exits with a clear message) unless owned
  by the user and not group/world-writable. No file: one host, `{"id": "local", "label": <hostname>}`. Leaving
  `local` out is allowed (the console then manages only other machines).
* `ssh` is `user@address` (letters, digits, `.`, `_`, `-`; neither the user nor the address starts with `-` or `.`), or an ssh client config alias
  when `ssh_config: true`. Hub key: `~/.config/engine-console/ssh/id_ed25519` (0600, no passphrase).
* ssh argv: `ssh -T -e none [-F none] -o BatchMode=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no
  -o VerifyHostKeyDNS=no -o IdentityAgent=none -o PKCS11Provider=none -o PasswordAuthentication=no
  -o KbdInteractiveAuthentication=no -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no
  -o ForwardX11=no -o ClearAllForwardings=yes -o PermitLocalCommand=no -o ConnectTimeout=5
  -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o IdentityFile=<hub key> -o IdentitiesOnly=yes [-p port]
  -- <ssh> /usr/local/lib/ecnode/ecnode --stdio`, with a scrubbed environment (`HOME`, `PATH`, `LANG`, `USER`).
  The hub never starts ssh while its key file is missing (the link says to run `hosts init-key`): with `-i` and a
  missing file, ssh warns and falls back to the user's default identities. `-o IdentityFile=` is used because it
  fails closed even if the file disappears after that check. Key paths with whitespace, quotes or `%` are refused.
* Local link argv: `[sys.executable, "-I", "-m", "ecnode", "--stdio"]` (the node package is a backend
  dependency), plus `--policy <path>` when the `NODE_POLICY` setting is set (`NODE_COMMAND`, when set, is the
  complete argv). Its environment is a whitelist: `HOME`, `PATH`, `USER`, `LOGNAME`, `LANG`, `XDG_RUNTIME_DIR`,
  `DOCKER_HOST`, `DOCKER_CONTEXT` and `DOCKER_CONFIG` (no other `DOCKER_*`), so secrets in the console's environment
  (an HF token, API keys) never reach the node.
* Link states: `connecting` → `up` (hello received, protocol matches) → `stale` (no sample for 6 s) → `down`
  (exponential backoff 1-60 s) | `incompatible` (protocol mismatch; retry every 300 s) | `disabled` (hosts.json
  `enabled: false`). ssh failures become human text: login refused (key not authorized), host key unknown or
  changed (connect once by hand), node not installed (exit 127), no route. Auth, host-key and not-installed
  failures retry every 300 s.
* A remote node whose `hello.forced_cmd` is false was reached with an unrestricted key: `LinkStatus.notes` says so.
* CLI (`engine-console hosts ...`): `init-key --from <addresses>` creates the hub key and prints the key line to
  add on each machine; `check` connects to every host once and prints state, versions and policy mode; `list`.
* The hub keeps the latest `hello`, `sample`, `disks` and `inventory` per host in memory, and the last good
  `models.list` snapshot per host in SQLite (a host that is down still shows its last-known library, greyed out).

### 10.4 Instances in v2

* `Instance` gains `host_id`, `host_label`, `management`, `script`, `vram[]`, `model`, `model_location`,
  `controls` (`domain/models.py`). Managed ids stay `inst_*`; external ids become `ext-<host>-<slug>`; script ids
  `scr-<host>-<script name>`. Each `-` of the host id is doubled (`gpu-2` gives `ext-gpu--2-<slug>`): host ids and slugs
  both hold `-`, and without that two hosts could produce the same id (security review M1).
* Managed instances: the `instances` table gains `host_id` (default `local`), `model` (JSON ModelRef) and the
  chosen `image`; every docker action goes to that host's node. Port allocation is per host (the node's
  `port_range` minus ports in use in its inventory); `port_in_use` from `gateway.start` makes the hub try the
  next port.
* Preflight per host: link up and policy `manage`; image allowed by the node policy and present there
  (`docker.image_present`); model present on that host (its library); fit against that host's GPUs (a
  unified-memory host budgets VRAM and RAM from the same pool); GPU clash with that host's other instances.
* Image choice: `InstanceCreate.image` if given (must be allowed by the node policy), else the settings pin for
  `(host, engine)`, else the pin for the engine, else `adapter.image_for(<host arch>)`.
* Model mounts (see `LaunchContext` in `adapters/base.py`): hf → the host's HF root, read-only, at the container
  HF cache path; gguf → the file's directory, read-only, at `/models`, with `model_path` = `/models/<file>`;
  plain → the directory at `/models`; ollama → nothing.
* `vram[]`: from the host sample, the GPU processes whose container is this instance's container (managed, script
  or external alike). `host_memory`: from the sample's per-container cgroup memory.
* `model_location`: for managed instances from the ModelRef; for others from the container mounts matched to a
  model root (the inventory reports `root`/`rel` per mount).
* `controls`: managed → all true while the host is up and in manage mode; script → start/stop/restart map to the
  script's actions (start = `up`, stop = `down`, restart = down then up) with `reason` naming the script;
  external → `stop`/`start` only when that node's policy has `external_control: "stop_start"`, else all false with
  a reason; `configure` is true for externals (opens Launch prefilled from the parsed params: "clone as managed").

### 10.5 Model library

* Built from every host's `models.list`, refreshed on `models_changed`, plus a manual rescan.
* A **copy** is one model on one host's root (`LibraryCopy`, id = first 16 hex of `sha1("host|root|rel")`).
  Copies group into a **library model** under `key = "<format>:<name>"` (hf: the repo id; gguf: the repo id or
  file stem plus `:<quant>`; ollama: `name:tag`; plain: the directory name).
* **Shared storage**: when host B's root sits on an NFS mount whose `remote_host` is one of host A's addresses
  and the export path covers the copy, the copy belongs to A and B is added to `visible_on` instead of a second
  copy being listed. When the owning machine is not a configured host, B's copy stays, with `shared_from` = the
  address.
* `in_use_by`: instances whose container mounts the copy itself (or a part of it), or mounts a folder that holds it
  (a shared HF cache) **and** loads that model: its parsed model arguments (`model`, `model_path`, draft model,
  tokenizer...) or its console label name the repo id, the repo's cache folder or the same path. A container whose
  model argument cannot be parsed counts as using everything it mounts, so a delete is refused rather than pulled
  from under a running engine.
* `engines`: adapters whose `model_formats` contain the format. Fit badges are computed on demand (`POST /fit`
  with `host_id` + `model`).
* Delete: one copy, only when its node allows deletes, the root is writable, and nothing uses it.

### 10.6 Downloads and transfers

* `POST /downloads` gains `host_id` and `root_id` (a writable root of kind `hf-home`, `hf-hub`, `gguf` or `auto`;
  default: the host's first writable HF root). `allow_patterns` selects files, e.g. one GGUF quant (`GET
  /hf/models/{repo}` returns `gguf_groups` for that).
* The hub lists the repo's files (sha256 per LFS file), asks the node `fs.stat` for existing blobs and partials,
  then streams each missing file from HF straight into `fs.write` (resuming from the partial; sha256 checked by the
  node). HF cache layout (`blobs/`, `snapshots/<sha>/`, `refs/main`) is written with `fs.write` + `fs.symlink` +
  `fs.put_small`; a plain or gguf root receives the files under `<repo dir name>/`.
* Copies: `POST /library/copies/{id}/copy {host_id, root_id?}` creates a download row with `kind: "copy"`; files
  stream `fs.read` → `fs.write` with the same resume and verification.
* State machine, pause/resume/cancel, queue and SSE are unchanged from v1. The disk pre-check uses the target
  mount's free space from the node's `disks`.

### 10.7 Learn mode

* `GET /api/v1/learn` returns `LearnBundle` (glossary, metrics, concepts, params per engine, presets). Params
  served by `GET /engines/{engine}/params` carry their `explain` and `level`.
* Coverage gate (`tests/learn`): every `ParamSpec` of every registered adapter has an Explain; every metric key of
  section 5 plus the host metrics has one; every `Explain.terms` and `Term.see_also` key exists; the length limits
  in `learn/README.md` hold; no empty `summary`.
* Concept keys: `state.<instance state>`, `link.<link state>`, `management.<class>`, `verdict.<fit verdict>`,
  `check.<preflight code>`, `format.<model format>`, `root.<kind>`, `alert.<code>`, `page.<view>` (what a page is
  for), `field.<ui field>` for non-parameter fields (host, GPU pick, root pick...).
* UI: an info button (ⓘ) next to every parameter, metric, state badge, verdict and column header; it opens a
  popover (summary, details, when to change it, risks, example, glossary links). A **Beginner mode** setting shows
  summaries inline and starts launching as a guided wizard (model → machine → goal preset → review → launch); a
  **Glossary** page lists every term, with search.

### 10.8 HTTP API additions (prefix `/api/v1`; shapes in `domain/fleet.py`)

```
GET  /hosts                                        {items: HostSummary[], next_cursor: null}
GET  /hosts/{host_id}                              HostDetail
POST /hosts/{host_id}/reconnect                    (admin) HostSummary
GET  /fleet/stream                                 SSE: event "host" (HostSummary) on change, at least every 5 s
GET  /hardware?host=<id>                           HardwareReport for that host (default local; v1 shape + host_id)
GET  /instances?host=<id>                          v2 Instance shape; POST /instances body gains host_id, model, image
POST /instances/preflight                          same body as POST /instances
GET  /library?host=&format=&q=                     {items: LibraryModel[], next_cursor}
POST /library/rescan {host_id?}                    (admin) 202 {hosts: [ids]}
DELETE /library/copies/{copy_id}                   (admin) 204
POST /library/copies/{copy_id}/copy                (admin) CopyRequest -> Download (kind "copy")
GET  /models                                       v1 shape kept: HF models of every host, each with host_id
POST /downloads                                    body gains host_id, root_id
GET  /hf/models/{repo_id}                          gains gguf_groups: [{quant, files: [path], total_bytes}]
GET  /scripts                                      {items: ScriptInfo[]}
POST /scripts/{host_id}/{name}/{action}            (admin) ScriptRunRequest -> ScriptJob (202)
GET  /scripts/jobs/{job_id}                        ScriptJob;   GET /scripts/jobs/{job_id}/stream: SSE "line" + "end"
GET  /learn                                        LearnBundle
GET  /engines                                      each engine gains model_formats, images_by_arch, multi_model
GET  /settings, PUT /settings                      gain host_image_pins
POST /fit                                          body gains host_id, model
```

### 10.9 Metrics in v2

The hub sends each node `metrics.configure` with every adapter's metric-name prefixes; the node scrapes its engine
endpoints and pushes `metrics` messages; the hub maps each one to an instance (by host + port or container) and
parses it with the adapter (managed) or the generic map (others). Per-host system series:
`GET /metrics/system?host=`. Console `/metrics` gains `engine_console_host_up{host}`,
`engine_console_host_gpu_used_gib{host,gpu}` and `engine_console_instance_vram_gib{instance,host,gpu}`.

### 10.10 Security changes (full table in docs/SECURITY.md)

* New assets: the hub's ssh key (0600, never printed or sent anywhere) and each node's policy (root-owned).
* The hub treats every node as untrusted (`proto.sanitize_*`); node text is never rendered as HTML and never used
  in SQL, paths or a shell. Per-link flood limit: DATA is bounded by the credit the hub grants (so a fast model
  copy is never mistaken for a flood); JSON frames and empty DATA frames above 20 MB or 50,000 messages in 10 s
  drop the link.
* Engine secret env values travel hub → node inside the ssh session and reach docker only through the child
  environment. The HF token never leaves the hub.
* Node HTTP proxy: loopback only, known engine ports only, allowlisted paths and methods, no redirects.
* File operations are confined to policy roots (`realpath` + `O_NOFOLLOW`); secret-looking files are never read.
* Scripts: argv from the node policy only, never a shell, one argument from an allowlist, minimal environment.

### 10.11 NodeError → HTTP

| node code | HTTP |
|---|---|
| node_unavailable, busy | 503 |
| monitor_only, conflict, in_use, port_in_use, cancelled | 409 |
| policy_denied, not_owned | 403 |
| not_found | 404 |
| no_space | 507 |
| hash_mismatch, size_mismatch, docker_error, http_error, io_error, internal | 502 |
| timeout | 504 (the action may or may not have taken effect; the UI says so) |
| unsupported | 501 |
| bad_args, unknown_op | 500 (a hub bug; logged) |

Problems keep the RFC 7807 shape; `code` is the node's code, `detail` the node's message prefixed with the host
label.

### 10.12 Quality bar additions

* Node: standard library only; `ruff`, `mypy --strict` and offline `pytest` on Python 3.12 (`make node-test`) and on
  3.12/3.13/3.14 (`make node-matrix`). No test touches real docker, GPUs, the test machine's `/proc` (inject
  roots) or the network.
* Hub: offline tests use an in-process fake node that speaks real `ecnode.proto` frames in both directions, so the
  protocol is exercised end to end without docker.
* Live verification (by the orchestrator, recorded in docs/FEATURES.md): the local host through the real node
  subprocess; discovery of the running engines; per-engine VRAM attribution; a managed launch on a GPU with free
  memory; a real HF download through the node; remote hosts once their operator has installed nodes.
