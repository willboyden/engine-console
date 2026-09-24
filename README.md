# Engine Console

A single-host control plane and observability UI for **vLLM** and **SGLang** containers:
search and download Hugging Face models, estimate whether they fit your GPUs, launch and supervise
engine containers, watch live metrics, benchmark, chat, compare models in an arena, and audit every
change. Built as an engine-agnostic core plus one adapter per engine.

> Status: v0.1, offline-tested design. The author did not run the test suite or a live engine when writing
> these docs; treat claims of behaviour as "implemented in code", not "verified on hardware".

## Architecture at a glance

```
browser --loopback--> FastAPI (127.0.0.1:8791) --docker CLI (rootless)--> gateway (127.0.0.1:18000-18099) --> engine
   static UI (no build)     |  SQLite WAL, SSE, /metrics, OTLP                  (internal network, no ports, no egress)
                            `--HTTPS via allowlisting proxy (opt-in)--> huggingface.co
```

- Hexagonal core; `EngineAdapter` port; UI forms generated from each adapter's parameter catalog.
- Host process, not a container, so no container ever holds the Docker socket.
- Zero-npm frontend; zero-CDN; hand-rolled SVG charts.
- **Host RAM visibility:** per-engine anon / page cache / shared memory / kernel from the container cgroup, system memory and swap with alerts, time series, and a host-RAM line in the fit estimate (live-verified on three engines; the fit part with fakes only).
- **Prefilled dashboard:** engines already running (vLLM, SGLang, ollama, llama.cpp and other OpenAI-compatible servers) are
  discovered read-only and shown as monitor-only instances, with metrics from discovery onward (ADR-0013). Live-verified on
  three engines; some states and engine classes are tested with fakes only.
- Fit estimator that degrades to "unknown" instead of guessing.

Read next: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (contract), [`docs/adr/`](docs/adr/) (13 decisions with
trade-offs), [`docs/diagrams/`](docs/diagrams/) (Mermaid C4, sequences, state machines),
[`docs/FEATURES.md`](docs/FEATURES.md) (honest comparison with oMLX and Open WebUI),
[`docs/SECURITY.md`](docs/SECURITY.md), [`docs/FIT-ESTIMATOR.md`](docs/FIT-ESTIMATOR.md),
[`docs/OPERATIONS.md`](docs/OPERATIONS.md).

## Quick start

```bash
make setup      # uv venv --python 3.13 + uv sync --frozen
make run        # http://127.0.0.1:8791   API docs: /api/docs
make all        # lint + typecheck + backend tests + frontend tests (offline)
```

Example hardware: developed for 2x RTX PRO 6000 Blackwell (sm_120); nothing is hard-wired to it.

Prerequisites: Docker (rootless recommended), NVIDIA driver and container toolkit, `uv`, Node 22 (tests only).
Frontend without a backend: `cd frontend && node dev/mock-server.mjs`.

| Target | Does |
|---|---|
| `make setup` | create venv, install locked deps |
| `make run` | serve API and UI |
| `make test` / `lint` / `typecheck` | pytest, ruff, mypy --strict |
| `make frontend-test` | `node --test` |
| `make all` | lint, typecheck, test, frontend-test |

## Security posture (summary)

Loopback bind with Host allowlist, Origin and CSRF-header checks; bearer keys (admin/viewer); audit log; secret params never in the DB;
engines on an internal network with no ports or egress behind a per-instance gateway; console egress through an allowlisting
proxy (opt-in, fail-closed); `--cap-drop ALL`. Known gaps: local processes are still admin unless
`TRUST_LOOPBACK=false`; egress proxy is off by default; GPU engines under `cap-drop ALL` and real proxy interception are unverified. Full STRIDE table in `docs/SECURITY.md`.

## Deploy (opt-in templates, nothing self-applies)

- `deploy/engine-console.service`: hardened systemd `--user` unit.
- `deploy/grafana-engine-console.json`: dashboard for the console's own `/metrics`.
- `deploy/egress-proxy/`: example allowlist and addon for a mitmproxy-based egress proxy.

## Layout

```
backend/    FastAPI app, services, adapters, migrations, tests
frontend/   ES modules + web components, i18n/en.json, node tests, mock server
docs/       architecture, ADRs, diagrams, feature matrix, security, ops, fit estimator
deploy/     systemd unit, Grafana dashboard
```

## Limitations

Only vLLM and SGLang; NVIDIA only; single node, single user; no LRU model swapping, RAG or tool use;
English only; fit estimates uncalibrated; no browser end-to-end tests; several views verified only against a mock server.
