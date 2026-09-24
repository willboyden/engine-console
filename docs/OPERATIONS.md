# Operations

Commands here were derived from the code (re-read 2026-09-24) and Makefile. `make -n all` was run; the server, systemd
unit, backups, the host proxy and the router opt-in were **not** run by the author of this document (unverified).
The coordinator reports a live self-test on real rootless Docker verified the internal network, an engine with no
egress or ports, and the gateway on 127.0.0.1. Not verified: GPU engines under `--cap-drop ALL`/TP, real mitmproxy
interception and CA trust, real HF downloads.

## Prerequisites

- Ubuntu host with rootless Docker (`docker context ls` shows `rootless`) and the `ai-lab` network
  (`docker network ls`), NVIDIA driver plus container toolkit (see repo `AGENTS.md`).
- `uv` and Python 3.13 (`uv venv --python 3.13`; the system 3.14 is too new). Node 22 for frontend tests only.
- The gateway image (`nginx` pinned by digest, `GATEWAY_IMAGE` in `config.py`) must be present locally along with the engine images: the console checks presence at preflight and never pulls.
  The default pins come from the adapters (`vllm/vllm-openai:v0.23.0`, `lmsysorg/sglang:v0.5.14-cu130` in the
  code at time of writing).

## Install and run (foreground)

```bash
cd clients/engine-console
make setup     # uv venv --python 3.13 && uv sync --frozen
make run       # http://127.0.0.1:8791  (UI at /, API docs at /api/docs)
```

First start writes an admin key to `~/.local/share/engine-console/bootstrap-admin.key` (mode 0600) and
logs only the path. Loopback clients need no key by default.

## Configuration (environment, no prefix)

| Variable | Default | Meaning |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8791` | Bind address. Do not widen without a TLS proxy and keys. |
| `DATA_DIR` | `~/.local/share/engine-console` | SQLite DB, bootstrap key |
| `HF_CACHE_DIR` | `/fast/models/hf` | Environment only (not changeable via the API); absolute, no `:` `,` or newline, not a symlink. Downloads land in `<dir>/hub` |
| `HF_TOKEN` / `HF_TOKEN_FILE` | unset / `~/.config/engine-console/hf_token` | For gated models; prefer the file (0600) |
| `DOCKER_CONTEXT` | `rootless` | |
| `ENGINE_NETWORK` | `ai-lab-engines` | Internal network for engines; created `--internal` if missing, refused if it exists but is not internal |
| `GATEWAY_IMAGE` | pinned nginx digest | Per-instance TCP gateway; must already be pulled |
| `EGRESS_PROXY` | unset (direct) | e.g. `http://127.0.0.1:8082`; see "Egress proxy" |
| `EGRESS_CA_BUNDLE` | repo `security/egress/mitmproxy/ca/mitmproxy-ca-cert.pem` if it exists | Certificate only |
| `REQUIRE_EGRESS_PROXY` | `false` | `true` refuses all HF traffic unless `EGRESS_PROXY` is set |
| `ENGINE_IMAGE_ALLOWLIST` | vllm, sglang prefixes + gateway digest | Images not matching a prefix are refused |
| `ALLOWED_HOSTS` | empty | Extra Host header values (needed behind a proxy with its own hostname) |
| `MAX_SSE_CONNECTIONS`, `SSE_IDLE_TIMEOUT_S` | `8`, `300` | |
| `PORT_RANGE_START` / `PORT_RANGE_END` | `18000` / `18099` | Loopback ports published by the gateways (engines publish none) |
| `OTLP_ENDPOINT`, `OTLP_ENABLED` | `localhost:4317`, `true` | Trace export (gRPC, no TLS) |
| `TRUST_LOOPBACK` | `true` | Set `false` to require a key even from localhost |
| `ENGINE_HF_OFFLINE` | `true` | Engines get `HF_HUB_OFFLINE=1` |
| `HF_CACHE_READONLY` | `true` | Engines mount the cache read-only; only the console writes it |
| `STARTUP_TIMEOUT_S`, `DOWNLOAD_CONCURRENCY`, `SCRAPE_INTERVAL_S` | `1800`, `2`, `2.0` | |

## Egress proxy (recommended, opt-in)

By default the console reaches Hugging Face directly and `/api/v1/health` reports `egress_mode: direct` with a
warning. To route through the repo's mitmproxy chokepoint and fail closed:

```bash
# terminal 1: start the host proxy (logs and allowlists every request)
LISTEN_PORT=8082 security/egress/mitmproxy/run.sh
# terminal 2 (or in the unit's Environment=): point the console at it
EGRESS_PROXY=http://127.0.0.1:8082 REQUIRE_EGRESS_PROXY=true make run
```

The console trusts only the certificate in `EGRESS_CA_BUNDLE` (default: the repo's `ca/mitmproxy-ca-cert.pem` when
present; `run.sh` documents `~/.mitmproxy/mitmproxy-ca-cert.pem` as mitmproxy's own generated location, so set
`EGRESS_CA_BUNDLE` if yours is there). Never point it at the CA private key. If the proxy is down or the CA is missing,
HF calls fail with 503 `egress_proxy_unavailable`; there is no direct fallback. A 403 "egress blocked" means the host is
missing from `security/egress/mitmproxy/allowlist.txt`. The proxy log (`flows.mitm`) contains HF request headers
including the token: keep it out of git and treat it as a secret. This flow was not run by the author (unverified).

## Letting the router reach an engine (opt-in)

Engines sit on `ai-lab-engines` with no published ports, and the router cannot see them by default. To route
LiteLLM to an engine, a human edits `inference/litellm/docker-compose.yml` (not done by the console):

```yaml
services:
  litellm:
    networks: [ai-lab, ai-lab-hermes, ai-lab-switchyard, default, ai-lab-engines]
networks:
  ai-lab-engines:
    external: true      # created by the console on first launch (--internal)
```

Then use the instance's `internal_endpoint` (`http://<container>:<port>`, shown on the instance detail and in
`GET /api/v1/instances/{id}`) as the `api_base` of a model in `inference/litellm/config.yaml`, and recreate the router.
The network must exist first (launch one instance, or `docker --context rootless network create --internal ai-lab-engines`).
Effect: that engine can also reach the router container. Exact router networks above are copied from the compose at
time of writing and may have changed; the router path was not exercised (unverified).

## systemd user unit (opt-in)

`deploy/engine-console.service` is a template; nothing installs it for you (AGENTS.md rule 4).

```bash
mkdir -p ~/.local/share/engine-console ~/.config/engine-console
# edit the paths marked CHANGEME first
cp deploy/engine-console.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now engine-console
journalctl --user -u engine-console -f          # JSON logs
systemd-analyze --user security engine-console  # inspect exposure
```

`ProtectSystem=strict` makes everything read-only except `ReadWritePaths`. If the unit fails with a
`NAMESPACE` error, one of those paths does not exist: create it or edit the list. If a
different `HF_CACHE_DIR` is used, add it to `ReadWritePaths`. `loginctl enable-linger $USER` is needed for
it to run without an open session (not done by this repo).

Prometheus and Grafana: `deploy/grafana-engine-console.json` is a dashboard template (opt-in import; see
`observability/README.md` for the provisioning directory). `/metrics` requires a bearer key when scraped
from anywhere but loopback, so a Prometheus container on the `ai-lab` network needs a viewer key in its scrape
`authorization` config (not set up here).

## Backup

State to protect: `<data_dir>/console.db` (+ `-wal`, `-shm` while running), `<data_dir>/secrets/` (engine secret params, plaintext), `bootstrap-admin.key`,
`~/.config/engine-console/hf_token`. Model files in the HF cache are re-downloadable and are not part of this backup.
These are secrets-bearing (conversations, key hashes, token), so follow AGENTS.md rule 6: gitignored
destination, encrypt fail-closed, `umask 077`, never print contents.

Consistent copy of the DB while running (SQLite online backup via Python, so no `sqlite3` CLI is required):

```bash
umask 077
python3 - <<'PY'
import sqlite3, pathlib
src = sqlite3.connect(pathlib.Path.home() / ".local/share/engine-console/console.db")
dst = sqlite3.connect("/path/to/gitignored/dir/console.db.bak")   # CHANGEME
src.backup(dst); dst.close(); src.close()
PY
```

Then encrypt it (`gpg --symmetric` or `--encrypt -r <key>`) and remove the plaintext copy; if encryption fails,
delete the partial output. Restore: stop the service, put the file at `<data_dir>/console.db`, delete stale
`-wal`/`-shm`, start.

## Upgrade

```bash
git pull                      # or check out the new revision
make setup                    # uv sync --frozen against the committed uv.lock
make all                      # lint, typecheck, tests (offline)
systemctl --user restart engine-console
```

Migrations are ordered SQL files applied at start (`migrations/`); take a DB backup first, and treat
downgrades as unsupported (no down-migrations exist). Running engines are containers and survive a console
restart; on start the console adopts labelled containers that are still running. Bumping an engine
image is a setting (image pin) plus a new adapter param catalog: follow the repo's "prefer a validated bump"
practice and validate on a copy first.

## Troubleshooting

| Symptom | Likely cause / check |
|---|---|
| `401 unauthorized` from a proxy or non-loopback client | Send `Authorization: Bearer <key>`; a request carrying `X-Forwarded-For` is never loopback-trusted |
| Preflight fails "image not present" or `image_not_allowed` | Pull the image (`docker --context rootless pull <image>`); the console does not pull. It must also match `ENGINE_IMAGE_ALLOWLIST` (includes the gateway digest) |
| 421 `misdirected_request` | The Host header is not `127.0.0.1:8791`/`localhost:8791`; add the name to `ALLOWED_HOSTS` |
| 403 `csrf_header_required` or `cross_origin` | Non-GET request without `X-Engine-Console: 1` (or a bearer key), or from another origin |
| 503 `egress_proxy_unavailable` | Proxy not running, wrong `EGRESS_PROXY`, or CA missing; see the Egress proxy section |
| Instance `failed`: gateway missing or stopped | Restart the instance; `docker --context rootless ps -a --filter label=ai-lab.console=1` |
| `engine_network_not_internal` | A network named `ai-lab-engines` exists without `--internal`; remove or rename it |
| Instance stuck in `loading` | `GET /api/v1/instances/{id}/logs`; the state fails after `STARTUP_TIMEOUT_S` (1800 s) and stores the last log lines |
| Fit says `wont_fit` although the model should fit | Free VRAM is counted: check `fits_if_stop` and other resident engines (ComfyUI can hold VRAM too) |
| Download fails `insufficient_disk` (HTTP 507) | Free space under `HF_CACHE_DIR` (2 % margin required) |
| Gated model error | Set `HF_TOKEN` (file preferred), accept the licence on huggingface.co |
| `egress ... is not on the HF allowlist` | A CDN host changed; extend `HF_ALLOWED_HOSTS` deliberately and mirror it in the egress allowlist |
| No traces in Tempo/Phoenix | Collector not up (`observability/`); export failures are silent by design. Set `OTLP_ENABLED=false` to silence |
| Port conflict on 18000-18099 | Console picks a free gateway port; check for stale containers with `docker --context rootless ps -a --filter label=ai-lab.console=1` |
| Unit fails with `NAMESPACE`/`203` | Missing `ReadWritePaths` directory, see the unit section |
| GPU stats missing | NVML failure falls back to `nvidia-smi`; check `nvidia-smi -L` |

Always use `docker --context rootless ...`; the default context is the wrong daemon here.
