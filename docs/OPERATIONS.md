# Operations

Commands here were derived from the code (re-read 2026-09-24) and Makefile. `make -n all` was run; the server, systemd
unit, backups, the egress proxy, the container opt-in and external engine discovery were **not** run by the author of this document (unverified).
A live self-test on real rootless Docker verified the internal network, an engine with no egress or ports, and the
gateway on 127.0.0.1. Not verified: GPU engines under `--cap-drop ALL` or tensor parallelism, real proxy TLS interception
and CA trust, real HF downloads. Hardware example only: development targeted 2x RTX PRO 6000 Blackwell (sm_120).

## Prerequisites

- Linux host with rootless Docker (`docker context ls` shows a `rootless` context), the NVIDIA driver and the NVIDIA container toolkit. The console creates its own internal engine network on first launch.
- `uv` and Python 3.13 (`uv venv --python 3.13`; newer interpreters often lack wheels). Node 22 for frontend tests only.
- The gateway image (`nginx` pinned by digest, `GATEWAY_IMAGE` in `config.py`) must be present locally along with the engine images: the console checks presence at preflight and never pulls.
  The default pins come from the adapters (`vllm/vllm-openai:v0.23.0`, `lmsysorg/sglang:v0.5.14-cu130` in the
  code at time of writing).

## Install and run (foreground)

```bash
git clone https://github.com/willboyden/engine-console && cd engine-console
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
| `HF_CACHE_DIR` | `~/.cache/huggingface` (a symlinked default is resolved to its target) | Environment only (not changeable via the API); absolute, no `:` `,` or newline, not a symlink. Downloads land in `<dir>/hub` |
| `HF_TOKEN` / `HF_TOKEN_FILE` | unset / `~/.config/engine-console/hf_token` | For gated models; prefer the file (0600) |
| `DOCKER_CONTEXT` | `rootless` | |
| `ENGINE_NETWORK` | `engine-console-engines` | Internal network for engines; created `--internal` if missing, refused if it exists but is not internal |
| `GATEWAY_IMAGE` | pinned nginx digest | Per-instance TCP gateway; must already be pulled |
| `EGRESS_PROXY` | unset (direct) | e.g. `http://127.0.0.1:8082`; see "Egress proxy" |
| `EGRESS_CA_BUNDLE` | unset | Path to the proxy's CA certificate (certificate only, never the key). Required when `EGRESS_PROXY` is set |
| `DISCOVERY_ENABLED` | `true` | Turn external engine discovery off/on (ADR-0013) |
| `DISCOVERY_PORTS` | `8000, 30000, 11434, 8001, 18000-18099` | Loopback ports probed (JSON list in the environment) |
| `DISCOVERY_INTERVAL_S` | `10` | Refresh interval |
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

### Persistent settings: the env file

Settings can also live in a per-user file so they survive restarts without exporting variables each time:
`~/.config/engine-console/env` (override the path with `ENGINE_CONSOLE_ENV_FILE`). It uses `KEY=VALUE` lines with the
same names as the environment variables below, for example:

```
HF_CACHE_DIR=/data/models/hf
EGRESS_PROXY=http://127.0.0.1:8082
EGRESS_CA_BUNDLE=/path/to/proxy-ca-cert.pem
```

Real environment variables win over the file. Keep it `chmod 600` if it ever holds a token. Tests ignore this file.

## Egress proxy (recommended, opt-in)

By default the console reaches Hugging Face directly and `/api/v1/health` reports `egress_mode: direct` with a
warning. Any allowlisting HTTP(S) proxy works. A minimal, self-contained option is the upstream mitmproxy container
with the small addon in `deploy/egress-proxy/` (an example allowlist of Hugging Face hosts plus a 20-line addon that
answers 403 "egress blocked" for everything else). Not run by the author of this document (unverified).

```bash
# 1. start the proxy on loopback (CHANGEME: pin a mitmproxy image tag you have reviewed)
docker run -d --name egress-proxy --restart unless-stopped -p 127.0.0.1:8082:8080 \
  -v "$PWD/deploy/egress-proxy:/addon:ro" -v "$HOME/.mitmproxy:/home/mitmproxy/.mitmproxy" \
  mitmproxy/mitmproxy:CHANGEME mitmdump --set block_global=false -s /addon/allowlist_addon.py
# 2. mitmproxy writes its CA on first start; the certificate (not the key) is:
ls "$HOME/.mitmproxy/mitmproxy-ca-cert.pem"
# 3. point the console at it and refuse to run without it
EGRESS_PROXY=http://127.0.0.1:8082 \
EGRESS_CA_BUNDLE="$HOME/.mitmproxy/mitmproxy-ca-cert.pem" \
REQUIRE_EGRESS_PROXY=true make run
```

Edit `deploy/egress-proxy/allowlist.txt` (one host glob per line; the addon re-reads it on every request):

```
huggingface.co
*.huggingface.co
*.hf.co
```

The console trusts only the certificate in `EGRESS_CA_BUNDLE`; it never routes around the proxy (`trust_env=False`).
If the proxy is down or the CA is missing, HF calls fail with 503 `egress_proxy_unavailable`; there is no direct
fallback. A 403 "egress blocked" means the host is missing from the proxy allowlist. Keep `HF_ALLOWED_HOSTS` and the
proxy allowlist in step. If the container cannot write `~/.mitmproxy` under rootless Docker, adjust ownership of that
directory (unverified). The proxy sees request headers including the HF token; treat its logs as secrets.

## Letting another container reach an engine (opt-in)

Engines sit on the internal engine network with no published ports, so other containers cannot see them by default.
To let, for example, an OpenAI-compatible router reach one, an operator joins that container to the engine network
in its own compose file (the console never edits other services):

```yaml
services:
  router:
    networks: [default, engines]
networks:
  engines:
    name: engine-console-engines         # the value of ENGINE_NETWORK
    external: true                       # created by the console on first launch (--internal)
```

Then use the instance's `internal_endpoint` (`http://<container>:<port>`, shown on the instance detail and in
`GET /api/v1/instances/{id}`) as that service's upstream URL. The network must exist first (launch one instance, or
`docker --context rootless network create --internal <name>`). Effect: that engine can also reach the other container.
This path was not exercised (unverified).

## External engine discovery (monitor-only)

Engines that were already running before the console started (vLLM, SGLang, ollama, llama.cpp, TensorRT-LLM, Dynamo, or any
OpenAI-compatible server) appear on the dashboard without being launched through it. Live-verified 2026-09-24 against one
SGLang and two vLLM containers (all detected and `ready`, vLLM metrics returned, no secret in the JSON). The
`auth_required` and `unreachable` states, ollama, and the llama.cpp/TensorRT-LLM/Dynamo classification are covered by
tests with fakes only.

- Sources: running containers (`docker ps` + `docker inspect`; containers labelled `engine-console=1` are skipped) and
  loopback ports in `DISCOVERY_PORTS`. A container is listed only if its image or command line matches a known engine or it
  answers like an OpenAI-compatible server; routers, UIs, proxies and databases are never listed.
- Each appears as an instance with `managed=false`, `source=external`, id `ext-<name>`, plus `container_name`, `image`,
  `engine`, `endpoint`, `served_models`, `state_reason` and `history_since` (epoch seconds: metrics start when it was
  discovered). States: `ready`; `auth_required` (HTTP 401/403, so no metrics or chat; the console sends no key);
  `unreachable` (container found but no published `127.0.0.1` port answers: publish one to monitor it); `stopped`
  (shown for one cycle after the container disappears). Discovered instances are not stored in the database.
- Start, stop, restart, remove, patch, logs and command return 409 `instance_not_managed`. To manage an engine, relaunch it
  through the console. Chat works when `ready`. A benchmark sends real load, so `POST /bench` needs `confirm_external: true`
  (otherwise 400 `confirm_external_required`).
- `POST /api/v1/instances/discover` (admin) forces a refresh. Settings: `DISCOVERY_ENABLED`, `DISCOVERY_PORTS`,
  `DISCOVERY_INTERVAL_S`.
- Read-only, loopback only, GET only. Container environment values are never read or returned; secrets in command-line
  arguments are redacted. Text from a discovered engine is untrusted data.

## systemd user unit (opt-in)

`deploy/engine-console.service` is a template; nothing installs it for you.

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
it to run without an open session (not done by the template).

Prometheus and Grafana: `deploy/grafana-engine-console.json` is a dashboard template (opt-in; import it in Grafana or copy it into your provisioning directory). `/metrics` requires a bearer key when scraped
from anywhere but loopback, so a Prometheus container needs a viewer key in its scrape
`authorization` config (not set up here).

## Backup

State to protect: `<data_dir>/console.db` (+ `-wal`, `-shm` while running), `<data_dir>/secrets/` (engine secret params, plaintext), `bootstrap-admin.key`,
`~/.config/engine-console/hf_token`. Model files in the HF cache are re-downloadable and are not part of this backup.
These are secrets-bearing (conversations, key hashes, token), so: write only to a directory outside version control, encrypt fail-closed (never leave a plaintext copy if encryption fails), use `umask 077`, and never print contents.

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
image is a setting (image pin) plus a new adapter param catalog: pins mean tested, not frozen: validate a bump against a copy of your database first.

## Troubleshooting

| Symptom | Likely cause / check |
|---|---|
| `401 unauthorized` from a proxy or non-loopback client | Send `Authorization: Bearer <key>`; a request carrying `X-Forwarded-For` is never loopback-trusted |
| Preflight fails "image not present" or `image_not_allowed` | Pull the image (`docker --context rootless pull <image>`); the console does not pull. It must also match `ENGINE_IMAGE_ALLOWLIST` (includes the gateway digest) |
| 421 `misdirected_request` | The Host header is not `127.0.0.1:8791`/`localhost:8791`; add the name to `ALLOWED_HOSTS` |
| 403 `csrf_header_required` or `cross_origin` | Non-GET request without `X-Engine-Console: 1` (or a bearer key), or from another origin |
| 503 `egress_proxy_unavailable` | The detail says which: "TLS verification failed" means `EGRESS_PROXY` is set but `EGRESS_CA_BUNDLE` is not (`/health` warns about this); "unreachable" means the proxy is not running or `EGRESS_PROXY` is wrong; see the Egress proxy section |
| Local library shows fewer models than are on disk | The console reads `HF_CACHE_DIR` (default `~/.cache/huggingface`); point it at the directory that contains `hub/`. Models whose `refs/main` names a missing snapshot are still listed |
| Instance `failed`: gateway missing or stopped | Restart the instance; `docker --context rootless ps -a --filter label=engine-console=1` |
| `engine_network_not_internal` | A network with the `ENGINE_NETWORK` name exists without `--internal`; remove or rename it |
| Instance stuck in `loading` | `GET /api/v1/instances/{id}/logs`; the state fails after `STARTUP_TIMEOUT_S` (1800 s) and stores the last log lines |
| Fit says `wont_fit` although the model should fit | Free VRAM is counted: check `fits_if_stop` and other resident engines (other GPU processes hold VRAM too) |
| Download fails `insufficient_disk` (HTTP 507) | Free space under `HF_CACHE_DIR` (2 % margin required) |
| Gated model error | Set `HF_TOKEN` (file preferred), accept the licence on huggingface.co |
| `egress ... is not on the HF allowlist` | A CDN host changed; extend `HF_ALLOWED_HOSTS` deliberately and mirror it in the egress allowlist |
| No traces in your tracing backend | Collector not up on `localhost:4317`; export failures are silent by design. Set `OTLP_ENABLED=false` to silence |
| Port conflict on 18000-18099 | Console picks a free gateway port; check for stale containers with `docker --context rootless ps -a --filter label=engine-console=1` |
| Unit fails with `NAMESPACE`/`203` | Missing `ReadWritePaths` directory, see the unit section |
| GPU stats missing | NVML failure falls back to `nvidia-smi`; check `nvidia-smi -L` |

Always use `docker --context rootless ...`; the default context may point at a different daemon.
