# Engine Console

One page to manage GPU inference engines and the models they serve, on one machine or on several. A hub (this
application) talks to a small per-machine agent, the **node** (`ecnode`), which does all docker, GPU and file work on its
machine and obeys a local **policy file** that decides whether the console may only watch the machine or also change it.
Five engines are supported (vLLM, SGLang, TensorRT-LLM, llama.cpp, Ollama), there is a cross-machine model library with
downloads and copies, engines that an operator starts with their own scripts are handled without bypassing those scripts,
and every setting has a plain-language explanation ("learn mode").

> Status: 0.4.0, "the fleet". Verified live on **one** workstation (two GPUs, rootless Docker, mostly in monitor mode);
> remote machines over ssh and the frontend against the real backend are **not verified yet**. The rest is covered by
> offline tests with fakes, or is unverified (for example the installer). `docs/FEATURES.md` lists exactly what ran where.

## Architecture at a glance

```
browser / curl --Bearer key--> hub: FastAPI 127.0.0.1:8791 (static UI, SQLite WAL, SSE, /metrics, OTLP)
                                |
                                |-- host "local":     subprocess ------------------> ecnode --stdio --> docker, GPUs, model roots
                                |-- host "gpu-box-2": ssh, restricted hub key ------> ecnode --stdio --> docker, GPUs, model roots
                                |-- host "edge-1":    ssh, restricted hub key ------> ecnode --stdio --> docker, GPUs, model roots
                                `-- HTTPS via an allowlisting proxy (opt-in) ---------> huggingface.co

on every machine:  policy.json (mode monitor | manage, allowed images, model roots, scripts, ...)  =  the hard boundary
                   engine container (internal network, no ports, cap-drop ALL)  <--TCP--  loopback gateway 127.0.0.1:18000-18099
                   the hub reaches an engine's API only through its node (http.request); nothing new listens on the network
```

- The hub never runs docker, reads `/proc` or opens a model directory itself; every machine, its own included, is a host
  reached through a node. A bug or compromise of the hub can do at most what each node's policy allows.
- Hexagonal core with an `EngineAdapter` port; the launch forms are generated from each adapter's parameter catalog.
- Zero-npm, zero-CDN frontend (ES modules, web components, hand-rolled SVG charts).
- Engines run as labelled sibling containers on an `--internal` network behind a per-instance loopback gateway, on every
  machine. Containers the console did not create are discovered (read-only) by each node and shown as `script` or `external`
  instances.
- Fit estimator that degrades to "unknown" instead of guessing, with host-RAM and partial-offload accounting.

What the UI shows (from the contract, `docs/ARCHITECTURE.md` sections 6 and 10.7): the fleet's GPUs and engines, every
machine and its models, launch forms with a live fit meter, instances with logs and metrics, downloads, benchmarks, chat and
arena, usage, settings, an info button next to every setting, a Beginner mode and a Glossary. The views, from
`frontend/js/app.js`:

| View | What it is for |
|---|---|
| Dashboard (`#/`) and a page per machine (`#/hosts/<id>`) | Link state, GPUs, RAM, drives and the engines on each machine; Reconnect |
| Library (`#/library`) | Every model on every machine and drive, with format, size, where it is, what can run it, copy and delete |
| Discover (`#/discover`) | Hugging Face search, then download to the machine and drive you pick |
| Downloads (`#/downloads`) | Downloads and machine-to-machine copies, with progress, pause and resume |
| Launch (`#/launch`, `#/launch/wizard`) | The settings form for each engine with a live fit meter; in Beginner mode a guided wizard (model, machine, goal, review) |
| Instances (`#/instances`) | Managed, script and external engines with their controls, logs and the RAM and VRAM each holds |
| Metrics, Benchmark, Chat, Arena, Usage | Engine metrics over time, load tests, chatting with one or two engines side by side, token usage |
| Glossary (`#/learn`) | Every term and metric in plain language |
| Settings (`#/settings`) | Beginner mode, theme, per-machine image choices, API keys and the audit log |

These were checked against the mock server (`frontend/dev/mock-server.mjs`) with headless screenshots of every route; see
docs/FEATURES.md for what was verified against a real backend.

## Quick start: one machine

Prerequisites: Linux, Docker (rootless recommended) with the NVIDIA driver and container toolkit, `uv`, and the engine
images you want to use already pulled (the console never pulls). Node 22 is needed only for the frontend tests.

```bash
make setup      # backend venv (Python 3.13 via uv, installs the node package too) + node venv (Python 3.12)
```

The `engine-console` and `ecnode` commands live in the backend virtual environment. Below, `engine-console ...` and
`ecnode ...` mean `cd backend && uv run --frozen engine-console ...` (or `ecnode ...`), or the same after activating
`backend/.venv`.

**A policy for the local node.** With no policy file the node runs in monitor mode: the console can watch this machine but
cannot start, stop, download or delete anything. `ecnode policy-template` prints a starter (monitor mode, with the model
folders and docker context it detects) and writes nothing:

```bash
mkdir -p ~/.config/ecnode
(cd backend && uv run --frozen ecnode policy-template) > ~/.config/ecnode/policy.json
chmod 644 ~/.config/ecnode/policy.json     # a group- or world-writable policy is refused; umask 002 would make it 664
```

The node uses the first of these that exists: `--policy PATH` (the `NODE_POLICY` setting for the hub's own node),
`/etc/ecnode/policy.json`, `~/.config/ecnode/policy.json`. A policy in your home directory is a convenience: the hub runs
as you, so it is only a guard against hub bugs. `sudo node/install.sh` installs a root-owned `/etc/ecnode/policy.json`
(monitor mode) that the hub's user cannot edit; use that when the boundary matters.

```bash
make run        # http://127.0.0.1:8791  (the OpenAPI JSON is at /api/v1/openapi.json and needs a key; no Swagger page)
```

**Sign in.** Loopback is no longer trusted: the UI and every API call need an API key. The first start writes an admin key
to `~/.local/share/engine-console/bootstrap-admin.key` (mode 0600; the path is logged, the key never is). Put it in your
password manager, then delete the file or mint named keys (`POST /api/v1/keys`).

**Switch to manage mode** when you have read the policy: edit the file (as root if it lives in `/etc`), set
`"mode": "manage"`, `"writable": true` on the model folder where downloads may land, and `gateway_image` to an nginx image
pinned by digest that is already on the machine. The node follows within about two seconds; press **Reconnect** on the host
in the UI (or `POST /api/v1/hosts/local/reconnect`) so the hub re-reads the policy. Details, every field and the error
messages are in `docs/OPERATIONS.md`.

```bash
engine-console hosts check      # connects to every host once, prints state, node version, GPUs and policy mode
make all                        # lint, typecheck, backend, frontend and node tests (offline)
```

| Target | Does |
|---|---|
| `make setup` | create the backend and node venvs, install locked dependencies |
| `make run` | serve API and UI on 127.0.0.1:8791 |
| `make test` / `lint` / `typecheck` | backend pytest, ruff, mypy --strict |
| `make node-test` / `node-lint` / `node-typecheck` / `node-matrix` | the node agent's tests (3.12; matrix 3.12, 3.13, 3.14) |
| `make frontend-test` | `node --test` |
| `make all` | all of the above except `node-matrix` |

Frontend without a backend: `cd frontend && node dev/mock-server.mjs`.

## Add another machine (five steps)

The names and addresses are examples: the machine is `gpu-box-2` at `10.0.0.5`, the login user is `ops`, the hub is at
`10.0.0.2`.

1. **Install the node on the machine** (needs `sudo`, system `python3` 3.10 or newer, and Docker for that user): from a
   checkout of this repository, `sudo node/install.sh`. It copies the code to `/usr/local/lib/ecnode` (root-owned), writes a
   monitor-mode `/etc/ecnode/policy.json` only if none exists, and ends with `ecnode check`, a read-only report of what the
   machine would tell the hub. It edits no ssh or sudo configuration and starts nothing.
2. **Create the hub's key** on the hub: `engine-console hosts init-key --from 10.0.0.2`. It creates the key once (mode 0600,
   never printed) and prints one line that restricts the key to the node command and to the hub's address:
   `restrict,from="10.0.0.2",command="/usr/local/lib/ecnode/ecnode --stdio" ssh-ed25519 AAAA... engine-console-hub`.
3. **Authorize the key** on the machine: add that one line to the authorized keys of the login user (`ops`), the user whose
   Docker the node should use.
4. **Pin the host key and register the machine.** On the hub, as the hub's user, connect once by hand (`ssh ops@10.0.0.5 true`)
   and check the fingerprint before accepting; a final `Permission denied` is fine, the host key is stored before
   authentication. The console never writes known_hosts and refuses an unknown or changed key. Then add the machine to
   `~/.config/engine-console/hosts.json` (a plain file you own, not group- or world-writable):
   `{"hosts": [{"id": "local", "label": "Workstation"}, {"id": "gpu-box-2", "label": "GPU box 2", "ssh": "ops@10.0.0.5"}]}`.
5. **Check it:** `engine-console hosts check`, then restart the console (it reads `hosts.json` only at start-up). A remote
   node starts in monitor mode; its administrator switches it to manage in `/etc/ecnode/policy.json`, and you press
   Reconnect.

If a link is not `up`, `docs/OPERATIONS.md` ("Link states") gives the exact message and the fix for each case.

## Engines

| Engine | Pinned image (the adapter's default) | Model formats | Notes |
|---|---|---|---|
| vLLM | `vllm/vllm-openai:v0.27.1` | `hf`, `plain` | Prometheus metrics; compile caches kept in named volumes |
| SGLang | `lmsysorg/sglang:v0.5.20-cu130` | `hf`, `plain` | Prometheus metrics |
| TensorRT-LLM | `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` | `hf`, `plain` | PyTorch backend; `/metrics` is a slow JSON iteration-stats array; no API key option |
| llama.cpp | `ghcr.io/ggml-org/llama.cpp@sha256:4c0ece46...` (build b10156, pinned by digest) | `gguf` | one model per server, partial GPU offload; CORS limited to `localhost` unless set; setting an API key makes `/metrics` answer 401, so live metrics are lost |
| Ollama | `ollama/ollama:0.30.11` | `ollama` (models already in an Ollama store) | one server, many models; no Prometheus metrics, no API key; importing a GGUF or Hugging Face model into a store is not supported |

Formats: `hf` is a Hugging Face repository in a cache folder, `gguf` a `.gguf` file or split set, `plain` a model directory
with `config.json` and weights outside any cache, `ollama` a name:tag in an Ollama store. A library entry also has `other` for a
folder the console cannot read as a model (listed with its size only). The pins live in the adapters
(`backend/src/engine_console/adapters/*_params.py`); `GET /api/v1/engines` shows the effective default, and `PUT /api/v1/settings`
can pin another image per engine or per machine. What may actually run is each node's `engine_images`. The adapters carry no
arm64 images; on an aarch64 machine set an image pin that exists for it (unverified). TensorRT-LLM, llama.cpp, Ollama and
SGLang launches have been checked against the pinned images' `--help` only; no model was started for them.

## Security posture (summary)

Policy on every machine is the hard boundary (monitor or manage); the hub's ssh key is its own, restricted to the node command
and to the hub's address; ssh runs with `-F none`, `BatchMode`, strict host keys, no agent and no forwarding; engines run with
`--cap-drop ALL`, `no-new-privileges` and a pids limit on an internal network, behind a loopback gateway; every node is treated
as untrusted by the hub; loopback is not trusted (API key required, plus Host, Origin and CSRF checks); secrets are `[set]`
markers or 0600 files; model files are confined to the policy's roots. Residual risks, stated plainly in `docs/SECURITY.md`:
a user in a rootful `docker` group is root-equivalent on that machine, whoever controls the hub controls whatever each policy
allows, an engine runs model code when `trust_remote_code` is set, the hub's ssh key has no passphrase, and a delete from a
shared model cache relies on the hub's in-use rule. No penetration test or independent review has been done.

## Read next

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): the contract (v1 sections 1-9; v2, the fleet, in section 10) and
  [`docs/FLEET-PROTOCOL.md`](docs/FLEET-PROTOCOL.md): the node wire protocol and the policy file.
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md): the operator guide (hosts, ssh, policy, scripts, library, troubleshooting).
- [`docs/SECURITY.md`](docs/SECURITY.md): STRIDE table, trust boundaries, findings fixed, residual risks.
- [`docs/FEATURES.md`](docs/FEATURES.md): status matrix with what is verified live, with fakes only, or not yet.
- [`docs/adr/`](docs/adr/): 21 decisions with their trade-offs and rejected alternatives (14-21 are the fleet).
- [`docs/diagrams/`](docs/diagrams/): Mermaid C4, sequences and state machines (hand-written, not rendered or validated).
- [`docs/FIT-ESTIMATOR.md`](docs/FIT-ESTIMATOR.md), [`node/README.md`](node/README.md) and
  [`backend/src/engine_console/learn/README.md`](backend/src/engine_console/learn/README.md) (how the help text is written).

## Deploy (opt-in templates, nothing self-applies)

- `deploy/engine-console.service`: a hardened systemd `--user` unit. It was written for 0.3 and has never been started;
  see the unit section of `docs/OPERATIONS.md` before using it with 0.4.
- `deploy/grafana-engine-console.json`: dashboard for the console's own `/metrics`.
- `deploy/egress-proxy/`: example allowlist and addon for a mitmproxy-based egress proxy.
- `node/install.sh`: installs the node on a machine (needs `sudo`, `--uninstall` removes it and keeps the policy).

## Layout

```
backend/    FastAPI hub: services, adapters (5 engines), learn content, migrations, tests
node/       ecnode, the per-machine agent (standard library only), install.sh, example policies, tests
frontend/   ES modules + web components, i18n/en.json, node tests, mock server
docs/       contract, protocol, ADRs, diagrams, feature matrix, security, operations, fit estimator
deploy/     systemd unit, Grafana dashboard, example egress proxy
```

## Limitations

- NVIDIA GPUs and Linux machines only (the node reads `/proc`, `/sys` and NVML or `nvidia-smi`); up to 32 hosts; one hub, one
  user at a time, SQLite, no high availability. Engines start with `--restart no`: after a reboot they stay stopped.
- **Not verified:** remote machines over ssh (offline tests and a real `ssh -G` check only), copies between real machines,
  script actions, external stop and start, multi-GPU launches, launches of any engine but vLLM, aarch64 and unified-memory
  machines, and the frontend against the real backend (still being finished).
- The hub does not pull images and does not install nodes; you do both on each machine. A node's `allow_pull` switch exists
  but nothing sends a pull yet.
- A change to a node's policy file is enforced by the node within seconds but is shown in the hub only after the host
  reconnects.
- Copies stream through the hub, so every byte crosses it twice; an Ollama store, an `other` folder and a plain model directory
  whose files cannot be counted exactly cannot be copied; a copy that starts in a Hugging Face cache needs the hub to reach
  Hugging Face. Shared (NFS) storage is recognised only when the exporting machine is one of the configured hosts.
- Ollama serves only models already in a store. Dynamo is recognised for monitoring and cannot be launched.
- Fit estimates are uncalibrated apart from one vLLM overhead measurement; partial offload is modelled for llama.cpp only.
- No LRU model swapping, RAG, tools or web search; English only; no browser end-to-end tests; no penetration test.
