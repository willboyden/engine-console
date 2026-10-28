# Engine Console

One page to manage GPU inference engines and the models they serve, on one machine or on several. A hub (this
application) talks to a small per-machine agent, the **node** (`ecnode`), which does all docker, GPU and file work on its
machine and obeys a local **policy file** that decides whether the console may only watch the machine or also change it.
Five engines are supported (vLLM, SGLang, TensorRT-LLM, llama.cpp, Ollama), there is a cross-machine model library with
downloads and copies, engines that an operator starts with their own scripts are handled without bypassing those scripts,
and every setting has a plain-language explanation ("learn mode").

> Status: 0.5.0, "the Rust rewrite". The hub and the node agent are now Rust (one Cargo workspace in `backend/`); the HTTP
> API, the node wire protocol, the persisted state and the frontend are unchanged from 0.4.x. The Rust code is
> parity-checked against the last Python release (tag `python-final`): an HTTP differential run of 674 steps (665 same, 9
> allowed differences that are headers only, 0 different), the node protocol cross-matrix and both mixed hub/node stacks,
> and all 2487 tests of that release mapped to Rust tests (2484) or marked not applicable (3). The browser end-to-end suite
> renders every page against the Rust hub and node (fake CLIs). **Not yet done with the Rust code:** a run on real GPUs, a
> real Docker launch, a real remote machine over ssh, an aarch64 machine, and the systemd unit. Everything described as
> "verified live" in these docs was seen with the Python 0.4.x implementation (one workstation, mostly monitor mode; five
> remote machines monitored over ssh on 2026-10-15; managing a remote machine was never verified).
> `docs/FEATURES.md` lists exactly what ran where.

## Architecture at a glance

```
browser / curl --Bearer key--> hub: engine-console (Rust) 127.0.0.1:8791 (static UI, SQLite WAL, SSE, /metrics, OTLP)
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

These were checked against the mock server (`frontend/dev/mock-server.mjs`) with headless screenshots of every route, and
`make e2e` (`frontend/e2e/run.mjs`) opens every route, with Beginner mode on and off, in a real headless Chrome against the
real backend and its real local node (the Rust hub and node since 0.5.0). That run is hermetic: a throwaway home and data directory, a monitor-mode policy with
no Docker binary and no port probes, Hugging Face pointed at a closed port, and two tiny seeded models. It fails on an error
or not-found state, the sign-in dialog, an uncaught exception, a console error, an API 5xx, or a frontend file that is not
served with `Cache-Control: no-cache`. It renders pages; it does not click through actions.

## Quick start: one machine

Prerequisites: Linux; a Rust toolchain via [rustup](https://rustup.rs) (the pinned 1.99.1 installs itself from
`backend/rust-toolchain.toml` the first time you build); on the machines that run engines, Docker (rootless recommended)
with the NVIDIA driver and container toolkit, and the engine images you want to use already pulled (the console never
pulls). Node 22 is needed only for the frontend tests and the browser suite, and `uv` only for the parity reference
(`make parity`, `make proto`). No Python is needed to build or run the console or the node.

```bash
make setup      # cargo build --locked (debug binaries in backend/target/debug) + enable the git hooks
make release    # optimised binaries: backend/target/release/engine-console and backend/target/release/ecnode
```

Below, `engine-console ...` and `ecnode ...` mean `backend/target/release/engine-console ...` and
`backend/target/release/ecnode ...` (or put that directory on your `PATH`). The hub starts the `ecnode` that sits next to
its own binary as the node of the local machine.

**A policy for the local node.** With no policy file the node runs in monitor mode: the console can watch this machine but
cannot start, stop, download or delete anything. `ecnode policy-template` prints a starter (monitor mode, with the model
folders and docker context it detects) and writes nothing:

```bash
mkdir -p ~/.config/ecnode
backend/target/release/ecnode policy-template > ~/.config/ecnode/policy.json
chmod 644 ~/.config/ecnode/policy.json     # a group- or world-writable policy is refused; umask 002 would make it 664
```

The node uses the first of these that exists: `--policy PATH` (the `NODE_POLICY` setting for the hub's own node),
`/etc/ecnode/policy.json`, `~/.config/ecnode/policy.json`. A policy in your home directory is a convenience: the hub runs
as you, so it is only a guard against hub bugs. `sudo node/install.sh --binary ...` installs a root-owned
`/etc/ecnode/policy.json` (monitor mode) that the hub's user cannot edit; use that when the boundary matters.

A node runs `docker`, `nvidia-smi` and `ip` only from the fixed path `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`,
never from your `PATH`. If `command -v docker` prints a path outside it, set `"docker": {"binary": "<that absolute path>"}`
in the policy.

```bash
make run        # builds the release binaries, then serves http://127.0.0.1:8791 in the foreground
                # (the OpenAPI JSON is at /api/v1/openapi.json and needs a key; no Swagger page)
```

On a Linux desktop, run `make release` and then `make install-desktop-icon` to add an "Engine Console" launcher to the app
menu and the desktop: it starts the console in the background when it is not running and opens the UI (right-click it for
"Stop Engine Console"). Details in `docs/OPERATIONS.md`.

**Sign in.** Loopback is no longer trusted: the UI and every API call need an API key. The first start writes an admin key
to `~/.local/share/engine-console/bootstrap-admin.key` (mode 0600; the path is logged, the key never is). Put it in your
password manager, then delete the file or mint named keys (`POST /api/v1/keys`).

**Switch to manage mode** when you have read the policy: edit the file (as root if it lives in `/etc`), set
`"mode": "manage"`, `"writable": true` on the model folder where downloads may land, and `gateway_image` to an nginx image
pinned by digest that is already on the machine. The node follows within about two seconds; press **Reconnect** on the host
in the UI (or `POST /api/v1/hosts/local/reconnect`) so the hub re-reads the policy. Details, every field and the error
messages are in `docs/OPERATIONS.md`.

```bash
backend/target/release/engine-console hosts check   # connects to every host once, prints state, node version, GPUs and policy mode
make all        # lint, tests, test map, cargo-deny, cargo-audit, frontend tests and the browser suite (needs a Chrome)
```

| Target | Does |
|---|---|
| `make setup` | `cargo build --locked` (debug) and enable the git hooks (`make hooks`) |
| `make build` / `make release` | debug binaries / optimised binaries in `backend/target/{debug,release}/` (`release` remaps build-machine paths and refuses a binary that still holds one) |
| `make run` | `make release`, then serve API and UI on 127.0.0.1:8791 |
| `make test` | every Rust test (hub, node, protocol), hermetic: fakes, temp dirs, closed ports |
| `make lint` / `fmt` / `clippy` / `typecheck` | rustfmt check and clippy with warnings as errors / rustfmt in place / clippy / the same as `clippy` |
| `make deny` / `audit` / `crate-age` | cargo-deny (advisories, licences, bans, sources) / cargo-audit / the 14-day dependency cooldown (queries crates.io) |
| `make test-map` | every test of the last Python release names a Rust test that exists and is compiled (`backend/TEST-MAP.md`) |
| `make parity` | HTTP differential run of the Rust hub against the last Python release (needs `uv`), then Python against Python as a self-test |
| `make proto` | the node protocol cross-matrix (Python and Rust driver against Python and Rust node) and both mixed stacks |
| `make node-test` | the node's and the protocol's tests only |
| `make node-dist` | `backend/target/dist/ecnode-<version>-<arch>` and its `.sha256`, for `node/install.sh --binary` |
| `make frontend-test` | `node --test` |
| `make e2e` | every route in headless Chrome against the real hub and node (needs a Chrome: `CHROME_BIN`, `git config engine-console.chrome <path>` or one on PATH); `scripts/e2e-gate.sh` builds first and records a pass that the `pre-push` hook requires |
| `make all` | `lint test test-map deny audit frontend-test e2e` (`crate-age`, `parity` and `proto` are separate) |
| `make install-desktop` / `install-desktop-icon` / `uninstall-desktop` | the app-menu launcher (and a desktop copy); needs `make release` first |

Frontend without a backend: `cd frontend && node dev/mock-server.mjs`.

## Add another machine (five steps)

The names and addresses are examples: the machine is `gpu-box-2` at `10.0.0.5`, the login user is `ops`, the hub is at
`10.0.0.2`.

1. **Install the node on the machine** (needs `sudo` and Docker for that user; no Python, compiler or network). Get the
   `ecnode-<version>-<arch>` binary and its `ecnode-<version>-<arch>.sha256` (a release asset, or `make node-dist` on a
   machine of that architecture), copy both to the machine, and from a checkout of this repository run
   `sudo node/install.sh --binary ecnode-<version>-<arch>`. It checks a copy of the file (sha256, ELF header, machine
   type, and that `ecnode version` runs), swaps `/usr/local/lib/ecnode` in with one rename (root-owned), writes a
   monitor-mode `/etc/ecnode/policy.json` only if none exists, and ends with `ecnode check`, a read-only report of what the
   machine would tell the hub. It edits no ssh or sudo configuration and starts nothing. The checksum catches a corrupted or
   wrong-architecture file, not a malicious release: the binary and its `.sha256` come from the same place.
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

To move a machine that already runs the Python node (0.4.x) to the Rust node, no key or `authorized_keys` change is needed;
see "Upgrade" and "Rollback" in `docs/OPERATIONS.md`. If a link is not `up`, the same document ("Link states") gives the exact
message and the fix for each case.

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
(`backend/crates/ec-hub/src/adapters/*_params.rs`); `GET /api/v1/engines` shows the effective default, and `PUT /api/v1/settings`
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
- [`docs/adr/`](docs/adr/): 22 decisions with their trade-offs and rejected alternatives (14-21 are the fleet, 22 is the Rust rewrite).
- [`docs/diagrams/`](docs/diagrams/): Mermaid C4, sequences and state machines (hand-written, not rendered or validated).
- [`docs/FIT-ESTIMATOR.md`](docs/FIT-ESTIMATOR.md), [`node/README.md`](node/README.md),
  [`backend/PORTING.md`](backend/PORTING.md) (the porting record; its section 14 lists the deliberate deviations, DV1-DV50) and
  [`backend/crates/ec-hub/learn/README.md`](backend/crates/ec-hub/learn/README.md) (how the help text is written).

## Deploy (opt-in templates, nothing self-applies)

- `deploy/engine-console.service`: a hardened systemd `--user` unit that runs `backend/target/release/engine-console`. It has
  never been started (not with the Python hub, not with the Rust one); see the unit section of `docs/OPERATIONS.md` first.
- `deploy/grafana-engine-console.json`: dashboard for the console's own `/metrics`.
- `deploy/egress-proxy/`: example allowlist and addon for a mitmproxy-based egress proxy.
- `node/install.sh`: installs a prebuilt `ecnode` binary on a machine (needs `sudo`; `--uninstall` removes it and keeps the policy).

## Layout

```
backend/    Cargo workspace (Rust 1.99.1, edition 2024):
              crates/ec-hub     the hub (library + binary engine-console): services, adapters (5 engines), API, migrations/, learn/
              crates/ecnode     the per-machine node agent (binary ecnode)
              crates/ec-proto   the wire protocol both binaries link (+ the dev driver ecproto-drive)
              parity/           differential harness against the last Python release (Node 22, uv); PORTING.md, TEST-MAP.md
node/       install.sh (binary-only node installer), README.md, example policies in examples/
frontend/   ES modules + web components (no build step), i18n/en.json, node tests, mock server, e2e
docs/       contract, protocol, ADRs, diagrams, feature matrix, security, operations, fit estimator
deploy/     systemd unit, Grafana dashboard, example egress proxy
```

## Limitations

- NVIDIA GPUs and Linux machines only (the node reads `/proc`, `/sys` and NVML or `nvidia-smi`); up to 32 hosts; one hub, one
  user at a time, SQLite, no high availability. Engines start with `--restart no`: after a reboot they stay stopped.
- **Not verified with the Rust code (0.5.0):** anything on real GPUs, a real Docker daemon, a real remote machine over ssh, an aarch64
  machine, or under systemd (the arm64 CI job has not run yet). **Not verified at all, with either implementation:** managing remote machines over ssh (the links and monitoring ran live on five machines in monitor mode;
  launches, copies and manage mode only with offline tests and a real `ssh -G` check), copies between real machines,
  script actions, external stop and start, multi-GPU launches, launches of any engine but vLLM, launches and fit estimates on
  aarch64 and unified-memory machines (their nodes ran live in monitor mode), and the frontend's actions against a manage-mode machine (the browser suite renders every page in monitor mode).
- The hub does not pull images and does not install nodes; you do both on each machine. A node's `allow_pull` switch exists
  but nothing sends a pull yet.
- A change to a node's policy file is enforced by the node within seconds but is shown in the hub only after the host
  reconnects.
- Copies stream through the hub, so every byte crosses it twice; an Ollama store, an `other` folder and a plain model directory
  whose files cannot be counted exactly cannot be copied; a copy that starts in a Hugging Face cache needs the hub to reach
  Hugging Face. Shared (NFS) storage is recognised only when the exporting machine is one of the configured hosts.
- Ollama serves only models already in a store. Dynamo is recognised for monitoring and cannot be launched.
- Fit estimates are uncalibrated apart from one vLLM overhead measurement; partial offload is modelled for llama.cpp only.
- No LRU model swapping, RAG, tools or web search; English only; the browser suite renders every page but does not click
  through actions; no penetration test.
