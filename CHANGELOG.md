# Changelog

## 0.4.0: the fleet

One console now manages engines and models on several GPU machines, for five engines, with a cross-machine model
library and plain-language help for every setting. Read "Breaking changes" before upgrading from 0.3.

### Breaking changes

- **Loopback is no longer trusted (`TRUST_LOOPBACK=false` is the new default).** Every API call except
  `GET /api/v1/health` (and the static app shell) needs `Authorization: Bearer <key>`; that includes `GET /metrics` and
  `/api/v1/openapi.json`. The UI asks for a key. The key written on the very first start is in
  `<DATA_DIR>/bootstrap-admin.key` (default `~/.local/share/engine-console/bootstrap-admin.key`). If you upgrade an existing
  database and no longer have a key, start once with `TRUST_LOOPBACK=true`, create one with `POST /api/v1/keys`, and restart
  without the flag. A Prometheus scrape needs a `viewer` key. `TRUST_LOOPBACK=true` still works and makes `/api/v1/health`
  warn.
- **The hub needs the node package, and no policy means no control.** Docker, GPU, `/proc` and model-folder work moved
  into `ecnode` (`node/`), which the hub starts itself: as a subprocess for its own machine and over ssh for others. `ecnode`
  is now a runtime dependency of the backend (`../node`, a path dependency): run `make setup` again. A node with no policy
  file, or one it refuses, runs in **monitor mode**: the console can watch the machine but cannot start, stop, download,
  copy, delete or run scripts. To manage the local machine, write a policy with `ecnode policy-template` and set
  `"mode": "manage"` (OPERATIONS.md).
- **Where models live changed.** The hub no longer owns a model cache: models are found, downloaded and mounted through the
  model roots of each node's policy. `HF_CACHE_DIR` no longer has any effect (it is only shown in `GET /settings`). The
  template policy lists `~/.cache/huggingface/hub` as a read-only root; set `"writable": true` to let downloads land there.
- **Hub settings that no longer do anything:** `HF_CACHE_DIR`, `DOCKER_BIN`, `GATEWAY_IMAGE`, `GATEWAY_IMAGE_ALLOWLIST`,
  `DISCOVERY_ENABLED`, `DISCOVERY_PORTS`, `DISCOVERY_INTERVAL_S`, and `ENGINE_NETWORK` (now only shown in `GET /settings`
  and a fallback in the "copy as command" snippet). Their replacements live in each node's policy: `docker.binary` and
  `docker.context`, `engine_network`, `gateway_image`, `engine_images` and `discovery_ports`. `DOCKER_CONTEXT` no longer
  drives docker from the hub (`/health` and `/settings` still echo it); if set in the hub's environment it is passed to the
  local node, which uses it only when its policy names no context. `ENGINE_IMAGE_ALLOWLIST` now only limits the global image
  pins set through `PUT /settings`; what may run is the node policy's `engine_images`.
- **Database migrations (automatic at start, no down-migrations).** `0003_fleet`: `instances` gains `host_id` (existing rows
  become `local`), `model` and `image_req`. `0004_library_and_transfers`: new `library_snapshots` table, and `downloads`
  gains `host_id`, `root_id`, `kind`, `source_copy_id`, `done_json` and `current_dest`. Back up `console.db` first; going
  back to 0.3 is unsupported. A download left paused by 0.3 can only be resumed if the local node is in manage mode and has a
  writable Hugging Face root.
- **Engine containers carry more labels, and mount model roots instead of host paths.** Four labels are new:
  `engine-console.format`, `.gpu_ids`, `.root` and `.rel`. The ownership label `engine-console=1`, the container naming and the
  other labels are unchanged, and a 0.3 container with only the old labels is adopted as a Hugging Face model (covered by an
  offline test, not run against a real 0.3 container). Engines are never given a Hugging Face token (0.3 passed it
  by name when `ENGINE_HF_OFFLINE=false`); they run with `HF_HUB_OFFLINE=1` and read the cache mounted read-only.
- **Instance ids of engines the console did not create changed:** `ext-<slug>` became `ext-<host>-<slug>`, and engines
  started by a registered script are `scr-<host>-<script>`. Each `-` of the host id is doubled (`gpu-2` gives
  `ext-gpu--2-<slug>`), so ids from two hosts can never collide. API clients that stored the old ids need updating.
- **Every model mount is read-only, and `HF_CACHE_READONLY` is gone.** The node refuses `ro: false`, and refuses a mount
  whose source holds a key- or token-looking name up to four folders down (HF model trees are skipped). From an `hf-home` root the console mounts
  only `hub/` (at `<HF_HOME>/hub` in the engine), never the HF_HOME with its `token` file.
- **Monitor mode is stricter.** Engine HTTP through a monitor-mode node is GET, plus POST to inference paths only (chat,
  completions, embeddings, Ollama chat/generate/embed/show). Pulling or deleting Ollama models through the console needs
  manage mode.
- **Docker is no longer run by the hub, so the unit template needs a second look.** `deploy/engine-console.service` was
  written for 0.3 and has not been started: with `ProtectSystem=strict` the local node can write only to the unit's
  `ReadWritePaths`, which must now include every writable model root and `~/.local/state/ecnode` (untested).

### Security

An independent review of the 0.4.0 tree (2026-10-04) found three high-severity problems in the node's file and mount
handling, all fixed before release with regression tests: a symlink target could reach the root's parent through another
link (H1), a delete could be redirected outside the root during its in-use check (H2), and read-write mounts let containers
plant links (H3). Also fixed: ambiguous instance ids across hosts (M1), a quadratic, unbounded stream reader in the hub
(M2), a CPU-burning safetensors header (M3), monitor mode allowing mutating engine POSTs (M4), mounts exposing token files
(M5), a library/lifecycle recursion (L2), non-JSON NaN/Infinity metrics (L3), a 500 on deleting some instances (L6), and an
audit log that masked fewer names than the secret store (L8). The open items are listed in docs/SECURITY.md.

### New

- **Fleet.** `~/.config/engine-console/hosts.json` lists the machines (up to 32); no file means one host, `local`. Remote
  nodes are reached with `ssh` using the hub's own key, which each machine restricts to the node command
  (`restrict,from=...,command=...`): `ssh -F none`, `BatchMode`, `StrictHostKeyChecking=yes` with known_hosts read-only, no
  agent, no forwarding, no password. New CLI: `engine-console hosts init-key --from <address>`, `hosts check`, `hosts list`.
  Links have the states `connecting`, `up`, `stale`, `down`, `incompatible` and `disabled`, and failures are explained in
  words (login refused, host key unknown or changed, node not installed, no route).
- **The node, `ecnode`** (`node/`, standard library only, Python 3.10 or newer): reports GPUs (NVML, or `nvidia-smi`), per-process
  VRAM attributed to containers, host and per-container memory, drives and mounts including NFS, containers and engine
  endpoints, and the models in each root; runs, stops and removes the engines and loopback gateways it is asked to; moves
  model files; relays engine HTTP calls to loopback ports. `node/install.sh` installs it (needs `sudo`, never edits ssh
  or sudo configuration, writes a monitor-mode starter policy only if none exists); `ecnode check` and `ecnode
  policy-template` are read-only.
- **A node-local policy is the hard boundary.** Monitor or manage mode, allowed engine images, model roots (with kinds and a
  writable switch), port range, scripts, and switches for image pulls, deletes, host IPC and stopping engines the console did
  not create. A policy file the node does not accept means monitor mode, with the reason shown.
- **Five engines:** vLLM and SGLang (catalogs re-derived from the pinned images), and new adapters for **TensorRT-LLM**,
  **llama.cpp** (GGUF, partial GPU offload) and **Ollama** (one server, many models from an existing store). The adapters
  carry the pinned images; each launch runs under `--cap-drop ALL`, `no-new-privileges`, a pids limit, on the internal network
  behind a loopback gateway, on every machine.
- **A cross-machine model library:** Hugging Face caches, GGUF files (including quants kept inside a cache), Ollama stores
  and plain checkpoint folders, per machine and per drive, grouped by model. A machine whose root is an NFS mount of another
  configured machine's disk shares the same copy instead of listing a second one. A machine that is down shows its
  last-known library, marked stale. Deleting a copy needs the node's `allow_delete`, a writable root, and no running engine
  that loads it.
- **Downloads to any machine and drive, and copies between machines** (streamed through the hub, resumable, size-checked
  and, where Hugging Face supplies a sha256, hash-checked on the machine that stores the file; a loose GGUF file or plain
  directory copied between machines is checked for size only). Hugging Face cache, GGUF and plain-directory sources can be
  copied; an Ollama store cannot.
- **Script-managed engines.** An operator's own start/stop scripts are registered in the node's policy and appear as
  instances whose start, stop and restart run those scripts, so ordering rules are never bypassed. Engines the console did not
  create can be stopped and started if the node's policy says `external_control: "stop_start"`.
- **Learn mode.** Every parameter, preset, metric, state, link state, launch check and term has a plain-language explanation
  in `engine_console/learn/*.json`, served at `GET /api/v1/learn` (parameters from `GET /engines/{engine}/params` carry
  `explain` and `level`); a test fails the build when one is missing. The info buttons, Beginner mode and Glossary page are
  being finished in the frontend.
- **Fit estimator:** `weights_gpu_fraction` and `kv_gpu_fraction` for partial offload, a unified-memory budget for machines
  where the GPU shares RAM (GB10), per-machine estimates, and a measured 2.5 GiB that vLLM holds on top of its share (one
  measurement). The free-VRAM check now follows how each engine sizes its share, as read in the pinned images' source:
  vLLM must find its whole `gpu_memory_utilization` share free at start (plus that 2.5 GiB), and SGLang's
  `mem_fraction_static` is a share of the memory free at launch, so an engine already on the card shrinks it and
  `fits_if_stop` names it (FIT-ESTIMATOR.md, "How an engine sizes its share").
- **API:** `GET /hosts`, `/hosts/{id}`, `POST /hosts/{id}/reconnect`, `GET /fleet/stream`; `host`, `host_id`, `model` and
  `image` on hardware, instances, downloads and fit; `GET /library`, `POST /library/rescan`, `DELETE /library/copies/{id}`,
  `POST /library/copies/{id}/copy`; `GET /scripts` and `POST /scripts/{host}/{name}/{action}` with job streams;
  `GET /learn`; `host_image_pins` in settings. Prometheus gains `engine_console_host_up{host}`,
  `engine_console_host_gpu_used_gib{host,gpu}` and `engine_console_instance_vram_gib{instance,host,gpu}`.
- **Security findings fixed while building it** (details in SECURITY.md): ssh used to fall back to the user's default
  identities when its key file was missing; a shared model cache made every model look in use (and an unparseable engine
  command now counts as using everything it mounts); `memlock` is refused as a container ulimit; llama.cpp's CORS default
  reflected any origin; the node never reads container environment values.

### Verification

- **Verified live** on one real workstation with two GPUs and rootless Docker, by the maintainers during this build, in
  monitor mode except for one approved managed test: the local node over a subprocess; host facts, GPUs, per-process VRAM
  attributed to containers, per-container memory, drives including an NFS mount; discovery of running engines (one
  script-managed, one external); a library scan of 8 roots (32 models, about 3 TB) including Hugging Face caches, GGUF
  inside a cache, an Ollama store and plain checkpoint folders (one in an unknown weight format); a real Hugging Face
  download through an allowlisting egress proxy into a writable root; a managed vLLM launch on one GPU under `--cap-drop ALL`,
  `no-new-privileges`, a pids limit, the internal network, a read-only model mount and the loopback gateway (ready in about
  a minute); chat, plain and streamed, through the node; stop and remove; a library delete; a 401 without an API key and the
  CSRF header enforced.
- **Fakes and tests only:** remote nodes over ssh (the command line is checked with the real `ssh -G`), copies between machines
  (between two local node processes), script actions, stopping and starting external engines, multi-GPU launches,
  TensorRT-LLM, llama.cpp, Ollama and SGLang managed launches (catalogs checked against the pinned images' `--help`, no model
  started), and GB10 unified-memory machines.
- **Not yet:** the frontend against the real backend (still being finished) and any remote machine (no node is installed on
  another machine yet).
- The documentation was written without running the server. `docs/diagrams/` holds Mermaid sources that were not rendered.

## 0.3.0
- **Host RAM monitoring.** Per-engine host memory (anon, page cache, shared memory, kernel) from the container cgroup, system
  memory and swap with alerts (`host_ram_low`, `swap_heavy`, `shmem_high`), time series (`host_ram_*` keys, `GET /metrics/system`,
  Prometheus gauges), and a `host_ram` section in the fit estimate. Verified live on three engines (an SGLang engine at
  68.6 GiB, 64 GiB of it shared memory, and two vLLM engines); the fit `host_ram` verdict and the RSS fallback for cgroup v1
  hosts are tested with fakes only. Not modelled: vLLM `swap_space`; the SGLang per-TP-rank assumption for CPU offload and
  HiCache is unconfirmed.

## 0.2.1
- Optional per-user env file (`~/.config/engine-console/env`) so settings survive restarts.
- Egress: a missing proxy CA is now reported as a CA problem (with the fix), not "proxy unreachable"; `/health` warns when `EGRESS_PROXY` is set without `EGRESS_CA_BUNDLE`.
- Local library lists models whose `refs/main` points at a missing snapshot folder instead of hiding them.
- Layout: below 1024 px the navigation is a compact sticky top bar (it used to stretch to half the screen); added a small-screen tier for filters and tables.

## 0.2.0
- **External engine discovery.** Engines already running before the console starts (containers and `127.0.0.1` port
  probes) appear as monitor-only instances with metrics from discovery onward; lifecycle actions return 409
  `instance_not_managed`, and benchmarks need `confirm_external`. New settings `discovery_enabled`, `discovery_ports`,
  `discovery_interval_s`. Verified live on three engines (one SGLang, two vLLM); other states and engine classes are
  tested with fakes only.
- **Standalone documentation.** Docs, ADRs (now 13), deploy templates and README no longer depend on any host repository;
  a minimal example egress proxy (`deploy/egress-proxy/`) is included.
- **Breaking: renamed ownership label and engine network.** The label is now `engine-console` and the default engine
  network is `engine-console-engines`. Containers and networks created by 0.1.x carry the old names and are no longer
  recognised as the console's own: stop and remove them (and the old network) before upgrading, or relaunch them
  through the console. The default model cache is `~/.cache/huggingface`.

## 0.1.1
Fix clean-clone CI: the vLLM log fixtures were being ignored (not committed), and a test that depended on a
file outside this repository now skips when that file is absent. `v0.1.0` predates this fix and fails CI on a clean clone.

## 0.1.0
First public cut. vLLM and SGLang adapters behind one engine-agnostic core; HF search and resumable
downloads; fit estimator; schema-driven launch form; instances with logs; metrics; benchmark; chat and
arena; usage; audit. Engines run on an internal network behind a per-instance loopback gateway; console
egress goes through a mitmproxy chokepoint and fails closed.

Known gaps are listed in `docs/FEATURES.md` and `docs/SECURITY.md` (residual risks). Not yet verified on
real hardware: GPU engines under `--cap-drop ALL` with tensor parallelism, and full model downloads through
the proxy.
