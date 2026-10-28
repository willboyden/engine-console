# Changelog

## 0.5.0: the Rust rewrite

The hub and the node agent are rewritten in Rust: one `engine-console` binary for the hub and one `ecnode` binary per
architecture (x86_64, aarch64) for every machine, and no Python on any machine. The HTTP API, the wire protocol
(protocol 1), the files the console keeps, the command lines and the UI are unchanged, so a 0.4 hub drives a 0.5 node
and a 0.5 hub drives a 0.4 node: a fleet can be switched one machine at a time and switched back the same way. Read
"Upgrading" before switching a remote machine.

**How far it is verified.** The Rust code was checked against the last Python code (tag `python-final`, which stays
runnable as the reference of the parity harness):

- the HTTP differential harness sends the same 674 requests to the Python hub and the Rust hub, with no fleet, with a
  real local node in monitor mode (fake `docker`, `nvidia-smi` and `ip`), and with a fake Hugging Face behind an
  allowlisting egress proxy: 665 answers are the same, 9 differ only in the security headers the Rust hub adds to its
  catch-all 500 (deviation DV30), none differ otherwise;
- the node protocol cross-matrix (the Python and the Rust driver against the Python and the Rust node, 9 scenarios)
  equals the recorded Python transcripts except the documented `hello.host.python` field and the release number; the
  Python hub driving the Rust node answers like the all-Python stack apart from those two node fields, and the Rust hub
  driving the Python node answers exactly like it;
- each of the 2487 tests of the last Python release is mapped to a Rust test that exists and is compiled (2484), or
  marked N/A for Python-only mechanics (3) (`backend/TEST-MAP.md`); the Rust suite runs hermetically (fakes, temp dirs,
  closed ports);
- the browser suite opens every page against the Rust hub and the Rust node.

**Not yet verified with the Rust code:** a run on real GPUs (NVML or `nvidia-smi`), a real Docker launch, a real remote
machine over ssh, and an aarch64 machine (CI builds and tests `ecnode` on GitHub's arm64 runner; that job has not run
yet). Everything the documentation calls "verified live" was seen with the Python implementation of 0.4.

### Upgrading

- **The hub.** Stop the 0.4 console first, with the launcher's "Stop" action or the 0.4 `engine-console stop`: the 0.5
  `stop` stops only a console the 0.5 `open` started (DV7). Then `make release` (needs a Rust toolchain through rustup;
  the pinned 1.99.1 installs itself) and `make install-desktop` (and `install-desktop-icon` if you use it): the launcher
  file keeps its name and now runs `backend/target/release/engine-console`. The old launcher points at
  `backend/.venv/bin/engine-console`, which is gone. Every file is read as 0.4 wrote it: `~/.config/engine-console/`
  (env file, `hosts.json`, the hub's ssh key), `~/.local/share/engine-console/` (`console.db`, `bootstrap-admin.key`)
  and each machine's `/etc/ecnode/policy.json`. The hub's own machine needs nothing else: its local node is the `ecnode`
  next to `engine-console`. The systemd template (`deploy/engine-console.service`) runs the release binary.
- **Remote machines, one at a time.** On each machine: (1) as the node's user, run `command -v docker`; if the path is
  outside `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`, set `"docker": {"binary": "<that path>"}` in
  `/etc/ecnode/policy.json` first, because the Rust node runs its tools only from that fixed PATH (DV31; `nvidia-smi` and
  `ip` must be on it); (2) copy `ecnode-0.5.0-<arch>` and `ecnode-0.5.0-<arch>.sha256` there (release assets, or `make
  node-dist` on a machine of that architecture); (3) `sudo node/install.sh --binary ecnode-0.5.0-<arch>`; (4) on the hub,
  `engine-console hosts check`. The node stays at `/usr/local/lib/ecnode/ecnode`, so the key line in `authorized_keys`
  (`command="/usr/local/lib/ecnode/ecnode --stdio"`) does not change and nothing is re-keyed. Read what
  `install.sh` prints from `ecnode check`: 0.5 refuses a policy whose model root contains the home directory, or is,
  contains or lies inside `/etc/ecnode`, the policy file's folder, `~/.ssh`, `~/.config/ecnode` or
  `~/.local/state/ecnode` (DV45). A refused policy means monitor mode with no port probing (DV44), so fix the root and
  check again.
- **Going back.** One machine: in a checkout of the tag `python-final` (or `v0.4.0`), `sudo node/install.sh` installs the
  Python node at the same path again (it needs the machine's system `python3` 3.10 or newer). The hub: run the 0.4 hub of
  that checkout (`make setup`, `make run`); it reads the same files.
- **A fleet installer script of your own** changes its program path only: `<checkout>/backend/target/release/engine-console
  hosts init-key ...` and `... hosts check` instead of `uv run engine-console ...` (same arguments, output and exit codes);
  it ships `ecnode-<version>-<arch>` with its `.sha256` instead of `node/` and passes `--binary`.

### Changed

- **One Rust workspace in `backend/`** (Rust 1.99.1, pinned by `backend/rust-toolchain.toml`): `crates/ec-hub` (the hub:
  axum on its own hyper listener, SQLite through rusqlite, reqwest with rustls, OTLP over gRPC), `crates/ecnode` (the node;
  NVML loaded at run time when the driver has it, else `nvidia-smi`) and `crates/ec-proto` (the wire protocol both link).
  Migrations and learn content are compiled into the hub. The Python packages in `backend/` and `node/` are removed;
  `node/` keeps the installer and the policy examples.
- **`node/install.sh --binary FILE`** installs a prebuilt `ecnode`: it checks the file against `FILE.sha256` and that it is
  a 64-bit ELF executable for this machine's `uname -m` (on the copy it is about to install), runs `ecnode version`, swaps
  the whole `/usr/local/lib/ecnode` in with one rename (an old Python install goes with the old directory), writes
  `/etc/ecnode/policy.json` once from `ecnode policy-template` and runs `ecnode check`. Without `--binary` it refuses and
  says how to go back to the Python node. The checksum catches a damaged or wrong file, not a malicious release: the
  binary and its `.sha256` come from the same place.
- **Release binaries:** `make release` builds `engine-console` and `ecnode` with no path of the build machine inside (the
  build fails if one remains); `make node-dist` writes `ecnode-<version>-<arch>` and its `.sha256`. On a version tag, CI
  builds `ecnode` on Ubuntu 22.04 for x86_64 and aarch64 (glibc 2.35: runs on Ubuntu 22.04 and 24.04 and DGX OS 7) and
  attaches both, with their checksums, to a draft release.
- **Developer workflow:** `make setup | build | release | run | test | lint | clippy | deny | audit | crate-age | test-map |
  parity | proto | frontend-test | e2e | all | node-test | node-dist` (`make help`). Dependencies come from the committed
  `Cargo.lock` (`--locked` everywhere), are checked by cargo-deny and cargo-audit, and must have been published for 14
  days (security fixes excepted, each listed with its advisory). CI runs formatting, clippy, the tests and the test map,
  the supply-chain checks, the node on an arm64 runner, the frontend tests, the browser suite and the parity runs against
  the Python reference.
- **The parity harness stays** (`backend/parity/`): `make parity` and `make proto` run the last Python release, exported
  from git into `backend/target/` and set up with `uv`, against the Rust binaries; `EC_PYTHON_REF` or `python@<ref>`
  picks another release.

### Deviations from 0.4

The full list, with reasons, is `backend/PORTING.md` section 14 (DV1 to DV50). The ones you can notice:

- a bad setting prints `engine-console: invalid setting <field>: <reason>` and exits 1, without a traceback that could
  show the value (DV1); log timestamps are UTC and there is no access log or `server: uvicorn` header (DV2);
- the node reports no Python version, and `ecnode check` names the implementation in its first line (DV5, DV24); it uses
  NVML when the driver library is present (`gpu.source` `"nvml"`; `ECNODE_NO_NVML=1` turns it off) (DV6);
- hub and node run `ssh`, `ssh-keygen`, `docker`, `nvidia-smi`, `ip` and the desktop helpers only from the fixed PATH
  above, never from the caller's (DV31, DV38);
- limits where 0.4 had none: the hub's listener (10 s to send the headers, 5 s idle keep-alive, 60 s for a request body,
  256 connections, then 503) (DV37); output of child programs and reads of files are capped, and a FIFO, socket or
  device where a regular file belongs is refused instead of read (DV32, DV33, DV39, DV40);
- route roles are enforced in the request pipeline from each route's declared access (DV36); a panic in a background
  loop is logged with secrets masked and the loop goes on (DV41); the catch-all 500 carries the security headers (DV30);
- found by an independent audit of the Rust code before the release: a node with no policy file or a refused one probes
  no port and reaches only ports a container publishes (DV44); model roots may not contain or overlap the node's and the
  login user's control files (`/etc/ecnode`, the policy's folder, `~/.ssh`, `~/.config/ecnode`,
  `~/.local/state/ecnode`, the home directory), and writes refuse secret-looking and start-up file names such as
  `.bashrc` or `.netrc` anywhere in a path (DV45; a model file such as `tokens.txt` can no longer be downloaded or copied);
  a port that only answers `/metrics` with an engine's name is no longer an engine (DV46); a path, container name, image
  or command line that is too long is refused instead of cut short (DV48);
- `api_keys.last_used_at` is written at most once a minute per key (DV49); chat replies are written off the request's
  thread (DV50); `ENGINE_CONSOLE_TOOL_PATH` and `ECNODE_TOOL_PATH` exist in debug builds only, so a release binary
  always runs its tools from the fixed PATH (DV31, DV38);

### Python bugs found during the port

Behaviour of 0.4 that 0.5 does not have:

- the node could block forever on a FIFO, or read a device, where it expected a regular file (`/proc`, `/sys` and `/etc`
  reads, model metadata, GGUF headers) (DV33);
- an error while handling one message from the hub stopped the node's reader thread; the node then read nothing more and
  ended only when its watchdog noticed the hub's silence (DV34);
- the hub followed a symlinked secrets file or bootstrap key, and could hang on a FIFO in place of a file it reads, such
  as the hub key's `.pub` (DV39);
- the tools both programs run came from whatever PATH they inherited (DV31, DV38), and their output was read without a
  limit (DV32, DV40);
- the HTTP server had no time limit for headers or request bodies and no connection cap (DV37);
- a metrics `window` such as `²m` passed `isdigit()` and then failed `int()`: a 500; it is now a 400 `invalid_window`
  (DV12);
- the catch-all 500 had none of the security headers every other answer has (DV30);
- an unconfigured or refused node probed ports 8000, 8080, 11434 and 30000 and forwarded requests to whatever answered
  (DV44);
- a writable model root that held the node's user configuration let a hub rewrite the user's policy and add a script,
  because roots were checked as exact paths and writes never checked names (DV45);
- a `/metrics` page that merely mentioned an engine made a bare port an engine, contrary to the documentation (DV46);
- an open file whose location could not be read back counted as inside the model root (DV47);
- over-long paths, container names, images and command lines were cut to their limit and acted on (DV48).

Kept as they were, because they are harmless and clients may rely on them (`backend/PORTING.md` section 15):
`PUT /settings` accepts `"default_gpu_ids": [true]` as GPU 1; `HEAD /api/v1/health` and `HEAD /` answer 405 while
`HEAD /api/v1/openapi.json` answers 200; `GET /api//v1/health` answers 401; `GET /api/v1/hosts/BAD` answers 404 while the
scripts routes answer 422 for a bad id.

### Also in 0.5.0

Fixed in the last Python code before the rewrite (tag `python-final`); the Rust hub and node keep every one.

#### Fixed

- **A signed-out tab no longer polls forever.** Views refresh on timers, and after a 401 the UI asked for a key but kept
  sending the refused requests. After a 401 the API client now holds every request made with the same key (or with none)
  and answers it locally with a 401, until the user signs in again (even with the same key) or a request succeeds;
  `/health` stays open. The same fix went into Nemo Console, where a signed-out tab was measured sending a refused
  request every 5 s.
- **Pages could break after an upgrade.** Frontend files were served without `Cache-Control`, so a browser could keep old
  ES modules for days (heuristic caching) and pair them with new pages, which then failed their import with "Something went
  wrong" (Nemo Console 0.3.0, which serves its frontend the same way, shipped broken for a user like this). Every frontend
  file is now served with `Cache-Control: no-cache` and its ETag, and a request that already holds that version gets a 304.
  A page whose code fails to load now says to reload with Ctrl+Shift+R. Covered by a backend test and the browser suite.
- Found by the new browser suite on its first runs, each with a regression test:
  - a 403 that is not about the key's role (the Hugging Face egress allowlist refusing an address, a gated model) told the
    reader "Your key's role does not allow this... use an admin key", even when the key was an admin key. `egress_denied`
    and `hf_gated` now have their own advice, and an unknown 403 says only that the console refused the request;
  - the local machine was warned about "an unrestricted SSH key". Its node runs as a subprocess, not over ssh, so it always
    reports `forced_cmd: false`; the UI now warns only for remote machines, as the hub already did.
- Settings showed "Model cache directory (read-only)" with the hub's `HF_CACHE_DIR`, which has had no effect since 0.4.0
  (models live in each machine's model folders). It now says where models live. The Downloads page's empty state pointed at
  a "Models page" that is now called Discover.

#### Added

- **Browser end-to-end suite** (`frontend/e2e/run.mjs`, `make e2e`, part of `make all`): the real backend with its real local
  node, signed in with the bootstrap key the way a person does (loopback not trusted), and a real headless Chrome driven over
  the DevTools protocol with Node 22's built-in WebSocket (no npm). Every route of `js/app.js` is opened with Beginner mode on
  and off; the run fails on an error or not-found state, the sign-in dialog, an uncaught exception, a console error, an API
  5xx, a frontend file served without `no-cache`, or a key warning on the local machine. It is hermetic, so a laptop, a GPU
  server and CI see the same thing: a throwaway home and data directory, OTLP off, Hugging Face pointed at a closed
  loopback port (Discover's "not on the HF allowlist" message is the one excused error), and a monitor-mode policy with no
  Docker binary, no port probes and two tiny seeded models (a Hugging Face cache and a plain folder). Since 0.5.0 the node
  also reads a fake `nvidia-smi` with NVML off, so the run never reads the machine's GPUs (`E2E_LIVE=1` does, read-only).
- **Ship gate:** a versioned git `pre-push` hook (`.githooks/pre-push`, enabled by `make hooks`, which `make setup` now runs)
  refuses to push a commit or tag unless the suite passed on its exact `backend/`, `frontend/` and `node/` trees
  (`scripts/e2e-gate.sh`, record kept in `.git`); when the pushed commit is the clean HEAD it runs the suite first. A Chrome
  that is not on PATH is named once per checkout with `git config engine-console.chrome <path>`.
- **CI:** an `e2e` job runs the browser suite (and, before the rewrite, a `node` job ran the Python node's tests on Python
  3.12, 3.13 and 3.14).

## 0.4.0: the fleet

One console now manages engines and models on several GPU machines, for five engines, with a cross-machine model
library and plain-language help for every setting. Read "Breaking changes" before upgrading from 0.3.

### Breaking changes

- **Five generic settings need the `ENGINE_CONSOLE_` prefix.** `HOST`, `PORT`, `ALLOWED_HOSTS`, `TRUST_LOOPBACK` and
  `DATA_DIR` are now read only as `ENGINE_CONSOLE_HOST`, `ENGINE_CONSOLE_PORT`, `ENGINE_CONSOLE_ALLOWED_HOSTS`,
  `ENGINE_CONSOLE_TRUST_LOOPBACK` and `ENGINE_CONSOLE_DATA_DIR`, in the environment and the env file alike: another
  program's exported `HOST=0.0.0.0` used to widen the bind. The bare names are ignored; the console logs one warning at
  start-up naming any that are set. Every other setting keeps its name.
- **Loopback is no longer trusted (`ENGINE_CONSOLE_TRUST_LOOPBACK=false` is the new default).** Every API call except
  `GET /api/v1/health` (and the static app shell) needs `Authorization: Bearer <key>`; that includes `GET /metrics` and
  `/api/v1/openapi.json`. The UI asks for a key. The key written on the very first start is in
  `<data dir>/bootstrap-admin.key` (default `~/.local/share/engine-console/bootstrap-admin.key`). If you upgrade an existing
  database and no longer have a key, start once with `ENGINE_CONSOLE_TRUST_LOOPBACK=true`, create one with
  `POST /api/v1/keys`, and restart without the flag. A Prometheus scrape needs a `viewer` key.
  `ENGINE_CONSOLE_TRUST_LOOPBACK=true` still works and makes `/api/v1/health` warn.
- **Viewer keys no longer reach Hugging Face.** `GET /api/v1/hf/*` is admin-only (it spends the operator's HF token), and
  a viewer's `POST /fit` judges only copies a machine already holds (`422 model_info_unavailable` otherwise).
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
audit log that masked fewer names than the secret store (L8).

Later the same review's remaining items were closed too: hub metric state keyed by node-chosen names is capped per host and
expires, with rollup retention and migration 0005 (M7); every program the node starts gets an environment built from scratch
(L1); live streams are capped per key (4 of the 8) and streamed chat replies at 16, 4 per key, with slots released however a
stream ends and the engine request stopped even when the client leaves before the first byte (L4); the generic settings need the `ENGINE_CONSOLE_` prefix (L5); the node installer sets its own `PATH` and
umask, leaves `/usr/local/lib` alone and prints the commit it installs, and the policy template prefers digest pins (L7);
CI actions are pinned by commit and the systemd template is updated for 0.4 (L9). M6 (copy provenance) stays open. A zero-context
AI review of these fixes and the desktop launcher (2026-10-15) found no High or Medium issue; its one Low finding is fixed.
docs/SECURITY.md lists what is open.

### New

- **Desktop launcher (Linux).** `make install-desktop` adds "Engine Console" to the app menu and
  `make install-desktop-icon` also puts it on the desktop (GNOME: marked trusted). It runs `engine-console open`, which
  starts the console in the background if nothing is serving, waits for `/api/v1/health` and opens the UI; its "Stop
  Engine Console" action runs `engine-console stop`, which stops only a console that `open` started. It opens a page only
  on a loopback address whose every listening socket belongs to you and answers like the console, never reads or passes a
  key, and serialises double clicks with a lock file.
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
- **Verified live on five remote machines** (2026-10-15, monitor mode only; four aarch64 GB10 machines with Python 3.12 and
  one x86_64 RTX 5090 machine with Python 3.14, all with rootful Docker): `node/install.sh` run as real root, and the hub's ssh
  links with the restricted key all `up` (`engine-console hosts check`). The desktop launcher's open, re-open and stop were run
  against a real console on a scratch port and data directory, with the browser opener stubbed.
- **Fakes and tests only:** work on remote nodes beyond the link and monitoring (the command line is checked with the real
  `ssh -G`), copies between machines
  (between two local node processes), script actions, stopping and starting external engines, multi-GPU launches,
  TensorRT-LLM, llama.cpp, Ollama and SGLang managed launches (catalogs checked against the pinned images' `--help`, no model
  started), and fit estimates and launches on GB10 unified-memory machines.
- **Not yet:** the frontend against the real backend (still being finished) and any managed (manage-mode) work on a remote
  machine.
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
