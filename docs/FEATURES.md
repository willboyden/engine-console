# Feature matrix

Status words used here. **Implemented**: the code exists in this repository (a grep of `backend/crates`, `node/install.sh` and
`frontend/js` finds it). **Partial**: some of it exists; the text says what is missing. **Not planned**: out of scope by
design. Verification is a separate word, and it says against what: **verified live** (run against real software on real
hardware), **fakes only** (covered by offline tests with fake docker, fake nodes or fake networks), **unverified** (read
from code only). "Implemented" never implies "verified live".

**Every "verified live" in this document, including the 0.4.0 record, was seen with the Python implementation (0.4.x).**
Release 0.5.0 replaced the hub and the node with Rust code; its verification is the separate record below, and none of the
live runs has been repeated with it. The "Where" column names the Rust files of 0.5.0 (module map: `backend/PORTING.md`
section 3: the Python file `backend/src/engine_console/services/x.py` became `backend/crates/ec-hub/src/services/x.rs`, and `node/src/ecnode/x.py` became
`backend/crates/ecnode/src/x.rs`).

Compared against **oMLX** and **Open WebUI** as summarised in this project's design brief (not re-checked against their
current releases). Engine Console differs in kind: it controls inference-engine containers on NVIDIA GPUs across one or several
machines, whereas oMLX serves models itself on Apple silicon and Open WebUI is a chat front end for any OpenAI-compatible
endpoint.

## Verification record for 0.5.0 (the Rust rewrite)

The hub, the node and the wire protocol were rewritten in Rust (ADR-0022). This is the record of what was run against the
Rust code before the release and of what the test suites cover.

**Parity with the last Python release (tag `python-final`), shown by differential runs:**

- the HTTP differential harness (`make parity`, `backend/parity/`) in three modes: no fleet; a real local node in
  monitor mode with fake `docker`, `nvidia-smi` and `ip` programs (the reference's Python node behind both hubs, and in a
  second run the Rust node behind both); and a fake Hugging Face behind an allowlisting egress
  proxy. 674 steps: 665 same, 9 allowed (the catch-all 500 now carries the security headers, DV30; headers only), 0
  different;
- the node protocol cross-matrix (`make proto`): a Python and a Rust driver against a Python and a Rust node, 9 scenarios
  (monitor, manage, truncated stream and six malformed-input cases). The results equal the recorded Python transcripts except
  for the documented DV5 field (`hello.host.python` is null) and the release number;
- the Python hub driving the Rust node matches the all-Python stack except for those same two node fields; the Rust hub
  driving the Python node matches exactly.

**Tests:** all 2487 tests of the last Python release are mapped: 2484 to Rust tests that exist and are compiled, 3 marked not
applicable because they test Python-only mechanics (`backend/TEST-MAP.md`, checked by `make test-map`). The Rust test suite is
hermetic (fakes, temporary directories, closed ports). `backend/crates/ecnode/tests/test_install_sh.rs` runs the installer in a
sandboxed copy. The browser end-to-end suite (`make e2e`) renders every page against the Rust hub and Rust node with fake
command-line tools.

**Not yet done with the Rust code (unverified live):**

- a run on real GPUs (NVML or `nvidia-smi`);
- a real Docker launch;
- a real remote machine over ssh, and `node/install.sh --binary` as real root on a real machine;
- an aarch64 machine: the arm64 CI job builds and tests `ecnode` on GitHub's arm64 runner, but has not run yet because
  nothing has been pushed;
- the systemd unit, which has never been started (with either implementation).

The deliberate differences from the Python release are listed in `backend/PORTING.md` section 14 (DV1 to DV50).

## Verification record for 0.4.0 (the fleet; Python implementation, kept as history)

Nothing in this section was run by the author of this document; it is the record of what the maintainers ran, with the Python
implementation, during the 0.4.0 build (all v2 commits are dated 2026-10-04) and of what the test suites cover.

**Verified live, on one real workstation with two GPUs and rootless Docker, in monitor mode except for one approved managed
test:**

- the local node over a subprocess;
- host facts, GPUs, per-process VRAM attributed to containers, cgroup memory per container, and drives including an NFS mount;
- discovery of running engines, including one script-managed engine and one external engine (both monitor-only);
- a library scan of 8 roots (32 models, about 3 TB): Hugging Face caches, GGUF files inside a Hugging Face cache, an Ollama
  store, and plain checkpoint folders, one of them in an unknown weight format;
- a real Hugging Face download through an allowlisting egress proxy into a writable root (Hugging Face cache layout);
- a managed vLLM launch on one GPU under `--cap-drop ALL`, `no-new-privileges`, a pids limit, the internal network, a
  read-only model mount and the loopback gateway, ready in about a minute;
- chat, plain and streamed (SSE), through the node;
- stop and remove of that instance, and a library delete;
- the API refusing a request without a key (401) and enforcing the CSRF header.

**Verified live on five remote machines (2026-10-15, monitor mode only):** four aarch64 GB10 unified-memory machines (Ubuntu
24.04, Python 3.12) and one x86_64 machine with an RTX 5090 (Ubuntu 26.04, Python 3.14), all with rootful Docker:

- the Python `node/install.sh` run as real root (`sudo`), writing a monitor-mode policy and passing its own `ecnode check` (the GB10
  GPU reported as unified memory over NVML);
- the hub's links to all five over ssh with the restricted key (`engine-console hosts check`: every link `up`, node 0.4.0,
  policy `monitor`), with the machines' host keys pinned.

**Verified with fakes and tests only (offline):**

- everything on a remote node beyond the link and monitoring above (launches, copies, script actions and manage mode): the
  hub's ssh command line is checked with the real `ssh -G`, key creation with a real `ssh-keygen` in a temporary directory, and
  the whole link logic against an in-process fake node that speaks the real frames;
- copies between machines (between two local node processes; an opt-in test, `ENGINE_CONSOLE_LIVE_NODE=1`, uses real
  `ecnode --stdio` subprocesses);
- script actions, and stopping and starting external engines;
- multi-GPU launches;
- TensorRT-LLM, llama.cpp, Ollama and SGLang managed launches: the catalogs were checked against the pinned images' `--help`,
  and no model was started for any of them;
- fit estimates and launches on GB10-style unified-memory machines.

**Unverified (read from code only):** the systemd unit template. In 0.4 no automated test ran the Python `node/install.sh`; its live runs are
listed above. (The 0.5.0 binary-only installer is tested in a sandboxed copy, see the 0.5.0 record.)

**Verified in a real browser (`make e2e`, every push):** every route of the frontend, with Beginner mode on and off, against the
real backend and its real local node, hermetic (monitor mode, no Docker binary, no network, two tiny seeded models). Its
first runs found and fixed two bugs: a refused Hugging Face address told an admin to "use an admin key", and the local
machine was warned about an unrestricted ssh key it does not use. Pages are rendered; actions are not clicked.

**Not yet:** any managed (manage-mode) work on a remote machine, and the frontend's actions against a manage-mode machine.

## Fleet and nodes

| Capability | Status | Verification | Where |
|---|---|---|---|
| Host list `hosts.json` (up to 32, strict schema, owner and mode checks, read at start-up) | Implemented | fakes only | `services/hosts.rs` |
| Local node as a subprocess with a whitelisted environment | Implemented | verified live | `services/hosts.rs`, `services/fleet.rs` |
| Remote nodes over ssh with the hub's own key, restricted to the node command | Implemented | fakes only (`ssh -G`, real `ssh-keygen` in a temp dir); no remote login yet | `services/hosts.rs`, ADR-0015 |
| `engine-console hosts init-key`, `check`, `list` | Implemented | fakes only | `cli.rs` |
| Link states `connecting`, `up`, `stale`, `down`, `incompatible`, `disabled`, with backoff and failures explained in words | Implemented | fakes only | `services/fleet.rs` |
| Node agent `ecnode` (GPUs, memory, drives, containers, endpoints, models; run and stop engines; files; scripts) | Implemented | local machine verified live; the rest fakes only | `backend/crates/ecnode/src/` |
| Node policy: monitor and manage mode, images, roots, port range, scripts, switches | Implemented | monitor mode live and one managed test live; every rule offline | `backend/crates/ecnode/src/policy.rs`, ADR-0016 |
| The hub shows a node's policy change | Partial | The node enforces an edited policy within ~2 s, but the hub reads a node's policy from `hello` only and sends no `node.info`, so the change appears after the host reconnects | `services/fleet.rs` |
| Node installer and uninstaller (binary-only since 0.5.0: sha256, ELF and machine checks, one-rename swap) | Implemented | fakes only (sandboxed copy of the script in `test_install_sh.rs`); the Python installer of 0.4.x ran as real root on five machines, the new one has not | `node/install.sh`, `backend/crates/ecnode/tests/test_install_sh.rs` |
| Image pulls from the hub | Partial | The node implements `docker.pull` behind `allow_pull` (fakes only); no hub code sends it, so images must be pulled on each machine | `ecnode/src/ops.rs` |
| Hub and node protocol compatibility check at `hello` | Implemented | fakes only | `backend/crates/ec-proto/`, `services/fleet.rs` |

## Monitoring

| Capability | Status | Verification | Where |
|---|---|---|---|
| Per-machine GPU stats (NVML, else `nvidia-smi`) and per-process VRAM attributed to the container | Implemented | verified live | `ecnode/src/gpus.rs`, `ecnode/src/agent.rs` |
| Unified-memory GPUs (GB10): VRAM and RAM from one pool | Implemented | fakes only | `ecnode/src/gpus.rs`, `services/fitting.rs` |
| Host RAM, shared memory and swap with alerts `host_ram_low`, `shmem_high`, `swap_heavy` | Implemented | the numbers verified live; the alert thresholds fakes only | `services/hostmem.rs` |
| Per-engine host memory from the container cgroup (anon, cache, shared, kernel; RSS fallback on cgroup v1) | Implemented | cgroup path verified live; the RSS fallback fakes only | `ecnode/src/memory.rs` |
| Drives, mounts and NFS exports per machine | Implemented | verified live (including an NFS mount) | `ecnode/src/disks.rs` |
| Engines the console did not create: classified by image and command, loopback ports probed, shown monitor-only | Implemented | verified live (script-managed and external engines) | `ecnode/src/discovery.rs` |
| Engine metrics scraped by the node, parsed by the adapter, ring buffer and rollups, per-host system series, Prometheus gauges | Implemented | fakes only | `services/metrics.rs`, `services/telemetry.rs` |
| Fit estimate against that machine's GPUs, partial offload, "fits if you stop X" | Implemented | fakes only; accuracy uncalibrated except one vLLM measurement | `services/fit.rs`, `docs/FIT-ESTIMATOR.md` |

## Engines

| Capability | Status | Verification | Where |
|---|---|---|---|
| vLLM adapter, managed launch on one GPU | Implemented | verified live | `adapters/vllm*.rs` |
| vLLM tensor-parallel launch | Implemented | fakes only | `adapters/vllm.rs` |
| SGLang adapter | Implemented | fakes only; catalog checked against the pinned image | `adapters/sglang*.rs` |
| TensorRT-LLM adapter (`trtllm-serve`, PyTorch backend) | Implemented | fakes only; catalog checked against the pinned image; live iteration statistics unverified | `adapters/trtllm*.rs` |
| llama.cpp adapter (GGUF, split sets, quants in a cache, partial offload) | Implemented | fakes only; catalog checked against the pinned image; metrics and start-up log fixtures are synthetic | `adapters/llamacpp*.rs` |
| Ollama adapter (one server, many models from an existing store, mounted read-only) | Partial | fakes only. Importing a GGUF or Hugging Face model into a store is not implemented (the launch refuses it with a clear message) | `adapters/ollama*.rs` |
| Dynamo and other OpenAI-compatible servers | Partial | Recognised by image or command (Dynamo) or by their answers (any OpenAI-compatible server) and shown monitor-only; discovery was verified live for the engines found on the workstation, which did not include Dynamo; there is no launch adapter | `ecnode/src/discovery.rs` |
| Per-engine and per-machine image pins (`image_pins`, `host_image_pins`) | Implemented | fakes only | `services/settings.rs` |
| Every launch: `--cap-drop ALL`, `no-new-privileges`, pids limit, internal network, loopback gateway, read-only model mount | Implemented | verified live for one vLLM instance on one GPU | `ecnode/src/docker.rs` |
| Preflight per machine (link, mode, image allowed and present, model present, fit, ports, GPUs, gateway image) | Implemented | live for one launch; the blocking paths fakes only | `services/lifecycle.rs` |
| Start, restart, idle TTL, pin, remove, logs of managed instances | Implemented | stop and remove verified live; the rest fakes only | `services/lifecycle.rs` |
| Adoption of labelled containers when a link comes up | Implemented | fakes only | `services/lifecycle.rs` |

## Library, downloads and copies

| Capability | Status | Verification | Where |
|---|---|---|---|
| Model library across machines, grouped by model, per root kind (`hf-home`, `hf-hub`, `gguf`, `plain`, `ollama`, `other`, `auto`) | Implemented | verified live (8 roots, 32 models) | `ecnode/src/models.rs`, `services/library.rs` |
| Last-known library kept in SQLite for a machine that is down | Implemented | fakes only | `services/library.rs` |
| Shared storage: an NFS mount of another configured machine's disk lists the bytes once | Implemented | an NFS mount seen live; the folding between two machines fakes only | `services/library.rs`, `services/fleet_status.rs` |
| Library delete with the in-use rule | Implemented | verified live (one delete) | `services/library.rs`, `services/lifecycle.rs` |
| Hugging Face download into any writable root of any machine (cache layout, `allow_patterns` such as one GGUF quant) | Implemented | verified live (one real download through the proxy into a writable root) | `services/downloads.rs` |
| Resume, pause, cancel, queue, disk pre-check, sha256 where Hugging Face supplies one | Implemented | fakes only | `services/downloads.rs`, `ecnode/src/fsops.rs` |
| Copies between machines or drives | Partial | Hugging Face cache folders, loose GGUF files and plain directories whose scan counted every file can be copied; an Ollama store or an `other` folder cannot (`copy_unsupported`, the node protocol has no directory listing). Fakes only (two local node processes) | `services/downloads.rs` |

## Script-managed and external engines

| Capability | Status | Verification | Where |
|---|---|---|---|
| Scripts registered in a node policy, shown as instances, start and stop through `up` and `down`, restart as one job | Implemented | discovery of a script-managed engine verified live; running actions fakes only | `services/scripts.rs`, `ecnode/src/scripts.rs`, ADR-0019 |
| Script output streaming, one run per script, timeout, whole-process-group kill | Implemented | fakes only | `ecnode/src/scripts.rs` |
| `external_control: "stop_start"` for engines the console did not create | Implemented | fakes only | `ecnode/src/ops.rs` |

## Learn mode, security and the UI

| Capability | Status | Verification | Where |
|---|---|---|---|
| Plain-language explanations for parameters, presets, metrics, states, links, checks and terms, served at `GET /api/v1/learn`, with a build-failing coverage gate | Implemented | the gate runs offline; whether each text is true for the pinned engine versions is unverified | `backend/crates/ec-hub/learn/`, `tests/test_learn_content.rs`, ADR-0020 |
| Info buttons, Beginner mode, Glossary page | Implemented | rendered against the real backend in headless Chrome by `make e2e`, Beginner mode on and off; the buttons are not clicked | `frontend/` |
| API key required, loopback not trusted by default; CSRF header; Host and Origin checks | Implemented | 401 without a key and the CSRF header verified live; Host and Origin unverified against a real browser | `api/security.rs`, ADR-0021 |
| Security headers, body and SSE caps, audit of mutating calls | Implemented | fakes only | `api/security.rs` |
| Console egress through an allowlisting proxy, fail closed | Implemented | one real download through a proxy verified live; failure paths fakes only | `services/hf_http.rs`, ADR-0012 |
| Console self-observability (OTLP, `/metrics`, JSON logs) | Implemented | fakes only | `services/telemetry.rs` |
| All views of the frontend | Implemented | mock server: frontend unit tests and headless screenshots of every route at desktop and phone widths; real backend: every route rendered in headless Chrome by `make e2e` (monitor mode, seeded models); actions not clicked | `frontend/js/views/` |
| Frontend files revalidated on every load (`Cache-Control: no-cache`, ETag, 304 while unchanged), so an upgrade never pairs a new page with an old module | Implemented | backend test; `make e2e` checks the header on every file a real browser loads | `api/statics.rs` |

## Versus oMLX

| oMLX capability | Engine Console | Status | Evidence / gap |
|---|---|---|---|
| Model downloader | HF search with filters, model card, resumable queued downloads to any machine and drive, pause/resume/cancel, sha256 check where Hugging Face supplies one, disk pre-check | Implemented; the backend download verified live, the view not verified against the real backend | `services/hf.rs`, `services/downloads.rs`, `frontend/js/views/discover.js`, `downloads.js` |
| Per-model settings | Profiles may carry a `repo_id`, and the launch form applies profile params; no separate persistent per-model settings page that auto-applies on every launch | Partial | `services/profiles.rs` |
| Profiles | Named param bundles, engine presets, YAML import/export, diff, copy-as-command | Implemented (fakes only) | `services/profiles.rs`, `/profiles/*` |
| Benchmark | `quick` and `standard` suites: TTFT, decode throughput at concurrency 1/4/16/64, optional prefix-cache probe, stored and comparable, run through the node | Implemented (fakes only; no real benchmark run recorded) | `services/bench.rs` |
| Hot/cold KV cache (tiered) | Only reads engine KV usage and prefix-hit metrics | Not planned | Engine-side feature |
| Multi-model LRU / pinning / TTL | Several instances per machine on distinct GPUs and ports, pin, idle TTL auto-stop; **no LRU eviction** or on-demand swap (Ollama swaps models inside its own server) | Partial | `services/lifecycle.rs` |
| Usage heatmaps | Hourly heatmap, per-model and per-day totals, CSV export | Implemented (fakes only) | `services/usage.rs` |
| Chat | Streaming chat through the node to any machine's engine, history, prompt library, image upload, reasoning fold, export | Implemented; plain and SSE chat verified live; image upload against the mock only | `services/chat.rs` |
| API keys | Create/revoke, hashed at rest, admin and viewer roles | Implemented | `services/settings.rs` |
| 8 languages | i18n mechanism (flat keys, plurals) but only `en` ships | Partial | `frontend/i18n/en.json` only |
| Vendored offline dependencies | Frontend has zero third-party JS and no CDN references; backend and node dependencies are pinned by `Cargo.lock` and fetched from crates.io at build time (`--locked`, checked by cargo-deny and cargo-audit, 14-day cooldown), not vendored; nothing is fetched at run time | Partial | `backend/Cargo.lock`, `backend/deny.toml` |

## Versus Open WebUI

| Open WebUI capability | Engine Console | Status | Evidence / gap |
|---|---|---|---|
| Admin panel | Settings (defaults, image pins per machine, token status), keys, audit log; each host's node policy is readable (read-only) through the API | Partial | no user management beyond keys |
| Custom model presets | Engine presets and saved profiles change *engine launch* parameters; no chat-level "custom models" | Partial | `adapters/*.rs` presets, `services/profiles.rs` |
| Arena / Elo evaluation | Side-by-side matches, blind mode, votes, Elo leaderboard | Implemented (view against the mock only) | `/arena/*` |
| Prompt library | CRUD prompts, picker in chat | Implemented | `/prompts` |
| RAG / knowledge bases | None | Not planned | Out of scope; use a chat front end in front of the engines |
| Tools / function-calling UI, MCP | None | Not planned | Out of scope |
| Web search | None (it would need new egress) | Not planned | Conflicts with deny-by-default egress |
| RBAC | Two roles (`admin`, `viewer`); no groups or per-model permissions | Partial | `api/security.rs` |

## Other capabilities

| Capability | Status | Verification | Where |
|---|---|---|---|
| Access from another container to an engine (`internal_endpoint`, opt-in per machine) | Implemented | not exercised | OPERATIONS.md |
| Secret engine params kept out of the database and API | Implemented | fakes only | `services/secrets_store.rs` |
| Browser end-to-end tests | Implemented | `make e2e`: every route, Beginner mode on and off, real backend and node, hermetic; run by CI and required by the `pre-push` hook; renders pages, does not click through actions | `frontend/e2e/`, `scripts/e2e-gate.sh` |
