# Fleet protocol (ecnode wire protocol v1) and node policy

Status: **LOCKED contract** (contract v2). The code twin is the `ec-proto` crate (`backend/crates/ec-proto/src/`; until 0.4.x it was `proto.py`): framing, limits, every
message and argument shape, and the whitelist cleaners. This page explains what each message *means* and what a
node must check before acting. Changes are additive only and are announced in `docs/ARCHITECTURE.md` §10. Protocol version 1 is
unchanged by the Rust rewrite (0.5.0): a hub of either implementation drives a node of either (exercised by the protocol
cross-matrix and both mixed stacks, `make proto`).

## 1. Pieces

```
Engine Console (hub, one machine)                          each machine (including the hub's own)
  FleetService ── NodeLink "local" ── subprocess ─────────▶ ecnode --stdio   (as the login user)
               └─ NodeLink "<host>" ─ ssh, restricted key ─▶ ecnode --stdio ──▶ docker CLI, /proc, /sys,
                                                                                NVML / nvidia-smi, model roots
```

* One **node** (`ecnode`) per machine: a single Rust binary, no Python or other runtime on the machine. It reads NVML
  (`libnvidia-ml.so.1`, loaded at run time) when the driver library is present, else `nvidia-smi`.
* The **hub** is Engine Console. It starts every node itself (subprocess or ssh) and talks to it over the node's
  stdin/stdout. Nothing listens on the network anywhere.
* A node runs docker, `nvidia-smi`, `ip` and any other program given by bare name only from the fixed path
  `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`; a policy may name an absolute `docker.binary` instead.
* A node obeys its **local policy file** (§6) before anything the hub asks. A hub can read the policy (it is in
  `hello`), never change it.

## 2. Stream, frames, requests

See the header comment of `backend/crates/ec-proto/src/lib.rs` for the exact bytes. In short:

* Node writes `MAGIC` (`ecnode-proto 1\n`) first; the hub skips up to 64 KiB of shell noise before it.
* Frames: `u32 length | u8 kind | body`. JSON frames ≤ 1 MiB; DATA frames carry `u32 stream id` + ≤ 1 MiB of bytes.
* `req` → zero or more `event` / DATA → exactly one `res`. `cancel` stops early (still one `res`).
* Uploads: hub sends DATA on the request's stream, then `end`.
* Flow control: every stream starts with `WINDOW` (8 MiB) of credit in each direction; `credit` grants more.
  JSON frames are never credit-limited.
* Liveness: `ping`/`pong` every 10 s; either side drops the link after 30 s of silence. The node exits on stdin
  EOF, on 30 s without a hub message, or when a write to stdout is blocked for 10 s (an orphaned node must not
  poll forever).
* At most `MAX_INFLIGHT` (64) requests run at once on a node; more get `busy`.

## 3. Pushed messages (node → hub)

| type | when | content (see `ec-proto`'s `specs.rs` for fields) |
|---|---|---|
| `hello` | first frame | protocol + node version, `node_id` (sha256 of `/etc/machine-id`, 16 hex), host facts (hostname, arch, OS, kernel, CPU, RAM; `python` is null since 0.5.0, the Rust node has none), GPU driver/CUDA, static GPUs (uuid, name, total memory, compute capability, unified memory), docker facts (available, context, rootless, version, nvidia runtime), interface addresses, **policy summary**, `forced_cmd` (sshd set `SSH_ORIGINAL_COMMAND`), `via_ssh` |
| `sample` | every 2 s | per-GPU live stats keyed by UUID (only UUIDs from `hello` are kept by the hub) incl. per-process VRAM with the **container name** each process belongs to (pid → `/proc/<pid>/cgroup` → container id → name); host memory from `/proc/meminfo`; CPU load/util; cgroup v2 memory of every **engine** container (`memory.current` + `memory.stat`: anon, file, shmem, kernel) |
| `disks` | every 30 s and at start | mounts relevant to the node: `/`, every model root's filesystem, network mounts. Size/used/free via `statvfs`; fstype and source from `/proc/self/mountinfo`; block-device model from `/sys/block/<dev>/device/model`; NFS/CIFS exports split into `remote_host`/`remote_path`; `roots` lists the model roots on each mount |
| `inventory` | on change, ≥ every 30 s | running and console-owned containers (sanitized `docker inspect`: **argv redacted on the node**, env never read), engine classification by signature, GPU devices, mounts (with the model root they fall in), compose project/service labels; probed engine endpoints (`/v1/models`, `/health`, `/metrics` on loopback ports: published ports of engine containers plus `discovery_ports`; with no usable policy, published ports only, §6); script status (§6.3) |
| `metrics` | per engine endpoint, every `interval_s` | raw Prometheus text, filtered on the node to the metric-name prefixes the hub configured with `metrics.configure`, ≤ 512 KiB |
| `models_changed` | after a scan finds a difference | new `generation`; the hub then pages through `models.list` |
| `log`, `fatal`, `pong`, `credit` | as needed | `fatal` precedes an exit (e.g. the node could not start; a bad policy is NOT fatal: it means monitor mode) |

## 4. Operations

Every op's argument and result shapes are in `ec-proto`'s `specs.rs` (the argument specs, `RESULTS` and `EVENTS`). **M** = allowed in
monitor mode (`MONITOR_OPS` in the same file); everything else needs policy `"mode": "manage"` and answers `monitor_only`
otherwise.

| op | M | what the node does, and must check |
|---|---|---|
| `node.info` | M | fresh `hello` content (re-reads the policy file) |
| `metrics.configure` | M | sets scrape interval (clamped 1–60 s) and per-engine metric-name prefixes |
| `docker.render` | M | the exact `docker run` argv it *would* run for a spec, secret values omitted. Same validation as `docker.run` |
| `docker.run` | | validate the spec (§5), write `files` into `~/.local/state/ecnode/instances/<instance>/files/` (0700) and mount each read-only, then `docker --context <policy> run -d …`. Always: `--cap-drop ALL`, `--security-opt no-new-privileges:true`, `--pids-limit 4096`, `--restart no`, the policy engine network (created `--internal` if missing, refused if it exists and is not internal), label `engine-console=1` plus `engine-console.*` labels from the spec |
| `docker.stop` / `docker.start` / `docker.rm` | | only containers labelled `engine-console=1` (and, with `role`/`instance`, exactly that role and instance): `not_owned` otherwise |
| `docker.inspect` | M | sanitized view of one container (same shape as inventory) |
| `docker.logs`, `docker.logs_follow` | M | **console-owned containers only** (logs of other containers can hold secrets). Tail clamped 1–5000; follow streams `{"line"}` events until cancelled |
| `docker.image_present`, `docker.images` | M | local image facts |
| `docker.pull` | | policy `allow_pull` and the image matches `engine_images`; streams progress lines |
| `docker.external` | | policy `external_control == "stop_start"`; target is a running/stopped container classified as an engine, **not** console-owned and **not** matched by any script's `containers` patterns; action `stop` or `start` only |
| `net.ensure_internal` | | create/verify the policy engine network (`--internal`) |
| `gateway.start` | | nginx stream gateway from the policy `gateway_image` (exact match), on the default bridge, `-p 127.0.0.1:<host_port>:8080`, read-only, cap-drop ALL, user 101, then connected to the engine network. `host_port` inside `port_range` and free (`port_in_use` otherwise). `engine_host` must be a console-owned container name |
| `http.request` | M | HTTP/1.1 to `127.0.0.1:<port>` **only** for a port that is a console-owned gateway's published port or an engine endpoint in the current inventory; methods `GET`/`POST`; path must start with one of `/v1/`, `/health`, `/metrics`, `/version`, `/api/` (Ollama), `/get_model_info`, `/get_server_info`, `/server_info`; hop-by-hop and `Host` headers dropped. In monitor mode `POST` is forwarded only to inference paths (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, Ollama `/api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings`, `/api/show`); anything else answers `policy_denied`. A bare (non-container) endpoint is an engine only when its `/v1/models` answers a JSON object with a `data` list, or 401/403 while `/metrics` names an engine (a `/metrics` marker alone names the engine, it never qualifies a port). With no usable policy only ports a running container publishes on loopback are reachable. Request body (`body_len` > 0) arrives as upload DATA + `end`. Response: one event `{status, headers}`, the body as DATA, res `{status, bytes}`. Never follows redirects. Default timeout 600 s (streams can be long) |
| `fs.stat` | M | for each path (relative to the root): exists, is_dir, size, and the size of `<path>.incomplete` (`partial`) |
| `fs.write` | | upload into `<root>/<rel>.incomplete` (writable root only), starting at `offset`, which must equal the current partial size (or 0, which truncates). On `end`: size must equal `size`; if `sha256` is given the node hashes the **whole** file and compares; then `os.replace` to `<rel>`. `no_space` when the filesystem would drop below 1 GiB free. Mismatch: the partial is deleted and `hash_mismatch` / `size_mismatch` returned |
| `fs.read` | M | stream `<root>/<rel>` from `offset` (for machine-to-machine copies) |
| `fs.symlink` | | create `<root>/<link>` → relative `target`; `..` only as leading segments (`../../blobs/<hash>`), and the realpath of the joined target must be inside the same root (HF snapshot links) |
| `fs.put_small` | | write ≤ 96 KiB (base64) atomically to `<root>/<rel>` (HF `refs/main`, small metadata) |
| `fs.delete` | | policy `allow_delete`; refuse `in_use` when a running container mounts a path at or under the target; refuse the root itself; the parent and the target are opened once and removed through those fds (a path re-pointed during the in-use check answers `conflict`, never follows the link); returns bytes freed |
| `models.scan` | M | rescan roots in the background (bounded: ≤ 50 000 directory entries and ≤ 120 s per root); answers at once with the generation it will produce |
| `models.list` | M | page through the latest scan (`cursor`, `limit` ≤ 200) |
| `models.config` | M | the `config.json` of one model (bounded JSON) |
| `scripts.run` | | a script action from the policy (§6.3) by name; `arg` must be one of that action's `args`. argv only, never a shell; streams stdout/stderr lines; one run per script at a time (`busy` otherwise) |

### Path rules (every `root` + `rel`)

`root` is a model root id from the policy. `rel` is relative: no leading `/`, no `\`, no NUL, no empty, `.` or
`..` segment. The node resolves the **parent directory** with `realpath` and requires it to be inside the root's
own `realpath`; files are opened with `O_NOFOLLOW`. Writes only into roots with `"writable": true`. Never read
anything that is not a model file: the scanner and `fs.read` skip names that look like keys or secrets
(`id_*`, `*.pem`, `*.key`, `.env*`, `token*`, `credentials*`) even inside a root.

## 5. Run spec validation (node side, `docker.run` / `docker.render`)

| field | rule |
|---|---|
| `name` | `^ec-[a-z0-9][a-z0-9-]{0,58}$` |
| `instance` | `^[A-Za-z0-9_-]{1,64}$` |
| `image` | matches a policy `engine_images` entry: exact reference, or `repository:*` / `repository@sha256:*` with a literal repository |
| `argv` | ≤ 512 items, each ≤ 4096 chars, no NUL; passed as separate arguments after the image |
| `container_port` | 1–65535 |
| `gpu_uuids` | each one a GPU this node reported in `hello` |
| `network_alias` | same pattern as `name` |
| `labels` | keys start with `engine-console.`; the node adds `engine-console=1` itself |
| `env`, `secret_env` | names `^[A-Za-z_][A-Za-z0-9_]{0,127}$`; `secret_env` values reach docker only through the child process environment (`-e NAME`), never argv, never logged. `secret_env` names must contain key, token, secret or password (any case; else `bad_args`) and must not be one that changes the docker CLI (`LD_*`, `DOCKER_*`, `OTEL_*`, `PATH`, ...; `policy_denied`); the docker child's environment is otherwise built from scratch |
| `mounts` | `root` is a policy root, `rel` passes the path rules, `dest` is an absolute container path without `..`; always read-only (`ro: false` answers `policy_denied`); refused when the source, or a folder directly in it, holds a key- or token-looking name (an HF_HOME's `token`: mount an `hf-hub` root on `hub/` instead) |
| `volumes` | names `^ec-[a-z0-9][a-z0-9_.-]{0,60}$` (compile caches), absolute `dest` |
| `shm_size` | `^[0-9]{1,4}[kmg]$`, at most policy `max_shm_gib` |
| `ipc_host` | only if policy `allow_ipc_host` (multi-GPU NCCL needs it) |
| `files` | ≤ 4 files, ≤ 128 KiB each, absolute container paths, mounted read-only |
| `ulimits` | (additive, v1.1) at most 4; names `stack` and `nofile` only, values `-1`, a number or `soft:hard`. `memlock` is refused: unlimited locked memory lets one container pin host RAM, and it only matters for RDMA, which will get its own policy switch |

Metrics of slow engines (additive, v1.1): engines whose `/metrics` is slow and drains a statistics queue on every
read (TensorRT-LLM's iteration stats) are never probed on `/metrics` by discovery (`metrics_status` stays null); the
scraper reads them once per interval when `health_status` is 200, with a 10 s read timeout.

## 6. Node policy file

Location: `--policy PATH`, else `/etc/ecnode/policy.json` if it exists, else `~/.config/ecnode/policy.json`.
Installed copies are root-owned, mode 0644: the console's ssh key runs as the login user and cannot edit it.
The node refuses a policy file that is group/world-writable, or (for `/etc`) not owned by root, and then runs in
**monitor mode** with `policy.error` saying why. No file at all also means monitor mode. Without a usable policy the
node also probes no discovery port (since 0.5.0; DV44). A root that is, contains or lies inside `/etc/ecnode`, the policy
file's folder, `~/.ssh`, `~/.config/ecnode` or `~/.local/state/ecnode`, or that contains the home directory, makes the
policy invalid (DV45).

```json
{
  "version": 1,
  "mode": "monitor",
  "docker": {"binary": "docker", "context": null},
  "engine_images": ["vllm/vllm-openai:*", "lmsysorg/sglang:*", "nvcr.io/nvidia/tensorrt-llm/release:*",
                    "ghcr.io/ggml-org/llama.cpp@sha256:*", "ollama/ollama:*"],
  "gateway_image": "nginx@sha256:<digest>",
  "engine_network": "engine-console-engines",
  "port_range": [18000, 18099],
  "discovery_ports": [8000, 8080, 11434, 30000],
  "model_roots": [
    {"id": "hf", "path": "/data/hf", "kind": "hf-home", "writable": true, "label": "HF cache"},
    {"id": "gguf", "path": "/data/gguf", "kind": "gguf", "writable": true, "label": "GGUF files"},
    {"id": "shared", "path": "/mnt/shared-models", "kind": "auto", "writable": false, "label": "Shared (NFS)"}
  ],
  "scripts": [],
  "allow_pull": false,
  "allow_delete": false,
  "allow_ipc_host": true,
  "external_control": "none",
  "max_shm_gib": 64
}
```

### 6.1 Root kinds

| kind | layout | scanned as |
|---|---|---|
| `hf-home` | `HF_HOME`: `<root>/hub/models--org--name/{blobs,snapshots,refs}` | `hf` (GGUF-only repos as `gguf`) |
| `hf-hub` | hub cache itself: `<root>/models--org--name/…` | same |
| `gguf` | `.gguf` files anywhere below (depth ≤ 6); split sets `*-00001-of-0000N.gguf` grouped | `gguf` |
| `plain` | directories holding `config.json` + weights (depth ≤ 3) | `plain` |
| `ollama` | an Ollama home or its `models/` dir (`manifests/`, `blobs/`) | `ollama`: name:tag from the manifest path, size from layers, facts from the config blob |
| `other` | anything | `other`: one entry per top-level directory, size only |
| `auto` | detect per root: `hub/` → hf-home, `models--*` → hf-hub, `models/manifests` or `manifests` → ollama, else plain + gguf | as detected |

### 6.2 Modes

`monitor`: samples, disks, inventory, metrics, model scans, engine HTTP (GET, plus POST to inference paths only: chat
and benchmarks through the console), reading model files for copies. Nothing on the machine changes. `manage`: everything else the policy allows.
The installer writes a monitor-mode starter policy; the operator switches to manage after reading it.

### 6.3 Scripts (engines the operator already starts with their own scripts)

```json
{"name": "big-model", "label": "Big model (2 GPUs)", "engine": "sglang",
 "description": "Starts before the small model; the script handles the order.",
 "containers": ["myproj-sglang-*"],
 "status_action": "status",
 "timeout_s": 900,
 "actions": [
   {"name": "up", "label": "Start", "argv": ["/opt/lab/model.sh", "up"], "confirm": true},
   {"name": "down", "label": "Stop", "argv": ["/opt/lab/model.sh", "down"], "confirm": true, "danger": true},
   {"name": "status", "label": "Check status", "argv": ["/opt/lab/model.sh", "status"]},
   {"name": "switch", "label": "Switch model", "argv": ["/opt/lab/use-model.sh"], "args": ["model-a", "model-b"],
    "confirm": true}
 ]}
```

* `argv` stays on the node (the hub only sees names, labels, allowed `args`). The chosen `arg`, if any, is
  appended as one argument. Executables must be absolute paths.
* Environment: `PATH` (absolute entries only), `HOME`, `USER`, `LOGNAME`, `LANG`, `XDG_RUNTIME_DIR`, `DOCKER_HOST`, `DOCKER_CONTEXT`
  from the node's own environment; nothing from the hub.
* Status: `up` when a running container matches one of `containers` (shell-style patterns), else `down`.
  `status_action` runs only when asked (the UI's "Check status").
* Containers matching a script's patterns show as `management: "script"`; `docker.external` refuses them.

## 7. Errors

`res.error.code` values are the constants in `backend/crates/ec-proto/src/lib.rs` (`bad_args`, `unknown_op`, `monitor_only`,
`policy_denied`, `not_found`, `not_owned`, `conflict`, `in_use`, `port_in_use`, `no_space`, `hash_mismatch`,
`size_mismatch`, `io_error`, `docker_error`, `http_error`, `timeout`, `cancelled`, `busy`, `unsupported`,
`internal`). `message` is a sentence a person can act on, never a stack trace, never a secret value.

## 8. Install (per machine; opt-in, needs the operator's sudo)

`node/install.sh --binary ecnode-<version>-<arch>` (or `--uninstall`): checks a copy of the binary (sha256 from the
`.sha256` file next to it, ELF header and machine type, `ecnode version` runs), swaps `/usr/local/lib/ecnode/` in with one
rename (root-owned; the binary is `/usr/local/lib/ecnode/ecnode`), writes `/etc/ecnode/policy.json` (monitor mode, detected
roots and docker context) only if absent, and finishes with `ecnode check` (what this machine would report). Details:
`node/README.md`. The hub prints the
authorized_keys line for its key:

```
restrict,from="<hub addresses>",command="/usr/local/lib/ecnode/ecnode --stdio" ssh-ed25519 AAAA… engine-console-hub
```
