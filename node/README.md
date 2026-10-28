# ecnode: the Engine Console node agent

`ecnode` runs on every machine of a local GPU cluster. Engine Console (the hub) starts it, as a local subprocess
on its own machine or over ssh on the others, and talks to it on its stdin/stdout with the framed protocol in
[`docs/FLEET-PROTOCOL.md`](../docs/FLEET-PROTOCOL.md) (code twin: the `ec-proto` crate, `backend/crates/ec-proto/src/`).
Nothing listens on the network. Since 0.5.0 the node is a Rust program: the source is `backend/crates/ecnode/` and this
folder holds only the installer (`install.sh`), this file and example policies (`examples/`).

The node:

* **reports** the machine: GPUs (NVML when the driver's `libnvidia-ml.so.1` can be loaded, else `nvidia-smi`; `ECNODE_NO_NVML=1`
  switches NVML off; zero GPUs and unified-memory GPUs are fine), per-process VRAM mapped to the container each process belongs to, host RAM and
  CPU, cgroup v2 memory of every engine container, disks and mounts (NFS/CIFS split into host and export),
  docker containers (sanitized: argv redacted on the node, environment never read), engine endpoints found by
  loopback probes, Prometheus metrics filtered to the families the hub asks for, and the models in each model
  root (Hugging Face caches, GGUF files, plain model directories, Ollama stores);
* **does** what the hub asks within its local policy: runs and removes engine containers and their loopback
  gateways, pulls allowed images, moves model files in and out of model roots, runs the operator's own
  lifecycle scripts, and forwards engine HTTP calls (chat, benchmarks, health) to loopback ports;
* **obeys its local policy file first**. The hub can read the policy (it is in `hello`), never change it.

One program per architecture (x86_64 and aarch64, dynamically linked against glibc only, 2.35 or
newer: Ubuntu 22.04 and 24.04, DGX OS 7). It needs no Python, compiler or network on the machine. It is not a musl build,
because a static binary could not load the NVIDIA driver's NVML library. The last Python node stays available at the tag
`python-final` for rollback (see "Rollback").

## Install (per machine, needs the operator's sudo)

Get the node binary for the machine's architecture and its checksum file, side by side:

* a release asset: `ecnode-<version>-<arch>` and `ecnode-<version>-<arch>.sha256` (`<arch>` is `x86_64` or `aarch64`). CI's
  release job on a `v*` tag builds them on an x86_64 and an arm64 runner and attaches them to a **draft** GitHub release
  that the repository owner publishes; or
* `make node-dist` on a machine of that architecture, which writes both files to `backend/target/dist/`.

```
sudo node/install.sh --binary ecnode-<version>-<arch>   # binary -> /usr/local/lib/ecnode/ecnode (root-owned), policy -> /etc/ecnode/policy.json
sudo node/install.sh --uninstall                        # removes /usr/local/lib/ecnode, keeps /etc/ecnode
```

What `install.sh --binary FILE` does, in order:

1. refuses a symlink or a file that is not a regular file;
2. copies FILE into a staging directory next to `/usr/local/lib/ecnode` and checks **the copy**: its sha256 must equal the
   digest in `FILE.sha256` (`sha256sum` format), it must be a 64-bit little-endian ELF executable (EXEC or PIE) whose machine
   matches `uname -m` (x86_64 or aarch64), and `ecnode version` must run;
3. swaps the whole `/usr/local/lib/ecnode` prefix in with one rename (an old Python install's `lib/` and launcher go with
   the old prefix, so a failed install leaves the old one);
4. writes `/etc/ecnode/policy.json` from `ecnode policy-template` (run as the invoking user, so the node validates its own
   template; root:root, 0644, monitor mode) **only if it does not exist**, and never overwrites it;
5. runs `ecnode check --policy /etc/ecnode/policy.json` as the invoking user.

It is safe to re-run; it never edits ssh or sudo configuration and starts nothing. It runs with a fixed `PATH` and
`umask 022`, never changes the owner or mode of `/usr/local/lib` (it refuses one that anyone but root can write to), and
needs only bash and coreutils. Honest limit: the sha256 check catches a corrupted or wrong-architecture file, not a malicious
release, because the binary and its `.sha256` come from the same release page. A test
(`backend/crates/ecnode/tests/test_install_sh.rs`) runs the script in a sandboxed copy (paths in a temp dir, the root-only
steps dropped): good install, re-run over an old prefix, policy kept, bad or missing checksum, wrong architecture, not ELF,
32-bit, symlink, directory, `--uninstall` and argument errors. The script has not been run as real root on a real machine
with this binary-only version.

**Docker outside the fixed PATH.** The node runs `docker`, `nvidia-smi` and `ip` only from
`/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`; it never searches the caller's `PATH`. As the node's user run
`command -v docker`; if the path is outside that list, set `"docker": {"binary": "<that absolute path>"}` in
`/etc/ecnode/policy.json` before connecting the hub.

### Rollback

In a checkout of the tag `python-final` (the last Python code; `v0.4.0` also works), run that checkout's
`sudo node/install.sh` (the Python installer; needs the machine's system `python3` 3.10 or newer). It puts the Python node
back at the same path. The key line, the policy and the hub need no change.

Then, on the hub, `engine-console hosts init-key --from <hub addresses>` prints the key line to add to the login
user's authorized keys on this machine. The line restricts the hub's key to exactly the node command:

```
restrict,from="10.0.0.2",command="/usr/local/lib/ecnode/ecnode --stdio" ssh-ed25519 AAAA... engine-console-hub
```

## Commands

```
ecnode --stdio [--policy PATH]   speak the hub protocol on stdin/stdout (what the hub starts)
ecnode check [--policy PATH]     print what this machine would report (read-only)
ecnode policy-template           print a starter monitor-mode policy for this machine (writes nothing)
ecnode version
```

`check`'s first line is `ecnode <version> (rust) · protocol 1 · node id <id>`. It prints host, GPUs, docker, the policy (and why it was refused, if it was), each model root with the
filesystem it lives on and its free space, and a count of containers and endpoints. It never prints an
environment value or a container's command line. It exits 1 when a policy file exists but was refused.

## Policy

Full reference: [`docs/FLEET-PROTOCOL.md` section 6](../docs/FLEET-PROTOCOL.md). Examples:
[`examples/policy.monitor.json`](examples/policy.monitor.json), [`examples/policy.manage.json`](examples/policy.manage.json).

* **Lookup:** `--policy PATH`, else `/etc/ecnode/policy.json` if it exists, else `~/.config/ecnode/policy.json`.
* **Refused files run the node in monitor mode**, with `policy.error` saying why: a symlink or non-regular file,
  a group- or world-writable file, a file owned by anyone but root or the node's user, a file under `/etc` not
  owned by root (or in a directory there that is not root's alone), a file in a world-writable directory, more
  than 256 KiB, invalid JSON, or anything the schema rejects. Unknown keys are errors (a typo in
  `"allow_delete"` must not be silently ignored); keys starting with `_` are comments. No file at all also means
  monitor mode. The node re-checks the file every couple of seconds, so an edit takes effect without a restart
  (the hub sees the new summary on its next `node.info`).
* **Docker context:** `docker.context` names the context every docker call uses. When it is `null`, the CLI's
  own choice applies, with one exception for rootless-only machines: a node started with a scrubbed environment
  (the hub's ssh session has no `DOCKER_HOST`) whose current context is `default` and who cannot use the default
  socket uses an existing `rootless` context. Naming the context in the policy avoids the guesswork;
  `ecnode policy-template` does.
* **Images:** an exact reference, `repository:*` (any tag of exactly that repository) or `repository@sha256:*`.
  The repository is compared literally: `vllm/vllm-openai:*` allows neither `evil/vllm/vllm-openai:x` nor
  `vllm/vllm-openai-evil:x` nor `docker.io/vllm/vllm-openai:x`.
* **Model roots** are directories that hold only models. `/`, `/etc`, `/home`, the user's home directory itself
  and similar are refused, and so is a directory that contains the home directory or that is, contains or lies
  inside `/etc/ecnode`, the policy file's directory, `~/.ssh`, `~/.config/ecnode` or `~/.local/state/ecnode`; paths
  with `:` or `,` are refused (they would break docker mount syntax). Writes never create or replace a file with a
  secret-looking or write-protected name (shell start-up files, `.netrc`, `.git-credentials`, `authorized_keys`, ...). Prefer the
  HF hub cache (`.../huggingface/hub`, kind `hf-hub`) and an Ollama `models/` directory over their parent
  directories, so token files, keys and prompt history sit outside every root (the template does this).

### Monitor vs manage

`monitor` (the default and the template): samples, disks, inventory, metrics, model scans, engine HTTP through
the console (chat and benchmarks), and reading model files for machine-to-machine copies. Nothing on the machine
changes; every other op answers `monitor_only`. `manage`: everything else the policy allows (`allow_pull`,
`allow_delete`, `allow_ipc_host`, `external_control`, writable roots, scripts). Switch by editing the file as
root after reading it.

## Security model

* **The hub is the operator, the policy is the boundary.** Arguments reach handlers only through
  `proto.clean_args`; policy checks come after. A hub cannot widen the policy.
* **Containers** are started with `--cap-drop ALL`, `--security-opt no-new-privileges:true`,
  `--pids-limit 4096`, `--restart no`, `--pull never` and only on the policy's engine network, which must be
  `--internal`. Only containers labelled `engine-console=1` are stopped, started or removed (with role/instance
  checked when given), acting on the verified container id. The per-instance nginx gateway is read-only,
  cap-drop ALL, user 101, publishes only `127.0.0.1:<port in port_range>`, and its config holds only a checked
  container name and port.
* **Secrets:** `secret_env` values reach docker only through the docker CLI's own environment (`-e NAME` on the
  command line), never argv, logs or results. A `secret_env` name must look secret (key, token, secret or password),
  and names that would change how the docker CLI itself runs (`LD_*`, `DOCKER_*`, `OTEL_*`, `PATH`, proxies...) are
  refused. `docker.render` shows the same argv, so it never holds a
  value either. Container argv is redacted on the node (`--api-key X`, `--token=X`, `NAME=X`, inside `sh -c`
  strings too); `Config.Env` is never read; logs, script output and error text are scrubbed of secret-looking
  values. Logs are available for console-owned containers only.
* **Child environments:** every program the node starts (docker, nvidia-smi, ip) gets an environment built from
  scratch: `HOME`, `XDG_RUNTIME_DIR`, `DOCKER_HOST` and `DOCKER_CONTEXT` from the node's own when set, the fixed `PATH`
  above (programs named by bare name are resolved there to an absolute path before the exec), and `LANG=C.UTF-8`. `DOCKER_CONFIG`, proxies, loader, Go runtime and OpenTelemetry variables
  never reach them.
* **Files:** every `(root, rel)` is relative, without `..`, `.`, empty segments, backslashes or control
  characters; the parent directory's realpath must lie inside the root's realpath, the directory fd is
  re-checked, and final components are opened with `O_NOFOLLOW`. Names that look like keys or secrets
  (`id_*`, `*.pem`, `*.key`, `.env*`, `token`, `credentials*`, `.ssh`, ...) are never read, even inside a root.
  Uploads go to `<rel>.incomplete`, are checked (size, whole-file sha256) and renamed into place; a disk is never
  filled below 1 GiB free. Deletes need `allow_delete`, a writable root, and no running container mounting the
  target or anything under it; a root itself, or a directory holding another root, is never deleted.
* **Engine HTTP** goes only to `127.0.0.1` ports that are console gateways or engine endpoints in the current
  inventory, GET/POST only, under the engine API paths, with hop-by-hop and Host headers dropped, and no
  redirects followed.
* **Scripts** run argv from the policy only (absolute program path, never a shell), with at most one argument
  picked from that action's allowlist, and a minimal environment (`PATH`, `HOME`, `USER`, `LOGNAME`, `LANG`,
  `XDG_RUNTIME_DIR`, `DOCKER_HOST`, `DOCKER_CONTEXT`; `PATH` keeps absolute entries only). One run per script; a run is stopped (its whole process
  group) on timeout, on cancel, or when the hub's session ends.
* **Bounded work:** a short command keeps at most 32 MiB of stdout and 1 MiB of stderr and a truncated `docker container
  inspect` is refused instead of parsed; readers of `/proc`, `/sys`, `/etc`, GGUF headers and model metadata accept regular files
  only, so a FIFO, socket or device cannot hang the node; a panic costs one frame or one loop round, not the node.
* **The stream:** in `--stdio` mode the node keeps the protocol on private copies of stdin/stdout and points
  fds 0/1 at `/dev/null` and stderr, so a stray print or a child process cannot corrupt frames. It never sends
  more DATA than the hub granted, buffers uploads only up to the credit it granted, keeps bulk JSON (log lines,
  inventory, metrics) well under the hub's flood limit, exits on stdin EOF or 30 s without a hub message, and
  exits at once (status 3) when a write to the hub has been blocked for 10 s.

## Notes on op results

* Upload errors (`fs.write`, `http.request` with a body) are answered as soon as they are known; the rest of
  that upload is discarded. A successful upload is answered only after the hub's `end`.
* `fs.read` returns `{"size": <bytes sent>}`; `fs.stat` reports `partial` (the size of `<rel>.incomplete`, 0 when
  there is none) and counts a symlink as existing only when it resolves inside the root.
* `docker.stop` / `docker.start` / `docker.rm` answer `not_found` for a container that does not exist.
* `models.scan` answers with the generation its scan will publish; a requested scan always publishes (and pushes
  `models_changed`) even when nothing changed. `force` skips the "a scan finished less than 10 s ago" shortcut.
  `models.list` pages by offset (`cursor`, `next_cursor`), at most 200 entries and about 768 KiB per page.
* `rel` of each `models.list` entry: `hub/models--<org>--<name>` in an `hf-home` root, `models--<org>--<name>` in
  an `hf-hub` root, `models/manifests/<registry>/<namespace>/<name>/<tag>` in an Ollama home and
  `manifests/...` in an Ollama `models/` directory, the file (first shard of a split set) in a `gguf` root, the
  model directory in a `plain` root; `auto` roots use the form of the layout they were detected as. A GGUF-only
  repo in an HF cache is one `gguf` entry at its repo directory; its quant files (or split sets) are listed in
  `meta.gguf_files` with their path below the repo directory (snapshot entries are symlinks into `blobs/`, so a
  launch mounts the cache, not a snapshot directory).
* Mounts in a run spec: `rel: ""` is the root directory itself (still realpath-confined). Every mount is read-only;
  `ro: false` is refused, and so is a source holding a key- or token-looking name up to four folders down (HF model trees are skipped).
  Everywhere else an empty `rel` is refused.
* GPU processes are attributed to containers through `/proc/<pid>/cgroup` (`.../docker-<id>.scope` with the
  systemd cgroup driver, rootless or rootful, or `.../docker/<id>` with cgroupfs), and the first sample waits
  briefly for the first inventory so it already carries container names and cgroup memory.

## Development

```
make node-test          # from the repository root: cargo test for the ecnode and ec-proto crates (hermetic)
make test               # every Rust test, hub included
make lint               # rustfmt check + clippy with warnings as errors
```

The tests are offline: fake `docker`, `nvidia-smi` and `ip` programs first on the fixed tool path (`ECNODE_TOOL_PATH`, which
exists for development and tests only), fake `/proc`, `/sys` and `/etc` trees under a temporary directory, and a fake engine
on an ephemeral `127.0.0.1` port. The end-to-end tests drive the real server loop through OS pipes with real protocol
frames, including a credit-limited upload larger than the window, a credit-limited download, cancel, monitor-mode refusals
and the write watchdog. The node has not been run on real GPUs or on an aarch64 machine since the rewrite; the arm64 CI job
builds and tests it but has not run yet.
