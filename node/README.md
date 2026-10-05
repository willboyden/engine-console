# ecnode: the Engine Console node agent

`ecnode` runs on every machine of a local GPU cluster. Engine Console (the hub) starts it, as a local subprocess
on its own machine or over ssh on the others, and talks to it on its stdin/stdout with the framed protocol in
[`docs/FLEET-PROTOCOL.md`](../docs/FLEET-PROTOCOL.md) (code twin: `src/ecnode/proto.py`). Nothing listens on the
network.

The node:

* **reports** the machine: GPUs (NVML when `pynvml` is importable, else `nvidia-smi`; zero GPUs and
  unified-memory GPUs are fine), per-process VRAM mapped to the container each process belongs to, host RAM and
  CPU, cgroup v2 memory of every engine container, disks and mounts (NFS/CIFS split into host and export),
  docker containers (sanitized: argv redacted on the node, environment never read), engine endpoints found by
  loopback probes, Prometheus metrics filtered to the families the hub asks for, and the models in each model
  root (Hugging Face caches, GGUF files, plain model directories, Ollama stores);
* **does** what the hub asks within its local policy: runs and removes engine containers and their loopback
  gateways, pulls allowed images, moves model files in and out of model roots, runs the operator's own
  lifecycle scripts, and forwards engine HTTP calls (chat, benchmarks, health) to loopback ports;
* **obeys its local policy file first**. The hub can read the policy (it is in `hello`), never change it.

Standard library only, Python 3.10 or newer, so it runs on the machine's system Python.

## Install (per machine, needs the operator's sudo)

```
sudo node/install.sh              # code -> /usr/local/lib/ecnode (root-owned), policy -> /etc/ecnode/policy.json
sudo node/install.sh --uninstall  # removes the code, keeps /etc/ecnode
```

`install.sh` copies the package to `/usr/local/lib/ecnode/lib/`, writes the launcher
`/usr/local/lib/ecnode/ecnode` (it runs `/usr/bin/python3 -I` with only that root-owned copy added to the path),
writes `/etc/ecnode/policy.json` from `ecnode policy-template` **only if it does not exist** (root:root, 0644,
monitor mode), and finishes with `ecnode check`. It is safe to re-run; it never edits ssh or sudo configuration
and starts nothing.

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

`check` prints host, GPUs, docker, the policy (and why it was refused, if it was), each model root with the
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
  and similar are refused; paths with `:` or `,` are refused (they would break docker mount syntax). Prefer the
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
  command line), never argv, logs or results; names that would change how the docker CLI itself runs
  (`LD_*`, `DOCKER_*`, `PATH`, proxies...) are refused. `docker.render` shows the same argv, so it never holds a
  value either. Container argv is redacted on the node (`--api-key X`, `--token=X`, `NAME=X`, inside `sh -c`
  strings too); `Config.Env` is never read; logs, script output and error text are scrubbed of secret-looking
  values. Logs are available for console-owned containers only.
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
  `XDG_RUNTIME_DIR`, `DOCKER_HOST`, `DOCKER_CONTEXT`). One run per script; a run is stopped (its whole process
  group) on timeout, on cancel, or when the hub's session ends.
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
  `ro: false` is refused, and so is a source holding a key- or token-looking name at its top or one folder down.
  Everywhere else an empty `rel` is refused.
* GPU processes are attributed to containers through `/proc/<pid>/cgroup` (`.../docker-<id>.scope` with the
  systemd cgroup driver, rootless or rootful, or `.../docker/<id>` with cgroupfs), and the first sample waits
  briefly for the first inventory so it already carries container names and cgroup memory.

## Development

```
cd node && uv run ruff check src tests && uv run mypy --cache-dir .mypy_cache/node src && uv run pytest -q
make node-matrix        # from the repository root: the tests on Python 3.12, 3.13 and 3.14
```

The tests are offline: fake `docker`, `nvidia-smi` and `ip` programs first on `PATH`, fake `/proc`, `/sys` and
`/etc` trees under a temporary directory, and a fake engine on an ephemeral `127.0.0.1` port. The end-to-end
tests drive the real server loop through OS pipes with real protocol frames, including a credit-limited upload
larger than the window, a credit-limited download, cancel, monitor-mode refusals and the write watchdog.
