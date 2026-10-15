# Operations

The operator guide for Engine Console 0.4.0 (contract v2, the fleet). Everything here was derived from the code at this
revision, the `Makefile` and the `--help` output of `engine-console hosts` and `ecnode` (those were run). The server, the
installer, ssh to a remote machine, the systemd unit, backups and the proxy were **not** run by the author of this document;
what the maintainers ran live is listed in `FEATURES.md`. Names and addresses (`gpu-box-2`, `edge-1`, `ops`, `10.0.0.5`,
`/data/hf`) are examples.

## Concepts in one minute

- The **hub** is this application (one process, `engine-console`, on one machine). It keeps no docker, GPU or model-file code
  of its own.
- A **host** is a machine the hub manages. Every host, the hub's own (`local`) included, has one **node**: `ecnode --stdio`,
  which the hub starts itself (a subprocess for `local`, ssh for the others). The pair is called a **link**.
- Each node reads a **policy file** on its own machine. The policy is the hard boundary: in **monitor** mode the console can
  only watch the machine; in **manage** mode it can do what the policy lists (images, model folders, scripts, switches).
- Engines run as containers on the machine that has the GPUs, on an internal docker network, behind a loopback gateway. The hub
  reaches an engine only through that machine's node.
- Model files live in **model roots** (folders named in the policy). The **library** is every machine's roots merged.

## Prerequisites

- **Hub machine:** Linux, `uv` and Python 3.13 (`uv venv --python 3.13`; newer interpreters often lack wheels), Docker with the
  NVIDIA driver and container toolkit (rootless Docker recommended) if engines will run there, and, for remote machines, the
  OpenSSH client (`ssh`, `ssh-keygen`). Node 22 is needed only for the frontend tests.
- **Every machine that runs engines (the hub's too):** Docker for the node's login user, the NVIDIA driver and container
  toolkit, the engine images already pulled, and an nginx image pinned by digest for the gateways (the console never pulls). Check
  the user's docker with `docker --context <context> ps` before blaming the console.
- **Every remote machine:** `sudo` once, system `python3` 3.10 or newer (Ubuntu 24.04 and DGX OS ship 3.12), a login user whose
  authorized keys you can edit, and an ssh server. The node needs no pip packages and no internet access.

## Install and run the hub (foreground)

```bash
git clone https://github.com/willboyden/engine-console && cd engine-console
make setup     # backend venv (uv, Python 3.13) incl. the node package, node venv (Python 3.12)
make run       # http://127.0.0.1:8791  (UI at /; OpenAPI JSON at /api/v1/openapi.json, needs a key; no Swagger page)
```

The `engine-console` and `ecnode` commands live in the backend virtual environment: in this guide `engine-console ...` and
`ecnode ...` mean `cd backend && uv run --frozen engine-console ...` (or `ecnode ...`), or the same after activating
`backend/.venv`. On remote machines `ecnode` is the installed launcher, `/usr/local/lib/ecnode/ecnode`.

**First sign-in.** Loopback is not trusted: the UI and every API call except `GET /api/v1/health` need an API key. The first
start writes an admin key to `<data dir>/bootstrap-admin.key` (default `~/.local/share/engine-console/bootstrap-admin.key`,
mode 0600) and logs only the path. Sign in with it, copy it into a password manager, mint named keys (`POST /api/v1/keys`,
roles `admin` and `viewer`) and delete the file; the console logs a reminder on every start while the file exists. With curl:
`curl -H "Authorization: Bearer $KEY" http://127.0.0.1:8791/api/v1/hosts` (bearer requests need no CSRF header; others
need `X-Engine-Console: 1`). A request without a key gets `401 unauthorized`. In the UI, the key button in the top bar
(or the dialog that opens on the first 401) takes the key; it is kept in that browser tab's session storage only and is
gone when the tab closes.

If you upgraded from 0.3 and no longer have a key, start once with `ENGINE_CONSOLE_TRUST_LOOPBACK=true`, create a key, and restart
without it
(`/api/v1/health` warns while the flag is on).

### Desktop launcher (Linux, optional)

```bash
make install-desktop        # app menu entry: ~/.local/share/applications/engine-console.desktop
make install-desktop-icon   # the same, plus a copy on the desktop (GNOME: marked trusted so a double click runs it)
make uninstall-desktop      # remove both
```

The launcher runs `engine-console open`: if nothing answers on the console's address it starts the console in the
background (`python -I -m engine_console.main` from the backend environment, detached from the launcher), waits up to 60 s
for `/api/v1/health`, then opens the UI with `xdg-open`. If a console is already running, however it was started, it only
opens the page. The menu entry's **Stop Engine Console** action runs `engine-console stop`, which stops the console that
`open` started and refuses any other process (a console started with `make run` is stopped with Ctrl-C). Both read the same
settings as the console (environment and env file). A desktop launch has no terminal, so a refusal or a failed start is
also shown as a desktop notification.

`open` refuses to open a page when `ENGINE_CONSOLE_HOST` is neither a loopback address nor a wildcard (a console bound to
`0.0.0.0` or `::` that is already running is opened through loopback, but the launcher never starts one), when something that does not answer like
the console holds the port, or when any listening socket on the port belongs to another user; it never starts a console
bound to every interface. It never reads or passes an API key: the UI asks for one as usual, and its sign-in box is a login
form the browser's password manager can save.

| File | Purpose |
|---|---|
| `<data dir>/console.log` | the background console's output (0600; rotated to `console.log.1` above 10 MiB, but only when `open` starts a console: one that runs for weeks keeps writing to the same file) |
| `<data dir>/console.pid` | the pid `stop` may signal, checked against its owner and command line first |
| `<data dir>/console.lock` | serialises launches, so a double click starts one console |

The checkout path goes into the launcher's `Exec` line, so `install-desktop` refuses a path with characters outside
`[A-Za-z0-9._/+-]`. Move the checkout or run the console with `make run` instead.

## Configuration (environment)

Environment variables use the setting's name in upper case, except five generic ones that need the `ENGINE_CONSOLE_` prefix:
`ENGINE_CONSOLE_HOST`, `ENGINE_CONSOLE_PORT`, `ENGINE_CONSOLE_ALLOWED_HOSTS`, `ENGINE_CONSOLE_TRUST_LOOPBACK` and
`ENGINE_CONSOLE_DATA_DIR`. The bare `HOST`, `PORT`, `ALLOWED_HOSTS`, `TRUST_LOOPBACK` and `DATA_DIR` are ignored, in the
environment and the env file alike (another program's exported `HOST=0.0.0.0` used to widen the bind); `engine-console` logs one
warning at start-up naming any that are still set. Lists and objects are JSON (`HF_ALLOWED_HOSTS='["huggingface.co"]'`).

| Variable | Default | Meaning |
|---|---|---|
| `ENGINE_CONSOLE_HOST` / `ENGINE_CONSOLE_PORT` | `127.0.0.1` / `8791` | Bind address. Do not widen without a TLS proxy and keys. An empty host is refused. |
| `ENGINE_CONSOLE_DATA_DIR` | `~/.local/share/engine-console` | SQLite DB (`console.db`), `secrets/`, bootstrap key |
| `ENGINE_CONSOLE_TRUST_LOOPBACK` | `false` | `true` makes any local process or loopback browser page an admin (health warns). Keep `false`. |
| `HOSTS_FILE` | `~/.config/engine-console/hosts.json` | The machines to manage (see "The host list"); read at start-up only |
| `HUB_KEY_FILE` | `~/.config/engine-console/ssh/id_ed25519` | The hub's own ssh key (`hosts init-key` creates it) |
| `NODE_POLICY` | unset | Policy file for the hub's own node (`--policy`). Unset: the node's own lookup (`/etc/ecnode/policy.json`, then `~/.config/ecnode/policy.json`) |
| `NODE_COMMAND` | unset | Complete argv of the local node (JSON list); development and tests only |
| `REMOTE_NODE_COMMAND` | `/usr/local/lib/ecnode/ecnode --stdio` | What the restricted key runs on remote machines; must equal the `command=` of the installed key line |
| `FLEET_ENABLED` | `true` | `false`: no node links at all (every host `disabled`) |
| `RESTART_SETTLE_S` | `4.0` | Pause between stop and start in a restart, so the fit check sees the freed VRAM |
| `HF_TOKEN` / `HF_TOKEN_FILE` | unset / `~/.config/engine-console/hf_token` | For gated models; prefer the file (0600, a group- or world-readable file is refused). Used by the hub only |
| `HF_ENDPOINT` | `https://huggingface.co` | |
| `HF_ALLOWED_HOSTS` | `huggingface.co`, `cdn-lfs*.huggingface.co`, `cas-bridge.xethub.hf.co`, `*.hf.co` | The console's own egress allowlist (JSON list) |
| `EGRESS_PROXY` | unset (direct) | e.g. `http://127.0.0.1:8082`; see "Egress proxy" |
| `EGRESS_CA_BUNDLE` | unset | Path to the proxy's CA certificate (certificate only, never the key); needed when `EGRESS_PROXY` is set |
| `REQUIRE_EGRESS_PROXY` | `false` | `true` refuses all HF traffic unless `EGRESS_PROXY` is set |
| `ENGINE_HF_OFFLINE` | `true` | Engines get `HF_HUB_OFFLINE=1` |
| `HF_CACHE_CONTAINER_PATH` | `/root/.cache/huggingface` | Where an engine sees the cache |
| `ENGINE_IMAGE_ALLOWLIST` | empty | Extra exact image references allowed as **global image pins** (`PUT /settings` `image_pins`); each adapter's own default is always allowed. It does not decide what may run: the node policy's `engine_images` does |
| `ENGINE_CONSOLE_ALLOWED_HOSTS` | empty | Extra Host header values (needed behind a proxy with its own hostname) |
| `MAX_SSE_CONNECTIONS`, `MAX_SSE_PER_KEY`, `SSE_IDLE_TIMEOUT_S` | `8`, `4`, `300` | Live streams at once, in total and per API key (requests without a key share one bucket); `429 too_many_streams` when full. A dashboard tab holds two |
| `MAX_CHAT_STREAMS`, `MAX_CHAT_STREAMS_PER_KEY` | `16`, `4` | Streamed chat replies at once, in total and per key; `429 too_many_chat_streams`. Non-streaming chat and the arena are not counted |
| `MAX_DOWNLOAD_QUEUE`, `DOWNLOAD_CONCURRENCY` | `50`, `2` | Unfinished download rows allowed; transfers at once |
| `STARTUP_TIMEOUT_S`, `SCRAPE_INTERVAL_S`, `SUPERVISOR_INTERVAL_S` | `1800`, `2.0`, `2.0` | |
| `BODY_CAP_BYTES`, `CHAT_BODY_CAP_BYTES`, `IMPORT_BODY_CAP_BYTES` | 1 MiB, 32 MiB, 256 KiB | |
| `OTLP_ENDPOINT`, `OTLP_ENABLED` | `localhost:4317`, `true` | Trace export (gRPC, no TLS) |
| `PORT_RANGE_START` / `PORT_RANGE_END` | `18000` / `18099` | Only a fallback for a node whose policy sends no range |
| `ENGINE_CONSOLE_ENV_FILE` | `~/.config/engine-console/env` | Optional `KEY=VALUE` file with the same names (the `ENGINE_CONSOLE_` prefix included for the five generic ones); real environment variables win |

**Settings that no longer do anything** (they still parse): `HF_CACHE_DIR` (the hub has no model cache; it is only echoed by
`GET /settings`), `DOCKER_BIN`, `GATEWAY_IMAGE`, `GATEWAY_IMAGE_ALLOWLIST`, `DISCOVERY_ENABLED`, `DISCOVERY_PORTS`,
`DISCOVERY_INTERVAL_S`, `ENABLE_API_DOCS`. `ENGINE_NETWORK` and `DOCKER_CONTEXT` are only echoed by `GET /settings`
(`DOCKER_CONTEXT` also by `/health`), and `ENGINE_NETWORK` is a fallback in the "copy as command" snippet; the node uses its
policy's `engine_network` and `docker.context`. A `DOCKER_CONTEXT` set in the hub's environment is passed to the **local** node, which uses it only
when its policy names no context. The replacements are node-policy fields: `docker`, `engine_network`, `gateway_image`,
`engine_images` and `discovery_ports`.

### Persistent settings: the env file

`~/.config/engine-console/env` (override with `ENGINE_CONSOLE_ENV_FILE`) uses the same names, the `ENGINE_CONSOLE_` prefix included for
the five generic ones:

```
EGRESS_PROXY=http://127.0.0.1:8082
EGRESS_CA_BUNDLE=/path/to/proxy-ca-cert.pem
HOSTS_FILE=/home/ops/engine-console-hosts.json
ENGINE_CONSOLE_ALLOWED_HOSTS=["console.example.internal"]
```

Real environment variables win over the file. Keep it `chmod 600` if it ever holds a token. Tests ignore this file.

## The host list (`hosts.json`)

`~/.config/engine-console/hosts.json` (override with `HOSTS_FILE`) lists the machines. No file means one host, `local`, labelled
with the machine's hostname. It is read **once at start-up**: edit it, then restart the console. `engine-console hosts check`
and `hosts list` read it fresh each time.

```json
{"hosts": [
  {"id": "local", "label": "Workstation"},
  {"id": "gpu-box-2", "label": "GPU box 2", "ssh": "ops@10.0.0.5"},
  {"id": "edge-1", "label": "Edge 1", "ssh": "ops@10.0.0.6", "port": 2222, "enabled": false}
]}
```

| Field | Rules |
|---|---|
| top level | exactly `{"hosts": [...]}` and nothing else; 1 to 32 hosts |
| `id` | required; lowercase letters, digits and `-`, 1 to 32 characters, not starting with `-`; unique. `local` is the machine the console runs on, and only `local` may use that id |
| `label` | 1 to 64 printable characters; defaults to the id |
| `ssh` | required for every host except `local`; `user@address` or just `address` (letters, digits, `.`, `_`, `-`; the address part starts with a letter or digit, so it can never be read as an option; no `%`). A bare IPv6 address is refused (it has a `:`); use a DNS name or an ssh_config alias |
| `port` | optional, 1 to 65535 |
| `ssh_config` | optional, default `false`. `true` lets `ssh` read your ssh configuration (so `ssh` may be an alias, and a ProxyJump applies), **and** lets that configuration add identity files or other options. The hub then notes any login that was not through the restricted key. Leave it off unless you need it |
| `enabled` | optional, default `true`; `false` shows the host as `disabled` and never connects |

Validation is strict on purpose: an unknown key, a duplicate key, `NaN`, a symlink, a file that is not owned by the user running
the console, a group- or world-writable file (`chmod 600`), or a file over 64 KiB is an error. The console then **refuses to
start** with `engine-console: <path>: <reason>` and exit status 2, rather than guess which machines it may run ssh against. You
can leave `local` out (the console then manages only other machines), but API calls that omit `host` default to `local` and
answer `404 unknown_host`.

## Adding a machine

The five steps are in the README. In detail:

### 1. Install the node (on the machine, needs `sudo`)

```bash
sudo node/install.sh              # from a checkout of this repository on that machine
sudo node/install.sh --uninstall  # removes the code, keeps /etc/ecnode
```

`install.sh` checks for `/usr/bin/python3` 3.10 or newer, copies the package to `/usr/local/lib/ecnode/lib/` (root-owned, staged
next to the old copy and swapped in with one rename, so a failed install leaves the old one), writes the launcher
`/usr/local/lib/ecnode/ecnode` (it runs `/usr/bin/python3 -I` with only that copy on the path), writes `/etc/ecnode/policy.json`
from `ecnode policy-template` **only if none exists** (root:root, 0644, monitor mode), and ends with `ecnode check` as the user
who called `sudo`. It is safe to re-run. It opens no port, creates no user, edits no ssh or sudo configuration and starts no
service. It runs with a fixed `PATH` and `umask 022`, never changes the owner or mode of `/usr/local/lib` (it creates it only if
missing and refuses one that anyone but root can write to), and prints the git commit it installs, read with git as the user who
called `sudo` (`<sha> (plus uncommitted changes in node/src/ecnode)`, or `unknown (not a git checkout)`). The commit is also
written into the launcher's docstring. No test runs the installer. It was run on 2026-10-14 in a user and mount namespace
(mapped root, scratch directories over `/usr/local/lib` and `/etc`, docker stubbed), and on 2026-10-15 as real root (`sudo`)
on five machines: four aarch64 machines (Ubuntu 24.04, Python 3.12) and one x86_64 machine (Ubuntu 26.04, Python 3.14).

`ecnode` commands (all read-only except `--stdio`, which is what the hub starts):

| Command | Does |
|---|---|
| `ecnode --stdio [--policy PATH]` | speak the hub protocol on stdin and stdout |
| `ecnode check [--policy PATH]` | print what this machine would report: host, GPUs, docker, the policy (and why it was refused), each model root with its filesystem and free space, containers and endpoints; exit status 1 when a policy file exists but was refused |
| `ecnode policy-template` | print a starter **monitor-mode** policy for this machine (detected model folders, docker context, a local nginx digest if there is one); writes nothing |
| `ecnode version` | print the version |

### 2. Create the hub's key and print the line

```bash
engine-console hosts init-key --from 10.0.0.2      # the hub's address as the machine sees it; comma-separate several
```

It creates `~/.config/engine-console/ssh/id_ed25519` (ed25519, no passphrase, mode 0600 in a 0700 directory) unless one exists
(an existing key is never replaced), never prints the private half, and prints one line:

```
restrict,from="10.0.0.2",command="/usr/local/lib/ecnode/ecnode --stdio" ssh-ed25519 AAAA... engine-console-hub
```

- `restrict` turns off pty allocation, port, agent and X11 forwarding and user rc files for this key.
- `command="..."` makes sshd run the node whatever the client asked for. It must equal `REMOTE_NODE_COMMAND`.
- `from="..."` lets only the hub's address use the key. Prefer addresses or CIDR ranges (`10.0.0.0/24`); `--from '*'` works
  from anywhere and the CLI warns about it.
- The comment `engine-console-hub` replaces whatever comment the public key had.

### 3. Authorize the key (on the machine)

Add that **one line** to the authorized keys of the login user the node should run as (the user whose Docker it will use). The
console never does this itself.

### 4. Pin the host key

The hub's ssh uses `StrictHostKeyChecking=yes` with known_hosts **read-only**, so it refuses a host key it has not seen. From the
hub machine, as the user that runs the console, connect once by hand with the same address (and `-p` if you set a port):

```bash
ssh ops@10.0.0.5 true
```

Compare the fingerprint with the machine's own (`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` on it) before you accept. The
host key is stored before authentication, so a final `Permission denied` is fine. Then add the host to `hosts.json`.

### 5. Check it

```bash
engine-console hosts check
```

It connects to every host once (the local node subprocess too) and prints a table, then one line per problem:

```
HOST       STATE  NODE   ARCH    GPUS  POLICY
local      up     0.4.0  x86_64  2     monitor
gpu-box-2  up     0.4.0  x86_64  1     monitor
```

(illustrative; `NODE` is the node's version, `POLICY` its mode). `--timeout N` waits N seconds (default 20). The exit status is 0 when
every host is `up`, `stale` or `disabled`, and 1 otherwise. `engine-console hosts list` prints the configured hosts without
connecting. Then restart the console so it reads the new `hosts.json`. A link that is not `up` is explained in "Link states".

### Why `ssh -F none` and the other options

The hub runs exactly this command for a remote host (the options are from `services/hosts.py`; `-F none` is left out for a host
with `ssh_config: true`):

```
ssh -T -e none -F none -o BatchMode=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o VerifyHostKeyDNS=no
    -o IdentityAgent=none -o PKCS11Provider=none -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no
    -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no -o ForwardX11=no -o ClearAllForwardings=yes
    -o PermitLocalCommand=no -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=3
    -o IdentityFile=<hub key> -o IdentitiesOnly=yes [-p <port>] -- <user@address> /usr/local/lib/ecnode/ecnode --stdio
```

with an environment of only `HOME`, `PATH`, `LANG=C.UTF-8` and `USER`.

| Option | Why |
|---|---|
| `-F none` | Ignore the user's and the system's ssh configuration, so the command line is the whole truth: no ProxyCommand, LocalCommand, extra identities or aliases surprise you. |
| `-o IdentityFile=<key>` and `IdentitiesOnly=yes` | Offer exactly the hub's key. `-i` is **not** used: with `-i` and a missing file ssh only warns and then offers the user's default identity files (seen with `ssh -G`), which would log in with an unrestricted key. The hub also refuses to run ssh while the key file is missing. |
| `BatchMode=yes`, `PasswordAuthentication=no`, `KbdInteractiveAuthentication=no` | Never prompt; nothing is unattended-safe about a password. |
| `StrictHostKeyChecking=yes`, `UpdateHostKeys=no`, `VerifyHostKeyDNS=no` | A host key that is unknown or changed is an error, never a silent trust; known_hosts is read and never written; DNS does not vouch for keys. |
| `IdentityAgent=none`, `ForwardAgent=no`, `ForwardX11=no`, `ClearAllForwardings=yes`, `PKCS11Provider=none` | No agent, no forwarding, no smart-card provider: the key can only reach the node. |
| `ControlMaster=no`, `ControlPath=none` | No shared connections with other ssh sessions. |
| `PermitLocalCommand=no` | No local command hooks. |
| `-T -e none` | No pty, no escape character: the stream carries binary frames. |
| `ConnectTimeout=5`, `ServerAliveInterval=5`, `ServerAliveCountMax=3` | A dead machine is noticed in seconds, not minutes. |
| `--` before the destination | The destination can never be read as an option. |

A key path with whitespace, quotes or `%` is refused. The key's path is replaced by `<hub key>` in link error text.

## Link states

`GET /api/v1/hosts` (and the Hosts page, the `host` events of `GET /fleet/stream`, and `engine-console hosts check`) shows each
link's `state`, an `error` in words, `retry_in_s` and `notes`. After an attempt that needs a person to fix something, the hub
retries only every 300 s; **Reconnect** (`POST /api/v1/hosts/<id>/reconnect`, admin) skips the wait and tries now.

| State | Meaning | What to do |
|---|---|---|
| `connecting` | An attempt is in progress, waiting for the node's `hello` (up to 25 s). `hosts check` shows "no answer yet" if it never settles. | Wait. If it stays here, the error of the previous attempt is the clue. |
| `up` | The node said hello and the protocol matches. | Read `notes` (below). |
| `stale` | Connected, but no sample for 6 s (samples come every 2 s). Numbers shown may be old. | Wait; the node may be blocked on a slow call. If nothing at all arrives for 30 s the link drops to `down` with "no message from the node for 30 s". |
| `down` | The connection failed or ended. The hub retries after 1 s, doubling to 60 s, or after 300 s when only a person can fix it. | Find the exact `error` below. |
| `incompatible` | Hub and node speak different protocol versions; retried every 300 s. | Update the side the message names (the same repository revision on both sides speaks the same protocol). |
| `disabled` | `enabled: false` in `hosts.json`, or `FLEET_ENABLED=false`. | Change the file or the setting and restart. |

Errors you may see in `error`, and the remedy (messages are the code's own; `<label>` is the host's label, `<dest>` its `ssh`
value):

| Exact message (start) | Cause and fix |
|---|---|
| `ssh refused the login to <dest>: the hub's key is not authorized there. Run engine-console hosts init-key ...` | The key line is missing or wrong on that machine, or `from=` does not match the hub's address, or the line is in another user's authorized keys. Re-run `hosts init-key --from <hub address>` and add the printed line for the login user named in `ssh`. Retried every 300 s. |
| `the hub's ssh key file is readable by other users, so ssh refuses it: chmod 600 the key` | `chmod 600 ~/.config/engine-console/ssh/id_ed25519`. |
| `the host key of <dest> is unknown or has changed. Check it, then connect once by hand to trust it: ssh <dest> true` | Unknown host (do step 4) or a changed key (a reinstalled machine, or something in the middle: compare fingerprints, then remove the old known_hosts entry). The hub never writes known_hosts. |
| `cannot resolve <dest>: check the name in hosts.json` | The name does not resolve from the hub; use an address or fix DNS. |
| `cannot reach <dest>: <ssh's last line>` | No route, connection refused or timed out: the machine is off, the address or `port` is wrong, or a firewall blocks ssh. |
| `ssh to <dest> failed: <last line>` | Any other ssh failure; the last line of ssh's own error is included. |
| `the ecnode node is not installed on <label> (expected /usr/local/lib/ecnode/ecnode): run its install.sh there` | Exit status 127 from the forced command: run `sudo node/install.sh` on that machine, or fix `REMOTE_NODE_COMMAND` and the `command=` of the key line so they agree. Retried every 300 s. |
| `the console has no ssh key yet: run engine-console hosts init-key ...` | The key file does not exist. Create it; the next retry (or Reconnect) picks it up. |
| `the hub key path is not usable` or `... must not contain spaces, quotes or '%'` | `HUB_KEY_FILE` points somewhere ssh cannot take. |
| `the ecnode package is missing from the console's Python environment: ...` | Local node only: the backend environment lacks `ecnode`; run `make setup`. |
| `could not start ssh: ...` | The `ssh` program is not installed or not on the hub's `PATH`. |
| `the node gave up: <reason>` | The node sent `fatal` before its hello (for example "the node could not start: <exception>"). Run `ecnode check` on that machine. |
| `no answer from the node within 25 s` | ssh connected but the node printed nothing: a slow start, or something else is on the other end. Run the remote command by hand: `/usr/local/lib/ecnode/ecnode check`. |
| `protocol error from the node: ...` | The far side printed something that is not the protocol (the hub skips up to 64 KiB of shell noise first). Check shell startup files and the forced command. |
| `<label> runs ecnode X (protocol N); this Engine Console speaks protocol 1. Update <the Engine Console hub / ecnode on label>.` | State `incompatible`: update the side named. |
| `the node exited (N): ...`, `the node was stopped by signal N: ...`, `the connection closed`, `the node closed the connection` | The node ended; the tail of its stderr is appended. Run `ecnode check` on that machine and look at its Python version. |
| `no message from the node for 30 s` | The node hung or the network froze. The hub drops and retries. |
| `the node sent over 20 MB or 50000 messages in 10 s; dropped it`, `the node sent more data on stream N than the credit this hub granted it` | A node misbehaving or broken; the hub drops it. Reinstall the node from a known revision. |
| `reconnecting at your request` | You pressed Reconnect. |

`notes` (state `up` is still fine):

- `reached with an UNRESTRICTED ssh key: restrict it to the node command in authorized_keys (engine-console hosts init-key prints
  the line)`: the login did not come through a `command=` line (the node saw no `SSH_ORIGINAL_COMMAND`). Typical causes: the key
  line lacks `command=`, or another key (through `ssh_config: true`) was used. Fix the key line; the node then reports
  `forced_cmd: true`.
- `ignored N byte(s) the far side printed before the node started (a shell startup file writing to stdout?)`: harmless; remove
  the output from the login shell's startup files.

A request to a host whose link is not up fails at once with `503 node_unavailable`; the detail ends `... is not connected (<error>)`.

## The node policy

Location: `--policy PATH`, else `/etc/ecnode/policy.json` if it exists, else `~/.config/ecnode/policy.json`. The installed copy is
root-owned, 0644, so the hub's key (which runs as the login user) cannot edit it. The node refuses the file (and runs in
**monitor mode** with the reason in `policy.error`) if it is a symlink or not a regular file, group- or world-writable, owned by
anyone but root or the node's user, under `/etc` without being owned by root, in a directory others can write to or that another
user owns, larger than 256 KiB, not valid JSON, or invalid against the schema. Unknown keys are errors (a typo in `allow_delete`
is never silently ignored); keys starting with `_` are comments. No file at all also means monitor mode.

**Applying a change.** The node re-checks the file about every 2 seconds and enforces a new version at once. The hub learns a
node's policy from its `hello`, so the hub (preflight, the Hosts page) sees an edit **only after the host reconnects**: press
Reconnect on the host, or `POST /api/v1/hosts/<id>/reconnect`, or restart the console. (The hub sends no `node.info`.) A
half-edited or invalid file silently drops that machine to monitor mode until it is fixed; the host shows `policy.error`.

### A manage-mode policy, field by field

```json
{
  "_comment": "gpu-box-2: manage mode for the console's ops user. Paths and the digest are examples.",
  "version": 1,
  "mode": "manage",
  "docker": {"binary": "docker", "context": "rootless"},
  "engine_images": ["vllm/vllm-openai:*", "lmsysorg/sglang:*", "ghcr.io/ggml-org/llama.cpp@sha256:*"],
  "gateway_image": "nginx@sha256:<64 hex digits>",
  "engine_network": "engine-console-engines",
  "port_range": [18000, 18099],
  "discovery_ports": [8000, 30000],
  "model_roots": [
    {"id": "hf",     "path": "/data/hf/hub",       "kind": "hf-hub", "writable": true,  "label": "HF cache"},
    {"id": "gguf",   "path": "/data/gguf",         "kind": "gguf",   "writable": true,  "label": "GGUF files"},
    {"id": "shared", "path": "/mnt/shared-models", "kind": "auto",   "writable": false, "label": "Shared (NFS)"}
  ],
  "scripts": [],
  "allow_pull": false,
  "allow_delete": false,
  "allow_ipc_host": true,
  "external_control": "none",
  "max_shm_gib": 64
}
```

| Key | Default if absent | Meaning and validation |
|---|---|---|
| `version` | `1` | Must be `1`. |
| `mode` | `monitor` | `monitor` or `manage`. |
| `docker.binary` | `docker` | `docker` or an absolute path to the docker CLI. |
| `docker.context` | `null` | The docker context every call uses (`docker --context ...`). Naming it avoids guesswork: a node started with a scrubbed environment (the hub's ssh session has no `DOCKER_HOST`) whose current context is `default`, and that cannot use the default socket, falls back to an existing `rootless` context. `ecnode policy-template` fills it in. |
| `engine_images` | none (nothing can be launched) | At most 64 entries: an exact reference, `repository:*` (any tag of exactly that repository) or `repository@sha256:*` (any digest of it). The repository is compared literally: `vllm/vllm-openai:*` allows neither `evil/vllm/vllm-openai:x` nor `vllm/vllm-openai-evil:x` nor `docker.io/vllm/vllm-openai:x`. Tag patterns accept any tag, and a registry can repoint a tag; digest references are immutable. |
| `gateway_image` | `null` | The nginx image for loopback gateways, an exact reference (best `nginx@sha256:<digest>`) that is already on the machine. With `null`, launches there are blocked (`gateway_missing`). The hub no longer has a gateway setting of its own. |
| `engine_network` | `engine-console-engines` | The docker network engines join. Created `--internal` if missing; refused if it exists and is not internal. |
| `port_range` | `[18000, 18099]` | `[first, last]` for gateway ports on 127.0.0.1: first 1024 to 65535, and the range at most 1001 ports. |
| `discovery_ports` | `[8000, 8080, 11434, 30000]` | Up to 128 loopback ports probed for engines that have no container of their own (and that answer like an OpenAI-compatible server). |
| `model_roots` | none | Up to 32 folders that hold only models; see "Model roots". |
| `scripts` | none | See "Script-managed engines". |
| `allow_pull` | `false` | The node allows `docker.pull` for images matching `engine_images`. The hub sends no pull today, so this has no effect yet. |
| `allow_delete` | `false` | Allows deleting models (and removing a cancelled transfer's partial file) in writable roots. |
| `allow_ipc_host` | `false` | Lets multi-GPU engines use `--ipc host` (NCCL needs shared memory). When false the hub starts multi-GPU engines with a private IPC namespace. The starter template writes `true`. |
| `external_control` | `none` | `stop_start` lets the console stop and start recognised engine containers it did not create (see "External engines"). |
| `max_shm_gib` | `64` | The largest `--shm-size` an engine may be given (more than 0, at most 4096). |

Examples ship in `node/examples/policy.monitor.json` and `policy.manage.json`.

**Monitor mode.** The node answers: its own info and metrics configuration, rendering the `docker run` it would run (secrets
omitted), a sanitized container inspect, logs of console-owned containers, image facts, engine HTTP calls (so chat and
benchmarks through the console work), `fs.stat` and `fs.read` (reading model files for copies), and the model scan and list.
Everything else answers `monitor_only` (`409`): starting, stopping and removing engines, creating the network or gateways,
pulling images, writing, deleting, scripts and `external_control`. **Manage mode** allows those, limited by the other policy
fields.

### Policy errors you may see

In `policy.error` (shown on the host and in `ecnode check`); all but the first end with "; running in monitor mode":

| Message | Fix |
|---|---|
| `no policy file, so this node runs in monitor mode (write /etc/ecnode/policy.json, for example from ecnode policy-template)` | Write one. |
| `policy file <path> is group- or world-writable (run: sudo chmod 644 <path>)` | `chmod 644`. A file created with `umask 002` is 664 and is refused. |
| `policy file <path> must be owned by root (run: sudo chown root:root <path>)` | For a file under `/etc`. |
| `policy file <path> is owned by another user`, `... is a symlink; point --policy at the real file`, `... is in a directory others can write to (<dir>)`, `... is in a directory owned by another user (<dir>)` | Fix the owner, the link or the directory. |
| `policy file <path> is larger than 262144 bytes`, `... is not valid JSON (...)` | Trim or repair the JSON. |
| `policy file <path>: <field>: ...` | The field named is invalid, for example `policy.model_roots[0].path: '/' is too broad for a model root; name the directory that holds the models`, `policy.engine_images[1]: ... must be an exact image reference, 'repository:*' or 'repository@sha256:*' with a literal repository`, `policy: unknown key 'allow_delet' (allowed: ...)`. |

## Model roots and kinds

A **model root** is a folder in the policy that holds only models. The node scans it, serves files from it, mounts parts of it
read-only into engines, and (if `writable`) accepts downloads and copies into it. Root rules: an absolute path of at most 512
characters; no `:`, `,`, `\` or `.` and `..` segments (they would break docker mount syntax); not `/`, `/etc`, `/home`, `/tmp`, `/opt`,
`/srv`, `/mnt`, `/media`, `/var`, `/usr`, `/root` or the user's home directory itself (a folder below them is fine); ids are
`[a-z0-9][a-z0-9_-]{0,31}` and unique. Files with secret-looking names (`id_*`, `*.pem`, `*.key`, `.env*`, `token*`, `credentials*`,
`.ssh`, `.gnupg`, `.aws`, `.config`, `.local`, ...) are never read, even inside a root.

| Kind | Layout the node looks for | Listed as | Mounted into an engine |
|---|---|---|---|
| `hf-home` | `HF_HOME`: `<root>/hub/models--org--name/{blobs,snapshots,refs}` | `hf` (a GGUF-only repo as `gguf`) | the whole root, read-only, at the engine's HF cache path |
| `hf-hub` | the hub cache itself: `<root>/models--org--name/...` | same | the whole root at `<HF path>/hub` |
| `gguf` | `.gguf` files anywhere below, up to 6 levels; split sets `*-00001-of-0000N.gguf` grouped | `gguf` | the file's folder at `/models`, read-only (for a quant inside an HF cache: the whole repo folder) |
| `plain` | directories holding `config.json` and weights, up to 3 levels | `plain` | that directory at `/models` |
| `ollama` | an Ollama home or its `models/` folder (`manifests/`, `blobs/`) | `ollama` (`name:tag`) | the store's `models` folder at `/root/.ollama/models`, read-only |
| `other` | anything | `other`: one entry per top-level folder, size only | never |
| `auto` | detected per root: `hub/` means `hf-home`, `models--*` means `hf-hub`, `manifests` means `ollama`, else `plain` and `gguf` | as detected | as detected |

Recommendations:

- Point an HF root at the cache's `hub/` folder (`kind: hf-hub`) rather than `HF_HOME`, and an Ollama root at its `models/`
  folder, so token files, keys and prompt history sit outside every root (the starter template does this). The node refuses to
  mount a folder with a key- or token-looking name up to four folders down (HF model trees are skipped); from an `hf-home` root the console mounts
  only `hub/`. An engine is given the whole root, so it can read every model in it.
- Make a root `writable: true` only where downloads and copies should land. Engines always mount roots read-only; the node
  refuses a read-write mount (the old `HF_CACHE_READONLY` setting is gone).
- A root on NFS is normally `writable: false`. If it is another configured machine's disk the library lists the bytes once (see
  "Shared storage").
- The scanner runs when the node starts, again when a root's directory signature changes (checked every 30 s), every 10 minutes,
  and when asked (`POST /api/v1/library/rescan`, admin, optional `{"host_id": "gpu-box-2"}`). Each root is bounded to 50,000
  directory entries and 120 s; beyond that the library warns `<label>: the scan of root <id> did not finish` and sizes are lower
  bounds.
- A checkpoint folder whose weights are in a format the common engines do not load (for example `*.ftw`) is still listed, with the
  warning `the weights are in a format the common engines do not load (...); listed so its disk space is visible`.

## Engines and images

The adapters carry pinned default images (`vllm/vllm-openai:v0.27.1`, `lmsysorg/sglang:v0.5.20-cu130`,
`nvcr.io/nvidia/tensorrt-llm/release:1.2.1`, `ghcr.io/ggml-org/llama.cpp@sha256:4c0ece46...` and `ollama/ollama:0.30.11`; the
adapters are the authority and `GET /api/v1/engines` shows the effective default). The console **never pulls**. Pull on each
machine, as the node's login user and with the context named in the policy:

```bash
docker --context rootless pull vllm/vllm-openai:v0.27.1
docker --context rootless pull nginx@sha256:<digest>          # the gateway image named in the policy
```

(`GATEWAY_IMAGE` in `backend/src/engine_console/config.py` is the digest earlier versions used; review it or pick your own.)

Which image a launch uses: the `image` in the request (it must match the node's `engine_images`), else the pin for that machine
and engine, else the global pin for the engine, else the adapter's default for the machine's architecture. Pin per machine:

```bash
curl -X PUT -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"host_image_pins": {"gpu-box-2": {"vllm": "vllm/vllm-openai:v0.27.1"}}}' http://127.0.0.1:8791/api/v1/settings
```

The pin table is replaced as a whole (send `{}` to clear it); each image must be a pinned tag or digest, never `:latest`. The
adapters carry no arm64 images, so an aarch64 machine needs a pin that exists for it (unverified).

## Launching on a machine

The launch form (or `POST /api/v1/instances`) takes the machine, the model (a library copy on **that machine**, or visible to it
over shared storage), the engine, the GPUs **by index on that machine**, and the settings. `POST /api/v1/instances/preflight`
runs the same checks without starting anything. Example body: `{"engine": "vllm", "host_id": "gpu-box-2", "model": {"format":
"hf", "repo_id": "org/model"}, "gpu_ids": [0], "params": {"max_model_len": 8192}}`. The engine must be able to load the model's
format (`format_unsupported` otherwise). Per host, the console picks the first free port of the node's `port_range`.

What the node starts: the engine container named `ec-<name>-<id6>` with `--cap-drop ALL`, `--security-opt no-new-privileges:true`,
`--pids-limit 4096`, `--restart no`, `--pull never`, the GPUs by UUID (`--gpus device=...`), `--shm-size` (the adapter's size, at most
`max_shm_gib`), `--ipc host` only for multi-GPU engines when allowed, the model mounts read-only, `HF_HUB_OFFLINE=1`, and labels
(`engine-console=1`, `.role`, `.instance`, `.engine`, `.model`, `.port`, `.name`, `.format`, `.gpu_ids`, `.gpus`, and `.root`, `.rel` for
non-Hugging Face models) on the internal network; then a gateway `<container>-gw` published only on `127.0.0.1:<port>`. Engines
restart **neither** after a crash nor after a reboot (`--restart no`): start them again from the console. Engines survive a
console restart, and the console adopts them again from their labels when each link comes up. `GET /api/v1/instances/<id>/command`
shows the exact `docker run` the node would use (secret values omitted).

### Launch checks (preflight codes) and fixes

Each check has a `code` and a `level` (`ok`, `warn` or `block`); a launch is refused (`409 preflight_failed`) when any check blocks.

| Code | Level | Message (start) and fix |
|---|---|---|
| `host_down` | block | `<label> is not connected (...)`: fix the link first. |
| `monitor_only` | block | `<label>'s node is in monitor mode (<policy path>)`: its administrator sets `"mode": "manage"`, then Reconnect. |
| `invalid_params` | block | An engine parameter is unknown or out of range; the message lists them. |
| `format_unsupported` | block | The engine cannot load that model format (for example llama.cpp needs `gguf`). |
| `model_missing` | block | `<model> is not in <label>'s model library: download or copy it to that machine first`. Rescan if you just added files. |
| `fit_unavailable` | warn | No fit estimate: no readable `config.json` yet; rescan. |
| `fit` | ok, warn or block | `fit verdict: ...`; `fits if you stop X` names resident engines. |
| `trust_remote_code` | warn | The model repository's code runs inside the container; off unless you set it. |
| `no_free_port` | block | `no free port in <lo>-<hi> on <label>`: widen `port_range` or stop instances. |
| `no_gpu`, `gpu_unknown` | block | `<label> reports no GPU` / `<label> has no GPU N (it has ...)`: GPU indices are the machine's own. |
| `gpu_shared` | warn | `GPU already used by <name>; relying on free VRAM`. |
| `image_not_allowed` | block | `image <image> is not allowed by <label>'s policy (engine_images)`: add a matching pattern to that node's policy, or pick another image. |
| `image_missing` | block | `image <image> is not present on <label>: pull it on that machine (the console never pulls)`. |
| `image_check_failed` | warn | The node could not check for the image; see the detail. |
| `gateway_missing` | block | `<label>'s policy names no gateway_image ...`: set `gateway_image` to an nginx digest present on that machine. |
| `secret_in_argv` | block | The adapter would put a secret parameter on the command line: unset it or use an engine that takes it from the environment. |
| `invalid_launch` | block | The run spec breaks a node rule, for example `shm_size 16g is above this machine's policy limit of N GiB` (raise `max_shm_gib`) or a bad mount; the message says which. |
| `invalid_model` | block | The model reference is malformed (a GGUF model needs `root_id` and a `.gguf` `rel`). |
| other codes | warn or block | Engine compatibility rules (`check.*` in the learn content), each with its fix in the message. |

### Instances that do not start or stay up

| State and `error` | Meaning and fix |
|---|---|
| `failed`, `container exited with code N` | The engine crashed; `last_logs` holds the tail (secrets masked). Read it, fix the setting, start again. |
| `failed`, `container disappeared` | The container was removed outside the console. |
| `failed`, `startup timed out` | No healthy answer within `STARTUP_TIMEOUT_S` (1800 s). Look at the logs: a huge model, a failing download, or a wrong health path. |
| `failed`, `gateway container missing or stopped; restart the instance` | The gateway is gone: restart the instance. |
| `unreachable`, `<label> is not connected: ...` | The host's link is down; the containers keep running. Fix the link; the instance returns on reconnect. |
| stuck in `loading` | `GET /api/v1/instances/<id>/logs`; the state moves on the adapter's log lines and the health probe. |
| `409 preflight_failed` on start | Run the preflight; the blocking check says what changed since the instance was created. |

## Script-managed engines

Many labs start an engine with their own script because the order matters (start the big model before the small one, stop one
before the other). The console must not bypass that: stopping or starting such a container directly could break the ordering. So
the operator **registers the script in the node's policy**, and the console runs the script's actions and nothing else.

```json
"scripts": [
  {"name": "big-model", "label": "Big model (2 GPUs)", "engine": "sglang",
   "description": "Starts before the small model; the script handles the order.",
   "containers": ["myproj-sglang-*"],
   "status_action": "status",
   "timeout_s": 900,
   "actions": [
     {"name": "up",     "label": "Start",        "argv": ["/opt/lab/model.sh", "up"],   "confirm": true},
     {"name": "down",   "label": "Stop",         "argv": ["/opt/lab/model.sh", "down"], "confirm": true, "danger": true},
     {"name": "status", "label": "Check status", "argv": ["/opt/lab/model.sh", "status"]},
     {"name": "switch", "label": "Switch model", "argv": ["/opt/lab/use-model.sh"], "args": ["model-a", "model-b"], "confirm": true}
   ]}
]
```

- `argv` stays on the node; the hub sees only names, labels, allowed `args`, `confirm` and `danger`. `argv[0]` must be an absolute
  path; no shell is ever involved. A run names a script, an action and, if the action lists `args`, exactly one of them, appended
  as a single argument. Names are `[a-z0-9][a-z0-9_-]{0,31}`; at most 32 scripts with 16 actions each.
- `containers` are shell-style patterns for the container names the script owns. Every running container that matches (and is not
  console-owned) is shown as **one** instance, `scr-<host>-<script>` (each `-` of the host id doubled), with `management: "script"`. `docker.external` refuses to
  touch those containers. A pattern that is too broad swallows other engines into the script's instance; one that matches nothing
  leaves the instance `stopped` (the node's script status is then `down`, or `unknown` when the list is empty).
- Controls: start runs `up`, stop runs `down`, restart runs `down` then `up` as **one** job and stops at the first failure. While
  an action runs the instance shows `starting` or `stopping`. `status_action` runs only when asked (run it like any action).
- Environment of a script: `PATH`, `HOME`, `USER`, `LOGNAME`, `LANG`, `XDG_RUNTIME_DIR`, `DOCKER_HOST` and `DOCKER_CONTEXT` from the
  node's own environment; nothing from the hub. Working directory: the user's home. One run per script at a time. A run is
  stopped, with its process group, after `timeout_s` (default 900, at most 86400), on cancel, or when the link ends.
- Running an action is admin-only: `POST /api/v1/scripts/<host>/<script>/<action>` (optional body `{"arg": "model-b"}`) answers
  `202` with a job; follow it with `GET /api/v1/scripts/jobs/<id>` or the SSE `.../stream` (events `line` and `end`). Output is
  scrubbed of secret-looking values; the job keeps the last ~20 kB.

| Code (HTTP) | Meaning |
|---|---|
| `unknown_script` (404), `unknown_action` (404) | The name is not in that node's policy (as reported at its last hello: Reconnect after editing). |
| `arg_required` (400), `invalid_arg` (400) | The action needs one of its `args`, or the value is not allowed. |
| `script_busy` (409) | A run of that script is already in progress; wait. |
| `monitor_only` (409) / `node_unavailable` (503) | The node is in monitor mode, or its link is down. |
| `instance_not_managed` (409) | You tried an unsupported action on a script or external instance (for example remove, logs, or restart of an external one). |
| `not_found` (404) with `the script program '/x' does not exist on this machine` | The absolute path in `argv` is wrong. |
| `timeout` (504) with `the script ran longer than its timeout_s (N s) and was stopped` | Raise `timeout_s` or fix the script. |

## External engines (monitor-only, or stop and start)

Each node lists the engines it did not start: running containers (classified by image and command signatures: vLLM, SGLang,
Ollama, llama.cpp, TensorRT-LLM, Dynamo; routers, UIs, proxies and databases are never listed) and loopback ports from
`discovery_ports` that answer like an OpenAI-compatible server. They appear as `ext-<host>-<slug>` (each `-` of the host id doubled, so `gpu-2` gives `ext-gpu--2-...`), `management: "external"`.
Probes are GET-only on `127.0.0.1` with short timeouts, a 1 MB cap and no redirects; container **environment values are never
read** and command-line secrets are redacted on the node. States: `ready`, `loading` (HTTP 503), `auth_required` (HTTP 401 or 403:
metrics and chat unavailable), `unreachable` (no published loopback port answers, or none is published: publish one on
`127.0.0.1` to monitor it) and `stopped` (shown for 30 s after a container disappears).

By default nothing can change them: start, stop and remove answer `409 instance_not_managed`. With `"external_control":
"stop_start"` in that node's policy (and manage mode) the console may **stop** (`docker stop -t 30`) and **start** a recognised engine
container that is neither console-owned nor claimed by a script; never restart, remove or read its logs. A benchmark sends real
load, so `POST /api/v1/bench` on an engine the console did not create needs `"confirm_external": true` (otherwise `400
confirm_external_required`). The launch form can be prefilled from an external engine's parsed command line ("clone as managed").

## Downloads and copies

`POST /api/v1/downloads` with `{"repo_id": "org/model", "host_id": "gpu-box-2", "root_id": "hf", "allow_patterns": [...]}`
downloads a Hugging Face repository (or some files; `GET /api/v1/hf/models/<repo>` returns `gguf_groups` to pick one quant) into
a **writable** root of that machine. With no `root_id` the machine's first writable Hugging Face root is used. Where files land:

| Target root kind | Layout |
|---|---|
| `hf-home`, `hf-hub` | the Hugging Face cache layout: `blobs/<sha256>`, `snapshots/<commit>/<path>` as relative links, `refs/<revision>`; the links and ref are created only after every blob is in place, so a half-downloaded repository never looks like a model |
| `gguf` | the files under `<org>--<name>/<path>` (the root must receive at least one `.gguf`) |
| `auto` | what its last scan showed (a cache stays a cache, loose model files get files, an empty root takes a cache for a repository) |
| `plain`, `ollama`, `other` | not for Hugging Face downloads (`root_kind_unsupported`); `plain` takes only a copy of a plain model directory |

How it flows: the hub lists the repository's files (with a sha256 for each LFS file) through its allowlisted client, asks the node
which blobs and partial files already exist (`fs.stat`), then streams each missing file from Hugging Face into the node's
`fs.write`. The node writes `<file>.incomplete`, checks the declared size, hashes the whole file against the expected sha256 when
the hub knows one, and renames it into place; on a mismatch the partial is deleted. The node never fills a disk below 1 GiB free,
and the hub checks the target mount's free space first (2 % margin). Two transfers run at once, at most 50 rows are unfinished.

**Copies.** `POST /api/v1/library/copies/<copy id>/copy` with `{"host_id": "edge-1", "root_id": "gguf"}` copies one library
copy to another machine's root (or another drive of the same machine). Bytes go `fs.read` on the source node, through the hub,
into `fs.write` on the target node, so every byte crosses the hub twice. What can be copied: a Hugging Face cache folder (the
large LFS files move machine to machine; the hub fetches the few small files and the file listing from Hugging Face, so the hub
must reach it), a loose `.gguf` file or split set, and a plain model directory whose scan counted every file (found by its usual
file names and proved complete). An Ollama store or an `other` folder cannot be copied. A GGUF file or plain directory copied
between machines, and a repository's small non-LFS files, are checked for size only (there is no expected hash).

**Resume, pause, cancel.** A transfer interrupted for any reason keeps its `.incomplete` partial on the target; **Resume** continues
from the partial's size (the node insists the offset equals the partial). Pause keeps the partial. **Cancel** asks the node to
delete the file in flight, but only if that node's policy has `allow_delete`; otherwise the partial stays and a later download of
the same file resumes from it. When the console restarts, rows left `running` or `queued` become `paused`. A download row from 0.3
can be resumed only if the local node is in manage mode and has a writable Hugging Face root. A node that stops accepting data for
120 s fails the row with `the machine stopped accepting data; resume to continue`.

| `code` (HTTP) | Meaning and fix |
|---|---|
| `monitor_only` (409) | `this machine's node policy is monitor-only, so nothing can be written to it`: manage mode first. |
| `unknown_root` (422) | `<label> has no model root '<id>'`: the policy (as the hub last read it) has no such root; Reconnect after editing. |
| `root_not_writable` (409) | The root is `"writable": false` in that policy. |
| `no_writable_root` (409) | `<label> has no writable model root that can hold ...`: add or enable one. |
| `root_kind_unsupported` (422) | The root's kind or layout cannot hold this (see the table above). |
| `copy_same_location` (409) | The target root is where the model already is, or storage it shares. |
| `copy_unsupported` (501) | An Ollama store, an `other` folder, or a plain directory without exact counts cannot be copied. |
| `copy_listing_incomplete` (501) | A plain directory has files the protocol could not find by name (a subfolder, an unusual name, or it changed since the scan): rescan, or copy another way. |
| `copy_source_mismatch` (409) | The source copy is at another revision or incomplete; a shard is missing. |
| `copy_source_gone` | The source copy is no longer in the library. |
| `download_exists` (409) | The same download or copy to that location is already queued, running or paused. |
| `download_queue_full` (409) | 50 unfinished rows: cancel or finish some. |
| `insufficient_disk` (507) | `<label>: need X GB but only Y GB free at <root path>`, or the node's `no_space` (less than 1 GiB would remain). Free space on that mount. |
| `hash_mismatch`, `size_mismatch` (502) | The bytes did not match; the partial was deleted. Retry; repeated failures point at the source or the network. |
| `hf_gated` (403) | `Hugging Face denied access to <repo>: the model is gated or private. Set HF_TOKEN and accept the license at ...`. |
| `egress_denied` (403) | `egress to <host> is not on the HF allowlist` or `the egress proxy blocked this host; add it to the proxy's allowlist`. |
| `egress_proxy_unavailable` (503) | `TLS verification failed through the egress proxy: set EGRESS_CA_BUNDLE ...` or `egress proxy unreachable: refusing to connect directly`. |
| `no_files`, `no_gguf_files` (400) | `allow_patterns` matched nothing; a `gguf` root needs a `.gguf` among the files. |
| `io_error` (502) | The node could not write; for example `cannot write ...: Read-only file system` (see the systemd section). |

## Shared storage (NFS)

When a machine's model root is an NFS mount, the node reports the export's server and path in its `disks`. If that server is the
address (or hostname) of **another configured host**, and the export path covers the folder, then host B's copy and host A's are
the same bytes: the library lists A's copy once, with `visible_on` naming both machines, and an engine started on B mounts it
through B's own NFS mount. A copy onto the same storage is refused (`copy_same_location`). Rules and limits:

- The match needs the server's address to be one the owner reports in its `hello` addresses (loopback, link-local and container
  bridge interfaces are ignored), or its hostname. An address that two hosts both report (such as a docker bridge `172.17.0.1`)
  identifies nobody.
- If the exporting machine is **not** a configured host, B's copy stays as its own entry, with the NFS source shown (`shared_from`).
- Deleting a shared copy deletes the real files for everyone: the owner's node does it, and the in-use check looks at every machine
  that can see them (the owner's path and every NFS alias).
- A root on NFS is normally read-only (`"writable": false`): downloads and copies into it are refused unless you make it writable.

## The library

`GET /api/v1/library?host=&format=&q=` lists every model once, grouped by `key` (`hf:<repo id>`, `gguf:<repo or file>:<quant>`,
`ollama:<name:tag>`, `plain:<dir>`), with its copies, sizes, drives and the engines that can load the format. A machine that is
down shows its last-known library (kept in SQLite) with the warning `<label> is <state>: showing its last-known library, which
cannot be changed until it reconnects`; other warnings are `copies are at different revisions: ...` and the node's own (for
example `1 file(s) still downloading`).

**Deleting a copy** (`DELETE /api/v1/library/copies/<id>`, admin) needs: the machine's link up, manage mode, `allow_delete`, a
writable root, and no running engine that uses the copy. An engine uses a copy if its container mounts the copy itself (or part of
it), or mounts a folder that holds it (a shared cache) **and** loads that very model (its parsed model arguments or label name
the repository, the cache folder or the path). A container whose model argument cannot be parsed counts as using everything it
mounts, so a delete is refused rather than pulled from under a running engine. The node also refuses `fs.delete` while a running
container mounts the target or anything under it. Never a root itself or a folder that holds another root.

| `code` (HTTP) | Meaning |
|---|---|
| `copy_not_found` (404) | The id is not in the library (rescan; ids are stable per machine, root and path). |
| `node_unavailable` (503) | `the machine is not connected (<state>)`. |
| `policy_denied` (403) | `this machine's policy does not allow deleting models` (`allow_delete`), or `that model root is read-only in this machine's policy`. |
| `monitor_only` (409) | `this machine's policy is monitor-only`. |
| `model_in_use` (409) | `the model is used by <instance ids>; stop it first`. |
| `root_not_deletable` (409) | The entry is a whole model root. |
| `in_use` (409) | The node refused: `a running container mounts '<path>'; stop it before deleting`. |

## Egress proxy (recommended, opt-in)

By default the hub reaches Hugging Face directly and `/api/v1/health` reports `egress_mode: direct` with a warning. Any
allowlisting HTTP(S) proxy works. A minimal option is the upstream mitmproxy container with the small addon in
`deploy/egress-proxy/` (an example allowlist of Hugging Face hosts plus a 20-line addon that answers 403 "egress blocked" for
everything else). One real model download through an allowlisting proxy was verified live; the proxy and addon in this repository
were not run by the author of this document (unverified).

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

The console trusts only the certificate in `EGRESS_CA_BUNDLE`; it never routes around the proxy (`trust_env=False`). If the proxy
is down or the CA is missing, Hugging Face calls fail with `503 egress_proxy_unavailable`; there is no direct fallback. A 403
"egress blocked" means the host is missing from the proxy allowlist. Keep `HF_ALLOWED_HOSTS` and the proxy allowlist in step. Since
downloads stream through the hub, the proxy now carries every model byte, and it sees request headers including the Hugging Face
token: treat its logs as secrets. Nodes do not use the proxy; they need no internet.

## Letting another container reach an engine (opt-in, per machine)

Engines sit on the internal engine network of **their own machine** with no published ports, so other containers cannot see them
by default. To let, for example, an OpenAI-compatible router on that machine reach one, join the router to that machine's engine
network in its own compose file (the console never edits other services):

```yaml
services:
  router:
    networks: [default, engines]
networks:
  engines:
    name: engine-console-engines         # the policy's engine_network on that machine
    external: true                       # created by the console on first launch (--internal)
```

Use the instance's `internal_endpoint` (`http://<container>:<port>`, in `GET /api/v1/instances/<id>`) as that service's upstream
URL. The network must exist first (launch one instance there, or `docker --context <ctx> network create --internal <name>`).
Effect: that engine can also reach the other container. Not exercised (unverified).

## Reading host memory

Each node reads memory on its machine from `/proc/meminfo` and, per engine container, from the container's cgroup v2
`memory.current` and `memory.stat` (the counters `docker stats` uses); the hub only displays it. The cgroup path comes from
`/proc/<pid>/cgroup`, must match a strict character set, contain no `..`, and resolve under `/sys/fs/cgroup`. Hosts on cgroup v1
fall back to summed process memory, labelled `rss` (fakes only). Per-container cgroup memory was verified live on one workstation.

| Term | Meaning |
|---|---|
| anon | Memory the engine allocated for itself (heap, pinned buffers). Cannot be dropped without killing or swapping the process. |
| cache | File-backed page cache (for example model weight files); the kernel reclaims it under pressure. |
| shm | Shared memory: tmpfs, `/dev/shm` and CUDA IPC buffers. Accounted like cache but **cannot be reclaimed**; multi-GPU engines can hold tens of GiB. |
| kernel | Kernel structures charged to the container (slab, page tables, stacks). Usually small. |

Because shm hides inside "cached" figures, free-memory tools can look healthy while little is really available: use *available*,
not *free*. Alerts: `host_ram_low` warns under 10 % available and is critical under 5 %; `shmem_high` warns above 25 % shared
memory; `swap_heavy` warns above 50 % swap used. Series: `GET /api/v1/metrics/system?window=15m&host=<id>`. See the same numbers
by hand on that machine (the cgroup path differs between systemd and cgroupfs drivers):

```bash
pid=$(docker --context rootless inspect -f '{{.State.Pid}}' <container>)
cat /proc/$pid/cgroup                       # e.g. 0::/user.slice/.../docker-<id>.scope
cat /sys/fs/cgroup/<path>/memory.current    # total bytes
grep -E '^(anon|file|shmem|kernel|slab) ' /sys/fs/cgroup/<path>/memory.stat
free -h; grep -E 'MemAvailable|Shmem:|SwapFree' /proc/meminfo
```

cache = `file` minus `shmem`. When an instance's `host_memory` is null: the node's user cannot read that cgroup, the machine is on
cgroup v1 and `/proc/<pid>` is unreadable, the container is not on that machine, or the instance is stopped.

## Prometheus and Grafana

`GET /metrics` needs a bearer key (a `viewer` key is enough) and the scrape's Host header must be one the console accepts
(`127.0.0.1:8791` or `localhost:8791`, else add the name to `ENGINE_CONSOLE_ALLOWED_HOSTS`). A Prometheus job (not run by the author of this
document):

```yaml
scrape_configs:
  - job_name: engine-console
    metrics_path: /metrics
    authorization: {type: Bearer, credentials_file: /etc/prometheus/engine-console-viewer.key}
    static_configs: [{targets: ["127.0.0.1:8791"]}]
```

New gauges: `engine_console_host_up{host}`, `engine_console_host_gpu_used_gib{host,gpu}`,
`engine_console_instance_vram_gib{instance,host,gpu}`; the v1 gauges (`engine_console_host_*_gib`,
`engine_console_instance_host_ram_gib{instance}`, `engine_console_instances{state}`, request counters) describe the console's own
machine. `deploy/grafana-engine-console.json` is a dashboard template (opt-in; it predates the per-host gauges).

## systemd user unit (opt-in; updated for 0.4, never started)

`deploy/engine-console.service` is a template; nothing installs it for you, and it has never been started. It was parsed with
`systemd-analyze --user verify` on a filled-in copy (2026-10-14).

```bash
mkdir -p ~/.local/share/engine-console ~/.config/engine-console ~/.local/state/ecnode
chmod 700 ~/.local/share/engine-console ~/.local/state/ecnode
# edit every line marked CHANGEME first, including one ReadWritePaths entry per writable model root
cp deploy/engine-console.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now engine-console
journalctl --user -u engine-console -f          # JSON logs
systemd-analyze --user security engine-console  # inspect exposure
```

`ProtectSystem=strict` makes everything read-only except `ReadWritePaths`; if the unit fails with a `NAMESPACE` error one of those
paths does not exist: create it or edit the list. `loginctl enable-linger $USER` lets it run without an open session.

What the 0.4 template does:

- The local node is a child of the console, so it inherits the unit's sandbox and can write only to `ReadWritePaths`. The
  template lists the data dir, `~/.local/state/ecnode` (where the node writes the config files that llama.cpp templates and
  TensorRT-LLM option files need), the backend venv and the uv cache, plus a `/CHANGEME/writable-model-root` placeholder that makes
  the start fail until you replace it with **one line per writable model root** of the local policy (`ecnode check` lists the
  roots and which are writable). A missing root shows up as `io_error` (`cannot write ... Read-only file system`) on a download.
- `PrivateUsers=yes` is explicit: in a user unit, the namespacing options run the service in a user namespace where only your
  own uid and gid exist. A root-owned Docker socket (rootful Docker, the `docker` group) is then unreachable, so the unit
  **needs rootless Docker**; the rootless socket under `$XDG_RUNTIME_DIR` is yours and works.
- `ProtectProc=` and `ProtectControlGroups=` are not set: systemd does not support them in per-user units.
- `UMask=0077`: files the hub creates are 0600, and so are the model files the local node downloads. Rootless engines still
  read them (container root is you); a root shared with other users or machines needs `UMask=0022`.
- The node runs `docker` and the hub runs `ssh` from the unit's `PATH`; set `Environment=PATH=...` if yours live elsewhere. The
  environment whitelist passes `HOME`, `PATH`, `USER`, `LOGNAME`, `LANG`, `XDG_RUNTIME_DIR`, `DOCKER_HOST`, `DOCKER_CONTEXT` and
  `DOCKER_CONFIG` from the hub to its node (which does not hand `DOCKER_CONFIG` on to docker), so keep `Environment=DOCKER_CONTEXT=...` if the policy names no context. If the
  journal shows the docker CLI failing to write under `~/.docker`, add it to `ReadWritePaths`.
- A reverse proxy in front of the console needs its hostname in `ENGINE_CONSOLE_ALLOWED_HOSTS`.

Not verified, because the unit has never been started: the real rootless socket and NVML device access inside the user
namespace, whether the docker CLI writes under `~/.docker`, and how `ProtectSystem` behaves in a user unit on a given kernel.

## Backup

State to protect on the hub: `<data_dir>/console.db` (+ `-wal`, `-shm` while running), `<data_dir>/secrets/` (engine secret
params, plaintext), `bootstrap-admin.key` if you kept it, `~/.config/engine-console/hosts.json`, **the hub ssh key**
(`~/.config/engine-console/ssh/id_ed25519`), and `~/.config/engine-console/hf_token` and `env`. On each machine: its policy
(`/etc/ecnode/policy.json`) and, if you edit it, the scripts it names. Model files are re-downloadable and are not part of this
backup; the library snapshots in the database are a cache and rebuild from the nodes. These are secret-bearing (conversations,
key hashes, tokens, an ssh key): write only to a directory outside version control, encrypt fail-closed (never leave a plaintext
copy if encryption fails), use `umask 077`, and never print contents.

Consistent copy of the DB while running (SQLite online backup via Python, no `sqlite3` CLI needed):

```bash
umask 077
python3 - <<'PY'
import sqlite3, pathlib
src = sqlite3.connect(pathlib.Path.home() / ".local/share/engine-console/console.db")
dst = sqlite3.connect("/path/to/gitignored/dir/console.db.bak")   # CHANGEME
src.backup(dst); dst.close(); src.close()
PY
```

Then encrypt it (`gpg --symmetric` or `--encrypt -r <key>`) and remove the plaintext copy; if encryption fails, delete the partial
output. Restore: stop the service, put the file at `<data_dir>/console.db`, delete stale `-wal`/`-shm`, start.

## Upgrade

**Hub.**

```bash
git pull                      # or check out the new revision
make setup                    # uv sync --frozen against the committed uv.lock (also installs the node package)
make all                      # lint, typecheck, tests (offline)
systemctl --user restart engine-console
```

Migrations are ordered SQL files applied at start (`backend/src/engine_console/migrations/`); take a database backup first and
treat downgrades as unsupported (no down-migrations). Upgrading from 0.3 applies `0003_fleet` (instances gain `host_id`, which is
`local` for old rows, plus `model` and `image_req`) and `0004_library_and_transfers` (`library_snapshots`; downloads gain host,
root, kind and bookkeeping columns). Engines are containers and survive a console restart; on start the console adopts labelled
containers on each host as its link comes up. The breaking changes of 0.4.0 are listed first in `CHANGELOG.md`.

**Nodes.** Re-run `sudo node/install.sh` from the new revision on each machine: the code is swapped atomically, the policy is never
touched. Hub and node match as long as their protocol version (`PROTO`) matches; a mismatch shows as `incompatible` with the side
to update. A node that is running keeps the code it loaded (unverified); Reconnect the host to start the new one. Bumping an engine
image is an image pin plus a new adapter catalog: pins mean tested, not frozen: validate a bump against a copy of your database first.

## Uninstalling a node

```bash
sudo node/install.sh --uninstall   # removes /usr/local/lib/ecnode and keeps /etc/ecnode
```

By hand, if you want it all gone: remove the key line from the login user's authorized keys; `sudo rm -r /etc/ecnode`;
`rm -r ~/.local/state/ecnode`; remove the host from `hosts.json`. **Not removed:** engine containers and their gateways that the
console created (`docker --context <ctx> ps -a --filter label=engine-console=1`, then `docker rm -f` them) and the engine network
(`docker network rm engine-console-engines`).

## Data and configuration locations

| What | Where | Notes |
|---|---|---|
| Hub database | `<data dir>/console.db` (+ `-wal`, `-shm`) | 0600; includes the library snapshots |
| Engine secret params | `<data dir>/secrets/<instance>.json` | 0600, plaintext |
| Bootstrap admin key | `<data dir>/bootstrap-admin.key` | 0600; delete after copying |
| Desktop launcher state | `<data dir>/console.log`, `console.pid`, `console.lock` | 0600; written by `engine-console open` |
| Desktop launcher | `~/.local/share/applications/engine-console.desktop` (+ a copy on the desktop) | `make install-desktop[-icon]` |
| Host list | `~/.config/engine-console/hosts.json` (`HOSTS_FILE`) | owned by you, not group/world-writable |
| Hub ssh key | `~/.config/engine-console/ssh/id_ed25519` (+ `.pub`) (`HUB_KEY_FILE`) | 0600, no passphrase |
| known_hosts | your default ssh files | read by the hub's ssh, never written by it |
| HF token | `~/.config/engine-console/hf_token` (`HF_TOKEN_FILE`) | 0600 |
| Env file | `~/.config/engine-console/env` | optional |
| Node code (remote) | `/usr/local/lib/ecnode/` | root-owned; launcher `ecnode` and `lib/ecnode/` |
| Node policy | `/etc/ecnode/policy.json`, else `~/.config/ecnode/policy.json` | `--policy` overrides |
| Node state | `~/.local/state/ecnode/instances/<instance>/files/` | 0700; generated engine config files, removed with the instance |
| Compile caches | docker named volumes `ec-vllm-cache`, `ec-flashinfer-cache` | on the machine that runs vLLM |
| Gateway ports | `127.0.0.1:18000-18099` per machine | the policy's `port_range` |

## Troubleshooting

Always use the docker context the node's policy names (for example `docker --context rootless ...`) when you check a machine by
hand; the default context may point at a different daemon.

The node does not hand its own environment to docker. `DOCKER_CONFIG`, `DOCKER_TLS_VERIFY`, `DOCKER_CERT_PATH`, proxy variables
and `SSH_AUTH_SOCK` do not reach the docker CLI. A daemon that needs them (TLS over `tcp://`, or `ssh://` with an agent) must be
reached through a docker context named in the policy (`docker.context`); the CLI reads its contexts from `~/.docker`. Node refusals reach you as RFC 7807 problems:
`{"status": 409, "code": "monitor_only", "detail": "<host label>: <the node's sentence>", "host_id": "gpu-box-2"}`. The `code` is the
node's own; the status mapping is:

| Node `code` | HTTP | Typical cause |
|---|---|---|
| `node_unavailable`, `busy` | 503 | link not up; the node is already running 64 requests |
| `monitor_only`, `conflict`, `in_use`, `port_in_use`, `cancelled` | 409 | monitor mode; a name, network or upload clash; a model mounted by a running engine; a taken port |
| `policy_denied`, `not_owned` | 403 | the policy forbids it; the container was not created by the console |
| `not_found` | 404 | no such container, image, root or file |
| `no_space` | 507 | less than 1 GiB would remain |
| `hash_mismatch`, `size_mismatch`, `docker_error`, `http_error`, `io_error`, `internal` | 502 | bad bytes; docker said no; the engine did not answer; a file operation failed; a node bug |
| `timeout` | 504 | the action may or may not have taken effect |
| `unsupported` | 501 | for example `the model scanner is not running on this node` |
| `bad_args`, `unknown_op` | 500 | a hub bug (logged): please report it |

### Start-up and access

| Symptom | Likely cause / check |
|---|---|
| `engine-console: <path>: <reason>` and the process exits with status 2 | `hosts.json` is not acceptable (see "The host list"); the message says what to fix. |
| `401 unauthorized`, detail `a valid bearer API key is required` | Send `Authorization: Bearer <key>`; a request carrying `X-Forwarded-For` is never loopback-trusted. The key may be revoked. |
| `403 forbidden`, `this key has the viewer role (read-only)` or `admin role required for this action` | Use an admin key. The Hugging Face search and model-card routes (`/api/v1/hf/*`) are admin-only: they spend the operator's HF token. A viewer's `POST /fit` judges only copies a machine already holds (`422 model_info_unavailable` otherwise). |
| `403 csrf_header_required` | A non-GET request without a bearer key needs `X-Engine-Console: 1`. |
| `403 cross_origin` | The request came from another origin. |
| `421 misdirected_request`, `unrecognised Host header` | The Host header is not `127.0.0.1:8791` or `localhost:8791`; add the name to `ENGINE_CONSOLE_ALLOWED_HOSTS`. |
| `413 payload_too_large`, `415 unsupported_media_type` | Body over the cap; profile import needs a JSON or YAML content type. |
| `429 too_many_streams` | 8 SSE streams are open on the console, or 4 for this key (`scope` says which); close one. |
| `429 too_many_chat_streams` | 16 streamed chat replies are running, or 4 for this key; wait for one to finish. |
| `404` on `/api/docs` | There is no Swagger page; the OpenAPI JSON is `/api/v1/openapi.json` and needs a key. |
| `404 unknown_host`, `no host '<id>' is configured` | The id is not in `hosts.json`. Calls that omit `host` mean `local`. |
| `409 host_disabled` on Reconnect | The host is `disabled`. |
| No traces in your tracing backend | Collector not up on `localhost:4317`; export failures are silent by design. `OTLP_ENABLED=false` silences. |
| GPU stats missing for a host | `GET /api/v1/hosts/<id>` shows `node.gpu_source` (`nvml`, `nvidia-smi` or `none`); run `ecnode check` on that machine for the reason (no driver, `nvidia-smi` not on `PATH`, NVML failing). Without `pynvml` the node reads `nvidia-smi`. |
| `the node has not sent its first sample yet; try again in a few seconds` | Right after a connect; wait. |

### Policy, library and transfers

See the tables in "Policy errors you may see", "Launch checks", "Downloads and copies", "The library" and "Script-managed
engines". Extra checks:

| Symptom | Likely cause / check |
|---|---|
| A host shows monitor mode although you set `"manage"` | The file was refused (see `policy.error`; `ecnode check` on that machine prints it), or the hub has not re-read it: Reconnect. |
| The library lacks models that are on disk | The folder is not in a policy root, the root's `kind` does not match the layout (try `auto`), the scan hit its budget (warning on the model or `did not finish`), the machine is down (last-known library), or the files sit under a folder named like a credential store (`.config`, `.local`, `.ssh`, ...) inside the root, which is never read. Run `POST /api/v1/library/rescan`. |
| `model_missing` for a model you can see in the library | It is on another machine. Download or copy it to the machine you launch on (or share the storage). |
| Download fails `no_space` or `insufficient_disk` | Free space under the target root's mount. |
| Engines on a machine cannot be controlled, `monitor_only` | Manage mode on that machine's policy. |
| `docker_error`, `docker run failed: ...` | Read the rest of the line. `cannot reach the docker daemon` means the node's user cannot use docker (or the policy names a wrong context); `the image is not on this machine; pull it first` means pull it there; `the host port is already in use` is retried on the next port; `a container with that name already exists` is a clash. |
| `conflict`, `docker network '<name>' exists but is not --internal` | A network with the policy's `engine_network` name exists without `--internal`; remove or rename it, or name another in the policy. |
| `http_error`, `nothing is listening on 127.0.0.1:N (is the engine running?)` | The engine is not up, or its gateway is gone. |
| `policy_denied`, `port N is not an engine endpoint or console gateway on this node` | The node's inventory does not list that port (yet): wait one cycle, or the instance's gateway is missing. |
| `timeout`, `the engine on port N did not answer within X s` | A long generation; raise the client's timeout or check the engine. |
| Fit says `wont_fit` although the model should fit | Free VRAM is counted: check `fits_if_stop` and other resident engines; another GPU process also holds VRAM. Estimates are uncalibrated (FIT-ESTIMATOR.md). |
| Gated model error | Set `HF_TOKEN` (file preferred) on the hub, accept the licence on huggingface.co. |
| Port conflict | The console picks a free gateway port from the node's range; check for stale containers with `docker --context <ctx> ps -a --filter label=engine-console=1`. |
