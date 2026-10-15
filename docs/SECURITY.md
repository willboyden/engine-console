# Security

Basis: reading the code of this revision (0.4.0, contract v2), the offline test suites, and the live checks listed below.
No penetration test, fuzzing or independent review has been done. Status words used here: **verified live** (run against
real software on real hardware, and by whom), **fakes only** (covered by tests that use fake docker, fake nodes or fake
networks), **unverified** (read from code only).

**What was exercised live (by the maintainers, on one workstation with two GPUs and rootless Docker, during the 0.4.0
build):** a request without an API key gets 401; the CSRF header is enforced; a managed vLLM launch under `--cap-drop ALL`,
`no-new-privileges`, a pids limit, the internal network, a read-only model mount and the loopback gateway; one real Hugging
Face download through an allowlisting egress proxy into a writable root; a library delete. An earlier self-test on rootless
Docker (a test container, before this version) had verified the internal network, an engine with no egress and no published
ports, and the gateway on 127.0.0.1.

**Not verified:** any ssh login to a remote node, a restricted key line in a real sshd, a hostile node or hostile hub against
the real implementation (the tests use an in-process fake node that speaks the real frames), tensor-parallel engines under
`--cap-drop ALL`, the proxy's failure paths (CA trust, proxy down) against a real proxy, the frontend (still being finished)
against the real backend, and how a real browser treats the CSP and the sign-in form.

## Assets

| Asset | Where | Protection |
|---|---|---|
| **Hub ssh key** (new) | `~/.config/engine-console/ssh/id_ed25519` and `.pub` on the hub's machine (`HUB_KEY_FILE`) | Created by `hosts init-key` with umask 077 (directory 0700, key 0600), no passphrase, never printed or sent; passed to ssh as a path that is masked in error text. The remote side restricts it to the node command and to the hub's address. |
| **Hosts file** (new) | `~/.config/engine-console/hosts.json` (`HOSTS_FILE`) | Decides what the hub runs `ssh` against: read once at start-up, with `O_NOFOLLOW`, refused unless it is a plain file owned by the running user, not group- or world-writable, at most 64 KiB, strict schema, no duplicate keys. |
| **Node policies** (new) | `/etc/ecnode/policy.json` (root-owned) or `~/.config/ecnode/policy.json` on each machine | The hard boundary of that machine. A file the node does not accept means monitor mode. The hub can read a summary and never change it. |
| Node code | `/usr/local/lib/ecnode` (root-owned) on remote machines; the hub's virtual environment for `local` (an editable install of the repository's `node/` folder) | Installed by `node/install.sh` with an atomic swap and a launcher that runs `python -I`. The local node runs from the hub's own environment, which the hub's user can edit. |
| API keys | SQLite `api_keys` (SHA-256 of a 256-bit secret), `<data_dir>/bootstrap-admin.key` (0600) | Shown once; roles admin and viewer. |
| Hugging Face token | `HF_TOKEN` or `~/.config/engine-console/hf_token` (0600), hub only | Used only in the `Authorization` header of allowlisted requests. It never goes to a node or an engine. Spent only on behalf of admin keys: `/hf/*` is admin-only, and a viewer's fit estimate reads only local copies. |
| Engine secret parameters (an API key for an engine) | `<data_dir>/secrets/<instance>.json` (0600); `[set]` in the database and API | Travel from the hub to the node inside the ssh session and reach docker only through the child process environment (`-e NAME`), never argv. |
| Model files | The model roots of each node | Confined to the policy's roots; written through `.incomplete` partials and renamed after verification. |
| Library snapshots, conversations, usage, audit | SQLite `console.db` | 0600 file; the audit table rejects UPDATE and DELETE at the database level. |
| known_hosts | The hub user's default ssh files | Read by the hub's ssh, never written by it. |
| Node state | `~/.local/state/ecnode/instances/<instance>/files/` (0700) | Generated config files (a chat template, a TensorRT-LLM options file) mounted read-only into the engine; no secrets are placed there. |

## Trust boundaries

```
[browser / curl] --HTTP, Bearer key--> [hub process, user uid, 127.0.0.1:8791]
    Host allowlist, Origin check, CSRF header, bearer auth + roles, body and SSE caps, CSP and other headers
                                          |-- pipe (local) -------------> [node "local": ecnode --stdio, whitelisted env]
                                          |-- ssh, hub key, forced cmd -> [node on each other machine, login user]
                                          |       (a node is UNTRUSTED data to the hub: whitelists, caps, flood and credit limits)
                                          |-- HTTPS via EGRESS_PROXY ---> [allowlisting proxy] --> huggingface.co + CDNs
                                          |-- gRPC, plaintext ----------> [OTel collector localhost:4317]
                                          `-- files: data dir, hosts.json, hub key, hf_token
[node] --argv lists, --context from policy--> [docker daemon of the login user]
[node] --realpath + O_NOFOLLOW--> [model roots from the policy]
[node] --loopback http.request--> [gateway 127.0.0.1:18000-18099] --TCP--> [engine on the --internal network: no ports, no egress]
[other container, e.g. a router] --opt-in: operator joins that machine's engine network--> engine
```

1. **Client to hub.** Host allowlist, then Origin and CSRF checks, then a bearer key. Loopback is not trusted by default
   (ADR-0021).
2. **Hub to node** (new). The hub starts the node itself: a pipe for `local`, ssh for the others. Nothing listens on either
   side. The hub treats everything a node says as untrusted data; a node treats the hub as its operator but obeys its policy
   first (ADR-0014, ADR-0015, ADR-0016).
3. **Node to docker and the filesystem** (new). The node runs the docker CLI with argv lists and a context named by the
   policy, touches only containers labelled `engine-console=1`, and confines every file operation to the policy's roots.
4. **Engine to anything else.** No route (internal network, no published ports). Weights come from a read-only mount.
5. **Hub to the Internet.** Hugging Face only, through the proxy when `EGRESS_PROXY` is set (ADR-0012), plus the in-app
   allowlist. Nodes need no internet access at all.
6. **Model-derived and node-derived content** (chat output, model cards, logs, container names, library entries) is
   untrusted data rendered in the UI. The contract says it is never rendered as HTML; the frontend side is unverified.

## Controls

| Control | Where | Notes |
|---|---|---|
| Policy as the boundary: monitor refuses every op outside `proto.MONITOR_OPS`; manage allows only what the policy lists | `node/.../policy.py`, `server.py` | A policy the node refuses means monitor mode; unknown keys are errors; re-checked every ~2 s. |
| Whitelisted request arguments (`proto.clean_args`) before any handler, then policy checks | `proto.py`, `server.py` | Wrong types become null; strings and lists are capped. |
| Pushed node messages rebuilt from a whitelist, numbers finite, strings capped; unknown GPU UUIDs in a sample are dropped | `proto.sanitize_*`, `services/fleet.py` | |
| Per-link limits: JSON frames and empty DATA frames over 20 MB or 50,000 messages in 10 s drop the link; DATA is bounded by the credit the hub grants; hello within 25 s; silence over 30 s drops the link | `services/fleet.py` | |
| ssh: own key, `-F none`, `IdentitiesOnly`, `BatchMode`, `StrictHostKeyChecking=yes`, `UpdateHostKeys=no`, no agent, no forwarding, no passwords, scrubbed environment; never starts without its key | `services/hosts.py` | The command line was checked with the real `ssh -G` (offline). |
| Remote key line `restrict,from=...,command=...`; the hub notes an unrestricted login (`forced_cmd` false) | `hosts.py`, `node/install.sh` docs | Fakes only for the note; no real sshd was involved. |
| Local node environment whitelist (`HOME`, `PATH`, `USER`, `LOGNAME`, `LANG`, `XDG_RUNTIME_DIR`, `DOCKER_HOST`, `DOCKER_CONTEXT`, `DOCKER_CONFIG`) | `services/hosts.py` | The hub's own environment (token, API keys) never reaches the node. |
| Run spec validation: container name, image (policy match with a literal repository), mounts as `root` + relative path inside the root's realpath (never a host path, no `:` or `,`), GPU UUIDs the node reported, label keys `engine-console.*`, env names, `shm_size` at most `max_shm_gib`, `--ipc host` only if `allow_ipc_host`, at most four generated files, ulimits `stack` and `nofile` only | `node/.../docker.py` | |
| Containers: `--cap-drop ALL`, `--security-opt no-new-privileges:true`, `--pids-limit 4096`, `--restart no`, `--pull never`, the policy's `--internal` network (an existing network that is not internal is refused), every model mount read-only (`ro: false` is refused), and no mount whose source holds a key- or token-looking name at its top or one folder down (the hub mounts only `hub/` of an HF_HOME root) | `docker.py`, `services/runspec.py` | Verified live for one vLLM instance on one GPU. No memory or CPU limit is set. |
| Gateway: nginx from the policy's `gateway_image`, read-only, `--cap-drop ALL`, user 101, pids limit 256, publishes only `127.0.0.1:<port in port_range>`; its config holds only a checked container name and port | `docker.py`, `ops.py` | |
| Ownership: stop, start, remove and logs only for containers labelled `engine-console=1` (and the right role and instance); `docker.external` only for recognised engine containers that are neither owned nor claimed by a script | `docker.py`, `ops.py` | `not_owned` otherwise. |
| The docker CLI runs with an environment built from scratch (`util.child_env`) plus the `secret_env` values; a `secret_env` name must look secret (key, token, secret, password) and must not change how the docker CLI runs (`LD_*`, `DOCKER_*`, `OTEL_*`, `PATH`, proxies, ...); a plain `env` name that looks secret is refused | `util.py`, `docker.py` | |
| Argv of other containers redacted on the node (`--api-key X`, `--token=X`, `NAME=X`, inside `sh -c` strings too); container environment values never read; logs, script output and errors scrubbed; logs only for owned containers | `util.py`, `discovery.py`, `ops.py` | Offline test with a fake container whose environment holds a marker value. |
| File confinement: relative paths only, no `..`, backslash or control characters; parent `realpath` inside the root and the directory fd re-checked; final component opened with `O_NOFOLLOW`; names that look like keys or secrets never read; a symlink target may use `..` only as leading segments and the `realpath` of the joined target must be inside the root | `confine.py`, `fsops.py` | |
| Uploads go to `<file>.incomplete`, must match the declared size and, when an expected sha256 is known, the whole-file hash, then are renamed; a disk is never filled below 1 GiB free | `fsops.py` | |
| Deletes: policy `allow_delete`, a writable root, never a root itself or a folder that holds another root, and refused while a running container mounts the target or anything under it; the parent and the target are opened once and the tree is removed through those fds, so a path re-pointed during the in-use check answers `conflict` | `fsops.py` | See residual risk 6. |
| Engine HTTP: loopback ports of console gateways or inventory endpoints only, GET and POST, allowlisted path prefixes, no redirects, hop-by-hop and Host headers dropped; in monitor mode POST only to inference paths (chat, completions, embeddings, Ollama chat/generate/embed/show); a bare endpoint counts as an engine only with a real `/v1/models` listing (or 401/403) | `httpproxy.py`, `discovery.py` | |
| Scripts: argv only from the policy (absolute program path, no shell), at most one argument from the action's allowlist, minimal environment, one run per script, a timeout, the whole process group killed on timeout, cancel or link loss | `scripts.py` | Fakes only. |
| Protocol stream hygiene: the node keeps the protocol on private copies of stdin and stdout and points fds 0 and 1 elsewhere; it exits on stdin EOF, on 30 s without a hub message, and when a write is blocked for 10 s | `cli.py`, `server.py` | |
| Host header allowlist (421); same-origin Origin check on non-GET (403 `cross_origin`); `X-Engine-Console: 1` on non-GET without a bearer key (403) | `api/security.py` | Unverified against a real browser. |
| Bearer keys, roles `admin` and `viewer`; a viewer may `POST` only chat, fit, conversations and arena routes, matched exactly, and every router also enforces it (`enforce_roles`); the `GET /hf/*` routes are admin-only | `api/security.py`, `api/deps.py`, `api/routers/system.py` | |
| `ENGINE_CONSOLE_TRUST_LOOPBACK=false` by default | `config.py` | `/api/v1/health` warns when it is turned on. |
| The generic settings (`HOST`, `PORT`, `ALLOWED_HOSTS`, `TRUST_LOOPBACK`, `DATA_DIR`) are read only with the `ENGINE_CONSOLE_` prefix; the bare names are ignored with a start-up warning, and an empty host is refused | `config.py`, `main.py` | |
| Security headers on every response (CSP without `unsafe-inline`, `nosniff`, `no-referrer`, `X-Frame-Options: DENY`, `Permissions-Policy`), `Cache-Control: no-store` on `/api/` | `api/security.py` | |
| Static files: only the app shell and the `css`, `js` and `i18n` folders, containment checked | `main.py` | |
| Body caps: 1 MiB, 32 MiB for chat, 256 KiB for profile import (413), content-type check on import (415); 8 SSE streams (4 per key), 16 streamed chat replies (4 per key) and a 300 s SSE idle timeout | `api/security.py`, `api/deps.py` | |
| Audit of every mutating call: key id, role, method, path, status, redacted params (metadata only for chat, arena, conversations, prompts); the table refuses UPDATE and DELETE | `api/security.py`, migration 0001 | |
| Image pins: a pinned tag or digest, never `:latest`; a global pin must also be an adapter default or on `ENGINE_IMAGE_ALLOWLIST` (exact match) | `services/settings.py` | What may run is the node policy's `engine_images`. |
| Adoption of containers by label validates every label (ids, names, ports inside the node's range, model references) before it reaches SQL, paths or the UI | `services/lifecycle.py` | |
| Console egress: HTTPS only, per-redirect-hop allowlist, `trust_env` off, optional proxy, fail closed with 503 `egress_proxy_unavailable` | `services/hf_http.py` | One real download through a proxy verified live; the failure paths are fakes only. |
| Secrets: engine secret parameters are `[set]` in the database and API; a secret that an adapter would put on the command line is refused (`secret_in_argv`) | `services/secrets_store.py`, `lifecycle.py` | The value is plaintext in a 0600 file. |

## Threat model (STRIDE)

| Threat | Scenario | Mitigation present | Residual risk |
|---|---|---|---|
| Spoofing | A web page in the operator's browser calls the API (cross-site request, DNS rebinding) | Host allowlist (421), Origin check, CSRF header, and loopback is not trusted, so the page would also need a key | Unverified against a real browser. |
| Spoofing | Another local process calls the API | A key is required (`ENGINE_CONSOLE_TRUST_LOOPBACK=false`) | A process running as the hub's user can read the bootstrap key file, the database, the hub ssh key and the HF token. A compromised hub account is a compromise of everything the hub can reach. |
| Spoofing | A reverse proxy without `X-Forwarded-For` makes remote clients look like loopback | With loopback untrusted this no longer grants anything; a key is still needed | Only with `ENGINE_CONSOLE_TRUST_LOOPBACK=true` is a proxy mistake a full bypass. A proxy that rewrites Host needs `ENGINE_CONSOLE_ALLOWED_HOSTS`. |
| Spoofing | Something else answers at a node's address, or sits in the middle | `StrictHostKeyChecking=yes`, known_hosts read-only, no host key updates, no DNS host-key lookup | Trust on first use: the operator must check the fingerprint when connecting once by hand. A name in `hosts.json` that DNS can spoof is only as good as DNS: prefer addresses. |
| Spoofing | A stolen hub key is used from another machine | `from="<hub address>"` and `restrict` in each machine's key line; the key can only start a node | Useless protection if the line was installed with `--from '*'` (the CLI warns), or if the attacker is on the hub's address. |
| Spoofing | ssh quietly offers the operator's own, unrestricted key | The hub refuses to run ssh without its own key, uses `-o IdentityFile` plus `IdentitiesOnly`, `-F none`; an unrestricted login is reported as a note | A host with `ssh_config: true` lets the operator's ssh configuration add identity files; then only the note tells. |
| Spoofing / tampering | A hostile or compromised node reports false GPUs, containers, models, health or metrics, or crafted names | Whitelisted, capped, finite messages; unknown GPU UUIDs dropped; ids and labels validated; node text never used in SQL, paths or a shell; library entries are cleaned again when read from SQLite | A node can still lie about anything it reports. Wrong numbers are possible; code execution through the data is not expected. HTML escaping in the UI is unverified. |
| Tampering | A viewer key mutates state | Exact viewer list, `enforce_roles` on every router | None known. |
| Tampering | Crafted parameters or ids inject docker flags, mounts or paths | Hub: typed parameters, `RepoId`, mounts by root and relative path. Node: re-validation of every field, argv lists, mounts inside the realpath of a policy root, `:` and `,` refused, label keys, env names, ulimit allowlist | Not fuzzed. An admin can still set any flag an adapter's catalog offers, including `trust_remote_code`. |
| Tampering | A malicious or stolen hub asks a node for something harmful | Monitor mode changes nothing; manage mode allows only the policy's images, roots, scripts and switches; every argument is validated again on the node | Within the policy the hub can still start any allowed image with any catalog flag, write (and, with `allow_delete`, delete) in writable roots, run registered scripts with allowed arguments, and stop and start engines if `external_control` allows. Tag patterns such as `vllm/vllm-openai:*` accept any tag, and a registry tag can be repointed. |
| Tampering | A policy file is edited by the wrong person | Root-owned `/etc/ecnode/policy.json`; the node refuses files that are writable by others, symlinks, or in writable directories | A policy under the hub user's home is only a guard against hub bugs: the hub's user can edit it. |
| Tampering | A model root is changed under the node (links swapped), by another user or by the hub itself through node ops | `realpath` before use, directory fd re-checked, `O_NOFOLLOW`, secret-looking names never read; links can use `..` only as leading segments and must resolve inside the root, so node ops cannot create an outward link; deletes go through pinned fds; containers get read-only mounts only | Before the 2026-10-04 review fixes the hub itself could create the swap (H1-H3 below). A root on storage that other users can write (NFS, a shared folder) still leaves a short window between the mount check and docker's mount, because docker is given a path, not a pinned fd. Keep roots writable by nobody but the node's user. |
| Tampering | A model file is corrupted or replaced in transit | Hugging Face files with an LFS hash and every copy of such files are verified by sha256 on the receiving node, and a size mismatch deletes the partial | A loose GGUF file or plain directory copied between machines, and the small non-LFS files of a repository, are checked for size only: there is no expected hash. |
| Tampering | A pre-existing engine network without `--internal`, or a gateway or engine name collision | The node refuses a non-internal network (`conflict`) and a name that exists | None known. |
| Tampering | A malicious `hosts.json` planted by another user | Refused unless owned by the user, a regular file, not group- or world-writable, no symlink | A process running as the same user can edit it. |
| Repudiation | Someone denies a change | Every mutating API call is audited with the key id; script runs, library deletes, copies and reconnects are mutating calls | The audit table can be edited by whoever owns the database file. A node keeps no audit log of its own: an action on a machine is traceable only to the hub's audit row. |
| Information disclosure | A token or key in logs, API output or the audit | Redaction in logs and audit, `[set]` markers, JSON logs without tracebacks, node-side scrubbing, ssh key path masked in link errors | Secret parameter values sit in 0600 files; pattern-based redaction can miss free text; chat content is plaintext in SQLite. |
| Information disclosure | Engine secrets in transit and at rest on a node | Inside the ssh session; into docker only through the child environment, never argv | A user who can use docker on that machine can read an engine's environment with `docker inspect`; so can root. |
| Information disclosure | A hub or key holder reads files through the node | Only inside policy roots, secret-looking names refused, relative paths with realpath checks | Any file in a root that does not look like a secret can be read with `fs.read` (that is how copies work). Roots should hold only models; the starter policy names the `hub/` folder of a cache, not the whole HF home. |
| Information disclosure | A viewer or any key holder sees other containers' details | Argv redacted on the node, environment never read, argv and names length-capped | Container names, images, ports, redacted command lines and, for managed engines, logs are visible to every key holder, viewers included. |
| Information disclosure | XSS through model output, a model card or node text | CSP with `script-src 'self'` and no inline script, hand-written markdown renderer, contract rule that node text is never HTML | The renderer was not audited and the new frontend is unverified. |
| Information disclosure | The egress proxy sees credentials | The proxy is operator-run and on loopback | With a proxy configured, it carries every downloaded byte and sees the token in each request header; its flow log is sensitive. |
| Denial of service | A hostile node floods the hub | Flood limit, credit limit, silence timeout, hello timeout; the link is dropped and retried with backoff | A node can still make its own link flap. |
| Denial of service | A hostile or buggy hub floods a node | 64 requests in flight (then `busy`), upload credit enforcement, bulk JSON rate limit, idle and silence exits | The hub can occupy a node's request slots with long calls. |
| Denial of service | Disk exhaustion on a machine | Hub pre-check with a 2 % margin, node refuses to go below 1 GiB free, queue of 50 rows, two transfers at a time | `.incomplete` partials stay when the policy forbids deletes. The SQLite database can grow without a retention job. |
| Denial of service | A hung network mount, a huge model folder, or a stuck script | Filesystem calls are abandoned after a timeout, scans stop after 50,000 entries or 120 s per root, scripts have a timeout and are killed as a group | A dead NFS server leaves one blocked thread on the node. |
| Denial of service | An engine takes all host memory or CPU | pids limit 4096; the console watches host RAM and alerts | No `--memory` or `--cpus` limit is set. An engine can exhaust a machine. |
| Denial of service | Request or SSE flood at the hub | Body caps, 8 SSE connections (4 per key), 16 streamed chat replies (4 per key), idle timeout | No rate limiting; non-streaming chat and arena calls are not capped; the caps are per process; not load-tested. |
| Elevation of privilege | A compromised hub starts arbitrary containers everywhere | Policies, label ownership, image match, `--cap-drop ALL` | See "Tampering: a malicious or stolen hub" above. |
| Elevation of privilege | A stolen hub key | Starts a node, from the hub's address, which obeys its policy | A bug in the node's frame parser or an operation handler would run as the login user. |
| Elevation of privilege | The login user can use a rootful docker daemon | Rootless Docker is the tested setup | A user in the `docker` group is root-equivalent on that machine; so is the node, and any script the policy registers. |
| Elevation of privilege | An engine container escapes | Rootless Docker where used, `--cap-drop ALL`, `no-new-privileges`, pids limit, internal network, read-only model mount | Kernel or runtime escapes are not addressed; Docker's default seccomp profile only; GPU passthrough widens the surface; the engine's root file system is writable. |
| Elevation of privilege | An engine loads hostile model code or weights (`trust_remote_code`, pickle-based files) | `trust_remote_code` is off unless set and preflight warns; no route out of the container | The code runs inside the container with GPU access and can read every model in the mounted folder. |
| Elevation of privilege | `--ipc host` for multi-GPU engines | Only if the policy sets `allow_ipc_host` (false when absent, true in the starter template), and only when more than one GPU is used | A container then shares the host's IPC namespace. |
| Elevation of privilege | A registered script | argv fixed in the policy, one argument from an allowlist, minimal environment, no hub input | Scripts are the operator's code with the login user's privileges. |
| Elevation of privilege | The node installed from a modified checkout | After install the code is root-owned and run with `python -I` | Nothing verifies the source: whoever can modify the checkout you install from controls the node. |
| Elevation of privilege | An engine pivots through an opted-in router | Opt-in only | An opted-in engine can reach the router container and what it exposes on that network. |
| Exfiltration | An engine or a compromised model sends data out | No route out (internal network, no published ports) | Verified live for the launch flags of a GPU engine and, in the earlier self-test, for a test container's lack of egress; an explicit egress probe from a GPU engine was not part of the 0.4.0 run. |
| Availability | Egress proxy down | Fail closed (503), no direct fallback when a proxy is configured | `REQUIRE_EGRESS_PROXY` defaults to false: without `EGRESS_PROXY` the console runs direct and health warns. |

## Security findings fixed during the 0.4.0 build

1. **ssh could fall back to the operator's default identities.** With `-i <file>` and a missing file, ssh only warns and then
   offers the user's default identity files, even with `IdentitiesOnly=yes` (observed with `ssh -G`). The hub would have
   logged in with an unrestricted key, defeating the forced command. Fix: the hub refuses to build the command while its key
   file is missing (`NoHubKey`: the link says to run `hosts init-key`), passes the key as `-o IdentityFile=` (which fails
   closed even if the file vanishes after the check), keeps `IdentitiesOnly=yes` and `-F none`, rejects key paths with
   whitespace, quotes or `%`, and notes any login that did not come through the restricted line. Tests:
   `backend/tests/fleet/test_hosts.py`.
2. **The in-use rule for shared model caches.** Engines mount a whole cache folder, so the first rule (a mount that holds the
   model means in use) made every model in it look in use, and none could be deleted while any engine ran. This was found
   live: a freshly downloaded test model was refused because two unrelated engines mount the cache. Fix: a container that
   mounts a folder holding the model now counts only if its parsed model arguments (model, draft model, tokenizer, label)
   name that model's repository, cache folder or path; a container whose model argument cannot be parsed still counts as
   using everything it mounts, so a delete is refused rather than pulled from under a running engine. This rule lives in the
   hub (`services/lifecycle.py`); see residual risk 6.
3. **No `memlock` ulimit.** Unlimited locked memory lets one container pin host RAM, and it only matters for RDMA, which would
   need its own policy switch. The node accepts only `stack` and `nofile` as container ulimits (TensorRT-LLM asks for a 64 MiB
   stack); any other name, including `memlock`, is refused. Test: `backend/tests/core/test_runspec_ulimits.py`.
4. **llama.cpp's CORS default reflected any origin.** Measured on the pinned build: with its default (`*`) the server echoes any
   origin in `Access-Control-Allow-Origin`, so a web page open in the operator's browser could read answers from the loopback
   port. Fix: the adapter passes `--cors-origins localhost` unless the user sets `cors_origins`. The console never needs CORS.
5. **The node never reads container environment values.** Environment values are the likeliest place for secrets, so the node's
   inspect view takes only names, image, command, labels, ports, mounts, state and a few timestamps and ids; the command line
   is redacted on the node before it leaves the machine. The raw `docker inspect` document does contain the environment in the node process's memory, but no
   code copies a value out of it. Test: `node/tests/test_discovery.py` (a fake container whose environment holds a marker value
   that must never appear in anything reported).

## Findings from the independent review of 2026-10-04

An audit of the 0.4.0 tree with zero prior context (code reading, throwaway probes against the real node modules,
450 iterations of hostile node messages through the hub) returned BLOCK on three findings. All fixed items below have a
regression test that reproduces the audit's sequence and fails on the code before the fix; they are verified offline only
(no real sshd, no remote node, no Docker daemon race was run).

| # | Finding | Fix | Test |
|---|---|---|---|
| H1 | `fs.symlink` checked its target lexically, so `y -> sub/..` then `z -> y/..` pointed `z` at the root's parent | `..` only as leading segments; `realpath` of the joined target must be inside the root | `node/tests/test_fsops.py` |
| H2 | `fs.delete` re-resolved the path after the in-use check (which runs docker), so a parent swapped for an outward link redirected the delete outside the root | parent and target pinned by fd, removal through the fds, `conflict` when the target moved | `node/tests/test_fsops.py` |
| H3 | `ro: false` gave containers read-write mounts, from which they could plant links and swap directories | every mount read-only; `ro: false` refused; the hub's `HF_CACHE_READONLY` switch removed | `node/tests/test_docker.py`, `backend/tests/core/test_model_mounts.py` |
| M1 | `ext-<host>-<slug>` was ambiguous for hyphenated host ids, so a stop could reach another host | each `-` of the host id is doubled in derived ids | `backend/tests/core/test_externals.py` |
| M2 | the hub's engine stream reader was quadratic and unbounded on a line with no newline | linear scan; a line over 1 MiB or a stream over 256 MiB is cut off | `backend/tests/core/test_nodehttp_chat_bench.py` |
| M3 | a hostile safetensors header cost minutes of node CPU per file | dims, tensor count and the running product bounded | `node/tests/test_models.py` |
| M4 | monitor mode forwarded any POST (Ollama pull, create, delete); an HTML catch-all page could qualify as an engine | POST only to inference paths in monitor mode; `/v1/models` must be a listing, and a 401/403 there counts only when `/metrics` names a known engine | `node/tests/test_httpproxy.py`, `node/tests/test_discovery.py` |
| M5 | mounting a whole HF_HOME showed its `token` file to the container | a mount is refused if a key- or token-looking name sits up to four levels below its source (HF model trees skipped; names like `hf_token`, `my-secret.txt`, `apikey` included), or if the source is too big to check; the hub mounts only `hub/` | `node/tests/test_docker.py`, `backend/tests/core/test_model_mounts.py` |
| L2 | library and lifecycle called each other without end while a managed instance was active | the lifecycle uses `library.locate()`, which does not ask who uses the copy | `backend/tests/core/test_library_lifecycle_wiring.py` |
| L3 | NaN/+Inf metric values reached SSE as non-JSON (NaN also broke the rollup insert) | dropped at parse time | `backend/tests/core/test_metrics_fleet.py` |
| L6 | the secret store refused ids with `-`, so deleting such an instance answered 500 | one id rule for every instance id shape; delete never raises | `backend/tests/core/test_audit_fixes.py` |
| L8 | the audit log masked fewer parameter names than the secret store treats as secret | one `SENSITIVE_KEY` rule for both | `backend/tests/core/test_audit_fixes.py` |
| - | the contract said an ssh user never starts with `-`, but the check allowed it (`--` kept it harmless) | the user part must start with a letter, digit or `_` | `backend/tests/fleet/test_hosts.py` |
| M7 | hub state keyed by node-chosen names (metric rings, counters, rollups) was never pruned; a node reporting new names forever grew memory and the database | at most 256 series per host, the least recently updated evicted first (with its rollup rows), and one host cannot evict another's; a series not updated for 1 h leaves memory; rollup rows are deleted after 14 days (hourly) or 400 days (daily); a host that left hosts.json follows the same expiry; the prune runs in the existing metrics loop at most every 5 min | `backend/tests/core/test_metrics_prune.py` |
| L1 | the docker child inherited the node's whole environment, and the deny-list for `secret_env` names missed loader and tracing names (`OTEL_*`, `LOCPATH`, ...) | every program the node starts (docker, nvidia-smi, ip) gets an environment built from scratch: `HOME`, `XDG_RUNTIME_DIR`, `DOCKER_HOST`, `DOCKER_CONTEXT` from the node's own when set, `PATH` with absolute entries only, `LANG=C.UTF-8`; `docker run` adds the `secret_env` values, whose names must look secret (the hub's own split rule) and pass a wider deny-list (`OTEL_*`, `GRPC_*`, `LOCPATH`, `SSLKEYLOGFILE`, `ZONEINFO` added); scripts keep their documented list with the same `PATH` rule. Also run against a real rootless daemon (`docker info`, `docker ps`) and `nvidia-smi` | `node/tests/test_util.py`, `node/tests/test_docker.py`, `node/tests/test_scripts.py` |
| L7 | the installer inherited `PATH` and umask, re-set the owner and mode of `/usr/local/lib`, did not say what it installed, and the policy template invited mutable tags | fixed `PATH`, `umask 022`, `CDPATH` unset; `/usr/local/lib` is created only if missing, never chown/chmod-ed, and refused if anyone but root can write to it; the installer prints the git commit (read as the checkout's owner) or `unknown (...)` and records it in the launcher; the template allows llama.cpp by digest only and carries an `_engine_images` note asking for digest pins; the manage example pins llama.cpp and the gateway by real digests | `node/tests/test_cli.py`; the installer was run in a user and mount namespace (mapped root, scratch directories), not as real root |
| L4 | the SSE cap of 8 was global, streamed chat replies were uncapped per key, and viewers could spend the HF token through `/hf/*` | 4 SSE streams per key under the global 8; streamed chat replies capped at 16, 4 per key (`429 too_many_chat_streams`); a slot is released however a stream ends, including a client gone before the first byte (which leaked a slot under ASGI 2.4); `/hf/*` admin-only, and a viewer's `POST /fit` judges only copies a machine already holds; Discover shows a viewer a plain note | `backend/tests/core/test_stream_caps.py` |
| L5 | settings used unprefixed environment names, so a stray exported `HOST=0.0.0.0` widened the bind | `HOST`, `PORT`, `ALLOWED_HOSTS`, `TRUST_LOOPBACK` and `DATA_DIR` are read only as `ENGINE_CONSOLE_<NAME>`, from the environment and the env file; the bare names are ignored with one start-up warning (names only); an empty host is refused | `backend/tests/core/test_env_prefix.py` |
| L9 (CI) | CI actions were pinned by tag | each `uses:` is pinned to the commit of its release, with the release in a comment (checkout v4.4.0, setup-uv v5.4.2, setup-node v4.4.0), checked against the GitHub API | `.github/workflows/ci.yml` |

A second zero-context review of the fixes (same day) found H1, H2, M1-M4, L2, L3, L6 and L8 closed and returned
ALLOW-WITH-CHANGES. It asked for a deeper secret scan before mounting and a stricter rule for locked `/v1/models`
endpoints; both are in the table above. The mount check-then-use window it also named is the residual in the
"model root is changed under the node" row: it needs someone who can already write inside a root.

On 2026-10-14, M7, L1, L4, L5, L7 and the CI half of L9 were closed (the rows above). Those fixes have not had an independent
review yet. Within its cap of 256 series a node can still churn names and evict its own older series and their history, and
rollup rows written before migration 0005 carry no host, so only their age removes them.

Still open from that review:

- **M6** copy provenance: loose GGUF and plain copies are size-checked only (residual risk 7), and shared-storage ownership
  rests on addresses and names the nodes report about themselves.
- **L9** (the systemd half): the unit template was updated for 0.4 but has never been started (residual risk 15).

## Residual risks

1. **A docker-group user is root-equivalent on that machine.** If the node's login user can use a rootful Docker daemon, the
   node and every script registered in its policy are root-equivalent. The policy limits what the hub asks, not what a bug in
   the node, or a script, could do. Rootless Docker is the setup this project is developed and tested on.
2. **Whoever controls the hub controls what each node policy allows.** The policy bounds the damage; it does not remove it.
   On the hub's own machine a policy in the user's home directory is only a guard against hub bugs, because the hub runs as
   that user. Put the policy in root-owned `/etc/ecnode/policy.json` where it matters.
3. **Engines run model code if `trust_remote_code` is set**, and pickle-based weights can run code on load. The code runs in
   the container, with GPU access and read-only access to every model in the mounted folder, and without a route out.
4. **The hub's ssh key has no passphrase** and lives in the hub user's files. Anyone who can read it can start nodes on every
   machine that authorized it, from the hub's address. Anything running as the hub's user can read it.
5. **A node is a program running as a login user and parses frames from the hub.** A stolen hub key plus a bug in the node
   would give code execution as that user. The node is standard-library Python with a small, whitelisted protocol surface;
   it has had no external review.
6. **A delete from a shared model cache relies on the hub's in-use rule.** The node refuses `fs.delete` while a running
   container mounts the target or anything under it; a container that mounts a parent folder (a whole cache) is only caught by
   the hub's rule (finding 2 above). A compromised hub can bypass it; `allow_delete: false` is the node-side answer.
7. **Copies of loose GGUF files and plain directories are not hash-checked**, and neither are the small non-LFS files of a
   repository: there is no expected hash to compare. A hostile source node could change them.
8. **No memory or CPU limits** on engine containers, and Docker's default seccomp profile only.
9. **Trust on first use for host keys.** The operator must compare the fingerprint when connecting once by hand.
10. **`ssh_config: true`** hands part of the ssh command line back to the operator's configuration (identities, ProxyCommand).
11. **Egress enforcement is opt-in:** the default is direct (`REQUIRE_EGRESS_PROXY=false`); with a proxy, the proxy's log can
    hold credentials and every downloaded byte passes through it.
12. **Secret parameter files are plaintext on disk (0600)** and chat content is plaintext in SQLite.
13. **The gateway has no authentication of its own:** loopback reachability on each machine is the only control. Any local user
    of a machine can reach its engines' gateway ports.
14. **The audit table is not tamper-evident** (the database file is editable), and nodes keep no audit log.
15. **The systemd unit template has never been started.** It was updated for 0.4 (the local node's writable roots and state dir
    in `ReadWritePaths`, `PrivateUsers=yes` so it needs rootless Docker, options unsupported in user units removed) and only
    parsed with `systemd-analyze --user verify`.
16. **One independent review, no penetration test.** The 2026-10-04 review (below) was an AI auditor reading the code, with
    probes and protocol fuzzing; no human review or penetration test has been done.

## Secrets handling

- `HF_TOKEN` or `~/.config/engine-console/hf_token` (mode 0600; a group- or world-readable file is refused). The token is read
  by the hub only and sent only in the `Authorization` header of allowlisted requests. It is never given to a node or an
  engine, and engines run with `HF_HUB_OFFLINE=1` by default.
- API keys: 256-bit random, shown once, stored as SHA-256. The first-start admin key is in `<data_dir>/bootstrap-admin.key`
  (0600); the console logs the path and reminds you on every start to copy the key and delete the file.
- The hub ssh key: created with umask 077 inside a 0700 directory, never printed, never returned by an API, not logged
  (its path is replaced by `<hub key>` in link errors).
- Engine secret parameters: `[set]` everywhere except the 0600 file on the hub; sent to the node only inside `docker.run`.
- Backups of `<data_dir>` hold key hashes, secret files, conversations, the audit and library snapshots, and `hosts.json` and
  the hub key sit next to it: encrypt them (OPERATIONS.md).

## Security principles this project follows

| Principle | How the console relates |
|---|---|
| Deny egress by default | Engines have no route out; console egress goes through an allowlisting proxy and fails closed when configured; still opt-in by default. Nodes need no internet. |
| Isolate what you run | Engines on an internal network, no ports, `cap-drop ALL`; the node only touches containers it labelled; model files only inside policy roots. |
| Least authority | A node-local policy bounds the hub; monitor mode changes nothing. |
| Never read or leak credentials | The node never reads container environment values and refuses secret-looking file names; the hub's environment never reaches a node; no path touches cloud or browser credential stores. |
| No host-level changes without the operator | Installing a node, authorizing the key, writing a policy, the systemd unit, the proxy and joining a router to an engine network are operator actions; nothing self-applies. |
| Observable from the first request | OTLP, `/metrics`, JSON logs, audit (ADR-0010). |
| Secrets never on disk unprotected | 0600 files, `[set]` markers; backups must be encrypted (OPERATIONS.md). |
| Do not claim what was not verified | The live checks are listed at the top; everything else is marked fakes only or unverified. |

## Recommended posture

- Keep the hub bound to loopback and `ENGINE_CONSOLE_TRUST_LOOPBACK=false`; never expose port 8791 without a TLS proxy that sets
  `X-Forwarded-For` and whose hostname is in `ENGINE_CONSOLE_ALLOWED_HOSTS`.
- Install node policies under `/etc/ecnode/` (root-owned), start every machine in monitor mode, and switch to manage only
  after reading the file. Keep `allow_delete`, `allow_pull` and `external_control` off until you need them. Give each writable
  root its own folder that holds only models, and prefer digest-pinned images in `engine_images`.
- Install the key line with `--from <the hub's address>`, never `*`, and check the host key fingerprint before the first
  connection. Leave `ssh_config` off.
- Prefer rootless Docker for the node's login user. Do not add that user to a rootful `docker` group.
- Set `EGRESS_PROXY` and `REQUIRE_EGRESS_PROXY=true`; give Prometheus a `viewer` key; encrypt backups.
