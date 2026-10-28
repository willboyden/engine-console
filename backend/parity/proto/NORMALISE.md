# ecnode protocol cross-matrix: scenarios, transcripts, normalisation

Files in this directory:

| file | what |
|---|---|
| `drive.py` | hub-side driver, Python, using only `ecnode.proto` (the reference) |
| `../../crates/ec-proto/src/bin/ecproto-drive.rs` | the same driver in Rust (`target/debug/ecproto-drive`) |
| `scenarios/*.json` | the steps each run performs |
| `golden/*.json` | normalised transcripts of `drive.py` against the **Python** node (`--record`) |
| `env.mjs` | the hermetic environment of one run (temp HOME, fake bin dir, seeded model roots, policy) |
| `fakes/*.cjs` | fake `docker`, `nvidia-smi`, `ip` (Node, so the matrix outlives the Python reference) |
| `matrix.mjs` | runs {py, rust} driver x {py, rust} node over every scenario and diffs against golden |
| `hub-node.mjs` | the Python hub's real NodeLink driving the Python node, then the Rust node; diffs the API |
| `expected-diffs.json` | every diff the Rust node (or the Rust hub, `rust_hub`) may show, with the reason |
| `gen_vectors.py` / `vectors.json` | Python `ecnode.proto` outputs for 875 hostile inputs; `crates/ec-proto/tests/test_vectors.rs` |

Run: `node backend/parity/proto/matrix.mjs` (builds the two Rust binaries first; `--no-build`, `--only <name>`,
`--all`, `--show-expected`), `node backend/parity/proto/hub-node.mjs`, `node backend/parity/proto/rust-hub.mjs`, and
re-record golden from the Python reference with `node backend/parity/proto/matrix.mjs --record` (refuses to write a
transcript that holds this machine's identity). The Python side is the last Python release, prepared by
`lib/python-release.mjs` (the tag `python-final`; `EC_PYTHON_REF` picks another ref).

Rules the matrix enforces:

* `py` driver x `py` node and `rust` driver x `py` node must equal golden exactly (the second one proves the Rust
  framing, JSON encoding, credit accounting and upload/download handling against the real Python node);
* `*` x `rust` node may differ only where `expected-diffs.json` lists the path (glob: `*` within a segment, `**` the
  rest; a pattern also covers everything below it). Entries that no longer match anything are printed as notes.

## The hermetic environment (env.mjs)

* HOME, the fake bin dir and the policy live in a fresh temp dir; `PATH=<tmp>/bin:/usr/bin:/bin`, `LANG=C.UTF-8`;
  nothing else of the caller's environment reaches the driver or the node (no SSH_*, DOCKER_*, XDG_*).
* `docker` is a fake with two containers (`ec-owned-1`, console-owned; `other-engine`, not), both **stopped** so the
  node reads no `/proc/<pid>` of this machine; two images; one log; context `testctx`. It never reaches a daemon.
* `nvidia-smi` is a fake with two GPUs (`GPU-aaaaaaaa-1111-...`, `GPU-bbbbbbbb-1111-...`) and no processes; `ip`
  reports documentation addresses (192.0.2.5, 2001:db8::5). The node venv has no `nvidia-ml-py`, so the Python node
  uses the fake (the hub venv does have it, which is why hub-node.mjs starts the Python node from the node venv of the
  Python reference, `lib/python-release.mjs`).
* Model roots: an HF cache with `e2e/tiny-llama`, a plain checkpoint `tiny-plain`, `big/blob.bin` (9 MiB + 4321 bytes,
  more than WINDOW so a download needs a credit grant), and in manage scenarios a writable root `w`.
* Policy: monitor or manage, docker context `testctx`, `discovery_ports: []` (no probe of this machine's real ports),
  `engine_images: ["vllm/vllm-openai:*"]`, a fake gateway digest, `allow_delete` in manage scenarios.
* The node still reads this machine's `/proc`, `/etc/machine-id`, hostname and mounts: the normaliser below removes
  them, and every transcript is scanned for this machine's hostname, user name, home path, machine id (and its
  hash), interface addresses and real GPU UUIDs. A hit fails the run; `--record` refuses to write the file.

## Scenario steps and transcript records

The transcript is `{scenario, skipped, hello, initial: {sample, disks, inventory}, steps: [...]}`: `skipped` is the
preamble byte count, `hello` the node's first frame, `initial` the first push of each of those types (the driver waits
up to 15 s for all three before the first step).

| step | sends | records |
|---|---|---|
| `ping {id}` | ping | `pong` (or null after 10 s) |
| `req {op|op_missing, args|args_missing, id?, upload?, grant?, cancel_after_bytes?, cancel_after_events?, timeout?}` | req; upload DATA within the node's credit then `end`; download credit in WINDOW/4 grants; `cancel` after N bytes or N events | `id`, `res` (raw), `events`, `data` ({bytes, sha256}, or {bytes_ge_cancel} for a cancelled download, null when none), `credit_from_node`, `credit_violation` (the node sent DATA beyond its credit), `timeout` when no res came |
| `send_reqs {op, args (with {i}), from, count}` | count requests, no wait | `sent` |
| `send_req {id, op, args}` | one request, no wait (duplicate ids) | - |
| `cancel_all {from, count}` | cancel each | every `res`, in id order |
| `send {msg}` / `raw {hex}` | any JSON frame / raw bytes | - |
| `sleep {s}` | - (frames are still read) | - |
| `await_push {type, count, timeout}` | - | the latest push of that type once `count` were seen, `timeout` |
| `close {stderr_contains}` / `wait_exit` | close stdin / nothing | `exit` code (or "timeout"), `stderr_contains: {substring: bool}` |

Upload payloads are `byte i = (i * 31 + 7) % 251`; `"sha256": "$upload_sha256"` in args is replaced by the payload's
digest. Request ids count from 1 unless a step gives one.

## Normalisation (matrix.mjs `normalise`, applied to golden and to every run)

1. The run's temp dir is replaced by `<tmp>` in the raw text (paths in policy, error messages, rendered argv).
2. Numbers keep their kind: a float lexeme is stored as `{"$f": value}` (JSON.parse source access), so `5` vs `5.0`
   is a diff.
3. Any numeric `t`, `mtime`, `atime`, `scanned_at`, `duration_s`, `boot_time` becomes `"<num>"`; any numeric
   `generation`, `seq` becomes `"<int>"` (wall clock, scan timing, push counters).
4. `credit_from_node` > 0 becomes `"<granted>"` (how much upload credit a node grants depends on how fast its disk
   consumes; that the upload completed is checked by its res).
5. In a hello-shaped object (`proto` + `node_id` + `host`): `node_id` becomes `"<node_id>"`, and every non-null
   `host.*` value becomes `"<host.KEY>"` (hostname, OS, kernel, CPU, RAM, boot time, Python version).
6. In a sample-shaped object (`mem` + `cpu` + `gpus` + `containers`): numeric `mem.*` become `"<int>"`, numeric
   `cpu.*` become `"<num>"` (this machine's live memory and load). GPU numbers come from the fake and are kept.
7. A `mounts` list of disk mounts (objects with a `mount` key) becomes the list of its non-empty `roots` arrays
   (which seeded roots share a filesystem); the mounts themselves are this machine's.

hub-node.mjs additionally replaces the API's link timings (`last_seen`, `connected_since`, `rtt_ms`, `skew_s`,
`retry_in_s`, `updated_at`, `checked_at`), the host `label` (the hostname), host `memory` / `cpu` numbers, `alerts`
(they follow this machine's RAM pressure), node facts (`hostname`, `os`, `kernel`, `cpu_*`, `python`) and the detail's
`disks` (to their roots) with placeholders. It prints diff paths only, never values.
