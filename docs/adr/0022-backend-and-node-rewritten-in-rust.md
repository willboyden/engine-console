# ADR-0022: Hub and node rewritten in Rust

Status: accepted (release 0.5.0). Implemented in the Cargo workspace `backend/`: `crates/ec-hub` (the hub, binary
`engine-console`), `crates/ecnode` (the node, binary `ecnode`) and `crates/ec-proto` (the wire protocol both binaries
link). Replaces the implementation named in ADR-0002, ADR-0005 and ADR-0014 (FastAPI, a virtual environment, a
standard-library Python node); it does not change what those ADRs decided.
**Verified:** against the last Python release by differential tests (see "Consequences"). **Not verified with the Rust
code:** real GPUs, a real Docker launch, a real remote machine over ssh, an aarch64 machine, and the systemd unit.

## Context

Through 0.4.x the hub was a FastAPI application in a uv-managed virtual environment (its dependencies installed from PyPI),
and the node was a standard-library Python package that needed the machine's system `python3` 3.10 or newer. Every
machine in the fleet therefore needed a suitable system Python, and every upgrade of a node copied a Python package whose
behaviour depended on that interpreter (the node was run on 3.12 and 3.14 during 0.4). Installing, checking and rolling
back one file per architecture is simpler than that: the installer can verify a single binary, and "what is installed"
is one path.

The change had to be invisible to everything else. The HTTP API (including the OpenAPI document, the status codes, the
error bodies and the 422 validation messages the frontend reads), the node wire protocol (version 1, frame for frame), the
persisted state (`console.db`, `hosts.json`, the hub ssh key, the bootstrap key, node policies) and the command line had to
stay as they were, so that the frontend, existing `authorized_keys` lines and existing data directories keep working and
the hub and each node can be switched one machine at a time.

## Decision

Rewrite the hub, the node and the protocol library in Rust, in one Cargo workspace.

- The hub (`ec-hub`) uses axum 0.8 on its own hyper listener, rusqlite with bundled SQLite in WAL mode, reqwest with
  rustls, and OpenTelemetry over gRPC (tonic). Routing, request validation and the security middleware are written to
  reproduce the Python behaviour exactly (`backend/PORTING.md` sections 1 and 5 to 13).
- The node (`ecnode`) is one binary per architecture (x86_64 and aarch64, glibc, dynamically linked). It reads NVML by
  loading `libnvidia-ml.so.1` at run time when the driver library is present, and falls back to `nvidia-smi`.
- The wire protocol is byte-compatible: same frames, limits and flow control, same JSON values. Both binaries link
  `ec-proto`, so they cannot disagree.
- The HTTP API and the persisted state are unchanged. SQL migrations and the learn content stay the same files and are
  embedded at build time.
- `node/install.sh` installs a prebuilt `ecnode` binary only (`--binary`), checking its sha256 and its ELF header.
- Correctness is shown by differential parity, not by promise: the last Python release (tag `python-final`) stays
  runnable as the reference, and `make parity` and `make proto` compare the Rust code against it (`backend/parity/`,
  `python@<ref>` or `EC_PYTHON_REF` picks another reference). Every test of that release is mapped to a Rust test
  (`backend/TEST-MAP.md`, checked by `make test-map`).
- Deliberate differences are listed with a reason in `backend/PORTING.md` section 14 (DV1 to DV50), not left to chance.

## Consequences

Positive:

- A node needs no Python, compiler or network on its machine: copy one binary and its `.sha256`, run `install.sh`.
  Rollback of a machine is the Python installer of the `python-final` tag, which puts the Python node back at the same
  path.
- The hub no longer needs a virtual environment, uv or PyPI at run time. The dependency set is `backend/Cargo.lock`,
  built with `--locked`, checked by cargo-deny and cargo-audit, and subject to a 14-day crate cooldown (`make crate-age`).
- Hardening that was cheap to add while porting: bounded child output and file reads, special files refused instead of
  hanging, tool programs run only from a fixed `PATH`, route roles enforced in the request pipeline, the hub's own listener
  limits, a panic in a background loop logged and survived (DV31 to DV41).
- The hub and any node can be switched independently, in any order, because protocol 1 is unchanged and was exercised
  across all four hub/node combinations.

Negative:

- The Rust code is new. It has been compared with the Python release, but it has not run on real GPUs, against a real
  Docker daemon, over ssh to a real machine, on an aarch64 machine, or under systemd. Everything in the docs that is
  marked "verified live" was seen with the Python 0.4.x implementation.
- Behaviour was reproduced, including quirks (backend/PORTING.md section 15). Some Python mechanisms have no Rust twin and
  are documented deviations, for example: a bad setting prints `engine-console: invalid setting <field>: <reason>` and
  exits 1 (DV1); log timestamps are UTC and there is no uvicorn access log (DV2); the node reports no Python version (DV5);
  `gpu.source` is `nvml` when the driver library is present (DV6); `engine-console stop` only stops a console that the same
  implementation's `open` started (DV7); the catch-all 500 now carries the security headers (DV30).
- A node no longer searches the caller's `PATH`. If `docker` is not in `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`,
  the policy must name its absolute path (DV31); a machine upgraded without doing that has a node that cannot find docker.
- Building needs a Rust toolchain (pinned by `backend/rust-toolchain.toml`). The node is dynamically linked against glibc,
  and CI builds each architecture natively (an x86_64 and an arm64 runner) instead of cross-linking.
- The parity harness needs uv, Python 3.13 (hub) and Python 3.12 (node) to run the reference, and keeps a copy of the old code reachable.
- The `sha256` check in `install.sh` catches a corrupted or wrong-architecture file, not a malicious release: the binary and
  its checksum come from the same place. Nothing in this repository signs releases.
- The glibc floor is 2.35 (Ubuntu 22.04 and 24.04, DGX OS 7). Older distributions are not supported by the prebuilt binary.

## Alternatives considered

- **Keep Python.** Zero risk of new bugs, and the live verification of 0.4.x would still apply. Rejected because every
  machine would still need a suitable system Python and the hub a PyPI-fed environment, which is the cost this change
  removes.
- **Package the Python node with PyInstaller or a zipapp.** Keeps one code base. A zipapp still needs the system
  interpreter; PyInstaller ships an interpreter and every bundled library as one large opaque file whose contents are harder
  to audit than a small binary, and it does not remove the Python hub. Rejected.
- **Go.** A single binary per architecture and a good standard library. Not chosen: a fully static Go binary would hit the
  same limit as musl (it cannot load the NVIDIA driver's NVML library at run time), and a cgo build gives up the static
  advantage. No Go prototype was built or measured; this is a judgement from the constraints, not a comparison.
- **Static musl build of the Rust node.** Smallest dependency on the target. Rejected for the same NVML reason: a static
  musl binary cannot `dlopen` the NVIDIA driver's library.
- **Rewrite only the node.** Would have removed the per-machine Python requirement and left the hub on PyPI. Rejected because
  the protocol crate is shared and the two halves were easier to prove together.
