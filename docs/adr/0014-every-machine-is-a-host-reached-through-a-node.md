# ADR-0014: Every machine is a host reached through a node

Status: accepted (contract v2, `docs/ARCHITECTURE.md` section 10.1, decision 14); implemented in `node/` and
`backend/src/engine_console/services/fleet.py`.
**Verified live:** the local node as a subprocess of the hub, on one real workstation (host facts, GPUs, per-process
VRAM, cgroup memory, drives, discovery, a library scan, a managed launch, chat, stop and remove, a library delete).
**Fakes only:** remote machines through ssh (no remote node is installed yet) and every protocol path the live run did
not touch.

**Status note (0.5.0).** Since 0.5.0 the node is the Rust binary `ecnode` (`backend/crates/ecnode`), not a standard-library Python package: it needs no system Python, and for `local` it is the `ecnode` next to the hub binary (`NODE_COMMAND` overrides it), started with `--stdio`. The protocol is defined once in the `ec-proto` crate that both binaries link, in place of `proto.py` (the frames and JSON values are unchanged, protocol version 1). `hello.host.python` is now null (DV5). "Every machine needs the node package" now reads "every machine needs the `ecnode` binary", installed by `node/install.sh --binary`. The decision stands. Every "verified live" statement in this ADR was made with the Python 0.4.x implementation; the Rust code has not been run live. See ADR-0022.

## Context

Version 0.3 ran docker, NVML, `/proc` reads and model-cache access inside the FastAPI process and assumed one machine.
The new requirement is one page that manages engines and models on several GPU machines. Those machines have their own
docker (rootless or not), their own GPUs, disks and sometimes NFS mounts, and their own system Python (3.12 on current
Ubuntu LTS and DGX OS). The hub is an asyncio process, and docker, `nvidia-smi` and `/proc` reads block.

## Decision

- Every machine, including the one the hub runs on (`local`), is a **host** with exactly one **node**: `ecnode --stdio`,
  a standard-library-only Python 3.10+ package (`node/`). On the hub's machine it is a subprocess of the hub
  (`python -I -m ecnode --stdio`, environment reduced to a whitelist). On other machines it is the same program started
  over ssh (ADR-0015).
- The node does all docker, `/proc`, `/sys`, NVML or `nvidia-smi`, and model-root file work. The hub does none of it.
- The wire is a framed binary protocol on the node's stdin and stdout: length-prefixed JSON frames, DATA frames with
  per-stream credit windows, ping and pong. It is defined once in `node/src/ecnode/proto.py`; both ends import that
  module (the hub depends on the `ecnode` package), so they agree byte for byte. The node pushes `hello`, `sample`
  (every 2 s), `disks` (30 s), `inventory`, `metrics` and `models_changed`; the hub sends requests and receives exactly
  one result for each (`docs/FLEET-PROTOCOL.md`).
- The hub treats every node as untrusted data: each pushed message is rebuilt from a whitelist, numbers must be finite,
  strings and lists are capped, a node that floods, goes silent, breaks framing or exceeds its credit is dropped.
  `PROTO` is checked at `hello`; a mismatch puts the link in `incompatible` and names the side to update.

## Consequences

- One code path for the local and every remote machine. A hub bug cannot reach past what a node allows (ADR-0016). A
  slow docker call blocks a node thread, not the hub's event loop. Engines keep running when the hub restarts, and the
  hub adopts them again from their labels when each link comes up. Offline tests use an in-process fake node that
  speaks the real frames in both directions.
- The protocol is a contract surface. Changes are additive; anything else bumps `PROTO`.
- Every machine needs the node package. For `local` it comes with the hub's virtual environment; elsewhere
  `node/install.sh` copies it (needs `sudo` once) and relies on the machine's `/usr/bin/python3`.
- The hub's picture of a machine is what the node last pushed. A node that sends no sample for 6 s shows `stale`; one
  that sends nothing at all for 30 s is dropped (`down`). The last-known library of a machine that is down stays
  visible, marked stale.
- The node's policy summary reaches the hub in `hello`. No hub code sends `node.info` today, so an edit of a node's
  policy file is enforced by the node within a couple of seconds but only shows up in the hub after the link
  reconnects (the Reconnect action).
- Machines without NVML and without `nvidia-smi` report no GPUs.
- Linux only: the node reads `/proc` and `/sys` (container memory from cgroup v2, with a process-memory fallback).

## Alternatives considered

- **The hub drives `docker -H ssh://host` (docker contexts over ssh).** Needs a login that can use docker, which is
  root-equivalent with no policy layer in between; gives no view of GPUs, `/proc`, disks or model roots; leaves local and
  remote on different code paths.
- **A node daemon that listens on a TCP port (HTTP or gRPC with TLS).** Adds a listening service per machine plus
  certificates and an authentication scheme to operate; ssh is already how the operator reaches these machines.
- **A cluster manager (Kubernetes, Nomad, Slurm).** Far larger to install and run than a few workstations warrant, and
  it would own container lifecycle, which this project keeps to labelled sibling containers (ADR-0003).
- **One ssh command per action.** No streaming for logs, chat or copies, no pushed telemetry, a new login for each
  action.
- **Keep the hub's local code path and add a remote agent only.** Two implementations of every operation to keep in
  step, and local would not exercise the remote path.
