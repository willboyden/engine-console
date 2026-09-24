# ADR-0003: Engines as labelled sibling containers via the docker CLI

Status: accepted (contract v1, see `docs/ARCHITECTURE.md` section 1); amended 2026-09-24 after the hardening pass

## Context

Engines are commonly run as containers pinned to a GPU by UUID. The console needs to create and destroy them without touching containers it did not create.

## Decision

`services/docker.py` builds `docker --context rootless run ...` as an argv list executed without a shell, with a timeout per call. Every container gets an ownership label, `--security-opt no-new-privileges:true`, `--restart no`, GPUs by UUID, and a published port bound to `127.0.0.1` from range 18000-18099. `ps` and adoption filter on the label. Engines run with `HF_HUB_OFFLINE=1` by default.

## Consequences

- The console cannot remove containers it did not label.
- CLI over the Docker SDK avoids a dependency and mirrors what the operator would type, so 'copy as command' output is the same thing the console executes.
- Cost: parsing CLI output (`inspect` JSON is used where possible); each call forks a process.
- `--restart no` means engines do not come back after a reboot; adoption at console start only re-detects running containers.

## Alternatives considered

- Docker SDK for Python / raw socket: rejected, extra dependency, less transparent.
- Compose files generated per instance: rejected for lifecycle (state lives in compose projects); compose YAML is offered only as a copyable snippet.

## Amendment (2026-09-24)

Superseded in part by ADR-0011: engines no longer publish ports; they sit on the internal network
the internal engine network behind a per-instance gateway. Additional hardening on engine containers: `--cap-drop ALL`,
`--pids-limit 4096`, the HF cache mounted read-only (`HF_CACHE_READONLY` now defaults to true), image names
checked against `ENGINE_IMAGE_ALLOWLIST` (prefix match, so a pinned tag is not enforced, only the image repository), and
`stop`/`rm`/`logs` verify the console ownership label (plus role and owning-instance labels for gateways) before
touching a container. Unverified on real hardware: GPU engines under `--cap-drop ALL`, tensor-parallel/NCCL, and
RDMA transports (which would need `IPC_LOCK` back).
