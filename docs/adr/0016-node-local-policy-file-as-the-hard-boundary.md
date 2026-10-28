# ADR-0016: A node-local policy file is the hard boundary

Status: accepted (contract v2, `docs/ARCHITECTURE.md` section 10.1, decision 16); implemented in
`node/src/ecnode/policy.py` and enforced in `server.py`, `ops.py`, `docker.py`, `confine.py`, `fsops.py`.
**Offline tests:** loading, validation, every refusal rule and the monitor-mode refusals (node test suite).
**Verified live:** the local node ran in monitor mode with a starter policy, and one approved managed test ran with a
manage-mode policy. No policy has been installed on a remote machine yet.

**Status note (0.5.0).** Since 0.5.0 the policy is enforced by `backend/crates/ecnode/src/policy.rs`, and the other files named above exist in that crate under the same names (`server.rs`, `ops.rs`, `docker.rs`, `confine.rs`, `fsops.rs`). The decision and the policy format are unchanged. Every "verified live" statement in this ADR was made with the Python 0.4.x implementation; the Rust code has not been run live. See ADR-0022.

## Context

A compromised hub, a bug in the hub, or a stolen hub key must not be able to widen what the console can do on a machine.
Operators also want to start by only watching a machine and switch on control deliberately. The person who administers a
machine is not necessarily the person who runs the hub.

## Decision

- Each node reads a JSON policy: `--policy PATH`, else `/etc/ecnode/policy.json` if it exists, else
  `~/.config/ecnode/policy.json`. The installed copy is root-owned, mode 0644, so the hub's ssh key (which runs as the login
  user) cannot edit it. The hub can read a summary of the policy (it is in `hello`) and never change it.
- `mode` is `monitor` or `manage`. In monitor mode the node answers only the operations that change nothing
  (`proto.MONITOR_OPS`: info, metrics configuration, rendering a `docker run` without running it, a sanitized container
  inspect, logs of console-owned containers, image facts, engine HTTP calls, `fs.stat`, `fs.read`, and the model scan and
  list). Everything
  else answers `monitor_only`.
- In manage mode the policy still decides what is allowed: `engine_images` (an exact reference, `repository:*` or
  `repository@sha256:*`, with the repository compared literally), `gateway_image`, `engine_network`, `port_range`,
  `discovery_ports`, `model_roots` (id, path, kind, `writable`, label), `scripts`, `allow_pull`, `allow_delete`,
  `allow_ipc_host`, `external_control` and `max_shm_gib`.
- The node refuses a policy file (and runs in monitor mode, saying why in `policy.error`) when it is a symlink or not a
  regular file, is group- or world-writable, is owned by anyone but root or the node's user, sits under `/etc` without
  being owned by root, lives in a directory others can write to, is larger than 256 KiB, is not valid JSON, or is invalid
  against the schema. Unknown keys are errors, so a typo in `allow_delete` is never silently ignored; keys that start with
  `_` are comments. No file at all also means monitor mode. The file is re-checked about every 2 s.
- `node/install.sh` writes a monitor-mode starter policy only if none exists. The operator reads it and switches to manage.
- The hub mirrors some rules (mode, image patterns, gateway image present) so preflight can say no before a round trip.
  The node checks everything again.

## Consequences

- Whoever controls the hub controls at most what each machine's own administrator allowed.
- An administrator can tighten a policy without touching the hub, and the node follows within about 2 s.
- There are two places to configure: the hub's `hosts.json` and each node's policy.
- The hub learns a node's policy from `hello`, so a policy edit shows up in the hub only after the link reconnects (see
  ADR-0014). Until then the hub may offer an action the node now refuses, or refuse one the node now allows.
- Failing closed has a cost: a typo anywhere makes the whole file invalid and silently takes the machine back to monitor
  mode. The reason is visible as `policy.error` on the host and in `ecnode check`.
- On the hub's own machine a policy under the user's home directory is only as strong as that user account, because the
  hub runs as that user. A root-owned `/etc/ecnode/policy.json` is the real boundary there too.
- The policy does not protect against root on that machine, against a container-runtime or kernel escape, or against a
  script the policy itself registers.
- Some switches have no hub-side user yet: the node honours `allow_pull`, but the hub sends no image pulls today.

## Alternatives considered

- **Policy kept on the hub.** A compromised hub edits it; the boundary would protect nothing.
- **Only an sshd `command=` restriction.** It restricts which program runs, not what that program does.
- **AppArmor or SELinux profiles.** Host-level changes that need an administrator and differ per distribution. They are a
  good addition on top of the policy, not a replacement for it.
- **A policy pushed and signed by the hub.** A compromised hub signs anything.
- **Capability tokens from a separate authorization service.** More moving parts than a few machines justify.
