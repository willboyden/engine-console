# ADR-0011: Engine network isolation and per-instance gateway

Status: accepted 2026-09-24. Live self-test on real rootless Docker verified: the internal network, an engine
with no egress and no published ports, and the gateway forwarding on 127.0.0.1. **Not verified:** GPU engines
under `--cap-drop ALL` or tensor parallelism.

## Context

Originally each engine published `127.0.0.1:<port>` and joined the shared `ai-lab` bus. That gave a model
server (which loads untrusted weights and runs a large Python stack) a route to the Internet and to every
other service on the bus, contradicting deny-by-default egress (AGENTS.md rule 1).

## Decision

- Engines run on the Docker network `ai-lab-engines`, created with `--internal` (no external routing).
  `ensure_internal_network` refuses to reuse an existing network of that name that is not internal.
- Engines publish no ports. Each instance gets a sidecar **gateway**: nginx (pinned by digest, must already be
  present; the console never pulls) doing a TCP `stream` forward to `<container>:<port>`. It is created on the
  default bridge so it can publish `127.0.0.1:<port>` (range 18000-18099), then attached to the internal network.
  The gateway is read-only, `--cap-drop ALL`, non-root (101), `--pids-limit 256`, and its config is generated from a
  slugged name and an integer port only. A TCP forward does not buffer, so SSE passes through unchanged.
- The gateway is labelled `ai-lab.console=1`, role `gateway` and its owning instance id; stop/remove verify all three.
- The console reaches an engine at `127.0.0.1:<port>` (the gateway). The supervisor treats a missing or stopped
  gateway as a failed instance.
- **Router access is opt-in.** The router cannot see the engines by default. An operator who wants LiteLLM to reach
  an engine adds `ai-lab-engines` to the router's `networks` and uses `instance.internal_endpoint`
  (`http://<container>:<port>`). This also lets that engine reach the router container over that network, so it is a deliberate, human-approved change (AGENTS.md rule 4). The console does not edit the router compose.

## Consequences

- An engine cannot exfiltrate, and no other container or LAN host can reach it. Weights come only from the read-only cache mount.
- Cost: one extra container and a TCP hop per instance; a gateway image to keep pinned and present; more state
  to reconcile on adopt/stop (legacy pre-gateway containers that publish their own port are still recognised).
- nginx quirk found only in the live test: it needs `daemon off;` or PID 1 exits. Mocks did not catch it.
- The gateway has no auth: anything on the host that can reach the loopback port can talk to the engine (as before).
- The router opt-in is a manual step and is documented in OPERATIONS.md; it has not been exercised end to end here.

## Alternatives considered

- Keep publishing engine ports directly on 127.0.0.1: simplest, rejected because the engine keeps its egress path.
- iptables/nftables rules per engine: host-level change needing approval, not portable across rootless setups.
- Attach the router to the engine network automatically: rejected, it silently widens the router's blast radius.
- A Python TCP proxy inside the console: rejected, puts a data path in the control plane and loses process isolation.
