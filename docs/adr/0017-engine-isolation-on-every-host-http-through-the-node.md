# ADR-0017: Engines stay isolated on every host, and the hub reaches them through the node

Status: accepted (contract v2, `docs/ARCHITECTURE.md` section 10.1, decision 17); implemented in
`node/src/ecnode/docker.py` (engine run and gateway), `node/src/ecnode/httpproxy.py` and
`backend/src/engine_console/services/nodehttp.py`. Extends ADR-0011.
**Verified live:** one managed vLLM launch on one GPU under `--cap-drop ALL`, `no-new-privileges`, a pids limit, the
internal network, a read-only model mount and the loopback gateway, ready in about a minute; chat, plain and streamed as
SSE, through the node. ADR-0011 listed GPU engines under `--cap-drop ALL` as unverified; this check covers one GPU, not
tensor parallelism. **Fakes only:** the same path on a remote machine, multi-GPU launches, and every engine other than
vLLM.

## Context

ADR-0011 gave each engine an `--internal` network, no published ports, and a per-instance nginx gateway that publishes
`127.0.0.1:<port>`. In 0.3 the hub reached that loopback port directly. With several machines the gateway of a remote
engine is on another machine's loopback, so the hub cannot reach it, and exposing it would undo the isolation.

## Decision

- The isolation of ADR-0011 applies on every host, from that host's policy: the engine runs with `--cap-drop ALL`,
  `--security-opt no-new-privileges:true`, `--pids-limit 4096`, `--restart no` and `--pull never` on the policy's
  `engine_network` (created `--internal`, refused if it already exists without it). The gateway comes from the policy's
  `gateway_image` and is read-only, `--cap-drop ALL`, user 101, publishing only `127.0.0.1:<port in port_range>`.
- The hub reaches an engine's HTTP API through the node: `http.request` connects to `127.0.0.1:<port>` on the node's own
  machine. The port must be a console gateway's published port or an engine endpoint in the node's current inventory;
  the method is `GET` or `POST`; the path starts with `/v1/`, `/health`, `/metrics`, `/version`, `/api/` (Ollama),
  `/get_model_info`, `/get_server_info` or `/server_info`; hop-by-hop and `Host` headers are dropped; redirects are never
  followed; the default timeout is 600 s. The response head arrives as one event, the body as DATA frames under the hub's
  credit.
- Health probes, chat (plain and SSE), benchmarks and the metrics pull fallback all use this one path, for local and
  remote engines alike.

## Consequences

- A remote engine needs no network exposure, no extra port and no credential. Chat, benchmarks and health checks behave
  the same everywhere.
- Every response byte crosses the node, the ssh session and the hub. One link carries a node's whole traffic: a stream has
  an 8 MiB credit window, and a node runs at most 64 requests at once (more get `busy`, HTTP 503). A benchmark at high
  concurrency, a chat stream and a model copy on the same machine share that one pipe. Throughput over ssh has not been
  measured.
- A new engine API path needs a change to the protocol's path allowlist.
- Other applications on the LAN cannot use the hub as a proxy to remote engines. The opt-in of ADR-0011 stays a
  per-machine docker matter: the operator joins another container to that machine's engine network.
- Readiness is a 200 on the adapter's health path through the node. While a node's link is down the instance is shown as
  `unreachable` and the supervisor leaves its state alone instead of inventing a failure; the containers keep running.

## Alternatives considered

- **Publish engine ports on the LAN and let the hub connect.** Exposes engines to the network and needs an authentication
  story per engine, which several of them do not have (Ollama and TensorRT-LLM have no API key option).
- **ssh port forwarding to the gateway.** Forwarding is switched off on purpose (`ClearAllForwardings=yes`, `restrict` in the
  key line) so the key can never be used to reach anything but the node.
- **A reverse proxy on each machine.** One more listening service per machine to configure and secure.
- **Run the hub's HTTP client on each machine.** That is what `http.request` is, with the allowlists in the node.
