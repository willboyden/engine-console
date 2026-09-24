# ADR-0013: External engine discovery (monitor-only)

Status: accepted and implemented (`services/discovery.py`, `tests/core/test_discovery.py`).
**Verified live on 2026-09-24** against three real engines (one SGLang and two vLLM containers): all detected and
`ready`, metrics returned for a vLLM engine, no secret in the returned JSON. **Fakes only (not live):** the
`auth_required` and `unreachable` states, ollama, and the llama.cpp, TensorRT-LLM and Dynamo classification.

## Context

Operators usually already have vLLM or SGLang containers running before they install the console (started from
compose files or by hand). A console that shows an empty dashboard until every engine is re-launched through it is
a poor first impression; oMLX-style dashboards are useful immediately. But the console did not create these engines:
it must not stop, restart, reconfigure or delete them, and it must not trust what they report.

## Decision

- **Discovery.** At start-up, every `discovery_interval_s` (default 10 s) and on `POST /instances/discover` (admin only),
  the console lists engines it did not launch: (1) running containers, via `docker ps` and `docker inspect`, skipping
  containers that carry the console's own ownership label (`engine-console=1`), classified by data-driven image and
  command-line signatures (vLLM, SGLang, ollama, llama.cpp, TensorRT-LLM, Dynamo, generic OpenAI-compatible); images that
  are known not to be engines (routers, UIs, proxies, databases, telemetry) are never listed; and (2) `127.0.0.1` ports
  in `discovery_ports` (default 8000, 30000, 11434, 8001 and 18000-18099) that are not already claimed by a container.
  Probes are GET-only on `/v1/models`, `/health` and `/metrics`, with 1 s connect and 2 s read timeouts, a 1 MB body cap
  and no redirects.
- **Monitor-only instances.** They appear in the normal instance list with `managed=false`, `source="external"`, id
  `ext-<slug>`, and `container_name`, `image`, `engine`, `endpoint`, `served_models`, `state_reason` and `history_since`
  (epoch seconds; metrics history starts at discovery and is never back-filled). States: `ready`, `auth_required`
  (the engine answered 401/403 on the probes), `unreachable` (container found but no published loopback port answers, or
  no published port at all) and `stopped` (a vanished container is shown as stopped for one cycle, then dropped).
  Discovered instances are held in memory, not persisted. Params are display-only, parsed from a redacted command line.
- **No control.** Start, stop, restart, remove, patch, logs and command on a discovered instance return HTTP 409
  `instance_not_managed`. Chat works when the engine is `ready`. A benchmark sends real load to an engine the console
  does not own, so `POST /bench` needs `confirm_external: true`; without it the response is 400
  `confirm_external_required`.
- **Settings.** `discovery_enabled` (default true), `discovery_ports`, `discovery_interval_s` (default 10).
- **Safety properties.** Read-only; loopback only; GET only. `Config.Env` of a container is never read into the
  console's data structures, so environment values are never returned. Secrets in command-line arguments are redacted.
  Container names, images, labels and model names are untrusted: length-capped, control characters stripped, never
  used to build commands, SQL or paths. Metadata that `docker inspect` returns for foreign containers (image, command
  line, published ports, labels, name) is read; environment is not.

## Consequences

- A useful dashboard on first run, and a migration path: observe first, adopt (relaunch as managed) later.
- Cost: more Docker inspection and periodic probes; the fit estimator and preflight are blind to VRAM used by
  discovered engines except through measured free VRAM (already used by `fits_if_stop`-style checks).
- A discovered engine is outside the console's isolation model: it may publish ports on other interfaces, have
  network egress, or run with looser capabilities. The UI must not imply otherwise.
- A port that answers no probe is not listed as a bare endpoint (only containers can be `unreachable`).
- A loopback port probe can misidentify an unrelated local service; the engine is identified by its response shape,
  so a hostile local service could masquerade as an engine and feed arbitrary metrics and names to the UI.
  Escaping and size limits are the mitigation; see SECURITY.md.
- Metrics history for discovered engines is only as long as the console has been watching.

## Alternatives considered

- Adopt and manage discovered containers automatically: rejected, it would let the console mutate containers it did not
  create and bypass its own ownership-label check.
- Require every engine to be launched through the console: rejected, poor first-run experience.
- Read the container's environment to detect API keys or model names: rejected, environment values are the most likely
  place for secrets.
