# ADR-0019: Three management classes: managed, script, external

Status: accepted (contract v2, `docs/ARCHITECTURE.md` section 10.1, decision 19, and section 10.4); implemented in
`backend/src/engine_console/services/externals.py`, `services/scripts.py`, `services/lifecycle.py` and, on the node,
`scripts.py` and the `docker.external` operation. Supersedes the mechanism (not the decision) of ADR-0013: discovery now
runs in the node, and the hub-side `DISCOVERY_*` settings no longer have an effect.
**Verified live:** discovery of running engines on one workstation, including one script-managed engine and one external
engine, both shown monitor-only. **Fakes only:** running a script action, stopping and starting an external engine,
the busy and timeout paths of scripts.

## Context

Labs often start engines with their own scripts, and those scripts encode ordering rules (start the big model before the
small one, stop one before the other). ADR-0013 made engines the console did not create monitor-only. That is safe, but it
leaves an operator who wants to start or stop such an engine from the console with no way to do it that respects the
script. The console must use those scripts and never bypass them.

## Decision

`Instance.management` has three values.

- **`managed`**: containers the console created (label `engine-console=1`). Full control: start, stop, restart, remove,
  logs, settings. Ids are `inst_*`.
- **`script`**: an engine started by a script that is registered in that node's policy. Containers whose names match a
  script's `containers` patterns (shell-style) become one instance per script (`scr-<host>-<script>`). Start, stop and
  restart exist only as that script's actions: start is `up`, stop is `down`, restart is `down` then `up` as one job that
  stops at the first failure. The script's `status_action` runs only when asked.
- **`external`**: any other engine container or bare loopback endpoint (`ext-<host>-<slug>`, each `-` of the host id doubled so ids from two hosts never collide). Monitor-only, unless that
  node's policy sets `external_control` to `stop_start`, which allows plain stop and start (never restart, remove or logs)
  of a recognised engine container that is neither console-owned nor claimed by a script.

Scripts live only in the node's policy. The argv (an absolute program path, then fixed arguments) stays on the node; the
hub sees a script's name, label, engine, description, container patterns, and each action's name, label, allowed
arguments and `confirm` and `danger` flags. A run names a script, an action and at most one argument chosen from that
action's allowlist, which is appended as one argument. The program is started without a shell, in its own process group,
with only `PATH`, `HOME`, `USER`, `LOGNAME`, `LANG`, `XDG_RUNTIME_DIR`, `DOCKER_HOST` and `DOCKER_CONTEXT` taken from the
node's own environment (nothing from the hub). One run per script at a time (`busy` otherwise). A run is stopped, with
its whole process group, on its `timeout_s` (default 900 s), on cancel, and when the hub's session ends. Output lines are
scrubbed of secret-looking values before they leave the node.

## Consequences

- Ordering rules stay in the operator's scripts. The console cannot start a script's engine any other way.
- A script is the operator's own code and runs as the login user. If that user is root-equivalent through docker, so is the
  script; the policy cannot make an allowed script safer than it is.
- The hub learns only an exit code and the output. It cannot inspect or tune what a script starts, and it shows no logs for
  script or external containers (logs of other containers can hold secrets, so `docker.logs` is for console-owned
  containers only). The launch form can prefill from an external engine's parsed command line ("clone as managed").
- Matching is by container name. A pattern that is too broad folds other engines into one script instance, and a pattern
  that matches nothing leaves the script instance permanently `stopped`.
- A script-managed instance shows `starting` or `stopping` while an action runs, and `loading` when its container runs but
  does not answer yet.
- Discovery moved from the hub to each node: the node classifies containers by image and command signatures (vLLM, SGLang,
  Ollama, llama.cpp, TensorRT-LLM, Dynamo), redacts their argv on the node, never reads container environment values, and
  probes loopback ports from the policy's `discovery_ports`. Dynamo is recognised for monitoring only; there is no adapter.

## Alternatives considered

- **Adopt discovered containers and manage them directly** (rejected in ADR-0013): would bypass the ordering a script
  encodes and the ownership label that the node's stop, start and remove checks rely on.
- **Plain stop and start of externals only, no scripts.** Would let the console break an ordering rule the operator wrote
  a script to enforce.
- **Keep the script argv on the hub.** A compromised hub could run any command on every machine; with the argv in the
  node's policy the hub can only pick among what the machine's administrator listed.
- **Free-form command strings run through a shell.** An injection surface with no benefit over an argv list.
