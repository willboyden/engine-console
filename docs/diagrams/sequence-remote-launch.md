# Sequence: starting an engine on another machine

Contract v2 (docs/ARCHITECTURE.md sections 10.4 and 10.11, docs/FLEET-PROTOCOL.md sections 4 and 5; code:
`services/lifecycle.py`, `services/fleet.py`, `node/src/ecnode/ops.py`). The machine here is `gpu-box-2`. The link to it
is already up: the hub started `ssh ... -- user@10.0.0.5 /usr/local/lib/ecnode/ecnode --stdio` earlier, sshd ran the
forced command from the restricted key, and the node sent its `hello` (including its policy summary).

For the hub's own machine the sequence is the same with the ssh hop replaced by the subprocess pipes. Status of this
sequence: the local variant was exercised live by the maintainers (one vLLM instance on one GPU, real rootless Docker); the ssh
variant has only been exercised against fakes and a real `ssh -G` check of the
command line.

```mermaid
sequenceDiagram
  actor Op as Operator
  participant UI as Frontend
  participant API as Hub API
  participant LC as LifecycleService
  participant NL as NodeLink gpu-box-2
  participant ND as ecnode on gpu-box-2
  participant DK as Docker on gpu-box-2
  participant GW as Gateway sidecar
  participant ENG as Engine container
  Op->>UI: Launch (machine, model, engine, GPUs, settings)
  UI->>API: POST /instances/preflight
  API->>LC: preflight
  Note over LC,NL: reads what the node pushed: hello (policy mode, engine_images, gateway_image, port_range), sample, inventory, disks, and the library
  LC->>LC: link up, mode manage, image allowed, model in that machine library, fit against its GPUs, free port
  LC->>NL: docker.image_present
  NL->>ND: request over the ssh session
  ND->>DK: docker image inspect
  DK-->>ND: present
  ND-->>NL: result
  LC-->>UI: checks with level ok, warn or block
  UI->>API: POST /instances
  API->>LC: create (per-host lock, preflight again, row state starting)
  LC->>NL: net.ensure_internal
  NL->>ND: request
  ND->>ND: refuse unless policy mode is manage
  ND->>DK: create the policy network with --internal, or verify it is internal
  LC->>NL: docker.run (spec, secret values only inside this request)
  NL->>ND: request
  ND->>ND: clean arguments, check image against engine_images, mounts are root plus relative path inside the realpath of a policy root, GPU uuids are known, shm below max_shm_gib
  ND->>DK: docker run -d --pull never --cap-drop ALL --security-opt no-new-privileges:true --pids-limit 4096 --restart no (secrets via child environment, never argv)
  DK->>ENG: create and start
  ND-->>NL: result with container id
  LC->>NL: gateway.start (host port from the policy range)
  NL->>ND: request
  ND->>DK: create gateway from the policy gateway_image on the default bridge, publish 127.0.0.1:port, connect to the engine network, start
  DK->>GW: start
  alt port already in use
    ND-->>NL: error port_in_use
    LC->>NL: gateway.start with the next free port, up to 8 tries
  end
  LC-->>UI: 201, state starting
  loop every 2 s until ready, failed or STARTUP_TIMEOUT_S
    LC->>LC: read the container from the pushed inventory
    LC->>NL: docker.logs (tail 40)
    NL->>ND: request
    ND-->>NL: scrubbed log text, parsed by the adapter for phase and percent
    LC->>NL: http.request GET health path on the gateway port
    NL->>ND: request
    ND->>GW: GET 127.0.0.1:port/health
    GW->>ENG: TCP forward
    ENG-->>LC: 200 gives state ready (answer travels back through the node)
  end
  Note over Op,ENG: Chat and benchmarks use the same hop: http.request POST /v1/chat/completions, the response body streamed back as DATA frames
  Note over LC,ND: if the link drops the instance shows unreachable and keeps its last state, the containers keep running, and on reconnect the hub adopts them again from the labels
```

What the node refuses, and the code the hub shows (full list in OPERATIONS.md, troubleshooting):

- policy mode is `monitor`: `monitor_only` (HTTP 409).
- image not matched by the policy's `engine_images`: `policy_denied` (403); the hub's own preflight says
  `image_not_allowed` first.
- the policy names no `gateway_image`: preflight check `gateway_missing`; `gateway.start` would answer `policy_denied`.
- a mount outside every policy root, or a path that leaves its root: `policy_denied` / `bad_args`.
- a node that cannot be reached: `node_unavailable` (503), and preflight check `host_down`.

Not rendered: this diagram was written by hand and has not been run through a Mermaid renderer.
