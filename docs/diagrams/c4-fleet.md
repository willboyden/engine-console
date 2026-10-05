# C4 fleet view: hub, links, nodes, policies and trust boundaries

Contract v2 (docs/ARCHITECTURE.md section 10, docs/FLEET-PROTOCOL.md). Names such as `gpu-box-2`, `edge-1` and the
addresses are examples. The second diagram is the state machine of one link (`LinkState` in `domain/fleet.py`).

## Structure and trust boundaries

```mermaid
flowchart LR
  subgraph hub["Hub machine"]
    direction TB
    svc["Hub services<br/>lifecycle, library, downloads,<br/>scripts, metrics, fit"]
    fleet["FleetService<br/>one NodeLink per host in hosts.json"]
    keyf["Hub ssh key 0600<br/>hosts.json 0600, owned by the user"]
    svc --> fleet
    fleet --- keyf
  end
  subgraph m0["Host 'local' (the hub's machine)"]
    direction TB
    n0["ecnode --stdio<br/>subprocess, env whitelist"]
    p0["policy: /etc/ecnode/policy.json<br/>or ~/.config/ecnode/policy.json"]
    d0[["docker, NVML, /proc, /sys"]]
    r0[("model roots")]
    n0 --- p0
    n0 --> d0
    n0 --> r0
  end
  subgraph m1["Host 'gpu-box-2' (10.0.0.5)"]
    direction TB
    s1["sshd<br/>restrict,from=hub,command=ecnode"]
    n1["ecnode --stdio"]
    p1["policy.json<br/>root-owned, mode monitor"]
    d1[["docker, NVML, /proc, /sys"]]
    r1[("model roots<br/>/data/hf, /data/gguf")]
    s1 --> n1
    n1 --- p1
    n1 --> d1
    n1 --> r1
  end
  subgraph m2["Host 'edge-1' (10.0.0.6)"]
    direction TB
    s2["sshd<br/>restrict,from=hub,command=ecnode"]
    n2["ecnode --stdio"]
    p2["policy.json<br/>root-owned, mode manage"]
    d2[["docker"]]
    r2[("model roots<br/>NFS mount of gpu-box-2")]
    s2 --> n2
    n2 --- p2
    n2 --> d2
    n2 --> r2
  end
  fleet ==>|"subprocess pipes"| n0
  fleet ==>|"ssh, hub key"| s1
  fleet ==>|"ssh, hub key"| s2
  r2 -.-|"same bytes: listed once, visible_on both"| r1
  hf[("Hugging Face")]
  proxy["Egress proxy (optional)"]
  svc -->|"HTTPS, in-app allowlist"| proxy --> hf
```

Trust boundaries, from the outside in:

1. Browser or curl to the hub: Host allowlist, Origin and CSRF checks, bearer key (loopback is not trusted).
2. Hub to node: a pipe (local) or an ssh session whose key sshd restricts to the node command. The hub treats every
   node as untrusted data (`proto.sanitize_*`, flood and credit limits). The node treats the hub as its operator but
   obeys its own policy first and never lets the hub edit it.
3. Node to docker and the filesystem: argv lists only, owned containers only (label `engine-console=1`), file
   operations confined to the policy's model roots.
4. Engine to anything else: internal network, no published ports, no route out.

## One link

```mermaid
stateDiagram-v2
  [*] --> disabled: enabled false in hosts.json, or FLEET_ENABLED=false
  [*] --> connecting: link starts
  connecting --> up: hello received and protocol matches
  connecting --> incompatible: hello protocol differs, retry in 300 s
  connecting --> down: no hello in 25 s, ssh failed, node not installed, login refused
  up --> stale: no sample for 6 s
  stale --> up: a sample arrives
  up --> down: node closed, silent for 30 s, protocol error, flood, over credit
  stale --> down: same causes
  down --> connecting: wait 1 s doubling to 60 s, or 300 s when only a person can fix it
  incompatible --> connecting: after 300 s
```

`up` and `stale` are the only states in which numbers from the node are shown as live. A request to a node whose link is
`connecting`, `down`, `incompatible` or `disabled` fails at once with `node_unavailable` (HTTP 503). The Reconnect
action skips the wait and restarts the attempt.

Not rendered: these diagrams were written by hand and have not been run through a Mermaid renderer.
