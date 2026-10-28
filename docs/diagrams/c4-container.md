# C4 level 2: containers (deployable units)

This is the contract v2 picture (docs/ARCHITECTURE.md section 10): the hub never runs docker, reads `/proc` or opens a
model directory itself. Every machine, the hub's own included, is reached through a node (`ecnode --stdio`). The next
page, `c4-fleet.md`, shows the same system from the fleet's point of view (links, policies, trust boundaries).

```mermaid
flowchart TB
  op(["Operator<br/>browser or curl, API key"])
  hf[("Hugging Face<br/>allowlisted hosts only")]
  otel["OpenTelemetry collector :4317"]
  subgraph hubm["Hub machine (one user uid)"]
    fe["Static frontend<br/>ES modules + web components"]
    api["Hub: engine-console (Rust binary) 127.0.0.1:8791<br/>Host / Origin / CSRF checks, bearer keys"]
    db[("SQLite WAL console.db<br/>incl. last-known library per host")]
    sec["0600 files: hub ssh key, hosts.json,<br/>hf_token, secrets/ID.json, bootstrap key"]
    proxy["Allowlisting HTTPS proxy<br/>(optional) 127.0.0.1:8082"]
    lnode["Node 'local': ecnode --stdio<br/>subprocess, whitelisted environment"]
    lpol["Node policy (monitor | manage)"]
    ldock[["Docker (rootless)"]]
    subgraph lnet["engine network (--internal) + loopback gateway"]
      lgw["Gateway sidecar per instance<br/>nginx stream, read-only<br/>127.0.0.1:18000-18099"]
      leng["Engine container<br/>no ports, cap-drop ALL"]
    end
    lroots[("Model roots from the policy<br/>mounted read-only")]
  end
  subgraph remote["Each other machine (reached over ssh)"]
    sshd["sshd: hub key restricted to<br/>command=ecnode --stdio"]
    rnode["Node: ecnode --stdio<br/>runs as the login user"]
    rpol["Node policy, root-owned<br/>/etc/ecnode/policy.json"]
    rdock[["Docker"]]
    subgraph rnet["engine network (--internal) + loopback gateway"]
      rgw["Gateway sidecar"]
      reng["Engine container"]
    end
    rroots[("Model roots, maybe an NFS mount")]
  end
  fe -. "served by" .-> api
  op -->|"HTTP, Authorization: Bearer"| api
  api --- db
  api --- sec
  api -->|"subprocess stdin/stdout"| lnode
  lnode --> lpol
  lnode -->|docker CLI| ldock
  ldock --> lgw
  ldock --> leng
  lroots -->|"mount, read-only"| leng
  lnode -->|"fs ops, confined to roots"| lroots
  lnode -->|"http.request, loopback only"| lgw
  lgw -->|"TCP forward"| leng
  api ==>|"ssh, framed protocol"| sshd
  sshd --> rnode
  rnode --> rpol
  rnode -->|docker CLI| rdock
  rdock --> rgw
  rdock --> reng
  rroots -->|"mount, read-only"| reng
  rnode -->|"http.request, loopback only"| rgw
  rgw -->|"TCP forward"| reng
  api -->|"HTTPS, EGRESS_PROXY"| proxy -->|allowlist| hf
  api -->|OTLP| otel
```

Reading the picture:

- The only link from the hub to another machine is the ssh session that starts that machine's node. Nothing new
  listens on the network on either side. The hub's own machine uses a plain subprocess instead of ssh.
- Downloads and copies are not drawn: bytes go from Hugging Face (or from one node's `fs.read`) through the hub into a
  node's `fs.write`; the hub keeps no model files (ADR-0018).
- Engines discovered by a node (containers it did not create, bare loopback ports) are listed from the node's
  inventory. They are never attached to the engine network or given a gateway by the console (ADR-0013, ADR-0019).
- Access from another container (for example an OpenAI-compatible router) to an engine is still opt-in and is a
  per-machine docker matter: the operator joins that container to that machine's engine network. See OPERATIONS.md.

Not rendered: this diagram was written by hand and has not been run through a Mermaid renderer.
