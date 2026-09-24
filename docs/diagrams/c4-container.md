# C4 level 2: containers (deployable units)

```mermaid
flowchart TB
  subgraph host["Workstation (user uid)"]
    fe["Static frontend<br/>ES modules + web components"]
    api["FastAPI process<br/>127.0.0.1:8791<br/>Host/Origin/CSRF checks"]
    db[("SQLite WAL console.db")]
    sec["0600 files: admin key,<br/>hf_token, secrets/ID.json"]
    cache[("HF cache dir<br/>read-write for console only")]
    proxy["Host mitmproxy<br/>127.0.0.1:8082"]
  end
  dk[["Rootless Docker"]]
  subgraph bridge["default bridge"]
    gw["Gateway sidecar per instance<br/>nginx stream, read-only<br/>publishes 127.0.0.1:18000-18099"]
  end
  subgraph internal["ai-lab-engines (--internal, no external route)"]
    eng["Engine container<br/>no published ports, cap-drop ALL"]
  end
  router["LiteLLM router<br/>(opt-in: joined to ai-lab-engines)"]
  hf[("Hugging Face")]
  otel["OTel collector :4317"]
  fe -. "served by" .-> api
  api --- db
  api --- sec
  api -->|writes| cache
  cache -->|"mount, read-only"| eng
  api -->|docker CLI| dk
  dk --> gw
  dk --> eng
  api -->|"HTTP via loopback port"| gw
  gw -->|"TCP forward"| eng
  api -->|"HTTPS, EGRESS_PROXY"| proxy -->|allowlist| hf
  api -->|OTLP| otel
  router -. "internal_endpoint, opt-in" .-> eng
```
