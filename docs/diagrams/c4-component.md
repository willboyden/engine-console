# C4 level 3: hub components

Rust crate `backend/crates/ec-hub` (source under `src/`). Since contract v2 the hub reaches docker, GPUs and model
files only through a node (see `c4-fleet.md`), so it has no command-runner or GPU-probe port of its own.

```mermaid
flowchart LR
  subgraph http["api/ + app.rs + server.rs"]
    sec["Request pipeline (app.rs handle_request)<br/>Host, CSRF, auth, roles, audit, metrics, span"]
    routes["api/routes: system, downloads, library, instances,<br/>profiles, observe, chat, admin, hosts, scripts"]
  end
  subgraph core["services/ (application core)"]
    fit[FitService] --- fitfn["fit::estimate_fit<br/>(pure)"]
    lc[LifecycleService]
    fl[FleetService]
    lib[LibraryService]
    dl[DownloadService]
    pr[ProfileService]
    mt[MetricsService]
    bn[BenchService]
    ch[ChatService]
    us[UsageService]
    st[SettingsService]
    au[AuditService]
  end
  subgraph ports["domain/ports.rs, domain/fleet.rs, adapters/base.rs"]
    ea{{EngineAdapter trait}}
    hub{{HubClient trait}}
    fp{{FleetPort / NodeClient traits}}
  end
  vllm[vLLM, SGLang, TensorRT-LLM,<br/>llama.cpp, Ollama adapters] --> ea
  hfh[hf_http: HF client<br/>egress allowlist] --> hub
  nl["NodeLink per host<br/>(ecnode --stdio over a pipe or ssh)"] --> fp
  sec --> routes --> core
  lc --> ea & fp & fit & dl
  fit --> ea & fp
  dl --> hub & fp
  mt --> ea
  core --> store[("Store: SQLite (rusqlite)")]
```

Wiring lives in `container.rs` (`build_container`), which accepts fakes for the ports so the tests run offline.

Not rendered: this diagram was written by hand and has not been run through a Mermaid renderer.
