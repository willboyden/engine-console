# Engine Console: the Rust port (porting record)

Engine Console's hub (`engine-console`: the API and UI server, the desktop launcher and the `hosts` CLI), its node
agent (`ecnode`) and the wire protocol between them (ecnode protocol v1) were ported from Python to Rust. The Rust code replaced the Python
code in release 0.5.0. The last Python code is the tag `python-final` (commit 15ac7a6); it stays runnable as the parity
reference: the harness, the protocol scripts and the golden generators export it and run it next to the Rust binaries
(§17).

This document records the port: the design decisions (D1-D17), the map from each Rust file to its Python source, the
interfaces, the routes, the wire protocol, the settings, the persisted state, the metrics, the subprocess argv and the
security controls that had to stay equal, the deliberate deviations from Python (DV1-DV50), the Python behaviour kept
on purpose, the aarch64 build, the tests and gates that prove parity, what the cutover had to preserve outside the
repository, the risks, and the addenda (§A1-§A14), the decisions and changes made while the port was done, in order. An
addendum that amends an earlier section names it. Python paths (`config.py`, `services/*.py`, `node/src/ecnode/*.py`,
`backend/tests/**`) are those of the `python-final` tree.

Contents: 1 Decisions · 2 Workspace · 3 Module map · 4 Areas · 5 Interfaces · 6 Routes · 7 Wire protocol ·
8 Settings and files · 9 Persisted state · 10 Metrics and tracing · 11 Subprocess argv · 12 Security controls ·
13 Request validation (422) · 14 Deviations · 15 Python behaviour kept · 16 aarch64 · 17 Testing, harness, gates ·
18 Consumers outside the repo / cutover · 19 Risks · 20 Addenda

---

## 1. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | Cargo workspace in `backend/` (in `rust/` until the cutover, §A14): `ec-proto` (wire protocol, lib + dev bin `ecproto-drive`), `ec-hub` (lib + bin `engine-console`), `ecnode` (lib + bin `ecnode`). Edition 2024, Rust 1.99.1 (`backend/rust-toolchain.toml`), harness2-workbench's lints. | One protocol crate both binaries link; the hub and the node ship separately. |
| D2 | HTTP server: axum 0.8 / hyper as transport only. **Routing is our own Starlette-compatible router** (`api/router.rs`): match the percent-decoded path, routes in Python's registration order, `{x}` = one segment, `{x:path}` = rest, `$` also before one trailing `\n`, path match + method mismatch = 405 only when nothing matches fully, slash redirect only when nothing matches at all. Every request goes through one function (`app.rs handle_request`) that is the SecurityMiddleware. | axum's router matches raw paths, answers 405/404 differently and has no partial-match semantics; parity needs Starlette's exact order. |
| D3 | Request validation: our own pydantic-v2-compatible validator (`api/validate.rs` + schemas in `api/schemas.rs`), producing FastAPI's 422 `errors` list (`loc`/`msg`/`type`) and lax coercions. Pinned by 720 vectors generated from pydantic itself (`tests/data/validation_vectors.json`). | serde errors cannot reproduce pydantic's messages; the frontend and tests read them. |
| D4 | Dependencies: versions already vetted in the harness2-workbench project's `Cargo.lock` reused where they fit. New crates: `nvml-wrapper 0.13.0` (+ `libloading` 0.8.9), `opentelemetry-otlp` with `grpc-tonic` (`tonic 0.14.6`, prost; no `protoc` needed, generated code is shipped), `md-5 0.11.0`, `fancy-regex 0.19.2` (ecnode `util.rs`: Python's secret patterns use look-arounds), nix feature `hostname`. **Cooldown: every crates.io package in `Cargo.lock` was published at least 14 days ago**, checked against crates.io by the gate `python3 -I backend/tools/crate-age.py` (§17.1, §A3); the only younger versions are three security fixes listed with their RustSec advisories in `backend/tools/crate-age-exceptions.toml`. `clippy.toml large-error-threshold = 512` (problem errors carry the RFC 7807 body). | Owner's policy; harness2's lock is the vetted set. |
| D5 | `GET /api/v1/openapi.json` serves Python's generated OpenAPI document (`crates/ec-hub/assets/openapi.json`, regenerated from the Python reference by `backend/parity/gen/openapi_python.py`) verbatim, except `info.version`, which is the workspace version since 0.5.0 (§A14). | Exact parity without a schema generator; the port does not change the API surface. |
| D6 | The SQL migrations (`backend/crates/ec-hub/migrations/*.sql`) and the learn content (`backend/crates/ec-hub/learn/**/*.json`) are **embedded at build time** (`include_str!`). During the port they stayed in the Python package (`backend/src/engine_console/migrations/`, `backend/src/engine_console/learn/`) and the Rust hub embedded them from there; they moved into ec-hub at the cutover (§A14). | One source of truth while both implementations ran. |
| D7 | Node binary targets `x86_64-unknown-linux-gnu` and `aarch64-unknown-linux-gnu` (glibc, dynamically linked against libc only), not musl. See §16. | Optional NVML is a `dlopen` of `libnvidia-ml.so.1`, impossible from a static musl binary. |
| D8 | The Rust hub's **local node** is `NODE_COMMAND` when set (same JSON list as Python), else `<directory of the running engine-console>/ecnode --stdio`, plus `--policy NODE_POLICY` when set. (Python: `sys.executable -I -m ecnode --stdio [--policy]`.) Remote nodes are unchanged: `ssh ... /usr/local/lib/ecnode/ecnode --stdio`. | A single-binary deployment ships both binaries side by side; the cross-matrix swaps nodes with NODE_COMMAND. |
| D9 | Desktop launcher: `engine-console open` starts `[<this executable>, "--from-launcher"]` (any non-subcommand argv serves, as Python's `run()` ignores argv); `stop` signals only a recorded pid whose owner is the user and whose `/proc/<pid>/cmdline` is exactly that argv. | Python's `stop` keys on `[python, -I, -m, engine_console.main]`; the Rust launcher needs an equally distinctive argv. |
| D10 | Errors: every service returns `AppResult<T>` = `Result<T, AppError>`, `AppError = Problem | Node(NodeError)` (`domain/errors.rs`). The HTTP layer renders problems exactly as `api/errors.py` (title = code with spaces; node errors labelled with the host through `services/nodeerr.rs`). **Code not yet ported answered 501 `{"type":"urn:engine-console:not_ported","title":"not ported","status":501,"code":"not_ported","detail":"not ported: <python module>"}`** (a superset of a bare `{"detail":...}`, so the UI shows it like any problem). Where a stub would otherwise have answered a misleading 200/404, it carried a `PORTED` flag (`services::library::PORTED`, `EngineAdapter::ported()`) that the route checks after validation. Every route is ported now: `services::library::PORTED` is true, every adapter keeps the default `ported()` (true), and the parity run has 0 unported steps (§A10). | One error type across the code; the harness classifies 501s as "unported", never as diffs. |
| D11 | JSON: `serde_json` with `preserve_order` + `float_roundtrip`. API responses are compact UTF-8 (FastAPI's JSONResponse); Python-formatted text where bytes matter is produced by `pyjson.rs` (`json.dumps` default separators + `ensure_ascii` for SQLite/audit/secret files, compact + ensure_ascii for SSE frames, `repr(float)`, prometheus' `floatToGoString`). Parity is on JSON **values**: key order is free, list order, null vs absent, int vs float are not. | Python's dict order is insertion order; serde preserves it, but only values are compared. |
| D12 | Concurrency: the hub runs on a multi-threaded tokio runtime; shared service state uses `Mutex`/`RwLock`; the node keeps Python's thread design on `std::thread` (no tokio). Late-bound links (library ⇄ lifecycle, fit → residents, fleet → resolver/counts) are `Arc<dyn Fn>` closures over `Weak` refs, wired only in `container.rs`. | Mirrors Python's object graph without cycles. |
| D13 | Prometheus text is hand-written (`services/telemetry.rs`) to match `prometheus_client.generate_latest` byte for byte (order, `_created` gauges, sorted labels, Go-style floats, escaping). | Grafana and the parity harness read it. |
| D14 | Logs: JSON lines `{"ts","level","logger","msg"}` with secrets redacted, like `JsonFormatter`; timestamps in UTC (deviation DV2, §14). No access log. | Logs are not an API; no new dependency for local time zones. |
| D15 | Wire compatibility = identical frame bytes (header, kinds, limits, DATA layout, MAGIC) + identical JSON **values**; every INT field stays an integer, every NUM may be int or float as Python sends it. | Python's peers parse values; int-vs-float changes `clean()` results. |
| D16 | The shared registries (the `mod` lists, the route table `api/routes/mod.rs`, the service container `container.rs`, the adapter registry `adapters/mod.rs`) are core files that list every module from the start; the rest of this decision was about how the port was organised. | One place wires every module. |
| D17 | `services/{usage,nodeview,hostmem,nodeerr,libport}.rs` are core (pure helpers used by three or more areas); `services/metrics.rs` belongs to lifecycle, not core (§4). Otherwise a note about how the port was organised. | `metrics.rs` is lifecycle-coupled. |

---

## 2. Workspace

```
backend/
  Cargo.toml  Cargo.lock  rust-toolchain.toml  rustfmt.toml  clippy.toml  deny.toml
  PORTING.md (this)  TEST-MAP.md
  crates/ec-proto/   src/{lib,clean,specs,sanitize,msg,asyncio,pyfmt}.rs  src/bin/ecproto-drive.rs  tests/
  crates/ec-hub/     src/ (see §3)  assets/openapi.json  migrations/*.sql  learn/**/*.json
                     tests/{common/kit.rs, test_*.rs, data/}
  crates/ecnode/     src/ (see §3)  tests/
  parity/            run.mjs routes-coverage.mjs check-test-map.mjs lib/ corpus/ fakes/ gen/ proto/
                     routes.python.json allowed-diffs.json test-map.json README.md
  tools/             crate-age.py crate-age-exceptions.toml
  target/            (gitignored; also holds python-release/<commit>/, the Python reference, §17)
node/                install.sh  README.md  examples/   (the Python package was removed in 0.5.0, §A14)
```

Build: `make build` (debug: `cd backend && cargo build --locked --workspace --bins`) or `make release` (§16), with
rustup's cargo on PATH.

---

## 3. Module map

Every Rust file and the Python source it ports. Hub paths are relative to `backend/src/engine_console/`, node paths to
`node/src/ecnode/`, both of `python-final`. Which area each file belongs to: §4.

### 3.1 Hub: `crates/ec-hub/src`

| Rust file | Python source | Notes |
|---|---|---|
| `main.rs` | `main.py run()` | |
| `lib.rs` | `__init__.py` | |
| `config.rs` | `config.py` | |
| `container.rs` | `container.py` | wires every service |
| `app.rs` | `main.py create_app` + `api/security.py` middleware | |
| `server.rs` | uvicorn (the listener) | §A12, DV37 |
| `sync.rs`, `panics.rs`, `fsutil.rs`, `tools.rs` | – (hardening: poison-proof locks, the panic hook and guards, bounded reads of regular files, the fixed tool PATH) | §A12, DV38-DV42 |
| `cli.rs` | `cli.py` | the `hosts` commands' work is in `services/hosts.rs` and `services/fleet.rs` |
| `desktop.rs` | `desktop.py` | |
| `pyjson.rs` | (Python text formats) | |
| `pyyaml/{mod,load,pyobj}.rs` | PyYAML 6.0.3 `safe_dump` (mod.rs) and `safe_load` (load.rs, pyobj.rs) | the hub's only YAML emitter and loader (§A9, §A10) |
| `learn/mod.rs` | `learn/__init__.py` | |
| `api/{http,router,deps,security,statics,validate,schemas}.rs` | `api/{errors,deps,security}.py`, `main.py` static | |
| `api/routes/{system,downloads,library,instances,profiles,observe,chat,admin,hosts,scripts}.rs` | `api/routers/*.py` | every route, its validation and serialisation; each calls one service method |
| `domain/{errors,fleet,learn,models,ports,refs,repo_id}.rs` | `domain/*.py` | the shared shapes (§5) |
| `adapters/{mod,base,stub}.rs` | `adapters/__init__.py`, `adapters/base.py` | the `EngineAdapter` trait; `try_*` forms §A5 |
| `adapters/pycompat.rs` | CPython semantics the adapters rely on | shared by every adapter (§A5, §A7) |
| `adapters/{vllm,vllm_params,vllm_parse}.rs` | `adapters/vllm*.py` | |
| `adapters/{sglang,sglang_params,sglang_parsers}.rs` | `adapters/sglang*.py` | |
| `adapters/{trtllm,trtllm_params,trtllm_parse}.rs` | `adapters/trtllm*.py` | |
| `adapters/{llamacpp,llamacpp_params,llamacpp_parse}.rs` | `adapters/llamacpp*.py` | |
| `adapters/{ollama,ollama_params}.rs` | `adapters/ollama*.py` | |
| `services/{mod,audit,common,hostmem,libport,nodeerr,nodeview,secrets_store,settings,store,telemetry,usage}.rs` | same-named `services/*.py` | |
| `services/{fleet,fleet_status,hosts,hardware,nodehttp}.rs` | same-named `services/*.py` | |
| `services/{library,downloads,hf,hf_http,fit,fitting,gguf_info,chat,bench,profiles}.rs` | same-named `services/*.py` | |
| `services/{lifecycle,runspec,docker,externals,discovery,scripts,metrics}.rs` | same-named `services/*.py` | |
| `assets/openapi.json` | `app.openapi()` | generated (D5) |
| `../migrations/*.sql`, `../learn/**/*.json` | `migrations/*.sql`, `learn/**/*.json` | the same files, embedded at build time (D6) |
| `tests/common/kit.rs` | `tests/core/conftest.py`, `fakes.py` | the test kit; includes `fakefleet.rs` as `kit::fakefleet` (§A4) |
| `tests/common/fakefleet.rs` (the kit's submodule, §A4) | `tests/core/fakefleet.py` (full op behaviour) | |
| `tests/test_*.rs` | `backend/tests/**` | TEST-MAP names the Rust test(s) for every Python test (§17.4) |

### 3.2 Node: `crates/ecnode/src`

| Rust file | Python source |
|---|---|
| `main.rs`, `entry.rs` | `__main__.py` (argparse-compatible CLI, `--stdio` fd handling) |
| `lib.rs` | `__init__.py` |
| `server.rs` | `server.py` (writer, requests, credit, reader, watchdog, push loops) |
| `errors.rs` | `errors.py` (OpError, Cancelled, Stream trait) |
| `agent.rs`, `policy.rs`, `ops.rs`, `util.rs`, `cli.rs`, `confine.rs`, `fsops.rs`, `docker.rs`, `discovery.rs`, `disks.rs`, `gpus.rs`, `memory.rs`, `hostinfo.rs`, `httpproxy.rs`, `models.rs`, `gguf.rs`, `scripts.rs` | same-named `*.py` |
| `util/proc.rs` | `util.py`'s subprocess half |
| `util/py.rs` | the Python semantics the node's text and path handling depends on (`str` methods, `float()`/`int()`, `round()`, posixpath, `json.dumps`, `fnmatch`, `shlex.split`) |
| `httpproxy/client.rs` | the part of Python's `http.client` the node uses (`httpproxy.py` `request`, `discovery.py` `http_get`) |
| `util/{fsread,panics,sync}.rs` | – (hardening: bounded reads of regular files, contained panics, poison-proof locks; §A11) |

### 3.3 Protocol: `crates/ec-proto` (complete port of `node/src/ecnode/proto.py`; §7).

### 3.4 Harness: `backend/parity/**` (§17).

---

## 4. Areas

The code and its tests are grouped in seven areas. `backend/TEST-MAP.md` and `backend/parity/test-map.json` use exactly
these names; the counts per area are in the header of `backend/TEST-MAP.md` (generated; §17.4). A test row whose area
differs from its file's carries the reason in its note.

| Area | Scope | Rust files | Python sources | pytest ids |
|---|---|---|---|---|
| `core` | configuration, wiring, the HTTP layer and listener, domain types, CLI, launcher, learn content, the shared services; the wire protocol and the node's server and entry point | hub `main.rs`, `lib.rs`, `config.rs`, `container.rs`, `app.rs`, `server.rs`, `sync.rs`, `panics.rs`, `fsutil.rs`, `tools.rs`, `cli.rs`, `desktop.rs`, `pyjson.rs`, `pyyaml/`, `learn/`, `api/**`, `domain/**`, `adapters/{mod,base,stub,pycompat}.rs`, `services/{audit,common,hostmem,libport,nodeerr,nodeview,secrets_store,settings,store,telemetry,usage}.rs`, `tests/common/kit.rs`; `ec-proto`; ecnode `main.rs`, `entry.rs`, `lib.rs`, `server.rs`, `errors.rs` | config, container, main, api/*, domain/*, cli, desktop, learn, services/{audit,common,hostmem,libport,nodeerr,nodeview,secrets_store,settings,store,telemetry,usage}; node proto/server/__main__/errors | 262 (259 mapped to Rust tests, 3 N/A) |
| `adapters: vLLM, SGLang` | the vLLM and SGLang adapters, and every adapter test that is not in the next area, including tests that cover all engines | `adapters/{vllm,vllm_params,vllm_parse,sglang,sglang_params,sglang_parsers}.rs` | `adapters/vllm*.py`, `adapters/sglang*.py` | 171 |
| `adapters: TensorRT-LLM, llama.cpp, Ollama` | those three adapters; every `tests/trtllm/` row, plus rows in shared test files whose id or parameters name trtllm, llamacpp or ollama | `adapters/{trtllm,trtllm_params,trtllm_parse,llamacpp,llamacpp_params,llamacpp_parse,ollama,ollama_params}.rs` | `adapters/trtllm*.py`, `adapters/llamacpp*.py`, `adapters/ollama*.py` | 609 |
| `fleet` | the hub's link to its nodes: hosts file and ssh, node links, fleet status, hardware, HTTP through a node | `services/{fleet,fleet_status,hosts,hardware,nodehttp}.rs`, `tests/common/fakefleet.rs` | `services/{fleet,fleet_status,hosts,hardware,nodehttp}.py`, `tests/core/fakefleet.py` | 430 |
| `library and data` | model library, downloads, Hugging Face, fit, chat, bench, profiles | `services/{library,downloads,hf,hf_http,fit,fitting,gguf_info,chat,bench,profiles}.rs` | the same-named `.py` | 425 |
| `node agent` | the node's operations, policy, confinement, docker, discovery, GPUs, models, scripts | every ecnode file not in `core` (§3.2) | `node/src/ecnode/*.py` except proto/server/__main__/errors | 349 |
| `lifecycle` | instances, run specs, docker, externals, discovery, scripts, metrics | `services/{lifecycle,runspec,docker,externals,discovery,scripts,metrics}.rs` | the same-named `.py` | 241 |

Total 2487 pytest ids (backend 2097, node 390): 2484 mapped to Rust tests, 3 N/A.

The areas were ported in parallel: the tests of one area run against fakes of the others' interfaces
(`tests/common/kit.rs`). An area counted as ported when every one of its TEST-MAP rows was mapped
(`check-test-map.mjs --owner <area>`), its routes answered the parity run with 0 diffs and none unported, its
protocol and e2e expectations were removed, and every gate passed.

---

## 5. Interfaces

Signatures are in the source files; their doc comments document them. Summary:

* **Domain shapes** (`domain/models.rs`, `domain/fleet.rs`, `domain/refs.rs`, `domain/learn.rs`): every pydantic API
  model as a serde struct, fields in Python order, `None` serialised as `null`, floats `f64`, ints `i64`.
* **Errors** (`domain/errors.rs`): `Problem {status, code, detail, extra}`, `NodeError {host_id, code, message}`,
  `AppError`, constructors `not_found`/`conflict`/`bad_request`/`unprocessable`/`forbidden`/`unauthorized`/
  `upstream_error`/`gated_model`/`too_many_streams`/`egress_proxy_unavailable`/`egress_denied`/`internal`,
  `not_ported(module)`.
* **Node ports** (`domain/fleet.rs`): `NodeClient` (host, link, latest(kind) -> `Option<Arc<Obj>>`, `call(op,
  args, timeout)`, `open(op, args) -> Box<dyn NodeStream>`), `NodeStream` (next/write/end/result/cancel),
  `FleetPort` (hosts, node), `Fleet: FleetPort` (start, stop, summaries, summary, detail, reconnect,
  set_instance_resolver, set_counts_provider). Pushed messages are re-published on the EventBus as topic
  `node.<type>` with payload `{"host_id", "msg"}`; link changes as topic `fleet` (HostSummary dump).
* **Hub client** (`domain/ports.rs`): `HubClient` (search, model, card, open_file) + `HubModel`, `RepoFile`,
  `ByteStream`.
* **Adapters** (`adapters/base.rs`): `EngineAdapter` trait (identity, catalog, presets, validate, build_launch[_ctx],
  compatibility, memory_model, host_memory_gib, parse_metrics, parse_startup_log, metric_prefixes, ported, and the
  `try_*` forms of §A5) and the types `ParamSpec`, `Explain`, `ModelInfo`, `Hardware`, `LaunchSpec`, `LaunchContext`,
  `Compat`, `FloatMap`. Each adapter module exposes `pub fn adapter() -> Arc<dyn EngineAdapter>`; the registry
  (`adapters/mod.rs`) lists the five in Python's order.
* **Library port** (`services/libport.rs`): `LibraryPort` (find, locate, model_info, hf_root, ollama_root, copies)
  + `NullLibrary`. `LibraryService` implements it; `LibraryService::find_copy(host, ref, own)` is Python's
  `find(..., own=)`.
* **Services** (constructors and public methods, all `AppResult`): `HardwareService` (node,
  host_memory, gpus, report, hardware), `NodeHttp` (open, request; `HttpOpts`, `HttpResult`, `HttpStream`
  chunk/line/read/close; caps `MAX_BUFFERED`, `MAX_LINE`, `MAX_STREAM`), `FleetService::new(cfg, bus)`,
  `hosts.rs` free functions (default_hosts, load_hosts, parse_hosts, check_remote_command, ssh_argv, ssh_env,
  local_env, local_argv, key_line, init_key; `HostsError::{Invalid, NoHubKey}`), `HfService` (model_info, detail,
  list_files, search; `SearchHit`, `GgufGroup`, `ModelDetail`, `SearchParams`), `HfHttpClient::new(endpoint,
  allowed, token_fn, EgressConfig)`, `FitService` (set_resident_provider, set_library, resolve_gpus, info_for,
  assess, assess_info), `LibraryService` (start, stop, models, model, copy, find_copy, local_models, roots, root,
  rescan, delete_copy + LibraryPort), `DownloadService` (recover, get, list_page, create, copy, pause, resume,
  cancel, delete, shutdown), `ChatService` (open_stream -> `ChatStream{lines}`, complete, conversations, prompts,
  arena), `BenchService` (recover, shutdown, get, list_page, start), `ProfileService` (get, list_page, create,
  update, delete, diff, export_yaml, import_yaml), `LifecycleService::new(LifecycleDeps)` (set_library,
  set_scripts, externals, get, list_page, all_instances, all_active, lookup, instance_for_container, counts,
  require_managed, touch, resident, instances_using, launch_for, preflight, create, start, stop, restart, remove,
  patch, logs, follow_logs, command, tick, and `adopt`, §A4), `ScriptService` (scripts, script, busy, run, restart,
  job, jobs, stream, shutdown), `MetricsService` (series, system_series, latest, run; `parse_window`), plus the core
  `SettingsService`, `AuditService`, `UsageService`, `Telemetry`, `EventBus`, `SseLimiter`, `Store`, `SecretStore`.
* These signatures were fixed for the duration of the port; each area could add public methods, private fields and
  private types in its own files. The changes made to shared interfaces are recorded as addenda: §A4
  (`Overrides::library`, `LifecycleService::adopt`) and §A5 (the `try_*` forms of `EngineAdapter`).

---

## 6. Routes

`backend/parity/routes.python.json` is the Python route table generated from the live app (`backend/parity/gen/
routes_python.py`: method, path, order, module, status code, path/query params with defaults and patterns, body
model): **79 API/static routes + `GET`/`HEAD /api/v1/openapi.json` (a plain starlette route) = 81**. The Rust
registry `api/routes/mod.rs` registers all of them in the same order. Handler rules (FastAPI's):

1. the router dependency `enforce_roles` (mutating methods: admin, or viewer on the exact viewer list), then the
   endpoint's own dependencies in parameter order (`Admin` = 403 "admin role required");
2. path, query and body errors **collected together** into one 422 (path first, then query, then body);
3. one service call; 201/202/204 as declared; JSON content type `application/json`, problems
   `application/problem+json`, logs `text/plain; charset=utf-8`, CSV `text/csv; charset=utf-8` +
   `Content-Disposition: attachment; filename="usage.csv"`, YAML `application/yaml`, SSE `text/event-stream;
   charset=utf-8` + `X-Accel-Buffering: no`, `/metrics` `text/plain; version=0.0.4; charset=utf-8`.
4. Unmatched: any path with a non-matching method is 405 `method_not_allowed` ("Method Not Allowed", no Allow
   header); GET of an unknown path is the SPA's 404 "no such file" (or Starlette's 404 "Not Found" when the
   frontend dir does not exist).
5. SSE responses take a slot from `MAX_SSE_CONNECTIONS`/`MAX_SSE_PER_KEY` (429 `too_many_streams`, extra members
   `scope` ("key"|"total") and `limit`), end after `SSE_IDLE_TIMEOUT_S` without an event; streamed chat replies
   take a slot from `MAX_CHAT_STREAMS`/`MAX_CHAT_STREAMS_PER_KEY` (429 `too_many_chat_streams`). Frames are
   `sse()` of services/common.py: `event: <name>\n` then one `data: ` line per payload line, then a blank line.
6. Static: only `/`, `/index.html`, and files under `css/`, `js/`, `i18n/` that resolve inside their directory;
   `Cache-Control: no-cache`, Starlette's ETag `"md5(str(st_mtime)-size)"`, `Last-Modified`, 304 on a matching
   `If-None-Match` (weak or strong); Content-Type from Python's mimetypes (built-in table + `/etc/mime.types`).

Coverage per route: `node backend/parity/routes-coverage.mjs --results <run.json>` (§17.2).

The route roles are also enforced structurally from each route's declared `Access` (DV36, §A12).

---

## 7. Wire protocol (ecnode protocol v1)

Reference: `node/src/ecnode/proto.py` (of `python-final`) and `docs/FLEET-PROTOCOL.md`. Rust: `crates/ec-proto`
(complete; 17 ported tests + 875 cross-language vectors). Golden transcripts: `backend/parity/proto/golden/*.json`
(recorded with the Python driver against the Python node; machine identity normalised away, see
`backend/parity/proto/NORMALISE.md`).

**Transport.** The node speaks on stdin/stdout: a subprocess for host `local`, `ssh` through a key restricted to
the node command elsewhere. stderr is free-form diagnostics (the hub keeps the last 2048 bytes to explain a drop).
In `--stdio` mode the node keeps the original fds 0/1 for the protocol and points fd 0 at `/dev/null` and fd 1 at
stderr, so a stray print can never corrupt the stream.

**Preamble.** The node first writes `MAGIC` = `b"ecnode-proto 1\n"`. The hub skips up to `MAX_PREAMBLE` (64 KiB) of
shell noise looking for a line ending in MAGIC (the async reader accepts one line up to the stream limit,
`MAX_DATA + 4096`); the node's first frame is its `hello` (or `fatal`).

**Frames.** `u32 LENGTH (big-endian, bytes that follow) | u8 KIND | BODY`.
* `KIND_JSON = 1`: one UTF-8 JSON object, 2..=`MAX_JSON` (1 MiB) bytes. NaN/Infinity/overflow-to-inf are invalid.
* `KIND_DATA = 2`: `u32 STREAM (big-endian, a request id 1..=2^31-1) | payload` of at most `MAX_DATA` (1 MiB).
* A length outside the kind's range, an unknown kind, a truncated frame, an out-of-range stream id or a non-object
  JSON body is a protocol error: drop the link (the node exits with code 2 and says why on stderr).

**Requests.** Hub: `{"type":"req","id":N,"op":"...","args":{...}}`. Node: zero or more
`{"type":"event","id":N,"event":{...}}` and DATA frames on stream N, then exactly one
`{"type":"res","id":N,"ok":true,"result":{...}}` or `{"type":"res","id":N,"ok":false,"error":{"code","message"}}`
(message capped at 1000 characters). `{"type":"cancel","id":N}` asks the node to stop; it still answers once
(`cancelled` unless already finished). Uploads (`fs.write`, `http.request` with `body_len > 0`): DATA on stream N
then `{"type":"end","id":N}`; the res comes after `end` or `cancel`.

**Flow control.** Each stream starts with `WINDOW` = 8 MiB of credit in each direction; the receiver grants more
with `{"type":"credit","id":N,"bytes":k}` as it consumes (the hub in batches of WINDOW/4; the node in steps of at
least 1 MiB). Sending more DATA than granted is a protocol error. JSON frames are never credit-limited.

**Liveness.** Hub `{"type":"ping","id":N}` every `PING_S` = 10 s; node `{"type":"pong","id":N}`. The hub drops a
node silent for `SILENT_S` = 30 s; the node exits on stdin EOF, on `HUB_SILENCE_S` = 30 s without a hub frame, or
when a write is blocked 10 s (exit code 3). `HELLO_TIMEOUT_S` = 25. Samples every `SAMPLE_S` = 2 s, disks every 30 s,
inventory on change and at least every 30 s. At most `MAX_INFLIGHT` = 64 requests per node: more answer `busy`.

**Pushed messages** (node → hub, whitelist in `ec_proto::specs`, field order as proto.py): `hello` (proto, version,
node_id, host{hostname, arch, os, kernel, cpu_model, cpu_cores, cpu_threads, ram_total_bytes, boot_time, python},
gpu{driver, cuda, source, error}, gpus[GPU_STATIC], docker{...}, addresses[], policy{POLICY}, forced_cmd,
via_ssh), `sample` (seq, t, gpus{uuid: GPU_SAMPLE}, mem, cpu, containers{name: {mem: CG_MEM, cpu_pct}}), `disks`,
`inventory` (t, containers[CONTAINER], endpoints[ENDPOINT], scripts[SCRIPT_STATUS]), `metrics`, `models_changed`,
`log`, `fatal`, `pong`, `credit`. The hub rebuilds every one with `sanitize_push` (unknown fields dropped, numbers
finite and ≤ 1e15, strings capped in characters with NUL removed, lists/maps capped with nulls dropped; a `hello`
without `proto` is a protocol error; `hello.gpus` keep only entries with a uuid; a `sample` keeps only GPUs announced
in hello).

**Operations** (args / result / events / required / monitor-mode), exactly `ec_proto::specs::{args, results,
events, required, MONITOR_OPS, UPLOAD_OPS}`:

| op | monitor | required args | result | events |
|---|---|---|---|---|
| node.info | ✓ | – | hello fields | – |
| metrics.configure | ✓ | families | {} | – |
| docker.render | ✓ | spec | {argv} | – |
| docker.run | | spec | {id} | – |
| docker.stop | | name | {} | – |
| docker.start | | name | {} | – |
| docker.rm | | name | {} | – |
| docker.inspect | ✓ | name | {info: CONTAINER} | – |
| docker.logs | ✓ | name | {text, truncated} | – |
| docker.logs_follow | ✓ | name | {lines} | {line} |
| docker.image_present | ✓ | image | {present, id} | – |
| docker.images | ✓ | – | {images[IMAGE]} | – |
| docker.pull | | image | {id} | {line} |
| docker.external | | name, action | {} | – |
| net.ensure_internal | | – | {network} | – |
| gateway.start | | name, instance, host_port, engine_host, engine_port | {id} | – |
| http.request | ✓ (POST only to inference paths) | port, method, path | {status, bytes} | {status, headers} once, before DATA |
| fs.stat | ✓ | root, paths | {items[]} | – |
| fs.write | | root, rel, size | {size, sha256} | – (upload) |
| fs.read | ✓ | root, rel | {size} | – (DATA) |
| fs.symlink | | root, link, target | {} | – |
| fs.put_small | | root, rel, content_b64 | {} | – |
| fs.delete | | root, rel | {freed} | – |
| models.scan | ✓ | – | {generation} | – |
| models.list | ✓ | – | {generation, entries[MODEL_ENTRY], next_cursor, roots[ROOT_SCAN]} | – |
| models.config | ✓ | root, rel | {config: JSONV} | – |
| scripts.run | | name, action | {rc} | {stream, line} |

The node cleans args with `clean_args` (wrong types → null) and refuses: an unknown op (`unknown_op`, text
"this node does not know the op <repr> (ecnode protocol 1)"), a non-monitor op in monitor mode (`monitor_only`,
text naming the policy file), missing required args (`bad_args` "<op> needs a, b"), a full slot table (`busy`). Error
codes: bad_args, unknown_op, monitor_only, policy_denied, not_found, not_owned, conflict, in_use, port_in_use,
no_space, hash_mismatch, size_mismatch, io_error, docker_error, http_error, timeout, cancelled, busy, unsupported,
internal; the hub maps them to HTTP (`services/nodeerr.rs status_of`: node_unavailable/busy 503; monitor_only/
conflict/in_use/port_in_use/cancelled 409; policy_denied/not_owned 403; not_found 404; no_space 507; timeout 504;
unsupported 501; bad_args/unknown_op 500; anything else 502), detail "<host label>: <message>", extra `host_id`.

**Versioning.** `PROTO = 1`. Additive only (a new op, a new optional field, a new message type): an older peer
ignores what it does not know (the hub drops unknown pushed types; the node answers unknown ops with `unknown_op`).
Anything else bumps PROTO; peers whose PROTO differs refuse each other at hello and the hub shows the link as
`incompatible` naming the side to update.

**Cross-matrix proof.** `node backend/parity/proto/matrix.mjs`: {Python, Rust} driver × {Python, Rust} node over 9
scenarios (monitor, manage, 6 malformed-frame cases, truncated). Python driver × Python node and **Rust driver ×
Python node equal golden exactly** (Rust framing, encoding, credit, 9 MiB upload and download against the real
Python node). The Rust node passes every protocol-level step; its only remaining differences are DV5
(`hello.host.python` is null) and its release number (§A14), each listed with its reason in
`backend/parity/proto/expected-diffs.json`. `node backend/parity/proto/hub-node.mjs` runs the **Python hub's real
NodeLink against the Rust node** and diffs `/api/v1/hosts`. `node backend/parity/proto/rust-hub.mjs` runs the reverse
full stack, **the Rust hub's NodeLink against the Python node**, compared with the Python hub driving the same node
(`GET /api/v1/hosts`, `/hosts/local`, `/hardware?host=local`, the `/fleet/stream?once=true` snapshot, then
`POST /hosts/local/reconnect`); its list of expected differences (`rust_hub`) is empty. The Python side of all three
scripts is the `python-final` reference (§17).

---

## 8. Settings and files

Settings (`config.rs`, every field of `Settings` in config.py): names case-insensitive, no prefix, except `host`,
`port`, `allowed_hosts`, `trust_loopback`, `data_dir` which are read **only** as `ENGINE_CONSOLE_<NAME>` (a bare one
is ignored with one warning naming it). Sources: environment over the env file over defaults. Env file:
`$ENGINE_CONSOLE_ENV_FILE` or `~/.config/engine-console/env` (python-dotenv subset: `export`, quotes, `#` comments,
`${VAR}` / `${VAR:-default}`). List fields are JSON. Bool: 1/0/true/false/yes/no/on/off/t/f/y/n. Validators: host
not blank; `HF_CACHE_DIR` absolute, no `:` `,` CR LF NUL, not a symlink. Defaults (selection): host 127.0.0.1, port
8791, data dir `~/.local/share/engine-console`, HF cache `~/.cache/huggingface` (its target when a symlink),
hf_endpoint `https://huggingface.co`, hf_token_file `~/.config/engine-console/hf_token` (refused when group/other
bits are set), hf_allowed_hosts `huggingface.co, cdn-lfs*.huggingface.co, cas-bridge.xethub.hf.co, *.hf.co`,
docker_context `rootless`, engine_network `engine-console-engines`, gateway_image pinned nginx digest, egress proxy
off, body caps 1 MiB / chat 32 MiB / import 256 KiB, SSE 8 total / 4 per key, chat streams 16 / 4, SSE idle 300 s,
hosts file `~/.config/engine-console/hosts.json`, hub key `~/.config/engine-console/ssh/id_ed25519`,
remote_node_command `/usr/local/lib/ecnode/ecnode --stdio`, otlp `localhost:4317` on, trust_loopback **off**.

Files read: `hosts.json` (`{"hosts":[{id,label,ssh,port,ssh_config,enabled}]}`; strict, owner-only, ≤ 64 KiB, no
symlink; `services/hosts.rs`), the node policy (`--policy`, else `/etc/ecnode/policy.json`, else
`~/.config/ecnode/policy.json`; schema `docs/FLEET-PROTOCOL.md` §6, keys starting with `_` are comments; refused →
monitor mode with `error`; ecnode `policy.rs`).

---

## 9. Persisted state (Rust reads what Python wrote and writes the same)

| What | Where | Format | Mode | Area |
|---|---|---|---|---|
| SQLite DB | `<data>/console.db` (+`-wal`/`-shm`) | WAL, foreign_keys ON, synchronous NORMAL; tables of migrations 0001–0005 (`profiles`, `downloads`, `instances`, `metric_rollups`, `bench_runs`, `conversations`, `messages`, `prompts`, `arena_matches`, `arena_ratings`, `usage_events`, `audit` (UPDATE/DELETE triggers abort), `api_keys`, `settings`, `library_snapshots`, `schema_migrations`); migrations applied by file name in one transaction each, recorded with `strftime('%s','now')`; JSON columns hold Python `json.dumps` text (`", "` / `": "`, ensure_ascii) | 0600 (dir 0700) | core (the store); each table's service |
| API keys | `api_keys` | `ec_` + token_urlsafe(32); stored `sha256(secret)` hex, prefix = first 8 chars; compared in constant time | – | core |
| Bootstrap key | `<data>/bootstrap-admin.key` | the secret + `\n`, written on first start when no key exists | 0600 | core |
| Instance secrets | `<data>/secrets/<iid>.json` | `{"param": "value"}` (json.dumps); ids `[A-Za-z0-9][A-Za-z0-9_.-]{0,127}`; DB/API carry `[set]` | 0600 / dir 0700 | core (store), lifecycle (use) |
| Launcher | `<data>/console.pid` (`<pid>\n`), `console.lock` (flock), `console.log` (rotated to `.1` past 10 MiB) | – | 0600, no symlinks/FIFOs | core |
| Hub ssh key | `~/.config/engine-console/ssh/id_ed25519(.pub)` | OpenSSH ed25519, created with ssh-keygen under umask 077 | 0600 / dirs 0700 | fleet |
| Node files | `~/.local/state/ecnode/instances/<instance>/files/` | generated engine config files, mounted read-only | dirs 0700, files 0644 | node agent |
| Library snapshots | `library_snapshots` | per host: generation, roots_json, entries_json, scans_json, facts_json | – | library and data |
| Metric rollups | `metric_rollups` | (instance_id, res h/d, bucket_ts, key) sum/count/max, host_id; retention 14 d / 400 d | – | lifecycle |

No migration step: the Rust store opens a Python-written DB as it is (`schema_migrations` names match) and vice
versa (tests in `services/store.rs`; tests with a database the other implementation wrote were left to the library
and data and the lifecycle areas for their tables).

---

## 10. Metrics and tracing

`GET /metrics` (auth required): `engine_console_http_requests_total{method,route,status}` (+ `_created`),
`engine_console_http_request_seconds{method,route}` histogram (buckets .005 … 10, + `_created`),
`engine_console_bench_runs_total` (+ `_created`), `engine_console_instances{state}` (8 states),
`engine_console_host_{ram_used,ram_available,shmem,swap_used}_gib`, `engine_console_instance_host_ram_gib{instance}`,
`engine_console_host_up{host}`, `engine_console_host_gpu_used_gib{host,gpu}`,
`engine_console_instance_vram_gib{instance,host,gpu}` — names, types, help texts and label names exactly as
Python. `route` label: the matched route template **without** `/api/v1` (FastAPI keeps the router-relative path),
`/metrics`, `/{path:path}` for the SPA, the partial route for a 405, `unmatched` for guarded requests refused before
routing or the OpenAPI route, `static` for unguarded ones. Method label `OTHER` for unknown methods.

OTLP/gRPC to `OTLP_ENDPOINT` (insecure), resource `service.name=engine-console`, tracer `engine_console`, one span
per request named `HTTP <METHOD>` with `http.method`, `http.route`, `http.status_code`.

---

## 11. Subprocess argv (never a shell; same timeouts, same environment scrubbing)

Hub: local node = D8 with the environment `local_env()` (HOME, PATH, USER, LOGNAME, LANG, XDG_RUNTIME_DIR,
DOCKER_HOST, DOCKER_CONTEXT, DOCKER_CONFIG only; PATH is the fixed tool PATH, DV38); remote node = `ssh -T -e none
[-F none] -o BatchMode=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o VerifyHostKeyDNS=no -o
IdentityAgent=none -o PKCS11Provider=none -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o
ControlMaster=no -o ControlPath=none -o ForwardAgent=no -o ForwardX11=no -o ClearAllForwardings=yes -o
PermitLocalCommand=no -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o IdentityFile=<key> -o
IdentitiesOnly=yes [-p PORT] -- <dest> <remote command>` with `ssh_env()` (hosts.py `ssh_argv`; `services/hosts.rs`);
`ssh-keygen -t ed25519 -N "" -C engine-console-hub -q -f <key>` / `ssh-keygen -y -f <key>` (init-key;
`services/hosts.rs`); `xdg-open <url>`, `notify-send -- "Engine Console" <escaped message>` (`desktop.rs`).

Node (ecnode; Python: `node/src/ecnode/docker.py`, `gpus.py`, `hostinfo.py`, `scripts.py`): every child gets
`/dev/null` stdin and an environment built from scratch (HOME, XDG_RUNTIME_DIR, DOCKER_HOST, DOCKER_CONTEXT, PATH,
LANG=C.UTF-8; plus `secret_env` values for `docker run` only). PATH: Python passes the absolute entries of the
node's own PATH; the Rust node passes the fixed tool PATH (DV31, §A11). docker: `[docker] [--context C] context
show|ls`, `info --format {{json .}}`, `ps --no-trunc --format ...` (+ `-a --filter label=engine-console=1`),
`container inspect ...`, `image inspect`, `image ls -q --no-trunc`, `pull`, `logs [--tail N] [-f]`, `stop -t N`,
`start`, `rm -f`, `network inspect --format {{.Internal}}` / `network create --internal --label engine-console=1`,
`run -d --pull never --name N --network NET --network-alias A --restart no --cap-drop ALL --security-opt
no-new-privileges:true --pids-limit 4096 [--gpus device=...] --shm-size S --ipc host|private [--ulimit] --label ...
-v src:dest:ro ... -e K=V ... -e SECRET_NAME ... IMAGE ARGV...`, gateway `create --pull never --name ec-... --network
bridge --restart no --read-only --cap-drop ALL --security-opt no-new-privileges:true --pids-limit 256 --user 101:101
--tmpfs ... -p 127.0.0.1:<port>:8080 --label ... -e NGINX_CONF=... --entrypoint /bin/sh IMAGE -c ...` then `network
connect`, `start`. `nvidia-smi --query-gpu=index,uuid,name,memory.total,compute_cap,pci.bus_id,driver_version
--format=csv,noheader,nounits` (and the sample/process queries); `ip -j addr`; policy script argv from the policy
only (one optional argument from the action's allowed list).

---

## 12. Security controls (Python → Rust → test)

Hub tests are in `crates/ec-hub`, node tests in `crates/ecnode`. "Ported tests" are the Rust ports of the named
Python test files; TEST-MAP names each one.

| Control | Python | Rust | Test |
|---|---|---|---|
| Loopback bind default, host not blank | `config.py:72,143` | `config.rs Settings::defaults/validate` | `config::tests::test_validators` |
| Generic names only with ENGINE_CONSOLE_ prefix (L5) | `config.py:28-41` | `config.rs prefix_only` | `test_bare_host_and_port_are_ignored`, `test_env_file_prefix_rule_and_env_wins` |
| HF cache dir checks (docker -v injection, symlink) | `config.py:150-160` | `config.rs validate` | `test_validators` (+ ported tests) |
| Host allowlist → 421 (DNS rebinding) | `api/security.py:91-93` | `api/security.rs precheck`, `allowed_hosts` | `test_host_allowlist_returns_421`, `security::tests::test_precheck_order_and_gates` |
| Origin must equal Host on unsafe methods | `security.py:95-97` | `security.rs precheck` | `test_unsafe_methods_need_custom_header_and_same_origin` |
| CSRF header `X-Engine-Console: 1` unless bearer | `security.py:98-101` | `security.rs precheck` | same + `test_bearer_clients_are_exempt_from_custom_header` |
| Profile import content types → 415 | `security.py:102-103` | `security.rs precheck` | `test_profile_import_requires_json_or_yaml_content_type` |
| Body caps (Content-Length precheck + streamed count) → 413 | `security.py:104-106,141-149,170-172` | `security.rs precheck/cap`, `api/http.rs Ctx::read_body` | `test_body_caps` |
| Bearer auth; loopback trusted only when configured and no X-Forwarded-For | `security.py:72-83` | `security.rs principal` | ported tests (`test_security.py`, `test_audit_fixes.py`) |
| Key hash sha256, constant-time compare, revoked keys refused | `settings.py:196-202` | `services/settings.rs authenticate` | `settings::tests::test_keys_create_authenticate_revoke` |
| Viewer: only the exact mutating allowlist | `security.py:28-31,120-123`, `deps.py:51-63` | `security.rs viewer_may`, `deps.rs enforce_roles` | `security::tests::test_viewer_allowlist_is_exact` + ported tests |
| Admin-only reads (/hf/*, /audit, /keys) | routers `_: Admin` | `deps.rs require_admin` in each route | ported tests; `tests/test_route_roles.rs` (DV36) |
| Security headers on every response, `no-store` on /api/ | `security.py:33-40,151-158` | `app.rs add_security_headers` | `test_security_headers_on_every_answer` |
| Audit every mutating guarded call; chat/prompt bodies metadata only; query/body redacted; never fails the request | `security.py:187-216`, `audit.py` | `app.rs finalize`, `security.rs audit_params`, `services/audit.rs` | `security::tests::test_audit_params_never_stores_chat_text` + ported tests |
| Audit table append-only | migration 0001 triggers | same SQL | `store::tests::test_migrations_apply_once_and_audit_is_append_only` |
| Secrets never in DB/API/audit; secret files 0600, dir 0700, id regex | `secrets_store.py:23-68` | `services/secrets_store.rs` | `secrets_store::tests::test_split_and_round_trip_0600` |
| redact/scrub (L8 one rule) | `common.py:53-80` | `services/common.rs` | `common::tests::test_redact_masks_keys_scrubs_values_and_truncates` |
| Bootstrap key 0600 (fchmod over a looser file) | `settings.py:204-215` | `settings.rs ensure_bootstrap_key` | `settings::tests::test_bootstrap_key_written_once_0600` |
| HF token file refused when group/other readable; status never echoes it | `settings.py:84-110` | `settings.rs hf_token/token_status` | `settings::tests::test_token_status_refuses_open_file` |
| Image pins: pinned tag, exact allowlist (no prefix match), host-pin shapes | `settings.py:33-66,154-166` | `settings.rs update/clean_host_image_pins` | `settings::tests::test_pins_validation` |
| repo_id / revision validation ("--", ".." refused) | `domain/repo_id.py` | `domain/repo_id.rs`, `api/schemas.rs` | parity corpus 11-system/13-downloads + ported tests |
| Static files: only shell + asset dirs, no traversal/symlink escape | `main.py:116-133` | `api/statics.rs spa` | `statics::tests::test_spa_serves_only_asset_dirs` |
| SSE / chat stream caps per key and total, idle timeout | `deps.py:71-154` | `api/deps.rs` | `common::tests::test_sse_limiter_per_key_and_total` + ported tests (`test_stream_caps.py`) |
| Validation errors never echo input (`input`/`ctx` dropped) | `api/errors.py:44-47` | `api/validate.rs Issue` | `test_validation_vectors_match_pydantic` |
| Unhandled errors → generic 500, no traceback | `api/errors.py:49-52` | `app.rs` catch_unwind + `http.rs fail_response` | ported tests |
| Egress: https only, host allowlist on every hop, no ambient proxy env, fail-closed proxy/CA | `hf_http.py:26-128` | `services/hf_http.rs` | ported tests (`test_security.py` egress rows); parity mode `hf` (§17.2) |
| hosts.json owner-only plain file, strict schema | `hosts.py:149-189` | `services/hosts.rs` | `tests/test_hosts.rs` (port of `tests/fleet/test_hosts.py`) |
| ssh argv: only the hub key, no agent, strict host keys, `--` | `hosts.py:198-249` | `services/hosts.rs ssh_argv` | `tests/test_hosts.rs` |
| Node env scrubbing for the local node | `hosts.py:252-275` | `services/hosts.rs local_env` | `tests/test_hosts.rs` |
| authorized_keys line: restrict, from=, command= | `hosts.py:288-319` | `services/hosts.rs key_line` | `tests/test_hosts.rs`, `tests/test_cli.rs`, `tests/test_hosts_cli_golden.rs` |
| Launcher: loopback-only URL, listeners must be ours, pid argv check, private files (O_NOFOLLOW, uid, 0600) | `desktop.py:57-160,273-298` | `desktop.rs` | `desktop::tests::*` + ported tests (`test_desktop.py`) + `test_open_then_stop_end_to_end_with_a_fake_xdg_open` (the real binary: background start, fake `xdg-open`/`notify-send` first on PATH, `stop`, modes 0700/0600) |
| Node: monitor mode refuses every non-monitor op | `server.py:431` | `ecnode/src/server.rs` | `ecnode test_server.rs`, proto matrix (monitor scenario) |
| Node: arguments only through clean_args, required args | `server.py:436-439` | `server.rs` | `test_server.rs`, matrix |
| Node: upload beyond granted credit refused | `server.py:266-276` | `server.rs Request::add_data` | `test_server.rs` |
| Node: MAX_INFLIGHT busy, duplicate ids ignored | `server.py:443-455` | `server.rs` | `test_busy_when_every_slot_is_taken`, `test_duplicate_in_flight_id_is_ignored_with_a_warning` |
| Node: writer stall / hub silence watchdog | `server.py:488-500` | `server.rs` | `test_server.rs` |
| Node: stdio fd isolation | `__main__.py:22-31` | `ecnode/src/entry.rs` | `test_stdio_child_process_protects_its_stream` |
| Node: policy file checks (regular, not group/world writable, owner, /etc root-owned, parent dir) → monitor | `policy.py:394-479` | `policy.rs` | `ecnode tests/test_policy.rs` (port of `test_policy.py`) |
| Node: model root sanity (forbidden roots, no `..`) | `policy.py:56-60,241-250` | `policy.rs` | `tests/test_policy.rs` |
| Node: image patterns exact / repo:* / repo@sha256:* | `policy.py:177-194` | `policy.rs` | `tests/test_policy.rs` |
| Node: path rules (relative, no `..`/control chars, realpath inside root, fd re-check via /proc/self/fd, O_NOFOLLOW) | `confine.py:27-219` | `confine.rs` | `tests/test_fsops.rs` (port of `test_fsops.py`), `tests/test_confine.rs` |
| Node: secret-looking names never read | `confine.py:48-58`, `fsops.py:224`, `util.py:300` | `confine.rs`, `fsops.rs`, `util.rs is_secret_name` | `tests/test_fsops.rs`, `tests/test_util.rs` |
| Node: fs.write partial + size/sha256 + 1 GiB free floor + flock; fs.delete pinned fds, in-use, root refusal | `fsops.py:121-439` | `fsops.rs` | `tests/test_fsops.rs` |
| Node: run spec (cap-drop ALL, no-new-privileges, pids, internal network, ro mounts, secret_env by name only, denied env names, ulimit allowlist) | `docker.py:250-512` | `docker.rs` | `tests/test_docker.rs` (port of `test_docker.py`) |
| Node: http.request ports/paths/methods, monitor POST only to inference, no redirects | `httpproxy.py:30-158` | `httpproxy.rs` | `tests/test_httpproxy.rs` (port of `test_httpproxy.py`) |
| Node: scripts argv from the policy only, one allowed argument, one run per script | `scripts.py:39-131` | `scripts.rs` | `tests/test_scripts.rs` (port of `test_scripts.py`) |
| Node: log/argv redaction | `util.py:241-300` | `util.rs scrub`, `redact_argv` | `tests/test_util.rs` |
| Hub: every node message sanitized | `proto.py sanitize_*` | `ec_proto::sanitize_*` | `test_proto.rs`, `test_vectors.rs` |

The hardening beyond Python (route roles, listener limits, fixed tool PATH, regular files only, bounded child output,
contained panics) is in DV31-DV43, §A11, §A12 and §A13.

---

## 13. Request validation (422)

Body `{"type":"urn:engine-console:validation_error","title":"validation error","status":422,"code":
"validation_error","detail":"request validation failed","errors":[{"loc":[...],"msg":"...","type":"..."}]}`.
Locations: `["body", field, ...]`, `["query", name]`, `["path", name]`; JSON that does not parse:
`["body", <char position>]` type `json_invalid` msg "JSON decode error"; a body whose content type is not JSON (or
absent): model bodies `model_attributes_type` "Input should be a valid dictionary or object to extract fields
from", `dict` bodies `dict_type`; no body for a required model: `["body"]` `missing` "Field required". Coercions and
messages are pydantic v2's lax mode (python mode), pinned by `tests/data/validation_vectors.json` (regenerate with
`backend/parity/gen/validation_vectors.py`). Unions report every member with the member name in `loc` (`str`,
`dict[str,any]`); a model validator (`give repo_id or model`) runs only when all fields passed (`value_error`,
loc `["body"]`); query parameters: the last value of a repeated name wins.

---

## 14. Deviations (deliberate; everything else must match)

Some code comments cite deviations by earlier names: D-SETTINGS-ERR is DV1, D-LOG-TZ (D-LOG) is DV2, D-JSONPOS is
DV3, D-500-HEADERS is DV30.

| # | Deviation | Reason |
|---|---|---|
| DV1 | A bad setting at start-up prints `engine-console: invalid setting <field>: <reason>` and exits 1; Python dies with a pydantic traceback that includes the offending value. | Never echo a possibly secret value; same exit code. |
| DV2 | Logs: timestamps in UTC (`+0000`), no uvicorn access lines, no `server: uvicorn` header. | No time-zone database dependency; logs/headers are not an API. |
| DV3 | `json_invalid` position derived from serde_json's line/column; equal on the tested cases (`{bad` → 1, trailing data → 13), may differ for exotic malformed JSON. | Python's JSONDecodeError positions are parser-specific. |
| DV4 | ec-proto: integers beyond 64 bits are parsed as floats (`clean_json` keeps a float where Python gives null); a lone UTF-16 surrogate in a string is refused (frame dropped); nesting deeper than 128 is refused (Python ~1000). Pinned by `test_documented_deviations`. | serde_json limits; no Python peer sends any of these. |
| DV5 | ecnode `hello.host.python` is null. | There is no Python. |
| DV6 | An ecnode on a machine with `libnvidia-ml.so.1` reports `gpu.source: "nvml"` where the deployed stdlib-only Python node reports `"nvidia-smi"` (same shapes; equals Python with the `nvml` extra). `ECNODE_NO_NVML=1` switches NVML off (§A2). | Optional NVML was a requirement of the port. |
| DV7 | `engine-console stop` only stops a console started by the same implementation's `open` (D9); the background console gets a new process group, not a new session (setsid would need `unsafe` pre-exec). | Distinctive argv per implementation; `unsafe_code = forbid`. |
| DV8 | SSE snapshot routes (`/downloads/stream`, `/bench/{id}/stream`) compute the snapshot before the response starts: an error becomes a problem response instead of a broken stream. | Clearer failure; same data on success. |
| DV9 | `ENGINE_CONSOLE_PORT` outside 0–65535 is refused at start-up (Python fails later at bind). | Same outcome, earlier. |
| DV10 | Shutdown waits at most 3 s for open responses (SSE streams never end by themselves); uvicorn waits for connections. | A stuck stream must not block exit. |
| DV11 | A NaN/Infinity float in a response would serialise as `null` (Python: 500 because `allow_nan=False`). Services must never produce one. | serde_json has no NaN. |
| DV12 | Metrics `window` accepts ASCII digits only (Python's `str.isdigit` also accepts other scripts' digits and 500s on superscripts). | Superset of sane input. |
| DV13 | Desktop probe is a minimal HTTP/1.1 client (no httpx): same 2 s total budget, 64 KiB cap, JSON shape check. | No extra dependency. |
| DV14 | `engine-console hosts` help texts are argparse's output at its default width (80 columns), byte for byte; argparse re-wraps them for `COLUMNS` or the terminal's width, the Rust texts never re-wrap. Usage and error lines are identical at any width. | Static text; `tests/test_hosts_cli_golden.rs` pins the 80-column output. |
| DV15 | A node process (the local `ecnode` or `ssh` to a remote one) is started in its own process group (`process_group(0)`), not a new session (Python: `start_new_session=True`); the link kills the whole group. | `setsid` needs `unsafe` pre-exec (`unsafe_code = forbid`); the group is what the kill needs. |
| DV16 | hosts.json that is not JSON: the message keeps Python's prefix `<path>: not valid JSON: ` but the rest is serde_json's error text (Python: the `json` module's), still cut to 120 characters. | Parser-specific text; the golden test compares up to the prefix. |
| DV17 | `py_isprintable` (host labels, node text) treats unassigned code points (Unicode Cn) other than noncharacters as printable; Python's `str.isprintable()` treats them as not printable. | No Unicode database in the build; an unassigned character in a label is harmless. |
| DV18 | The link's clock skew `skew_s` is rounded as `(x * 10).round_ties_even() / 10`; Python's `round(x, 1)` rounds the exact binary value, so the two can differ by 0.1 s on a value within an ulp of a tie. | Display value only; no exact decimal rounding in std. |
| DV19 | ecnode policy.json with a JSON syntax error: the message uses the `json` module's wording where serde can tell which error it is, but the line/column/char position is derived from serde's position, which can differ from Python's for some malformed inputs. | Parser-specific positions. |
| DV20 | ecnode without `ip`: IPv4 addresses come from /proc/net/fib_trie (local addresses) matched to the directly connected routes of /proc/net/route; Python asks the kernel per interface with the SIOCGIFADDR/SIOCGIFNETMASK ioctls (primary address only). The two can differ for secondary addresses and for an address without a connected route. IPv6 (/proc/net/if_inet6) is identical. (nix `getifaddrs` was considered and not taken: §A6.) | The ioctl needs `unsafe`; the /proc reader is testable with fixtures; only machines without iproute2 take this path. |
| DV21 | Model-scanner entries show a file name that is not UTF-8 with U+FFFD; Python would carry surrogate escapes, which the frame encoder then refuses (the message would be dropped). The file itself is always reached through its real path. | A name the hub can receive. |
| DV22 | `int()` of a float outside the 64-bit range saturates (Python grows a big integer). | No big integers; no input reaches that range in practice. |
| DV23 | Symlink resolution in ecnode's path confinement gives up after 256 levels and treats the link as unresolvable (Python raises RecursionError near 1000). | Bounded recursion; both refuse the path. |
| DV24 | `ecnode check`'s first line is `ecnode <version> (rust) · protocol 1 · node id <id>`; Python's is `ecnode <version> · protocol 1 · Python <x.y.z> · node id <id>`. Only people read this line: `node/install.sh` and the fleet installer use `check`'s exit code, and the hub never runs it (checked during the port). Follows from DV5. | There is no Python on the Rust node. |
| DV25 | An adapter result holding an integer beyond 64 bits serialises it as a JSON float (serde_json carries no bigger integer and reads one back as a float); Python emits the exact integer. (`pycompat::BigInt::to_value`) | No arbitrary-precision JSON; no real parameter reaches 2^63. |
| DV26 | The numeric tower's integer modulo with an operand beyond 128 bits raises `OverflowError` (the generic 500 on an HTTP path) where Python computes it. | Only products of absurd JSON integers get there. |
| DV27 | The adapters' Python `re` patterns are translated to Rust regexes (`pycompat::compile`: Python's `\d`, `\s`, `\w`, IGNORECASE with dotted/dotless i); look-arounds and a few escapes are not supported and raise `re.error`. The patterns are adapter constants, all compiled by the tests, so no request reaches this. | The regex crate has no look-around. |
| DV28 | The adapters' `json.loads` stops at a fixed nesting of 9998 levels (Python 3.13's measured top-of-stack limit) with RecursionError; Python inside a deep server stack gives up slightly earlier. | Python's limit depends on the stack it is called from. |
| DV29 | `pycompat::sum` requires a non-empty list (Python returns the int 0 for an empty one); no caller passes an empty list. | Float-only accumulator. |
| DV30 | **D-500-HEADERS.** Python's catch-all 500 (Starlette's ServerErrorMiddleware, outside the security-headers middleware) carries no security headers; the Rust port's carries them (`cache-control: no-store`, CSP, `permissions-policy`, `referrer-policy`, `x-content-type-options`, `x-frame-options`). Status and body are equal. The parity harness accepts exactly this on the listed steps only (`backend/parity/allowed-diffs.json`). The same decision as in the two sibling ports (§A13). | Stricter; never weaken a response's headers to match. |
| DV31 | ecnode runs docker, nvidia-smi, ip, uname, date and any program given by bare name only from the fixed PATH `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin` (or `ECNODE_TOOL_PATH`, absolute entries, honoured by debug builds only: tests, harnesses, development; a release node never reads it, §A15); it resolves the program to an absolute path before the exec, and every child, policy scripts included, gets that PATH. Python searches the node's own PATH. A machine whose docker CLI lives elsewhere (e.g. a rootless install in ~/bin) sets an absolute `docker.binary` in its policy (§18). | The caller's PATH (ssh session, launcher, hub) must not choose which binary the node runs. |
| DV32 | A short ecnode command keeps at most 32 MiB of stdout and 1 MiB of stderr and drains the rest while the child runs (`CmdResult.truncated`); a truncated `docker container inspect` is refused (DOCKER_ERROR) instead of parsed. Python keeps all output. | Engines influence inspect output; memory stays bounded and a child never blocks on a full pipe. |
| DV33 | ecnode readers accept regular files only: /proc, /sys and /etc reads, GGUF headers and model metadata treat a FIFO, socket, device or directory as unreadable (Python blocks on a FIFO, reads a device). `fs.read` refuses one before opening it (same message as Python's later refusal); `fs.write`, `fs.put_small` and `fs.symlink` refuse to replace one, and `fs.write` refuses a special partial file, all as CONFLICT. | A FIFO must never hang the node. |
| DV34 | An ecnode panic prints one redacted stderr line (the hook) plus, in loops, `<what>: panic: <redacted>`; a panic while handling one hub frame costs that frame (Python: an exception in `_on_message` kills the reader thread and the node ends at the hub-silence watchdog); a writer-thread panic ends the session. | One bad frame or round must not take the node down. |
| DV35 | ecnode's secret patterns treat a fancy-regex match-time error (backtrack limit), or a pattern that failed to compile, as a match (redact). Python's `re` has no backtrack limit. | Fail closed. |
| DV36 | Route roles are checked in the request pipeline (after authentication, before the handler) from each route's declared `Access` (open / caller with the viewer rules / admin), pinned to the live Python app by `backend/parity/gen/route_roles_python.py` (`tests/data/route_roles.json`, `tests/test_route_roles.rs`). FastAPI decodes a JSON body before it runs dependencies, so in theory a `json_invalid` could precede a 403; no reachable request shows it (the middleware already refuses non-admin mutating calls off the viewer list). | Structural enforcement: a handler cannot forget its role. |
| DV37 | The hub's own HTTP listener (`server.rs`): header read 10 s, keep-alive idle 5 s (uvicorn's value), request body 60 s in total (then 408), at most 256 connections (one more gets an immediate 503 and is closed, never queued), shutdown waits at most `SHUTDOWN_GRACE`. HTTP/1.1 without exactly one Host is h11's 400 "Invalid HTTP request received." (with the security headers). A body cut short is FastAPI's 400 "There was an error parsing the body" and never reaches a handler (uvicorn drops the reply of a half-closed client). uvicorn has no head timeout, body deadline or connection cap. | Slow or abusive clients cannot exhaust the hub. |
| DV38 | The hub resolves ssh, ssh-keygen, xdg-open, notify-send and a bare node command on the fixed tool PATH `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`, and its children (ssh, the local node, xdg-open, notify-send) get that PATH, never the caller's. Override: `ENGINE_CONSOLE_TOOL_PATH` (process environment only, absolute entries), compiled into debug builds only (tests, harnesses, development); a release build never reads it, so the fixed tool PATH always holds there (§A15). Python inherits PATH. | The caller's PATH must not choose which binary the hub runs. |
| DV39 | Hub-owned files must be regular files and are size-capped: env file 1 MiB (else ignored), HF token 64 KiB, secrets 1 MiB, frontend files 32 MiB (else 404), mime.types 4 MiB with 4 KiB lines, learn files 4 MiB, `/proc/net/tcp*` 64 MiB (else "not ours"). A symlinked secrets file or bootstrap key is refused (Python follows it); a non-regular file at the database path or its `-wal`/`-shm` fails start-up and a non-regular hub key `.pub` is "cannot read" (Python would hang on a FIFO). | A FIFO or a huge file must never hang or exhaust the hub. |
| DV40 | ssh-keygen output is capped at 64 KiB per stream while it is read. | Bounded memory. |
| DV41 | A panic in a background loop (supervisor and metrics rounds, pushed metric messages, fleet watcher, link pinger and watchdog, script job steps, library events and fetches, each connection) is logged redacted and the loop goes on; the process hook prints the thread, location and masked message without a backtrace; a panicking script step fails its job with `[internal] panic`; a panicking download or bench run fails its row ("panic: <redacted>"), as Python's `except Exception` around `_run` does. Every std lock recovers from poisoning (`crate::sync`); the store rolls back a transaction a panic left open. | One bug must not stop a loop or lock every caller out. |
| DV42 | Redaction fails closed: a pattern that cannot compile redacts everything. | Fail closed. |
| DV43 | Outside a checkout, the default frontend dir is `frontend/` next to the binary (derived from the running executable; `ENGINE_CONSOLE_FRONTEND_DIR` / `frontend_dir` still win); no compile-time path is used at run time. | Runtime paths from the install, not the build machine. |
| DV44 | With no policy file or a refused one, the node probes no discovery port (Python probes 8000, 8080, 11434 and 30000), guesses no port of a host-network container, and `http.request` reaches only ports a running container publishes on a loopback address, also against an inventory built while the policy was still valid; `hello` reports `discovery_ports: []`. A valid policy without `discovery_ports` keeps Python's defaults. | A node nobody configured must have no side effects on the machine's loopback services, and a port that merely answers must never become a forwarding target (§A15 F1). |
| DV45 | `policy.parse` refuses a model root (read-only or writable) that is, contains or lies inside `/etc/ecnode`, the policy file's directory, `~/.ssh`, `~/.config/ecnode` or `~/.local/state/ecnode`, or that contains the home directory (compared as written and by realpath). `fs.write`, `fs.put_small` and `fs.symlink` refuse any path segment that is a secret-looking name (`is_secret_name`, as reads already did) or one of `WRITE_PROTECTED_NAMES` (shell and session start-up files, `.pgpass`, `.my.cnf`, `.mylogin.cnf`, `.gitconfig`, `.git`, `.netrc`, `.git-credentials`, `authorized_keys`), and refuse a directory whose real location below the root has such a segment, before anything is created in it. Python checked only exact forbidden roots and the home directory itself, and names on reads only. | A hub with a writable root must not be able to rewrite the node's policy, keys or login scripts, which would be code execution as the login user (§A15 F2). A model file with a secret-looking name (e.g. `tokens.txt`) can no longer be written either; reads already refused it. |
| DV46 | A bare discovery port becomes an endpoint only with a real `/v1/models` listing (a JSON object with a `data` list, or 401/403 while `/metrics` names an engine) plus `/health` or `/metrics` answering, or an engine hint; Python also accepted a port whose `/metrics` merely contained an engine marker. The hint still names the engine; a native Ollama (listing only) still counts. | docs/SECURITY.md and docs/FLEET-PROTOCOL.md promise the listing; a `/metrics` substring alone made any page an `http.request` target (§A15 F3). |
| DV47 | confine's re-check of a directory fd and `open_inside`'s re-check of an opened file refuse when `readlink /proc/self/fd/N` fails; Python let both pass. | Fail closed, as `fs.delete` already did (§A15 F4). |
| DV48 | The node refuses (`bad_args`) a request whose `rel`, `link`, `target`, `name`, `image` or `argv` (anywhere in the args, e.g. `spec.mounts[i].rel`, `spec.argv[i]`) is longer than its protocol cap, has more items than its cap, or holds a NUL; Python cut it (and dropped NULs) and acted on the result. `ec_proto::clean` is unchanged (hub-side cleaning of pushed data, the 875 parity vectors). | Acting on a silently cut path, container name, image or argv targets something the hub did not name, and the node's own length checks could never fire (§A15 F5). Neither hub sends such values except a Hugging Face file path of more than about 860 characters in a download, which now fails with `bad_args` instead of being written to a cut path. |
| DV49 | `api_keys.last_used_at` is written on a key's first use, then at most once every 60 s per key (and at once when the recorded value lies ahead of the clock); Python writes it on every authenticated request, so `GET /keys` can show a last use up to 60 s old. | No database write per API call on the request path (§A15 F7). |
| DV50 | A streamed chat reply is written on the blocking pool: at the stream's end the response ends only after the write (as Python's `finally`); a stream dropped mid-reply hands the partial reply to a writer instead of writing it while being dropped, and the end of the lifespan waits for replies still being written. Python writes in the generator's `finally` on the event loop. | No SQLite work inside `Drop` on an async worker; no reply is lost (§A15 F7). |

---

## 15. Python behaviour kept although it looks wrong

Changing one of these makes it a deviation, which belongs in §14.

* `PUT /settings {"default_gpu_ids": [true]}` is accepted and stored as GPU 1 (pydantic lax bool→int).
* `HEAD /api/v1/health` and `HEAD /` answer 405, while `HEAD /api/v1/openapi.json` answers 200.
* `GET /api//v1/health` answers 401 (the double slash takes it off the open-path list): more restrictive, harmless.
* `GET /api/v1/hosts/BAD` answers 404 `unknown_host` (no pattern check) while the scripts routes answer 422.
* `parse_window("²m")` passes `isdigit()` and then fails `int()`: 500 in Python (the Rust hub refuses it earlier,
  DV12).

---

## 16. aarch64 (DGX Spark) build

Decision: **glibc targets, native arm64 CI builds.**
* `x86_64-unknown-linux-gnu`: built natively (the development machine; CI on `ubuntu-latest` and, for releases,
  `ubuntu-22.04`).
* `aarch64-unknown-linux-gnu`: `rustup target add aarch64-unknown-linux-gnu` works without sudo and
  `cargo check -p ecnode -p ec-proto --target aarch64-unknown-linux-gnu` passes (checked early in the port,
  nvml-wrapper included). Linking needs an aarch64 linker + glibc sysroot, which the development machine does not
  have (no sudo/apt; zig is not installed and the port's network rule did not allow fetching it), so the arm64
  binary is built **natively on GitHub's `ubuntu-22.04-arm` runner** (the repository is public). CI's `node-arm64`
  job runs the ecnode and ec-proto tests there and makes a release build with remapped paths (it checks that no
  build-machine path remains, runs `ecnode version` and prints `ldd`). The `release` job, on `v*` tags only, builds
  ecnode on `ubuntu-22.04` and `ubuntu-22.04-arm` (glibc 2.35 floor: runs on Ubuntu 22.04/24.04 and DGX OS 7),
  refuses a tag that differs from the workspace version, and attaches `ecnode-<version>-<arch>` + `.sha256`
  (sha256sum's format) to a draft release of the tag with the runner's GitHub CLI. **These CI jobs have not run yet:
  nothing has been pushed.**
* Not musl: a static musl binary cannot `dlopen` the NVIDIA driver's `libnvidia-ml.so.1` (glibc); NVML would be
  silently unavailable. The nvidia-smi fallback still works there, so musl stays a fallback option if a glibc floor
  ever bites.
* Release binaries are built with `make release`: `--remap-path-prefix` for the home directory, the cargo home and
  the workspace (Cargo's `trim-paths` is not stable in 1.99.1; a plain release build of ecnode embeds ~95 absolute
  `~/.cargo/registry/src/...` paths), and the target refuses a binary that still names a build-machine path. CI's
  arm64 and release builds use the same flags. `make node-dist` writes `backend/target/dist/ecnode-<version>-<arch>`
  + `.sha256` for this machine's architecture: the files `node/install.sh --binary` takes (§18).
* Runtime dependencies of `ecnode`: glibc only (`ldd` will show libc, libm, libgcc_s, ld-linux); NVML optional
  (loaded with libloading when present, else nvidia-smi as in Python).

---

## 17. Testing, harness, gates

**The Python reference.** Since the cutover the Python code exists only in git history.
`backend/parity/lib/python-release.mjs` exports `backend/` and `node/` of the tag `python-final` (else commit 15ac7a6,
the last commit with the Python code) with `git archive` (no worktree) into `backend/target/python-release/<commit>/`
and runs `uv sync --frozen` there (Python 3.13 for the hub, 3.12 for the node: the oldest interpreter a managed machine
runs); uv's cache is used, or the network the first time, and nothing outside `backend/target` is written.
`python@<ref>` (run.mjs) or `EC_PYTHON_REF=<ref>` (every harness script) picks another ref; an absolute path to a
prepared tree (`backend/` and `node/`, each with its `.venv`) is used as it is. The differential harness, the three
protocol scripts and the golden generators (`backend/parity/gen/*.py`, through `gen/pyref.py`; `EC_PYTHON_TREE`
names a tree explicitly) all use it.

### 17.1 Gates
Every cargo gate runs against the committed lockfile (`--locked`). Makefile targets: `make lint test test-map deny
audit crate-age parity proto frontend-test e2e` and `make release` (§16); `make all` = lint test test-map deny audit
frontend-test e2e (crate-age, parity and proto run on their own: they need crates.io, or uv's cache or the network
for the Python reference). The commands behind them:
`cargo fmt --all --check` · `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings` ·
`cargo test --locked --workspace --all-features` · `cargo deny --locked check` · `cargo audit` ·
`python3 -I backend/tools/crate-age.py` (dependency cooldown, §A3; needs crates.io, cached in `backend/target/`) ·
`node backend/parity/check-test-map.mjs --cargo` (every mapped Rust test carries a test attribute and is compiled
into a test binary: listed by `cargo test --locked -- --list`) · `node --test frontend/tests/*.test.mjs` ·
`node backend/parity/run.mjs` (`python-final` vs Rust) and `node backend/parity/run.mjs --b python` (Python vs
Python: 0 diffs) · `node backend/parity/routes-coverage.mjs --results <json>` · `node backend/parity/proto/matrix.mjs` ·
`node backend/parity/proto/hub-node.mjs` · `node backend/parity/proto/rust-hub.mjs` (the Rust hub's NodeLink driving
the Python node; diffs only from `expected-diffs.json` "rust_hub", which is empty) · `node frontend/e2e/run.mjs`
(`make e2e` = `scripts/e2e-gate.sh run`, §17.5).

CI (`.github/workflows/ci.yml`) jobs: `rust` (fmt, clippy, tests, the test map with `--cargo`), `supply-chain`
(cargo-deny, cargo-audit, crate-age), `node-arm64` (§16), `frontend`, `e2e`, `parity` (`python-final` vs Rust:
`run.mjs` and the three proto scripts) and `release` (§16). These jobs have not run yet: nothing has been pushed.

The hosts CLI goldens (`tests/data/hosts_cli_golden.json`) are regenerated with
`python3 -I backend/parity/gen/hosts_cli_golden.py` (it runs the reference's Python CLI) and must come out
unchanged.

### 17.2 Differential HTTP harness (`backend/parity/`, Node 22, zero deps)
`node backend/parity/run.mjs` runs the same corpus against two hubs: by default A = the Python reference (`python`)
and B = the Rust hub (`backend/target/debug/engine-console`); `--b python` is the self-test (Python vs Python must
give 0 diffs, which proves the normalisation removes only volatile values). `--a`/`--b` take `python`,
`python@<ref>` or `rust`; `--node python|rust|default` picks the local node both hubs run in fleet mode (default: the
reference's Python node). Both backends get identical hermetic setups (temp HOME/data/HF cache, ephemeral ports, OTLP
off, HF at a closed port; fleet mode = a real local node in monitor mode with fake docker/nvidia-smi/ip on PATH) in
three modes: `nofleet`, `fleet`, and `hf` (the fleet setup plus a fake Hugging Face behind a loopback CONNECT egress
proxy). The corpus covers all 81 routes: 477 steps over 13 corpus files at the start of the port, 674 steps over the
three modes once every area was ported (§A10; the corpus now has 20 files). Compared: status, chosen headers and
normalised JSON values (int vs float kept); normalisation rules N1-N12 in `corpus/NORMALISE.md` (N11 §A10, N12 §A14).
A Rust 501 `not ported` is classified "unported", not a diff; a difference listed in `allowed-diffs.json` (DV30) is
"allowed" on its listed steps only. Each area added corpus steps for its own scenarios (the `hf` mode came with the
library and data area).

### 17.3 Protocol cross-matrix: §7 (`make proto` runs the three scripts).

### 17.4 TEST-MAP
`python3 -I backend/parity/gen/test_map.py` writes `backend/TEST-MAP.md` from `pytest --collect-only` of the
`python-final` tree's two suites (`backend/`, `node/`; through `gen/pyref.py`) and merges `backend/parity/test-map.json`
(`{pytest id: {owner, rust: [crate::fn] | "N/A: reason", note}}`, where `owner` holds the row's area; TEST-MAP's column
is "area"). The sidecar is merged, never rebuilt: a row's `rust` value survives every rerun. `check-test-map.mjs`
verifies that every named Rust test exists (with `--cargo`: and is compiled and listed by cargo) and counts mapped /
N/A / unmapped rows per area (`--owner <area>` for one area). Rust tests mirror Python names. An area was not done
while one of its rows was unmapped; none is (2484 mapped, 3 N/A).

### 17.5 Browser end-to-end run
`frontend/e2e/run.mjs` runs only the Rust hub and the Rust node (the debug binaries
`backend/target/debug/{engine-console,ecnode}`; `scripts/e2e-gate.sh run` builds them first). `E2E_BACKEND` and
`E2E_NODE` are gone (a value other than `rust`/`default` stops the run with an error naming the removal). The run is
hermetic by default (§A2, §A11, §A12): `E2E_LIVE=1` is still the only opt-in to the caller's PATH and the hub's own
default local node (the ecnode next to engine-console, which then reads this machine's real GPUs, read-only).

---

## 18. Consumers outside the repo (cutover checklist)

The record of what the cutover had to preserve; what was done is at the end of this section.

Known consumers outside the repository (no API client, Prometheus scrape, Grafana provisioning or systemd unit was
found using the console):
1. **Desktop launchers** written by `make install-desktop[-icon]`: their `TryExec`/`Exec` hard-code
   `<checkout>/backend/.venv/bin/engine-console open|stop` (a dock may pin the file id `engine-console.desktop`).
   Cutover: the make target must point at the Rust binary (keep the file name); stop the Python console first (DV7).
2. **A fleet installer script** (kept outside this repository) runs three steps: (a) `uv run engine-console
   hosts init-key --from <addresses>` in `backend/`, keeping the first stdout line that starts with `restrict,`;
   (b) ships `node/` with `git archive` and runs `sudo bash node/install.sh` on each machine (before 0.5.0 that
   installed the Python package and a Python launcher; the machine needed only its system Python 3.10+); (c) `uv run
   engine-console hosts check`. What the Rust hub keeps, and the cutover:
   * **`hosts init-key --from ADDRS [--key-file PATH]` stdout is byte-identical to Python's** (`cli.rs`; the key
     and the line come from `services::hosts::init_key`): `created <key> (private half 0600, never printed)` or
     `using the existing key <key>`; an empty line; `Add this ONE line to authorized_keys of the login user on every
     machine the console should manage` / `(the user that machine's node was installed for):`; an empty line; the line
     `restrict,from="<ADDRS>",command="<remote_node_command>" <key type> <base64> <comment>` (default command
     `/usr/local/lib/ecnode/ecnode --stdio`); with `--from '*'` (or `*.*.*.*`) an empty line and the WARNING sentence;
     an empty line and `Then list the machines in <hosts file> and run: engine-console hosts check`. Exactly one stdout
     line starts with `restrict,`. Exit 0; a refused key file or argument: `engine-console: <reason>` on stderr, exit
     1; a usage error: argparse's text, exit 2. **An existing key is reused, never regenerated**, so every machine's
     authorized line stays valid. A golden test proves it (`tests/test_hosts_cli_golden.rs`, §A4): the same temp key
     file, Python's and Rust's stdout compared byte for byte (generator under `backend/parity/gen/`), for a new and for
     an existing key.
   * **Hub cutover:** the script calls the Rust binary (`make release`, then
     `<checkout>/backend/target/release/engine-console`) instead of `uv run engine-console`; `hosts init-key|check|list`
     keep their argv, stdout, stderr and exit codes, so the program path is the only edit. Both hubs read the same
     `hosts.json` and key (§9), so steps (a) and (c) may run with either.
   * **Node cutover (`node/install.sh`):** a new option `--binary <file>` installs a prebuilt `ecnode` instead of the
     Python package: verify `<file>` against `<file>.sha256` (the release assets of §16), refuse a file that is not an
     ELF executable for this machine's `uname -m` (`x86_64` or `aarch64`), stage it as `ecnode` (root:root, 0755) in
     the same `mktemp` directory and swap the whole `/usr/local/lib/ecnode` prefix in with the same rename (so the old
     Python `lib/` copy disappears with the old prefix), then the unchanged steps: `ecnode version` must run (prints
     `ecnode <version>`, which install.sh echoes), the policy is written once with `ecnode policy-template` as the
     invoking user and never overwritten, and `ecnode check` runs as that user. No Python, toolchain or network is
     needed on the machine. The fleet script then ships `ecnode-<version>-<arch>` and its `.sha256` next to `node/`
     (`uname -m` on the target picks the arch) and passes `--binary`.
   * **Docker outside the fixed PATH (DV31):** before switching a machine to the Rust node, run `command -v docker` as
     the node's user; if the answer is outside `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`, set
     `"docker": {"binary": "<absolute path>"}` in its policy. nvidia-smi and ip have no override and must be in that
     PATH (they are on Ubuntu and DGX OS).
3. **The forced command on every managed machine** stays `restrict,from="...",command="/usr/local/lib/ecnode/ecnode
   --stdio"`: the binary is installed at exactly that path, so no authorized line changes and nothing is re-keyed.
   The binary keeps `--stdio`, `--policy`, `check`, `policy-template`, `version`, argparse-identical usage and errors
   (exit 2) and protocol 1. A hub of either implementation drives a node of either (§7 cross-matrix), so a fleet can
   be switched one machine at a time and rolled back the same way.
4. **State**: `~/.config/engine-console/{env,hosts.json,ssh/}`, `~/.local/share/engine-console/{console.db,
   bootstrap-admin.key}` (a browser may have the bootstrap key saved), `/etc/ecnode/policy.json` (`_comment` keys
   ignored) — all read as they are (§9).
5. Docs that say `uv run engine-console` go stale but break nothing.

**Done at the cutover (0.5.0):**
* `node/install.sh` is binary-only, with `--binary <file>` implemented as specified above: sha256 against
  `<file>.sha256` (only its first field is used, so the `.sha256` file cannot point at another file); a 64-bit
  little-endian ELF executable (or PIE) whose machine matches `uname -m`; the checks run on the copy in the staging
  directory, so they and the install see the same bytes; the whole prefix is swapped in with one rename (the old one
  is put back if that fails); `ecnode version` must run; the policy is written once from `ecnode policy-template`;
  `ecnode check` runs last. `--uninstall` removes the prefix and keeps `/etc/ecnode`. Its sandboxed test is
  `crates/ecnode/tests/test_install_sh.rs` (§A14).
* The Python install path was removed from this checkout: `node/install.sh` without `--binary` or `--uninstall`
  refuses to run and names the rollback. The per-machine rollback is `node/install.sh` of the tag `python-final` (or
  `v0.4.0`), which installs the Python node at the same path, so the forced command stays valid either way.
* `make install-desktop` points the launcher at `backend/target/release/engine-console` (same file name
  `engine-console.desktop`; it needs `make release` first).
* The fleet installer outside the repository needs only its program path changed (`<checkout>/backend/target/release/
  engine-console` instead of `uv run engine-console`) and ships the binary + `.sha256` with `--binary`.

## 19. Risks

* Open: Python-only semantics that are hard to mirror exactly: pydantic messages for types not yet in the vectors,
  Python `json` decode positions, `str.isprintable`/`repr` edge cases in node messages (ec-proto `pyfmt.rs` covers the
  used ones), Python float formatting in free text (`{:.1f}` ties).
* Open: some literal value sets are copied into tests instead of read from code (Verdict, HostRamFit.verdict,
  Confidence, DownloadState, Management, Download.kind in `tests/test_learn_content.rs`); they could become consts in
  `domain/`.
* Resolved: the fake fleet in `tests/common/kit.rs` was a skeleton until `fakefleet.py`'s op behaviour was ported
  (until then, tests that needed fleet ops wrote op handlers with `FakeNode::on`); the kit now includes the full
  `fakefleet.rs` and the skeleton fakes are gone (§A4).
* Open: timing-sensitive behaviour (SSE idle timeouts, link backoff, watchdogs) is validated by tests, not by the
  harness.
* Open: `cargo deny`/`cargo audit` and `backend/tools/crate-age.py` need network (advisory DB, crates.io), and the
  parity and proto runs need uv's cache or the network for the Python reference. A security fix younger than 14 days
  enters only through `backend/tools/crate-age-exceptions.toml` (§A3).
* Open: the Rust hub and node have not yet run against real GPUs, real Docker launches, real remote machines or an
  aarch64 machine. Every test, the harness and the default e2e run use fakes (§A2); on the development machine the
  real Docker and GPU paths of the node could at most be checked read-only (`nvidia-smi`, `docker ps`).
* Open: the CI jobs, including the native arm64 build and the release job, have not run yet (§16, §17.1).

---

## 20. Addenda

### §A1 The engine adapters, in two parts
The engine adapters were ported in two parts, which are the two adapter areas of §4:

| Area | Scope |
|---|---|
| `adapters: vLLM, SGLang` | `adapters/{vllm*,sglang*}.rs` and every adapter test that is not in the other part, including tests that cover all engines |
| `adapters: TensorRT-LLM, llama.cpp, Ollama` | `adapters/{trtllm*,llamacpp*,ollama*}.rs`; every `tests/trtllm/` row, plus rows in shared test files whose id or parameters name trtllm, llamacpp or ollama |

`gen/test_map.py` assigns every adapter row by this rule (§A8).

### §A2 Hermetic by construction
Nothing a test, harness or e2e run starts may reach a live service, a real docker or this machine's GPUs unless the
run says so explicitly.
* **Rust ecnode: `ECNODE_NO_NVML`** (implemented in `gpus.rs`). When the variable is set to a non-empty value, the
  node does not load NVML at all and reads GPUs with `nvidia-smi` from PATH, exactly as the stdlib-only Python node
  does; `hello.host.gpu.source` is then `"nvidia-smi"`. Python ignores the variable (its equivalent is a venv without
  pynvml). A hub passes its local node only HOME, PATH, USER, LOGNAME, LANG, XDG_RUNTIME_DIR and DOCKER_*, so
  harnesses put it in the command: `NODE_COMMAND = ["/usr/bin/env", "ECNODE_NO_NVML=1", "<ecnode>", "--stdio",
  "--policy", "<policy>"]`. Every node test that starts the binary sets it (`tests/test_cli.rs` does).
* **Test kit** (`tests/common/kit.rs`): `TestEnv` settings point Hugging Face at `CLOSED_URL` (`https://127.0.0.1:9`),
  OTLP at `CLOSED_HOSTPORT`, docker at the fake `parity/fakes/bin/docker`, the hub key into the temp dir, no egress
  proxy, no node command or policy. `FAKES` and `hermetic_path()` give the PATH for any child a test starts (fakes,
  then `/usr/bin:/bin`); children start with `env_clear()`. A test that needs one of Python's real defaults sets it
  explicitly and still talks to a fake (wiremock, a fake CLI).
* **Parity harness:** backends get an environment built from scratch (no inherited variable), PATH = the fakes, HF
  on a closed port, OTLP off; the Rust node runs with `ECNODE_NO_NVML=1` (`lib/backend.mjs`, `proto/env.mjs`,
  `proto/hub-node.mjs`). `--node default` is the only live opt-in (the Python hub's own node reads real GPUs through
  its venv's pynvml, read-only).
* **e2e (`frontend/e2e/run.mjs`), by default:** environment built from scratch; PATH =
  `backend/parity/fakes/bin` then `/usr/bin:/bin` (fake docker, a fake `nvidia-smi` with two fixed GPUs, fake `ip`);
  HF at `https://127.0.0.1:9`; OTLP off and at `127.0.0.1:9`; the local node in MONITOR mode with a temp policy
  (docker binary `/nonexistent/e2e/docker`, no discovery ports, seeded temp model roots); NVML blocked (Rust node:
  `ECNODE_NO_NVML=1`; the Python node, while the run still offered it: `pynvml` made unimportable for that process).
  `E2E_LIVE=1` is the explicit opt-in to a live run: the caller's PATH and the hub's own default local node, which
  reads this machine's real GPUs (read-only; NVML when the driver has it, else nvidia-smi). **Before this rule the
  Python default** already built its environment from scratch, used a temp HOME and data dir, OTLP off, HF on a
  closed port and the monitor-mode policy with a nonexistent docker binary and no discovery ports, so it never reached
  docker, a local service port or the internet; but it inherited the caller's PATH and ran the hub's own default local
  node from the backend venv, which has pynvml: that node read the machine's real GPUs through NVML (read-only), and
  `nvidia-smi`/`ip` from the caller's PATH. That is what `E2E_LIVE=1` still does, with the Rust hub and node.
* Amended by §A11 and §A12: every test or harness launch of a Rust node or hub also sets its tool PATH variable.

### §A3 Dependency cooldown gate
`python3 -I backend/tools/crate-age.py` (gate, §17.1) checks every crates.io package of `Cargo.lock` against
crates.io's publication time: at least 14 days before now, or listed in `backend/tools/crate-age-exceptions.toml` as
the smallest release fixing a RustSec advisory (security fixes bypass the cooldown, version updates never do).
`--suggest` prints the `cargo update --precise` downgrade per too-new package. When the gate was introduced, 31 lock
entries resolved past the cooldown were moved back to the newest release of their semver class that was 14 days old
(fancy-regex to 0.19.2 in `crates/ecnode/Cargo.toml`, the rest in `Cargo.lock` only); rustls 0.23.46, rustls-webpki
0.103.16 and webpki-root-certs 1.0.10 stay as security exceptions (RUSTSEC-2026-0346..0349, -0382, -0383); tokio is at
1.53.3 (old enough, and fixes the RUSTSEC-2026-0386 unsound warning of 1.53.2). A new crate is added at its newest
release that is at least 14 days old.

### §A4 Test kit, library override, hosts CLI, adoption
* **One test environment.** `tests/common/kit.rs` includes `fakefleet.rs` as its submodule `kit::fakefleet` and
  re-exports `FakeNode`, `FakeFleet`, `FakeLibrary`, `FleetEnv`, `HelloOpts`, `make_hello`.
  `TestEnv` is `FleetEnv` (Python's `Env`) plus the kit's request helpers: `TestEnv(pub FleetEnv)` with `Deref`, so
  `env.node`, `env.fleet`, `env.library`, `env.add_host(..)`, `env.seed_local_model(..)` and the rest of `FleetEnv`
  work on a `TestEnv`, while `env.get/post/put/patch/delete(..)` return the kit's `Req` (`chunked()`, `open()`) and
  `env.admin_key()` stays. The kit's skeleton fakes are gone. **Include only the kit** (`#[path = "common/kit.rs"]
  mod kit;`) and write `use kit::fakefleet;`: a test file that also includes `common/fakefleet.rs` as its own module
  (an earlier pattern) fails clippy's `duplicate_mod` under `-D warnings` and would hold two copies of every type.
  Every test file uses the kit form.
  The hermetic defaults of §A2 live in `FleetEnv::with`, so both spellings get them.
* **`Overrides::library: Option<Arc<dyn LibraryPort>>`** (`container.rs`): the library the lifecycle sees instead of
  the real one (Python's `library=` override). The routes and `DownloadService` keep the real `LibraryService`.
  `FleetEnv` passes its `FakeLibrary` this way (no post-build `set_library`).
* **`FakeLibrary`'s info builder** is `services::hf::build_model_info` by default (§A10); `set_info_builder` replaces
  it for a test.
* **`hosts check` warnings:** the `hosts` commands install a "last resort" tracing layer (`cli.rs`): every WARN or
  ERROR event, from any target, is written to stderr as its bare message, as Python's `logging.lastResort` does when
  nothing configured logging (e.g. `link gpu-box: down: cannot reach ...: Connection refused (retry in 1 s)`).
* **hosts CLI golden test:** `tests/test_hosts_cli_golden.rs` replays 24 Python transcripts (help, usage errors,
  list, init-key new/existing/`*`/key file/bad address/remote command, check with a fake `ssh`, no hub key) with the
  Rust binary: exit code, stdout and stderr byte for byte after normalising the temp dir and the key's base64. It also
  pins the §18 contract: exactly one stdout line starting `restrict,` per successful init-key.
* **`LifecycleService::adopt`:** `pub async fn adopt(&self, host_id: Option<&str>) -> AppResult<usize>` — Python's
  `adopt(host_id=None) -> int`: re-attach to console-owned engine containers that outlived the console on one host
  (or every host of the fleet when `None`), returning how many instances were adopted; gateways are skipped, a known
  container's row moves to `loading` (running) or `stopped` (a clean exit), an unknown managed container with valid
  labels becomes a new row. The supervisor's per-host adoption (`tick`) calls it, and
  `test_adoption_ignores_gateways_and_only_adopts_engines` calls it directly.

### §A5 Python exceptions in adapters
Python adapters raise exceptions nobody in the adapter catches; FastAPI answers an HTTP caller with its generic 500,
while non-HTTP callers either catch them (lifecycle preflight `_resolve_gpus` catches ANY exception from
`memory_model` and falls back to the requested GPUs) or log and carry on (pollers, ticks, background tasks).
* `adapters/pycompat.rs` (shared by every adapter; it began as the vLLM adapter's `vllm_parse/py.rs`, §A7) defines
  `PyExc { kind, msg }`, `type PyResult<T> = Result<T, PyExc>`, `exc(kind, msg)`, `raise(kind, msg) -> !`
  (`std::panic::panic_any(PyExc)`, the one justified `#[allow(clippy::panic)]`), `catch(f) -> PyResult<T>` (a `PyExc`
  panic becomes `Err`; any other panic is resumed, so genuine bugs still surface) and `or_raise(PyResult<T>) -> T`.
* `EngineAdapter` (base.rs) has `try_validate`, `try_compatibility`, `try_memory_model`, `try_host_memory_gib`,
  `try_parse_metrics`, `try_parse_startup_log` and `try_build_launch_ctx`, each returning `PyResult<..>` with the
  default `catch(|| self.<plain method>(..))`. The vLLM and SGLang adapters use the defaults; the TensorRT-LLM,
  llama.cpp and Ollama adapters implement `try_*` natively (`Err(exc(..))`) and the plain method as
  `or_raise(self.try_x(..))` (§A8).
* **Rule:** every non-HTTP caller uses only the `try_*` forms: MetricsService poll/tick, the lifecycle supervisor and
  tick, preflight `_resolve_gpus` (`Err` -> the requested GPUs, as Python), bench, any spawned task. HTTP handlers may
  call the plain forms: a raise there becomes FastAPI's generic 500 through app.rs's `catch_unwind` (faithful).
* **Tests:** pycompat (`catch` converts a `PyExc`, resumes any other panic); lifecycle (preflight falls back when
  `memory_model` raises; a raising `parse_metrics` stops neither the metrics poller nor the supervisor tick); core (a
  raise on an HTTP path is the generic 500).
* **Safety net:** ec-hub's lib.rs refuses `panic = "abort"` at compile time (`#[cfg(panic = "abort")]
  compile_error!`); a panic hook keeps caught `PyExc` panics off stderr (debug log) and chains every other panic to
  the default hook; app.rs logs a `PyExc` payload by kind.
* Why not `try_*`-only everywhere: the vLLM and SGLang adapters' numeric helpers raise deep inside arithmetic (the
  numeric tower's TypeErrors); threading `Result` through them would rewrite finished, tested code, while `catch`
  around a synchronous, pure adapter call is sound once unwinding is guaranteed.

### §A6 Hermetic agents, `check` output, address fallback
* **The real node probes ports and queries docker.** An `Agent` built with `Agent::new` (or a node started with the
  default policy) probes the default discovery ports 8000, 8080, 11434 and 30000 on 127.0.0.1 and runs `docker` from
  PATH. Tests therefore never use `Agent::new`: `tests/test_server.rs` builds `hermetic_agent(store)`
  (`Agent::with_parts`: PATH = the parity fakes then /usr/bin:/bin, `ECNODE_NO_NVML=1`, a HOME that does not exist,
  NVML off, a prober, quiet prober and metrics GET that refuse every port without opening a socket, a docker socket
  default that points nowhere), and the node agent's own tests use `tests/common/wsd_agent.rs` (`World`, the safe
  prober).
* **A node started from a test** gets a policy with `"discovery_ports": []` and `"docker": {"binary":
  "/nonexistent/docker"}`, written with an explicit mode (0644 or 0600), and the test asserts the hello's
  `policy.error` is null: under a 002 umask (as on the development machine) a plain write is group-writable, the node
  refuses the file and falls back to the DEFAULT policy, which still says "monitor" but probes the default ports.
  Applied to `tests/test_cli.rs` (ecnode) and `tests/test_wsb_security.rs` (ec-hub, which also passes
  `ECNODE_NO_NVML=1`). The harness and e2e policies already set both and chmod them.
* **Proof:** the whole workspace test suite run under `strace -f -e trace=connect` (all 51 test binaries,
  `--no-fail-fast`): 143 connects, none to 127.0.0.1:8000/8001/8080/11434/30000 or any other local service port, none
  off loopback; the only Unix socket is the local nscd cache. Re-run this audit after any change to discovery, the
  http proxy or the test kits.
* **`ecnode check` / `policy-template` write to `main_with`'s `out`** (stdout in the binary, as Python), so a caller
  that captures `out` captures the report; `check`'s first line names the implementation (DV24).
* **IPv4 address fallback:** the alternative to the /proc/net/fib_trie reader (nix `getifaddrs`) was declined; the
  reader stays (DV20).

### §A7 Shared pycompat, exceptions implemented, log fixtures
* **`adapters/pycompat.rs`** is the vLLM adapter's `vllm_parse/py.rs`, moved (git history kept) and shared by every
  adapter; `adapters::vllm_parse::py` stays as a re-export so existing paths compile.
* **§A5 implemented:** `PyExc`, `PyResult`, `exc`, `raise` (now `panic_any(PyExc)`), `raise_exc`, `or_raise`, `catch`
  and `install_quiet_panic_hook` in pycompat; the seven `try_*` defaults on `EngineAdapter`; ec-hub refuses
  `panic = "abort"` at compile time; `main` installs the quiet hook before serving; app.rs logs a caught `PyExc` by
  kind and still answers Python's catch-all 500. Tests: `tests/test_adapter_exceptions.rs` (catch converts and
  resumes, `or_raise` round trip, `try_*` Err/Ok, a raise on an HTTP route is the 500 and the next request is served).
  The non-HTTP callers' tests (preflight fallback, poller and tick survive a raise) are lifecycle tests (§A10).
* **`*.log` fixtures were silently dropped:** the root `.gitignore` excludes `*.log` and then only re-included
  `backend/tests/**/fixtures/*.log` (the Python tests' fixtures); the vLLM adapter's `startup_v0.27.1.log` and
  `tqdm_line.log` never reached the repository (the tests passed in the working tree that had the files and failed
  once the code was merged without them). `.gitignore` now keeps `backend/crates/*/tests/data/**/*.log`; the two files
  were added (identical to the Python fixtures). Before committing fixtures, check
  `git status --ignored -- backend | grep -v target` is empty.
* **TEST-MAP:** regenerated from `test-map.json` (`gen/test_map.py`) after this change; the adapter rows were split
  into their two areas later (§A8).

### §A8 One exception type, native try_* forms, the adapter area split
* **One `PyExc` for every adapter.** The TensorRT-LLM adapter first shipped its own `PyExc`/`raise`/`catch` in
  `trtllm_parse::py`. A raise of that type is not a `pycompat::PyExc`, so `pycompat::catch` (the trait's `try_*`
  defaults, app.rs) re-raised it as a genuine bug: the first TensorRT-LLM, llama.cpp or Ollama exception in a poller or
  tick would have killed it. `trtllm_parse::py` now re-exports pycompat's exception API;
  `tests/test_adapter_exceptions.rs` pins it.
* **Native `try_*` forms.** The three adapters' inherent `try_*` methods are now the trait's `try_*` overrides (no
  catch round trip). Their inherent `try_build_launch_ctx` (Python's `build_launch`, a ValueError as `Err`) is
  `py_build_launch_ctx`; the trait's `try_build_launch_ctx` maps a ValueError to the inner 422 refusal and any other
  exception to `Err`, and `build_launch_ctx` is `or_raise` of it.
* **ecnode is a dev-dependency of ec-hub** (tests only): the test kit calls the node's real
  `ecnode::agent::filter_metrics` instead of a copy.
* **Area split applied** (§A1): `gen/test_map.py` gives every adapter row whose id names trtllm, llamacpp or ollama
  to `adapters: TensorRT-LLM, llama.cpp, Ollama` and the rest to `adapters: vLLM, SGLang`, reassigned rows included;
  the notes of the ten learn-coverage rows were refreshed. `check-test-map.mjs --owner <area>` passes for both.
* The ten tests of these three adapters that need `estimate_fit` were ignored until `services::fit::estimate_fit` was
  ported; they run on it since (§A10).

### §A9 One PyYAML emitter
* **`src/pyyaml.rs`** (now `src/pyyaml/mod.rs`, §A10) is the hub's only `yaml.safe_dump`: `Node` (Null, Bool, Int as
  canonical digits, Float, Str, Seq, insertion-ordered Map) and `safe_dump(&Node, sort_keys) -> String`. Built from
  the TensorRT-LLM adapter's line-by-line port of PyYAML 6.0.3's emitter; the lifecycle's own compose emitter
  (services/docker.rs `mod yaml`) is gone (`pub use crate::pyyaml as yaml`, `safe_dump(.., false)`); trtllm keeps
  `yaml::safe_dump(&PyVal)` as a shim (`sort_keys=True`).
* Both golden sets pass on it: the TensorRT-LLM adapter's 1300 cases (`tests/test_ws_a2_goldens.rs`) and the
  lifecycle's PyYAML vectors (`services::docker` tests). Unifying them found a bug in the adapter's port:
  `check_simple_key` must count the prepared tag (`!!str`, 5 characters), so a 123-character key is complex
  (`? key`); fixed and pinned by `pyyaml::tests` (expectations checked against PyYAML 6.0.3).
* **Profiles:** the profile export (`export_yaml`) uses `crate::pyyaml::safe_dump` (Python: `yaml.safe_dump(...,
  sort_keys=False)`); no third emitter.
* `EngineAdapter::named_volumes()` was proposed and not added: no caller needs it yet.
* Two lifecycle tests that needed the TensorRT-LLM adapter (`test_wse_shared_rows`' ulimits row,
  `test_no_stale_concepts`) run since that adapter was ported; the check-code scanner accepts an inline
  `if c { "block" } else { "warn" }` level.

### §A10 Parity complete, allowed diffs, one YAML module
* **Parity is complete:** Python vs Rust over 674 steps in the three modes (nofleet, fleet, hf): 665 same, 9 allowed
  (DV30 D-500-HEADERS), 0 diff, 0 unported. `backend/parity/allowed-diffs.json` names each accepted difference, its
  status, its headers and its steps; the harness gives a listed step the verdict `allowed` only when both sides answer
  that status, status and body are equal, and the only differences are those headers absent on Python's side. run.mjs
  prints `STALE ALLOWANCE` for a listed step that no longer needs it. Python vs Python ignores the list.
* **N11** (corpus/NORMALISE.md): live host-RAM readings formatted into sentences (`<x> GiB of host RAM`, `<x> GiB of
  <n> GiB`, `(<x> of <n> GiB)`) are `<live>`. Without it a run under load (a parallel cargo build) showed up to 33
  diffs: each hub's node reads /proc/meminfo at its own moment. Numbers in fields were already N5.
* **Profiles:** a JSON body whose `yaml` member is present but not a string raises `AttributeError` (Python's
  `yaml.safe_load(<not a str>)` fails on `.read`): the catch-all 500, as in Python.
* **Test kit:** `FakeLibrary` builds a copy's info with `services::hf::build_model_info` by default; the adapter kit's
  (`tests/common/a2_kit.rs`) `estimate_fit` is `services::fit::estimate_fit`; `test_externals`' `sse_engine` streams
  only for `stream: true` (Python's fake); the L2 row maps to
  `test_wsc_l2_library_and_lifecycle_read_each_other_without_recursing` (the variant needing a real FleetService link
  was dropped). The lifecycle's §A5 test of the preflight fallback now uses a model the library does not have
  (`model_missing`, no info): with the info known, Python's `assess_info` does not catch the memory model's exception
  either (500, now asserted too). Every test that was ignored while it waited for another area's code runs now:
  `cargo test --workspace -- --include-ignored` runs everything.
* **One YAML module:** the profiles' loader (`services/profiles/{yaml_load,pyobj}.rs`, copied from the Nemo Console
  port) moved to `src/pyyaml/{load,pyobj}.rs`; `src/pyyaml/mod.rs` is the dump; `pyyaml::safe_load` is re-exported.
  The loader's own exception value type is `pyobj::PyError` (never raised as a panic; not `pycompat::PyExc`). All
  golden sets pass (the TensorRT-LLM adapter's 1300 dump cases, the lifecycle's dump vectors, the profile goldens).

### §A11 The node's audit fixes
* ecnode: regular files only with bounded reads (`util/fsread.rs`), child output capped while it is read
  (`util/proc.rs run_capped`), tools from a fixed PATH (`ECNODE_TOOL_PATH` for tests), panics contained and reported
  redacted (`util/panics.rs`), poisoned locks recovered (`util/sync.rs`), regex errors fail closed. DV31-DV35.
* **Hermetic launches (amends §A2/§A6):** setting PATH no longer isolates a Rust node; every test or harness launch of
  the Rust node also sets `ECNODE_TOOL_PATH` to the fakes, or the node runs the system's real tools. The harness
  (`lib/backend.mjs`, `proto/env.mjs`, `proto/hub-node.mjs`), the e2e runner and `test_wsb_security.rs` do. An strace
  of every ecnode test binary and of the harness's Rust-node runs shows tools executed only from the fakes or the
  fixed PATH.
* §11: a child of the node gets the fixed tool PATH, not the node's.

### §A12 The hub's audit fixes
* New core modules: `server.rs` (the listener, DV37), `sync.rs` (poison-proof locks), `panics.rs` (one process hook
  that masks secrets and keeps adapter `PyExc` panics quiet; `guarded`/`guarded_sync` for every background step),
  `fsutil.rs` (regular files, bounded reads, private creation), `tools.rs` (the fixed tool PATH, `run_capped`).
  `main` installs `panics::install_panic_hook` (it replaces pycompat's quiet hook and keeps its PyExc rule).
* Route roles: `Router::table()` exposes every route with its `Access`; `tests/test_route_roles.rs` checks the table
  against the live Python app and asserts, for every guarded route, 401 without a key and 403 for a viewer on every
  admin route and mutating method (never 422/404/200).
* Header values are decoded as latin-1 like Starlette's (a non-ASCII Origin is refused, not ignored): a fix toward
  Python, not a deviation.
* `tests/test_hub_server.rs` boots the real listener on an ephemeral port (dispatch, bad Host 400/421, loopback trust
  from the socket address only, `X-Forwarded-For` never trusted, idle/slow/truncated/over-cap clients, a panicking
  handler is a 500 and still audited).
* **Hermetic launches (amends §A2):** every test or harness launch of the Rust hub sets `ENGINE_CONSOLE_TOOL_PATH` to
  the fakes (harness `lib/backend.mjs`, `proto/rust-hub.mjs`, the e2e runner); together with §A11 a Rust hub on a
  Rust node runs only fake tools.
* Library and data services: library events and fetches run guarded; a panicking download or bench run fails its
  row; fitting, hf and hf_http locks recover from poisoning (`tests/test_background_panics.rs`).

### §A13 The sibling ports' audit lessons, where each is answered
| Lesson | Answer | Proof |
|---|---|---|
| Structural role enforcement | Route `Access` checked in the pipeline (DV36) | `tests/test_route_roles.rs`: table = live Python app; every guarded route 401 without a key, 403 for a viewer on admin routes and mutating methods, never 422/404/200 |
| Panics in background tasks, redacting hook, poisoned mutexes | `panics.rs` / `sync.rs` (hub), `util/panics.rs` / `util/sync.rs` (node); downloads and bench fail their row (DV34, DV41) | hub `test_hub_hardening.rs`, `test_background_panics.rs`; node `test_node_hardening.rs`, `test_server.rs`, `test_cli.rs` |
| Server limits; truncated body 400; handler panic 500 + audit row | `server.rs` (DV37) | `tests/test_hub_server.rs` |
| Special files, bounded reads, line caps, copies refuse special files | `fsutil.rs` (hub, DV39), `util/fsread.rs` + fsops (node, DV33) | FIFO / symlink / oversized-line tests in both crates |
| Child output capped while reading | `tools::run_capped` (hub, DV40), `util/proc.rs run_capped` (node, DV32) | flood tests in both crates |
| No ambient PATH for tools | fixed tool PATH + `ENGINE_CONSOLE_TOOL_PATH` / `ECNODE_TOOL_PATH` (DV38, DV31); test kits resolve their own tools on the same PATH | strace of the whole suite and of a Rust-hub-on-Rust-node parity run: no lookup walks a directory under HOME, tools run only from the fakes or the fixed PATH |
| Regex errors fail closed | hub redaction (DV42), node secret patterns (DV35) | unit tests with injected failures |
| Header values decoded as latin-1 | as Starlette (a fix toward Python) | `test_headers_are_decoded_as_latin1`, real-socket case |
| Hermetic tests; `--locked` gates | every test and harness launch sets the tool PATHs and closed URLs; the Makefile targets and the CI jobs use `--locked` | the suite passes with an empty HOME and writes nothing into it; strace: 0 connects to local service or engine ports, 0 off loopback, no real `nvidia-smi` |
| Publish hygiene | no user names, home paths, worktree names, private host names or private IPs in configs, comments, goldens or tests; example LAN addresses are 203.0.113.x; the test ed25519 key spells FAKE-TEST-KEY-NOT-A-REAL-ONE-000; release binaries carry no build-machine path (`make release`) | scans before each commit; the release target refuses a binary with such a path |
| No compile-time paths at run time | the frontend dir comes from the running executable (DV43) | `test_runtime_code_has_no_build_time_paths` (node), hub statics tests |
| A test that boots the real listener | `tests/test_hub_server.rs` | dispatch, Host 400/421, loopback trust from the peer address only |
| check-test-map verifies real tests | test attribute required; `--cargo` requires each mapped test in `cargo test --locked -- --list` | 2484 mapped rows, all compiled and listed |
| Process jargon in public docs | answered at the cutover: `backend/PORTING.md` and `backend/TEST-MAP.md` were rewritten without process names; TEST-MAP rows are grouped by area (§4) | this document; TEST-MAP's `area` column |

### §A14 Cutover (0.5.0)
* **Layout.** The Cargo workspace moved from `rust/` to `backend/` (§2, D1). The Python trees were removed: `backend/`
  (`src/`, `tests/`, `pyproject.toml`, `uv.lock`) and `node/` (`src/`, `tests/`, `pyproject.toml`, `uv.lock`);
  `node/` keeps `install.sh`, `README.md` and `examples/`.
* **Embedded content.** The SQL migrations and the learn content moved into ec-hub (`crates/ec-hub/migrations/`,
  `crates/ec-hub/learn/`), still embedded at build time; the include paths were updated (D6).
* **A test that read Python source.** One test that read the TensorRT-LLM adapter's Python source at test time now
  reads a frozen golden, `crates/ec-hub/tests/data/ws_a2/python_add_codes.json` (the 32 `add("<level>", "<code>", ...)`
  codes of `trtllm.py`, in source order).
* **The Python reference** is prepared by `backend/parity/lib/python-release.mjs` (and found by the golden generators
  through `gen/pyref.py`): the tag `python-final`, else commit 15ac7a6 (§17). `run.mjs` defaults to python vs rust;
  `--b python` is the self-test.
* **Version 0.5.0.** `ec_hub::VERSION` is the workspace version (`backend/Cargo.toml`); the served OpenAPI document's
  `info.version` follows it (D5; `gen/openapi_python.py` writes it), pinned by `crates/ec-hub/tests/test_version.rs`.
* **N12** (`corpus/NORMALISE.md`): the hub's own release number in `GET /api/v1/health` `version` and in the OpenAPI
  `info.version` is compared as `<hub-version>` (the reference is 0.4.x, the Rust hub 0.5.0 and later). A node's
  `version` is not touched by N12.
* **Protocol expected diffs** (`backend/parity/proto/expected-diffs.json`): the node's release number was added for
  the Rust node (`hello.version` and node.info's `version`, scenario `"*"`; for `hub-node.mjs`, the node `version` in
  `/hosts` and `/hosts/local`); the stale `rust_hub` entries were removed (the list is empty).
* **`node/install.sh --binary`** (§18) and its sandboxed test `crates/ecnode/tests/test_install_sh.rs`: the script
  runs as shipped except for asserted edits (its paths point into a temp dir, the root-only steps are dropped, the
  parent directory must belong to the test's user), and the final `check` reads a hermetic policy, so no loopback
  port is probed; the node's tools are the parity fakes, NVML is off and HOME is a temp dir.
* **e2e** runs the Rust hub and node only (§17.5).
* `frontend/dev/dump-catalogs.py` became `frontend/dev/dump-catalogs.mjs`: it runs the debug hub on an ephemeral
  loopback port with a throwaway HOME and data dir (fleet off, OTLP off, Hugging Face at a closed port) and writes what
  `GET /engines` and `GET /engines/{id}/params` serve.
* **Makefile and CI:** the `rust-*` targets are gone (§17.1); CI jobs `rust`, `supply-chain`, `node-arm64`,
  `frontend`, `e2e`, `parity`, `release` (§16, §17.1), not run yet.
* **systemd:** the template `deploy/engine-console.service` runs the release binary
  (`backend/target/release/engine-console`).
* **Desktop launcher:** `make install-desktop` points at the release binary (§18).


### §A15 The independent audit before 0.5.0
An independent audit read the Rust tree statically (it could not run cargo) and found no way for a compromised hub to
cross the node boundary. Its findings and what was done, each fix with a test that fails without it:
* **F1 (medium), the policy fallback was not side-effect free:** DV44. `monitor_policy()` has no discovery ports; with
  a refused or missing policy only published container ports are probed and reachable
  (`tests/test_node_audit.rs` `test_audit_f1_*`; `test_check_flags_a_refused_policy` no longer expects probes).
* **F2 (medium), writable roots could reach the node user's control files:** DV45 (`policy.rs check_root_overlap`,
  `util.rs WRITE_PROTECTED_NAMES`, `confine.rs split_rel_for_write` / `check_write_dir`; `test_audit_f2_*`). Two
  parse tests now name a policy path below a directory (`/x/policy.json`), since a root inside the policy file's
  directory is refused.
* **F3 (medium), a bare port counted as an engine on a `/metrics` substring:** DV46 (`PortProbe::bare_engine`;
  `test_audit_f3_a_bare_port_needs_a_models_listing`). Strict `qualifies()` was not taken: it would drop a native
  Ollama, which answers `/v1/models` but neither `/health` nor `/metrics`. The Python golden of `build_endpoints` lost
  its one endpoint that had only a hint.
* **F4 (low):** DV47 (`confine::tests::test_audit_f4_an_fd_whose_location_cannot_be_read_is_refused`).
* **F5 (low), over-long args were cut instead of refused:** DV48. `ec_proto::overlong_args` runs in the node's request
  path right after `clean_args` (`test_server.rs`, `test_proto.rs` `test_audit_f5_*`). Both hubs bound container names
  (the `^ec-[a-z0-9][a-z0-9-]{0,58}$` pattern), argv (512 items of at most 4096 characters, no NUL) and images (the
  policy patterns); rels and names in other requests come from the node's own reports.
* **F6 (low), CI and release:** the `supply-chain`, `node-arm64` and `release` jobs (§16, §17.1), `node/install.sh
  --binary` with its checks (§18), and `overflow-checks = true` in `[profile.release]`.
* **F7 (low), blocking work on the async runtime:** every store call runs through `store::off_worker`: on a worker of
  the multi-threaded runtime it runs under tokio's `block_in_place`, so the worker's other tasks move to another
  thread while SQLite or the wait for the single connection blocks; elsewhere it runs as before (on a current-thread
  runtime, where `block_in_place` would panic, there is no other worker to free). `spawn_blocking` or a store thread
  would have made every caller async. Tests: `store::tests::test_a_slow_store_call_does_not_stall_unrelated_tasks`,
  `test_store_calls_work_on_every_kind_of_thread`. Also DV49 (`last_used_at`) and DV50 (chat replies;
  `ChatService::flush`, called by `lifespan_stop`).
* **F8 (low), deploy files and docs:** the systemd template runs the release binary and names
  `MemoryDenyWriteExecute` as a candidate (unverified); docs/OPERATIONS.md documents `docker.binary`, that `nvidia-smi`
  and `ip` have no override, and `ENGINE_CONSOLE_FRONTEND_DIR` / `frontend/` next to the binary.
* **F9 (low), hygiene:** the source headers no longer name owners or the way the port was organised; this document
  and TEST-MAP.md carry no process names.
* **F10 (info):** `ECNODE_TOOL_PATH` and `ENGINE_CONSOLE_TOOL_PATH` are honoured by debug builds only (DV31, DV38;
  `util::proc::tests::test_audit_f10_release_builds_ignore_the_tool_path_override`,
  `tools::tests::test_the_tool_path_override_exists_only_in_debug_builds`, each also run as a release-profile test).
  `services::common::random_bytes` panics when the operating system's RNG fails instead of returning zeros (Python's
  `os.urandom` raises; the panic is contained like any other: a 500, a failed row, or no start-up), tested with an
  injected failing source. The `fs.delete` in-use check sees running containers only, as in Python; documented in
  docs/SECURITY.md.

**Python bugs found by the audit** (each also in the Python release, fixed in 0.5.0 by the rows above): an
unconfigured or refused node probed 8000/8080/11434/30000 and forwarded to whatever answered (DV44); a writable root
holding `~/.config/ecnode` let a hub add a script to the user policy, because roots were checked as exact paths and
writes never checked names (DV45); a `/metrics` substring alone made a bare port an engine, contrary to the docs (DV46);
an unreadable `/proc/self/fd` link counted as inside the root (DV47); the node acted on over-long `rel`, `name`, `image`
and `argv` values cut to their caps (DV48).

**Gate results of the release tree (2026-10-28, after the audit fixes).** `cargo fmt --check` and `cargo clippy
--locked --workspace --all-targets --all-features -D warnings` clean; `cargo test --locked --workspace --all-features --
--include-ignored`: 99 test binaries, 1910 passed, 0 failed, 0 ignored. HTTP parity against python-final, three modes:
674 steps, 665 same, 9 allowed (DV30), 0 different, with the Python node and again with the Rust node behind both hubs;
Python vs Python 674 same; route coverage 81 of 81. Protocol: the matrix, the Python hub on the Rust node and the Rust
hub on the Python node all within `expected-diffs.json`. Test map: 2484 mapped (each compiled and listed by cargo), 3
N/A, 0 unmapped. Browser e2e: 34 pages and tabs. Frontend tests: 246 passed. cargo-deny clean, cargo-audit clean,
crate-age: 302 packages, none younger than 14 days except the 2 listed security fixes. `make release` refuses
build-machine paths. An strace of the whole test suite saw 278 connects, all to 127.0.0.1 (ephemeral or closed ports)
or the local nscd socket, none to an engine or service port (one timing test,
`test_incompatible_waits_are_cut_short_by_reconnect_too`, exceeds its 3 s wait under strace and passes without it).
The same gates were run again from a literal clone of the branch.
