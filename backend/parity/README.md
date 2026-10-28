# backend/parity: proving the Rust hub and node are drop-in replacements

Node 22, zero npm dependencies. The reference is the last Python release: the hub (`backend/`) and the node agent
(`node/`) of the tag `python-final` (else commit 15ac7a6, the code the port was verified against), exported with
`git archive` into `backend/target/python-release/<commit>/` and set up there once with `uv sync --frozen` (Python 3.13
for the hub, 3.12 for the node) by `lib/python-release.mjs`. `EC_PYTHON_REF=<ref>` (any commit or tag of this
checkout, or an absolute path to a prepared tree) changes the reference for every script; run.mjs also takes
`python@<ref>` as a backend or node name. Only this preparation needs uv (its cache, or the network the first time).
The route table and the corpus are committed data. Nothing here touches the live console (port 8791), Docker, a GPU or
the network: every backend runs on an ephemeral loopback port with a throwaway HOME and data dir, OTLP off, Hugging
Face pointed at a closed loopback port, and its local node (fleet mode) in monitor mode with fake `docker`,
`nvidia-smi` and `ip` (`fakes/bin/`) first on PATH.

| Tool | What it proves |
|---|---|
| `node backend/parity/run.mjs` | The Python reference vs the Rust hub (`backend/target/debug/engine-console`, build it first: `make build`) on the whole corpus in three modes. Green when every step matches, or differs only as an allowance of `allowed-diffs.json` says (headers only, status and body equal); a step a backend answers `501 "not ported: <module>"` would be listed per module. |
| `node backend/parity/run.mjs --b python` | Python vs Python: **0 diffs** proves the normalisation removes only volatile values (self-test). |
| `node backend/parity/routes-coverage.mjs` | Every route of `routes.python.json` (79) plus FastAPI's `GET`/`HEAD /api/v1/openapi.json` has at least one corpus step. With `--rust` (or `--results <run.json>`): no route Python serves is missing from the Rust server (a 404 "no such file" from the SPA fallback, or a 405, where Python answers). |
| `node backend/parity/proto/matrix.mjs`, `hub-node.mjs`, `rust-hub.mjs` | The node protocol: Python and Rust driver x Python and Rust node against recorded transcripts, the Python hub driving the Rust node, and the Rust hub driving the Python node (`proto/NORMALISE.md`). |
| `python3 -I backend/parity/gen/test_map.py [--autofill]` | Rewrites `backend/TEST-MAP.md` and merges `test-map.json`: one row per pytest node id of the reference's `backend/` and `node/` (collected with their own venvs), with its area and the Rust test(s) covering it. `--autofill` maps a row when exactly one `fn <same name>` exists in that suite's crates. |
| `node backend/parity/check-test-map.mjs [--owner <area>] [--cargo] [--allow-unmapped]` | Every Rust test named in `test-map.json` exists as a test (with `--cargo`: is compiled and listed by `cargo test --locked -- --list`); counts mapped / N/A / unmapped per area; fails on unmapped rows unless `--allow-unmapped`. |
| `backend/parity/gen/*.py` | The golden generators of the Rust tests' data files; each runs under the reference's own interpreter (see its docstring and `gen/pyref.py`), e.g. `"$(node backend/parity/lib/python-release.mjs)/backend/.venv/bin/python" -I backend/parity/gen/routes_python.py`. Re-running one must reproduce the committed file (apart from its noted volatile values). |

## run.mjs options

`--a python[@<ref>]|rust`, `--b python[@<ref>]|rust` (default: python vs rust), `--modes nofleet,fleet,hf` (default
all three), `--node python[@<ref>]|rust|default` (the local node both hubs run in fleet mode: `python` = the reference's
ecnode from its node venv, stdlib only, so it reads the fake `nvidia-smi`; `rust` = `backend/target/debug/ecnode`
started as `/usr/bin/env ECNODE_NO_NVML=1 ECNODE_TOOL_PATH=<fakes> ecnode ...`, so it reads the fake `nvidia-smi` and
runs only the fake tools too (PORTING.md §A2, §A11); `default` = each hub's own default; for the Python hub that is the
hub venv's ecnode, which reads the real GPUs through NVML, read-only, so its numbers are live), `--filter <corpus file
substring>`, `--json-out <file>` (full results, for `routes-coverage.mjs --results`), `--keep` (keep the temp dirs),
`--verbose` (every diff). Both hubs serve this checkout's `frontend/` (`FRONTEND_DIR`): the exported Python tree has
none.

Modes: `nofleet` runs both hubs with `FLEET_ENABLED=false` (every host disabled; most of the corpus). `fleet` runs a
real local node with the policy written by `lib/backend.mjs` `makeShared` (monitor mode, two seeded tiny models in
two roots, one policy script that is never run) and waits, polling both hubs in lock step, until each reports the
node up with its first sample, its disks and a scanned library. `hf` is the fleet setup plus a fake Hugging
Face hub (`lib/fakehf.mjs`): `HF_ENDPOINT=https://hf.test` (the .test TLD never resolves), `EGRESS_PROXY` a loopback
CONNECT proxy that tunnels only `hf.test` / `*.hf.test` to a local HTTPS server and answers any other target with the
lab proxy's block page, `EGRESS_CA_BUNDLE` a CA made for the run (openssl CLI, in the run's temp dir),
`REQUIRE_EGRESS_PROXY=true` and a fake `HF_TOKEN`; it serves search, model info with siblings, files (with Range),
the card, and the failures the client maps (gated, private, missing, a 500, a non-JSON answer, redirects inside and
outside the allowlist). The run prints the CONNECT targets the proxy saw.

## Corpus format (`corpus/*.json`, data only)

```json
{"modes": ["nofleet"], "about": "...", "scenarios": [{"name": "...", "steps": [
  {"method": "POST", "path": "/api/v1/keys", "query": {"limit": "5"}, "headers": {"Origin": "..."},
   "json": {"name": "x"}, "auth": "admin", "csrf": false, "capture": {"kid": "id"}, "expect": {"status": 201}},
  {"method": "DELETE", "path": "/api/v1/keys/{{kid}}"}
]}]}
```

* body: `json` (sent with `Content-Type: application/json` unless headers say otherwise), `raw` (a string),
  `raw_repeat` `{count, char, prefix?, suffix?}` (large bodies), `chunked` `{parts, size, char}` (no Content-Length).
* `auth`: `admin` (default: the backend's own bootstrap key), `viewer` (minted by `00-setup.json`), `none`, `bad`,
  `basic`. `csrf: true` adds `X-Engine-Console: 1`. `query` may be a list of pairs (repeated parameters).
* `capture` reads a value from that backend's raw answer (`a.b.0.c` in the JSON body, or `header:<name>`); `{{var}}`
  substitutes it later (also predefined: `{{port}}`, `{{admin_key}}`). Each backend keeps its own variables.
* `expect.status` is checked against backend A only: it keeps the corpus honest (a step that silently stopped
  testing what it says fails the run).
* `compare` forces the body kind (`json`, `text`, `sse`, `prom`, `none`, `length`); by default it follows the
  content type. SSE bodies are compared as `[{event, data}]` with `data` parsed as JSON.
* `route` overrides the route a step counts for in `routes-coverage.mjs` (normally the first route of the table that
  matches the path and method, like Starlette).

Compared for every step: the status, the headers `content-type`, `cache-control`, `content-security-policy`,
`x-content-type-options`, `referrer-policy`, `x-frame-options`, `permissions-policy`, `content-disposition`,
`x-accel-buffering`, `allow`, `location` (exact) and `etag`, `last-modified` (presence), and the normalised body.
The normalisation rules and their reasons: `corpus/NORMALISE.md`.

## Adding cases

Add steps to the corpus file of the router (or a new `NN-<topic>.json`); run `node backend/parity/run.mjs --b python`
(must stay 0 diffs), `node backend/parity/run.mjs` and `node backend/parity/routes-coverage.mjs`. A new volatile value gets a rule in `lib/normalise.mjs` and a row
in `corpus/NORMALISE.md`, never a per-step exception. The fakes are deterministic shell scripts; `fakes/bin/docker`
refuses every mutating verb, so a corpus step can never make a node change anything.
