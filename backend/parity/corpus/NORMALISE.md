# Normalisation rules of the differential harness

Every rule lives in `backend/parity/lib/normalise.mjs` (and the header choice in `lib/harness.mjs`). Only values that
are genuinely volatile are touched; everything else is compared exactly: status, the chosen headers, JSON values
(key order ignored, list order kept, `null` vs absent kept, integer vs float kept: Python's `1.0` never equals `1`).

One `Normaliser` exists per backend per run, so generated ids are renamed in order of first appearance: the same
request sequence yields the same placeholders on both sides, and a wrong or swapped id still shows up as a diff.

| Rule | What | Replaced by | Why it is volatile |
|---|---|---|---|
| N1 | Numbers under the keys `created_at`, `updated_at`, `ts`, `last_used_at`, `last_used`, `started_at`, `finished_at`, `last_request_at`, `voted_at`, `checked_at`, `last_seen`, `connected_since`, `scanned_at`, `history_since`, `mtime`, `atime`, `created`, `boot_time`, `revoked_at`; `t` of a metric point (`…points[i].t`) or of an SSE metrics event (`$.t`) | `"<ts>"` | wall-clock epoch seconds (or file times of the shared seeded models, which the two nodes' scans touch at different moments). A `null` stays `null`. |
| N2 | Numbers under `uptime_s`, `rtt_ms`, `skew_s`, `duration_s`, `retry_in_s`, `eta_s`, `speed_bps`, `avg_latency_ms`, `latency_ms` | `"<duration>"` | measured by the clock |
| N3 | Generated ids `<prefix>_<12 hex>` with prefix `key`, `prof`, `conv`, `prm`, `match`, `dl`, `inst`, `bench`, `job`, anywhere in a string (values, error details, audit actors) | `"<prefix:N>"`, N = order of first appearance among ids of that prefix (a 501 route on one side cannot shift the numbering of other kinds) | random (`secrets.token_hex(6)`) |
| N4 | API key secrets `ec_` + 43 url-safe characters anywhere in a string; a `prefix` field that is exactly `ec_` + 5 characters | `"<secret>"`, `"<prefix>"` | random (`secrets.token_urlsafe(32)`) |
| N5 | Live readings of the machine the local node runs on (fleet mode): `…memory.{used,available,free,cached,shmem,swap_used}_gib`, `…host_ram_free_gib`, `…cpu.{load1,load5,load15,util_pct}`, `…{drive,mount}.{used_bytes,free_bytes}`, `…disks[i].{used_bytes,free_bytes}`, `…host_ram.available_gib` | `"<live>"` | `/proc/meminfo`, the load average and `statvfs` change between two reads. Totals (`total_gib`, `size_bytes`), GPU numbers (from the fake `nvidia-smi`) and model sizes are compared. |
| N6 | `points` (and any `values` map) of `GET /api/v1/metrics/system` | `"<points>"` / `"<values>"` | how many 2-second samples the ring holds depends on how long the backend has run |
| N7 | Response headers outside the compared set (notably `date`, `server`, `content-length`) | not compared | `date` is the clock; `server: uvicorn` is a deliberate difference of the HTTP server, not of the API; `content-length` follows the body, which is compared |
| N8 | The run's temp dir and each backend's own root (HOME, data dir) anywhere in a string | `"<backend-root>"` | fresh per run and per backend |
| N9 | Prometheus text of `GET /metrics`: lines compared as a sorted list; sample values compared as numbers (`1.0` = `1`); series ending in `_created`, `_bucket`, `_sum`, and the host RAM gauges `engine_console_host_{ram_used,ram_available,shmem,swap_used}_gib` | `<volatile>` | `_created` are creation timestamps, `_bucket`/`_sum` are request latencies, the RAM gauges are live readings; the order of series is not part of the exposition format. Request **counts** (`engine_console_http_requests_total`, `_count`) are compared: both backends serve exactly the same requests (readiness is a TCP connect, never an HTTP request; the fleet readiness poll runs in lock step). |
| N10 | `etag`, `last-modified` headers | `"<present>"` | derived from file mtimes and the server's hashing scheme; the `If-None-Match` round trip (304) is compared through the status of the follow-up step |
| N11 | Live host-RAM readings formatted into text, anywhere in a string: `<x> GiB of host RAM`, `<x> GiB of <n> GiB` (available of total), `(<x> of <n> GiB)` (swap) | `<live>` for `<x>`; the total stays | the hostmem alerts and the fit's `host_ram.notes` quote `/proc/meminfo`, which each backend's node reads at its own moment (N5 for the same numbers in fields). Seen as flakiness under load (a parallel cargo build moves shared memory between the two reads). |
| N12 | The hub's own release number where the API states it: `$.version` of `GET /api/v1/health` and `$.info.version` of the OpenAPI document, when it equals that backend's package version (`pyproject.toml` of the Python reference, `backend/Cargo.toml` of the Rust hub) | `"<hub-version>"` | the reference is the last Python release (0.4.x); the Rust hub is 0.5.0 and later. A node's `version` (in `hello`, `/hosts`) is not touched: both hubs of a run drive the same kind of node. |

What is deliberately **not** normalised: error `detail` texts (the frontend shows them), 422 `errors` lists
(`loc`/`msg`/`type`), the machine's host name, kernel and CPU model (identical for both backends in one run), the
node's policy path (shared by both backends), GPU numbers, model sizes and quantisation, list order.
