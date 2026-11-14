# ADR-0023: The hub's own checks in the browser, as one WebAssembly module

Status: accepted (Unreleased, after 0.6.0). Amends ADR-0004: the frontend keeps zero runtime dependencies and no npm
package ships, but it now has one build step, `make wasm`. Implemented in `backend/crates/ec-engines`,
`backend/crates/ec-engines-wasm`, `frontend/js/engines-wasm.js` and `frontend/js/launch-preview.js`.
**Verified:** native unit tests of every export, the frontend's `node --test` suites against the built `.wasm`, and the
browser suite (`make e2e`) in headless Chrome against the real hub under its real CSP. **Not verified:** load time of the
module over a slow link or on a slow machine (no measurement), and browsers other than Chrome.

## Context

The Launch page's "copy as command" preview and its parameter validation were a hand-written JavaScript copy of the
hub's engine adapters, and the copy had drifted for four of the five engines (measured on 0.6.0 by the analysis behind
this ADR). For vLLM, a boolean set to false renders as `--no-<flag>` in the hub (for its negatable flags) and as
nothing in the preview, and the preview lacked the hub's `--host`, `--port`, `--served-model-name` and its environment;
SGLang's and llama.cpp's previews named the model with a flag the engine does not take; TensorRT-LLM's showed its
`yaml:` parameters as command-line flags, where the hub writes them into an options file. Cross-field rules (vLLM's `enable_auto_tool_choice` needs `tool_call_parser`, and others)
existed only in Rust, so the page let a launch through that preflight then refused. The Host page's image-policy badge
used a glob that allows `repo:tag@sha256:...` for `repo:*`, which the node refuses. The Config files page sent an env
draft, which may hold `HF_TOKEN`, to the hub only to have it checked.

The adapters are pure code: a parameter catalog, presets, validation and rendering, with no clock, file, thread or
network. They compile to `wasm32-unknown-unknown` unchanged.

## Decision

Compile the hub's own code to one small WebAssembly module and call it from the existing JavaScript UI. No UI rewrite.

- **One copy.** `ec-engines` holds the engine adapters, their Python-compatible JSON and YAML, the problem type, the
  env file's reader and value parsers (`envfile`), the node policy validator's mirror (`policy_check`) and the Config
  files page's env and policy checks (`config_check`). The hub links it natively and re-exports every moved path, so
  the hub's behaviour and its tests are unchanged. The image-policy rule moves to `ec-proto` (`image`), which the node,
  the hub and the module share.
- **The module.** `ec-engines-wasm` (cdylib, `wasm-bindgen = "=0.2.129"`, no `web-sys`) exports a JSON-in, JSON-out API:
  `engines`, `catalog`, `presets`, `validate`, `render` (the LaunchSpec: argv, env, files),
  `image_allowed`, `config_check` (POST /config/validate's answer for `env` and `policy`), `dotenv` (the hub's env
  reader, for the line editor's tests), `api_version` and `build_version`.
- **The pages.** `js/engines-wasm.js` loads it once, on the pages that use it (Launch, the wizard, a host's page, Config
  files). The Launch preview is the adapter's own output for the LaunchContext the hub would build
  (`js/launch-preview.js`), with the adapter's validation listed; the image badge and the engine choices use the node's
  rule; the env file and a node policy are checked in the page as they are typed. An env draft is never sent to the
  hub: without the module the page says it cannot check it (the API client refuses `file: env` outright).
- **Advisory only.** Nothing the module answers is trusted: preflight and launch validate on the hub, the node decides
  its policy, GET /instances/{id}/command stays the record after creation. No route, protocol message or stored value
  changed.
- **Failure is a degraded page, not a broken one.** If the module cannot load (not built, an older browser, a CSP
  mistake) the pages behave as before: the preview is the old approximation, labelled as such; image badges use the old
  glob; a policy draft goes to the hub. An env draft does not: the page says it cannot be checked. An adapter panic
  aborts the module on wasm32 (`panic = "abort"`; the hub keeps unwinding and turns the same panic into a 500): the
  loader marks the instance dead, never calls it again, and loads a fresh one, at most twice per page life (each dead
  instance stays in memory until the page goes). The call and input that trapped are remembered and answered
  "unavailable" without running the module again, so a page that repaints on reload cannot loop. One such input is
  known: SGLang's validation of `ep_size: null` (a panic kept on purpose: the hub answers it with the Python release's
  500, and the page never sends null).

## Build, size and supply chain

- `make wasm` builds the module with the `release-web` profile (size-optimised, fat LTO, panic = abort) and the same
  `--remap-path-prefix` flags as `make release`, then runs the `wasm-bindgen` CLI, which must be exactly the version of
  the `wasm-bindgen` crate in `Cargo.lock` (they share a private ABI; the target refuses another version). It fails if
  a build-machine path is left in the module or the glue, or if the module is over 2 MiB (`WASM_MAX_BYTES`). Output:
  `frontend/pkg/`, gitignored and Prettier-ignored, served by the hub under `/pkg/` (`application/wasm`, the same
  `no-cache` and ETag as every frontend file). Only what serves the UI depends on it (`make run`, `make install-desktop`,
  `make e2e`); the binaries, `make parity` and `make proto` need neither the wasm32 target nor the CLI. `make wasm` and
  `make clippy` name the missing target with the command that installs it.
- Size when this was written: 1,678,978 bytes raw, 577,479 bytes gzip -9 (the hub does not compress; the module is
  loaded only by the four pages above, and after the first load it is revalidated, not refetched).
- No new third-party crate in `Cargo.lock`: `wasm-bindgen` 0.2.129 and its macro and shared crates were already locked
  (dependencies of other crates for wasm targets; the same version harness2-workbench vetted). `ec-engines-wasm` makes
  `wasm-bindgen` a direct dependency, pinned with `=`; cargo-deny and the 14-day crate-age gate pass. The wasm32 target is in `rust-toolchain.toml`. clippy runs
  with `-D warnings` for wasm32 as well as natively (`make clippy`, CI).
- Still no npm package in the browser and no CDN. But the module is not only this repository's code: it statically
  links the third-party crates the adapters use (among them `serde`, `serde_json`, `regex`, `indexmap`, `tracing`
  and `wasm-bindgen`'s runtime), the same locked versions the hub links, covered by cargo-deny and the crate-age gate.
  The `wasm-bindgen` CLI that writes the JS glue is a build-time trust root outside those gates: it is installed by
  `cargo install --locked` at the pinned version, and its output is checked only for build-machine paths and by the
  CSP-hygiene test of the glue (no `eval`, no HTML sink). A clean clone has no complete UI before `make wasm`; without
  it the pages fall back as described above.

## Security

- **CSP.** `script-src 'self'` becomes `script-src 'self' 'wasm-unsafe-eval'`: the page may compile WebAssembly; it
  still cannot evaluate JavaScript strings or run inline script. Every other header is unchanged. The parity harness
  accepts exactly that pair of CSP values against the Python reference (`D-CSP-WASM` in
  `backend/parity/allowed-diffs.json`); the browser suite records every CSP violation and fails on one, and fails when
  the allowance is removed (checked by removing it).
- **Secrets.** The preview masks secret-looking env names and secret parameter values wherever the adapter put them,
  and says when one would land on the command line (the hub refuses that launch). An env draft never leaves the page;
  `make e2e` fails on any POST /config/validate with `file: env`, with the module loaded and with `/pkg/` blocked.
- **Trust.** The module is built from this repository's code and its locked crates and served by the hub from the
  checkout like every other frontend file; it is not a new origin and holds no key. Its answers never decide anything
  on the hub or the node.

## Consequences

- One copy of the adapters, the image rule and the env and policy checks for the hub, the node and the browser. The hub
  proves the browser's env check equals its own loader (`config::tests::test_env_draft_check_matches_the_loader` and a
  3000-case property test); the frontend tests prove the line editor reads values as the hub does.
- ADR-0004's "zero-build" is now "one build step, `make wasm`, with a pinned CLI". The supply-chain argument of ADR-0004
  holds only in part: no npm package and no CDN, but the browser now runs Rust crates from crates.io compiled into the
  module (above), and glue written by a tool.
- Contributors need the wasm32 target (installed from `rust-toolchain.toml`) and `cargo install --locked
  wasm-bindgen-cli@0.2.129` to build the full UI.
- The mock server (`frontend/dev/`) serves the module and sends the hub's `script-src`; it still answers parameter
  catalogs and config checks from its own samples (moving it to the module is open).
- Not done: the VRAM fit estimate stays on the hub (it needs model metadata, live free memory and resident engines).

## Alternatives considered

- **`POST /engines/{e}/render` on the hub.** Fixes the preview with less machinery, but every keystroke would be a
  request, the env draft would still be sent, and the mock server would need a fourth hand-written renderer. Rejected.
- **A full Rust UI (Leptos).** Shares the serde types too, but rewrites the whole JavaScript UI and its tests for no
  user-visible gain beyond this ADR. Rejected for now.
- **Fixing the JavaScript copy.** It would drift again; the measured drift is what started this. Rejected.
