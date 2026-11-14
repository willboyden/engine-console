//! Engine Console's engine adapters and the pure code they need, shared by the hub (native) and the browser (wasm32,
//! through `ec-engines-wasm`): one copy of each engine's parameter catalog, presets, validation, launch rendering
//! (argv, env, files) and compatibility rules; and the Config files page's env and policy draft checks
//! (`envfile`, `policy_check`, `config_check`), so the browser checks an env draft without sending it.
//!
//! Moved out of `ec-hub` unchanged (ADR 0023); the hub re-exports `adapters`, `pyyaml`, `pyjson` and `errors`, so
//! every `ec_hub::adapters::...` path keeps working. Nothing here may use a clock, a file, a thread or the network on
//! wasm32: the vLLM adapter's monotonic clock is read only by its metrics parser, and on wasm32 it is a constant.
//!
//! Panics: the adapters raise Python exceptions as unwinding panics that `adapters::pycompat::catch` turns back into
//! values (PORTING.md §A5). On wasm32 (panic = abort) a panic traps the module instead; the browser's loader
//! (frontend/js/engines-wasm.js) discards the instance and shows the preview as unavailable. The hub keeps
//! panic = "unwind" (ec-hub refuses to build otherwise).

pub mod adapters;
pub mod config_check;
pub mod envfile;
pub mod errors;
pub mod policy_check;
pub mod pyjson;
pub mod pyyaml;
pub mod served;
pub mod text;
