//! Engine Console hub (Rust port of the Python hub `engine_console`, which it replaced in 0.5.0; backend/PORTING.md
//! records the port, and the tag python-final stays runnable as the parity reference).

// Adapters model Python's uncaught exceptions as unwinding panics that `adapters::pycompat::catch` and app.rs turn back
// into values (PORTING.md §A5): an abort-on-panic build would kill the console on the first raised exception.
#[cfg(panic = "abort")]
compile_error!(
    "ec-hub needs panic = \"unwind\": adapters raise Python exceptions as panics (PORTING.md §A5)"
);

pub mod adapters;
pub mod api;
pub mod app;
pub mod cli;
pub mod config;
pub mod container;
pub mod desktop;
pub mod domain;
pub mod fsutil;
pub mod learn;
pub mod panics;
pub mod pyjson;
pub mod pyyaml;
pub mod server;
pub mod services;
pub mod sync;
pub mod tools;

/// The release number (`engine_console.__version__` of the Python hub): the workspace version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
