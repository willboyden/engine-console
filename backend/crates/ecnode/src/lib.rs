//! Engine Console node agent (Rust port of node/src/ecnode). One per machine; speaks the ecnode wire protocol
//! (crate `ec_proto`) on stdin/stdout to the hub, within its own local policy.
//!
//! The command line and the protocol server are `entry.rs`, `server.rs` and `errors.rs`; every other module is one
//! of the node's capabilities (backend/PORTING.md section 3.2 maps each to its Python source).

pub mod agent;
pub mod cli;
pub mod confine;
pub mod discovery;
pub mod disks;
pub mod docker;
pub mod entry;
pub mod errors;
pub mod fsops;
pub mod gguf;
pub mod gpus;
pub mod hostinfo;
pub mod httpproxy;
pub mod memory;
pub mod models;
pub mod ops;
pub mod policy;
pub mod scripts;
pub mod server;
pub mod util;
