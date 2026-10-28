//! Services, grouped by area (backend/PORTING.md §3 and §4).

// core
pub mod audit;
pub mod common;
pub mod hostmem;
pub mod libport;
pub mod nodeerr;
pub mod nodeview;
pub mod secrets_store;
pub mod settings;
pub mod store;
pub mod telemetry;
pub mod usage;

// fleet
pub mod fleet;
pub mod fleet_status;
pub mod hardware;
pub mod hosts;
pub mod nodehttp;

// library and data
pub mod bench;
pub mod chat;
pub mod downloads;
pub mod fit;
pub mod fitting;
pub mod gguf_info;
pub mod hf;
pub mod hf_http;
pub mod library;
pub mod profiles;

// lifecycle
pub mod discovery;
pub mod docker;
pub mod externals;
pub mod lifecycle;
pub mod metrics;
pub mod runspec;
pub mod scripts;
