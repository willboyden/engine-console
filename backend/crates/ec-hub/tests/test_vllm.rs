//! Port of backend/tests/vllm/*.py (offline: only the adapter and committed fixtures; no core code), plus the vLLM
//! rows of backend/tests/learn/test_learn_content.py. One test binary; the modules mirror the Python files and every
//! Rust test is the Python name with a `_vllm` suffix (backend/TEST-MAP.md).

#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::float_cmp
)]

#[path = "common/ws_a1_kit.rs"]
mod ws_a1_kit;

#[path = "vllm/test_catalog.rs"]
mod test_catalog;
#[path = "vllm/test_compat.rs"]
mod test_compat;
#[path = "vllm/test_help_snapshot.rs"]
mod test_help_snapshot;
#[path = "vllm/test_host_memory.rs"]
mod test_host_memory;
#[path = "vllm/test_learn_content.rs"]
mod test_learn_content;
#[path = "vllm/test_parse.rs"]
mod test_parse;
#[path = "vllm/test_validate_launch.rs"]
mod test_validate_launch;
