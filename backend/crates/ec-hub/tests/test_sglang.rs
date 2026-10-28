//! Port of backend/tests/sglang/*.py (offline: only the adapter and committed fixtures), plus the SGLang rows of
//! backend/tests/learn/test_learn_content.py. One test binary; the modules mirror the Python files and every Rust test
//! is the Python name with a `_sglang` suffix (backend/TEST-MAP.md).

#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::float_cmp
)]

#[path = "common/ws_a1_kit.rs"]
mod ws_a1_kit;

#[path = "sglang/test_host_memory.rs"]
mod test_host_memory;
#[path = "sglang/test_learn_content.rs"]
mod test_learn_content;
#[path = "sglang/test_sglang_adapter.rs"]
mod test_sglang_adapter;
#[path = "sglang/test_sglang_snapshot.rs"]
mod test_sglang_snapshot;
