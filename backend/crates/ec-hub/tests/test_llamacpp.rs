//! Port of backend/tests/llamacpp/*.py (offline: the adapter, the committed fixtures of tests/data/llamacpp; no core
//! code, no docker), plus the llama.cpp rows of backend/tests/learn/test_learn_content.py. One test binary; the
//! modules mirror the Python files and every Rust test is the Python name with a `_llamacpp` suffix (backend/TEST-MAP.md).

#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::semicolon_if_nothing_returned,
    clippy::too_many_lines
)]

#[path = "common/a2_kit.rs"]
mod a2;

#[path = "llamacpp/test_acceptance.rs"]
mod test_acceptance;
#[path = "llamacpp/test_catalog.rs"]
mod test_catalog;
#[path = "llamacpp/test_compat.rs"]
mod test_compat;
#[path = "llamacpp/test_learn.rs"]
mod test_learn;
#[path = "llamacpp/test_memory.rs"]
mod test_memory;
#[path = "llamacpp/test_parse.rs"]
mod test_parse;
#[path = "llamacpp/test_probes.rs"]
mod test_probes;
#[path = "llamacpp/test_validate_launch.rs"]
mod test_validate_launch;
