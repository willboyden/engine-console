//! Port of backend/tests/trtllm/*.py (offline: the adapter, the committed fixtures of tests/data/trtllm and the
//! Python-made goldens; no core code, no docker), plus the TensorRT-LLM rows of
//! backend/tests/learn/test_learn_content.py. One test binary; the modules mirror the Python files and every Rust test
//! is the Python name with a `_trtllm` suffix (backend/TEST-MAP.md).

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

#[path = "trtllm/test_trtllm_catalog.rs"]
mod test_trtllm_catalog;
#[path = "trtllm/test_trtllm_compat.rs"]
mod test_trtllm_compat;
#[path = "trtllm/test_trtllm_integration.rs"]
mod test_trtllm_integration;
#[path = "trtllm/test_trtllm_launch.rs"]
mod test_trtllm_launch;
#[path = "trtllm/test_trtllm_learn.rs"]
mod test_trtllm_learn;
#[path = "trtllm/test_trtllm_memory.rs"]
mod test_trtllm_memory;
#[path = "trtllm/test_trtllm_parse.rs"]
mod test_trtllm_parse;
