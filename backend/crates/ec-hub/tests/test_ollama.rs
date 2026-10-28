//! Port of backend/tests/ollama/*.py (offline: the adapter, the committed fixtures of tests/data/ollama; no core code,
//! no docker), plus the Ollama rows of backend/tests/learn/test_learn_content.py. One test binary; the modules mirror
//! the Python files and every Rust test is the Python name with an `_ollama` suffix (backend/TEST-MAP.md).

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

#[path = "ollama/test_catalog.rs"]
mod test_catalog;
#[path = "ollama/test_compat.rs"]
mod test_compat;
#[path = "ollama/test_learn.rs"]
mod test_learn;
#[path = "ollama/test_live_env.rs"]
mod test_live_env;
#[path = "ollama/test_memory.rs"]
mod test_memory;
#[path = "ollama/test_parse.rs"]
mod test_parse;
#[path = "ollama/test_probes.rs"]
mod test_probes;
#[path = "ollama/test_validate_launch.rs"]
mod test_validate_launch;
