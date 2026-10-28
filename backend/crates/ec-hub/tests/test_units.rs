//! Port of backend/tests/core/test_units.py, the registry rows. The store, redact/scrub, percentile and event
//! bus rows are unit tests in services/store.rs and services/common.rs; the hub parsing, egress allowlist and HTTP
//! client rows are in tests/test_wsc_units.rs (backend/TEST-MAP.md).

#![allow(clippy::panic)]

#[path = "common/kit.rs"]
mod kit;

use std::sync::Arc;

use ec_hub::adapters::base::EngineAdapter;
use ec_hub::adapters::{AdapterRegistry, load_default_adapters};

#[test]
fn test_registry_tolerates_missing_and_broken_modules() {
    // Python imports adapter modules by name and skips missing or broken ones; Rust adapters are compiled in, so only
    // the registry half is portable: lookups by id, and an unknown id is a 404 problem.
    let reg = AdapterRegistry::new([Arc::new(kit::FakeAdapter) as Arc<dyn EngineAdapter>]);
    assert!(reg.contains("fake"));
    assert_eq!(reg.get("fake").unwrap().display_name(), "Fake Engine");
    let Err(e) = reg.get("nope") else {
        panic!("an unknown engine was found")
    };
    assert_eq!((e.status, e.code.as_str()), (404, "unknown_engine"));
}

#[test]
fn test_default_registry_never_raises() {
    let reg = load_default_adapters();
    assert!(!reg.list().is_empty());
    assert!(reg.list().iter().all(|a| !a.id().is_empty()));
}
