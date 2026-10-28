//! One release number: the workspace version (Cargo.toml) is what `GET /api/v1/health` reports (`ec_hub::VERSION`) and
//! what the served OpenAPI document states, so a release bump cannot leave one of them behind.
#![allow(clippy::unwrap_used)]

#[test]
fn test_openapi_document_states_the_release_version() {
    let spec: serde_json::Value =
        serde_json::from_str(include_str!("../assets/openapi.json")).unwrap();
    assert_eq!(spec["info"]["version"], ec_hub::VERSION);
    assert_eq!(ec_hub::VERSION, env!("CARGO_PKG_VERSION"));
}
