//! pydantic v2's own answers (tests/data/validation_vectors.json, written by backend/parity/gen/validation_vectors.py)
#![allow(clippy::panic)] // test helper outside #[test]: an unknown vector type is a broken data file
//! replayed against api/validate.rs: same success values (int vs float kind included) or the same error list.

use ec_hub::api::validate::{T, check};
use serde_json::Value;

fn t_of(name: &str) -> T {
    match name {
        "int" => T::int(),
        "float" => T::Float,
        "bool" => T::Bool,
        "str" => T::str(),
        "int_ge1_le4096" => T::Int {
            ge: Some(1),
            le: Some(4096),
        },
        "int_ge0" => T::Int {
            ge: Some(0),
            le: None,
        },
        "str_min1_max5" => T::Str {
            min: Some(1),
            max: Some(5),
            pattern: None,
        },
        "host_id" => T::Str {
            min: None,
            max: None,
            pattern: Some(r"^[a-z0-9][a-z0-9-]{0,31}$"),
        },
        "literal_role" => T::Literal(&["admin", "viewer"]),
        "list_int" => T::list(T::int()),
        "dict_str" => T::dict(T::str()),
        "opt_int" => T::opt(T::int()),
        other => panic!("unknown type {other}"),
    }
}

#[test]
fn test_validation_vectors_match_pydantic() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/validation_vectors.json"
    ))
    .unwrap();
    let vectors: Vec<Value> = serde_json::from_str(&text).unwrap();
    let mut bad = Vec::new();
    for v in &vectors {
        let t = t_of(v["type"].as_str().unwrap());
        let mut errs = Vec::new();
        let out = check(&v["input"], &t, &[], &mut errs);
        let ok = v["ok"].as_bool().unwrap();
        if ok {
            let want = &v["value"];
            let non_finite = want == "<non-finite>";
            if !errs.is_empty() || (!non_finite && &out != want) {
                bad.push(format!(
                    "{} {} -> want {want}, got {out} {errs:?}",
                    v["type"], v["input"]
                ));
            }
        } else {
            let got: Vec<Value> = errs
                .iter()
                .map(|e| serde_json::json!({"loc": e.loc, "msg": e.msg, "type": e.typ}))
                .collect();
            if got != v["errors"].as_array().cloned().unwrap_or_default() {
                bad.push(format!(
                    "{} {} -> want {}, got {:?}",
                    v["type"], v["input"], v["errors"], got
                ));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} vectors differ:\n{}",
        bad.len(),
        vectors.len(),
        bad.join("\n")
    );
}
