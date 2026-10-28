//! Profiles YAML against Python's own ProfileService: the export bytes PyYAML writes, and what an import
//! creates or refuses, for the goldens made by backend/parity/gen/wsc_profile_goldens.py (PyYAML 6.0.3, pydantic and
//! the real vLLM / SGLang adapters).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use ec_hub::adapters::AdapterRegistry;
use ec_hub::domain::errors::AppError;
use ec_hub::domain::models::{Profile, ProfileIn};
use ec_hub::services::profiles::{ProfileService, import_fields, py_eq};
use ec_hub::services::store::Store;
use serde_json::{Value, json};

fn goldens() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/wsc_profile_goldens.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn service() -> (ProfileService, Arc<Store>) {
    let store = Arc::new(Store::memory().unwrap());
    let reg = AdapterRegistry::new([
        ec_hub::adapters::vllm::adapter(),
        ec_hub::adapters::sglang::adapter(),
    ]);
    (ProfileService::new(Arc::clone(&store), reg), store)
}

/// The golden's outcome shape for what the Rust service returned.
fn outcome(r: Result<Profile, AppError>) -> Value {
    match r {
        Ok(p) => {
            json!({"status": 201, "profile": {"name": p.name, "engine": p.engine, "repo_id": p.repo_id,
                                                  "description": p.description, "params": p.params}})
        }
        Err(AppError::Problem(p)) if p.status == 500 => json!({"status": 500}),
        Err(AppError::Problem(p)) => {
            let mut out = json!({"status": p.status, "code": p.code, "detail": p.detail});
            if !p.extra.is_empty() {
                out["extra"] = Value::Object(p.extra);
            }
            out
        }
        Err(other) => json!({"status": "node error", "error": format!("{other:?}")}),
    }
}

/// `import_yaml`, with an uncaught Python exception (a raise: FastAPI's catch-all 500) caught as `None`.
fn import(svc: &ProfileService, text: &str) -> Result<Profile, AppError> {
    ec_hub::adapters::pycompat::install_quiet_panic_hook();
    ec_hub::adapters::pycompat::catch(|| svc.import_yaml(text)).unwrap_or_else(|_| {
        Err(AppError::Problem(ec_hub::domain::errors::internal(
            "raised",
        )))
    })
}

/// Python's outcome without the exception class (a 500 is a 500 to the client).
fn expected(v: &Value) -> Value {
    if v["status"] == 500 {
        json!({"status": 500})
    } else {
        v.clone()
    }
}

#[test]
fn test_export_matches_pyyaml_byte_for_byte() {
    let g = goldens();
    for (i, case) in g["exports"].as_array().unwrap().iter().enumerate() {
        let (svc, store) = service();
        let row = &case["row"];
        let pid = format!("prof_{i:012x}");
        let params = ec_hub::pyjson::dumps(&row["params"]);
        store
            .execute(
                "INSERT INTO profiles VALUES(?,?,?,?,?,?,?,?)",
                &[
                    &pid,
                    &row["name"].as_str().unwrap(),
                    &row["engine"].as_str().unwrap(),
                    &row["repo_id"].as_str(),
                    &params,
                    &row["description"].as_str().unwrap(),
                    &1.0_f64,
                    &1.0_f64,
                ],
            )
            .unwrap();
        let text = svc.export_yaml(&pid).unwrap();
        assert_eq!(
            text,
            case["yaml"].as_str().unwrap(),
            "export {i}: {}",
            row["name"]
        );
    }
}

#[test]
fn test_import_matches_python_for_every_golden() {
    let g = goldens();
    let mut bad = Vec::new();
    for case in g["imports"].as_array().unwrap() {
        let (svc, _) = service();
        let text = case["yaml"].as_str().unwrap();
        let got = outcome(import(&svc, text));
        if got != expected(&case["result"]) {
            bad.push(format!(
                "{text:?}\n  python: {}\n  rust:   {got}",
                case["result"]
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of the import goldens differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

#[test]
fn test_import_twice_is_a_conflict() {
    let g = goldens();
    let d = &g["duplicate"];
    let (svc, _) = service();
    let text = d["yaml"].as_str().unwrap();
    assert_eq!(outcome(import(&svc, text)), expected(&d["first"]));
    assert_eq!(outcome(import(&svc, text)), expected(&d["result"]));
}

#[test]
fn test_every_export_imports_back_as_python_does() {
    let g = goldens();
    for case in g["roundtrips"].as_array().unwrap() {
        let (svc, _) = service();
        let text = case["yaml"].as_str().unwrap();
        assert_eq!(
            outcome(import(&svc, text)),
            expected(&case["result"]),
            "{text}"
        );
    }
}

/// A YAML-only Python object inside params is held back from the store: json.dumps would raise on it.
#[test]
fn test_import_flags_python_objects_json_cannot_hold() {
    let i =
        import_fields("name: x\nengine: vllm\nparams: {speculative_config: {when: 2024-01-01}}\n")
            .unwrap();
    assert!(i.foreign && i.stand_ins.is_empty());
    assert_eq!(i.fields["name"], "x");
    let plain = import_fields("name: x\nengine: vllm\nparams: {max_model_len: 4096}\n").unwrap();
    assert!(!plain.foreign);
    // a parameter whose own value has no JSON type keeps Python's str() for the engine's messages
    let d = import_fields(
        "name: x\nengine: vllm\nparams: {a: 2024-01-01, b: !!binary aGk=, c: .nan, d: [x]}\n",
    )
    .unwrap();
    assert_eq!(
        d.stand_ins,
        [
            ("a".to_owned(), "2024-01-01".to_owned()),
            ("b".to_owned(), "b'hi'".to_owned()),
            ("c".to_owned(), "nan".to_owned())
        ]
    );
    // bytes are text for a str field (pydantic decodes them) and invalid UTF-8 is not
    let f = import_fields("name: !!binary bmFtZQ==\nengine: vllm\n").unwrap();
    assert_eq!(f.fields["name"], "name");
    let f = import_fields("name: !!binary //79\nengine: vllm\n").unwrap();
    assert!(!f.fields["name"].is_string());
}

#[test]
fn test_crud_list_pagination_and_diff_on_the_store() {
    let (svc, _) = service();
    let mk = |name: &str, params: Value| ProfileIn {
        name: name.into(),
        engine: "vllm".into(),
        repo_id: None,
        preset: None,
        params: params.as_object().unwrap().clone(),
        description: String::new(),
    };
    let a = svc
        .create(&mk(
            "b-long",
            json!({"max_model_len": 8192, "block_size": 16}),
        ))
        .unwrap();
    let b = svc
        .create(&mk(
            "a-short",
            json!({"max_model_len": 4096.0, "enforce_eager": true}),
        ))
        .unwrap();
    let page = svc.list_page(1, None, None).unwrap();
    assert_eq!(page.items[0].name, "a-short");
    assert_eq!(page.next_cursor.as_deref(), Some("1"));
    let rest = svc.list_page(1, Some("1"), Some("vllm")).unwrap();
    assert!(rest.items[0].name == "b-long" && rest.next_cursor.is_none());
    assert!(
        svc.list_page(10, None, Some("sglang"))
            .unwrap()
            .items
            .is_empty()
    );
    let d = svc.diff(&a.id, &b.id).unwrap();
    assert_eq!(
        Value::Object(d.changed),
        json!({"max_model_len": {"a": 8192, "b": 4096.0}})
    );
    assert_eq!(Value::Object(d.only_a), json!({"block_size": 16}));
    assert_eq!(Value::Object(d.only_b), json!({"enforce_eager": true}));
    // 8192 == 8192.0 in Python: no change
    let c = svc
        .create(&mk("c", json!({"max_model_len": 8192.0})))
        .unwrap();
    assert!(svc.diff(&a.id, &c.id).unwrap().changed.is_empty());
    let err = svc.update(&b.id, &mk("b-long", json!({}))).unwrap_err();
    assert!(matches!(err, AppError::Problem(p) if p.code == "profile_exists" && p.status == 409));
    assert_eq!(
        svc.update(&b.id, &mk("renamed", json!({}))).unwrap().name,
        "renamed"
    );
    svc.delete(&b.id).unwrap();
    let gone = svc.get(&b.id).unwrap_err();
    assert!(
        matches!(gone, AppError::Problem(p) if p.code == "not_found" && p.detail == format!("no such profile: {}", b.id))
    );
}

#[test]
fn test_unknown_preset_lists_the_available_ones_sorted() {
    let (svc, _) = service();
    let p = ProfileIn {
        name: "x".into(),
        engine: "vllm".into(),
        repo_id: None,
        preset: Some("zzz".into()),
        params: serde_json::Map::new(),
        description: String::new(),
    };
    let AppError::Problem(e) = svc.create(&p).unwrap_err() else {
        panic!("not a problem")
    };
    assert_eq!(
        (e.status, e.code.as_str(), e.detail.as_str()),
        (422, "unknown_preset", "unknown preset 'zzz' for vllm")
    );
    assert_eq!(
        e.extra["available"],
        json!([
            "balanced",
            "long-context",
            "low-latency",
            "max-throughput",
            "reasoning",
            "tool-agent"
        ])
    );
}

#[test]
fn test_py_eq_follows_python() {
    assert!(
        py_eq(&json!(1), &json!(1.0))
            && py_eq(&json!(true), &json!(1))
            && py_eq(&json!(0.0), &json!(false))
    );
    assert!(!py_eq(&json!(1), &json!("1")) && !py_eq(&json!(null), &json!(false)));
    assert!(py_eq(&json!({"a": [1, 2.0]}), &json!({"a": [1.0, 2]})));
    assert!(!py_eq(&json!(u64::MAX), &json!(u64::MAX - 1)));
}
