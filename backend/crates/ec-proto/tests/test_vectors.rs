//! Cross-language vectors: backend/parity/proto/vectors.json holds the Python wire module's output (ecnode.proto) for a
//! corpus of hostile and edge inputs (backend/parity/proto/gen_vectors.py). The Rust port must give the same JSON value,
//! numbers compared with their int/float kind, for every vector.
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    unreachable_pub
)]

use ec_proto::clean::{l, m, s};
use ec_proto::{
    Obj, Spec, clean, clean_args, clean_json, missing_args, sanitize_event, sanitize_push,
    sanitize_res,
};
use serde_json::Value;

fn spec(name: &str) -> Spec {
    match name {
        "INT" => Spec::Int,
        "NUM" => Spec::Num,
        "BOOL" => Spec::Bool,
        "JSONV" => Spec::Json,
        "S5" => s(5),
        "S0" => s(0),
        "L_INT_3" => l(Spec::Int, 3),
        "L_S2_4" => l(s(2), 4),
        "M_S3_2" => m(s(3), 2),
        "M_INT_3" => m(Spec::Int, 3),
        other => panic!("unknown spec {other}"),
    }
}

/// Equality that also compares number KINDS (serde_json's `==` treats 1 and 1.0 as different already; this adds a
/// readable path to the first difference).
fn same(a: &Value, b: &Value, path: &str) -> Result<(), String> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            if x.is_f64() != y.is_f64() || x.as_f64() != y.as_f64() || (!x.is_f64() && x != y) {
                return Err(format!("{path}: {x} vs {y}"));
            }
            Ok(())
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Err(format!("{path}: length {} vs {}", x.len(), y.len()));
            }
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                same(p, q, &format!("{path}/{i}"))?;
            }
            Ok(())
        }
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().chain(y.keys()) {
                match (x.get(k), y.get(k)) {
                    (Some(p), Some(q)) => same(p, q, &format!("{path}/{k}"))?,
                    _ => return Err(format!("{path}/{k}: present on one side only")),
                }
            }
            Ok(())
        }
        _ if a == b => Ok(()),
        _ => Err(format!("{path}: {a} vs {b}")),
    }
}

fn obj(v: &Value) -> Obj {
    v.as_object().cloned().unwrap_or_default()
}

#[test]
fn test_python_vectors_match() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../parity/proto/vectors.json"
    );
    let doc: Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("vectors.json")).unwrap();
    let vectors = doc["vectors"].as_array().unwrap();
    assert!(vectors.len() > 500, "the corpus shrank: {}", vectors.len());
    let mut failures = Vec::new();
    for (i, vec) in vectors.iter().enumerate() {
        let kind = vec["kind"].as_str().unwrap();
        let input = &vec["input"];
        let op = vec.get("op").and_then(Value::as_str).unwrap_or("");
        let want_error = vec.get("error") == Some(&Value::Bool(true));
        let got: Result<Value, ()> = match kind {
            "clean" => Ok(clean(input, &spec(vec["spec"].as_str().unwrap()))),
            "clean_json" => Ok(clean_json(input)),
            "sanitize_push" => match input {
                Value::Object(o) => match sanitize_push(o) {
                    Ok(Some(x)) => Ok(Value::Object(x)),
                    Ok(None) => Ok(Value::Null),
                    Err(_) => Err(()),
                },
                _ => panic!("vector {i}: push input is not an object"),
            },
            "sanitize_res" => Ok(Value::Object(sanitize_res(&obj(input), op))),
            "sanitize_event" => Ok(Value::Object(sanitize_event(&obj(input), op))),
            "clean_args" => clean_args(op, input).map(Value::Object).map_err(|_| ()),
            "missing_args" => Ok(Value::Array(
                missing_args(op, &obj(input))
                    .into_iter()
                    .map(|k| Value::String(k.into()))
                    .collect(),
            )),
            other => panic!("unknown vector kind {other}"),
        };
        match (got, want_error) {
            (Err(()), true) => {}
            (Err(()), false) => {
                failures.push(format!("#{i} {kind} {op}: Rust refused, Python answered"));
            }
            (Ok(_), true) => {
                failures.push(format!("#{i} {kind} {op}: Python refused, Rust answered"));
            }
            (Ok(g), false) => {
                if let Err(e) = same(&vec["output"], &g, "") {
                    failures.push(format!("#{i} {kind} {op}: {e}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} vectors differ:\n{}",
        failures.len(),
        vectors.len(),
        failures.join("\n")
    );
}

/// The inputs gen_vectors.py lists under DEVIATIONS (they cannot be vectors): pin what the Rust peer does, so a
/// change is noticed. backend/PORTING.md "Deviations" carries the same list.
#[test]
fn test_documented_deviations() {
    // 1. an integer beyond 64 bits arrives as a float: INT and NUM drop it like Python does, JSONV keeps the float
    let o = ec_proto::decode_json(br#"{"n": 18446744073709551621}"#).unwrap();
    assert!(o["n"].is_f64());
    assert_eq!(clean(&o["n"], &Spec::Int), Value::Null);
    assert_eq!(clean(&o["n"], &Spec::Num), Value::Null);
    assert!(clean_json(&o["n"]).is_f64(), "Python: None");
    // 2. a lone surrogate is refused (Python's json accepts it)
    assert!(ec_proto::decode_json(br#"{"s": "\ud800"}"#).is_err());
    // 3. nesting between serde_json's limit (128) and Python's (~1000) is refused
    let deep = format!("{{\"a\":{}1{}}}", "[".repeat(200), "]".repeat(200));
    assert!(ec_proto::decode_json(deep.as_bytes()).is_err());
}
