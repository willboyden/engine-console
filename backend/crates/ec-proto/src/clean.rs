//! Whitelist cleaning (proto.py `clean` / `clean_json`).
//!
//! A spec is INT | NUM | BOOL | JSONV | S(n) | L(spec, cap) | M(spec, cap) | {field: spec}. `clean` never fails: a
//! wrong type becomes null, an object spec yields exactly its own keys in spec order (missing -> null), lists and
//! maps are capped and lose their null members, strings are capped (in characters) and lose NUL characters.

use serde_json::{Number, Value};

use crate::Obj;

pub const MAX_NUM: f64 = 1e15;
pub const JSON_MAX_DEPTH: usize = 8;
pub const JSON_MAX_ITEMS: usize = 512;
pub const JSON_MAX_NODES: i64 = 4000;
pub const JSON_MAX_STR: usize = 4096;

#[derive(Debug, Clone, PartialEq)]
pub enum Spec {
    Int,
    Num,
    Bool,
    /// any JSON value, bounded (depth, items, string length): untrusted data
    Json,
    Str(usize),
    List(Box<Spec>, usize),
    Map(Box<Spec>, usize),
    Obj(Vec<(&'static str, Spec)>),
}

/// `S(n)`
pub fn s(n: usize) -> Spec {
    Spec::Str(n)
}

/// `L(spec, cap)`
pub fn l(spec: Spec, cap: usize) -> Spec {
    Spec::List(Box::new(spec), cap)
}

/// `M(spec, cap)`
pub fn m(spec: Spec, cap: usize) -> Spec {
    Spec::Map(Box::new(spec), cap)
}

/// `{field: spec, ...}`
pub fn o(fields: &[(&'static str, Spec)]) -> Spec {
    Spec::Obj(fields.to_vec())
}

impl Spec {
    /// The fields of an object spec (empty for any other spec).
    pub fn fields(&self) -> &[(&'static str, Spec)] {
        match self {
            Spec::Obj(f) => f,
            _ => &[],
        }
    }
}

/// Python `str[:n].replace("\x00", "")`
pub fn cap_str(v: &str, n: usize) -> String {
    v.chars().take(n).filter(|&c| c != '\0').collect()
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// An integer within +-2**53, or None (floats, even integral ones, are not integers: Python's isinstance(v, int)).
fn int_of(n: &Number) -> Option<Value> {
    if let Some(i) = n.as_i64() {
        return (i.unsigned_abs() <= 1 << 53).then(|| Value::from(i));
    }
    if let Some(u) = n.as_u64() {
        return (u <= 1 << 53).then(|| Value::from(u));
    }
    None
}

pub fn clean_json(v: &Value) -> Value {
    let mut budget = JSON_MAX_NODES;
    clean_json_inner(v, 0, &mut budget)
}

fn clean_json_inner(v: &Value, depth: usize, budget: &mut i64) -> Value {
    *budget -= 1;
    if *budget < 0 || depth > JSON_MAX_DEPTH {
        return Value::Null;
    }
    match v {
        Value::Null | Value::Bool(_) => v.clone(),
        Value::Number(n) => {
            if n.is_f64() {
                // serde_json never holds a non-finite float; keep the value as it came
                return v.clone();
            }
            int_of(n).unwrap_or(Value::Null)
        }
        Value::String(s) => Value::String(cap_str(s, JSON_MAX_STR)),
        Value::Array(a) => Value::Array(
            a.iter()
                .take(JSON_MAX_ITEMS)
                .map(|x| clean_json_inner(x, depth + 1, budget))
                .collect(),
        ),
        Value::Object(map) => {
            let mut out = Obj::new();
            for (k, x) in map.iter().take(JSON_MAX_ITEMS) {
                let n = char_len(k);
                if n > 0 && n <= 256 {
                    out.insert(k.replace('\0', ""), clean_json_inner(x, depth + 1, budget));
                }
            }
            Value::Object(out)
        }
    }
}

/// Clean `v` against `spec`. Null in, null out.
pub fn clean(v: &Value, spec: &Spec) -> Value {
    if v.is_null() {
        return Value::Null;
    }
    match spec {
        Spec::Int => match v {
            Value::Number(n) if !n.is_f64() => int_of(n).unwrap_or(Value::Null),
            _ => Value::Null,
        },
        Spec::Num => match v {
            Value::Number(n) => {
                if !n.is_f64() && int_of(n).is_none() {
                    return Value::Null;
                }
                let f = n.as_f64().unwrap_or(f64::NAN);
                if f.is_finite() && f.abs() <= MAX_NUM {
                    v.clone()
                } else {
                    Value::Null
                }
            }
            _ => Value::Null,
        },
        Spec::Bool => match v {
            Value::Bool(_) => v.clone(),
            _ => Value::Null,
        },
        Spec::Json => clean_json(v),
        Spec::Str(n) => match v {
            Value::String(s) => Value::String(cap_str(s, *n)),
            _ => Value::Null,
        },
        Spec::List(sub, cap) => match v {
            Value::Array(a) => Value::Array(
                a.iter()
                    .take(*cap)
                    .map(|x| clean(x, sub))
                    .filter(|x| !x.is_null())
                    .collect(),
            ),
            _ => Value::Null,
        },
        Spec::Map(sub, cap) => match v {
            Value::Object(map) => {
                let mut out = Obj::new();
                for (k, x) in map.iter().take(*cap) {
                    let n = char_len(k);
                    if n > 0 && n <= 256 {
                        let c = clean(x, sub);
                        if !c.is_null() {
                            out.insert(k.replace('\0', ""), c);
                        }
                    }
                }
                Value::Object(out)
            }
            _ => Value::Null,
        },
        Spec::Obj(fields) => match v {
            Value::Object(map) => {
                let mut out = Obj::new();
                for (k, sub) in fields {
                    out.insert(
                        (*k).to_string(),
                        clean(map.get(*k).unwrap_or(&Value::Null), sub),
                    );
                }
                Value::Object(out)
            }
            _ => Value::Null,
        },
    }
}

/// `clean(...)` of an object spec as a map: `{}` when the value was not an object (`clean(...) or {}`).
pub fn clean_obj(v: &Value, spec: &Spec) -> Obj {
    match clean(v, spec) {
        Value::Object(o) => o,
        _ => Obj::new(),
    }
}
