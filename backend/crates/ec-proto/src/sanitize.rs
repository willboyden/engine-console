//! Sanitizers (hub side) and argument cleaning (node side), as proto.py.

use serde_json::Value;

use crate::clean::{Spec, clean, clean_obj};
use crate::{INTERNAL, Obj, ProtoError, specs};

/// A pushed node message rebuilt from its whitelist; `Ok(None)` for a type the hub does not know (ignore it).
/// A `hello` without a protocol version is a protocol error.
pub fn sanitize_push(msg: &Obj) -> Result<Option<Obj>, ProtoError> {
    let Some(Value::String(t)) = msg.get("type") else {
        return Ok(None);
    };
    let Some(spec) = specs::push(t) else {
        return Ok(None);
    };
    let mut out = clean_obj(&Value::Object(msg.clone()), spec);
    out.insert("type".into(), Value::String(t.clone()));
    if t == "hello" {
        if out.get("proto").is_none_or(Value::is_null) {
            return Err(ProtoError::invalid("hello carries no protocol version"));
        }
        let gpus: Vec<Value> = match out.get("gpus") {
            Some(Value::Array(a)) => a
                .iter()
                .filter(|g| g.get("uuid").is_some_and(truthy))
                .cloned()
                .collect(),
            _ => Vec::new(),
        };
        out.insert("gpus".into(), Value::Array(gpus));
    }
    Ok(Some(out))
}

/// Python truthiness of a JSON value.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `{"id", "ok", "result"|"error"}` for the response to `op` (the hub knows the op from its own request).
pub fn sanitize_res(msg: &Obj, op: &str) -> Obj {
    let mut out = Obj::new();
    out.insert(
        "id".into(),
        clean(msg.get("id").unwrap_or(&Value::Null), &crate::Spec::Int),
    );
    let ok = msg.get("ok") == Some(&Value::Bool(true));
    out.insert("ok".into(), Value::Bool(ok));
    if ok {
        let result = clean_obj(
            msg.get("result").unwrap_or(&Value::Null),
            specs::results(op),
        );
        out.insert("result".into(), Value::Object(result));
    } else {
        let err = clean_obj(msg.get("error").unwrap_or(&Value::Null), &specs::error());
        let pick = |k: &str, dflt: &str| -> Value {
            match err.get(k) {
                Some(v) if truthy(v) => v.clone(),
                _ => Value::String(dflt.into()),
            }
        };
        let mut e = Obj::new();
        e.insert("code".into(), pick("code", INTERNAL));
        e.insert(
            "message".into(),
            pick("message", "the node refused without saying why"),
        );
        out.insert("error".into(), Value::Object(e));
    }
    out
}

pub fn sanitize_event(msg: &Obj, op: &str) -> Obj {
    let mut out = Obj::new();
    out.insert(
        "id".into(),
        clean(msg.get("id").unwrap_or(&Value::Null), &crate::Spec::Int),
    );
    out.insert(
        "event".into(),
        Value::Object(clean_obj(
            msg.get("event").unwrap_or(&Value::Null),
            specs::events(op),
        )),
    );
    out
}

/// Node side: the whitelisted arguments of a request (wrong types -> null); ProtoError for an unknown op.
pub fn clean_args(op: &str, args: &Value) -> Result<Obj, ProtoError> {
    let Some(spec) = specs::args(op) else {
        return Err(ProtoError::invalid(format!("unknown op {}", py_repr(op))));
    };
    let input = if args.is_object() {
        args.clone()
    } else {
        Value::Object(Obj::new())
    };
    Ok(clean_obj(&input, spec))
}

/// Fields whose value names what the node acts on: a path, a link and its target, a container, an image, a command
/// line. `clean` would cut an over-long one to its cap (and drop its NUL characters), so the node would act on a
/// DIFFERENT path, container or argv than the hub asked for; the node refuses such a request instead (security
/// audit F5, beyond Python). `clean` itself is unchanged: it still bounds every pushed message the hub reads.
pub const STRICT_FIELDS: &[&str] = &["rel", "link", "target", "name", "image", "argv"];

/// Node side, before `clean_args`: why a request's STRICT_FIELDS would not survive cleaning unchanged (longer than
/// the spec's cap in characters, a list with more items than its cap, or a NUL character), else None. Other fields
/// keep Python's cleaning; an unknown op is clean_args' to refuse.
pub fn overlong_args(op: &str, args: &Value) -> Option<String> {
    let spec = specs::args(op)?;
    overlong_in(args, spec, "")
        .map(|why| format!("{op}: {why}; the node refuses it rather than cut it short"))
}

/// Search an argument for STRICT_FIELDS: through objects and lists of objects (`spec.mounts[i].rel`).
fn overlong_in(v: &Value, spec: &Spec, at: &str) -> Option<String> {
    match (spec, v) {
        (Spec::Obj(fields), Value::Object(map)) => fields.iter().find_map(|(k, sub)| {
            let x = map.get(*k)?;
            let here = if at.is_empty() {
                (*k).to_string()
            } else {
                format!("{at}.{k}")
            };
            if STRICT_FIELDS.contains(k) {
                strict(x, sub, &here)
            } else {
                overlong_in(x, sub, &here)
            }
        }),
        (Spec::List(sub, cap), Value::Array(a)) => a
            .iter()
            .take(*cap)
            .enumerate()
            .find_map(|(i, x)| overlong_in(x, sub, &format!("{at}[{i}]"))),
        _ => None,
    }
}

/// A strict value: a string within its cap and without NUL, or a list within its cap of such strings. A value of
/// another type is left to `clean` (it becomes null, and the op refuses a missing or invalid argument itself).
fn strict(v: &Value, spec: &Spec, at: &str) -> Option<String> {
    match (spec, v) {
        (Spec::Str(n), Value::String(s)) => {
            if s.contains('\0') {
                Some(format!("{at} contains a NUL character"))
            } else if s.chars().count() > *n {
                Some(format!("{at} is longer than {n} characters"))
            } else {
                None
            }
        }
        (Spec::List(sub, cap), Value::Array(a)) => {
            if a.len() > *cap {
                return Some(format!("{at} has more than {cap} items"));
            }
            a.iter()
                .enumerate()
                .find_map(|(i, x)| strict(x, sub, &format!("{at}[{i}]")))
        }
        _ => None,
    }
}

/// Required arguments that are missing, null, "" or [].
pub fn missing_args(op: &str, cleaned: &Obj) -> Vec<&'static str> {
    specs::required(op)
        .iter()
        .copied()
        .filter(|k| match cleaned.get(*k) {
            None | Some(Value::Null) => true,
            Some(Value::String(s)) => s.is_empty(),
            Some(Value::Array(a)) => a.is_empty(),
            Some(_) => false,
        })
        .collect()
}

/// Python's `repr()` of a str (as used in protocol messages: `f"... {op!r}"`).
pub fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c if is_py_unprintable(c) => {
                let n = c as u32;
                if n <= 0xff {
                    out.push_str(&format!("\\x{n:02x}"));
                } else if n <= 0xffff {
                    out.push_str(&format!("\\u{n:04x}"));
                } else {
                    out.push_str(&format!("\\U{n:08x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Approximation of Python's `str.isprintable()` == false for non-ASCII: C1 controls, NBSP-like separators and
/// format characters. Exact Unicode tables are not needed: these strings are short op names and paths.
fn is_py_unprintable(c: char) -> bool {
    let n = c as u32;
    (0x80..=0xa0).contains(&n)
        || n == 0xad
        || (0x2000..=0x200f).contains(&n)
        || (0x2028..=0x202f).contains(&n)
        || (0x205f..=0x206f).contains(&n)
        || n == 0x3000
        || n == 0xfeff
        || (0xfff9..=0xfffb).contains(&n)
        || (0xd800..=0xdfff).contains(&n)
}
