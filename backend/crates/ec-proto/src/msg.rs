//! Message builders (both sides). Same keys, in the same order, as proto.py.

use serde_json::{Value, json};

use crate::Obj;

fn obj(v: Value) -> Obj {
    match v {
        Value::Object(o) => o,
        _ => Obj::new(),
    }
}

pub fn req(rid: i64, op: &str, args: Option<Obj>) -> Obj {
    obj(
        json!({"type": "req", "id": rid, "op": op, "args": Value::Object(args.unwrap_or_default())}),
    )
}

pub fn res_ok(rid: i64, result: Option<Obj>) -> Obj {
    obj(
        json!({"type": "res", "id": rid, "ok": true, "result": Value::Object(result.unwrap_or_default())}),
    )
}

/// `message[:1000]` (characters, as Python slices).
pub fn res_err(rid: i64, code: &str, message: &str) -> Obj {
    let message: String = message.chars().take(1000).collect();
    obj(json!({"type": "res", "id": rid, "ok": false, "error": {"code": code, "message": message}}))
}

pub fn event(rid: i64, ev: Obj) -> Obj {
    obj(json!({"type": "event", "id": rid, "event": Value::Object(ev)}))
}

pub fn credit(rid: i64, nbytes: u64) -> Obj {
    obj(json!({"type": "credit", "id": rid, "bytes": nbytes}))
}

pub fn end(rid: i64) -> Obj {
    obj(json!({"type": "end", "id": rid}))
}

pub fn cancel(rid: i64) -> Obj {
    obj(json!({"type": "cancel", "id": rid}))
}

pub fn ping(rid: i64) -> Obj {
    obj(json!({"type": "ping", "id": rid}))
}

pub fn pong(rid: i64) -> Obj {
    obj(json!({"type": "pong", "id": rid}))
}
