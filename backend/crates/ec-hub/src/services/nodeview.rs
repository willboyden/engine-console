//! Read-only views over what a node last pushed (port of services/nodeview.py), shared by the fleet, library and
//! lifecycle services. Tests: tests/test_helpers.rs (port of tests/core/test_helpers.py).
//!
//! Everything a node says has already been through `ec_proto::sanitize_*`, but a value can still be missing or
//! null: every accessor tolerates that and returns a plain default instead of failing.

use std::sync::{Arc, LazyLock};

use regex::Regex;
use serde_json::Value;

use crate::domain::errors::Obj;
use crate::domain::fleet::NodeClient;

pub const GIB: f64 = 1_073_741_824.0;
static TAG: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,127}$").unwrap_or_else(|_| unreachable!())
});
static DIGEST: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[0-9a-f]{64}$").unwrap_or_else(|_| unreachable!()));

fn obj_of(v: Option<&Value>) -> Obj {
    match v {
        Some(Value::Object(o)) => o.clone(),
        _ => Obj::new(),
    }
}

fn arr_of(v: Option<&Value>) -> Vec<Value> {
    match v {
        Some(Value::Array(a)) => a.clone(),
        _ => Vec::new(),
    }
}

fn latest(node: &dyn NodeClient, kind: &str) -> Arc<Obj> {
    node.latest(kind).unwrap_or_default()
}

pub fn hello(node: &dyn NodeClient) -> Arc<Obj> {
    latest(node, "hello")
}

pub fn policy(node: &dyn NodeClient) -> Obj {
    obj_of(hello(node).get("policy"))
}

/// The node's policy mode. Anything unexpected is monitor mode: nothing may change on that machine.
pub fn mode(node: &dyn NodeClient) -> &'static str {
    if policy(node).get("mode").and_then(Value::as_str) == Some("manage") {
        "manage"
    } else {
        "monitor"
    }
}

pub fn arch(node: &dyn NodeClient) -> String {
    match hello(node).get("host").and_then(|h| h.get("arch")) {
        Some(Value::String(a)) if !a.is_empty() => a.clone(),
        _ => "x86_64".into(),
    }
}

pub fn link_up(node: &dyn NodeClient) -> bool {
    node.link().state == "up"
}

pub fn port_range(node: &dyn NodeClient, default: (i64, i64)) -> (i64, i64) {
    let pr = arr_of(policy(node).get("port_range"));
    if pr.len() == 2 {
        let (a, b) = (int_of(&pr[0]), int_of(&pr[1]));
        if let (Some(a), Some(b)) = (a, b)
            && 0 < a
            && a <= b
            && b <= 65535
        {
            return (a, b);
        }
    }
    default
}

/// An int that is not a bool (JSON has no bool/int confusion; floats are not ints).
pub fn int_of(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) if !n.is_f64() => n.as_i64(),
        _ => None,
    }
}

pub fn engine_images(node: &dyn NodeClient) -> Vec<String> {
    arr_of(policy(node).get("engine_images"))
        .into_iter()
        .filter_map(|p| p.as_str().map(str::to_string))
        .collect()
}

pub fn external_control(node: &dyn NodeClient) -> String {
    policy(node)
        .get("external_control")
        .and_then(Value::as_str)
        .unwrap_or("none")
        .to_string()
}

pub fn static_gpus(node: &dyn NodeClient) -> Vec<Obj> {
    arr_of(hello(node).get("gpus"))
        .into_iter()
        .filter_map(|g| match g {
            Value::Object(o) if o.get("uuid").is_some_and(ec_proto::sanitize::truthy) => Some(o),
            _ => None,
        })
        .collect()
}

pub fn uuid_index(node: &dyn NodeClient) -> Vec<(String, i64)> {
    static_gpus(node)
        .iter()
        .filter_map(|g| Some((value_str(g.get("uuid")?), int_of(g.get("index")?)?)))
        .collect()
}

/// `str(v)` of a JSON scalar as Python would print it for a str/int.
pub fn value_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

pub fn sample(node: &dyn NodeClient) -> Arc<Obj> {
    latest(node, "sample")
}

pub fn inventory(node: &dyn NodeClient) -> Arc<Obj> {
    latest(node, "inventory")
}

pub fn containers(node: &dyn NodeClient) -> Vec<Obj> {
    arr_of(inventory(node).get("containers"))
        .into_iter()
        .filter_map(|c| match c {
            Value::Object(o) if o.get("name").is_some_and(ec_proto::sanitize::truthy) => Some(o),
            _ => None,
        })
        .collect()
}

pub fn container_named(node: &dyn NodeClient, name: &str) -> Option<Obj> {
    containers(node)
        .into_iter()
        .find(|c| c.get("name").and_then(Value::as_str) == Some(name))
}

pub fn disk_mounts(node: &dyn NodeClient) -> Vec<Obj> {
    arr_of(latest(node, "disks").get("mounts"))
        .into_iter()
        .filter_map(|m| match m {
            Value::Object(o) if o.get("mount").is_some_and(ec_proto::sanitize::truthy) => Some(o),
            _ => None,
        })
        .collect()
}

pub fn endpoints(node: &dyn NodeClient) -> Vec<Obj> {
    arr_of(inventory(node).get("endpoints"))
        .into_iter()
        .filter_map(|e| match e {
            Value::Object(o) if o.get("port").and_then(int_of).is_some() => Some(o),
            _ => None,
        })
        .collect()
}

/// Host ports a container publishes that are reachable as 127.0.0.1 on its machine.
pub fn host_ports(c: &Obj) -> Vec<i64> {
    let mut out = Vec::new();
    for p in arr_of(c.get("ports")) {
        let Value::Object(p) = p else { continue };
        let ip = p.get("host_ip").and_then(Value::as_str).unwrap_or("");
        let hp = p.get("host_port").and_then(int_of);
        if ["", "0.0.0.0", "127.0.0.1", "::", "::1"].contains(&ip)
            && let Some(hp) = hp
            && !out.contains(&hp)
        {
            out.push(hp);
        }
    }
    out
}

/// Every host port the inventory says something listens on (container publishes and probed endpoints).
pub fn ports_in_use(node: &dyn NodeClient) -> Vec<i64> {
    let mut used: Vec<i64> = containers(node).iter().flat_map(host_ports).collect();
    used.extend(
        endpoints(node)
            .iter()
            .filter_map(|e| e.get("port").and_then(int_of)),
    );
    used.sort_unstable();
    used.dedup();
    used
}

/// root id -> root, in the policy's order.
pub fn model_roots(node: &dyn NodeClient) -> Vec<(String, Obj)> {
    let mut out: Vec<(String, Obj)> = Vec::new();
    for r in arr_of(policy(node).get("model_roots")) {
        if let Value::Object(o) = r
            && let Some(id) = o.get("id").filter(|v| ec_proto::sanitize::truthy(v))
        {
            let id = value_str(id);
            if let Some(slot) = out.iter_mut().find(|(k, _)| *k == id) {
                slot.1 = o;
            } else {
                out.push((id, o));
            }
        }
    }
    out
}

pub fn scripts(node: &dyn NodeClient) -> Vec<Obj> {
    arr_of(policy(node).get("scripts"))
        .into_iter()
        .filter_map(|s| match s {
            Value::Object(o) if o.get("name").is_some_and(ec_proto::sanitize::truthy) => Some(o),
            _ => None,
        })
        .collect()
}

pub fn addresses(node: &dyn NodeClient) -> Vec<String> {
    arr_of(hello(node).get("addresses"))
        .into_iter()
        .filter_map(|a| a.get("addr").and_then(Value::as_str).map(str::to_string))
        .collect()
}

/// `float(v)` of a number that is not a bool and not NaN.
pub fn num(v: Option<&Value>) -> Option<f64> {
    match v {
        Some(Value::Number(n)) => n.as_f64().filter(|x| !x.is_nan()),
        _ => None,
    }
}

pub fn gib(v: Option<&Value>) -> Option<f64> {
    num(v).map(|n| n / GIB)
}

pub fn ok_status(v: Option<&Value>) -> bool {
    v.and_then(int_of).is_some_and(|s| (200..300).contains(&s))
}

pub fn auth_status(v: Option<&Value>) -> bool {
    v.and_then(int_of).is_some_and(|s| s == 401 || s == 403)
}

/// The node policy's `engine_images` rule, mirrored so preflight can say no before a round trip.
pub fn image_allowed(image: &str, patterns: &[String]) -> bool {
    for p in patterns {
        if p == image {
            return true;
        }
        if let Some(repo) = p.strip_suffix("@sha256:*") {
            let digest = image.strip_prefix(&format!("{repo}@sha256:"));
            if !repo.is_empty() && !repo.contains('*') && digest.is_some_and(|d| DIGEST.is_match(d))
            {
                return true;
            }
        } else if let Some(repo) = p.strip_suffix(":*") {
            let tag = image.strip_prefix(&format!("{repo}:"));
            if !repo.is_empty()
                && !repo.contains('*')
                && !repo.contains('@')
                && tag.is_some_and(|t| TAG.is_match(t) && !t.ends_with('\n'))
            {
                return true;
            }
        }
    }
    false
}
