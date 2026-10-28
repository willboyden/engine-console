//! Small shared helpers (port of services/common.py): ids, clock, in-process pub/sub for SSE, redaction, SSE
//! framing, stream limits, fnmatch.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use regex::Regex;
use serde_json::{Map, Value};
use tokio::sync::mpsc;

use crate::pyjson;
use crate::sync::Recover as _;

/// `time.time()`
pub fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// `secrets.token_bytes(n)`. Panics when the operating system's random source fails (Python's `secrets` raises):
/// ids, keys and tokens are never made from a failed RNG. A request that gets here is answered 500 by app.rs's panic
/// containment; a background iteration is logged and skipped (`panics::guarded`).
pub fn random_bytes(n: usize) -> Vec<u8> {
    random_bytes_from(getrandom::fill, n)
}

/// `random_bytes` over a given random source (the OS's in the hub; a failing one in tests).
fn random_bytes_from(
    fill: impl FnOnce(&mut [u8]) -> Result<(), getrandom::Error>,
    n: usize,
) -> Vec<u8> {
    let mut buf = vec![0u8; n];
    if let Err(e) = fill(&mut buf) {
        // fail closed: never all-zero (or partly filled) bytes
        #[allow(clippy::panic)]
        {
            panic!("the operating system's random number generator failed: {e}");
        }
    }
    buf
}

/// `prefix + secrets.token_hex(6)`
pub fn new_id(prefix: &str) -> String {
    format!("{prefix}{}", hex::encode(random_bytes(6)))
}

/// `secrets.token_urlsafe(n)`: base64url of n random bytes, no padding. Errors if the OS has no entropy.
pub fn token_urlsafe(n: usize) -> std::io::Result<String> {
    use base64::Engine as _;
    let mut buf = vec![0u8; n];
    getrandom::fill(&mut buf).map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(buf))
}

// ── EventBus ─────────────────────────────────────────────────────────────────────────────────────────────
/// Topic -> bounded queues (256 each). A slow SSE client drops events instead of stalling producers.
type Subscribers = HashMap<String, Vec<(u64, mpsc::Sender<Value>)>>;

#[derive(Default)]
pub struct EventBus {
    subs: Mutex<Subscribers>,
    next: AtomicU64,
}

impl std::fmt::Debug for EventBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("EventBus")
    }
}

pub const BUS_QUEUE: usize = 256;

impl EventBus {
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::needless_pass_by_value)] // callers hand over a fresh value, as Python does
    pub fn publish(&self, topic: &str, event: Value) {
        let mut subs = self.subs.guard();
        if let Some(list) = subs.get_mut(topic) {
            list.retain(|(_, tx)| !tx.is_closed());
            for (_, tx) in list.iter() {
                let _ = tx.try_send(event.clone()); // full: this subscriber drops the event
            }
        }
    }

    /// A subscription; it unsubscribes when dropped.
    pub fn subscribe(self: &Arc<Self>, topic: &str) -> Subscription {
        let (tx, rx) = mpsc::channel(BUS_QUEUE);
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.subs
            .guard()
            .entry(topic.to_string())
            .or_default()
            .push((id, tx));
        Subscription {
            bus: Arc::clone(self),
            topic: topic.to_string(),
            id,
            rx,
        }
    }

    /// Number of live subscribers of a topic (tests).
    pub fn subscribers(&self, topic: &str) -> usize {
        self.subs
            .guard()
            .get(topic)
            .map_or(0, |l| l.iter().filter(|(_, t)| !t.is_closed()).count())
    }
}

pub struct Subscription {
    bus: Arc<EventBus>,
    topic: String,
    id: u64,
    rx: mpsc::Receiver<Value>,
}

impl Subscription {
    pub async fn recv(&mut self) -> Option<Value> {
        self.rx.recv().await
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        if let Some(list) = self.bus.subs.guard().get_mut(&self.topic) {
            list.retain(|(i, _)| *i != self.id);
        }
    }
}

// ── SSE framing ──────────────────────────────────────────────────────────────────────────────────────────
/// `sse(data, event)`: a JSON value (compact separators, ensure_ascii) or a string payload, one `data:` line per
/// line of the payload, then a blank line.
pub fn sse(data: &Value, event: Option<&str>) -> String {
    let payload = match data {
        Value::String(s) => s.clone(),
        other => pyjson::dumps_compact(other),
    };
    sse_text(&payload, event)
}

pub fn sse_text(payload: &str, event: Option<&str>) -> String {
    let head = event.map(|e| format!("event: {e}\n")).unwrap_or_default();
    let lines = py_splitlines(payload);
    let body = if lines.is_empty() {
        "data: ".to_string()
    } else {
        lines
            .iter()
            .map(|l| format!("data: {l}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!("{head}{body}\n\n")
}

/// `str.splitlines()` (the line boundaries Python recognises, without keeping them).
pub fn py_splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let is_break = matches!(
            c,
            '\n' | '\r'
                | '\u{b}'
                | '\u{c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if is_break {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r'
                && let Some(&(j, '\n')) = chars.peek()
            {
                chars.next();
                end = j + 1;
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

// ── secrets in text ──────────────────────────────────────────────────────────────────────────────────────
/// One rule for "this parameter is a secret": the secret store moves these values out of the DB and the audit log
/// masks them (review L8). Use [`is_sensitive_key`]: a pattern that failed to compile is `None`, and every name then
/// counts as a secret (fail closed), never as clean.
static SENSITIVE_KEY: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"(?i)(key|token|secret|password|passwd|authorization|cookie|credential)").ok()
});
static SECRET_VALUE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"\b(hf_[A-Za-z0-9]{8,}|ec_[A-Za-z0-9_\-]{16,}|sk-[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9._\-]{8,})")
        .ok()
});
/// What `scrub` leaves of a text it cannot check (its pattern did not compile): nothing of the text.
pub const REDACTED: &str = "[redacted]";

/// `SECRET_KEY.search(name)`: the parameter or field `name` holds a secret.
pub fn is_sensitive_key(name: &str) -> bool {
    sensitive_with(SENSITIVE_KEY.as_ref(), name)
}

/// `is_sensitive_key` with the pattern given; `None` (a pattern that did not compile) treats every name as secret.
pub fn sensitive_with(re: Option<&Regex>, name: &str) -> bool {
    re.is_none_or(|r| r.is_match(name))
}

/// Mask secret-looking tokens in free text (logs). No truncation, unlike `redact`.
pub fn scrub(text: &str) -> String {
    scrub_with(SECRET_VALUE.as_ref(), text)
}

/// `scrub` with the pattern given; `None` (a pattern that did not compile) masks the whole text (fail closed).
pub fn scrub_with(re: Option<&Regex>, text: &str) -> String {
    match re {
        Some(r) => r.replace_all(text, REDACTED).into_owned(),
        None if text.is_empty() => String::new(),
        None => REDACTED.to_string(),
    }
}

/// Deep copy with sensitive keys masked and secret-looking values scrubbed; long strings truncated.
pub fn redact(v: &Value) -> Value {
    redact_at(v, 0)
}

fn redact_at(v: &Value, depth: usize) -> Value {
    if depth > 8 {
        return Value::String("[truncated]".into());
    }
    match v {
        Value::Object(o) => {
            let mut out = Map::new();
            for (k, x) in o {
                let val = if is_sensitive_key(k) {
                    Value::String("[redacted]".into())
                } else {
                    redact_at(x, depth + 1)
                };
                out.insert(k.clone(), val);
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(
            a.iter()
                .take(100)
                .map(|x| redact_at(x, depth + 1))
                .collect(),
        ),
        Value::String(s) => {
            let s = scrub(s);
            let n = s.chars().count();
            if n <= 500 {
                Value::String(s)
            } else {
                let head: String = s.chars().take(500).collect();
                Value::String(format!("{head}...[{} more chars]", n - 500))
            }
        }
        other => other.clone(),
    }
}

/// Linear-interpolated percentile of `values` (p in 0..=100).
pub fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut s = values.to_vec();
    s.sort_by(f64::total_cmp);
    let k = (s.len() - 1) as f64 * p / 100.0;
    let lo = k.floor();
    let lo_i = float_index(lo);
    let hi_i = (lo_i + 1).min(s.len() - 1);
    Some(s[lo_i] + (s[hi_i] - s[lo_i]) * (k - lo))
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn float_index(x: f64) -> usize {
    x.max(0.0) as usize
}

// ── stream caps ──────────────────────────────────────────────────────────────────────────────────────────
/// Caps concurrent long-lived streams: `limit` in total and `per_key` for any one caller (an API key id).
#[derive(Debug)]
pub struct SseLimiter {
    pub limit: i64,
    pub per_key: i64,
    state: Mutex<(i64, HashMap<String, i64>)>,
}

impl SseLimiter {
    pub fn new(limit: i64, per_key: Option<i64>) -> Self {
        Self {
            limit,
            per_key: per_key.unwrap_or(limit),
            state: Mutex::new((0, HashMap::new())),
        }
    }

    /// How many key buckets are tracked (empty ones are dropped: the map stays small).
    pub fn tracked_keys(&self) -> usize {
        self.state.guard().1.len()
    }

    pub fn active(&self) -> i64 {
        self.state.guard().0
    }

    pub fn key_full(&self, key: &str) -> bool {
        self.state.guard().1.get(key).copied().unwrap_or(0) >= self.per_key
    }

    pub fn acquire(&self, key: &str) -> bool {
        let mut s = self.state.guard();
        if s.0 >= self.limit || s.1.get(key).copied().unwrap_or(0) >= self.per_key {
            return false;
        }
        s.0 += 1;
        *s.1.entry(key.to_string()).or_insert(0) += 1;
        true
    }

    pub fn release(&self, key: &str) {
        let mut s = self.state.guard();
        s.0 = (s.0 - 1).max(0);
        let n = s.1.get(key).copied().unwrap_or(0) - 1;
        if n > 0 {
            s.1.insert(key.to_string(), n);
        } else {
            s.1.remove(key);
        }
    }
}

// ── fnmatch ──────────────────────────────────────────────────────────────────────────────────────────────
/// Python's `fnmatch.fnmatchcase(name, pat)` (case-sensitive; `*`, `?`, `[seq]`, `[!seq]`).
pub fn fnmatchcase(name: &str, pat: &str) -> bool {
    Regex::new(&fnmatch_translate(pat)).is_ok_and(|re| re.is_match(name))
}

/// `fnmatch.translate`, anchored for a whole-string match.
pub fn fnmatch_translate(pat: &str) -> String {
    let chars: Vec<char> = pat.chars().collect();
    let mut out = String::from("(?s)^");
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        i += 1;
        match c {
            '*' => {
                while i < chars.len() && chars[i] == '*' {
                    i += 1;
                }
                out.push_str(".*");
            }
            '?' => out.push('.'),
            '[' => {
                let mut j = i;
                if j < chars.len() && chars[j] == '!' {
                    j += 1;
                }
                if j < chars.len() && chars[j] == ']' {
                    j += 1;
                }
                while j < chars.len() && chars[j] != ']' {
                    j += 1;
                }
                if j >= chars.len() {
                    out.push_str("\\[");
                } else {
                    let mut stuff: String = chars[i..j].iter().collect();
                    i = j + 1;
                    let negate = stuff.starts_with('!');
                    if negate {
                        stuff.remove(0);
                    }
                    let escaped: String = stuff
                        .chars()
                        .map(|ch| match ch {
                            '\\' | '[' | ']' | '^' | '&' | '~' => format!("\\{ch}"),
                            _ => ch.to_string(),
                        })
                        .collect();
                    if stuff.is_empty() {
                        out.push_str(if negate { "." } else { "(?!)" });
                    } else {
                        out.push('[');
                        if negate {
                            out.push('^');
                        }
                        out.push_str(&escaped);
                        out.push(']');
                    }
                }
            }
            other => out.push_str(&regex::escape(&other.to_string())),
        }
    }
    out.push('$');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_redact_masks_keys_scrubs_values_and_truncates() {
        let long = "x".repeat(600);
        let v = redact(
            &json!({"api_key": "s", "zz_access_key": 1, "note": "token hf_abcdefghijk", "long": long,
                               "list": [{"password": "p"}]}),
        );
        assert_eq!(v["api_key"], "[redacted]");
        assert_eq!(v["zz_access_key"], "[redacted]");
        assert_eq!(v["note"], "token [redacted]");
        assert_eq!(v["long"], format!("{}...[100 more chars]", "x".repeat(500)));
        assert_eq!(v["list"][0]["password"], "[redacted]");
    }

    #[test]
    fn test_sse_framing() {
        assert_eq!(
            sse(&json!({"a": 1}), Some("x")),
            "event: x\ndata: {\"a\":1}\n\n"
        );
        assert_eq!(sse(&json!("l1\nl2"), None), "data: l1\ndata: l2\n\n");
        assert_eq!(sse(&json!(""), Some("log")), "event: log\ndata: \n\n");
    }

    #[test]
    fn test_sse_limiter_per_key_and_total() {
        let l = SseLimiter::new(3, Some(2));
        assert!(l.acquire("a") && l.acquire("a"));
        assert!(!l.acquire("a") && l.key_full("a"));
        assert!(l.acquire("b"));
        assert!(!l.acquire("c"));
        l.release("a");
        assert!(l.acquire("c"));
    }

    #[test]
    fn test_fnmatch_like_python() {
        assert!(fnmatchcase(
            "cdn-lfs-us-1.huggingface.co",
            "cdn-lfs*.huggingface.co"
        ));
        assert!(fnmatchcase("x.hf.co", "*.hf.co"));
        assert!(!fnmatchcase("hf.co", "*.hf.co"));
        assert!(fnmatchcase("a1", "a[0-9]"));
        assert!(!fnmatchcase("a1", "a[!0-9]"));
        assert!(fnmatchcase("a.b", "a.b") && !fnmatchcase("axb", "a.b"));
    }

    #[test]
    fn test_percentile() {
        assert_eq!(percentile(&[], 50.0), None);
        assert_eq!(percentile(&[1.0], 95.0), Some(1.0));
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 50.0), Some(2.5));
    }

    #[tokio::test]
    async fn test_bus_drops_when_full_and_unsubscribes() {
        let bus = Arc::new(EventBus::new());
        let mut sub = bus.subscribe("t");
        for i in 0..300 {
            bus.publish("t", json!(i));
        }
        let mut n = 0;
        while let Ok(Some(_)) =
            tokio::time::timeout(std::time::Duration::from_millis(10), sub.recv()).await
        {
            n += 1;
        }
        assert_eq!(n, BUS_QUEUE);
        drop(sub);
        assert_eq!(bus.subscribers("t"), 0);
    }

    // Port of tests/core/test_units.py (names mirror the Python tests).
    #[test]
    fn test_redact_and_scrub() {
        let r = redact(
            &json!({"api_key": "x", "nested": {"HF_TOKEN": "y", "note": "use hf_abcdefgh12345 now"}, "big": "a".repeat(900)}),
        );
        assert_eq!(
            (&r["api_key"], &r["nested"]["HF_TOKEN"]),
            (&json!("[redacted]"), &json!("[redacted]"))
        );
        assert!(
            !r["nested"]["note"]
                .as_str()
                .unwrap()
                .contains("hf_abcdefgh")
        );
        assert!(r["big"].as_str().unwrap().contains("more chars"));
        assert_eq!(
            scrub("Authorization: Bearer abcdefghijkl"),
            "Authorization: [redacted]"
        );
    }

    /// Audit F10: ids, keys and tokens never come from a failed random source. `random_bytes` panics when the OS
    /// source fails (a request is then answered 500 by app.rs's panic containment, a background iteration is logged
    /// and skipped by `panics::guarded`); it used to return zeros.
    #[test]
    fn test_random_bytes_never_come_from_a_failed_rng() {
        let failed = std::panic::catch_unwind(|| {
            random_bytes_from(|_: &mut [u8]| Err(getrandom::Error::UNSUPPORTED), 6)
        });
        assert!(
            failed.is_err(),
            "bytes from a failed RNG: {:?}",
            failed.ok()
        );
        let fixed = random_bytes_from(
            |b: &mut [u8]| {
                b.fill(7);
                Ok(())
            },
            4,
        );
        assert_eq!(fixed, [7, 7, 7, 7]);
        assert_eq!(random_bytes(16).len(), 16);
        let id = new_id("key_");
        assert!(id.starts_with("key_") && id.len() == 16 && id != new_id("key_"));
    }

    /// A pattern that does not compile never lets a value through as clean (hardening finding 7).
    #[test]
    fn test_redaction_fails_closed_on_a_regex_error() {
        assert_eq!(scrub_with(None, "hf_abcdefghijkl and more"), REDACTED);
        assert_eq!(scrub_with(None, ""), "");
        assert!(sensitive_with(None, "harmless_name"));
        let re = Regex::new("x").ok();
        assert!(!sensitive_with(re.as_ref(), "abc") && sensitive_with(re.as_ref(), "xyz"));
        assert!(is_sensitive_key("HF_TOKEN") && !is_sensitive_key("model"));
    }

    /// One panic while a lock is held must not lock every later caller out (hardening finding 3).
    #[test]
    fn test_the_bus_and_the_limiter_survive_a_poisoned_lock() {
        let bus = Arc::new(EventBus::new());
        let l = Arc::new(SseLimiter::new(2, None));
        let (b2, l2) = (Arc::clone(&bus), Arc::clone(&l));
        let _ = std::thread::spawn(move || {
            let _a = b2.subs.lock();
            let _b = l2.state.lock();
            #[allow(clippy::panic)]
            {
                panic!("a holder panics");
            }
        })
        .join();
        assert!(bus.subs.is_poisoned() && l.state.is_poisoned());
        let sub = bus.subscribe("t");
        bus.publish("t", json!(1));
        assert_eq!(bus.subscribers("t"), 1);
        drop(sub);
        assert_eq!(bus.subscribers("t"), 0);
        assert!(l.acquire("k") && l.active() == 1);
        l.release("k");
        assert_eq!(l.active(), 0);
    }

    #[tokio::test]
    async fn test_event_bus_drops_for_slow_subscribers_without_blocking() {
        let bus = Arc::new(EventBus::new());
        let mut sub = bus.subscribe("t");
        let first = tokio::spawn(async move { sub.recv().await });
        tokio::task::yield_now().await;
        for i in 0..1000 {
            bus.publish("t", json!(i)); // must never panic or block, even past queue capacity
        }
        assert_eq!(first.await.unwrap(), Some(json!(0)));
    }
}
