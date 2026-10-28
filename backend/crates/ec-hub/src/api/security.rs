//! The security middleware's rules (port of api/security.py): Host allowlist (DNS rebinding), CSRF defences,
//! authentication (ADR 9), RBAC-lite, request body caps, audit of every mutating call.
//! The middleware itself (ordering, metrics, span, headers) is in app.rs `handle`.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use crate::config::Settings;
use crate::services::common::redact;
use crate::services::settings::{Principal, SettingsService};

pub const LOOPBACK: &[&str] = &["127.0.0.1", "::1", "localhost"];
pub const OPEN_PATHS: &[&str] = &["/api/v1/health"];
pub const SAFE: &[&str] = &["GET", "HEAD", "OPTIONS"];
pub const MUTATING: &[&str] = &["POST", "PUT", "PATCH", "DELETE"];
pub const KNOWN_METHODS: &[&str] = &["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"];
pub const CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; \
font-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; object-src 'none'; frame-ancestors 'none'";
/// Added to every response (replacing any the handler set).
pub const SECURITY_HEADERS: &[(&str, &str)] = &[
    ("content-security-policy", CSP),
    ("x-content-type-options", "nosniff"),
    ("referrer-policy", "no-referrer"),
    ("x-frame-options", "DENY"),
    (
        "permissions-policy",
        "camera=(), microphone=(), geolocation=()",
    ),
];
/// audit records metadata only for these: prompts and conversations are private content
pub const CONTENT_PATHS: &[&str] = &[
    "/api/v1/chat/completions",
    "/api/v1/arena",
    "/api/v1/conversations",
    "/api/v1/prompts",
];
pub const CHAT_PATH: &str = "/api/v1/chat/completions";
pub const IMPORT_PATH: &str = "/api/v1/profiles/import";
pub const IMPORT_TYPES: &[&str] = &[
    "application/json",
    "application/yaml",
    "application/x-yaml",
    "text/yaml",
    "text/x-yaml",
];
pub const CSRF_HEADER: &str = "x-engine-console";

/// The ONLY mutating (method, path) pairs a viewer may perform, matched whole.
static VIEWER_ALLOWED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^/api/v1/(chat/completions|fit|conversations|arena/matches|arena/matches/[^/]+/vote)$",
    )
    .unwrap_or_else(|_| unreachable!())
});

pub fn viewer_may(method: &str, path: &str) -> bool {
    method == "POST" && VIEWER_ALLOWED.is_match(path)
}

/// `_guarded`: the API and /metrics need a principal, except the open paths.
pub fn guarded(path: &str) -> bool {
    (path.starts_with("/api/") || path == "/metrics") && !OPEN_PATHS.contains(&path)
}

/// The accepted Host header values: 127.0.0.1/localhost/[::1] and the bind address at the port, plus
/// ENGINE_CONSOLE_ALLOWED_HOSTS, lower-cased.
pub fn allowed_hosts(cfg: &Settings) -> HashSet<String> {
    let p = cfg.port;
    let mut s: HashSet<String> = [
        format!("127.0.0.1:{p}"),
        format!("localhost:{p}"),
        format!("[::1]:{p}"),
        format!("{}:{p}", cfg.host),
    ]
    .into_iter()
    .map(|h| h.to_lowercase())
    .collect();
    s.extend(cfg.allowed_hosts.iter().map(|h| h.to_lowercase()));
    s
}

/// The body cap of a path.
pub fn cap(cfg: &Settings, path: &str) -> u64 {
    if path == CHAT_PATH {
        cfg.chat_body_cap_bytes
    } else if path == IMPORT_PATH {
        cfg.import_body_cap_bytes
    } else {
        cfg.body_cap_bytes
    }
}

/// `urlsplit(url).netloc`
pub fn urlsplit_netloc(url: &str) -> String {
    let rest = match url.split_once(':') {
        Some((scheme, rest))
            if !scheme.is_empty()
                && scheme
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic())
                && scheme
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c)) =>
        {
            rest
        }
        _ => url,
    };
    match rest.strip_prefix("//") {
        Some(r) => r.split(['/', '?', '#']).next().unwrap_or("").to_string(),
        None => String::new(),
    }
}

/// A refusal before authentication: (status, code, detail).
pub type Deny = (u16, &'static str, &'static str);

/// Host / CSRF / size gates that apply before authentication.
pub fn precheck(
    cfg: &Settings,
    hosts: &HashSet<String>,
    method: &str,
    path: &str,
    headers: &HashMap<String, String>,
) -> Option<Deny> {
    let host = headers
        .get("host")
        .map_or(String::new(), |h| h.to_lowercase());
    if !hosts.contains(&host) {
        return Some((421, "misdirected_request", "unrecognised Host header"));
    }
    if !SAFE.contains(&method) {
        if let Some(origin) = headers.get("origin")
            && (origin == "null" || urlsplit_netloc(origin).to_lowercase() != host)
        {
            return Some((403, "cross_origin", "cross-origin requests are not allowed"));
        }
        if !headers.contains_key("authorization")
            && headers.get(CSRF_HEADER).map(String::as_str) != Some("1")
        {
            return Some((
                403,
                "csrf_header_required",
                "send the X-Engine-Console: 1 header on non-GET requests",
            ));
        }
        if path == IMPORT_PATH {
            let ct = headers.get("content-type").map_or("", String::as_str);
            let main = ct.split(';').next().unwrap_or("").trim().to_lowercase();
            if !IMPORT_TYPES.contains(&main.as_str()) {
                return Some((
                    415,
                    "unsupported_media_type",
                    "profile import needs a JSON or YAML content-type",
                ));
            }
        }
        if let Some(cl) = headers.get("content-length")
            && !cl.is_empty()
            && cl.chars().all(|c| c.is_ascii_digit())
            && cl
                .parse::<u128>()
                .is_ok_and(|n| n > u128::from(cap(cfg, path)))
        {
            return Some((413, "payload_too_large", "request body too large"));
        }
    }
    None
}

/// The principal of a request, or None (401): a valid bearer key, or a trusted loopback caller.
pub fn principal(
    settings: &SettingsService,
    trust_loopback: bool,
    client_host: &str,
    headers: &HashMap<String, String>,
) -> Option<Principal> {
    let auth = headers.get("authorization").map_or("", String::as_str);
    if !auth.is_empty() {
        let (scheme, tok) = auth.split_once(' ').unwrap_or((auth, ""));
        let tok = tok.trim();
        if scheme.to_lowercase() == "bearer" && !tok.is_empty() {
            return settings.authenticate(tok).ok().flatten();
        }
        return None;
    }
    if trust_loopback && LOOPBACK.contains(&client_host) && !headers.contains_key("x-forwarded-for")
    {
        return Some(Principal::new("loopback", "admin", None));
    }
    None
}

/// The audit `params` of a mutating call: the query (redacted), and the body as metadata only for content paths,
/// redacted JSON for other JSON bodies, or a size note.
pub fn audit_params(
    path: &str,
    raw_query: &str,
    headers: &HashMap<String, String>,
    raw: &[u8],
    size: usize,
) -> Value {
    let mut params = Map::new();
    if !raw_query.is_empty() {
        let q = crate::api::validate::Query::parse(raw_query);
        let mut m = Map::new();
        for (k, v) in q.0 {
            m.insert(k, Value::String(v)); // dict(parse_qsl(...)): the last value of a repeated name wins
        }
        params.insert("query".into(), redact(&Value::Object(m)));
    }
    if CONTENT_PATHS.iter().any(|p| path.starts_with(p)) {
        let mut meta = Map::new();
        meta.insert("bytes".into(), Value::from(size));
        if !raw.is_empty()
            && let Ok(Value::Object(doc)) = serde_json::from_slice::<Value>(raw)
        {
            let mut fields: Vec<String> = doc.keys().cloned().collect();
            fields.sort();
            fields.truncate(30);
            meta.insert("fields".into(), Value::from(fields));
            if let Some(Value::Array(msgs)) = doc.get("messages") {
                meta.insert("messages".into(), Value::from(msgs.len()));
            }
        }
        params.insert("body".into(), Value::Object(meta));
    } else if !raw.is_empty()
        && headers
            .get("content-type")
            .is_some_and(|ct| ct.contains("json"))
    {
        let body = match serde_json::from_slice::<Value>(raw) {
            Ok(v) => redact(&v),
            Err(_) => Value::String("[unparseable json]".into()),
        };
        params.insert("body".into(), body);
    } else if !raw.is_empty() {
        params.insert(
            "body".into(),
            Value::String(format!("[{size} bytes, non-json]")),
        );
    }
    Value::Object(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn test_viewer_allowlist_is_exact() {
        assert!(viewer_may("POST", "/api/v1/fit"));
        assert!(viewer_may("POST", "/api/v1/arena/matches/m1/vote"));
        assert!(!viewer_may("DELETE", "/api/v1/conversations"));
        assert!(!viewer_may("POST", "/api/v1/conversations/x"));
        assert!(!viewer_may("POST", "/api/v1/fitx"));
    }

    #[test]
    fn test_precheck_order_and_gates() {
        let cfg = Settings::defaults();
        let hosts = allowed_hosts(&cfg);
        let ok = h(&[("host", "127.0.0.1:8791")]);
        assert_eq!(precheck(&cfg, &hosts, "GET", "/api/v1/hosts", &ok), None);
        assert_eq!(
            precheck(
                &cfg,
                &hosts,
                "GET",
                "/",
                &h(&[("host", "evil.example:8791")])
            )
            .map(|d| d.0),
            Some(421)
        );
        let x = h(&[
            ("host", "127.0.0.1:8791"),
            ("origin", "http://evil.example"),
        ]);
        assert_eq!(
            precheck(&cfg, &hosts, "POST", "/api/v1/keys", &x).map(|d| d.1),
            Some("cross_origin")
        );
        assert_eq!(
            precheck(&cfg, &hosts, "POST", "/api/v1/keys", &ok).map(|d| d.1),
            Some("csrf_header_required")
        );
        let imp = h(&[
            ("host", "127.0.0.1:8791"),
            ("x-engine-console", "1"),
            ("content-type", "text/plain"),
        ]);
        assert_eq!(
            precheck(&cfg, &hosts, "POST", IMPORT_PATH, &imp).map(|d| d.0),
            Some(415)
        );
        let big = h(&[
            ("host", "127.0.0.1:8791"),
            ("authorization", "Bearer x"),
            ("content-length", "99999999"),
        ]);
        assert_eq!(
            precheck(&cfg, &hosts, "POST", "/api/v1/keys", &big).map(|d| d.0),
            Some(413)
        );
        let same = h(&[
            ("host", "127.0.0.1:8791"),
            ("origin", "http://127.0.0.1:8791"),
            ("x-engine-console", "1"),
        ]);
        assert_eq!(precheck(&cfg, &hosts, "POST", "/api/v1/keys", &same), None);
        assert_eq!(
            precheck(
                &cfg,
                &hosts,
                "POST",
                "/x",
                &h(&[
                    ("host", "127.0.0.1:8791"),
                    ("origin", "null"),
                    ("x-engine-console", "1")
                ])
            )
            .map(|d| d.0),
            Some(403)
        );
    }

    #[test]
    fn test_urlsplit_netloc() {
        assert_eq!(urlsplit_netloc("http://localhost:8791/x"), "localhost:8791");
        assert_eq!(urlsplit_netloc("localhost:8791"), "");
        assert_eq!(urlsplit_netloc("//h:1"), "h:1");
    }

    #[test]
    fn test_audit_params_never_stores_chat_text() {
        let p = audit_params(
            "/api/v1/chat/completions",
            "",
            &h(&[("content-type", "application/json")]),
            br#"{"messages":[{"role":"user","content":"secret plan"}],"model":"m"}"#,
            70,
        );
        assert_eq!(
            p,
            serde_json::json!({"body": {"bytes": 70, "fields": ["messages", "model"], "messages": 1}})
        );
        let q = audit_params("/api/v1/keys", "api_key=s&x=1", &h(&[]), b"", 0);
        assert_eq!(
            q,
            serde_json::json!({"query": {"api_key": "[redacted]", "x": "1"}})
        );
    }
}
