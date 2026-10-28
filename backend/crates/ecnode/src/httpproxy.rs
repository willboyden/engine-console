//! Python: node/src/ecnode/httpproxy.py.
//!
//! http.request: one HTTP/1.1 exchange with an engine on 127.0.0.1, for the hub's chat, benchmarks and probes.
//!
//! Only ports that are a console-owned gateway's published port or an engine endpoint of the current inventory;
//! methods GET and POST; paths under the engine API prefixes of docs/FLEET-PROTOCOL.md §4; hop-by-hop and Host
//! headers from the hub are dropped. Redirects are never followed. The response goes back as one event
//! {status, headers}, then the body as DATA under the hub's credit, then res {status, bytes}.

pub mod client;

use std::collections::HashSet;
use std::sync::{Arc, LazyLock, Mutex};

use ec_proto::pyfmt::fmt_g;
use ec_proto::{BAD_ARGS, HTTP_ERROR, Obj, POLICY_DENIED, TIMEOUT};
use regex::Regex;
use serde_json::{Map, Value, json};

use crate::errors::{HandlerError, OpError, Stream};
use crate::util::py;

pub const PATH_PREFIXES: &[&str] = &[
    "/v1/",
    "/health",
    "/metrics",
    "/version",
    "/api/",
    "/get_model_info",
    "/get_server_info",
    "/server_info",
];
pub const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "expect",
];
pub const DEFAULT_TIMEOUT_S: f64 = 600.0;
pub const MAX_TIMEOUT_S: f64 = 3600.0;
pub const MAX_BODY: i64 = 1 << 30;
pub const CONNECT_S: f64 = 5.0;
/// Monitor mode is read-only: GET anything above, but POST only to inference. Ollama's /api/pull, /api/create,
/// /api/push, /api/copy, /api/delete and /api/blobs change the machine; they need manage mode.
pub const INFERENCE_POSTS: &[&str] = &[
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/api/chat",
    "/api/generate",
    "/api/embed",
    "/api/embeddings",
    "/api/show",
];

static TOKEN: LazyLock<Regex> =
    LazyLock::new(|| crate::policy::full(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}"));
static PATH: LazyLock<Regex> = LazyLock::new(|| crate::policy::full(r"/[\x21-\x7e]{0,2047}"));

fn op(code: &str, msg: impl Into<String>) -> OpError {
    OpError::new(code, msg)
}

/// `urllib.parse.unquote` of ASCII text: %XX escapes to bytes, the result read as UTF-8 (errors replaced).
pub fn unquote(s: &str) -> String {
    if !s.contains('%') {
        return s.to_string();
    }
    let b = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let hex = |c: u8| (c as char).to_digit(16);
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2]))
        {
            out.push(u8::try_from(h * 16 + l).unwrap_or(0));
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The path, if it is one this node forwards.
pub fn check_path(path: &str) -> Result<String, OpError> {
    if !PATH.is_match(path) || path.contains('#') {
        return Err(op(
            BAD_ARGS,
            "the path must start with '/' and hold only printable ASCII (no spaces)",
        ));
    }
    let route = path.split_once('?').map_or(path, |(r, _)| r);
    let decoded = unquote(route);
    if decoded.split('/').any(|s| s == "." || s == "..")
        || decoded.contains('\\')
        || decoded.chars().any(|c| u32::from(c) < 32)
    {
        return Err(op(
            BAD_ARGS,
            "the path may not contain '.' or '..' segments or control characters",
        ));
    }
    if !PATH_PREFIXES.iter().any(|p| route.starts_with(p)) {
        return Err(op(
            POLICY_DENIED,
            format!(
                "path {} is not an engine API path this node forwards ({})",
                py::repr_take(route, 200),
                PATH_PREFIXES.join(", ")
            ),
        ));
    }
    Ok(path.to_string())
}

/// The hub's headers without hop-by-hop ones (and those its Connection header names); names must be tokens and
/// values one line.
pub fn clean_headers(
    headers: Option<&Map<String, Value>>,
) -> Result<Vec<(String, String)>, OpError> {
    let empty = Map::new();
    let headers = headers.unwrap_or(&empty);
    let val = |v: &Value| py::str_of(v);
    let mut named: HashSet<String> = HashSet::new();
    for (k, v) in headers {
        if k.to_lowercase() == "connection" {
            for h in val(v).split(',') {
                named.insert(py::strip(h).to_lowercase());
            }
        }
    }
    let mut out = Vec::new();
    for (k, v) in headers {
        let lk = k.to_lowercase();
        if HOP_BY_HOP.contains(&lk.as_str()) || named.contains(&lk) {
            continue;
        }
        if !TOKEN.is_match(k) {
            return Err(op(
                BAD_ARGS,
                format!(
                    "header name {} is not a valid HTTP token",
                    py::repr_take(k, 60)
                ),
            ));
        }
        let v = val(v);
        if v.contains(['\r', '\n', '\0']) {
            return Err(op(
                BAD_ARGS,
                format!("header {} has a line break in its value", py::repr(k)),
            ));
        }
        out.push((k.clone(), v));
    }
    Ok(out)
}

fn int_arg(args: &Obj, key: &str) -> Option<i64> {
    args.get(key).and_then(Value::as_i64)
}

/// `request(args, stream, allowed_ports, monitor=)`: allowed_ports(refresh) -> engine endpoint ports + console
/// gateway ports (refresh=true re-reads docker). monitor: the node's policy is in monitor mode, so POST is limited
/// to INFERENCE_POSTS.
pub fn request(
    args: &Obj,
    stream: &dyn Stream,
    allowed_ports: &dyn Fn(bool) -> HashSet<u16>,
    monitor: bool,
) -> Result<Obj, HandlerError> {
    let port = int_arg(args, "port").unwrap_or(0);
    if !(1..=65535).contains(&port) {
        return Err(op(BAD_ARGS, "port must be 1-65535").into());
    }
    let port16 = u16::try_from(port).unwrap_or(0);
    if !allowed_ports(false).contains(&port16) && !allowed_ports(true).contains(&port16) {
        return Err(op(
            POLICY_DENIED,
            format!("port {port} is not an engine endpoint or console gateway on this node"),
        )
        .into());
    }
    let method = py::str_of(args.get("method").unwrap_or(&Value::Null)).to_uppercase();
    if method != "GET" && method != "POST" {
        return Err(op(POLICY_DENIED, "only GET and POST are forwarded to engines").into());
    }
    let path = check_path(&py::str_of(args.get("path").unwrap_or(&Value::Null)))?;
    let route = path.split_once('?').map_or(path.as_str(), |(r, _)| r);
    if monitor && method == "POST" && !INFERENCE_POSTS.contains(&route) {
        let mut sorted = INFERENCE_POSTS.to_vec();
        sorted.sort_unstable();
        return Err(op(
            POLICY_DENIED,
            format!(
                "this node is in monitor mode: POST is forwarded only to inference endpoints ({}); other engine \
                 requests need mode \"manage\" in the node's policy",
                sorted.join(", ")
            ),
        )
        .into());
    }
    let headers = clean_headers(args.get("headers").and_then(Value::as_object))?;
    let body_len = int_arg(args, "body_len").unwrap_or(0);
    if !(0..=MAX_BODY).contains(&body_len) {
        return Err(op(BAD_ARGS, "body_len must be 0 to 1 GiB").into());
    }
    if body_len != 0 && method == "GET" {
        return Err(op(BAD_ARGS, "a GET request has no body").into());
    }
    let timeout = args
        .get("timeout_s")
        .and_then(Value::as_f64)
        .filter(|t| *t != 0.0)
        .unwrap_or(DEFAULT_TIMEOUT_S)
        .clamp(1.0, MAX_TIMEOUT_S);

    let slot: Arc<Mutex<Option<std::net::TcpStream>>> = Arc::default();
    {
        let slot = Arc::clone(&slot);
        stream.on_cancel(Box::new(move || {
            if let Some(s) = crate::util::lock(&slot).as_ref() {
                client::shutdown(s);
            }
        }));
    }
    let outcome = exchange(
        port16, &method, &path, &headers, body_len, timeout, stream, &slot,
    );
    match outcome {
        Ok(res) => Ok(res),
        Err(Fail::Handler(e)) => Err(e),
        Err(Fail::Http(e)) => {
            if e.is_timeout() {
                if stream.cancelled() {
                    return Err(HandlerError::Cancelled);
                }
                return Err(op(
                    TIMEOUT,
                    format!(
                        "the engine on port {port} did not answer within {} s",
                        fmt_g(timeout)
                    ),
                )
                .into());
            }
            if e.is_refused() {
                return Err(op(
                    HTTP_ERROR,
                    format!("nothing is listening on 127.0.0.1:{port} (is the engine running?)"),
                )
                .into());
            }
            if stream.cancelled() {
                return Err(HandlerError::Cancelled);
            }
            Err(op(
                HTTP_ERROR,
                format!(
                    "the connection to the engine on port {port} failed ({})",
                    e.class_name()
                ),
            )
            .into())
        }
    }
}

enum Fail {
    Handler(HandlerError),
    Http(client::HttpError),
}

impl From<client::HttpError> for Fail {
    fn from(e: client::HttpError) -> Self {
        Fail::Http(e)
    }
}

impl From<HandlerError> for Fail {
    fn from(e: HandlerError) -> Self {
        Fail::Handler(e)
    }
}

impl From<OpError> for Fail {
    fn from(e: OpError) -> Self {
        Fail::Handler(e.into())
    }
}

#[allow(clippy::too_many_arguments)]
fn exchange(
    port: u16,
    method: &str,
    path: &str,
    headers: &[(String, String)],
    body_len: i64,
    timeout: f64,
    stream: &dyn Stream,
    slot: &Arc<Mutex<Option<std::net::TcpStream>>>,
) -> Result<Obj, Fail> {
    let mut conn = client::Conn::connect(port, CONNECT_S)?;
    *crate::util::lock(slot) = conn.handle().ok();
    conn.set_timeout(timeout)?;
    // a value that is not ISO-8859-1 cannot be sent: http.client raises UnicodeEncodeError, which nothing catches
    if headers
        .iter()
        .any(|(_, v)| v.chars().any(|c| u32::from(c) > 0xff))
    {
        return Err(HandlerError::internal("UnicodeEncodeError").into());
    }
    let mut head: Vec<(String, String)> = vec![("Host".into(), format!("127.0.0.1:{port}"))];
    head.extend(headers.iter().cloned());
    if method == "POST" {
        head.push(("Content-Length".into(), body_len.to_string()));
    }
    conn.send(&client::request_head(method, path, &head))?;
    let body_len = usize::try_from(body_len).unwrap_or(0);
    let mut sent = 0usize;
    while sent < body_len {
        let chunk = stream.read_upload()?;
        if chunk.is_empty() {
            break;
        }
        let take = &chunk[..chunk.len().min(body_len - sent)];
        conn.send(take)?;
        sent += take.len();
        if take.len() < chunk.len() {
            return Err(op(
                BAD_ARGS,
                format!("the hub sent more than body_len ({body_len}) bytes"),
            )
            .into());
        }
    }
    if sent != body_len {
        return Err(op(
            BAD_ARGS,
            format!("the hub sent {sent} of body_len {body_len} bytes"),
        )
        .into());
    }
    if body_len > 0 && !stream.read_upload()?.is_empty() {
        // the body is complete: what follows must be `end`
        return Err(op(
            BAD_ARGS,
            format!("the hub sent more than body_len ({body_len}) bytes"),
        )
        .into());
    }
    let mut resp = conn.get_response(false)?;
    let mut hdrs = Map::new();
    for (k, v) in &resp.headers {
        let lk = k.to_lowercase();
        if (lk != "content-length" && HOP_BY_HOP.contains(&lk.as_str()))
            || hdrs.len() >= 64
            || py::len(k) > 256
        {
            continue;
        }
        let joined = match hdrs.get(&lk).and_then(Value::as_str) {
            Some(prev) => py::take(&format!("{prev}, {v}"), 4096).to_string(),
            None => py::take(v, 4096).to_string(),
        };
        hdrs.insert(lk, Value::String(joined));
    }
    let mut ev = Obj::new();
    ev.insert("status".into(), json!(resp.status));
    ev.insert("headers".into(), Value::Object(hdrs));
    stream.event(ev);
    let mut total: i64 = 0;
    loop {
        stream.check().map_err(HandlerError::from)?;
        let chunk = resp.read1(65536)?;
        if chunk.is_empty() {
            break;
        }
        stream.send_data(&chunk)?;
        total += i64::try_from(chunk.len()).unwrap_or(0);
    }
    let mut res = Obj::new();
    res.insert("status".into(), json!(resp.status));
    res.insert("bytes".into(), json!(total));
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unquote_like_urllib() {
        assert_eq!(unquote("/v1/%2e%2e/admin"), "/v1/../admin");
        assert_eq!(unquote("/a%zz%4"), "/a%zz%4");
        assert_eq!(unquote("/%C3%A9"), "/\u{e9}");
        assert_eq!(unquote("/%ff"), "/\u{fffd}");
    }
}
