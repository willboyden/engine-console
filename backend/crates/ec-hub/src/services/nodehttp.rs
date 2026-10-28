//! Python: backend/src/engine_console/services/nodehttp.py
//! (tests/core/test_nodehttp_chat_bench.py, shared with the chat and bench services).
//!
//! An engine's HTTP API, reached through its host's node (`http.request`), for chat, benchmarks and health probes.
//!
//! The node connects to 127.0.0.1:<port> on ITS machine and relays: one event {status, headers}, the body as DATA
//! frames, then a result. A request body is uploaded with `write()` + `end()` first. Nothing here can reach an address
//! other than a loopback port the node's policy allows (the node refuses anything else); a remote engine therefore
//! needs no network exposure at all.
//!
//! Everything coming back is untrusted: bodies are read under a cap, lines are decoded incrementally (linear,
//! MAX_LINE / MAX_STREAM caps; security review M2).

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use serde_json::{Value, json};

use crate::domain::errors::{AppResult, NodeError, Obj, Problem, upstream_error};
use crate::domain::fleet::{FleetPort, NodeStream, StreamItem};
use crate::services::nodeerr::{NodeCaller, SLACK_S};

/// well under the protocol's 1 MiB DATA frame
pub const UPLOAD_CHUNK: usize = 256 * 1024;
/// a non-streaming engine answer larger than this is refused, not buffered
pub const MAX_BUFFERED: usize = 16 * 1024 * 1024;
/// characters in one streamed line
pub const MAX_LINE: usize = 1024 * 1024;
/// bytes in one streamed answer
pub const MAX_STREAM: usize = 256 * 1024 * 1024;
/// nodehttp.py `JSON_HEADERS`
pub const JSON_HEADERS: &[(&str, &str)] = &[
    ("content-type", "application/json"),
    ("accept", "application/json"),
];

fn long_line() -> Problem {
    upstream_error(format!(
        "the engine sent a line longer than {} KiB; the stream was cut off",
        MAX_LINE / 1024
    ))
}

/// Request options. `timeout_s` None: 600 s for `open`, 30 s for `request`; `max_bytes` None: MAX_BUFFERED.
#[derive(Debug, Clone, Default)]
pub struct HttpOpts {
    pub body: Option<Vec<u8>>,
    /// header name -> value (lower-cased by the node on the way back, not on the way in)
    pub headers: Vec<(String, String)>,
    pub timeout_s: Option<f64>,
    pub max_bytes: Option<usize>,
}

/// One buffered request/response.
#[derive(Debug, Clone, PartialEq)]
pub struct HttpResult {
    pub status: i64,
    /// lower-cased names
    pub headers: Vec<(String, String)>,
    pub body: Bytes,
}

impl HttpResult {
    /// A header by its lower-case name.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// Incremental UTF-8 decoding with replacement (Python's `codecs.getincrementaldecoder("utf-8")("replace")`): an
/// incomplete sequence at the end of a chunk waits for the next one; an invalid one becomes U+FFFD.
#[derive(Debug, Default)]
struct Utf8Decoder {
    left: Vec<u8>,
}

impl Utf8Decoder {
    fn decode(&mut self, chunk: &[u8], last: bool) -> String {
        let mut buf = std::mem::take(&mut self.left);
        buf.extend_from_slice(chunk);
        let mut out = String::new();
        let mut rest: &[u8] = &buf;
        loop {
            match std::str::from_utf8(rest) {
                Ok(s) => {
                    out.push_str(s);
                    break;
                }
                Err(e) => {
                    let (ok, bad) = rest.split_at(e.valid_up_to());
                    out.push_str(std::str::from_utf8(ok).unwrap_or_default());
                    if let Some(n) = e.error_len() {
                        out.push('\u{fffd}');
                        rest = &bad[n..];
                    } else {
                        // an incomplete sequence at the end
                        if last {
                            out.push('\u{fffd}');
                        } else {
                            self.left = bad.to_vec();
                        }
                        break;
                    }
                }
            }
        }
        out
    }
}

/// An open engine response. Read it with `chunk()` / `line()` / `read()`; dropping it before the end cancels the
/// node request (a browser that went away mid-stream).
pub struct HttpStream {
    pub status: i64,
    pub headers: Vec<(String, String)>,
    stream: Option<Box<dyn NodeStream>>,
    caller: Option<NodeCaller>,
    /// the response is complete (its result arrived): nothing to cancel
    done: bool,
    // line splitting
    decoder: Utf8Decoder,
    pending: String,
    plen: usize,
    total: usize,
    after_cr: bool,
    ready: VecDeque<String>,
    failed: Option<Problem>,
    lines_done: bool,
}

impl std::fmt::Debug for HttpStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpStream")
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}

/// `str.splitlines`-like split on `\r\n`, `\r` or `\n` (nodehttp.py `_EOL`), keeping the empty pieces.
fn split_eol(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let b = text.as_bytes();
    let mut start = 0;
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'\r' => {
                out.push(&text[start..i]);
                i += if b.get(i + 1) == Some(&b'\n') { 2 } else { 1 };
                start = i;
            }
            b'\n' => {
                out.push(&text[start..i]);
                i += 1;
                start = i;
            }
            _ => i += 1,
        }
    }
    out.push(&text[start..]);
    out
}

impl HttpStream {
    fn new(
        status: i64,
        headers: Vec<(String, String)>,
        stream: Box<dyn NodeStream>,
        caller: NodeCaller,
    ) -> Self {
        Self {
            status,
            headers,
            stream: Some(stream),
            caller: Some(caller),
            done: false,
            decoder: Utf8Decoder::default(),
            pending: String::new(),
            plen: 0,
            total: 0,
            after_cr: false,
            ready: VecDeque::new(),
            failed: None,
            lines_done: false,
        }
    }

    /// A header by its lower-case name.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// Next body chunk; None at the end (the node's error, if it ended with one, as a labelled problem).
    pub async fn chunk(&mut self) -> AppResult<Option<Bytes>> {
        let Some(s) = self.stream.as_mut() else {
            return Ok(None);
        };
        loop {
            match s.next().await {
                Some(StreamItem::Data(d)) if !d.is_empty() => return Ok(Some(d)),
                Some(_) => {}
                None => {
                    let res = s.result().await;
                    return match res {
                        Ok(_) => {
                            self.done = true;
                            self.stream = None;
                            Ok(None)
                        }
                        Err(e) => {
                            let p = self.problem(&e);
                            self.close().await;
                            Err(p.into())
                        }
                    };
                }
            }
        }
    }

    fn problem(&self, e: &NodeError) -> Problem {
        match &self.caller {
            Some(c) => c.problem(e),
            None => crate::services::nodeerr::problem_from_node(e, None),
        }
    }

    /// Split one decoded piece of text into lines (nodehttp.py `lines()`): linear in the stream, a partial line is
    /// kept and never re-scanned; a line over MAX_LINE characters is an error, after the lines before it.
    fn feed(&mut self, text: &str) {
        let mut text = text;
        if text.is_empty() {
            return;
        }
        if self.after_cr && text.starts_with('\n') {
            text = &text[1..];
        }
        self.after_cr = text.ends_with('\r');
        let segs = split_eol(text);
        if segs.len() > 1 {
            let first = segs[0];
            if self.plen + first.chars().count() > MAX_LINE {
                self.failed = Some(long_line());
                return;
            }
            let mut line = std::mem::take(&mut self.pending);
            line.push_str(first);
            self.ready.push_back(line);
            for seg in &segs[1..segs.len() - 1] {
                if seg.chars().count() > MAX_LINE {
                    self.failed = Some(long_line());
                    return;
                }
                self.ready.push_back((*seg).to_string());
            }
            self.plen = 0;
        }
        let last = segs[segs.len() - 1];
        self.pending.push_str(last);
        self.plen += last.chars().count();
        if self.plen > MAX_LINE {
            self.failed = Some(long_line());
        }
    }

    /// Next text line (no terminator; \n, \r\n or \r), decoded as UTF-8 with replacement.
    pub async fn line(&mut self) -> AppResult<Option<String>> {
        loop {
            if let Some(l) = self.ready.pop_front() {
                return Ok(Some(l));
            }
            if let Some(p) = self.failed.take() {
                self.lines_done = true;
                self.close().await;
                return Err(p.into());
            }
            if self.lines_done {
                return Ok(None);
            }
            match self.chunk().await {
                Err(e) => {
                    self.lines_done = true;
                    return Err(e);
                }
                Ok(None) => {
                    self.lines_done = true;
                    let mut tail = std::mem::take(&mut self.pending);
                    tail.push_str(&self.decoder.decode(b"", true));
                    if !tail.is_empty() {
                        return Ok(Some(tail));
                    }
                    return Ok(None);
                }
                Ok(Some(chunk)) => {
                    self.total += chunk.len();
                    if self.total > MAX_STREAM {
                        self.failed = Some(upstream_error(format!(
                            "the engine's stream passed {} MiB; it was cut off",
                            MAX_STREAM / (1024 * 1024)
                        )));
                        continue;
                    }
                    let text = self.decoder.decode(&chunk, false);
                    self.feed(&text);
                }
            }
        }
    }

    /// The whole body, refusing more than `limit` bytes (502 upstream_error).
    pub async fn read(mut self, limit: usize) -> AppResult<Bytes> {
        let mut out = Vec::new();
        loop {
            match self.chunk().await {
                Ok(Some(c)) => {
                    out.extend_from_slice(&c);
                    if out.len() > limit {
                        // over the cap: tell the node to stop sending
                        self.close().await;
                        return Err(upstream_error(format!(
                            "the engine's answer is larger than {} KiB; refusing to buffer it",
                            limit / 1024
                        ))
                        .into());
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    self.close().await;
                    return Err(e);
                }
            }
        }
        self.close().await;
        Ok(Bytes::from(out))
    }

    /// Cancel the node request unless the response is complete (idempotent).
    pub async fn close(&mut self) {
        if let Some(mut s) = self.stream.take()
            && !self.done
        {
            s.cancel().await;
        }
        self.done = true;
    }
}

#[derive(Debug, Clone)]
pub struct NodeHttp {
    caller: NodeCaller,
}

fn secs(s: f64) -> Duration {
    Duration::from_secs_f64(if s.is_finite() { s.max(0.0) } else { 0.0 })
}

impl NodeHttp {
    pub fn new(fleet: Arc<dyn FleetPort>) -> Self {
        Self {
            caller: NodeCaller::new(fleet),
        }
    }

    /// Send the request and wait for the response head (status + headers); the body is left to read.
    pub async fn open(
        &self,
        host_id: &str,
        port: i64,
        method: &str,
        path: &str,
        opts: HttpOpts,
    ) -> AppResult<HttpStream> {
        let timeout_s = opts.timeout_s.unwrap_or(600.0);
        let mut args = Obj::new();
        args.insert("port".into(), json!(port));
        args.insert("method".into(), json!(method));
        args.insert("path".into(), json!(path));
        args.insert("timeout_s".into(), json!(timeout_s));
        if !opts.headers.is_empty() {
            let h: Obj = opts
                .headers
                .iter()
                .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                .collect();
            args.insert("headers".into(), Value::Object(h));
        }
        let body = opts.body.filter(|b| !b.is_empty());
        if let Some(b) = &body {
            args.insert("body_len".into(), json!(b.len()));
        }
        let mut stream = self.caller.open(host_id, "http.request", Some(args))?;
        let started = tokio::time::timeout(
            secs(timeout_s + SLACK_S),
            Self::start(&mut stream, host_id, body.as_deref()),
        )
        .await;
        match started {
            Ok(Ok((status, headers))) => Ok(HttpStream::new(
                status,
                headers,
                stream,
                self.caller.clone(),
            )),
            Ok(Err(e)) => {
                stream.cancel().await;
                Err(self.caller.problem(&e).into())
            }
            Err(_) => {
                stream.cancel().await;
                Err(self
                    .caller
                    .problem(&NodeError::new(
                        host_id,
                        "timeout",
                        format!("the engine did not answer in {timeout_s:.0}s"),
                    ))
                    .into())
            }
        }
    }

    async fn start(
        stream: &mut Box<dyn NodeStream>,
        host_id: &str,
        body: Option<&[u8]>,
    ) -> Result<(i64, Vec<(String, String)>), NodeError> {
        if let Some(body) = body {
            for chunk in body.chunks(UPLOAD_CHUNK) {
                stream.write(chunk).await?;
            }
            stream.end().await?;
        }
        loop {
            let Some(item) = stream.next().await else {
                // raises the node's reason when it refused; an answer without a head is a fault
                stream.result().await?;
                return Err(NodeError::new(
                    host_id,
                    "http_error",
                    "the node answered without an HTTP status",
                ));
            };
            let StreamItem::Event(ev) = item else {
                continue;
            };
            let Some(status) = ev
                .get("status")
                .filter(|v| v.is_i64() || v.is_u64())
                .and_then(Value::as_i64)
            else {
                continue;
            };
            let mut headers: Vec<(String, String)> = Vec::new();
            if let Some(Value::Object(h)) = ev.get("headers") {
                for (k, v) in h {
                    let k = k.to_lowercase();
                    let v = ec_proto::pyfmt::str_value(v);
                    match headers.iter_mut().find(|(kk, _)| *kk == k) {
                        Some(slot) => slot.1 = v,
                        None => headers.push((k, v)),
                    }
                }
            }
            return Ok((status, headers));
        }
    }

    /// One buffered request/response (health probes, non-streaming chat).
    pub async fn request(
        &self,
        host_id: &str,
        port: i64,
        method: &str,
        path: &str,
        opts: HttpOpts,
    ) -> AppResult<HttpResult> {
        let max = opts.max_bytes.unwrap_or(MAX_BUFFERED);
        let opts = HttpOpts {
            timeout_s: Some(opts.timeout_s.unwrap_or(30.0)),
            ..opts
        };
        let resp = self.open(host_id, port, method, path, opts).await?;
        let (status, headers) = (resp.status, resp.headers.clone());
        let body = resp.read(max).await?;
        Ok(HttpResult {
            status,
            headers,
            body,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_eol_matches_the_python_regex() {
        assert_eq!(split_eol("a\r\nb\rc\nd"), ["a", "b", "c", "d"]);
        assert_eq!(split_eol("a\n"), ["a", ""]);
        assert_eq!(split_eol(""), [""]);
        assert_eq!(split_eol("\r"), ["", ""]);
    }

    #[test]
    fn test_utf8_decoder_waits_for_split_sequences_and_replaces_bad_ones() {
        let mut d = Utf8Decoder::default();
        let e = "é".as_bytes();
        assert_eq!(d.decode(&e[..1], false), "");
        assert_eq!(d.decode(&e[1..], false), "é");
        assert_eq!(d.decode(b"a\xffb", false), "a\u{fffd}b");
        assert_eq!(d.decode(b"\xe2\x82", false), "");
        assert_eq!(d.decode(b"", true), "\u{fffd}");
    }
}
