//! ecnode wire protocol, version 1: the hub (Engine Console) <-> one node (ecnode) per machine.
//!
//! A faithful port of the Python `node/src/ecnode/proto.py` (tag python-final), so 0.4 and 0.5 hubs and nodes talk to
//! each other; backend/PORTING.md section "Wire protocol" is the prose. Compatibility with the Python peers is at the
//! level of frame bytes (header, kinds, limits, DATA layout) and JSON *values* (key order and float spelling may
//! differ, both sides parse values).
//!
//! Frames: `u32 LENGTH (big-endian) | u8 KIND | BODY`. KIND_JSON: one UTF-8 JSON object of at most MAX_JSON bytes.
//! KIND_DATA: `u32 STREAM | payload` (payload at most MAX_DATA bytes). The node first writes MAGIC on a line of its
//! own; the hub skips up to MAX_PREAMBLE bytes of shell noise before it.

pub mod clean;
pub mod msg;
pub mod pyfmt;
pub mod sanitize;
pub mod specs;

use std::io::{BufRead, Read};

pub use clean::{Spec, clean, clean_json};
pub use sanitize::{
    clean_args, missing_args, overlong_args, sanitize_event, sanitize_push, sanitize_res,
};
use serde_json::{Map, Value};

pub type Obj = Map<String, Value>;

pub const PROTO: i64 = 1;
pub const MAGIC: &[u8] = b"ecnode-proto 1\n";
pub const MAX_PREAMBLE: usize = 64 * 1024;
pub const KIND_JSON: u8 = 1;
pub const KIND_DATA: u8 = 2;
pub const MAX_JSON: usize = 1024 * 1024;
pub const MAX_DATA: usize = 1024 * 1024;
pub const WINDOW: u64 = 8 * 1024 * 1024;
pub const MAX_ID: i64 = 2_147_483_647; // 2**31 - 1
pub const MAX_INFLIGHT: usize = 64;

pub const SAMPLE_S: f64 = 2.0;
pub const DISKS_S: f64 = 30.0;
pub const INVENTORY_S: f64 = 30.0;
pub const PING_S: f64 = 10.0;
pub const SILENT_S: f64 = 30.0;
pub const HUB_SILENCE_S: f64 = 30.0;
pub const HELLO_TIMEOUT_S: f64 = 25.0;

pub const MAX_METRICS_TEXT: usize = 512 * 1024;
pub const MAX_LOG_TEXT: usize = 768 * 1024;
pub const MAX_FILE_CONTENT: usize = 128 * 1024;

// res.error.code values
pub const BAD_ARGS: &str = "bad_args";
pub const UNKNOWN_OP: &str = "unknown_op";
pub const MONITOR_ONLY: &str = "monitor_only";
pub const POLICY_DENIED: &str = "policy_denied";
pub const NOT_FOUND: &str = "not_found";
pub const NOT_OWNED: &str = "not_owned";
pub const CONFLICT: &str = "conflict";
pub const IN_USE: &str = "in_use";
pub const PORT_IN_USE: &str = "port_in_use";
pub const NO_SPACE: &str = "no_space";
pub const HASH_MISMATCH: &str = "hash_mismatch";
pub const SIZE_MISMATCH: &str = "size_mismatch";
pub const IO_ERROR: &str = "io_error";
pub const DOCKER_ERROR: &str = "docker_error";
pub const HTTP_ERROR: &str = "http_error";
pub const TIMEOUT: &str = "timeout";
pub const CANCELLED: &str = "cancelled";
pub const BUSY: &str = "busy";
pub const UNSUPPORTED: &str = "unsupported";
pub const INTERNAL: &str = "internal";

/// The peer broke the protocol: the link is dropped. `Truncated` is the far side dying mid-frame (the hub then
/// explains the drop from the process exit status rather than from the framing).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProtoError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Truncated(String),
}

impl ProtoError {
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::Invalid(msg.into())
    }
}

/// One decoded frame.
#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    Json(Obj),
    Data { stream: u32, data: Vec<u8> },
}

/// `isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= MAX_ID`
pub fn valid_id(v: &Value) -> bool {
    v.as_i64().is_some_and(valid_id_i64) && !v.is_f64()
}

pub fn valid_id_i64(v: i64) -> bool {
    (1..=MAX_ID).contains(&v)
}

// ── encoding ─────────────────────────────────────────────────────────────────────────────────────────────
/// One JSON frame. serde_json cannot hold NaN/Infinity, so `allow_nan=False` is implicit.
pub fn encode_json(obj: &Obj) -> Result<Vec<u8>, ProtoError> {
    let body =
        serde_json::to_vec(obj).map_err(|e| ProtoError::invalid(format!("cannot encode: {e}")))?;
    if body.len() > MAX_JSON {
        return Err(ProtoError::invalid(format!(
            "message of {} bytes is over MAX_JSON ({MAX_JSON})",
            body.len()
        )));
    }
    let mut out = Vec::with_capacity(body.len() + 5);
    out.extend_from_slice(&u32_len(body.len() + 1).to_be_bytes());
    out.push(KIND_JSON);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Like [`encode_json`] for any JSON value that is an object (`ProtoError` otherwise).
pub fn encode_value(v: &Value) -> Result<Vec<u8>, ProtoError> {
    match v {
        Value::Object(o) => encode_json(o),
        _ => Err(ProtoError::invalid("message must be a JSON object")),
    }
}

pub fn encode_data(stream: i64, payload: &[u8]) -> Result<Vec<u8>, ProtoError> {
    if !valid_id_i64(stream) {
        return Err(ProtoError::invalid("stream id out of range"));
    }
    if payload.len() > MAX_DATA {
        return Err(ProtoError::invalid(format!(
            "data payload of {} bytes is over MAX_DATA ({MAX_DATA})",
            payload.len()
        )));
    }
    let mut out = Vec::with_capacity(payload.len() + 9);
    out.extend_from_slice(&u32_len(payload.len() + 5).to_be_bytes());
    out.push(KIND_DATA);
    out.extend_from_slice(&u32::try_from(stream).unwrap_or(0).to_be_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

fn u32_len(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

// ── decoding ─────────────────────────────────────────────────────────────────────────────────────────────
/// One JSON object. NaN, Infinity and numbers that overflow to infinity are refused (serde_json refuses all of
/// them as invalid JSON); so are non-objects, invalid UTF-8 and nesting deeper than serde_json's limit.
pub fn decode_json(body: &[u8]) -> Result<Obj, ProtoError> {
    let text = std::str::from_utf8(body)
        .map_err(|e| ProtoError::invalid(format!("not JSON: {}", trunc(&e.to_string(), 120))))?;
    let v: Value = serde_json::from_str(text)
        .map_err(|e| ProtoError::invalid(format!("not JSON: {}", trunc(&e.to_string(), 120))))?;
    match v {
        Value::Object(o) => Ok(o),
        _ => Err(ProtoError::invalid("message must be a JSON object")),
    }
}

fn trunc(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// (kind, body length) from the 5 header bytes.
pub fn parse_header(hdr: &[u8; 5]) -> Result<(u8, usize), ProtoError> {
    let length = i64::from(u32::from_be_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]));
    let kind = hdr[4];
    let body = length - 1;
    let ok = match kind {
        KIND_JSON => (2..=MAX_JSON as i64).contains(&body),
        KIND_DATA => (4..=(4 + MAX_DATA) as i64).contains(&body),
        _ => return Err(ProtoError::invalid(format!("unknown frame kind {kind}"))),
    };
    if !ok {
        return Err(ProtoError::invalid(format!(
            "frame length {length} out of range for kind {kind}"
        )));
    }
    Ok((kind, usize::try_from(body).unwrap_or(0)))
}

pub fn decode_body(kind: u8, body: &[u8]) -> Result<Frame, ProtoError> {
    if kind == KIND_JSON {
        return Ok(Frame::Json(decode_json(body)?));
    }
    if body.len() < 4 {
        return Err(ProtoError::invalid("data frame without a stream id"));
    }
    let stream = u32::from_be_bytes([body[0], body[1], body[2], body[3]]);
    if !valid_id_i64(i64::from(stream)) {
        return Err(ProtoError::invalid(
            "data frame with an out-of-range stream id",
        ));
    }
    Ok(Frame::Data {
        stream,
        data: body[4..].to_vec(),
    })
}

fn read_exact_sync<R: Read>(r: &mut R, n: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    let mut got = 0;
    while got < n {
        match r.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(k) => got += k,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    buf.truncate(got);
    Ok(buf)
}

/// Errors of the blocking reader: a broken protocol, or the stream itself failing.
#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    #[error(transparent)]
    Proto(#[from] ProtoError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Next frame from a blocking stream; `None` on a clean end of stream between frames.
pub fn read_frame_sync<R: Read>(r: &mut R) -> Result<Option<Frame>, ReadError> {
    Ok(read_frame_sized_sync(r)?.map(|(f, _)| f))
}

/// Like [`read_frame_sync`], plus the frame's size on the wire (5 + body).
pub fn read_frame_sized_sync<R: Read>(r: &mut R) -> Result<Option<(Frame, usize)>, ReadError> {
    let hdr = read_exact_sync(r, 5)?;
    if hdr.is_empty() {
        return Ok(None);
    }
    let Ok(hdr) = <[u8; 5]>::try_from(hdr.as_slice()) else {
        return Err(ProtoError::Truncated("stream ended inside a frame header".into()).into());
    };
    let (kind, n) = parse_header(&hdr)?;
    let body = read_exact_sync(r, n)?;
    if body.len() < n {
        return Err(ProtoError::Truncated("stream ended inside a frame".into()).into());
    }
    Ok(Some((decode_body(kind, &body)?, 5 + n)))
}

/// Skip shell noise up to and including the MAGIC line; returns how many noise bytes were skipped.
pub fn find_magic_sync<R: BufRead>(r: &mut R) -> Result<usize, ReadError> {
    let mut skipped = 0usize;
    loop {
        let line = read_line_capped(r, MAX_PREAMBLE + 1)?;
        if line.is_empty() {
            return Err(ProtoError::invalid(
                "the far side closed before speaking the ecnode protocol",
            )
            .into());
        }
        if line.ends_with(MAGIC) {
            return Ok(skipped + line.len() - MAGIC.len());
        }
        skipped += line.len();
        if skipped > MAX_PREAMBLE {
            return Err(ProtoError::invalid(
                "the far side printed something other than the ecnode protocol",
            )
            .into());
        }
    }
}

/// Python's `readline(limit)`: up to and including b"\n", at most `limit` bytes.
fn read_line_capped<R: BufRead>(r: &mut R, limit: usize) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::new();
    while out.len() < limit {
        let (done, used) = {
            let buf = match r.fill_buf() {
                Ok(b) => b,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            };
            if buf.is_empty() {
                return Ok(out);
            }
            let room = limit - out.len();
            let take = buf.len().min(room);
            match buf[..take].iter().position(|&b| b == b'\n') {
                Some(i) => {
                    out.extend_from_slice(&buf[..=i]);
                    (true, i + 1)
                }
                None => {
                    out.extend_from_slice(&buf[..take]);
                    (false, take)
                }
            }
        };
        r.consume(used);
        if done {
            break;
        }
    }
    Ok(out)
}

#[cfg(feature = "tokio")]
mod asyncio;
#[cfg(feature = "tokio")]
pub use asyncio::{find_magic_async, read_frame_async};
