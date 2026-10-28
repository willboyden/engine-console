//! Request context, handler failures and response helpers (the FastAPI/Starlette plumbing).
//!
//! Problem bodies are RFC 7807 problem+json with a stable `code` (api/errors.py `problem()`):
//! `{"type": "urn:engine-console:<code>", "title": <code with spaces>, "status", "code", "detail", ...extra}`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::body::Body as AxBody;
use bytes::{Bytes, BytesMut};
use http::{HeaderValue, Response, StatusCode, header};
use http_body_util::BodyExt;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::api::validate::{Body, Issue, Query};
use crate::container::Container;
use crate::domain::errors::{AppError, Problem};
use crate::services::nodeerr::to_problem;
use crate::services::settings::Principal;
use crate::sync::Recover as _;

pub type Resp = Response<AxBody>;
pub const PROBLEM_CT: &str = "application/problem+json";
/// audit records at most this much of a request body
pub const BODY_CAP_AUDIT: usize = 64 * 1024;

/// Why a handler did not produce its normal response.
#[derive(Debug)]
pub enum Fail {
    App(AppError),
    /// 422 validation_error with pydantic's error list
    Validation(Vec<Issue>),
    /// the body passed its cap while being read (413)
    TooLarge,
    Raw(Resp),
}

impl From<AppError> for Fail {
    fn from(e: AppError) -> Self {
        Self::App(e)
    }
}

impl From<Problem> for Fail {
    fn from(p: Problem) -> Self {
        Self::App(AppError::Problem(p))
    }
}

pub type HResult = Result<Resp, Fail>;

/// What the audit needs of the body actually read.
#[derive(Debug, Default)]
pub struct BodyCapture {
    pub head: Vec<u8>,
    pub total: usize,
}

/// One request as a handler sees it.
pub struct Ctx {
    pub c: Arc<Container>,
    pub method: String,
    /// percent-decoded path (Starlette's scope["path"])
    pub path: String,
    pub query: Query,
    pub raw_query: String,
    /// lower-cased names; the last value of a repeated header wins (the middleware's dict)
    pub headers: HashMap<String, String>,
    pub params: Vec<(String, String)>,
    pub principal: Principal,
    pub body_cap: u64,
    pub capture: Arc<Mutex<BodyCapture>>,
    body: Option<AxBody>,
    /// the whole body must have arrived by then (`server::Limits::body_read` after the request started)
    body_deadline: tokio::time::Instant,
}

/// FastAPI's answer when reading the body fails (`HTTPException(400, "There was an error parsing the body")` through
/// api/errors.py): here, a body that ended before its declared length (the client closed or half-closed early). The
/// partial body never reaches a handler as if it were complete.
pub fn body_incomplete() -> Fail {
    Fail::Raw(problem(
        400,
        "http_error",
        "http error",
        "There was an error parsing the body",
        &Map::new(),
    ))
}

/// The body did not arrive within the listener's body-read deadline (no Python counterpart: uvicorn has none).
pub fn body_timeout() -> Fail {
    Fail::Raw(problem(
        408,
        "http_error",
        "http error",
        "the request body was not received in time",
        &Map::new(),
    ))
}

impl Ctx {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c: Arc<Container>,
        method: String,
        path: String,
        raw_query: String,
        headers: HashMap<String, String>,
        principal: Principal,
        body_cap: u64,
        capture: Arc<Mutex<BodyCapture>>,
        body: AxBody,
    ) -> Self {
        Self {
            c,
            method,
            path,
            query: Query::parse(&raw_query),
            raw_query,
            headers,
            params: Vec::new(),
            principal,
            body_cap,
            capture,
            body: Some(body),
            body_deadline: tokio::time::Instant::now() + crate::server::BODY_READ_TIMEOUT,
        }
    }

    /// Move the body-read deadline (the pipeline sets it from the request's start and the listener's limits).
    pub fn set_body_deadline(&mut self, at: std::time::Instant) {
        self.body_deadline = tokio::time::Instant::from_std(at);
    }

    pub fn param(&self, name: &str) -> String {
        self.params
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(String::as_str)
    }

    /// The whole body (empty if already read), refusing more than the route's cap (413 payload_too_large), a body
    /// that ends before its declared length (400) and one that is still arriving at the deadline (408).
    pub async fn read_body(&mut self) -> Result<Bytes, Fail> {
        let Some(mut body) = self.body.take() else {
            return Ok(Bytes::new());
        };
        let mut buf = BytesMut::new();
        let cap = usize::try_from(self.body_cap).unwrap_or(usize::MAX);
        loop {
            let frame = match tokio::time::timeout_at(self.body_deadline, body.frame()).await {
                Err(_) => return Err(body_timeout()),
                Ok(None) => break,
                Ok(Some(Err(_))) => return Err(body_incomplete()),
                Ok(Some(Ok(frame))) => frame,
            };
            let Ok(chunk) = frame.into_data() else {
                continue;
            };
            {
                let mut cap_rec = self.capture.guard();
                cap_rec.total += chunk.len();
                if cap_rec.head.len() < BODY_CAP_AUDIT {
                    let room = BODY_CAP_AUDIT - cap_rec.head.len();
                    cap_rec
                        .head
                        .extend_from_slice(&chunk[..chunk.len().min(room)]);
                }
                if cap_rec.total > cap {
                    return Err(Fail::TooLarge);
                }
            }
            buf.extend_from_slice(&chunk);
        }
        Ok(buf.freeze())
    }

    /// The body as FastAPI reads it for a JSON body parameter.
    pub async fn body(&mut self) -> Result<Body, Fail> {
        let bytes = self.read_body().await?;
        Ok(Body::from_bytes(self.header("content-type"), &bytes))
    }
}

// ── responses ─────────────────────────────────────────────────────────────────────────────────────────────
pub fn with_ct(status: u16, content_type: &str, body: impl Into<AxBody>) -> Resp {
    let mut r = Response::new(body.into());
    *r.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    if let Ok(v) = HeaderValue::from_str(content_type) {
        r.headers_mut().insert(header::CONTENT_TYPE, v);
    }
    r
}

/// A JSON response (FastAPI's JSONResponse: compact, non-ASCII kept).
pub fn json<T: Serialize>(status: u16, v: &T) -> HResult {
    let body = serde_json::to_vec(v).map_err(|_| {
        Fail::App(AppError::Problem(crate::domain::errors::internal(
            "unexpected server error",
        )))
    })?;
    Ok(with_ct(status, "application/json", body))
}

pub fn ok<T: Serialize>(v: &T) -> HResult {
    json(200, v)
}

/// 204 No Content (no content-type, no body).
pub fn no_content() -> HResult {
    let mut r = Response::new(AxBody::empty());
    *r.status_mut() = StatusCode::NO_CONTENT;
    Ok(r)
}

/// `problem(status, code, title, detail, **extra)`
pub fn problem(
    status: u16,
    code: &str,
    title: &str,
    detail: &str,
    extra: &Map<String, Value>,
) -> Resp {
    let mut m = Map::new();
    m.insert(
        "type".into(),
        Value::String(format!("urn:engine-console:{code}")),
    );
    m.insert("title".into(), Value::String(title.into()));
    m.insert("status".into(), Value::from(status));
    m.insert("code".into(), Value::String(code.into()));
    m.insert("detail".into(), Value::String(detail.into()));
    for (k, v) in extra {
        m.insert(k.clone(), v.clone());
    }
    let body = serde_json::to_vec(&Value::Object(m)).unwrap_or_default();
    with_ct(status, PROBLEM_CT, body)
}

/// A problem whose title is its code with spaces (the exception handlers' rule).
pub fn problem_of(p: &Problem) -> Resp {
    problem(
        p.status,
        &p.code,
        &p.code.replace('_', " "),
        &p.detail,
        &p.extra,
    )
}

pub fn not_found_resp(detail: &str) -> Resp {
    problem(404, "not_found", "not found", detail, &Map::new())
}

/// The response for a failure (node errors are labelled with their host through the fleet).
pub fn fail_response(c: &Container, f: Fail) -> Resp {
    match f {
        Fail::Raw(r) => r,
        Fail::TooLarge => problem(
            413,
            "payload_too_large",
            "payload too large",
            "request body too large",
            &Map::new(),
        ),
        Fail::Validation(errs) => {
            let mut extra = Map::new();
            extra.insert(
                "errors".into(),
                Value::Array(errs.iter().map(Issue::to_json).collect()),
            );
            problem(
                422,
                "validation_error",
                "validation error",
                "request validation failed",
                &extra,
            )
        }
        Fail::App(e) => {
            if let AppError::Problem(p) = &e
                && p.status >= 500
                && p.code == "internal_error"
            {
                return problem(
                    500,
                    "internal_error",
                    "internal error",
                    "unexpected server error",
                    &Map::new(),
                );
            }
            let fleet = c.fleet_port();
            problem_of(&to_problem(&e, Some(&fleet)))
        }
    }
}
