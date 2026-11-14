//! The problem type of the adapters and services (port of domain/errors.py `ProblemError` and its subclasses).
//!
//! A [`Problem`] is an RFC 7807 problem with a stable `code`. The hub wraps it in its `AppError`
//! (ec-hub `domain::errors`, which re-exports everything here); the adapters return it from `build_launch`.

use serde_json::{Map, Value};

pub type Obj = Map<String, Value>;

/// `ProblemError`: status, code, detail and extra members of the problem body.
#[derive(Debug, Clone, PartialEq)]
pub struct Problem {
    pub status: u16,
    pub code: String,
    pub detail: String,
    /// extra members of the problem body (e.g. `errors`, `host_id`, `scope`, `limit`), in insertion order
    pub extra: Obj,
}

impl Problem {
    pub fn new(status: u16, code: &str, detail: impl Into<String>) -> Self {
        Self {
            status,
            code: code.to_string(),
            detail: detail.into(),
            extra: Obj::new(),
        }
    }

    /// Add one extra member (`ProblemError(..., **extra)`).
    #[must_use]
    pub fn with(mut self, key: &str, value: impl Into<Value>) -> Self {
        self.extra.insert(key.to_string(), value.into());
        self
    }

    /// Replace the code (`ProblemError(detail, code=...)`).
    #[must_use]
    pub fn code(mut self, code: &str) -> Self {
        self.code = code.to_string();
        self
    }
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}: {}", self.status, self.code, self.detail)
    }
}

// ── the ProblemError subclasses of domain/errors.py ────────────────────────────────────────────────────────
pub fn not_found(detail: impl Into<String>) -> Problem {
    Problem::new(404, "not_found", detail)
}
pub fn conflict(detail: impl Into<String>) -> Problem {
    Problem::new(409, "conflict", detail)
}
pub fn bad_request(detail: impl Into<String>) -> Problem {
    Problem::new(400, "bad_request", detail)
}
pub fn unprocessable(detail: impl Into<String>) -> Problem {
    Problem::new(422, "invalid_params", detail)
}
pub fn forbidden(detail: impl Into<String>) -> Problem {
    Problem::new(403, "forbidden", detail)
}
pub fn unauthorized(detail: impl Into<String>) -> Problem {
    Problem::new(401, "unauthorized", detail)
}
pub fn upstream_error(detail: impl Into<String>) -> Problem {
    Problem::new(502, "upstream_error", detail)
}
pub fn gated_model(detail: impl Into<String>) -> Problem {
    Problem::new(403, "hf_gated", detail)
}
pub fn too_many_streams(detail: impl Into<String>) -> Problem {
    Problem::new(429, "too_many_streams", detail)
}
pub fn egress_proxy_unavailable(detail: impl Into<String>) -> Problem {
    Problem::new(503, "egress_proxy_unavailable", detail)
}
pub fn egress_denied(detail: impl Into<String>) -> Problem {
    Problem::new(403, "egress_denied", detail)
}
pub fn internal(detail: impl Into<String>) -> Problem {
    Problem::new(500, "internal_error", detail)
}
