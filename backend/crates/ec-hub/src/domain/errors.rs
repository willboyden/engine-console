//! Domain errors (port of domain/errors.py + the NodeError of domain/fleet.py).
//!
//! Every error a service returns is an [`AppError`]: an RFC 7807 problem (`Problem`, with a stable `code`) or a node
//! refusal (`Node`), which the HTTP layer turns into a problem prefixed with the host's label (services/nodeerr.rs).

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

/// A node refused or failed a request. `code` is an ecnode.proto error code or "node_unavailable".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeError {
    pub host_id: String,
    pub code: String,
    pub message: String,
}

impl NodeError {
    pub fn new(host_id: &str, code: &str, message: impl Into<String>) -> Self {
        Self {
            host_id: host_id.to_string(),
            code: code.to_string(),
            message: message.into(),
        }
    }

    /// `NodeUnavailable`: the link to that host is not up.
    pub fn unavailable(host_id: &str, message: impl Into<String>) -> Self {
        Self::new(host_id, "node_unavailable", message)
    }
}

impl std::fmt::Display for NodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}: {}", self.host_id, self.code, self.message)
    }
}

/// What every service method returns on failure.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AppError {
    #[error("{0}")]
    Problem(Problem),
    #[error("{0}")]
    Node(NodeError),
}

impl From<Problem> for AppError {
    fn from(p: Problem) -> Self {
        Self::Problem(p)
    }
}

impl From<NodeError> for AppError {
    fn from(e: NodeError) -> Self {
        Self::Node(e)
    }
}

pub type AppResult<T> = Result<T, AppError>;

impl AppError {
    /// The problem, when this is one (tests and callers that branch on codes).
    pub fn problem(&self) -> Option<&Problem> {
        match self {
            Self::Problem(p) => Some(p),
            Self::Node(_) => None,
        }
    }

    pub fn status(&self) -> u16 {
        match self {
            Self::Problem(p) => p.status,
            Self::Node(e) => crate::services::nodeerr::status_of(&e.code),
        }
    }

    pub fn code(&self) -> &str {
        match self {
            Self::Problem(p) => &p.code,
            Self::Node(e) => &e.code,
        }
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

/// The answer for a handler or service that is not ported: 501 `not_ported`, detail `not ported: <python module>`
/// (D10). Every service is ported; the placeholder adapter is its only remaining user.
pub fn not_ported(module: &str) -> AppError {
    AppError::Problem(Problem::new(
        501,
        "not_ported",
        format!("not ported: {module}"),
    ))
}
