//! Domain errors (port of domain/errors.py + the NodeError of domain/fleet.py).
//!
//! Every error a service returns is an [`AppError`]: an RFC 7807 problem (`Problem`, with a stable `code`) or a node
//! refusal (`Node`), which the HTTP layer turns into a problem prefixed with the host's label (services/nodeerr.rs).

pub use ec_engines::errors::{
    Obj, Problem, bad_request, conflict, egress_denied, egress_proxy_unavailable, forbidden,
    gated_model, internal, not_found, too_many_streams, unauthorized, unprocessable,
    upstream_error,
};

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

/// The answer for a handler or service that is not ported: 501 `not_ported`, detail `not ported: <python module>`
/// (D10). Every service is ported; the placeholder adapter is its only remaining user.
pub fn not_ported(module: &str) -> AppError {
    AppError::Problem(Problem::new(
        501,
        "not_ported",
        format!("not ported: {module}"),
    ))
}
