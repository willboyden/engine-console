//! Port of node/src/ecnode/errors.py: the errors an op handler raises, the stream interface a
//! handler talks through, and the call a handler receives.
//!
//! Handlers never format a res themselves: they return a result object or an `OpError(code, message)` with a code
//! from `ec_proto` and a sentence a person can act on. The server turns anything else into "internal" without a
//! traceback in the message.

use ec_proto::Obj;

/// A refusal or failure with a protocol error code (`ec_proto::BAD_ARGS`, `POLICY_DENIED`, ...).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{code}: {message}")]
pub struct OpError {
    pub code: String,
    pub message: String,
}

impl OpError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
        }
    }
}

/// The hub cancelled the request (or the link went away) while it was running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("cancelled")]
pub struct Cancelled;

/// Everything a handler can end with besides its result.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HandlerError {
    #[error(transparent)]
    Op(#[from] OpError),
    #[error("cancelled")]
    Cancelled,
    /// A handler bug. `kind` names the failure class (Python shows the exception type name, e.g. "OSError");
    /// the server answers `internal` with it and never a traceback.
    #[error("internal error ({kind})")]
    Internal {
        kind: String,
        location: Option<String>,
    },
}

impl From<Cancelled> for HandlerError {
    fn from(_: Cancelled) -> Self {
        HandlerError::Cancelled
    }
}

impl HandlerError {
    pub fn internal(kind: impl Into<String>) -> Self {
        HandlerError::Internal {
            kind: kind.into(),
            location: None,
        }
    }
}

/// What a handler may do besides returning its result (errors.py `Stream`).
pub trait Stream: Send + Sync {
    /// True once the hub cancelled this request or the node is shutting down.
    fn cancelled(&self) -> bool;

    /// `Err(Cancelled)` if the hub cancelled this request.
    fn check(&self) -> Result<(), Cancelled>;

    /// Send one event frame for this request (cleaned with `ec_proto::specs::events(op)`).
    fn event(&self, ev: Obj);

    /// Send bytes on this request's stream; waits for credit. Fails with Cancelled, or with an `OpError(timeout)`
    /// when the hub stops granting credit.
    fn send_data(&self, data: &[u8]) -> Result<(), HandlerError>;

    /// Next uploaded chunk from the hub; empty once the hub sent `end`. Grants credit as it is consumed.
    fn read_upload(&self) -> Result<Vec<u8>, HandlerError>;

    /// Discard the rest of the upload until `end` (or cancel), still granting credit.
    fn drain_upload(&self);

    /// Call `f` (once, from another thread) when the request is cancelled or the node shuts down; at once if that
    /// already happened.
    fn on_cancel(&self, f: Box<dyn FnOnce() + Send>);
}

/// One request as a handler sees it (ops.py `Call`): arguments already through `ec_proto::clean_args` and the
/// REQUIRED check, the monitor-mode gate already passed.
pub struct Call<'a> {
    pub op: &'a str,
    pub args: &'a Obj,
    pub stream: &'a dyn Stream,
}
