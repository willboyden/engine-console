//! NodeError -> problem (contract §10.11) and the one way services talk to a node (port of services/nodeerr.py).
//! The HTTP error handler uses it too.
//!
//! A node refusal always becomes the same RFC 7807 problem: `code` is the node's code, `detail` is the node's message
//! prefixed with the host's label, `host_id` is an extra member.

use std::sync::Arc;
use std::time::Duration;

use crate::domain::errors::{AppError, NodeError, Obj, Problem};
use crate::domain::fleet::{FleetPort, NodeClient, NodeStream};

/// Outer bound around a node call on top of its own timeout (the client enforces `timeout`).
pub const SLACK_S: f64 = 10.0;

/// §10.11: the HTTP status of every node error code in the contract (nodeerr.py `HTTP_STATUS`).
pub const HTTP_STATUS: &[(&str, u16)] = &[
    ("node_unavailable", 503),
    ("busy", 503),
    ("monitor_only", 409),
    ("conflict", 409),
    ("in_use", 409),
    ("port_in_use", 409),
    ("cancelled", 409),
    ("policy_denied", 403),
    ("not_owned", 403),
    ("not_found", 404),
    ("no_space", 507),
    ("hash_mismatch", 502),
    ("size_mismatch", 502),
    ("docker_error", 502),
    ("http_error", 502),
    ("io_error", 502),
    ("internal", 502),
    ("timeout", 504),
    ("unsupported", 501),
    ("bad_args", 500),
    ("unknown_op", 500),
];

/// §10.11 status of a node error code. A code that is not listed (a newer node, say) is 502.
pub fn status_of(code: &str) -> u16 {
    HTTP_STATUS
        .iter()
        .find(|(c, _)| *c == code)
        .map_or(502, |(_, s)| *s)
}

/// The HTTP problem for a node refusal. `label` is the host's display name (falls back to its id).
pub fn problem_from_node(e: &NodeError, label: Option<&str>) -> Problem {
    if e.code == "bad_args" || e.code == "unknown_op" {
        tracing::error!(target: "engine_console.services.nodeerr", "hub bug: node {} answered {}: {}", e.host_id, e.code, e.message);
    }
    let detail = format!("{}: {}", label.unwrap_or(&e.host_id), e.message);
    Problem::new(status_of(&e.code), &e.code, detail).with("host_id", e.host_id.clone())
}

/// Resolve a host to its node, run one operation, translate failures. Holds no state.
#[derive(Clone)]
pub struct NodeCaller {
    pub fleet: Arc<dyn FleetPort>,
}

impl std::fmt::Debug for NodeCaller {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NodeCaller")
    }
}

impl NodeCaller {
    pub fn new(fleet: Arc<dyn FleetPort>) -> Self {
        Self { fleet }
    }

    /// 404 unknown_host when the host is not configured.
    pub fn node(&self, host_id: &str) -> Result<Arc<dyn NodeClient>, Problem> {
        self.fleet.node(host_id)
    }

    pub fn label(&self, host_id: &str) -> String {
        self.fleet
            .node(host_id)
            .map_or_else(|_| host_id.to_string(), |n| n.host().label.clone())
    }

    pub fn problem(&self, e: &NodeError) -> Problem {
        problem_from_node(e, Some(&self.label(&e.host_id)))
    }

    /// One request; a refusal is the labelled problem; no answer within timeout + SLACK_S is a 504 problem.
    pub async fn call(
        &self,
        host_id: &str,
        op: &str,
        args: Option<Obj>,
        timeout_s: f64,
    ) -> Result<Obj, AppError> {
        self.call_with_slack(host_id, op, args, timeout_s, SLACK_S)
            .await
    }

    /// [`Self::call`] with an explicit outer slack (tests: Python patches `nodeerr.SLACK_S`).
    pub async fn call_with_slack(
        &self,
        host_id: &str,
        op: &str,
        args: Option<Obj>,
        timeout_s: f64,
        slack_s: f64,
    ) -> Result<Obj, AppError> {
        let node = self.node(host_id)?;
        let fut = node.call(op, args, Duration::from_secs_f64(timeout_s.max(0.0)));
        match tokio::time::timeout(Duration::from_secs_f64((timeout_s + slack_s).max(0.0)), fut)
            .await
        {
            Ok(Ok(r)) => Ok(r),
            Ok(Err(e)) => Err(self.problem(&e).into()),
            Err(_) => Err(self
                .problem(&NodeError::new(
                    host_id,
                    "timeout",
                    format!("{op} did not answer in {timeout_s:.0}s"),
                ))
                .into()),
        }
    }

    pub fn open(
        &self,
        host_id: &str,
        op: &str,
        args: Option<Obj>,
    ) -> Result<Box<dyn NodeStream>, Problem> {
        Ok(self.node(host_id)?.open(op, args))
    }
}

/// Map an AppError raised by a service to the problem the client sees, labelling node errors through `fleet`.
pub fn to_problem(err: &AppError, fleet: Option<&Arc<dyn FleetPort>>) -> Problem {
    match err {
        AppError::Problem(p) => p.clone(),
        AppError::Node(e) => {
            let label = fleet
                .and_then(|f| f.node(&e.host_id).ok())
                .map(|n| n.host().label.clone());
            problem_from_node(e, label.as_deref())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Port of the pure rows of tests/core/test_nodeerr.py (names mirror the Python tests); the rows that need a
    // fleet, a node or the HTTP layer are in tests/test_nodeerr.rs.
    const TABLE: &[(&str, u16)] = &[
        ("node_unavailable", 503),
        ("busy", 503),
        ("monitor_only", 409),
        ("conflict", 409),
        ("in_use", 409),
        ("port_in_use", 409),
        ("cancelled", 409),
        ("policy_denied", 403),
        ("not_owned", 403),
        ("not_found", 404),
        ("no_space", 507),
        ("hash_mismatch", 502),
        ("size_mismatch", 502),
        ("docker_error", 502),
        ("http_error", 502),
        ("io_error", 502),
        ("internal", 502),
        ("timeout", 504),
        ("unsupported", 501),
        ("bad_args", 500),
        ("unknown_op", 500),
    ];

    #[test]
    fn test_every_code_in_the_contract_maps_to_its_status() {
        for (code, status) in TABLE {
            let p = problem_from_node(
                &NodeError::new("spark", code, "something happened"),
                Some("Spark box"),
            );
            assert_eq!((p.status, p.code.as_str()), (*status, *code));
            assert_eq!(p.detail, "Spark box: something happened");
            assert_eq!(
                p.extra,
                serde_json::json!({"host_id": "spark"})
                    .as_object()
                    .cloned()
                    .unwrap()
            );
        }
    }

    #[test]
    fn test_the_table_has_exactly_the_contracts_codes() {
        let mut got = HTTP_STATUS.to_vec();
        let mut want = TABLE.to_vec();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn test_an_unknown_code_from_a_newer_node_is_a_502_and_keeps_its_code() {
        let p = problem_from_node(&NodeError::new("a", "brand_new_code", "x"), None);
        // the id stands in for a missing label
        assert_eq!(
            (p.status, p.code.as_str(), p.detail.as_str()),
            (502, "brand_new_code", "a: x")
        );
    }

    #[test]
    fn test_node_unavailable_subclass_maps_like_the_base() {
        let p = problem_from_node(&NodeError::unavailable("a", "the link is down"), Some("A"));
        assert_eq!((p.status, p.code.as_str()), (503, "node_unavailable"));
    }
}
