"""NodeError -> ProblemError (contract §10.11) and the one way services talk to a node.

Every service that asks a host's node for something goes through `NodeCaller`, so a node refusal always becomes
the same RFC 7807 problem: `code` is the node's code, `detail` is the node's message prefixed with the host's
label (so a person with five machines knows which one said it)."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from engine_console.domain.errors import NotFound, ProblemError
from engine_console.domain.fleet import FleetPort, NodeClient, NodeError, NodeStream

log = logging.getLogger(__name__)

# §10.11. A code that is not listed (a newer node, say) is a node-side failure: 502.
HTTP_STATUS: dict[str, int] = {
    "node_unavailable": 503, "busy": 503,
    "monitor_only": 409, "conflict": 409, "in_use": 409, "port_in_use": 409, "cancelled": 409,
    "policy_denied": 403, "not_owned": 403,
    "not_found": 404,
    "no_space": 507,
    "hash_mismatch": 502, "size_mismatch": 502, "docker_error": 502, "http_error": 502, "io_error": 502,
    "internal": 502,
    "timeout": 504,                 # the action may or may not have taken effect: the UI says so
    "unsupported": 501,
    "bad_args": 500, "unknown_op": 500,   # a hub bug
}
SLACK_S = 10.0   # the client enforces `timeout`; this outer bound only protects against a client that does not


def problem_from_node(e: NodeError, label: str | None = None) -> ProblemError:
    """The HTTP problem for a node refusal. `label` is the host's display name (falls back to its id)."""
    if e.code in ("bad_args", "unknown_op"):
        log.error("hub bug: node %s answered %s: %s", e.host_id, e.code, e.message)
    detail = f"{label or e.host_id}: {e.message}"
    return ProblemError(detail, code=e.code, status=HTTP_STATUS.get(e.code, 502), host_id=e.host_id)


class NodeCaller:
    """Resolve a host to its node, run one operation, translate failures. Holds no state."""

    def __init__(self, fleet: FleetPort) -> None:
        self.fleet = fleet

    def node(self, host_id: str) -> NodeClient:
        return self.fleet.node(host_id)   # NotFound(code="unknown_host") when the host is not configured

    def label(self, host_id: str) -> str:
        try:
            return self.fleet.node(host_id).host.label
        except NotFound:
            return host_id

    def problem(self, e: NodeError) -> ProblemError:
        return problem_from_node(e, self.label(e.host_id))

    async def call(self, host_id: str, op: str, args: dict[str, Any] | None = None, *,
                   timeout: float = 30.0) -> dict[str, Any]:
        node = self.node(host_id)
        try:
            return await asyncio.wait_for(node.call(op, args, timeout=timeout), timeout + SLACK_S)
        except NodeError as e:
            raise self.problem(e) from None
        except TimeoutError:
            raise self.problem(NodeError(host_id, "timeout", f"{op} did not answer in {timeout:.0f}s")) from None

    def open(self, host_id: str, op: str, args: dict[str, Any] | None = None) -> NodeStream:
        return self.node(host_id).open(op, args)


async def shielded(coro: Any) -> None:
    """Run cleanup that must finish even when the surrounding task is being cancelled (a browser closing an SSE
    stream cancels its generator; the node request behind it still has to be told to stop)."""
    try:
        await asyncio.shield(asyncio.ensure_future(coro))
    except (asyncio.CancelledError, NodeError, ProblemError):
        return
    except Exception:   # noqa: BLE001 - cleanup is best effort; never mask the original outcome
        log.debug("cleanup failed", exc_info=True)
