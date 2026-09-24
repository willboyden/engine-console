"""Domain errors. The API layer maps every ProblemError to RFC 7807 problem+json with a `code`."""
from __future__ import annotations

from typing import Any


class ProblemError(Exception):
    status: int = 500
    code: str = "internal_error"

    def __init__(self, detail: str, *, code: str | None = None, status: int | None = None,
                 **extra: Any) -> None:
        super().__init__(detail)
        self.detail = detail
        if code is not None:
            self.code = code
        if status is not None:
            self.status = status
        self.extra = extra


class NotFound(ProblemError):
    status = 404
    code = "not_found"


class Conflict(ProblemError):
    status = 409
    code = "conflict"


class BadRequest(ProblemError):
    status = 400
    code = "bad_request"


class Unprocessable(ProblemError):
    status = 422
    code = "invalid_params"


class Forbidden(ProblemError):
    status = 403
    code = "forbidden"


class Unauthorized(ProblemError):
    status = 401
    code = "unauthorized"


class UpstreamError(ProblemError):
    status = 502
    code = "upstream_error"


class GatedModel(ProblemError):
    status = 403
    code = "hf_gated"


class EgressProxyUnavailable(ProblemError):
    """Fail-closed: the console will not fall back to a direct connection when the chokepoint is down."""
    status = 503
    code = "egress_proxy_unavailable"


class EgressDenied(ProblemError):
    status = 403
    code = "egress_denied"
