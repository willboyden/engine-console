"""Errors an op handler raises, and the stream interface a handler talks through.

Handlers never format a res themselves: they return a result dict or raise OpError(code, message) with a code
from proto.py and a sentence a person can act on. The server turns anything else into "internal" without a
traceback in the message.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class OpError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Cancelled(Exception):
    """The hub cancelled the request (or the link went away) while it was running."""


class Stream(Protocol):
    """What a handler may do besides returning its result."""

    @property
    def cancelled(self) -> bool: ...

    def check(self) -> None:
        """Raise Cancelled if the hub cancelled this request."""

    def event(self, ev: dict[str, Any]) -> None:
        """Send one event frame for this request."""

    def send_data(self, data: bytes) -> None:
        """Send bytes on this request's stream; waits for credit. Raises Cancelled."""

    def read_upload(self) -> bytes:
        """Next uploaded chunk from the hub; b"" once the hub sent `end`. Grants credit as it is consumed."""

    def drain_upload(self) -> None:
        """Discard the rest of the upload until `end` (or cancel), still granting credit."""

    def on_cancel(self, fn: Callable[[], None]) -> None:
        """Call fn (once, from another thread) when the request is cancelled or the node shuts down."""
