"""http.request: allowed ports and paths only, header hygiene, uploads, streaming, no redirects, timeouts."""
from __future__ import annotations

import threading
from typing import Any

import pytest

from ecnode import proto
from ecnode.errors import Cancelled, OpError
from ecnode.httpproxy import request

from .test_fsops import FakeStream


def run(engine, args: dict[str, Any], stream: FakeStream | None = None, ports: set[int] | None = None):
    s = stream or FakeStream()
    allowed = {engine.port} if ports is None else ports
    res = request(proto.clean_args("http.request", {"port": engine.port, "method": "GET", **args}), s,
                  lambda refresh: allowed)
    return res, s


def test_get_returns_one_event_then_the_body(engine):
    res, s = run(engine, {"path": "/v1/models"})
    assert res["status"] == 200 and res["bytes"] == len(b"".join(s.data))
    assert len(s.events) == 1 and s.events[0]["status"] == 200
    assert s.events[0]["headers"]["content-type"] == "application/json"
    assert b"org/model-a" in b"".join(s.data)


def test_post_body_arrives_as_upload_and_headers_are_cleaned(engine):
    body = b'{"model": "m", "messages": []}'
    s = FakeStream([body[:10], body[10:]])
    res, _ = run(engine, {"method": "POST", "path": "/v1/chat/completions", "body_len": len(body),
                          "headers": {"Authorization": "Bearer engine-key-123456", "Host": "evil.invalid",
                                      "Connection": "keep-alive, X-Drop", "X-Drop": "1", "Transfer-Encoding": "chunked",
                                      "Content-Type": "application/json"}}, s)
    assert res["status"] == 200 and b"".join(s.data) == b"echo:" + body
    seen = engine.seen[-1]
    assert seen["body"] == body and seen["headers"]["Host"] == f"127.0.0.1:{engine.port}"
    assert seen["headers"]["Authorization"] == "Bearer engine-key-123456"
    assert "X-Drop" not in seen["headers"] and "Transfer-Encoding" not in seen["headers"]


@pytest.mark.parametrize("args,code", [
    ({"path": "/admin"}, proto.POLICY_DENIED),
    ({"path": "/v1/../admin"}, proto.BAD_ARGS),
    ({"path": "/v1/%2e%2e/admin"}, proto.BAD_ARGS),
    ({"path": "/v1/models with space"}, proto.BAD_ARGS),
    ({"path": "v1/models"}, proto.BAD_ARGS),
    ({"path": "/v1/x\r\nHost: evil"}, proto.BAD_ARGS),
    ({"path": "/v1/models", "method": "PUT"}, proto.POLICY_DENIED),
    ({"path": "/v1/models", "method": "DELETE"}, proto.POLICY_DENIED),
    ({"path": "/v1/models", "headers": {"X-A": "a\r\nInjected: 1"}}, proto.BAD_ARGS),
    ({"path": "/v1/models", "headers": {"Bad Name": "x"}}, proto.BAD_ARGS),
    ({"path": "/v1/models", "body_len": 5}, proto.BAD_ARGS),
])
def test_refusals(engine, args, code):
    with pytest.raises(OpError) as e:
        run(engine, args)
    assert e.value.code == code


def test_only_engine_and_gateway_ports(engine):
    with pytest.raises(OpError) as e:
        run(engine, {"path": "/v1/models"}, ports=set())
    assert e.value.code == proto.POLICY_DENIED


def test_body_length_must_match(engine):
    for chunks in ([b"abc"], [b"abcdefgh"], [b"abcde", b"extra"]):
        with pytest.raises(OpError) as e:
            run(engine, {"method": "POST", "path": "/v1/chat/completions", "body_len": 5}, FakeStream(chunks))
        assert e.value.code == proto.BAD_ARGS


def test_redirects_are_returned_not_followed(engine):
    res, s = run(engine, {"path": "/v1/redirect"})
    assert res["status"] == 302 and s.events[0]["headers"]["location"] == "http://example.invalid/"
    assert all(r["path"] != "/" for r in engine.seen)


def test_chunked_streams_are_forwarded(engine):
    res, s = run(engine, {"path": "/v1/stream"})
    assert b"".join(s.data) == b"data: 0\n\ndata: 1\n\ndata: 2\n\n" and res["bytes"] == 27


def test_timeout_and_cancel(engine):
    with pytest.raises(OpError) as e:
        run(engine, {"path": "/v1/slow", "timeout_s": 1})
    assert e.value.code == proto.TIMEOUT
    s = FakeStream()
    timer = threading.Timer(0.3, lambda: (setattr(s, "_cancelled", True), [h() for h in s.hooks]))
    timer.start()
    with pytest.raises(Cancelled):
        run(engine, {"path": "/v1/slow"}, s)
    timer.cancel()


def test_nothing_listening(engine):
    import socket

    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        closed = sk.getsockname()[1]
    with pytest.raises(OpError) as e:
        request(proto.clean_args("http.request", {"port": closed, "method": "GET", "path": "/health"}), FakeStream(),
                lambda refresh: {closed})
    assert e.value.code == proto.HTTP_ERROR
