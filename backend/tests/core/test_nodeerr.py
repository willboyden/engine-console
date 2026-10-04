"""NodeError -> ProblemError (contract §10.11): one mapping, used everywhere."""
from __future__ import annotations

import asyncio
import logging

import pytest
from fastapi.testclient import TestClient

from engine_console.domain.errors import NotFound, ProblemError
from engine_console.domain.fleet import NodeError, NodeUnavailable
from engine_console.services.nodeerr import HTTP_STATUS, NodeCaller, problem_from_node, shielded

from .conftest import Env
from .fakefleet import until

P = "/api/v1"
TABLE = {
    "node_unavailable": 503, "busy": 503,
    "monitor_only": 409, "conflict": 409, "in_use": 409, "port_in_use": 409, "cancelled": 409,
    "policy_denied": 403, "not_owned": 403, "not_found": 404, "no_space": 507,
    "hash_mismatch": 502, "size_mismatch": 502, "docker_error": 502, "http_error": 502, "io_error": 502, "internal": 502,
    "timeout": 504, "unsupported": 501, "bad_args": 500, "unknown_op": 500,
}


@pytest.mark.parametrize(("code", "status"), sorted(TABLE.items()))
def test_every_code_in_the_contract_maps_to_its_status(code: str, status: int) -> None:
    p = problem_from_node(NodeError("spark", code, "something happened"), "Spark box")
    assert (p.status, p.code) == (status, code)
    assert p.detail == "Spark box: something happened" and p.extra == {"host_id": "spark"}


def test_the_table_has_exactly_the_contracts_codes() -> None:
    assert HTTP_STATUS == TABLE


def test_an_unknown_code_from_a_newer_node_is_a_502_and_keeps_its_code() -> None:
    p = problem_from_node(NodeError("a", "brand_new_code", "x"), None)
    assert p.status == 502 and p.code == "brand_new_code" and p.detail == "a: x"        # the id stands in for a missing label


def test_node_unavailable_subclass_maps_like_the_base() -> None:
    p = problem_from_node(NodeUnavailable("a", "the link is down"), "A")
    assert p.status == 503 and p.code == "node_unavailable"


def test_hub_bug_codes_are_logged_as_errors(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        problem_from_node(NodeError("a", "bad_args", "missing field"), "A")
        problem_from_node(NodeError("a", "unknown_op", "no such op"), "A")
        problem_from_node(NodeError("a", "docker_error", "daemon down"), "A")
    assert caplog.text.count("hub bug") == 2


async def test_the_caller_translates_labels_and_resolves_hosts(env: Env) -> None:
    caller = NodeCaller(env.fleet)
    assert caller.label("local") == "testbox" and caller.label("nope") == "nope"
    env.node.fail["docker.images"] = NodeError("local", "docker_error", "daemon down")
    with pytest.raises(ProblemError) as ei:
        await caller.call("local", "docker.images")
    assert ei.value.status == 502 and ei.value.detail == "testbox: daemon down"
    with pytest.raises(NotFound) as e2:
        await caller.call("nope", "docker.images")
    assert e2.value.code == "unknown_host"
    env.node.fail.clear()
    assert await caller.call("local", "docker.images") == {"images": []}


async def test_a_client_that_never_answers_is_a_504_not_a_hang(env: Env) -> None:
    from engine_console.services import nodeerr

    async def never(op: str, args: object = None, *, timeout: float = 0.0) -> dict[str, object]:
        await asyncio.sleep(3600)
        return {}

    env.node.call = never  # type: ignore[method-assign,assignment]
    old = nodeerr.SLACK_S
    nodeerr.SLACK_S = 0.05
    try:
        with pytest.raises(ProblemError) as ei:
            await NodeCaller(env.fleet).call("local", "docker.images", timeout=0.05)
    finally:
        nodeerr.SLACK_S = old
    assert ei.value.status == 504 and ei.value.code == "timeout"


async def test_shielded_cleanup_finishes_even_when_the_caller_is_cancelled() -> None:
    done: list[str] = []

    async def cleanup() -> None:
        await asyncio.sleep(0.05)
        done.append("cleaned")

    async def work() -> None:
        try:
            await asyncio.sleep(3600)
        finally:
            await shielded(cleanup())

    task = asyncio.create_task(work())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await until(lambda: done == ["cleaned"], what="the shielded cleanup to finish")


async def test_shielded_swallows_cleanup_failures() -> None:
    async def boom() -> None:
        raise NodeError("a", "timeout", "x")

    await shielded(boom())
    await shielded(asyncio.sleep(0))


def test_the_global_handler_formats_untranslated_errors(env: Env) -> None:
    from fastapi import FastAPI

    from engine_console.api.errors import install_error_handlers

    app = FastAPI()
    app.state.container = env.container
    install_error_handlers(app)

    @app.get("/boom")
    async def boom() -> None:
        raise NodeError("local", "policy_denied", "not allowed here")

    @app.get("/boom-unknown-host")
    async def boom2() -> None:
        raise NodeError("ghost", "timeout", "no answer")

    with TestClient(app) as c:
        r = c.get("/boom")
        assert r.status_code == 403 and r.headers["content-type"].startswith("application/problem+json")
        assert r.json()["code"] == "policy_denied" and r.json()["detail"] == "testbox: not allowed here"
        assert r.json()["host_id"] == "local" and r.json()["status"] == 403
        r2 = c.get("/boom-unknown-host")
        assert r2.status_code == 504 and r2.json()["detail"] == "ghost: no answer"          # an unknown host shows its id
