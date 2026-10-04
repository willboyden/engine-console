"""Engine HTTP through a host's node: the helper itself, then chat (SSE passthrough) and the benchmark on top of it."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from engine_console.domain.errors import ProblemError, UpstreamError
from engine_console.domain.fleet import NodeError
from engine_console.domain.models import InstanceCreate
from engine_console.services.bench import BenchService, ReqStat, one_request
from engine_console.services.nodehttp import NodeHttp

from .conftest import Env
from .fakefleet import ports

P = "/api/v1"


def sse_body(words: tuple[str, ...] = ("Hel", "lo", " there"), usage: bool = True) -> bytes:
    chunks: list[dict[str, Any]] = [{"choices": [{"delta": {"content": w}}]} for w in words]
    if usage:
        chunks.append({"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 3}})
    return ("".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


def engine(request: httpx.Request) -> httpx.Response:
    req = json.loads(request.content)
    if req.get("stream"):
        return httpx.Response(200, content=sse_body(), headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json={"choices": [{"message": {"content": "Hello there", "reasoning_content": "hmm"}}],
                                     "usage": {"prompt_tokens": 20, "completion_tokens": 3}})


async def ready(env: Env, **kw: Any) -> Any:
    env.seed_local_model()
    env.engine_up()
    inst = await env.container.lifecycle.create(InstanceCreate(engine="fake", repo_id="Qwen/Qwen3-32B",
                                                               params={"max_model_len": 4096}, **kw))
    await env.container.lifecycle.tick()
    assert env.container.lifecycle.get(inst.id).state == "ready"
    return env.container.lifecycle.get(inst.id)


# ---- the helper -----------------------------------------------------------------------------------------------------------------------
async def test_a_buffered_request_returns_status_headers_and_body(env: Env) -> None:
    inst = await ready(env)
    env.node.serve(None, "/v1/models", lambda r: httpx.Response(200, json={"data": []}, headers={"X-Thing": "1"}))
    r = await NodeHttp(env.fleet).request("local", inst.port, "GET", "/v1/models")
    assert r.status == 200 and json.loads(r.body) == {"data": []} and r.headers["x-thing"] == "1"
    sent = env.node.ops("http.request")[-1]
    assert sent["port"] == inst.port and sent["method"] == "GET" and "body_len" not in sent


async def test_a_request_body_is_uploaded_in_bounded_chunks_and_arrives_whole(env: Env) -> None:
    inst = await ready(env)
    seen: list[bytes] = []
    env.node.serve(None, "/v1/echo", lambda r: (seen.append(r.content), httpx.Response(200, text="ok"))[1])
    body = bytes(range(256)) * 2800                                          # ~700 KiB: more than two upload chunks
    r = await NodeHttp(env.fleet).request("local", inst.port, "POST", "/v1/echo", body=body,
                                          headers={"content-type": "application/json"})
    assert r.status == 200 and seen == [body]
    assert env.node.ops("http.request")[-1]["body_len"] == len(body)
    assert env.node.streams[-1].ended.is_set()


async def test_a_response_over_the_cap_is_refused_not_buffered(env: Env) -> None:
    inst = await ready(env)
    env.node.serve(None, "/v1/big", lambda r: httpx.Response(200, content=b"x" * 5000))
    with pytest.raises(UpstreamError):
        await NodeHttp(env.fleet).request("local", inst.port, "GET", "/v1/big", max_bytes=1000)
    assert env.node.streams[-1].cancelled                                     # and the node is told to stop sending


async def test_the_nodes_refusals_are_problems_naming_the_host(env: Env) -> None:
    http = NodeHttp(env.fleet)
    with pytest.raises(ProblemError) as e1:                                   # a port that is neither a gateway nor an endpoint
        await http.request("local", 9999, "GET", "/health")
    assert e1.value.status == 403 and e1.value.code == "policy_denied" and e1.value.detail.startswith("testbox: ")
    inst = await ready(env)
    with pytest.raises(ProblemError) as e2:
        await http.request("local", inst.port, "GET", "/admin/secrets")      # a path the node does not relay
    assert e2.value.code == "policy_denied"
    env.node.serve(None, "/v1/boom", lambda r: (_ for _ in ()).throw(NodeError("local", "http_error", "connection refused")))
    with pytest.raises(ProblemError) as e3:
        await http.request("local", inst.port, "GET", "/v1/boom")
    assert e3.value.status == 502 and e3.value.code == "http_error"
    env.node.state = "down"
    with pytest.raises(ProblemError) as e4:
        await http.request("local", inst.port, "GET", "/health")
    assert e4.value.status == 503 and e4.value.code == "node_unavailable"
    with pytest.raises(ProblemError) as e5:
        await http.request("nope", 1, "GET", "/health")
    assert e5.value.code == "unknown_host" and e5.value.status == 404


async def test_lines_are_split_across_chunk_boundaries_and_multibyte_characters_survive(env: Env) -> None:
    inst = await ready(env)
    env.node.http_chunk = 3                                                   # smaller than one "✓" (3 bytes) boundary
    env.node.serve(None, "/v1/lines", lambda r: httpx.Response(200, content="héllo ✓\r\nsecond\nthird".encode()))
    resp = await NodeHttp(env.fleet).open("local", inst.port, "GET", "/v1/lines")
    try:
        assert [ln async for ln in resp.lines()] == ["héllo ✓", "second", "third"]
    finally:
        await resp.aclose()


async def test_closing_a_stream_early_cancels_the_request_on_the_node(env: Env) -> None:
    inst = await ready(env)
    env.node.http_chunk = 4
    env.node.serve(None, "/v1/long", lambda r: httpx.Response(200, content=b"x" * 400))
    resp = await NodeHttp(env.fleet).open("local", inst.port, "GET", "/v1/long")
    chunks = resp.chunks()
    await chunks.__anext__()
    await resp.aclose()
    assert env.node.streams[-1].cancelled
    await chunks.aclose()


# ---- chat ---------------------------------------------------------------------------------------------------------------------------------
async def test_chat_streams_through_the_node_and_records_usage_and_the_conversation(env: Env) -> None:
    inst = await ready(env)
    env.node.serve(None, "/v1/chat/completions", engine)
    chat = env.container.chat
    conv = chat.create_conversation(__import__("engine_console.domain.models", fromlist=["ConversationIn"]).ConversationIn(
        title="t", instance_id=inst.id))
    msgs = [{"role": "user", "content": "hi"}]
    stream = await chat.open_stream({"instance_id": inst.id, "conversation_id": conv.id, "messages": msgs, "stream": True})
    text = "".join([line async for line in stream.aiter()])
    assert "Hel" in text and "[DONE]" in text and text.count("\n\n") >= 3      # event separators are preserved
    req = env.engine_requests[-1]
    sent = json.loads(req.content)
    assert req.url.port == inst.port and req.url.path == "/v1/chat/completions" and req.method == "POST"
    assert sent["model"] == "Qwen/Qwen3-32B" and "instance_id" not in sent and "conversation_id" not in sent
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    assert env.node.ops("http.request")[-1]["body_len"] == len(req.content)
    assert req.headers["content-type"] == "application/json"
    full = chat.get_conversation(conv.id, with_messages=True)
    assert [m["role"] for m in full.messages or []] == ["user", "assistant"] and (full.messages or [])[1]["content"] == "Hello there"
    usage = env.container.usage.query("model")
    assert usage[0].key == "Qwen/Qwen3-32B" and usage[0].requests == 1 and usage[0].prompt_tokens == 20


async def test_chat_non_streaming_and_reasoning(env: Env) -> None:
    inst = await ready(env)
    env.node.serve(None, "/v1/chat/completions", engine)
    data = await env.container.chat.complete({"instance_id": inst.id, "messages": [{"role": "user", "content": "hi"}]})
    assert data["choices"][0]["message"]["content"] == "Hello there"
    assert json.loads(env.engine_requests[-1].content)["stream"] is False
    assert env.container.usage.query("model")[0].completion_tokens == 3


async def test_a_client_that_goes_away_mid_stream_cancels_the_node_request_and_still_records_usage(env: Env) -> None:
    inst = await ready(env)
    env.node.http_chunk = 8
    env.node.serve(None, "/v1/chat/completions", engine)
    stream = await env.container.chat.open_stream({"instance_id": inst.id, "messages": [{"role": "user", "content": "hi"}]})
    gen = stream.aiter()
    first = await gen.__anext__()
    assert first.startswith("data:")
    await gen.aclose()                                                        # the browser closed the connection
    assert env.node.streams[-1].cancelled
    assert env.container.usage.query("model")[0].requests == 1               # the partial answer is still accounted for


async def test_engine_error_statuses_are_mapped_and_unreachable_engines_are_502_problems(env: Env) -> None:
    inst = await ready(env)
    chat = env.container.chat
    body = {"instance_id": inst.id, "messages": [{"role": "user", "content": "hi"}]}
    for status, want in ((400, 400), (404, 404), (422, 422), (500, 502), (503, 502)):
        env.node.serve(None, "/v1/chat/completions", lambda r, s=status: httpx.Response(s, text="engine says no"))
        with pytest.raises(UpstreamError) as ei:
            await chat.complete(body)
        assert ei.value.status == want and "engine says no" in ei.value.detail and f"HTTP {status}" in ei.value.detail
        with pytest.raises(UpstreamError) as ei2:
            await chat.open_stream({**body, "stream": True})
        assert ei2.value.status == want
    env.node.serve(None, "/v1/chat/completions", lambda r: (_ for _ in ()).throw(NodeError("local", "http_error", "refused")))
    with pytest.raises(ProblemError) as e3:
        await chat.complete(body)
    assert e3.value.status == 502 and e3.value.code == "http_error" and "testbox" in e3.value.detail
    env.node.serve(None, "/v1/chat/completions", lambda r: httpx.Response(200, text="<html>not json</html>"))
    with pytest.raises(UpstreamError):
        await chat.complete(body)


async def test_chat_to_an_instance_on_another_host_uses_that_hosts_node(env: Env) -> None:
    spark = env.add_host("spark", "Spark")
    env.seed_local_model(host_id="spark")
    spark.serve(None, "/health", lambda r: httpx.Response(200, text="ok"))
    spark.serve(None, "/v1/chat/completions", engine)
    inst = await env.container.lifecycle.create(InstanceCreate(engine="fake", repo_id="Qwen/Qwen3-32B", host_id="spark",
                                                               params={"max_model_len": 4096}))
    await env.container.lifecycle.tick()
    data = await env.container.chat.complete({"instance_id": inst.id, "messages": [{"role": "user", "content": "hi"}]})
    assert data["choices"] and len(spark.http_requests) >= 1 and spark.http_requests[-1].url.path == "/v1/chat/completions"
    assert not [r for r in env.node.http_requests if r.url.path == "/v1/chat/completions"]


async def test_chat_needs_a_ready_instance_and_a_known_one(env: Env) -> None:
    env.seed_local_model()
    inst = await env.container.lifecycle.create(InstanceCreate(engine="fake", repo_id="Qwen/Qwen3-32B",
                                                               params={"max_model_len": 4096}))
    from engine_console.domain.errors import Conflict, NotFound
    with pytest.raises(Conflict) as e1:
        await env.container.chat.complete({"instance_id": inst.id, "messages": [{"role": "user", "content": "x"}]})
    assert e1.value.code == "instance_not_ready"
    with pytest.raises(NotFound):
        await env.container.chat.complete({"instance_id": "inst_nope", "messages": [{"role": "user", "content": "x"}]})


# ---- bench ----------------------------------------------------------------------------------------------------------------------------------------
async def test_one_request_measures_ttft_and_counts_tokens_through_the_node(env: Env) -> None:
    inst = await ready(env)
    env.node.serve(None, "/v1/chat/completions", engine)
    st = await one_request(NodeHttp(env.fleet), "local", inst.port or 0, "m", "prompt", 8)
    assert st.error is None and st.ttft is not None and st.prompt_tokens == 20 and st.completion_tokens == 3 and st.e2e > 0


async def test_one_request_reports_http_and_node_errors_instead_of_raising(env: Env) -> None:
    inst = await ready(env)
    http = NodeHttp(env.fleet)
    env.node.serve(None, "/v1/chat/completions", lambda r: httpx.Response(500, text="x"))
    assert (await one_request(http, "local", inst.port or 0, "m", "p", 8)).error == "HTTP 500"
    env.node.serve(None, "/v1/chat/completions", lambda r: (_ for _ in ()).throw(NodeError("local", "timeout", "slow")))
    assert (await one_request(http, "local", inst.port or 0, "m", "p", 8)).error == "timeout"
    st = ReqStat()
    assert st.error is None


async def test_a_benchmark_runs_on_the_instances_host_and_stores_results(env: Env) -> None:
    inst = await ready(env)
    env.node.serve(None, "/v1/chat/completions", engine)
    bench: BenchService = env.container.bench
    run = await bench.start(inst.id, {"concurrency": [1, 2], "prompt_tokens": 28, "max_tokens": 4, "rounds": 1,
                                      "prefix_cache": True})
    await bench.wait(run.id)
    done = bench.get(run.id)
    assert done.state == "completed", done
    assert [lv["concurrency"] for lv in done.results["concurrency"]] == [1, 2]
    assert done.results["single"]["errors"] == 0 and done.results["concurrency"][1]["throughput_tps"] > 0
    assert "speedup" in done.results["prefix_cache"]
    assert {r.url.port for r in env.engine_requests if r.url.path == "/v1/chat/completions"} == {inst.port}


async def test_a_benchmark_with_a_failing_engine_completes_with_errors_counted(env: Env) -> None:
    inst = await ready(env)
    env.node.serve(None, "/v1/chat/completions", lambda r: httpx.Response(500, text="x"))
    run = await env.container.bench.start(inst.id, {"concurrency": [2], "prompt_tokens": 14, "max_tokens": 2, "rounds": 1})
    await env.container.bench.wait(run.id)
    done = env.container.bench.get(run.id)
    assert done.state == "completed" and done.results["concurrency"][0]["errors"] == 2


async def test_a_run_cut_off_by_shutdown_or_a_crash_is_closed_as_failed_not_left_running(env: Env) -> None:
    inst = await ready(env)
    env.node.serve(None, "/v1/chat/completions", engine)
    run = await env.container.bench.start(inst.id, {"concurrency": [1], "prompt_tokens": 14, "max_tokens": 2, "rounds": 5})
    await env.container.bench.shutdown()                   # cancelled before the task ever ran
    done = env.container.bench.get(run.id)
    assert done.state == "failed" and done.finished_at is not None and "interrupted" in (done.error or "")
    env.container.store.execute("UPDATE bench_runs SET state='running', error=NULL, finished_at=NULL WHERE id=?", (run.id,))
    env.container.bench.recover()                          # a crashed console leaves the same thing behind
    assert env.container.bench.get(run.id).state == "failed"


def test_ports_helper_is_a_published_mapping() -> None:
    assert ports(18000) == [{"host_ip": "0.0.0.0", "host_port": 18000, "container_port": 8000}]   # noqa: S104
