"""Lifecycle scripts registered in node policies: listing, validation against the node-reported policy, jobs, masking,
one-job-at-a-time, SSE, and the script instances' start/stop/restart that map onto them."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from engine_console.domain.errors import BadRequest, Conflict, NotFound, ProblemError
from engine_console.domain.fleet import NodeError

from .conftest import Env
from .fakefleet import BIG_MODEL_SCRIPT, FakeNode, ScriptRun, ports, until

P = "/api/v1"


def register(node: FakeNode, **kw: Any) -> None:
    node.policy["scripts"] = [{**json.loads(json.dumps(BIG_MODEL_SCRIPT)), **kw}]


async def finished(env: Env, job_id: str) -> Any:
    await env.container.scripts.wait(job_id)
    return env.container.scripts.job(job_id)


# ---- what the policies register ------------------------------------------------------------------------------------------------
def test_scripts_are_listed_per_host_with_actions_status_and_container_patterns(env: Env) -> None:
    register(env.node)
    spark = env.add_host("spark", "Spark")
    register(spark, name="other", label="Other")
    env.node.script_status["big-model"] = {"name": "big-model", "status": "up", "detail": "1 container", "checked_at": 5.0}
    items = env.container.scripts.scripts()
    assert [(s.host_id, s.name) for s in items] == [("local", "big-model"), ("spark", "other")]
    s = items[0]
    assert (s.label, s.engine, s.status, s.detail, s.checked_at) == ("Big model (2 GPUs)", "sglang", "up", "1 container", 5.0)
    assert s.containers == ["myproj-sglang-*"]
    assert [(a.name, a.confirm, a.danger, a.args) for a in s.actions] == [
        ("up", True, False, []), ("down", False, True, []), ("status", False, False, []),
        ("switch", True, False, ["model-a", "model-b"])]
    assert items[1].status == "unknown"                                            # no status pushed yet
    assert [i.name for i in env.container.scripts.scripts("spark")] == ["other"]
    assert "argv" not in json.dumps([i.model_dump() for i in items])                # the hub never sees a script's argv


def test_an_unreachable_hosts_script_status_is_unknown_not_last_seen(env: Env) -> None:
    register(env.node)
    env.node.script_status["big-model"] = {"name": "big-model", "status": "up", "detail": "x"}
    env.node.state = "down"
    s = env.container.scripts.scripts()[0]
    assert s.status == "unknown" and "not connected" in (s.detail or "")


def test_api_lists_scripts_as_a_page(client: TestClient, env: Env) -> None:
    register(env.node)
    body = client.get(f"{P}/scripts").json()
    assert [i["name"] for i in body["items"]] == ["big-model"] and body["next_cursor"] is None
    assert client.get(f"{P}/scripts", params={"host": "spark"}).json()["items"] == []
    assert client.get(f"{P}/scripts", params={"host": "Bad Host"}).status_code == 422


# ---- validation before anything is sent ------------------------------------------------------------------------------------------
async def test_run_validates_host_script_action_and_argument_against_the_reported_policy(env: Env) -> None:
    register(env.node)
    s = env.container.scripts
    with pytest.raises(NotFound) as e1:
        await s.run("nope", "big-model", "up")
    assert e1.value.code == "unknown_host"
    with pytest.raises(NotFound) as e2:
        await s.run("local", "not-a-script", "up")
    assert e2.value.code == "unknown_script"
    with pytest.raises(NotFound) as e3:
        await s.run("local", "big-model", "format-disk")
    assert e3.value.code == "unknown_action" and "up, down, status, switch" in e3.value.detail
    with pytest.raises(BadRequest) as e4:
        await s.run("local", "big-model", "switch")
    assert e4.value.code == "arg_required"
    with pytest.raises(BadRequest) as e5:
        await s.run("local", "big-model", "switch", "model-c; rm -rf /")
    assert e5.value.code == "invalid_arg"
    with pytest.raises(BadRequest) as e6:
        await s.run("local", "big-model", "up", "surprise")
    assert e6.value.code == "invalid_arg" and "takes none" in e6.value.detail
    assert not env.node.script_calls and not env.node.ops("scripts.run")               # nothing reached the node


async def test_run_refuses_when_the_node_is_down_or_in_monitor_mode(env: Env) -> None:
    register(env.node)
    env.node.set_mode("monitor")
    with pytest.raises(ProblemError) as e1:
        await env.container.scripts.run("local", "big-model", "up")
    assert e1.value.status == 409 and e1.value.code == "monitor_only" and "testbox" in e1.value.detail
    env.node.set_mode("manage")
    env.node.state, env.node.link_error = "down", "no route to host"
    with pytest.raises(ProblemError) as e2:
        await env.container.scripts.run("local", "big-model", "up")
    assert e2.value.status == 503 and "no route to host" in e2.value.detail
    assert not env.node.script_calls


# ---- jobs --------------------------------------------------------------------------------------------------------------------------
async def test_a_job_streams_the_scripts_lines_and_completes_with_the_return_code(env: Env) -> None:
    register(env.node)
    env.node.script_runs[("big-model", "up")] = ScriptRun(lines=[("out", "starting engine"), ("err", "warming up"), ("out", "done")])
    job = await env.container.scripts.run("local", "big-model", "up")
    assert job.state == "running" and job.host_id == "local" and job.name == "big-model" and job.action == "up"
    done = await finished(env, job.id)
    assert done.state == "completed" and done.rc == 0 and done.finished_at is not None
    assert done.output_tail.splitlines() == ["starting engine", "warming up", "done"]
    assert env.node.script_calls == [("big-model", "up", None)]
    assert env.node.ops("scripts.run") == [{"name": "big-model", "action": "up"}]       # no `arg` key when there is none


async def test_the_one_allowed_argument_is_passed_on(env: Env) -> None:
    register(env.node)
    job = await env.container.scripts.run("local", "big-model", "switch", "model-b")
    await finished(env, job.id)
    assert env.node.ops("scripts.run") == [{"name": "big-model", "action": "switch", "arg": "model-b"}]
    assert env.container.scripts.job(job.id).arg == "model-b"


async def test_a_nonzero_return_code_is_a_failed_job(env: Env) -> None:
    register(env.node)
    env.node.script_runs[("big-model", "up")] = ScriptRun(lines=[("err", "no GPU memory")], rc=3)
    done = await finished(env, (await env.container.scripts.run("local", "big-model", "up")).id)
    assert done.state == "failed" and done.rc == 3 and "no GPU memory" in done.output_tail


async def test_a_node_error_mid_run_is_a_failed_job_with_the_error_in_its_output(env: Env) -> None:
    register(env.node)
    env.node.script_runs[("big-model", "up")] = ScriptRun(lines=[("out", "half way")],
                                                          error=NodeError("local", "timeout", "script ran too long"))
    done = await finished(env, (await env.container.scripts.run("local", "big-model", "up")).id)
    assert done.state == "failed" and done.rc is None
    assert "half way" in done.output_tail and "[timeout] script ran too long" in done.output_tail


async def test_a_refusal_by_the_node_at_open_time_is_a_failed_job(env: Env) -> None:
    register(env.node)
    env.node.fail["scripts.run"] = NodeError("local", "busy", "the node is busy")
    done = await finished(env, (await env.container.scripts.run("local", "big-model", "up")).id)
    assert done.state == "failed" and "[busy]" in done.output_tail


async def test_output_is_masked_in_the_tail_and_in_what_late_subscribers_see(env: Env) -> None:
    register(env.node)
    leak = "hf_abcdefghijkl1234"
    env.node.script_runs[("big-model", "up")] = ScriptRun(lines=[
        ("out", f"token {leak}"), ("out", "Authorization: Bearer abcdefghijklmnop"), ("err", "key ec_abcdefghijklmnopqrstuv")])
    job = await env.container.scripts.run("local", "big-model", "up")
    done = await finished(env, job.id)
    assert leak not in done.output_tail and "abcdefghijklmnop" not in done.output_tail and "ec_abcdef" not in done.output_tail
    assert done.output_tail.count("[redacted]") == 3
    events = [e async for e in env.container.scripts.stream(job.id)]
    assert leak not in json.dumps(events) and events[-1][0] == "end"


async def test_the_tail_is_bounded_and_keeps_the_end(env: Env) -> None:
    register(env.node)
    env.node.script_runs[("big-model", "up")] = ScriptRun(lines=[("out", f"line {i:05d} " + "x" * 80) for i in range(1000)])
    done = await finished(env, (await env.container.scripts.run("local", "big-model", "up")).id)
    assert len(done.output_tail) <= 20_000 and done.output_tail.splitlines()[-1].startswith("line 00999")


async def test_control_characters_never_split_one_line_into_two(env: Env) -> None:
    register(env.node)
    env.node.script_runs[("big-model", "up")] = ScriptRun(lines=[("out", "a\x1b[31mred\x0cfeed sep")])
    done = await finished(env, (await env.container.scripts.run("local", "big-model", "up")).id)
    assert len(done.output_tail.splitlines()) == 1 and "\x1b" not in done.output_tail


# ---- one job per (host, script) -----------------------------------------------------------------------------------------------------
async def test_a_second_run_of_the_same_script_is_refused_until_the_first_ends(env: Env) -> None:
    register(env.node)
    gate = asyncio.Event()
    env.node.script_runs[("big-model", "up")] = ScriptRun(lines=[("out", "started")], gate=gate)
    first = await env.container.scripts.run("local", "big-model", "up")
    await until(lambda: bool(env.node.script_calls), what="the first run to reach the node")
    with pytest.raises(Conflict) as ei:
        await env.container.scripts.run("local", "big-model", "down")
    assert ei.value.code == "script_busy" and first.id in ei.value.detail
    assert env.container.scripts.busy("local") == {"big-model": "up"}
    gate.set()
    assert (await finished(env, first.id)).state == "completed"
    assert env.container.scripts.busy("local") == {}
    second = await env.container.scripts.run("local", "big-model", "down")            # free again
    assert (await finished(env, second.id)).state == "completed"


async def test_other_scripts_and_other_hosts_run_in_parallel(env: Env) -> None:
    register(env.node)
    env.node.policy["scripts"].append({**json.loads(json.dumps(BIG_MODEL_SCRIPT)), "name": "small-model", "label": "Small"})
    spark = env.add_host("spark", "Spark")
    register(spark)
    gate = asyncio.Event()
    for node in (env.node, spark):
        for name in ("big-model", "small-model"):
            node.script_runs[(name, "up")] = ScriptRun(lines=[("out", "x")], gate=gate)
    jobs = [await env.container.scripts.run("local", "big-model", "up"),
            await env.container.scripts.run("local", "small-model", "up"),
            await env.container.scripts.run("spark", "big-model", "up")]
    await until(lambda: len(env.node.script_calls) == 2 and len(spark.script_calls) == 1, what="all three runs to reach their nodes")
    assert all(env.container.scripts.job(j.id).state == "running" for j in jobs)
    gate.set()
    for j in jobs:
        assert (await finished(env, j.id)).state == "completed"


async def test_finished_jobs_are_kept_up_to_a_bound_and_running_ones_never_evicted(env: Env) -> None:
    from engine_console.services import scripts as sc
    register(env.node)
    for _ in range(sc.MAX_JOBS + 5):
        await finished(env, (await env.container.scripts.run("local", "big-model", "status")).id)
    assert len(env.container.scripts.jobs()) <= sc.MAX_JOBS + 1


# ---- restart = down then up ------------------------------------------------------------------------------------------------------------
async def test_restart_is_down_then_up_as_one_job(env: Env) -> None:
    register(env.node)
    job = await env.container.scripts.restart("local", "big-model")
    assert job.action == "restart"
    done = await finished(env, job.id)
    assert done.state == "completed" and [c[1] for c in env.node.script_calls] == ["down", "up"]
    assert done.output_tail.splitlines()[0] == "== down =="


async def test_restart_never_starts_what_failed_to_stop(env: Env) -> None:
    register(env.node)
    env.node.script_runs[("big-model", "down")] = ScriptRun(lines=[("err", "cannot stop")], rc=1)
    done = await finished(env, (await env.container.scripts.restart("local", "big-model")).id)
    assert done.state == "failed" and done.rc == 1 and [c[1] for c in env.node.script_calls] == ["down"]


# ---- SSE ----------------------------------------------------------------------------------------------------------------------------------
async def test_stream_replays_history_for_a_late_subscriber_then_ends(env: Env) -> None:
    register(env.node)
    env.node.script_runs[("big-model", "up")] = ScriptRun(lines=[("out", "one"), ("out", "two")])
    job = await env.container.scripts.run("local", "big-model", "up")
    await finished(env, job.id)
    events = [e async for e in env.container.scripts.stream(job.id)]
    assert [e for e, _ in events] == ["line", "line", "end"] and [d for _, d in events[:2]] == ["one", "two"]
    assert json.loads(events[-1][1])["state"] == "completed"


async def test_stream_follows_a_running_job_live(env: Env) -> None:
    register(env.node)
    gate = asyncio.Event()
    env.node.script_runs[("big-model", "up")] = ScriptRun(lines=[("out", "first"), ("out", "second")], gate=gate)
    job = await env.container.scripts.run("local", "big-model", "up")
    seen: list[tuple[str, str]] = []

    async def follow() -> None:
        async for ev in env.container.scripts.stream(job.id):
            seen.append(ev)

    task = asyncio.create_task(follow())
    await until(lambda: bool(seen), what="the first line")
    assert seen == [("line", "first")]                       # live: the first line arrived while the job is still running
    gate.set()
    await asyncio.wait_for(task, 2)
    assert [e for e, _ in seen] == ["line", "line", "end"]


async def test_stream_once_sends_what_there_is_and_closes(env: Env) -> None:
    register(env.node)
    gate = asyncio.Event()
    env.node.script_runs[("big-model", "up")] = ScriptRun(lines=[("out", "first"), ("out", "second")], gate=gate)
    job = await env.container.scripts.run("local", "big-model", "up")
    await until(lambda: bool(env.container.scripts.job(job.id).output_tail), what="the first line")
    events = [e async for e in env.container.scripts.stream(job.id, once=True)]
    assert [e for e, _ in events] == ["line", "end"] and json.loads(events[-1][1])["state"] == "running"
    gate.set()
    await finished(env, job.id)


def test_api_run_job_and_stream(client: TestClient, env: Env) -> None:
    register(env.node)
    env.node.script_runs[("big-model", "up")] = ScriptRun(lines=[("out", "hello"), ("out", "token hf_abcdefghijkl1234")])
    r = client.post(f"{P}/scripts/local/big-model/up")
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["state"] == "running" and job["action"] == "up"
    client.portal.call(env.container.scripts.wait, job["id"])  # type: ignore[union-attr]
    got = client.get(f"{P}/scripts/jobs/{job['id']}").json()
    assert got["state"] == "completed" and "hello" in got["output_tail"] and "hf_abcdef" not in got["output_tail"]
    stream = client.get(f"{P}/scripts/jobs/{job['id']}/stream")
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert stream.text.count("event: line") == 2 and "event: end" in stream.text and "hf_abcdef" not in stream.text
    once = client.get(f"{P}/scripts/jobs/{job['id']}/stream", params={"once": "true"})
    assert "event: end" in once.text
    assert client.get(f"{P}/scripts/jobs/nope").status_code == 404
    assert client.get(f"{P}/scripts/jobs/nope/stream").json()["code"] == "unknown_job"


def test_api_validation_errors_are_problems(client: TestClient, env: Env) -> None:
    register(env.node)
    assert client.post(f"{P}/scripts/local/nope/up").json()["code"] == "unknown_script"
    assert client.post(f"{P}/scripts/local/big-model/nope").json()["code"] == "unknown_action"
    assert client.post(f"{P}/scripts/nope/big-model/up").json()["code"] == "unknown_host"
    bad = client.post(f"{P}/scripts/local/big-model/switch", json={"arg": "model-c"})
    assert bad.status_code == 400 and bad.json()["code"] == "invalid_arg"
    ok = client.post(f"{P}/scripts/local/big-model/switch", json={"arg": "model-a"})
    assert ok.status_code == 202 and ok.json()["arg"] == "model-a"
    client.portal.call(env.container.scripts.wait, ok.json()["id"])  # type: ignore[union-attr]
    for evil in ("..%2Fx", "a b", "x" * 40, "-x"):
        assert client.post(f"{P}/scripts/local/{evil}/up").status_code in (404, 405, 422), evil   # never a job
    assert not env.node.script_calls[1:]


def test_running_a_script_is_admin_only(env: Env) -> None:
    from engine_console.main import create_app
    register(env.node)
    with TestClient(create_app(env.cfg, env.container), client=("192.168.1.9", 1)) as remote:
        boot = (env.cfg.data_dir / "bootstrap-admin.key").read_text().strip()
        adm = {"Authorization": f"Bearer {boot}"}
        viewer = remote.post(f"{P}/keys", json={"name": "v", "role": "viewer"}, headers=adm).json()["secret"]
        vh = {"Authorization": f"Bearer {viewer}"}
        assert remote.get(f"{P}/scripts", headers=vh).status_code == 200                    # reading is fine
        assert remote.post(f"{P}/scripts/local/big-model/up", headers=vh).status_code == 403
        assert not env.node.script_calls
        r = remote.post(f"{P}/scripts/local/big-model/up", headers=adm)
        assert r.status_code == 202
        remote.portal.call(env.container.scripts.wait, r.json()["id"])  # type: ignore[union-attr]


# ---- script instances: start / stop / restart map to up / down ---------------------------------------------------------------------------------
def test_script_instance_start_and_stop_run_the_scripts_up_and_down(client: TestClient, env: Env) -> None:
    register(env.node)
    inst = client.get(f"{P}/instances/scr-local-big-model").json()
    assert inst["state"] == "stopped" and inst["controls"]["start"] is True
    started = client.post(f"{P}/instances/scr-local-big-model/start")
    assert started.status_code == 200 and started.json()["state"] == "starting" and "script: up is running" in started.json()["phase"]
    client.portal.call(env.container.scripts.wait, env.container.scripts.jobs()[0].id)  # type: ignore[union-attr]
    assert env.node.script_calls == [("big-model", "up", None)]
    env.node.add_container("myproj-sglang-a", engine="sglang", host_ports=ports(18510, 30000))
    env.node.add_endpoint(18510, container="myproj-sglang-a", engine="sglang", models=("m",))
    assert client.get(f"{P}/instances/scr-local-big-model").json()["state"] == "ready"
    assert client.post(f"{P}/instances/scr-local-big-model/start").status_code == 409        # already running: no `start`
    stopped = client.post(f"{P}/instances/scr-local-big-model/stop")
    assert stopped.status_code == 200 and stopped.json()["state"] == "stopping"
    client.portal.call(env.container.scripts.wait, env.container.scripts.jobs()[0].id)  # type: ignore[union-attr]
    assert [c[1] for c in env.node.script_calls] == ["up", "down"]


def test_script_instance_restart_is_down_then_up_and_never_touches_docker(client: TestClient, env: Env) -> None:
    register(env.node)
    r = client.post(f"{P}/instances/scr-local-big-model/restart")
    assert r.status_code == 200, r.text
    client.portal.call(env.container.scripts.wait, env.container.scripts.jobs()[0].id)  # type: ignore[union-attr]
    assert [c[1] for c in env.node.script_calls] == ["down", "up"]
    assert not any(o in ("docker.stop", "docker.rm", "docker.run", "docker.external") for o, _ in env.node.calls)


def test_script_instance_without_the_action_or_in_monitor_mode_refuses_with_the_reason(client: TestClient, env: Env) -> None:
    register(env.node)
    env.node.set_mode("monitor")
    r = client.post(f"{P}/instances/scr-local-big-model/start")
    assert r.status_code == 409 and r.json()["code"] == "instance_not_managed" and "monitor mode" in r.json()["detail"]
    env.node.set_mode("manage")
    env.node.policy["scripts"][0]["actions"] = [a for a in env.node.policy["scripts"][0]["actions"] if a["name"] != "up"]
    r = client.post(f"{P}/instances/scr-local-big-model/start")
    assert r.status_code == 409
    for path in ("logs", "command"):
        assert client.get(f"{P}/instances/scr-local-big-model/{path}").status_code == 409
    assert client.delete(f"{P}/instances/scr-local-big-model").status_code == 409
    assert not env.node.script_calls
