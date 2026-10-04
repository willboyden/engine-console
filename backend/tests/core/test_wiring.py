"""The composition root and the app's lifespan: what is wired to what, what starts and stops, and the CLI entry point."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine_console import main as main_module
from engine_console.domain.fleet import HostCounts
from engine_console.main import create_app
from engine_console.services.hosts import HostsError

from .conftest import Env
from .fakefleet import cgroup, until

P = "/api/v1"


def test_the_fleet_is_told_how_to_name_instances_and_count_engines_and_models(env: Env) -> None:
    assert env.fleet.resolver == env.container.lifecycle.instance_for_container
    assert env.fleet.counts == env.container.lifecycle.counts
    assert env.container.fleet is env.fleet and env.container.library is env.library       # H1's router reads `Container.fleet`
    env.seed_local_model()
    counts = env.fleet.counts("local")                                                     # type: ignore[misc]
    assert isinstance(counts, HostCounts) and counts.models == 1 and counts.engines_total == 0


def test_every_service_that_talks_to_a_machine_goes_through_the_one_fleet(env: Env) -> None:
    c = env.container
    for svc in (c.lifecycle, c.scripts, c.hardware, c.nodehttp, c.metrics, c.library):
        found = [v for v in vars(svc).values() if v is env.fleet or getattr(v, "fleet", None) is env.fleet
                 or getattr(v, "_fleet", None) is env.fleet]
        assert found, type(svc).__name__


def test_the_lifespan_starts_and_stops_the_fleet_and_the_library_and_closes_what_the_last_run_left_open(env: Env) -> None:
    env.seed_local_model()
    env.container.store.execute("INSERT INTO bench_runs(id,instance_id,params,suite,state,results,created_at) "
                                "VALUES('b1','i','{}','{}','running','{}',1)")
    assert not env.fleet.started and not env.library.started
    with TestClient(create_app(env.cfg, env.container), client=("127.0.0.1", 1)) as c:
        assert env.fleet.started and env.library.started
        assert c.get(f"{P}/health").status_code == 200
        assert env.container.bench.get("b1").state == "failed"                         # a crashed run does not look alive
    assert not env.fleet.started and not env.library.started


def test_health_lists_engines_and_the_loopback_warning_follows_the_setting(env: Env) -> None:
    with TestClient(create_app(env.cfg, env.container), client=("127.0.0.1", 1)) as c:
        body = c.get(f"{P}/health").json()
        assert body["engines"] == ["fake"] and any("trust_loopback" in w for w in body["warnings"])


def test_the_background_supervisor_and_metrics_loops_run_when_enabled(env: Env) -> None:
    env.cfg.background_tasks = True
    env.cfg.supervisor_interval_s = 0.02
    env.cfg.scrape_interval_s = 0.02
    env.seed_local_model()
    env.engine_up()
    with TestClient(create_app(env.cfg, env.container), client=("127.0.0.1", 1), headers={"X-Engine-Console": "1"}) as c:
        inst = c.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 4096}}).json()
        env.node.cg[inst["container_name"]] = cgroup()
        deadline = time.time() + 5
        while time.time() < deadline and c.get(f"{P}/instances/{inst['id']}").json()["state"] != "ready":
            time.sleep(0.02)
        assert c.get(f"{P}/instances/{inst['id']}").json()["state"] == "ready"           # the supervisor promoted it by itself
        while time.time() < deadline and not c.get(f"{P}/metrics/system").json()["points"]:
            time.sleep(0.02)
        assert c.get(f"{P}/metrics/system").json()["points"]                              # and the metrics loop sampled the host
    assert env.node.ops("metrics.configure")                                              # nodes are configured when the loop starts


def test_a_supervisor_round_that_raises_does_not_end_the_loop(env: Env, caplog: pytest.LogCaptureFixture) -> None:
    import asyncio

    calls: list[int] = []

    async def flaky() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")

    async def run() -> None:
        task = asyncio.create_task(main_module._loop(flaky, 0.01))  # noqa: SLF001
        await until(lambda: len(calls) > 2, what="the loop to keep running after a failed round")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    assert len(calls) > 2 and "background loop error" in caplog.text


# ---- the entry point -------------------------------------------------------------------------------------------------------------------
def test_the_hosts_subcommand_goes_to_the_hosts_cli_with_its_own_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    import engine_console.cli as cli
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "hosts_cli", lambda argv: (seen.append(argv), 7)[1])
    monkeypatch.setattr(sys, "argv", ["engine-console", "hosts", "check", "--timeout", "3"])
    with pytest.raises(SystemExit) as ei:
        main_module.run()
    assert ei.value.code == 7 and seen == [["check", "--timeout", "3"]]


def test_a_hosts_file_the_console_refuses_stops_it_with_a_message_and_status_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                                              capsys: pytest.CaptureFixture[str]) -> None:
    def refuse(cfg: object = None, container: object = None) -> None:
        raise HostsError("hosts.json is group-writable: refusing to use it")

    monkeypatch.setattr(main_module, "create_app", refuse)
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(sys, "argv", ["engine-console"])
    with pytest.raises(SystemExit) as ei:
        main_module.run()
    assert ei.value.code == 2 and "group-writable" in capsys.readouterr().err


def test_serving_starts_uvicorn_with_the_configured_address(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(main_module, "create_app", lambda cfg=None, container=None: "app")
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module.uvicorn, "run", lambda app, **kw: started.append((app, kw)))
    monkeypatch.setattr(sys, "argv", ["engine-console"])
    main_module.run()
    assert started and started[0][0] == "app" and started[0][1]["host"] == "127.0.0.1" and started[0][1]["port"] == 8791
