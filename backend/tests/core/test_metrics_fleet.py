"""Metrics in v2: nodes are configured on link-up, pushed scrapes are routed to instances per host, per-host system series."""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from engine_console.domain.models import InstanceCreate
from engine_console.services.metrics import PREFIXES, MetricsService

from .conftest import Env
from .fakefleet import cgroup, ports, until
from .fakes import FakeMultiAdapter

P = "/api/v1"
TEXT1 = "running 2\nwaiting 1\nprompt_total 100\ngen_total 50\nkv 12.5\n"
TEXT2 = "running 3\nwaiting 0\nprompt_total 300\ngen_total 150\nkv 20\nunknown_key 9\n"


class Clock:
    def __init__(self, t: float = 10_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def use_clock(env: Env) -> Clock:
    clock = Clock()
    env.container.metrics._clock = clock  # noqa: SLF001
    return clock


def push(port: int | None = None, container: str | None = None, text: str = TEXT1, status: int = 200, engine: str = "fake") -> dict[str, Any]:
    return {"type": "metrics", "t": 1.0, "port": port, "container": container, "engine": engine, "status": status, "text": text}


async def managed(env: Env, name: str = "qwen", host: str = "local", **kw: Any) -> Any:
    env.seed_local_model(host_id=host)
    node = env.fleet.node(host)
    node.serve(None, "/health", lambda r: httpx.Response(200, text="ok"))
    inst = await env.container.lifecycle.create(InstanceCreate(engine="fake", repo_id="Qwen/Qwen3-32B", name=name, host_id=host,
                                                               params={"max_model_len": 4096}, **kw))
    await env.container.lifecycle.tick()
    return env.container.lifecycle.get(inst.id)


# ---- configuring the nodes ------------------------------------------------------------------------------------------------------
def test_families_cover_every_engine_the_hub_knows_and_honour_an_adapters_own_prefixes(env: Env) -> None:
    env.container.adapters.register(FakeMultiAdapter())
    fam = env.container.metrics.families()
    assert fam["vllm"] == ["vllm:"] and fam["sglang"] == ["sglang:"] and fam["llamacpp"] == ["llamacpp:"]
    assert fam["trtllm"] == ["trtllm", "tensorrt_llm"] and fam["dynamo"] == ["dynamo_"]
    assert fam["fakecpp"] == ["fakecpp:"]                      # declared by the adapter's `metric_prefixes`
    assert fam["fake"] == ["fake:"]                            # an adapter that declares none and is not in the table
    assert all(len(p) <= 64 for ps in fam.values() for p in ps) and len(fam) <= 16
    assert set(PREFIXES) <= set(fam)


async def test_each_node_is_configured_once_per_connection_and_again_after_a_reconnect(env: Env) -> None:
    m = env.container.metrics
    spark = env.add_host("spark", "Spark")
    await m.sync_hosts()
    await m.sync_hosts()
    for node in (env.node, spark):
        assert len(node.ops("metrics.configure")) == 1
    cfg = env.node.ops("metrics.configure")[0]
    assert cfg["interval_s"] == env.cfg.scrape_interval_s
    assert cfg["families"]["vllm"] == ["vllm:"] and cfg["families"]["fake"] == ["fake:"]
    spark.connected_since = 2000.0                               # the link dropped and came back
    await m.sync_hosts()
    assert len(spark.ops("metrics.configure")) == 2 and len(env.node.ops("metrics.configure")) == 1


async def test_a_host_that_is_down_is_not_configured_until_it_comes_up(env: Env) -> None:
    m = env.container.metrics
    spark = env.add_host("spark", "Spark")
    spark.state = "down"
    await m.sync_hosts()
    assert not spark.ops("metrics.configure")
    spark.state = "up"
    await m.sync_hosts()
    assert len(spark.ops("metrics.configure")) == 1


async def test_a_failed_configure_is_retried_on_the_next_round(env: Env) -> None:
    from engine_console.domain.fleet import NodeError
    env.node.fail["metrics.configure"] = NodeError("local", "timeout", "slow")
    await env.container.metrics.sync_hosts()
    env.node.fail.clear()
    await env.container.metrics.sync_hosts()
    assert len(env.node.ops("metrics.configure")) == 2 and env.node.metrics_config is not None


# ---- routing a push --------------------------------------------------------------------------------------------------------------------
async def test_l3_non_finite_values_never_reach_the_series(env: Env) -> None:
    """security review 2026-10-04 L3: Prometheus text allows NaN/+Inf; SSE then sent `Infinity`, which is not JSON."""
    import json

    clock = use_clock(env)
    inst = await managed(env)
    m = env.container.metrics
    m.ingest("local", push(port=inst.port, text="running NaN\nwaiting +Inf\nkv 1e999\ngen_total 5\n"))
    clock.t += 1
    pts = m.series(inst.id, "15m")["points"]
    vals = pts[-1]["values"]  # type: ignore[index]
    assert "requests_running" not in vals and "requests_waiting" not in vals and "kv_cache_usage_pct" not in vals
    json.dumps(m.series(inst.id, "15m"), allow_nan=False)                     # strict JSON, as the browser parses it


async def test_a_push_is_routed_to_the_managed_instance_by_gateway_port_or_container(env: Env) -> None:
    clock = use_clock(env)
    inst = await managed(env)
    m = env.container.metrics
    m.ingest("local", push(port=inst.port))
    clock.t += 1
    m.ingest("local", push(container=inst.container_name, text=TEXT2))
    clock.t += 1
    m.ingest("local", push(container=inst.container_name + "-gw", text=TEXT2))      # the gateway's name routes to its instance too
    pts = m.series(inst.id, "15m")["points"]
    assert len(pts) == 3 and pts[0]["values"]["requests_running"] == 2  # type: ignore[index]
    assert pts[1]["values"]["kv_cache_usage_pct"] == 20 and "unknown_key" not in pts[1]["values"]  # type: ignore[index]


async def test_pushes_are_routed_per_host_even_when_two_hosts_use_the_same_port_number(env: Env) -> None:
    clock = use_clock(env)
    a = await managed(env, "a")
    spark = env.add_host("spark", "Spark", port_range=(18000, 18099))
    b = await managed(env, "b", "spark")
    assert a.port == b.port == 18000 and spark is not None
    m = env.container.metrics
    m.ingest("local", push(port=18000, text="running 1\n"))
    clock.t += 1
    m.ingest("spark", push(port=18000, text="running 7\n"))
    assert m.latest(a.id).values["requests_running"] == 1 and m.latest(b.id).values["requests_running"] == 7  # type: ignore[union-attr]


async def test_pushes_for_externals_and_bare_endpoints_use_the_generic_map_or_their_adapter(env: Env) -> None:
    clock = use_clock(env)
    fake_vllm = type(env.container.adapters.get("fake"))()
    fake_vllm.id = "vllm"                                                       # stands in for the real vllm adapter's parser
    env.container.adapters.register(fake_vllm)
    env.node.add_container("vllm-a", engine="vllm", host_ports=ports(18500))
    env.node.add_endpoint(18500, container="vllm-a", engine="vllm", models=("m",))
    env.node.add_container("cpp-a", engine="llamacpp", image="ghcr.io/ggml-org/llama.cpp:server", host_ports=ports(18501, 8080))
    env.node.add_endpoint(18501, container="cpp-a", engine="llamacpp", models=("m",))
    env.node.add_endpoint(9100, engine="vllm", models=("bare",))
    m = env.container.metrics
    m.ingest("local", push(container="vllm-a", engine="vllm"))
    clock.t += 1
    m.ingest("local", push(port=18501, engine="llamacpp", text="llamacpp:requests_processing 4\nllamacpp:tokens_predicted_total 99\n"))
    clock.t += 1
    m.ingest("local", push(port=9100, engine="vllm", text="running 5\n"))
    assert m.latest("ext-local-vllm-a").values["requests_running"] == 2                       # type: ignore[union-attr]
    assert m.latest("ext-local-cpp-a").values == {"requests_running": 4.0, "generation_tokens_total": 99.0}  # type: ignore[union-attr]
    assert m.latest("ext-local-127-0-0-1-9100").values["requests_running"] == 5                # type: ignore[union-attr]


async def test_unknown_ports_failed_scrapes_and_garbage_are_ignored(env: Env) -> None:
    inst = await managed(env)
    m = env.container.metrics
    m.ingest("local", push(port=31337))                                   # nobody's port
    m.ingest("local", push(port=inst.port, status=500))                   # the scrape failed
    m.ingest("local", {"type": "metrics", "port": "x", "container": 5, "text": 7})   # malformed
    m.ingest("nope", push(port=inst.port))                                # a host that is not configured
    assert m.latest(inst.id) is None


async def test_rates_are_derived_from_counter_deltas_and_touch_the_instance(env: Env) -> None:
    clock = use_clock(env)
    inst = await managed(env)
    m = env.container.metrics
    m.ingest("local", push(port=inst.port))
    clock.t += 2
    m.ingest("local", push(port=inst.port, text=TEXT2))
    last = m.latest(inst.id).values                                       # type: ignore[union-attr]
    assert last["generation_tps"] == pytest.approx(50.0) and last["prompt_tps"] == pytest.approx(100.0)
    row = env.container.store.one("SELECT last_request_at FROM instances WHERE id=?", (inst.id,))
    assert row is not None and row["last_request_at"] is not None          # requests_running > 0 keeps it from idling out


async def test_rollups_average_per_hour_and_window_semantics(env: Env) -> None:
    clock = use_clock(env)
    inst = await managed(env)
    m = env.container.metrics
    m.ingest("local", push(port=inst.port))
    clock.t += 1
    m.ingest("local", push(port=inst.port, text=TEXT2))
    raw = m.series(inst.id, "15m")
    assert raw["resolution"] == "raw" and len(raw["points"]) == 2           # type: ignore[arg-type]
    hourly = m.series(inst.id, "1h")
    assert hourly["resolution"] == "1h" and hourly["points"][0]["values"]["requests_running"] == 2.5   # type: ignore[index]
    daily = m.series(inst.id, "7d")
    assert daily["resolution"] == "1d"


# ---- host RAM per instance -----------------------------------------------------------------------------------------------------------
async def test_host_ram_keys_ride_along_with_engine_metrics(env: Env) -> None:
    inst = await managed(env)
    env.node.cg[inst.container_name] = cgroup()
    env.container.metrics.ingest("local", push(port=inst.port))
    v = env.container.metrics.latest(inst.id).values            # type: ignore[union-attr]
    assert v["host_ram_gib"] == pytest.approx(68.6, abs=1e-3) and v["host_ram_anon_gib"] == pytest.approx(3.3, abs=1e-3)
    assert v["host_ram_shmem_gib"] == pytest.approx(64.0, abs=1e-3) and v["host_ram_cache_gib"] == pytest.approx(0.5, abs=1e-3)
    assert v["requests_running"] == 2


async def test_an_engine_without_metrics_still_gets_a_host_ram_series(env: Env) -> None:
    clock = use_clock(env)
    env.node.add_container("ollama-a", engine="ollama", image="ollama/ollama:0.5", host_ports=ports(18502, 11434), started_at=1.0)
    env.node.add_endpoint(18502, container="ollama-a", engine="ollama", models=("llama3",), health=None, models_status=200)
    env.node.cg["ollama-a"] = cgroup()
    clock.t += 100
    await env.container.metrics.scrape_once()
    pts = env.container.metrics.series("ext-local-ollama-a")["points"]
    assert len(pts) == 1 and set(pts[0]["values"]) == {"host_ram_gib", "host_ram_anon_gib", "host_ram_cache_gib", "host_ram_shmem_gib"}  # type: ignore[index]


async def test_instances_that_are_stopped_failed_or_on_a_down_host_get_no_points(env: Env) -> None:
    clock = use_clock(env)
    inst = await managed(env)
    env.node.cg[inst.container_name] = cgroup()
    env.node.state = "down"
    clock.t += 100
    await env.container.metrics.scrape_once()
    assert env.container.metrics.latest(inst.id) is None


# ---- the pull fallback ------------------------------------------------------------------------------------------------------------------
async def test_a_ready_managed_instance_nobody_pushed_for_is_read_through_the_node(env: Env) -> None:
    clock = use_clock(env)
    inst = await managed(env)
    env.node.serve(inst.port, "/metrics", lambda r: httpx.Response(200, text=TEXT1))
    clock.t += 100
    await env.container.metrics.scrape_once()
    assert env.container.metrics.latest(inst.id).values["requests_running"] == 2         # type: ignore[union-attr]
    assert any(r.url.path == "/metrics" and r.url.port == inst.port for r in env.node.http_requests)
    # pushes take over: while they keep arriving the hub does not poll
    env.container.metrics.ingest("local", push(port=inst.port))
    n = len(env.node.http_requests)
    clock.t += env.cfg.scrape_interval_s
    await env.container.metrics.scrape_once()
    assert len(env.node.http_requests) == n


async def test_a_failing_metrics_endpoint_is_skipped_without_raising(env: Env) -> None:
    clock = use_clock(env)
    inst = await managed(env)
    env.node.serve(inst.port, "/metrics", lambda r: httpx.Response(503, text="no"))
    clock.t += 100
    await env.container.metrics.scrape_once()
    assert env.container.metrics.latest(inst.id) is None


# ---- system series per host ---------------------------------------------------------------------------------------------------------------
async def test_system_series_are_kept_per_host_and_one_point_per_new_sample(env: Env) -> None:
    clock = use_clock(env)
    spark = env.add_host("spark", "Spark")
    spark.mem.update(total=128.0, available=100.0, shmem=2.0, swap_total=0.0, swap_free=0.0)
    m = env.container.metrics
    await m.scrape_once()
    await m.scrape_once()                                    # the same samples again: nothing new, nothing recorded
    assert len(m.system_series("15m")["points"]) == 1       # type: ignore[arg-type]
    clock.t += 2
    env.node.tick_sample()
    await m.scrape_once()
    local = m.system_series("15m", "local")
    assert len(local["points"]) == 2 and local["host_id"] == "local"   # type: ignore[arg-type]
    v = local["points"][0]["values"]                         # type: ignore[index]
    assert set(v) == {"ram_used_gib", "ram_available_gib", "shmem_gib", "swap_used_gib"}
    assert v["ram_used_gib"] == 192.0 and v["ram_available_gib"] == 54.0 and v["swap_used_gib"] == 24.0 and v["shmem_gib"] == 66.0
    sp = m.system_series("15m", "spark")["points"][0]["values"]  # type: ignore[index]
    assert sp["ram_used_gib"] == 28.0 and sp["swap_used_gib"] == 0.0
    hourly = m.system_series("1h", "spark")
    assert hourly["resolution"] == "1h" and hourly["points"][0]["values"]["shmem_gib"] == 2.0  # type: ignore[index]


def test_system_metrics_api_takes_a_host_and_validates_it(client: TestClient, env: Env) -> None:
    client.portal.call(env.container.metrics.scrape_once)  # type: ignore[union-attr]
    body = client.get(f"{P}/metrics/system", params={"window": "15m"}).json()
    assert body["host_id"] == "local" and len(body["points"]) == 1
    assert client.get(f"{P}/metrics/system", params={"host": "local"}).json()["points"]
    assert client.get(f"{P}/metrics/system", params={"host": "nope"}).status_code == 404
    assert client.get(f"{P}/metrics/system", params={"host": "Bad Host"}).status_code == 422
    assert client.get(f"{P}/metrics/system", params={"window": "soon"}).status_code == 400


async def test_the_sse_event_for_a_system_point_names_its_host(env: Env) -> None:
    env.add_host("spark", "Spark")
    events: list[Any] = []

    async def listen() -> None:
        async for ev in env.container.bus.subscribe("metrics"):
            events.append(ev)

    task = asyncio.create_task(listen())
    await asyncio.sleep(0)
    await env.container.metrics.scrape_once()
    await until(lambda: len(events) >= 2, what="both hosts' system events")
    task.cancel()
    assert {e["host_id"] for e in events} == {"local", "spark"} and {e["instance_id"] for e in events} == {"system"}


# ---- the bus ----------------------------------------------------------------------------------------------------------------------------------------
async def test_consume_routes_what_the_fleet_publishes_on_the_bus(env: Env) -> None:
    inst = await managed(env)
    task = asyncio.create_task(env.container.metrics.consume())
    await asyncio.sleep(0)
    env.container.bus.publish("node.metrics", {"host_id": "local", "msg": push(port=inst.port)})
    env.container.bus.publish("node.metrics", {"host_id": "local", "msg": "not a dict"})         # must not kill the consumer
    env.container.bus.publish("node.metrics", {"host_id": "local", "msg": push(port=inst.port, text=TEXT2)})
    await until(lambda: len(env.container.metrics.series(inst.id)["points"]) == 2,  # type: ignore[arg-type]
                what="both pushes to be recorded")
    task.cancel()


async def test_run_configures_nodes_then_samples_until_cancelled(env: Env) -> None:
    task = asyncio.create_task(env.container.metrics.run())
    await until(lambda: bool(env.node.ops("metrics.configure")) and bool(env.container.metrics.system_series("15m")["points"]),
                what="the loop to configure the node and sample it")
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


# ---- prometheus ------------------------------------------------------------------------------------------------------------------------------------
def test_prometheus_gauges_for_hosts_gpus_and_instance_vram(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    env.engine_up()
    spark = env.add_host("spark", "Spark")
    spark.state = "down"
    inst = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "gpu_ids": [1],
                                               "params": {"max_model_len": 4096}}).json()
    env.node.set_gpu_free([96.0, 36.0])
    env.node.gpu_procs = {1: [{"pid": 1, "container": inst["container_name"], "used_bytes": 60 * 1024**3}]}
    env.node.cg[inst["container_name"]] = cgroup()
    text = client.get("/metrics").text
    assert 'engine_console_host_up{host="local"} 1.0' in text and 'engine_console_host_up{host="spark"} 0.0' in text
    assert 'engine_console_host_gpu_used_gib{gpu="1",host="local"} 60.0' in text
    assert 'engine_console_host_gpu_used_gib{gpu="0",host="local"} 0.0' in text
    assert 'host="spark"' not in text.split("engine_console_host_gpu_used_gib")[1].split("# HELP")[0]   # a down host reports no GPUs
    assert f'engine_console_instance_vram_gib{{gpu="1",host="local",instance="{inst["id"]}"}} 60.0' in text
    assert "engine_console_host_ram_used_gib 192.0" in text and "engine_console_host_swap_used_gib 24.0" in text
    assert f'engine_console_instance_host_ram_gib{{instance="{inst["id"]}"}}' in text
    assert 'engine_console_instances{state="starting"} 1.0' in text
    # label sets of things that are gone do not linger
    client.delete(f"{P}/instances/{inst['id']}")
    assert f'instance="{inst["id"]}"' not in client.get("/metrics").text


def test_a_service_can_be_built_without_adapters(env: Env) -> None:
    from engine_console.adapters import AdapterRegistry
    m = MetricsService(env.container.store, env.container.lifecycle, AdapterRegistry(), env.fleet,  # type: ignore[arg-type]
                       env.container.nodehttp, env.container.bus)
    assert set(m.families()) == set(PREFIXES)
