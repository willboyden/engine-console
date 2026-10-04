"""Managed instances across the fleet: what the hub asks each host's node, in what order, and what it shows."""
from __future__ import annotations

import json
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from engine_console.domain.errors import Conflict, ProblemError
from engine_console.domain.fleet import NodeError
from engine_console.domain.models import InstanceCreate, InstancePatch, SettingsPatch
from engine_console.domain.refs import ModelRef

from .conftest import Env
from .fakefleet import FakeNode, cgroup, ports
from .fakes import FakeMultiAdapter

P = "/api/v1"
SECRET = "sk-live-abcdef123456"


def request(**kw: Any) -> InstanceCreate:
    base: dict[str, Any] = {"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 4096}}
    return InstanceCreate(**{**base, **kw})


async def launch(env: Env, **kw: Any) -> Any:
    host = kw.get("host_id", "local")
    if env.library.find(host, ModelRef(format="hf", repo_id="Qwen/Qwen3-32B")) is None:
        env.seed_local_model(host_id=host)
    return await env.container.lifecycle.create(request(**kw))


def codes(e: ProblemError) -> dict[str, str]:
    return {c["code"]: c["level"] for c in e.extra["preflight"]["checks"]}


async def preflight_codes(env: Env, **kw: Any) -> dict[str, str]:
    report, _ = await env.container.lifecycle.preflight(request(**kw))
    return {c.code: c.level for c in report.checks}


# ---- the happy path -----------------------------------------------------------------------------------------------
async def test_create_runs_engine_then_gateway_in_order_and_publishes_the_gateway_port(env: Env) -> None:
    inst = await launch(env)
    ops = [o for o, _ in env.node.calls if o in ("net.ensure_internal", "docker.rm", "docker.run", "gateway.start",
                                                 "docker.image_present")]
    assert ops.index("net.ensure_internal") < ops.index("docker.run") < ops.index("gateway.start")
    run = env.node.run_specs[0]
    gw = env.node.ops("gateway.start")[0]
    assert inst.state == "starting" and inst.host_id == "local" and inst.host_label == "testbox"
    assert 18000 <= inst.port <= 18099 and gw["host_port"] == inst.port and gw["engine_host"] == run["name"]
    assert gw["name"] == run["name"] + "-gw" and gw["instance"] == inst.id and gw["engine_port"] == 8000
    assert inst.container_name == run["name"] and run["instance"] == inst.id
    assert inst.gpu_uuids == ["GPU-uuid-0"] and run["gpu_uuids"] == ["GPU-uuid-0"]
    assert inst.model == ModelRef(format="hf", repo_id="Qwen/Qwen3-32B")
    assert inst.model_location is not None and inst.model_location.root_id == "hf"
    assert inst.controls.start and inst.controls.stop and inst.controls.remove and inst.controls.logs


async def test_run_spec_carries_labels_mounts_env_and_never_a_secret_in_argv_or_plain_env(env: Env) -> None:
    inst = await launch(env, params={"max_model_len": 4096, "api_key": SECRET})
    spec = env.node.run_specs[0]
    assert set(spec["labels"]) >= {"engine-console.role", "engine-console.instance", "engine-console.engine",
                                   "engine-console.model", "engine-console.port", "engine-console.name",
                                   "engine-console.format"}
    assert spec["labels"]["engine-console.instance"] == inst.id and spec["labels"]["engine-console.role"] == "engine"
    assert all(k.startswith("engine-console.") for k in spec["labels"]) and "engine-console" not in spec["labels"]
    # the HF cache root of that host, read-only, where the engine's HF_HOME expects it
    assert spec["mounts"] == [{"root": "hf", "rel": "", "dest": "/root/.cache/huggingface", "ro": True}]
    assert spec["env"]["HF_HUB_OFFLINE"] == "1" and spec["env"]["HF_HOME"] == "/root/.cache/huggingface"
    assert spec["secret_env"] == {"FAKE_API_KEY": SECRET}                      # the value rides only in secret_env
    for field in ("argv", "env", "labels", "mounts", "volumes", "files"):
        assert SECRET not in json.dumps(spec.get(field))
    assert spec["network_alias"] == spec["name"] and spec["container_port"] == 8000
    assert spec["shm_size"] == "16g" and spec["ipc_host"] is False             # one GPU: a private IPC namespace
    assert "image" not in spec["env"] and spec["image"] == "fake/engine:1.0"
    # nothing else the hub asked of the node carries the secret
    for op, args in env.node.calls:
        if op != "docker.run":
            assert SECRET not in json.dumps(args), op
    assert SECRET not in json.dumps(inst.model_dump())


async def test_hf_token_never_reaches_a_node(env: Env) -> None:
    env.cfg.hf_token = "hf_supersecrettokenvalue123"
    env.cfg.engine_hf_offline = False
    await launch(env)
    blob = json.dumps(env.node.calls)
    assert "hf_supersecret" not in blob and "HF_TOKEN" not in blob


async def test_multi_gpu_uses_host_ipc_only_when_the_policy_allows_it(env: Env) -> None:
    inst = await launch(env, params={"max_model_len": 4096, "tp": 2})
    assert env.node.run_specs[0]["ipc_host"] is True and inst.gpu_ids == [0, 1]
    other = env.add_host("nohostipc", "No host IPC", allow_ipc_host=False)
    env.seed_local_model(host_id="nohostipc")
    await env.container.lifecycle.create(request(host_id="nohostipc", params={"max_model_len": 4096, "tp": 2}))
    assert other.run_specs[0]["ipc_host"] is False


async def test_named_volumes_follow_the_adapter(env: Env) -> None:
    await launch(env)
    assert env.node.run_specs[0]["volumes"] == []     # the fake engine declares none (vllm declares its compile caches)
    from engine_console.services.lifecycle import NAMED_VOLUMES
    assert NAMED_VOLUMES["vllm"][0][0].startswith("ec-")


# ---- image choice ----------------------------------------------------------------------------------------------------
async def test_image_choice_order_request_then_host_pin_then_engine_pin_then_arch_default(env: Env) -> None:
    life = env.container.lifecycle
    env.cfg.engine_image_allowlist[:] = ["fake/engine:2", "fake/engine:2.0", "fake/engine:3"]
    env.node.policy["engine_images"] = ["fake/engine:*"]
    env.seed_local_model()
    first = await life.create(request())
    assert first.image == "fake/engine:1.0"                                    # the adapter's pinned default
    env.container.settings.update(SettingsPatch(image_pins={"fake": "fake/engine:2"}))
    second = await life.create(request(name="second"))
    assert second.image == "fake/engine:2"                                     # the engine pin
    env.container.settings.update(SettingsPatch(host_image_pins={"local": {"fake": "fake/engine:3"}}))
    third = await life.create(request(name="third"))
    assert third.image == "fake/engine:3"                                      # the host pin wins over the engine pin
    fourth = await life.create(request(name="fourth", image="fake/engine:9"))
    assert fourth.image == "fake/engine:9"                                     # the request wins over everything
    assert [s["image"] for s in env.node.run_specs] == ["fake/engine:1.0", "fake/engine:2", "fake/engine:3", "fake/engine:9"]


async def test_arm_hosts_get_the_arm_image_of_an_engine_that_has_one(env: Env) -> None:
    env.container.adapters.register(FakeMultiAdapter())
    arm = env.add_host("arm", "ARM box", arch="aarch64", engine_images=("fake/engine:*",))
    env.seed_local_model(host_id="arm")
    inst = await env.container.lifecycle.create(request(engine="fakecpp", host_id="arm"))
    assert inst.image == "fake/engine:1.0-arm64" and arm.run_specs[0]["image"] == "fake/engine:1.0-arm64"


# ---- preflight, one block per rule ---------------------------------------------------------------------------------------
async def test_preflight_ok_reports_every_check_and_the_chosen_gpus(env: Env) -> None:
    env.seed_local_model()
    report, port = await env.container.lifecycle.preflight(request())
    assert report.ok and port == 18000 and report.gpu_ids and report.fit is not None
    assert {c.code for c in report.checks} >= {"fit", "port"}
    assert not any(c.level == "block" for c in report.checks)


async def test_host_down_blocks_with_the_reason(env: Env) -> None:
    env.seed_local_model()
    env.node.state, env.node.link_error = "down", "no route to host"
    report, port = await env.container.lifecycle.preflight(request())
    assert not report.ok and port is None
    assert [c.code for c in report.checks] == ["host_down"] and "no route to host" in report.checks[0].message
    with pytest.raises(Conflict) as ei:
        await env.container.lifecycle.create(request())
    assert ei.value.code == "preflight_failed"
    assert not env.node.ops("docker.run")


async def test_monitor_mode_blocks_with_the_policy_path_and_the_advice(env: Env) -> None:
    env.seed_local_model()
    env.node.set_mode("monitor")
    report, _ = await env.container.lifecycle.preflight(request())
    c = next(c for c in report.checks if c.code == "monitor_only")
    assert c.level == "block" and "/etc/ecnode/policy.json" in c.message and "manage" in c.message
    with pytest.raises(Conflict):
        await env.container.lifecycle.create(request())
    assert not env.node.ops("docker.run")


async def test_image_not_allowed_by_the_node_policy_blocks_before_any_check_of_the_image(env: Env) -> None:
    env.seed_local_model()
    got = await preflight_codes(env, image="evil/miner:1")
    assert got["image_not_allowed"] == "block"
    assert not env.node.ops("docker.image_present")
    env.node.policy["engine_images"] = ["only/this:*"]
    assert (await preflight_codes(env))["image_not_allowed"] == "block"      # the adapter's default is not allowed either


async def test_node_policy_patterns_are_a_literal_repository_not_a_prefix(env: Env) -> None:
    env.seed_local_model()
    for image in ("fake/engine:evil/x", "fake/engine-evil:1", "other/fake/engine:1"):
        assert (await preflight_codes(env, image=image))["image_not_allowed"] == "block", image
    assert "image_not_allowed" not in await preflight_codes(env, image="fake/engine:7")


async def test_missing_image_blocks_and_says_to_pull_it_on_that_machine(env: Env) -> None:
    env.seed_local_model()
    env.node.missing_images.add("fake/engine:1.0")
    report, _ = await env.container.lifecycle.preflight(request())
    c = next(c for c in report.checks if c.code == "image_missing")
    assert c.level == "block" and "pull it on that machine" in c.message and "testbox" in c.message


async def test_model_missing_blocks_per_host(env: Env) -> None:
    env.seed_local_model()
    assert "model_missing" not in await preflight_codes(env)
    other = env.add_host("spark", "Spark")
    assert other is not None
    got = await preflight_codes(env, host_id="spark")                           # the model is only on `local`
    assert got["model_missing"] == "block"


async def test_format_unsupported_blocks_when_the_engine_cannot_load_the_format(env: Env) -> None:
    env.library.add("local", ModelRef(format="gguf", root_id="gguf", rel="a/model-Q4.gguf"), root_id="gguf", rel="a/model-Q4.gguf")
    got = await preflight_codes(env, repo_id=None, model=ModelRef(format="gguf", root_id="gguf", rel="a/model-Q4.gguf"))
    assert got["format_unsupported"] == "block"


async def test_fit_blocks_against_that_hosts_free_vram(env: Env) -> None:
    env.seed_local_model()
    env.node.set_gpu_free(10.0)
    report, _ = await env.container.lifecycle.preflight(request())
    assert not report.ok and report.fit is not None and report.fit.verdict == "wont_fit"
    spark = env.add_host("spark", "Spark")                                       # another machine, nothing resident
    env.seed_local_model(host_id="spark")
    assert spark is not None
    ok, _ = await env.container.lifecycle.preflight(request(host_id="spark"))
    assert ok.ok and ok.fit is not None and ok.fit.verdict in ("fits", "tight")


async def test_unified_memory_hosts_budget_vram_from_ram(env: Env) -> None:
    gb10 = env.add_host("gb10", "GB10", gpus=1, gpu_gib=119.0, unified=True, ram_gib=119.0)
    gb10.mem.update(total=119.0, available=30.0)
    env.seed_local_model(host_id="gb10")
    # the sample reports no per-GPU memory use (the driver cannot say): the pool is the host's RAM
    for g in gb10.hello["gpus"]:
        assert g["unified_memory"] is True
    report, _ = await env.container.lifecycle.preflight(request(host_id="gb10"))
    assert report.fit is not None and report.fit.per_gpu[0].free_gib < 30.0
    assert any("unified memory" in n for n in report.fit.notes)
    assert report.fit.verdict == "wont_fit"                                      # 65 GB of weights do not fit in 30 GiB free RAM


async def test_gpu_clash_warns_but_unknown_gpu_blocks(env: Env) -> None:
    first = await launch(env, gpu_ids=[0])
    assert first.gpu_ids == [0]
    env.node.containers[first.container_name]["state"] = "running"
    report, _ = await env.container.lifecycle.preflight(request(gpu_ids=[0], name="b"))
    clash = next(c for c in report.checks if c.code == "gpu_shared")
    assert clash.level == "warn" and first.name in clash.message
    assert (await preflight_codes(env, gpu_ids=[7]))["gpu_unknown"] == "block"


async def test_a_host_without_gpus_blocks(env: Env) -> None:
    env.add_host("cpu-box", "CPU box", gpus=0)
    env.seed_local_model(host_id="cpu-box")
    assert (await preflight_codes(env, host_id="cpu-box"))["no_gpu"] == "block"


async def test_a_policy_without_a_gateway_image_blocks(env: Env) -> None:
    env.seed_local_model()
    env.node.policy["gateway_image"] = None
    assert (await preflight_codes(env))["gateway_missing"] == "block"


async def test_shm_above_the_policy_limit_blocks_instead_of_being_clamped(env: Env) -> None:
    env.seed_local_model()
    env.node.policy["max_shm_gib"] = 8.0
    got = await preflight_codes(env)
    assert got["invalid_launch"] == "block"


async def test_secret_on_argv_is_refused_before_anything_runs(env: Env) -> None:
    env.seed_local_model()
    with pytest.raises(Conflict) as ei:
        await env.container.lifecycle.create(request(params={"leaky_key": SECRET}))
    assert codes(ei.value)["secret_in_argv"] == "block" and SECRET not in json.dumps(ei.value.extra)
    assert not env.node.ops("docker.run")


# ---- ports -------------------------------------------------------------------------------------------------------------------
async def test_ports_are_allocated_per_host_from_that_hosts_range(env: Env) -> None:
    a = await launch(env)
    other = env.add_host("spark", "Spark", port_range=(19000, 19010))
    env.seed_local_model(host_id="spark")
    b = await env.container.lifecycle.create(request(host_id="spark"))
    c = await launch(env, name="c")
    assert (a.port, c.port) == (18000, 18001) and b.port == 19000
    assert other.ops("gateway.start")[0]["host_port"] == 19000
    assert b.host_id == "spark" and b.host_label == "Spark" and b.gpu_uuids == ["GPU-uuid-0"]


async def test_ports_published_by_other_containers_are_skipped(env: Env) -> None:
    env.node.add_container("someone", image="redis", host_ports=ports(18000, 6379))
    env.node.add_endpoint(18001, container="other")
    inst = await launch(env)
    assert inst.port == 18002


async def test_gateway_port_in_use_retries_the_next_free_port(env: Env) -> None:
    env.node.port_taken.update({18000, 18001})                 # something listens there that the inventory does not show
    inst = await launch(env)
    assert [a["host_port"] for a in env.node.ops("gateway.start")] == [18000, 18001, 18002]
    assert inst.port == 18002 and env.container.lifecycle.get(inst.id).port == 18002
    assert env.node.containers[inst.container_name + "-gw"]["ports"][0]["host_port"] == 18002


async def test_gateway_gives_up_after_a_bounded_number_of_ports_and_leaves_no_engine_behind(env: Env) -> None:
    env.node.port_taken.update(range(18000, 18100))
    env.seed_local_model()
    with pytest.raises(ProblemError) as ei:
        await env.container.lifecycle.create(request())
    assert ei.value.code == "port_in_use" and ei.value.status == 409          # the node's own words, mapped to 409
    assert len(env.node.ops("gateway.start")) == 8                              # bounded: not every port of the range
    assert not any(c["state"] == "running" and c.get("managed") for c in env.node.containers.values())
    assert env.container.lifecycle.list_page().items[0].state == "failed"


async def test_a_gateway_that_fails_otherwise_tears_the_engine_down(env: Env) -> None:
    env.node.fail["gateway.start"] = NodeError("local", "docker_error", "cannot connect the network")
    env.seed_local_model()
    with pytest.raises(ProblemError) as ei:
        await env.container.lifecycle.create(request())
    assert ei.value.status == 502 and ei.value.code == "docker_error"
    assert not any(c.get("managed") for c in env.node.containers.values())
    assert env.container.lifecycle.list_page().items[0].state == "failed"


async def test_docker_run_failure_marks_the_instance_failed_with_the_nodes_words(env: Env) -> None:
    env.node.fail["docker.run"] = NodeError("local", "docker_error", "no such gpu")
    env.seed_local_model()
    with pytest.raises(ProblemError) as ei:
        await env.container.lifecycle.create(request())
    assert ei.value.status == 502 and "testbox" in ei.value.detail and "no such gpu" in ei.value.detail
    inst = env.container.lifecycle.list_page().items[0]
    assert inst.state == "failed" and "no such gpu" in (inst.error or "")


async def test_existing_non_internal_network_is_refused_and_nothing_is_started(env: Env) -> None:
    env.node.network_internal = False
    env.seed_local_model()
    with pytest.raises(ProblemError) as ei:
        await env.container.lifecycle.create(request())
    assert ei.value.status == 409 and not env.node.ops("docker.run")


# ---- stop / start / remove -------------------------------------------------------------------------------------------------
async def test_stop_stops_the_gateway_first_then_the_engine_with_role_and_instance(env: Env) -> None:
    inst = await launch(env)
    n0 = len(env.node.calls)
    stopped = await env.container.lifecycle.stop(inst.id)
    assert stopped.state == "stopped"
    stops = [a for o, a in env.node.calls[n0:] if o == "docker.stop"]
    assert [s["name"] for s in stops] == [inst.container_name + "-gw", inst.container_name]
    assert stops[0]["role"] == "gateway" and stops[0]["instance"] == inst.id and stops[1]["instance"] == inst.id
    assert env.container.lifecycle.get(inst.id).phase is None


async def test_remove_removes_both_containers_the_secrets_and_the_row(env: Env) -> None:
    inst = await launch(env, params={"max_model_len": 4096, "api_key": SECRET})
    sf = env.cfg.data_dir / "secrets" / f"{inst.id}.json"
    assert sf.exists()
    await env.container.lifecycle.remove(inst.id)
    assert inst.container_name not in env.node.containers and inst.container_name + "-gw" not in env.node.containers
    assert not sf.exists()
    with pytest.raises(ProblemError):
        env.container.lifecycle.get(inst.id)


async def test_stop_and_remove_tolerate_containers_that_are_already_gone(env: Env) -> None:
    inst = await launch(env)
    del env.node.containers[inst.container_name]
    del env.node.containers[inst.container_name + "-gw"]
    assert (await env.container.lifecycle.stop(inst.id)).state == "stopped"
    await env.container.lifecycle.remove(inst.id)


async def test_stop_on_an_unreachable_host_changes_nothing_and_says_503(env: Env) -> None:
    inst = await launch(env)
    env.node.state = "down"
    with pytest.raises(ProblemError) as ei:
        await env.container.lifecycle.stop(inst.id)
    assert ei.value.status == 503 and ei.value.code == "node_unavailable"
    row = env.container.store.one("SELECT state FROM instances WHERE id=?", (inst.id,))
    assert row is not None and row["state"] == "starting"                     # not "failed", not "stopping"
    assert env.container.lifecycle.get(inst.id).state == "unreachable"        # what the API shows meanwhile
    env.node.state = "up"
    assert env.container.lifecycle.get(inst.id).state == "starting"


async def test_start_recreates_the_containers_and_keeps_the_gpus(env: Env) -> None:
    inst = await launch(env, gpu_ids=[1])
    await env.container.lifecycle.stop(inst.id)
    again = await env.container.lifecycle.start(inst.id)
    assert again.state == "starting" and again.gpu_ids == [1] and again.container_name == inst.container_name
    assert len(env.node.run_specs) == 2 and env.node.run_specs[1]["gpu_uuids"] == ["GPU-uuid-1"]
    with pytest.raises(Conflict):
        await env.container.lifecycle.start(inst.id)                           # already starting


async def test_launch_will_not_remove_an_unlabelled_namesake(env: Env) -> None:
    inst = await launch(env)
    env.container.store.execute("UPDATE instances SET state='stopped' WHERE id=?", (inst.id,))
    victim = env.node.containers[inst.container_name]
    victim["labels"] = {}
    victim["managed"] = False
    with pytest.raises(ProblemError) as ei:
        await env.container.lifecycle.start(inst.id)
    assert ei.value.status == 403 and ei.value.code == "not_owned"
    assert inst.container_name in env.node.containers and victim["state"] == "running"
    assert env.container.lifecycle.get(inst.id).state == "failed"


# ---- the supervisor -----------------------------------------------------------------------------------------------------------
async def test_supervisor_walks_starting_loading_ready_from_logs_and_the_health_probe(env: Env) -> None:
    life = env.container.lifecycle
    inst = await launch(env)
    env.node.logs[inst.container_name] = "boot\nloading weights 55%\n"
    await life.tick()
    got = life.get(inst.id)
    assert got.state == "loading" and got.progress_pct == 55.0 and got.phase == "loading_weights"
    env.engine_up()
    await life.tick()
    assert life.get(inst.id).state == "ready"
    probes = [r for r in env.node.http_requests if r.url.path == "/health"]
    assert probes and probes[-1].url.port == inst.port and probes[-1].method == "GET"


async def test_supervisor_reads_the_inventory_and_only_asks_the_node_when_it_has_not_caught_up(env: Env) -> None:
    life = env.container.lifecycle
    inst = await launch(env)
    env.node.logs[inst.container_name] = "x\n"
    await life.tick()
    assert not env.node.ops("docker.inspect")                                  # the container is in the inventory
    del env.node.containers[inst.container_name]
    await life.tick()                                                          # just launched: ask the node, it says gone
    assert env.node.ops("docker.inspect") and life.get(inst.id).error == "container disappeared"


async def test_supervisor_trusts_a_missing_container_once_the_inventory_is_older_than_a_launch(env: Env) -> None:
    life = env.container.lifecycle
    inst = await launch(env)
    env.container.store.execute("UPDATE instances SET started_at=1 WHERE id=?", (inst.id,))
    del env.node.containers[inst.container_name]
    await life.tick()
    assert not env.node.ops("docker.inspect") and life.get(inst.id).error == "container disappeared"


async def test_crash_detection_captures_the_exit_code_and_scrubbed_logs(env: Env) -> None:
    life = env.container.lifecycle
    inst = await launch(env, params={"max_model_len": 4096, "api_key": SECRET})
    env.node.logs[inst.container_name] = f"CUDA error: out of memory\nkey {SECRET} hf_abcdefghijkl1234\n"
    env.node.containers[inst.container_name].update(state="exited", exit_code=137)
    await life.tick()
    got = life.get(inst.id)
    assert got.state == "failed" and "code 137" in (got.error or "") and "out of memory" in (got.last_logs or "")
    assert SECRET not in (got.last_logs or "") and "hf_abcdef" not in (got.last_logs or "")
    assert any(a["name"].endswith("-gw") for a in env.node.ops("docker.stop"))   # the gateway is stopped with it


async def test_startup_timeout_and_gateway_checks(env: Env) -> None:
    life = env.container.lifecycle
    inst = await launch(env)
    env.container.store.execute("UPDATE instances SET started_at=1 WHERE id=?", (inst.id,))
    await life.tick()
    assert life.get(inst.id).error == "startup timed out"
    second = await launch(env, name="two")
    env.node.containers[second.container_name + "-gw"]["labels"]["engine-console.instance"] = "inst_someone_else"
    env.node.containers[second.container_name + "-gw"]["instance"] = "inst_someone_else"
    await life.tick()
    got = life.get(second.id)
    assert got.state == "failed" and "gateway" in (got.error or "")
    assert not any(a["name"] == second.container_name + "-gw" and a["instance"] == "inst_someone_else"
                   for a in env.node.ops("docker.stop"))                                  # a foreign gateway is never touched


async def test_idle_ttl_stops_a_ready_instance_unless_it_is_pinned(env: Env) -> None:
    life = env.container.lifecycle
    env.engine_up()
    a = await launch(env)
    b = await launch(env, name="b", gpu_ids=[1])
    for i in (a, b):
        await life.tick()
        assert life.get(i.id).state == "ready"
    life.patch(a.id, InstancePatch(ttl_idle_s=1))
    env.container.store.execute("UPDATE instances SET last_request_at=? WHERE id=?", (time.time() - 100, a.id))
    await life.tick()
    assert life.get(a.id).state == "stopped" and life.get(b.id).state == "ready"


async def test_a_host_that_is_down_is_not_supervised_and_nothing_is_marked_failed(env: Env) -> None:
    life = env.container.lifecycle
    inst = await launch(env)
    env.node.state = "down"
    del env.node.containers[inst.container_name]
    env.container.store.execute("UPDATE instances SET started_at=1 WHERE id=?", (inst.id,))
    await life.tick()
    env.node.state = "up"
    row = env.container.store.one("SELECT state FROM instances WHERE id=?", (inst.id,))
    assert row is not None and row["state"] == "starting"


# ---- adoption ---------------------------------------------------------------------------------------------------------------------
GOOD = {"engine-console": "1", "engine-console.instance": "inst_ok", "engine-console.engine": "fake",
        "engine-console.model": "a/b", "engine-console.port": "18010", "engine-console.name": "ok",
        "engine-console.role": "engine"}


@pytest.mark.parametrize("labels", [
    {"engine-console.model": "a/b;rm -rf"}, {"engine-console.model": "a--b/c"}, {"engine-console.instance": "inst x"},
    {"engine-console.instance": "../x"}, {"engine-console.name": "a\nb"}, {"engine-console.name": "x" * 200},
    {"engine-console.instance": "i" * 100}, {"engine-console.port": "22"}, {"engine-console.port": "99999"},
    {"engine-console.port": "abc"}, {"engine-console.engine": "unknown"}, {"engine-console.format": "gguf"}])
async def test_adopt_skips_invalid_labels_without_raising(env: Env, labels: dict[str, str],
                                                          caplog: pytest.LogCaptureFixture) -> None:
    env.node.add_container("ec-bad", managed=True, labels={**GOOD, **labels}, image="fake/engine:1.0")
    assert await env.container.lifecycle.adopt() == 0
    env.node.containers["ec-bad"]["labels"] = dict(GOOD)
    assert await env.container.lifecycle.adopt("local") == 1


async def test_adopt_reattaches_labelled_containers_and_is_idempotent(env: Env) -> None:
    life = env.container.lifecycle
    env.node.add_container("ec-old-abc123", managed=True, image="fake/engine:1.0", labels={
        "engine-console.instance": "inst_old", "engine-console.engine": "fake", "engine-console.model": "Qwen/Qwen3-32B",
        "engine-console.port": "18007", "engine-console.name": "old", "engine-console.gpu_ids": "1",
        "engine-console.role": "engine"})
    env.node.add_container("ec-stranger", managed=True, labels={"engine-console.role": "engine"})   # no instance label
    assert await life.adopt() == 1
    inst = life.get("inst_old")
    assert (inst.state, inst.port, inst.gpu_ids, inst.name, inst.host_id) == ("loading", 18007, [1], "old", "local")
    assert inst.model == ModelRef(format="hf", repo_id="Qwen/Qwen3-32B")
    assert await life.adopt() == 0
    env.node.containers["ec-old-abc123"]["state"] = "exited"
    await life.adopt()
    assert life.get("inst_old").state == "stopped"


async def test_adopt_takes_the_real_port_from_the_gateway_and_ignores_gateways_as_instances(env: Env) -> None:
    life = env.container.lifecycle
    lab = {**GOOD, "engine-console.instance": "inst_x", "engine-console.port": "18020"}
    env.node.add_container("ec-x-gw", managed=True, labels={**lab, "engine-console.role": "gateway"},
                           host_ports=ports(18033, 8080, "127.0.0.1"))
    assert await life.adopt() == 0                                              # a gateway alone is not an instance
    env.node.add_container("ec-x", managed=True, labels={**lab, "engine-console.role": "engine"}, image="fake/engine:1.0")
    assert await life.adopt() == 1
    assert life.get("inst_x").port == 18033                                       # a retried gateway moved off the label's port


async def test_adopt_runs_per_host_when_its_link_comes_up(env: Env) -> None:
    life = env.container.lifecycle
    spark = env.add_host("spark", "Spark")
    spark.add_container("ec-s", managed=True, image="fake/engine:1.0", labels={**GOOD, "engine-console.instance": "inst_s"})
    spark.state = "down"
    await life.tick()
    assert not env.container.store.one("SELECT 1 FROM instances WHERE id='inst_s'")
    spark.state = "up"
    await life.tick()
    row = env.container.store.one("SELECT host_id FROM instances WHERE id='inst_s'")
    assert row is not None and row["host_id"] == "spark"


async def test_adopt_validates_a_gguf_models_root_and_path_labels(env: Env) -> None:
    life = env.container.lifecycle
    env.container.adapters.register(FakeMultiAdapter())
    lab = {**GOOD, "engine-console.engine": "fakecpp", "engine-console.format": "gguf", "engine-console.model": "m-Q4",
           "engine-console.root": "gguf", "engine-console.rel": "a/m-Q4.gguf"}
    env.node.add_container("ec-g", managed=True, labels=lab, image="fake/engine:1.0")
    assert await life.adopt() == 1
    assert life.get("inst_ok").model == ModelRef(format="gguf", root_id="gguf", rel="a/m-Q4.gguf")


# ---- several machines ---------------------------------------------------------------------------------------------------------------
async def test_instances_of_several_hosts_are_listed_together_and_filtered_by_host(env: Env) -> None:
    life = env.container.lifecycle
    spark = env.add_host("spark", "Spark")
    a = await launch(env)
    b = await launch(env, host_id="spark")
    spark.add_container("vllm-ext", engine="vllm", host_ports=ports(18500))
    spark.add_endpoint(18500, container="vllm-ext", engine="vllm", models=("org/big",))
    env.node.add_container("sgl-ext", engine="sglang", image="lmsysorg/sglang:v0.5", host_ports=ports(18501, 30000))
    env.node.add_endpoint(18501, container="sgl-ext", engine="sglang", models=("m",))
    items = {i.id: i for i in life.list_page().items}
    assert set(items) == {a.id, b.id, "ext-spark-vllm-ext", "ext-local-sgl-ext"}
    assert {i.host_id for i in items.values()} == {"local", "spark"}
    assert [i.id for i in life.list_page(host_id="spark").items if i.management == "managed"] == [b.id]
    assert {i.id for i in life.list_page(host_id="spark").items} == {b.id, "ext-spark-vllm-ext"}
    assert items["ext-spark-vllm-ext"].host_label == "Spark" and items[a.id].host_label == "testbox"


async def test_a_host_that_disappears_from_the_fleet_does_not_break_listing(env: Env) -> None:
    spark = env.add_host("spark", "Spark")
    env.seed_local_model(host_id="spark")
    inst = await env.container.lifecycle.create(request(host_id="spark"))
    assert spark is not None
    del env.fleet._nodes["spark"]   # noqa: SLF001
    got = env.container.lifecycle.get(inst.id)
    assert got.state == "unreachable" and "not configured" in (got.state_reason or "")
    await env.container.lifecycle.tick()   # supervising a host that is gone must not raise


# ---- VRAM, memory and counts ----------------------------------------------------------------------------------------------------------
async def test_vram_and_host_memory_come_from_the_hosts_sample_for_the_instances_container(env: Env) -> None:
    inst = await launch(env, gpu_ids=[1])
    env.node.gpu_procs = {1: [{"pid": 11, "container": inst.container_name, "used_bytes": 40 * 1024**3, "name": "python"},
                              {"pid": 12, "container": "someone-else", "used_bytes": 5 * 1024**3, "name": "python"}]}
    env.node.cg[inst.container_name] = cgroup()
    got = env.container.lifecycle.get(inst.id)
    assert [(v.gpu_index, v.gpu_uuid, round(v.used_gib or 0)) for v in got.vram] == [(1, "GPU-uuid-1", 40)]
    assert got.host_memory is not None and got.host_memory.source == "cgroup"
    assert got.host_memory.total_gib == pytest.approx(68.6, abs=1e-3) and got.host_memory.cache_gib == pytest.approx(0.5, abs=1e-3)
    env.node.gpu_procs = {1: [{"pid": 11, "container": inst.container_name, "name": "python"}]}   # the driver cannot say
    assert env.container.lifecycle.get(inst.id).vram[0].used_gib is None


async def test_counts_resident_and_instances_using_a_path(env: Env) -> None:
    life = env.container.lifecycle
    inst = await launch(env)
    c = life.counts("local")
    assert c.engines_total == 1 and c.engines_running == 1 and c.models == 1 and c.models_bytes == 65_530_000_000
    env.node.gpu_procs = {0: [{"pid": 1, "container": inst.container_name, "used_bytes": 50 * 1024**3}]}
    res = life.resident("local")
    assert [(r.name, r.gpu_ids, round(r.gib_per_gpu)) for r in res] == [(inst.name, [0], 50)]
    assert life.instances_using("local", "/data/hf/hub/models--Qwen--Qwen3-32B") == [inst.id]    # the HF root is mounted
    assert life.instances_using("local", "/data/hf") == [inst.id]
    assert life.instances_using("local", "/elsewhere/model") == []
    assert life.instance_for_container("local", inst.container_name) == inst.id
    assert life.instance_for_container("local", "nobody") is None


async def test_fit_fits_if_stop_names_the_resident_engine_on_that_host_only(env: Env) -> None:
    life = env.container.lifecycle
    inst = await launch(env, gpu_ids=[0])
    env.node.gpu_procs = {0: [{"pid": 1, "container": inst.container_name, "used_bytes": 80 * 1024**3}]}
    env.node.set_gpu_free([16.0, 96.0])
    report, _ = await life.preflight(request(gpu_ids=[0], name="second"))
    assert report.fit is not None and report.fit.verdict == "wont_fit" and report.fit.fits_if_stop == [inst.name]


# ---- the API ----------------------------------------------------------------------------------------------------------------------------
def test_api_create_list_get_filtered_by_host(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    env.add_host("spark", "Spark")
    env.seed_local_model(host_id="spark")
    a = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 4096}})
    b = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "host_id": "spark",
                                            "params": {"max_model_len": 4096}})
    assert a.status_code == 201 and b.status_code == 201, (a.text, b.text)
    assert a.json()["host_id"] == "local" and b.json()["host_id"] == "spark" and b.json()["host_label"] == "Spark"
    assert {i["id"] for i in client.get(f"{P}/instances").json()["items"]} == {a.json()["id"], b.json()["id"]}
    only = client.get(f"{P}/instances", params={"host": "spark"}).json()["items"]
    assert [i["id"] for i in only] == [b.json()["id"]]
    assert client.get(f"{P}/instances", params={"host": "Bad Host!"}).status_code == 422
    assert client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "host_id": "nope"}).status_code == 404


def test_api_preflight_and_unknown_host_problem_codes(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    r = client.post(f"{P}/instances/preflight", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "host_id": "nope"})
    assert r.status_code == 404 and r.json()["code"] == "unknown_host"
    env.node.state = "down"
    pf = client.post(f"{P}/instances/preflight", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B"}).json()
    assert pf["ok"] is False and pf["checks"][0]["code"] == "host_down"


def test_api_command_snippets_hold_no_secret_and_use_the_nodes_rendering(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    inst = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B",
                                               "params": {"max_model_len": 4096, "api_key": SECRET}}).json()
    cmd = client.get(f"{P}/instances/{inst['id']}/command")
    assert cmd.status_code == 200, cmd.text
    body = cmd.json()
    assert body["docker_run"].startswith("docker run -d --name ec-") and "--cap-drop ALL" in body["docker_run"]
    assert "-e FAKE_API_KEY" in body["docker_run"] and SECRET not in cmd.text and "s3cr3t" not in cmd.text
    assert "${FAKE_API_KEY}" in body["compose_yaml"] and "GPU-uuid-0" in body["compose_yaml"]
    assert "/data/hf:/root/.cache/huggingface:ro" in body["compose_yaml"] and "--model" in body["engine_cli"]
    render = env.node.ops("docker.render")[0]["spec"]
    assert SECRET not in json.dumps(render) and set(render["secret_env"]) == {"FAKE_API_KEY"}


def test_api_logs_follow_stream_and_scrub(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    inst = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B",
                                               "params": {"max_model_len": 4096, "api_key": SECRET}}).json()
    env.node.logs[inst["container_name"]] = f"args: api_key={SECRET} token hf_abcdefghijkl1234\nsecond line\n"
    logs = client.get(f"{P}/instances/{inst['id']}/logs").text
    assert SECRET not in logs and "hf_abcdef" not in logs and "[redacted]" in logs and "second line" in logs
    once = client.get(f"{P}/instances/{inst['id']}/logs/stream", params={"once": "true"})
    assert "event: log" in once.text and SECRET not in once.text
    live = client.get(f"{P}/instances/{inst['id']}/logs/stream")
    assert live.text.count("event: log") == 2 and SECRET not in live.text


def test_node_errors_become_problems_with_the_hosts_label_and_the_nodes_code(client: TestClient, env: Env) -> None:
    env.seed_local_model()
    env.node.fail["docker.run"] = NodeError("local", "docker_error", "boom")
    r = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B"})
    assert r.status_code == 502 and r.json()["code"] == "docker_error" and r.json()["detail"] == "testbox: boom"
    assert r.headers["content-type"].startswith("application/problem+json")


def test_unserved_node_error_is_still_a_problem_through_the_global_handler(client: TestClient, env: Env) -> None:
    env.node.state = "down"
    env.node.link_error = "connection refused"
    r = client.get(f"{P}/hardware")
    assert r.status_code == 503 and r.json()["code"] == "node_unavailable" and "connection refused" in r.json()["detail"]
    assert r.headers["content-type"].startswith("application/problem+json")


def test_spec_validation_helpers_are_not_reachable_with_hostile_names(env: Env) -> None:
    from engine_console.services.runspec import validate_run_spec
    spec = {"name": "ec-x", "instance": "inst_1", "image": "i:1", "argv": [], "container_port": 8000, "labels": {},
            "env": {}, "secret_env": {}, "mounts": [], "volumes": [], "shm_size": "16g"}
    assert validate_run_spec(spec) == []
    for bad in ({"name": "../x"}, {"name": "ec-A"}, {"instance": "a b"}, {"container_port": 0}, {"shm_size": "16 g"},
                {"env": {"1BAD": "x"}}, {"labels": {"x": "y"}}, {"mounts": [{"root": "hf", "rel": "../x", "dest": "/m"}]},
                {"mounts": [{"root": "hf", "rel": "a", "dest": "rel/path"}]}, {"volumes": [{"name": "evil", "dest": "/x"}]},
                {"argv": ["a\x00b"]}, {"files": {"relative": "x"}}):
        assert validate_run_spec({**spec, **bad}), bad
    assert validate_run_spec({**spec, "shm_size": "100g"}, max_shm_gib=64.0)


def test_fake_node_itself_rejects_dropped_arguments(env: Env) -> None:
    """The fake enforces what the real node's whitelist cleaning would do: a typo is a failure, not a silent drop."""
    import asyncio
    node: FakeNode = env.node
    with pytest.raises(NodeError) as ei:
        asyncio.run(node.call("docker.stop", {"name": "x", "typo_field": 1}))
    assert ei.value.code == "bad_args"
    node.set_mode("monitor")
    with pytest.raises(NodeError) as ei2:
        asyncio.run(node.call("docker.rm", {"name": "x"}))
    assert ei2.value.code == "monitor_only"


async def test_adoption_only_takes_ids_that_look_like_ours_and_survives_a_clash(env: Env) -> None:
    life = env.container.lifecycle
    base = {"engine-console.engine": "fake", "engine-console.model": "a/b", "engine-console.port": "18010",
            "engine-console.role": "engine"}
    # ids that would shadow derived instances in the API, or are simply not ours
    for i, bad in enumerate(("ext-evil", "scr-evil", "evil", "not_inst_1")):
        env.node.add_container(f"ec-bad{i}", managed=True, image="fake/engine:1.0",
                               labels={**base, "engine-console.instance": bad, "engine-console.name": f"n{i}"})
    assert await life.adopt() == 0
    # a container that claims the id of an instance that already exists (even on another host) does not abort the rest
    existing = await launch(env)
    env.node.add_container("ec-clash", managed=True, image="fake/engine:1.0",
                           labels={**base, "engine-console.instance": existing.id, "engine-console.name": "clash"})
    env.node.add_container("ec-fine", managed=True, image="fake/engine:1.0",
                           labels={**base, "engine-console.instance": "inst_fine", "engine-console.name": "fine"})
    assert await life.adopt() == 1
    assert life.get("inst_fine").host_id == "local" and life.get(existing.id).container_name == existing.container_name


async def test_a_script_job_that_never_ends_is_failed_at_the_upper_bound(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from engine_console.services import scripts as sc

    from .fakefleet import BIG_MODEL_SCRIPT, ScriptRun
    env.node.policy["scripts"] = [json.loads(json.dumps(BIG_MODEL_SCRIPT))]
    monkeypatch.setattr(sc, "JOB_MAX_S", 0.05)
    env.node.script_runs[("big-model", "up")] = ScriptRun(lines=[("out", "started")], gate=asyncio.Event())   # never released
    job = await env.container.scripts.run("local", "big-model", "up")
    await env.container.scripts.wait(job.id)
    done = env.container.scripts.job(job.id)
    assert done.state == "failed" and "[timeout]" in done.output_tail
    assert env.node.streams[-1].cancelled and env.container.scripts.busy("local") == {}


async def test_two_requests_to_start_the_same_instance_launch_it_once(env: Env) -> None:
    import asyncio
    inst = await launch(env)
    await env.container.lifecycle.stop(inst.id)
    n = len(env.node.run_specs)
    results = await asyncio.gather(env.container.lifecycle.start(inst.id), env.container.lifecycle.start(inst.id),
                                   return_exceptions=True)
    assert sum(1 for r in results if not isinstance(r, Exception)) == 1 and sum(1 for r in results if isinstance(r, Conflict)) == 1
    assert len(env.node.run_specs) == n + 1                                    # one launch, not two


async def test_a_stop_whose_request_went_away_does_not_leave_the_instance_stopping(env: Env) -> None:
    life = env.container.lifecycle
    inst = await launch(env)
    await life.sync_links()                                                    # the first look at a host adopts what runs there
    env.container.store.execute("UPDATE instances SET state='stopping' WHERE id=?", (inst.id,))
    await life.tick()
    assert life.get(inst.id).state == "stopping"                               # the container still runs: a stop may be under way
    env.node.containers[inst.container_name].update(state="exited", exit_code=0)
    env.node.containers[inst.container_name + "-gw"].update(state="exited", exit_code=0)
    await life.tick()
    assert life.get(inst.id).state == "stopped"


async def test_restart_waits_for_the_hosts_next_samples_before_the_fit_check(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    inst = await launch(env)
    env.cfg.restart_settle_s = 0.05
    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def spy(delay: float, *a: Any) -> None:
        slept.append(delay)
        await real_sleep(0)

    monkeypatch.setattr("engine_console.services.lifecycle.asyncio.sleep", spy)
    again = await env.container.lifecycle.restart(inst.id)
    assert again.state == "starting" and 0.05 in slept
