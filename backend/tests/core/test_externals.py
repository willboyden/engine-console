"""Engines the console did not create: external and script instances, read from each host's pushed inventory.

Classifying containers (image/command signatures) and redacting their argv happen on the node now; what the hub owns,
and what is tested here, is turning that inventory into instances: ids, states, who may control them, VRAM and RAM
attribution, model location, and that every mutating route stays refused when the node policy says monitor-only."""
from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from engine_console.domain.models import Instance

from .conftest import Env
from .fakefleet import BIG_MODEL_SCRIPT as SCRIPT
from .fakefleet import FakeNode, ports
from .test_api_smoke import problem_code, sse_engine

P = "/api/v1"
SECRET_ARG = "sk-live-argvsecret-987654321"
LEAKS = ("hf_supersecret", "argvsecret", "s3cr3t-key", "Bearer abc")
def seed_world(node: FakeNode) -> None:
    node.add_container("vllm-a", image="vllm/vllm-openai:v0.23.0", engine="vllm", host_ports=ports(18500),
                       params={"model": "org/big", "api_key": "[redacted]", "port": "8000"},
                       mounts=[{"source": "/data/hf", "dest": "/root/.cache/huggingface", "ro": True, "root": "hf", "rel": ""}],
                       gpus=["GPU-uuid-0"])
    node.add_endpoint(18500, container="vllm-a", engine="vllm", models=("org/big",), health=200, models_status=200, metrics=200)
    node.add_container("sgl-b", image="lmsysorg/sglang:v0.5.14", engine="sglang", host_ports=ports(18501, 30000))
    node.add_endpoint(18501, container="sgl-b", engine="sglang", health=401, models_status=401, auth_required=True)
    node.add_container("ollama-c", image="ollama/ollama:0.5", engine="ollama", host_ports=ports(18502, 11434))
    node.add_endpoint(18502, container="ollama-c", engine="ollama", models=("llama3",), health=None, models_status=200)
    node.add_container("router", image="ghcr.io/berriai/litellm:main", host_ports=ports(4000, 4000))        # not an engine
    node.add_container("vllm-noport", image="vllm/vllm-openai:v0.23.0", engine="vllm")
    node.add_container("postgres", image="postgres:16", host_ports=ports(5432, 5432))
    node.add_container("mine-gw", managed=True, labels={"engine-console.role": "gateway"}, host_ports=ports(18503, 8080, "127.0.0.1"))
    node.add_endpoint(18503, container="mine-gw", engine="vllm", models=("owned",))
    node.add_endpoint(9100, engine="vllm", models=("bare-model",), health=None, models_status=200, metrics=200)   # no container
    node.add_endpoint(9101, health=200, models_status=None)                                                      # health only: not an engine


def by_name(items: list[Instance]) -> dict[str, Instance]:
    return {i.name: i for i in items}


def externals(env: Env, host: str | None = None) -> list[Instance]:
    return [i for i in env.container.lifecycle.list_page(500, None, host).items if i.management != "managed"]


# ---- external instances ---------------------------------------------------------------------------------------------------------
def test_external_engines_are_listed_with_ids_states_and_fields_and_everything_else_is_ignored(env: Env) -> None:
    seed_world(env.node)
    got = by_name(externals(env))
    assert set(got) == {"vllm-a", "sgl-b", "ollama-c", "vllm-noport", "127.0.0.1:9100"}, set(got)
    a = got["vllm-a"]
    assert (a.id, a.managed, a.source, a.engine, a.state, a.management) == ("ext-local-vllm-a", False, "external", "vllm", "ready", "external")
    assert a.endpoint == "http://127.0.0.1:18500" and a.port == 18500 and a.served_models == ["org/big"]
    assert a.container_name == "vllm-a" and a.image == "vllm/vllm-openai:v0.23.0" and a.repo_id == "org/big"
    assert a.params["model"] == "org/big" and a.params["api_key"] == "[redacted]" and a.params["port"] == "8000"
    assert a.gpu_ids == [0] and a.gpu_uuids == ["GPU-uuid-0"] and a.host_id == "local" and a.host_label == "testbox"
    assert isinstance(a.history_since, float) and a.history_since == a.created_at
    assert got["sgl-b"].state == "auth_required" and "API key" in (got["sgl-b"].state_reason or "")
    assert got["ollama-c"].engine == "ollama" and got["ollama-c"].served_models == ["llama3"] and got["ollama-c"].state == "ready"
    assert got["vllm-noport"].state == "unreachable" and "no published host port" in (got["vllm-noport"].state_reason or "")
    assert got["vllm-noport"].port is None and got["vllm-noport"].endpoint is None
    bare = got["127.0.0.1:9100"]
    assert bare.engine == "vllm" and bare.container_name is None and bare.id == "ext-local-127-0-0-1-9100"
    blob = json.dumps([i.model_dump() for i in got.values()])
    for leak in (*LEAKS, "HF_TOKEN", "VLLM_API_KEY", "sk-live", "hf_super"):
        assert leak not in blob, leak
    assert "router" not in blob and "postgres" not in blob and "owned" not in blob   # routers, databases, our own gateways


def test_a_token_shaped_value_in_a_nodes_params_is_masked_even_if_the_node_forgot(env: Env) -> None:
    env.node.add_container("leaky", engine="vllm", host_ports=ports(18600), params={"hf_note": "token hf_abcdefghijkl1234 here"})
    env.node.add_endpoint(18600, container="leaky", engine="vllm", models=("m",))
    inst = by_name(externals(env))["leaky"]
    assert "hf_abcdefgh" not in json.dumps(inst.params) and "[redacted]" in json.dumps(inst.params)


def test_ids_are_per_host_and_stable_and_collisions_get_distinct_suffixes(env: Env) -> None:
    spark = env.add_host("spark", "Spark")
    for node in (env.node, spark):
        for n in ("My Engine", "my-engine"):
            node.add_container(n, engine="vllm", host_ports=ports(18700 if n == "My Engine" else 18701))
    first = [i.id for i in externals(env)]
    assert len(first) == len(set(first)) == 4
    assert first == [i.id for i in externals(env)]                                  # stable across refreshes
    assert sum(1 for i in first if i.startswith("ext-spark-")) == 2 and sum(1 for i in first if i.startswith("ext-local-")) == 2


async def test_m1_hyphenated_host_ids_cannot_collide_and_stop_reaches_only_the_row_clicked(env: Env) -> None:
    """security review 2026-10-04 M1: host `gpu` running `2-vllm` and host `gpu-2` running `vllm` both gave
    `ext-gpu-2-vllm`, and a stop on the compromised host's row was sent to the honest host."""
    honest = env.add_host("gpu-2", "Honest", external_control="stop_start", addresses=("10.0.0.22",))
    evil = env.add_host("gpu", "Compromised", external_control="stop_start", addresses=("10.0.0.21",))
    honest.add_container("vllm", engine="vllm", host_ports=ports(8000, 8000, "127.0.0.1"))
    honest.add_endpoint(8000, container="vllm", engine="vllm", models=("honest-model",))
    evil.add_container("2-vllm", engine="vllm", host_ports=ports(8000, 8000, "127.0.0.1"))
    evil.add_endpoint(8000, container="2-vllm", engine="vllm", models=("evil-model",))
    rows = {i.id: i.host_id for i in env.container.lifecycle.externals() if i.host_id in ("gpu", "gpu-2")}
    assert len(rows) == 2 and set(rows.values()) == {"gpu", "gpu-2"}
    evil_id = next(k for k, v in rows.items() if v == "gpu")
    assert env.container.lifecycle.get(evil_id).host_id == "gpu"
    await env.container.lifecycle.stop(evil_id)
    assert not [o for o, _a in honest.calls if o.startswith("docker.")]
    assert [o for o, _a in evil.calls if o.startswith("docker.")]


def test_hostile_container_and_model_names_are_neutralised(env: Env) -> None:
    hostile = "../../etc; rm -rf $(id) `x`"
    env.node.add_container(hostile, engine="vllm", host_ports=ports(18600))
    env.node.add_endpoint(18600, container=hostile, engine="vllm", models=("m\x1b[31m" + "x" * 150,))
    ext = next(i for i in externals(env) if i.container_name == hostile)
    assert re.fullmatch(r"ext-local-[a-z0-9-]{1,80}", ext.id) and ".." not in ext.id and "/" not in ext.id
    assert re.fullmatch(r"[A-Za-z0-9 _.:\-]{1,64}", ext.name)
    assert all("\x1b" not in m and len(m) <= 200 for m in ext.served_models)


def test_a_vanished_external_shows_stopped_for_a_while_then_disappears(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    import engine_console.services.lifecycle as lc
    clock = {"t": 1000.0}
    monkeypatch.setattr(lc, "now", lambda: clock["t"])
    seed_world(env.node)
    assert "vllm-a" in by_name(externals(env))
    del env.node.containers["vllm-a"]
    got = by_name(externals(env))
    assert got["vllm-a"].state == "stopped" and "no longer running" in (got["vllm-a"].state_reason or "")
    assert got["vllm-a"].vram == [] and got["vllm-a"].host_memory is None
    clock["t"] += lc.GONE_TTL_S + 1
    assert "vllm-a" not in by_name(externals(env))


def test_a_vanished_external_does_not_show_as_stopped_when_its_host_just_went_away(env: Env) -> None:
    seed_world(env.node)
    externals(env)
    env.node.state = "down"
    states = {i.name: i.state for i in externals(env)}
    assert set(states.values()) == {"unreachable"}                                    # not live, not "stopped"
    env.node.state = "up"
    assert by_name(externals(env))["vllm-a"].state == "ready"


def test_first_seen_is_kept_and_history_since_matches_it(env: Env) -> None:
    seed_world(env.node)
    t0 = by_name(externals(env))["vllm-a"].history_since
    assert t0 is not None and by_name(externals(env))["vllm-a"].history_since == t0


def test_get_one_external_instance_and_404_for_unknown(client: TestClient, env: Env) -> None:
    seed_world(env.node)
    one = client.get(f"{P}/instances/ext-local-vllm-a").json()
    assert one["state"] == "ready" and one["managed"] is False and one["management"] == "external"
    assert client.get(f"{P}/instances/ext-local-nope").status_code == 404
    rows = json.dumps([dict(r) for r in env.container.store.all("SELECT * FROM instances")])
    assert "vllm-a" not in rows                                                       # nothing external is persisted


def test_controls_and_clone_as_managed_for_externals_without_permission(env: Env) -> None:
    seed_world(env.node)
    c = by_name(externals(env))["vllm-a"].controls
    assert not (c.start or c.stop or c.restart or c.remove or c.logs) and c.configure       # clone as managed is always offered
    assert c.reason and "monitor-only" in c.reason


def test_controls_follow_the_nodes_external_control_policy_and_mode(env: Env) -> None:
    seed_world(env.node)
    env.node.policy["external_control"] = "stop_start"
    c = by_name(externals(env))["vllm-a"].controls
    assert c.stop and not c.start and not c.restart and not c.remove and not c.logs and c.configure
    env.node.set_mode("monitor")
    c = by_name(externals(env))["vllm-a"].controls
    assert not c.stop and "monitor mode" in (c.reason or "")
    env.node.set_mode("manage")
    env.node.state = "down"
    c = by_name(externals(env))["vllm-a"].controls
    assert not c.stop and "not connected" in (c.reason or "")


# ---- monitor-only enforcement ---------------------------------------------------------------------------------------------------
def test_every_mutating_instance_route_is_409_for_external_engines_by_default(client: TestClient, env: Env) -> None:
    seed_world(env.node)
    paths = [(m.upper(), p) for p, ops in client.app.openapi()["paths"].items() for m in ops  # type: ignore[attr-defined]
             if "/instances/{iid}" in p and m.upper() in ("POST", "PUT", "PATCH", "DELETE")]
    assert {m for m, _ in paths} == {"POST", "PATCH", "DELETE"} and len(paths) >= 5
    for method, path in paths:
        r = client.request(method, path.replace("{iid}", "ext-local-vllm-a"), json={"pinned": True})
        assert r.status_code == 409 and problem_code(r) == "instance_not_managed", (method, path, r.text)
    for path in ("logs", "logs/stream", "command"):
        r = client.get(f"{P}/instances/ext-local-vllm-a/{path}")
        assert r.status_code == 409 and problem_code(r) == "instance_not_managed", path
    assert not any(o in ("docker.stop", "docker.rm", "docker.run", "docker.start", "docker.external", "gateway.start")
                   for o, _ in env.node.calls)                                          # the node was never even asked


def test_stop_and_start_work_when_the_policy_allows_and_go_through_docker_external(client: TestClient, env: Env) -> None:
    seed_world(env.node)
    env.node.policy["external_control"] = "stop_start"
    r = client.post(f"{P}/instances/ext-local-vllm-a/stop")
    assert r.status_code == 200 and r.json()["state"] == "stopped"
    assert env.node.ops("docker.external") == [{"name": "vllm-a", "action": "stop"}]
    # the engine is no longer running, yet it stays listed so it can be started again
    listed = {i["id"]: i for i in client.get(f"{P}/instances").json()["items"]}
    assert listed["ext-local-vllm-a"]["state"] == "stopped" and listed["ext-local-vllm-a"]["controls"]["start"] is True
    assert client.post(f"{P}/instances/ext-local-vllm-a/restart").status_code == 409        # restart is not offered
    again = client.post(f"{P}/instances/ext-local-vllm-a/start")
    assert again.status_code == 200 and env.node.ops("docker.external")[-1] == {"name": "vllm-a", "action": "start"}
    # once it runs again it is a normal external instance
    assert client.get(f"{P}/instances/ext-local-vllm-a").json()["state"] == "ready"
    assert client.delete(f"{P}/instances/ext-local-vllm-a").status_code == 409              # never removable


def test_a_refusal_by_the_node_is_a_403_problem_naming_the_host_and_the_instance_is_left_alone(client: TestClient, env: Env) -> None:
    seed_world(env.node)
    env.node.policy["external_control"] = "stop_start"
    # the operator edited the node's policy after its last hello: the node refuses, and its words reach the user
    from engine_console.domain.fleet import NodeError
    env.node.fail["docker.external"] = NodeError("local", "policy_denied", "this node no longer allows controlling engines")
    r = client.post(f"{P}/instances/ext-local-vllm-a/stop")
    assert r.status_code == 403 and r.json()["code"] == "policy_denied" and r.json()["detail"].startswith("testbox: ")
    assert client.get(f"{P}/instances/ext-local-vllm-a").json()["state"] == "ready"
    env.node.fail.clear()
    env.node.policy["external_control"] = "none"               # and a flip the hub already sees is refused before the node
    n = len(env.node.calls)
    assert client.post(f"{P}/instances/ext-local-vllm-a/stop").status_code == 409
    assert len(env.node.ops("docker.external")) == 1 and len(env.node.calls) >= n


def test_viewer_cannot_force_discovery(env: Env) -> None:
    from engine_console.main import create_app
    with TestClient(create_app(env.cfg, env.container), client=("192.168.1.9", 1)) as remote:
        boot = (env.cfg.data_dir / "bootstrap-admin.key").read_text().strip()
        v = remote.post(f"{P}/keys", json={"name": "v", "role": "viewer"}, headers={"Authorization": f"Bearer {boot}"}).json()["secret"]
        assert remote.post(f"{P}/instances/discover", headers={"Authorization": f"Bearer {v}"}).status_code == 403
        assert remote.post(f"{P}/instances/discover", headers={"Authorization": f"Bearer {boot}"}).status_code == 200


# ---- VRAM, memory, model location ------------------------------------------------------------------------------------------------
def test_vram_host_memory_and_model_location_of_an_external(env: Env) -> None:
    from .fakefleet import cgroup
    seed_world(env.node)
    env.node.gpu_procs = {0: [{"pid": 7, "container": "vllm-a", "used_bytes": 60 * 1024**3, "name": "python"}]}
    env.node.cg["vllm-a"] = cgroup()
    env.node.disks = [{"mount": "/data", "fstype": "zfs", "source": "tank", "size": 10**12, "used": 10**11, "free": 9 * 10**11,
                       "roots": ["hf"]}]
    a = by_name(externals(env))["vllm-a"]
    assert [(v.gpu_index, v.gpu_uuid, round(v.used_gib or 0)) for v in a.vram] == [(0, "GPU-uuid-0", 60)]
    assert a.host_memory is not None and a.host_memory.source == "cgroup" and a.host_memory.shmem_gib == pytest.approx(64.0, abs=1e-3)
    loc = a.model_location
    assert loc is not None and (loc.host_id, loc.root_id, loc.root_label, loc.path) == ("local", "hf", "HF cache", "/data/hf")
    assert loc.drive == "/data (zfs)" and loc.fstype == "zfs" and loc.remote is False and loc.shared_from is None
    assert by_name(externals(env))["sgl-b"].vram == [] and by_name(externals(env))["sgl-b"].host_memory is None


def test_model_location_on_nfs_names_the_machine_that_owns_the_disk(env: Env) -> None:
    spark = env.add_host("spark", "Spark", addresses=("10.0.0.7",))
    assert spark is not None
    env.node.add_container("vllm-n", engine="vllm", host_ports=ports(18500), mounts=[
        {"source": "/mnt/shared/hf", "dest": "/root/.cache/huggingface", "ro": True, "root": "shared", "rel": "hf"}])
    env.node.policy["model_roots"].append({"id": "shared", "path": "/mnt/shared", "kind": "auto", "writable": False,
                                           "label": "Shared (NFS)"})
    env.node.add_endpoint(18500, container="vllm-n", engine="vllm", models=("m",))
    env.node.disks = [{"mount": "/mnt/shared", "fstype": "nfs4", "source": "10.0.0.7:/models", "remote": True,
                       "remote_host": "10.0.0.7", "remote_path": "/models", "roots": ["shared"]}]
    loc = by_name(externals(env, "local"))["vllm-n"].model_location
    assert loc is not None and loc.remote and loc.shared_from == "spark" and loc.drive == "NFS from Spark"
    env.node.disks[0]["remote_host"] = "10.9.9.9"                                      # an export from a machine that is not ours
    loc = by_name(externals(env, "local"))["vllm-n"].model_location
    assert loc is not None and loc.shared_from == "10.9.9.9"


# ---- chat, bench on externals ------------------------------------------------------------------------------------------------------
def test_chat_and_bench_can_target_external_with_confirmation(client: TestClient, env: Env) -> None:
    seed_world(env.node)
    env.port_routes[(18500, "/v1/chat/completions")] = sse_engine
    r = client.post(f"{P}/chat/completions", json={"instance_id": "ext-local-vllm-a", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and json.loads(env.engine_requests[-1].content)["model"] == "org/big"
    assert env.engine_requests[-1].url.port == 18500
    assert client.post(f"{P}/chat/completions", json={"instance_id": "ext-local-sgl-b", "messages": [{"role": "user", "content": "x"}]}).status_code == 409
    no = client.post(f"{P}/bench", json={"instance_id": "ext-local-vllm-a"})
    assert no.status_code == 400 and problem_code(no) == "confirm_external_required"
    assert client.post(f"{P}/bench", json={"instance_id": "ext-local-sgl-b", "confirm_external": True}).status_code == 409   # not ready
    ok = client.post(f"{P}/bench", json={"instance_id": "ext-local-vllm-a", "confirm_external": True,
                                          "suite": {"concurrency": [1], "prompt_tokens": 14, "max_tokens": 2, "rounds": 1}})
    assert ok.status_code == 202 and "external engine" in ok.json()["suite"]["note"]
    client.portal.call(env.container.bench.wait, ok.json()["id"])  # type: ignore[union-attr]
    assert client.get(f"{P}/bench/{ok.json()['id']}").json()["state"] == "completed"


def test_an_external_on_another_host_is_chatted_through_that_hosts_node(client: TestClient, env: Env) -> None:
    spark = env.add_host("spark", "Spark")
    spark.add_container("vllm-s", engine="vllm", host_ports=ports(18500))
    spark.add_endpoint(18500, container="vllm-s", engine="vllm", models=("org/spark-model",))
    spark.serve(18500, "/v1/chat/completions", sse_engine)
    r = client.post(f"{P}/chat/completions", json={"instance_id": "ext-spark-vllm-s", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert len(spark.http_requests) == 1 and not env.node.http_requests                 # only spark's node was asked
    assert json.loads(spark.http_requests[0].content)["model"] == "org/spark-model"


# ---- script instances --------------------------------------------------------------------------------------------------------------------
def with_script(env: Env, node: FakeNode | None = None) -> FakeNode:
    node = node or env.node
    node.policy["scripts"] = [json.loads(json.dumps(SCRIPT))]
    return node


def script_instance(env: Env, host: str = "local") -> Instance:
    return next(i for i in externals(env, host) if i.management == "script")


def test_a_script_is_one_instance_even_when_nothing_runs_and_its_containers_are_not_externals(env: Env) -> None:
    with_script(env)
    inst = script_instance(env)
    assert inst.id == "scr-local-big-model" and inst.script == "big-model" and inst.state == "stopped"
    assert inst.name == "Big model (2 GPUs)" and inst.engine == "sglang" and inst.container_name is None
    assert inst.managed is False and inst.management == "script"
    # two containers of the script's engine start: still ONE instance, and neither is listed as an external
    env.node.add_container("myproj-sglang-a", engine="sglang", image="lmsysorg/sglang:v0.5.14", host_ports=ports(18510, 30000),
                           gpus=["GPU-uuid-0", "GPU-uuid-1"])
    env.node.add_container("myproj-sglang-helper", image="busybox")
    env.node.add_endpoint(18510, container="myproj-sglang-a", engine="sglang", models=("qwen-big",))
    items = externals(env)
    assert [i.id for i in items] == ["scr-local-big-model"]
    inst = items[0]
    assert inst.state == "ready" and inst.port == 18510 and inst.container_name == "myproj-sglang-a"
    assert inst.served_models == ["qwen-big"] and inst.repo_id == "qwen-big" and inst.gpu_ids == [0, 1]


def test_a_script_engine_that_is_running_but_not_answering_yet_is_loading(env: Env) -> None:
    with_script(env)
    env.node.add_container("myproj-sglang-a", engine="sglang", host_ports=ports(18510, 30000))
    assert script_instance(env).state == "loading"
    env.node.add_endpoint(18510, container="myproj-sglang-a", engine="sglang", health=401, models_status=401)
    assert script_instance(env).state == "auth_required"


def test_script_controls_map_to_the_scripts_actions_and_name_the_script(env: Env) -> None:
    with_script(env)
    c = script_instance(env).controls
    assert c.start and not c.stop and c.restart and not c.remove and not c.logs and not c.configure
    assert c.reason and "Big model" in c.reason
    env.node.add_container("myproj-sglang-a", engine="sglang", host_ports=ports(18510, 30000))
    c = script_instance(env).controls
    assert c.stop and not c.start and c.restart
    env.node.policy["scripts"][0]["actions"] = [a for a in env.node.policy["scripts"][0]["actions"] if a["name"] != "down"]
    c = script_instance(env).controls
    assert not c.stop and not c.restart                                                  # no `down` action: nothing to map stop to
    env.node.set_mode("monitor")
    c = script_instance(env).controls
    assert not (c.start or c.stop or c.restart) and "monitor mode" in (c.reason or "")


def test_script_vram_and_memory_aggregate_over_the_scripts_containers(env: Env) -> None:
    from .fakefleet import cgroup
    with_script(env)
    env.node.add_container("myproj-sglang-a", engine="sglang", host_ports=ports(18510, 30000))
    env.node.add_container("myproj-sglang-b", engine="sglang")
    env.node.gpu_procs = {0: [{"pid": 1, "container": "myproj-sglang-a", "used_bytes": 30 * 1024**3},
                              {"pid": 2, "container": "myproj-sglang-b", "used_bytes": 10 * 1024**3}]}
    env.node.cg.update({"myproj-sglang-a": cgroup(10, 1, 9, 8, 0.1), "myproj-sglang-b": cgroup(5, 1, 4, 3, 0.1)})
    inst = script_instance(env)
    assert [round(v.used_gib or 0) for v in inst.vram] == [40]
    assert inst.host_memory is not None and inst.host_memory.total_gib == pytest.approx(15.0, abs=1e-3)


def test_a_policy_script_with_an_odd_name_is_not_listed_rather_than_trusted(env: Env) -> None:
    env.node.policy["scripts"] = [{**SCRIPT, "name": "../evil name"}]
    assert not [i for i in externals(env) if i.management == "script"]


def test_script_instance_ids_are_per_host(env: Env) -> None:
    spark = env.add_host("spark", "Spark")
    with_script(env)
    with_script(env, spark)
    assert {i.id for i in externals(env) if i.management == "script"} == {"scr-local-big-model", "scr-spark-big-model"}


def test_instance_for_container_resolves_managed_script_and_external_engines(env: Env) -> None:
    with_script(env)
    seed_world(env.node)
    env.node.add_container("myproj-sglang-a", engine="sglang")
    life = env.container.lifecycle
    life.externals()
    assert life.instance_for_container("local", "vllm-a") == "ext-local-vllm-a"
    assert life.instance_for_container("local", "myproj-sglang-a") == "scr-local-big-model"
    assert life.instance_for_container("local", "postgres") is None


def test_an_external_engine_answering_503_is_loading_not_unreachable(env: Env) -> None:
    env.node.add_container("sgl-warm", engine="sglang", host_ports=ports(18510, 30000))
    env.node.add_endpoint(18510, container="sgl-warm", engine="sglang", health=503, models_status=None)
    inst = by_name(externals(env))["sgl-warm"]
    assert inst.state == "loading" and "503" in (inst.state_reason or "") and inst.port == 18510
    env.node.add_endpoint(18510, container="sgl-warm", engine="sglang", health=200, models_status=200)
    assert by_name(externals(env))["sgl-warm"].state == "ready"
    env.node.add_endpoint(18510, container="sgl-warm", engine="sglang", health=None, models_status=None)
    assert by_name(externals(env))["sgl-warm"].state == "unreachable"
