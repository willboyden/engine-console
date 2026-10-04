"""Inventory, endpoints, samples and metrics: redaction on the node, no env values, loopback probes only."""
from __future__ import annotations

import json
import socket

from ecnode import discovery, proto
from ecnode.agent import filter_metrics
from ecnode.discovery import classify, container_view, parse_params, script_status
from ecnode.policy import Root, parse

from .conftest import (
    ARGV_SECRET,
    CID_ENGINE,
    ENV_VALUE,
    GPU_A,
    GPU_B,
    METRICS_TEXT,
    _w,
    container,
    manage_policy,
)


def test_classify_signatures_and_non_engines():
    assert classify("vllm/vllm-openai:v0.11.0", "") == "vllm"
    assert classify("lmsysorg/sglang:latest", "") == "sglang"
    assert classify("ghcr.io/ggml-org/llama.cpp:server-cuda", "") == "llamacpp"
    assert classify("nvcr.io/nvidia/tensorrt-llm/release:1.0", "") == "trtllm"
    assert classify("ollama/ollama:0.12", "") == "ollama"
    assert classify("custom/image:1", "python3 -m sglang.launch_server --model-path x") == "sglang"
    for img in ("vllm/vllm-router:1", "litellm/litellm:1", "nginx:1", "grafana/grafana:11"):
        assert classify(img, "vllm serve x") is None


def test_container_view_redacts_argv_and_never_reads_env(tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    doc = container("e1", cid=CID_ENGINE, image="vllm/vllm-openai:v1",
                    args=["vllm", "serve", "org/m", "--api-key", ARGV_SECRET, "--max-model-len", "8192",
                          "--enforce-eager"],
                    ports={"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18001"}],
                           "9000/tcp": [{"HostIp": "10.0.0.5", "HostPort": "9000"}]},
                    mounts=[{"Type": "bind", "Source": str(hf / "hub" / "models--a--b"), "Destination": "/m",
                             "RW": False}, {"Type": "volume", "Source": "/var/lib/x", "Destination": "/c", "RW": True}],
                    device_requests=[{"Driver": "", "Count": 0, "DeviceIDs": ["1", GPU_A]}])
    v = container_view(doc, roots=[(Root("hf", str(hf)), str(hf))], gpu_by_index={"0": GPU_A, "1": GPU_B})
    assert v is not None
    blob = json.dumps(v.info)
    assert ARGV_SECRET not in blob and ENV_VALUE not in blob and "Env" not in blob
    assert v.info["params"] == {"model": "org/m", "api_key": "[redacted]", "max_model_len": "8192",
                                "enforce_eager": "true"}
    assert v.info["gpus"] == [GPU_B, GPU_A] and v.info["all_gpus"] is False
    assert v.info["mounts"][0] == {"source": str(hf / "hub" / "models--a--b"), "dest": "/m", "ro": True,
                                   "root": "hf", "rel": "hub/models--a--b"}
    assert v.info["mounts"][1]["root"] is None
    assert v.loopback_ports == [18001] and v.published == {8000: 18001, 9000: 9000}
    assert v.info["created"] == 1788256800.123456789 or abs(v.info["created"] - 1788256800.12) < 0.01
    # the result survives the hub's own whitelist unchanged in shape
    assert proto.clean(v.info, proto.CONTAINER)["argv"] == v.info["argv"]


def test_all_gpus_and_cdi_devices():
    doc = container("e", cid="1" * 64, image="x:1", args=["x"],
                    device_requests=[{"Driver": "cdi", "DeviceIDs": ["nvidia.com/gpu=all"]}])
    assert container_view(doc).info["all_gpus"] is True
    doc = container("e", cid="1" * 64, image="x:1", args=["x"], device_requests=[{"Count": -1}])
    assert container_view(doc).info["all_gpus"] is True


def test_parse_params_display_only():
    assert parse_params(["vllm", "serve", "m", "--tp=2", "--x", "--y", "1"]) == {"model": "m", "tp": "2",
                                                                               "x": "true", "y": "1"}


def test_inventory_endpoints_gateways_and_bare_ports(agent_for, machine, fakebin, engine, prober):
    prober.allowed = {engine.port}
    state = fakebin.state()
    # the console engine publishes nothing (internal network); its gateway publishes the fake engine's port
    state["containers"].append(container(
        "ec-owned-1-gw", cid="9" * 64, image="nginx@sha256:" + "0" * 64, args=["/bin/sh", "-c", "nginx"],
        labels={"engine-console": "1", "engine-console.role": "gateway", "engine-console.instance": "inst_1"},
        ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(engine.port)}]}, pid=4545))
    fakebin.set_state(state)
    scripts = [{"name": "sg", "containers": ["other-*"], "actions": [{"name": "up", "argv": ["/bin/true"]}]},
               {"name": "none", "containers": ["nothing-*"], "actions": [{"name": "up", "argv": ["/bin/true"]}]}]
    a = agent_for(manage_policy(machine, discovery_ports=[engine.port, 1], scripts=scripts))
    a.hello()
    inv = a.refresh_inventory()
    names = {c["name"] for c in inv.msg["containers"]}
    assert names == {"ec-owned-1", "other-engine", "grafana", "ec-owned-1-gw"}
    eps = {e["port"]: e for e in inv.msg["endpoints"]}
    assert set(eps) == {engine.port}                  # the gateway claims the port: no second bare endpoint
    ep = eps[engine.port]
    assert ep["container"] == "ec-owned-1" and ep["engine"] == "vllm" and ep["models"] == ["org/model-a"]
    assert (ep["health_status"], ep["models_status"], ep["metrics_status"]) == (200, 200, 200)
    assert set(prober.tried) <= {engine.port, 1}      # nothing else was ever probed
    assert inv.gateway_ports == {engine.port} and engine.port in a.allowed_ports(False)
    st = {s["name"]: s for s in inv.msg["scripts"]}
    assert st["sg"]["status"] == "up" and "other-engine" in st["sg"]["detail"] and st["none"]["status"] == "down"
    blob = json.dumps(inv.msg)
    assert ENV_VALUE not in blob and ARGV_SECRET not in blob and "topsecretvalue" not in blob
    assert "proxy.invalid" not in json.dumps(a.hello())       # docker info's proxy credentials never leave


def test_bare_discovery_port_with_an_engine(agent_for, machine, fakebin, engine, prober):
    prober.allowed = {engine.port}
    fakebin.update(containers=[])
    a = agent_for(manage_policy(machine, discovery_ports=[engine.port]))
    eps = a.refresh_inventory().msg["endpoints"]
    assert eps == [{"port": engine.port, "container": None, "engine": "vllm", "models": ["org/model-a"],
                    "health_status": 200, "models_status": 200, "metrics_status": 200, "auth_required": False}]


def test_inventory_without_docker_still_probes(agent_for, machine, fakebin, engine, prober):
    prober.allowed = {engine.port}
    fakebin.update(daemon_down=True)
    a = agent_for(manage_policy(machine, discovery_ports=[engine.port]))
    inv = a.refresh_inventory()
    assert inv.msg["containers"] == [] and inv.error and "daemon" in inv.error
    assert [e["port"] for e in inv.msg["endpoints"]] == [engine.port]
    facts = a.hello()["docker"]
    assert facts["available"] is False and "Cannot connect" in facts["error"]


def test_sample_maps_gpu_processes_to_containers(agent_for, machine, fakebin):
    a = agent_for(manage_policy(machine))
    a.hello()
    a.refresh_inventory()
    s1 = a.sample(1)
    procs = s1["gpus"][GPU_A]["procs"]
    assert procs == [{"pid": 4242, "used_bytes": 2048 * 1024 * 1024, "name": "vllm", "container": "ec-owned-1"}]
    assert s1["gpus"][GPU_B]["procs"][0]["container"] is None          # a pid with no container
    assert s1["containers"]["ec-owned-1"]["mem"]["source"] == "cgroup"
    assert s1["containers"]["other-engine"]["mem"]["source"] == "rss"
    assert "grafana" not in s1["containers"]                            # only engine containers
    cg = machine.paths.cgroup / ("user.slice/user-1000.slice/user@1000.service/user.slice/docker-"
                                 f"{CID_ENGINE}.scope") / "cpu.stat"
    cg.write_text("usage_usec 6000000\n")
    s2 = a.sample(2)
    assert s2["containers"]["ec-owned-1"]["cpu_pct"] is not None and s2["containers"]["ec-owned-1"]["cpu_pct"] > 0
    assert proto.sanitize_push(s2)["gpus"][GPU_A]["procs"][0]["container"] == "ec-owned-1"


def test_sample_attribution_and_memory_for_every_cgroup_form(agent_for, machine, fakebin):
    gw_id, sys_id = "c" * 64, "d" * 64
    state = fakebin.state()
    state["containers"] += [
        container("ec-owned-1-gw", cid=gw_id, image="nginx@sha256:" + "0" * 64, args=["/bin/sh"], pid=4545,
                  labels={"engine-console": "1", "engine-console.role": "gateway",
                          "engine-console.instance": "inst_1"}),
        container("rootful-vllm", cid=sys_id, image="vllm/vllm-openai:v1", args=["vllm", "serve", "m"], pid=4646)]
    fakebin.set_state(state)
    proc = machine.paths.proc
    for pid, cg in ((4545, f"/docker/{gw_id}"), (4646, f"/system.slice/docker-{sys_id}.scope")):
        _w(proc / str(pid) / "cgroup", f"0::{cg}\n")
        _w(proc / str(pid) / "comm", "proc\n")
        _w(proc / str(pid) / "stat", f"{pid} (proc) S 1 {pid} {pid} 0 -1 0\n")
        _w(proc / str(pid) / "status", "VmRSS:\t  100 kB\n")
        _w(machine.paths.cgroup / cg.lstrip("/") / "memory.current", "1000\n")
        _w(machine.paths.cgroup / cg.lstrip("/") / "memory.stat", "anon 600\nfile 400\nshmem 0\nkernel 0\n")
    fakebin.smi(apps=[{"pid": 4646, "used_memory": "100", "gpu_uuid": GPU_B},
                      {"pid": 4242, "used_memory": "200", "gpu_uuid": GPU_A}])
    a = agent_for(manage_policy(machine))
    a.hello()
    a.refresh_inventory()
    s = a.sample(1)
    assert s["gpus"][GPU_B]["procs"][0]["container"] == "rootful-vllm"       # system.slice/docker-<id>.scope
    assert s["gpus"][GPU_A]["procs"][0]["container"] == "ec-owned-1"         # user.slice/.../docker-<id>.scope
    assert s["containers"]["ec-owned-1-gw"]["mem"] == {"current": 1000, "anon": 600, "file": 400, "shmem": 0,
                                                       "kernel": 0, "source": "cgroup"}   # /docker/<id>
    assert s["containers"]["rootful-vllm"]["mem"]["source"] == "cgroup"
    assert set(s["containers"]) == {"ec-owned-1", "other-engine", "ec-owned-1-gw", "rootful-vllm"}


def test_metrics_filter_keeps_families_and_their_help_lines():
    out = filter_metrics(METRICS_TEXT, ("vllm:",))
    assert out.splitlines() == ["# HELP vllm:num_requests_running Requests running.",
                                "# TYPE vllm:num_requests_running gauge",
                                'vllm:num_requests_running{model_name="org/model-a"} 1.0']
    big = "".join(f"vllm:x{i} 1\n" for i in range(100000))
    assert len(filter_metrics(big, ("vllm:",), cap=1000)) <= 1000


def test_scrape_needs_configuration_and_uses_prefixes(agent_for, machine, fakebin, engine, prober):
    prober.allowed = {engine.port}
    fakebin.update(containers=[])
    a = agent_for(manage_policy(machine, discovery_ports=[engine.port]))
    a.refresh_inventory()
    assert a.scrape() == []                                             # not configured yet
    a.configure_metrics(0.2, {"vllm": ["vllm:"]})
    assert a.metrics_conf.interval_s == 1.0                             # clamped to 1-60 s
    msgs = a.scrape()
    assert len(msgs) == 1 and msgs[0]["status"] == 200 and "vllm:num_requests_running" in msgs[0]["text"]
    assert "process_cpu" not in msgs[0]["text"] and proto.sanitize_push(msgs[0])["port"] == engine.port


def test_script_status_unknown_without_patterns():
    pol = parse({"scripts": [{"name": "s", "actions": [{"name": "up", "argv": ["/bin/true"]}]}]}, None)
    assert script_status(pol, ["x"], 1.0)[0]["status"] == "unknown"


def test_http_get_never_follows_redirects(engine):
    p = discovery.http_get(engine.port, "/v1/redirect")
    assert p.status == 302
    with socket.socket() as sk:                                         # a loopback port nobody listens on
        sk.bind(("127.0.0.1", 0))
        closed = sk.getsockname()[1]
    assert discovery.http_get(closed, "/v1/models").status is None


def test_huge_inventories_are_shrunk_to_one_frame():
    from ecnode.agent import shrink_inventory
    from ecnode.util import json_size

    def big(name: str, engine: str | None) -> dict:
        return {"name": name, "engine": engine, "managed": False, "argv": ["x" * 1000] * 400,
                "params": {f"p{i}": "v" * 200 for i in range(60)}, "labels": {f"l{i}": "w" * 256 for i in range(48)}}

    msg = {"t": 1.0, "containers": [big(f"c{i}", "vllm" if i < 2 else None) for i in range(256)], "endpoints": [],
           "scripts": []}
    out = shrink_inventory(msg, budget=900 * 1024)
    assert json_size(out) <= 900 * 1024
    assert {c["name"] for c in out["containers"]} >= {"c0", "c1"}       # engines are kept
    assert len(proto.encode_json({"type": "inventory", **out})) <= proto.MAX_JSON


def test_slow_metrics_engines_are_never_probed_on_metrics_and_are_scraped_by_health():
    from .conftest import container
    doc = container("trt", cid="cd" * 32, image="nvcr.io/nvidia/tensorrt-llm/release:1.2.1",
                    args=["trtllm-serve", "serve", "org/m", "--port", "8000"],
                    ports={"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18077"}]})
    view = discovery.container_view(doc)
    assert view.info["engine"] == "trtllm"
    loud, quiet = [], []

    def ok(status: int = 200) -> discovery.Probe:
        return discovery.Probe(status=status, body=b'{"data": [{"id": "org/m"}]}')

    eps = discovery.build_endpoints([view], [], prober=lambda p: loud.append(p) or discovery.PortProbe(p, ok(), ok(), ok()),
                                    quiet_prober=lambda p: quiet.append(p) or discovery.PortProbe(p, ok(), ok(), discovery.Probe()))
    assert quiet == [18077] and loud == []                       # /metrics is never touched by discovery
    assert eps[0]["engine"] == "trtllm" and eps[0]["health_status"] == 200 and eps[0]["metrics_status"] is None
