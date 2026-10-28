"""Write backend/crates/ec-hub/tests/data/wse_externals_vectors.json: Python's derived instances
(services/externals.py `build_host_instances`) for several inventories built with the Python fake node
(tests/core/fakefleet.py): the node's latest hello / sample / inventory / disks and link, the other hosts, the busy
scripts, and the Instance list Python builds from them. tests/test_wse_vectors.rs (externals part) rebuilds the same
node in Rust and compares every field (states, reasons, controls, VRAM/RAM, model location, ids).

    "$(node backend/parity/lib/python-release.mjs)/backend/.venv/bin/python" -I backend/parity/gen/wse_externals_vectors.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyref import DATA, tree  # noqa: E402

BACKEND = tree() / "backend"  # the reference's tests/ (fakes, fixtures) are imported from here
sys.path.insert(0, str(BACKEND))

from engine_console.domain.fleet import HostConfig  # noqa: E402
from engine_console.services.externals import HostCtx, build_host_instances  # noqa: E402

from tests.core.fakefleet import BIG_MODEL_SCRIPT, FakeNode, cgroup, ports  # noqa: E402

OUT = DATA / "wse_externals_vectors.json"


def world(node: FakeNode) -> None:
    node.add_container("vllm-a", image="vllm/vllm-openai:v0.23.0", engine="vllm", host_ports=ports(18500),
                       params={"model": "org/big", "api_key": "[redacted]", "port": "8000", "note": "hf_abcdefghijkl1234"},
                       mounts=[{"source": "/data/hf", "dest": "/root/.cache/huggingface", "ro": True, "root": "hf", "rel": ""}],
                       gpus=["GPU-uuid-0"], started_at=500.0)
    node.add_endpoint(18500, container="vllm-a", engine="vllm", models=("org/big", "m\x1b2"), health=200, models_status=200,
                      metrics=200)
    node.add_container("sgl-b", image="lmsysorg/sglang:v0.5.14", engine="sglang", host_ports=ports(18501, 30000), all_gpus=True)
    node.add_endpoint(18501, container="sgl-b", engine="sglang", health=401, models_status=401, auth_required=True)
    node.add_container("ollama-c", image="ollama/ollama:0.5", engine="ollama", host_ports=ports(18502, 11434))
    node.add_endpoint(18502, container="ollama-c", engine="ollama", models=("llama3",), health=None, models_status=200)
    node.add_container("warm", engine="sglang", host_ports=ports(18505, 30000))
    node.add_endpoint(18505, container="warm", engine="sglang", health=503, models_status=None)
    node.add_container("dead", engine="vllm", host_ports=ports(18506))
    node.add_endpoint(18506, container="dead", engine="vllm", health=None, models_status=None)
    node.add_container("router", image="ghcr.io/berriai/litellm:main", host_ports=ports(4000, 4000))
    node.add_container("vllm-noport", image="vllm/vllm-openai:v0.23.0", engine="vllm")
    node.add_container("vllm-unprobed", engine="vllm", host_ports=ports(18509))
    node.add_container("stopped-one", engine="vllm", state="exited", host_ports=ports(18510))
    node.add_container("My Engine", engine="vllm", host_ports=ports(18700))
    node.add_container("my-engine", engine="vllm", host_ports=ports(18701))
    node.add_container("../../etc; rm -rf $(id)", engine="vllm", host_ports=ports(18702))
    node.add_container("mine-gw", managed=True, labels={"engine-console.role": "gateway"}, host_ports=ports(18503, 8080, "127.0.0.1"))
    node.add_endpoint(18503, container="mine-gw", engine="vllm", models=("owned",))
    node.add_endpoint(9100, engine="vllm", models=("bare-model",), health=None, models_status=200, metrics=200)
    node.add_endpoint(9101, health=200, models_status=None)
    node.add_endpoint(9102, models=("x",), health=200, models_status=200, metrics=None)
    node.add_container("myproj-sglang-a", engine="sglang", image="lmsysorg/sglang:v0.5.14", host_ports=ports(18520, 30000),
                       gpus=["GPU-uuid-1", "GPU-uuid-0"], started_at=300.0,
                       mounts=[{"source": "/mnt/shared/hf", "dest": "/models", "ro": True, "root": "shared", "rel": "hf"}])
    node.add_container("myproj-sglang-helper", image="busybox", started_at=200.0)
    node.add_endpoint(18520, container="myproj-sglang-a", engine="sglang", models=("qwen-big",))
    node.gpu_procs = {0: [{"pid": 7, "container": "vllm-a", "used_bytes": 60 * 1024**3, "name": "python"},
                          {"pid": 8, "container": "myproj-sglang-a", "used_bytes": 10 * 1024**3}],
                      1: [{"pid": 9, "container": "myproj-sglang-a", "name": "python"}]}
    node.cg.update({"vllm-a": cgroup(), "myproj-sglang-a": cgroup(10, 1, 9, 8, 0.1), "myproj-sglang-helper": cgroup(1, 1, 0, 0, 0)})
    node.disks = [{"mount": "/data", "fstype": "zfs", "source": "tank", "model": "Samsung 990", "roots": ["hf"]},
                  {"mount": "/mnt/shared", "fstype": "nfs4", "source": "10.0.0.7:/models", "remote": True,
                   "remote_host": "10.0.0.7", "remote_path": "/models", "roots": ["shared"]}]
    node.policy["model_roots"].append({"id": "shared", "path": "/mnt/shared", "kind": "auto", "writable": False,
                                       "label": "Shared (NFS)"})
    script = copy.deepcopy(BIG_MODEL_SCRIPT)
    script["containers"] = ["myproj-sglang-*"]
    other = {**copy.deepcopy(BIG_MODEL_SCRIPT), "name": "idle", "label": "  Idle\x07 one  ", "engine": None,
             "containers": ["nothing-*"], "actions": [{"name": "up", "label": "Start"}]}
    odd = {**copy.deepcopy(BIG_MODEL_SCRIPT), "name": "../evil"}
    node.policy["scripts"] = [script, other, odd]


def snapshot(node: FakeNode) -> dict[str, Any]:
    link = node.link()
    return {"hello": node.latest("hello"), "sample": node.latest("sample"), "inventory": node.latest("inventory"),
            "disks": node.latest("disks"), "link": link.model_dump()}


def case(name: str, node: FakeNode, busy: dict[str, str], others: dict[str, Any]) -> dict[str, Any]:
    ctx = HostCtx(host_id=node.host.id, label=node.host.label, node=node,
                  others={h: (lab, frozenset(addrs)) for h, (lab, addrs) in others.items()})
    seen: dict[str, float] = {}
    out = build_host_instances(ctx, seen, 1000.0, busy)
    return {"name": name, "host_id": node.host.id, "label": node.host.label, **snapshot(node), "busy": busy,
            "others": others, "instances": [i.model_dump(mode="json") for i in out], "first_seen": seen}


def main() -> None:
    others = {"local": ["testbox", ["10.0.0.5"]], "spark": ["Spark", ["10.0.0.7"]], "gpu-2": ["Two", ["10.0.0.9"]]}
    cases = []
    for variant in ("manage", "stop_start", "monitor", "down", "busy-up", "busy-down", "skew"):
        node = FakeNode(HostConfig(id="gpu-2" if variant == "skew" else "local", label="testbox"))
        world(node)
        busy: dict[str, str] = {}
        if variant == "stop_start":
            node.policy["external_control"] = "stop_start"
        elif variant == "monitor":
            node.set_mode("monitor")
        elif variant == "down":
            node.latest("sample")          # keep the last sample the hub saw
            node.state, node.link_error = "down", "no route to host"
        elif variant == "busy-up":
            busy = {"big-model": "up", "idle": "up"}
            del node.containers["myproj-sglang-a"]
        elif variant == "busy-down":
            busy = {"big-model": "down"}
        elif variant == "skew":
            node.skew_s = 12.5
        cases.append(case(variant, node, busy, others))
    OUT.write_text(json.dumps(cases, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name}: {len(cases)} cases, {sum(len(c['instances']) for c in cases)} instances")


if __name__ == "__main__":
    main()
