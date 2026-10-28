"""Write backend/crates/ec-hub/tests/data/wse_lifecycle_trace.json: one managed instance's whole life through the Python
hub's LifecycleService over the Python fake node (tests/core/fakefleet.py, conftest `Env`), recorded as
(a) every request the hub sent the node, with its arguments (the docker.run spec, gateway.start, docker.stop/rm with
role and instance, docker.logs, docker.render, ...), and (b) what the service answered after each step. The Rust
replay (tests/test_wse_lifecycle_trace.rs) drives the same steps over the Rust fake node and compares both, after
normalising only the random instance id and wall-clock times.

The fit estimator is kept out of the picture (no model description in the fake library, explicit GPUs), so the trace
pins the lifecycle itself; the fit is pinned by the fit tests.

    "$(node backend/parity/lib/python-release.mjs)/backend/.venv/bin/python" -I backend/parity/gen/wse_lifecycle_trace.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyref import DATA, tree  # noqa: E402

BACKEND = tree() / "backend"  # the reference's tests/ (fakes, fixtures) are imported from here
sys.path.insert(0, str(BACKEND))

import httpx  # noqa: E402

from engine_console.domain.models import InstanceCreate, InstancePatch  # noqa: E402
from engine_console.domain.refs import ModelRef  # noqa: E402

from tests.core.conftest import Env  # noqa: E402

OUT = DATA / "wse_lifecycle_trace.json"
SECRET = "sk-live-abcdef123456"


async def scenario(env: Env) -> list[dict[str, Any]]:
    life = env.container.lifecycle
    steps: list[dict[str, Any]] = []
    seen_calls = 0

    def record(name: str, value: Any) -> None:
        nonlocal seen_calls
        calls = [[op, args] for op, args in env.node.calls[seen_calls:]]
        seen_calls = len(env.node.calls)
        steps.append({"step": name, "value": value, "calls": calls})

    env.library.add("local", ModelRef(format="hf", repo_id="Qwen/Qwen3-32B"), root_id="hf",
                    rel="hub/models--Qwen--Qwen3-32B")
    req = InstanceCreate(engine="fake", repo_id="Qwen/Qwen3-32B", gpu_ids=[1], name="trace me",
                         params={"max_model_len": 4096, "api_key": SECRET, "kv_dtype": "fp8"})
    inst = await life.create(req)
    record("create", inst.model_dump(mode="json"))
    cn = inst.container_name
    env.node.logs[cn] = "boot\nloading weights 55%\n"
    await life.tick()
    record("tick-loading", life.get(inst.id).model_dump(mode="json"))
    env.engine_routes["/health"] = lambda r: httpx.Response(200, text="ok")
    await life.tick()
    record("tick-ready", life.get(inst.id).model_dump(mode="json"))
    env.node.logs[cn] = f"key {SECRET} token hf_abcdefghijkl1234\nsecond\n"
    record("logs", await life.logs(inst.id, 50))
    record("command", (await life.command(inst.id)).model_dump(mode="json"))
    record("patch", life.patch(inst.id, InstancePatch(pinned=True, name="renamed", ttl_idle_s=600)).model_dump(mode="json"))
    record("stop", (await life.stop(inst.id)).model_dump(mode="json"))
    record("start", (await life.start(inst.id)).model_dump(mode="json"))
    record("restart", (await life.restart(inst.id)).model_dump(mode="json"))
    env.node.logs[cn] = "CUDA error: out of memory\n"
    env.node.containers[cn].update(state="exited", exit_code=137)
    await life.tick()
    record("crash", life.get(inst.id).model_dump(mode="json"))
    await life.remove(inst.id)
    record("remove", None)
    env.node.add_container("ec-old-abc123", managed=True, image="fake/engine:1.0", labels={
        "engine-console.instance": "inst_old", "engine-console.engine": "fake", "engine-console.model": "Qwen/Qwen3-32B",
        "engine-console.port": "18007", "engine-console.name": "old", "engine-console.gpus": "GPU-uuid-1",
        "engine-console.role": "engine"})
    record("adopt", await life.adopt())
    record("adopted", life.get("inst_old").model_dump(mode="json"))
    record("list", life.list_page().model_dump(mode="json"))
    return steps


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        env = Env(Path(tmp))
        steps = asyncio.run(scenario(env))
    OUT.write_text(json.dumps(steps, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name}: {len(steps)} steps, {sum(len(s['calls']) for s in steps)} node calls")


if __name__ == "__main__":
    main()
