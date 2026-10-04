"""LaunchSpec.ulimits travel to the node's run spec and are checked hub-side with the node's own allowlist."""
from __future__ import annotations

from engine_console.adapters.base import LaunchSpec
from engine_console.services.runspec import build_run_spec, validate_run_spec


def _spec(ulimits: dict[str, str]) -> dict[str, object]:
    launch = LaunchSpec(image="img:1", argv=["x"], container_port=8000, health_path="/health", metrics_path="/metrics",
                        ulimits=ulimits)
    return build_run_spec(name="ec-t-1", instance="inst_1", image="img:1", launch=launch, gpu_uuids=[], labels={},
                          env={}, secret_env={}, mounts=[], volumes=[], ipc_host=False)


def test_ulimits_pass_through_and_are_validated() -> None:
    ok = _spec({"stack": "67108864", "nofile": "65536:65536"})
    assert ok["ulimits"] == {"stack": "67108864", "nofile": "65536:65536"} and validate_run_spec(ok) == []
    assert "ulimits" not in _spec({})
    assert any("ulimit memlock" in e for e in validate_run_spec(_spec({"memlock": "-1"})))
    assert any("ulimit stack" in e for e in validate_run_spec(_spec({"stack": "lots"})))


def test_trtllm_asks_for_the_documented_stack_size() -> None:
    from engine_console.adapters import load_default_adapters
    from engine_console.adapters.base import Hardware, LaunchContext
    trt = load_default_adapters().get("trtllm")
    hw = Hardware(gpu_ids=[0], gpu_uuids=["GPU-a"], gpu_names=["x"], gpu_total_gib=[96.0], gpu_free_gib=[90.0],
                  compute_capability="12.0", host_ram_gib=256.0)
    spec = trt.build_launch_ctx(LaunchContext(model="org/m", served_name="org/m",
                                              hf_cache_container_path="/root/.cache/huggingface", hw=hw))
    assert spec.ulimits == {"stack": "67108864"}
