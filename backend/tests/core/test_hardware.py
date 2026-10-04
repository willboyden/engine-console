"""Per-host hardware from what the node pushed (hello = static, sample = live), bytes -> GiB, never read from the hub itself."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from engine_console.domain.errors import NotFound
from engine_console.domain.fleet import NodeUnavailable
from engine_console.services.hostmem import (
    alerts_for,
    container_memory,
    meminfo_from_sample,
    parse_meminfo,
    sum_memory,
)

from .conftest import Env
from .fakefleet import GIB, cgroup

P = "/api/v1"
KB_PER_GIB = 1024**2
MEMINFO = (f"MemTotal: {246 * KB_PER_GIB} kB\nMemFree: {3 * KB_PER_GIB} kB\nMemAvailable: {54 * KB_PER_GIB} kB\n"
           f"Cached: {90 * KB_PER_GIB} kB\nShmem: {66 * KB_PER_GIB} kB\nSwapTotal: {32 * KB_PER_GIB} kB\n"
           f"SwapFree: {8 * KB_PER_GIB} kB\nBogus line\nHugePages_Total: 0\n")


# ---- the report ---------------------------------------------------------------------------------------------------------------------
def test_report_shape_numbers_and_alerts_from_hello_and_sample(env: Env) -> None:
    env.node.set_gpu_free([90.0, 40.0])
    rep = env.container.hardware.report("local")
    assert rep.host_id == "local" and rep.source == "nvml"
    assert [(g.index, g.uuid) for g in rep.gpus] == [(0, "GPU-uuid-0"), (1, "GPU-uuid-1")]
    g0, g1 = rep.gpus
    assert (g0.total_gib, g0.free_gib, g0.used_gib) == (96.0, 90.0, 6.0) and g1.free_gib == 40.0
    assert g0.util_pct == 5.0 and g0.temp_c == 40.0 and g0.power_w == 90.0 and g0.fan_pct == 30.0
    assert g0.compute_capability == "12.0" and g0.name == "RTX PRO 6000 Blackwell"
    assert rep.host_ram_gib == 246.0 and rep.host_ram_free_gib == 54.0
    assert rep.memory is not None and (rep.memory.used_gib, rep.memory.swap_used_gib, rep.memory.shmem_gib) == (192.0, 24.0, 66.0)
    assert {a.code for a in rep.alerts} == {"shmem_high", "swap_heavy"}


def test_each_host_reports_its_own_numbers(env: Env) -> None:
    spark = env.add_host("spark", "Spark", gpus=1, gpu_gib=119.0, ram_gib=128.0, source="nvidia-smi")
    spark.mem.update(total=128.0, available=100.0, shmem=1.0, swap_total=0.0, swap_free=0.0)
    rep = env.container.hardware.report("spark")
    assert len(rep.gpus) == 1 and rep.gpus[0].total_gib == 119.0 and rep.source == "nvidia-smi"
    assert rep.host_ram_gib == 128.0 and rep.alerts == []
    assert len(env.container.hardware.report("local").gpus) == 2


def test_unified_memory_gpus_take_their_budget_from_host_ram(env: Env) -> None:
    gb = env.add_host("gb", "GB10", gpus=1, gpu_gib=119.0, unified=True, ram_gib=119.0)
    gb.mem.update(total=119.0, available=30.0)
    g = env.container.hardware.report("gb").gpus[0]
    assert g.total_gib == 119.0 and g.free_gib == pytest.approx(30.0) and g.used_gib == pytest.approx(89.0)
    hw = env.container.hardware.hardware("gb")
    assert hw.unified_memory is True and hw.gpu_free_gib == [pytest.approx(30.0)]


def test_a_host_that_is_down_or_silent_has_no_numbers_rather_than_stale_ones(env: Env) -> None:
    env.node.state, env.node.link_error = "down", "no route to host"
    with pytest.raises(NodeUnavailable) as ei:
        env.container.hardware.report("local")
    assert "no route to host" in ei.value.message
    assert env.container.hardware.host_memory("local") is None
    env.node.state = "stale"
    assert env.container.hardware.report("local").gpus                      # stale: still the last sample
    with pytest.raises(NotFound) as e2:
        env.container.hardware.report("nope")
    assert e2.value.code == "unknown_host"


def test_before_the_first_sample_the_hub_says_so(env: Env) -> None:
    env.node.hello["gpus"] = env.node.hello["gpus"]
    original = env.node.latest
    env.node.latest = lambda kind: None if kind == "sample" else original(kind)  # type: ignore[method-assign,assignment]
    with pytest.raises(NodeUnavailable) as ei:
        env.container.hardware.report("local")
    assert "first sample" in ei.value.message


# ---- the adapter's view -------------------------------------------------------------------------------------------------------------------
def test_adapter_view_is_restricted_to_the_chosen_gpus_and_carries_host_facts(env: Env) -> None:
    env.node.set_gpu_free([90.0, 40.0])
    spark = env.add_host("arm", "ARM", arch="aarch64", cc="12.1")
    assert spark is not None
    hw = env.container.hardware.hardware("local", [1])
    assert (hw.gpu_ids, hw.gpu_uuids, hw.gpu_free_gib, hw.compute_capability) == ([1], ["GPU-uuid-1"], [40.0], "12.0")
    assert (hw.host_id, hw.arch, hw.unified_memory, hw.host_ram_gib) == ("local", "x86_64", False, 246.0)
    assert env.container.hardware.hardware("local", None).gpu_ids == [0, 1] and env.container.hardware.hardware("local", []).gpu_ids == [0, 1]
    arm = env.container.hardware.hardware("arm")
    assert arm.arch == "aarch64" and arm.host_id == "arm" and arm.compute_capability == "12.1"
    assert env.container.hardware.hardware("local", [9]).gpu_ids == []        # an index the host does not have selects nothing


def test_a_host_without_gpus_reports_an_empty_but_valid_hardware(env: Env) -> None:
    env.add_host("cpu", "CPU box", gpus=0)
    hw = env.container.hardware.hardware("cpu")
    assert hw.gpu_ids == [] and hw.compute_capability == "unknown"
    assert env.container.hardware.report("cpu").gpus == []


# ---- the API ---------------------------------------------------------------------------------------------------------------------------------
def test_hardware_route_defaults_to_local_and_takes_a_host(client: TestClient, env: Env) -> None:
    env.add_host("spark", "Spark", gpus=1)
    body = client.get(f"{P}/hardware").json()
    assert body["host_id"] == "local" and [g["uuid"] for g in body["gpus"]] == ["GPU-uuid-0", "GPU-uuid-1"] and body["source"] == "nvml"
    assert body["memory"]["total_gib"] == 246.0 and {a["code"] for a in body["alerts"]} == {"shmem_high", "swap_heavy"}
    sp = client.get(f"{P}/hardware", params={"host": "spark"}).json()
    assert sp["host_id"] == "spark" and len(sp["gpus"]) == 1
    assert client.get(f"{P}/hardware", params={"host": "ghost"}).json()["code"] == "unknown_host"
    assert client.get(f"{P}/hardware", params={"host": "Bad Host!"}).status_code == 422
    env.node.state = "down"
    down = client.get(f"{P}/hardware")
    assert down.status_code == 503 and down.json()["code"] == "node_unavailable"


# ---- the pure helpers (hostmem) -------------------------------------------------------------------------------------------------------------
def test_meminfo_text_parsing_is_unchanged() -> None:
    m = parse_meminfo(MEMINFO, 123.0)
    assert m is not None
    assert (m.total_gib, m.available_gib, m.free_gib, m.cached_gib, m.shmem_gib) == (246.0, 54.0, 3.0, 90.0, 66.0)
    assert m.used_gib == 192.0 and m.swap_total_gib == 32.0 and m.swap_used_gib == 24.0 and m.updated_at == 123.0
    assert parse_meminfo("MemTotal: 1 kB\n", 0) is None and parse_meminfo("", 0) is None
    no_swap = parse_meminfo("MemTotal: 1024 kB\nMemAvailable: 512 kB\n", 0)
    assert no_swap is not None and no_swap.swap_total_gib == 0.0 and no_swap.swap_used_gib == 0.0


def test_a_samples_mem_block_converts_bytes_to_gib_and_rejects_nonsense() -> None:
    m = meminfo_from_sample({"total": 246 * GIB, "available": 54 * GIB, "free": 3 * GIB, "cached": 90 * GIB, "shmem": 66 * GIB,
                             "swap_total": 32 * GIB, "swap_free": 8 * GIB}, 5.0)
    assert m is not None and (m.total_gib, m.used_gib, m.swap_used_gib, m.updated_at) == (246.0, 192.0, 24.0, 5.0)
    assert meminfo_from_sample(None, 1.0) is None and meminfo_from_sample({}, 1.0) is None
    assert meminfo_from_sample({"total": -1, "available": 5}, 1.0) is None
    assert meminfo_from_sample({"total": float("inf"), "available": 5}, 1.0) is None
    assert meminfo_from_sample({"total": True, "available": 5}, 1.0) is None
    swapless = meminfo_from_sample({"total": 2 * GIB, "available": GIB}, None)
    assert swapless is not None and swapless.swap_used_gib == 0.0 and swapless.updated_at == 0.0


def test_container_memory_cache_is_file_minus_shmem_and_never_negative() -> None:
    hm = container_memory(cgroup()["mem"])
    assert hm is not None and hm.source == "cgroup"
    assert hm.total_gib == pytest.approx(68.6, abs=1e-3) and hm.anon_gib == pytest.approx(3.3, abs=1e-3)
    assert hm.shmem_gib == pytest.approx(64.0, abs=1e-3) and hm.kernel_gib == pytest.approx(0.6, abs=1e-3)
    assert hm.cache_gib == pytest.approx(0.5, abs=1e-3)                      # file 64.5 - shmem 64.0: nothing double counted
    odd = container_memory({"current": GIB, "anon": GIB, "file": GIB, "shmem": 3 * GIB, "kernel": 0, "source": "rss"})
    assert odd is not None and odd.cache_gib == 0.0 and odd.source == "rss"
    assert container_memory(None) is None and container_memory({"anon": GIB}) is None


def test_sum_memory_adds_parts_and_is_rss_if_any_part_is() -> None:
    a, b = container_memory(cgroup(10, 1, 9, 8, 0.1)["mem"]), container_memory({**cgroup(5, 1, 4, 3, 0.1)["mem"], "source": "rss"})
    assert a is not None and b is not None
    total = sum_memory([a, b])
    assert total is not None and total.total_gib == pytest.approx(15.0) and total.source == "rss"
    assert sum_memory([]) is None


@pytest.mark.parametrize(("avail", "shmem", "swap_used", "expected"), [
    (50, 0, 0, []), (10.0, 0, 0, []), (9.9, 0, 0, [("warn", "host_ram_low")]), (5.0, 0, 0, [("warn", "host_ram_low")]),
    (4.9, 0, 0, [("crit", "host_ram_low")]), (50, 25, 0, []), (50, 25.1, 0, [("warn", "shmem_high")]),
    (50, 0, 16, []), (50, 0, 16.1, [("warn", "swap_heavy")]),
    (3, 40, 9, [("crit", "host_ram_low"), ("warn", "shmem_high"), ("warn", "swap_heavy")])])
def test_alert_thresholds(avail: float, shmem: float, swap_used: float, expected: list[tuple[str, str]]) -> None:
    from engine_console.domain.models import HostMemInfo
    swap_total = 32.0 if swap_used > 10 else 10.0 if swap_used else 0.0
    m = HostMemInfo(total_gib=100, used_gib=100 - avail, available_gib=avail, free_gib=1, cached_gib=1, shmem_gib=shmem,
                    swap_total_gib=swap_total, swap_used_gib=swap_used, updated_at=0)
    assert [(a.level, a.code) for a in alerts_for(m)] == expected


def test_the_reference_box_triggers_shmem_and_swap_alerts_with_an_explanation() -> None:
    m = parse_meminfo(MEMINFO, 0)
    assert m is not None
    got = {a.code: a for a in alerts_for(m)}
    assert set(got) == {"shmem_high", "swap_heavy"} and "cannot be reclaimed" in got["shmem_high"].message
