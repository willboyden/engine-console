"""Host RAM accounting against fake /proc and /sys/fs/cgroup trees built from measured numbers."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine_console.domain.models import HostMemInfo
from engine_console.services.fit import estimate_host_ram
from engine_console.services.hostmem import HostMemService, alerts_for, parse_meminfo

from .conftest import Env
from .test_api_smoke import P, tick
from .test_discovery import add, discover, models, ports, serve

GIB = 1024**3
KB_PER_GIB = 1024**2
CG = "/user.slice/docker-abc123.scope"
# measured on the reference box, SGLang container: current 68.6, anon 3.3, file 64.5, shmem 64.0, kernel 0.6 GiB
SGLANG = {"current": 68.6, "anon": 3.3, "file": 64.5, "shmem": 64.0, "kernel": 0.6}
MEMINFO = (f"MemTotal: {246 * KB_PER_GIB} kB\nMemFree: {3 * KB_PER_GIB} kB\nMemAvailable: {54 * KB_PER_GIB} kB\n"
           f"Cached: {90 * KB_PER_GIB} kB\nShmem: {66 * KB_PER_GIB} kB\nSwapTotal: {32 * KB_PER_GIB} kB\n"
           f"SwapFree: {8 * KB_PER_GIB} kB\nBogus line\nHugePages_Total: 0\n")


def gib(x: float) -> int:
    return int(x * GIB)


def build(root: Path, pid: int = 1234, cgroup_line: str | None = None, stats: dict[str, float] | None = SGLANG,
          meminfo: str = MEMINFO) -> Path:
    proc = root / "proc"
    (proc / str(pid)).mkdir(parents=True, exist_ok=True)
    (proc / "meminfo").write_text(meminfo)
    (proc / str(pid) / "cgroup").write_text(cgroup_line if cgroup_line is not None else f"0::{CG}\n")
    (proc / str(pid) / "stat").write_text(f"{pid} (engine) S 1 {pid} 0 0\n")
    (proc / str(pid) / "status").write_text("Name:\tengine\nVmRSS:\t   1048576 kB\n")     # 1 GiB
    if stats is not None:
        d = root / "sys" / "fs" / "cgroup" / CG.lstrip("/")
        d.mkdir(parents=True, exist_ok=True)
        (d / "memory.current").write_text(f"{gib(stats['current'])}\n")
        (d / "memory.stat").write_text("\n".join([f"anon {gib(stats['anon'])}", f"file {gib(stats['file'])}",
                                                  f"shmem {gib(stats['shmem'])}", f"kernel {gib(stats['kernel'])}",
                                                  "slab 999", "file_mapped 5"]) + "\n")
    return root


# ---- /proc/meminfo -----------------------------------------------------------------------------------------------------
def test_meminfo_parsing() -> None:
    m = parse_meminfo(MEMINFO, 123.0)
    assert m is not None
    assert (m.total_gib, m.available_gib, m.free_gib, m.cached_gib, m.shmem_gib) == (246.0, 54.0, 3.0, 90.0, 66.0)
    assert m.used_gib == 192.0 and m.swap_total_gib == 32.0 and m.swap_used_gib == 24.0 and m.updated_at == 123.0
    assert parse_meminfo("MemTotal: 1 kB\n", 0) is None and parse_meminfo("", 0) is None
    no_swap = parse_meminfo("MemTotal: 1024 kB\nMemAvailable: 512 kB\n", 0)
    assert no_swap is not None and no_swap.swap_total_gib == 0.0 and no_swap.swap_used_gib == 0.0


def test_host_unreadable_returns_none(tmp_path: Path) -> None:
    assert HostMemService(tmp_path).host() is None


# ---- per-container cgroup accounting ------------------------------------------------------------------------------------------
def test_cgroup_numbers_from_the_measured_sglang_container(tmp_path: Path) -> None:
    hm = HostMemService(build(tmp_path)).container(1234)
    assert hm is not None and hm.source == "cgroup"
    assert hm.total_gib == pytest.approx(68.6, abs=1e-3) and hm.anon_gib == pytest.approx(3.3, abs=1e-3)
    assert hm.shmem_gib == pytest.approx(64.0, abs=1e-3) and hm.kernel_gib == pytest.approx(0.6, abs=1e-3)
    assert hm.cache_gib == pytest.approx(0.5, abs=1e-3)              # file 64.5 - shmem 64.0: nothing double counted
    assert hm.anon_gib + hm.cache_gib + hm.shmem_gib + hm.kernel_gib <= hm.total_gib + 0.3   # parts (rounded) ~ current


def test_kernel_falls_back_to_component_sum_and_cache_never_negative(tmp_path: Path) -> None:
    root = build(tmp_path)
    (root / "sys/fs/cgroup" / CG.lstrip("/") / "memory.stat").write_text(
        f"anon {gib(1)}\nfile {gib(1)}\nshmem {gib(3)}\nkernel_stack {gib(0.25)}\nslab {gib(0.25)}\npagetables {gib(0.5)}\n")
    hm = HostMemService(root).container(1234)
    assert hm is not None and hm.kernel_gib == pytest.approx(1.0) and hm.cache_gib == 0.0


@pytest.mark.parametrize("line", ["0::/../../etc", "0::/a/../../etc", "0::/a b;rm -rf", "0::/x\x00y", "0::relative/path",
                                  "0::/" + "a" * 500, "0::/$(id)", "1:memory:/legacy"])
def test_hostile_cgroup_path_is_never_followed(tmp_path: Path, line: str) -> None:
    root = build(tmp_path, cgroup_line=line + "\n")
    # bait outside the cgroup root that a traversal would read
    (root / "etc").mkdir(exist_ok=True)
    (root / "etc/memory.current").write_text(f"{gib(500)}\n")
    (root / "etc/memory.stat").write_text(f"anon {gib(500)}\nfile 0\nshmem 0\nkernel 0\n")
    hm = HostMemService(root).container(1234)
    assert hm is not None and hm.source == "rss" and hm.total_gib == pytest.approx(1.0)   # fell back, bait unread


def test_symlink_escaping_the_cgroup_root_is_refused(tmp_path: Path) -> None:
    root = build(tmp_path, stats=None)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "memory.current").write_text(f"{gib(400)}\n")
    (outside / "memory.stat").write_text(f"anon {gib(400)}\nfile 0\nshmem 0\nkernel 0\n")
    link = root / "sys/fs/cgroup" / CG.lstrip("/")
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    hm = HostMemService(root).container(1234)
    assert hm is not None and hm.source == "rss" and hm.total_gib == pytest.approx(1.0)


def test_unreadable_cgroup_falls_back_to_summed_process_tree_rss(tmp_path: Path) -> None:
    root = build(tmp_path, stats=None)                                   # cgroup dir absent
    for child, ppid, kb in ((2000, 1234, 2 * KB_PER_GIB), (2001, 2000, KB_PER_GIB // 2), (3000, 1, 99 * KB_PER_GIB)):
        d = root / "proc" / str(child)
        d.mkdir()
        (d / "stat").write_text(f"{child} (worker) S {ppid} 0 0\n")
        (d / "status").write_text(f"VmRSS: {kb} kB\n")
    hm = HostMemService(root).container(1234)
    assert hm is not None and hm.source == "rss"
    assert hm.total_gib == pytest.approx(1.0 + 2.0 + 0.5)                # pid tree only; unrelated pid 3000 excluded
    assert hm.cache_gib == 0.0 and hm.shmem_gib == 0.0


def test_missing_pid_or_no_procfs_gives_none(tmp_path: Path) -> None:
    svc = HostMemService(build(tmp_path))
    assert svc.container(None) is None and svc.container(0) is None and svc.container(-5) is None
    assert svc.container(99999) is None
    assert HostMemService(tmp_path / "nothing").container(1234) is None


# ---- alerts ------------------------------------------------------------------------------------------------------------------------------------
def mem(total: float = 100, avail: float = 50, shmem: float = 0, swap_total: float = 0, swap_used: float = 0) -> HostMemInfo:
    return HostMemInfo(total_gib=total, used_gib=total - avail, available_gib=avail, free_gib=1, cached_gib=1, shmem_gib=shmem,
                       swap_total_gib=swap_total, swap_used_gib=swap_used, updated_at=0)


@pytest.mark.parametrize(("m", "expected"), [
    (mem(avail=50), []), (mem(avail=10.0), []), (mem(avail=9.9), [("warn", "host_ram_low")]),
    (mem(avail=5.0), [("warn", "host_ram_low")]), (mem(avail=4.9), [("crit", "host_ram_low")]),
    (mem(shmem=25), []), (mem(shmem=25.1), [("warn", "shmem_high")]),
    (mem(swap_total=32, swap_used=16), []), (mem(swap_total=32, swap_used=16.1), [("warn", "swap_heavy")]),
    (mem(swap_total=0, swap_used=0), []),
    (mem(avail=3, shmem=40, swap_total=10, swap_used=9), [("crit", "host_ram_low"), ("warn", "shmem_high"), ("warn", "swap_heavy")]),
])
def test_alert_thresholds(m: HostMemInfo, expected: list[tuple[str, str]]) -> None:
    assert [(a.level, a.code) for a in alerts_for(m)] == expected


def test_the_reference_box_triggers_shmem_and_swap_alerts_with_an_explanation() -> None:
    m = parse_meminfo(MEMINFO, 0)
    assert m is not None
    got = {a.code: a for a in alerts_for(m)}
    assert set(got) == {"shmem_high", "swap_heavy"} and "cannot be reclaimed" in got["shmem_high"].message


# ---- API: /hardware, instances, metrics ------------------------------------------------------------------------------------------------------------
def use_root(env: Env, root: Path) -> None:
    svc = HostMemService(root)
    env.container.hostmem = svc
    env.container.lifecycle.hostmem = svc
    env.container.metrics._hostmem = svc          # noqa: SLF001
    env.container.fit._hostmem = svc              # noqa: SLF001


def test_hardware_reports_memory_and_alerts(client: TestClient, env: Env, tmp_path: Path) -> None:
    use_root(env, build(tmp_path / "fs"))
    body = client.get(f"{P}/hardware").json()
    assert body["memory"]["total_gib"] == 246.0 and body["memory"]["swap_used_gib"] == 24.0 and body["memory"]["updated_at"]
    assert {a["code"] for a in body["alerts"]} == {"shmem_high", "swap_heavy"}
    use_root(env, tmp_path / "empty")
    body = client.get(f"{P}/hardware").json()
    assert body["memory"] is None and body["alerts"] == []


def test_managed_instance_carries_host_memory(client: TestClient, env: Env, tmp_path: Path) -> None:
    use_root(env, build(tmp_path / "fs", pid=4242))
    env.seed_local_model()
    env.engine_up()
    inst = client.post(f"{P}/instances", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B",
                                               "params": {"max_model_len": 4096}}).json()
    assert inst["host_memory"] is None                                   # pid not known until the supervisor inspects it
    env.runner.containers[inst["container_name"]]["pid"] = 4242
    tick(client, env)
    for got in (client.get(f"{P}/instances/{inst['id']}").json(), client.get(f"{P}/instances").json()["items"][0]):
        hm = got["host_memory"]
        assert hm["source"] == "cgroup" and hm["total_gib"] == pytest.approx(68.6, abs=1e-3)
        assert set(hm) == {"total_gib", "anon_gib", "cache_gib", "shmem_gib", "kernel_gib", "source"}
    client.post(f"{P}/instances/{inst['id']}/stop")
    assert client.get(f"{P}/instances/{inst['id']}").json()["host_memory"] is None     # not running: nothing to report


def test_external_instance_carries_host_memory_and_null_when_unreadable(client: TestClient, env: Env, tmp_path: Path) -> None:
    use_root(env, build(tmp_path / "fs", pid=4343))
    add(env, "ext-a", "vllm/vllm-openai:1", ["vllm", "serve", "m"], ports(18500), pid=4343)
    serve(env, 18500, models_h=models("m"), metrics="running 1\n")
    add(env, "ext-b", "vllm/vllm-openai:1", ["vllm", "serve", "m"], ports(18501), pid=99999)
    serve(env, 18501, models_h=models("m"))
    got = {i["name"]: i for i in discover(client)}
    assert got["ext-a"]["host_memory"]["source"] == "cgroup" and got["ext-a"]["host_memory"]["shmem_gib"] == pytest.approx(64.0, abs=1e-3)
    assert got["ext-b"]["host_memory"] is None
    assert client.get(f"{P}/instances/ext-ext-a").json()["host_memory"]["anon_gib"] == pytest.approx(3.3, abs=1e-3)


def test_metrics_keys_system_series_and_prometheus_gauges(client: TestClient, env: Env, tmp_path: Path) -> None:
    use_root(env, build(tmp_path / "fs", pid=4343))
    add(env, "ext-a", "ollama/ollama:1", ["ollama", "serve"], ports(18500, 11434), pid=4343)
    serve(env, 18500, models_h=models("llama3", owned_by="library"), health=None)      # no /metrics endpoint at all
    discover(client)
    client.portal.call(env.container.metrics.scrape_once)  # type: ignore[union-attr]
    pts = client.get(f"{P}/metrics/instances/ext-ext-a").json()["points"]
    assert len(pts) == 1                                                  # host RAM is sampled even without engine metrics
    v = pts[0]["values"]
    assert v["host_ram_gib"] == pytest.approx(68.6, abs=1e-3) and v["host_ram_anon_gib"] == pytest.approx(3.3, abs=1e-3)
    assert v["host_ram_shmem_gib"] == pytest.approx(64.0, abs=1e-3) and v["host_ram_cache_gib"] == pytest.approx(0.5, abs=1e-3)
    sysm = client.get(f"{P}/metrics/system", params={"window": "15m"}).json()
    sv = sysm["points"][0]["values"]
    assert set(sv) == {"ram_used_gib", "ram_available_gib", "shmem_gib", "swap_used_gib"}
    assert sv["ram_used_gib"] == 192.0 and sv["ram_available_gib"] == 54.0 and sv["swap_used_gib"] == 24.0
    hourly = client.get(f"{P}/metrics/system", params={"window": "1h"}).json()
    assert hourly["resolution"] == "1h" and hourly["points"][0]["values"]["shmem_gib"] == 66.0
    assert client.get(f"{P}/metrics/system", params={"window": "soon"}).status_code == 400
    prom = client.get("/metrics").text
    assert "engine_console_host_ram_used_gib 192.0" in prom and "engine_console_host_swap_used_gib 24.0" in prom
    assert 'engine_console_instance_host_ram_gib{instance="ext-ext-a"}' in prom
    assert prom.count("engine_console_instance_host_ram_gib{") == 1     # one label set per instance, nothing else


# ---- fit: host_ram ------------------------------------------------------------------------------------------------------------------------------------
HOST100 = mem(total=200, avail=100)


@pytest.mark.parametrize(("extra", "verdict"), [(76.0, "ok"), (76.5, "tight"), (96.0, "tight"), (96.5, "wont_fit")])
def test_host_ram_thresholds(extra: float, verdict: str) -> None:
    r = estimate_host_ram({"pool": extra}, HOST100)              # +4 GiB engine_process baseline => 80 / 80.5 / 100 / 100.5
    assert r.verdict == verdict and r.available_gib == 100.0 and r.total_gib == 200.0
    assert r.needed_gib == pytest.approx(extra + 4.0) and r.breakdown == {"pool": extra, "engine_process": 4.0}


def test_host_ram_unknown_never_fabricated() -> None:
    r = estimate_host_ram({"hicache": float("nan"), "cpu_weight_offload": 8.0}, HOST100)
    assert r.verdict == "unknown" and r.needed_gib == 12.0 and "hicache" not in r.breakdown
    assert any("could not be computed" in n and "hicache" in n for n in r.notes)
    none = estimate_host_ram({}, None)
    assert none.verdict == "unknown" and none.available_gib is None and any("could not be read" in n for n in none.notes)


def test_host_ram_notes_explain_page_cache_and_shared_memory() -> None:
    r = estimate_host_ram({}, mem(total=100, avail=50, shmem=40))
    assert r.verdict == "ok" and any("page cache" in n and "unreclaimable" in n for n in r.notes)
    assert any("shared memory" in n and "cannot be reclaimed" in n for n in r.notes)


def test_fit_endpoint_includes_host_ram(client: TestClient, env: Env, tmp_path: Path) -> None:
    use_root(env, build(tmp_path / "fs"))
    env.hub.add("Qwen/Qwen3-32B", {"config.json": b"{}"}, {"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 2,
                                                          "num_key_value_heads": 1}, params={"BF16": 1_000_000})
    r = client.post(f"{P}/fit", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 512}}).json()
    h = r["host_ram"]
    assert h["verdict"] == "ok" and h["available_gib"] == 54.0 and h["breakdown"] == {"engine_process": 4.0}
    use_root(env, tmp_path / "gone")
    r = client.post(f"{P}/fit", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 512}}).json()
    assert r["host_ram"]["verdict"] == "unknown"


