"""Review M7: series keyed by names a node chooses are capped per host, expire, and their rollups age out."""
from __future__ import annotations

import pytest

from engine_console.services import metrics as mm

from .conftest import Env
from .fakefleet import ports
from .test_metrics_fleet import push, use_clock


def engines(node: object, n: int, base: int = 18500) -> list[str]:
    """n external engines on a node (their series are keyed by the container names the node chose)."""
    names = []
    for i in range(n):
        name = f"vllm-{base + i}"
        node.add_container(name, engine="vllm", host_ports=ports(base + i))   # type: ignore[attr-defined]
        node.add_endpoint(base + i, container=name, engine="vllm", models=("m",))   # type: ignore[attr-defined]
        names.append(name)
    return names


@pytest.fixture(autouse=True)
def vllm_parser(env: Env) -> None:
    fake = type(env.container.adapters.get("fake"))()
    fake.id = "vllm"                                   # stands in for the real vllm adapter's parser
    env.container.adapters.register(fake)


def rollup_ids(env: Env) -> set[str]:
    return {r["instance_id"] for r in env.container.store.all("SELECT DISTINCT instance_id FROM metric_rollups")}


def test_one_host_cannot_create_more_series_than_the_cap_and_the_stalest_goes_first(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mm, "MAX_KEYS_PER_HOST", 4)
    clock = use_clock(env)
    m = env.container.metrics
    names = engines(env.node, 7)
    for n in names:
        m.ingest("local", push(container=n, engine="vllm", text="running 1\n"))
        clock.t += 1
    assert len(m._keys["local"]) == 4                                        # noqa: SLF001
    assert len(m._ring) <= 4                                                 # noqa: SLF001
    assert rollup_ids(env) == {f"ext-local-{n}" for n in names[-4:]}         # the three oldest lost memory and rows
    assert m.latest(f"ext-local-{names[0]}") is None and m.latest(f"ext-local-{names[-1]}") is not None


def test_a_flooding_host_does_not_evict_another_hosts_series(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mm, "MAX_KEYS_PER_HOST", 3)
    clock = use_clock(env)
    spark = env.add_host("spark", "Spark")
    m = env.container.metrics
    engines(spark, 1, 19000)
    m.ingest("spark", push(container="vllm-19000", engine="vllm"))
    for n in engines(env.node, 8):
        clock.t += 1
        m.ingest("local", push(container=n, engine="vllm"))
    assert m.latest("ext-spark-vllm-19000") is not None and len(m._keys["spark"]) == 1   # noqa: SLF001


def test_a_series_nobody_updates_expires_from_memory_and_its_rows_after_the_retention(env: Env) -> None:
    clock = use_clock(env)
    m = env.container.metrics
    (name,) = engines(env.node, 1)
    m.ingest("local", push(container=name, engine="vllm"))
    iid = f"ext-local-{name}"
    m.prune(clock.t + mm.KEY_TTL_S - 1, force=True)
    assert m.latest(iid) is not None                                          # a short silence keeps the history
    m.prune(clock.t + mm.KEY_TTL_S + 1, force=True)
    assert m.latest(iid) is None and "local" not in m._keys and iid not in m._last_counters   # noqa: SLF001
    assert iid in rollup_ids(env)                                             # rows outlive the memory
    m.prune(clock.t + mm.HOURLY_KEEP_S + 7200, force=True)
    left = env.container.store.all("SELECT res FROM metric_rollups WHERE instance_id=?", (iid,))
    assert {r["res"] for r in left} == {"d"}                                  # the hourly rows are gone, the daily stay
    m.prune(clock.t + mm.DAILY_KEEP_S + 2 * 86400, force=True)
    assert iid not in rollup_ids(env)


def test_the_bookkeeping_of_a_host_that_left_hosts_json_is_forgotten(env: Env) -> None:
    m = env.container.metrics
    m._last_seq["gone"] = 5                                                   # noqa: SLF001
    m._configured["gone"] = (True, 1.0)                                       # noqa: SLF001
    m._warned.update({("gone", 1.0), ("local", -1.0)})                        # noqa: SLF001
    m.prune(force=True)
    assert "gone" not in m._last_seq and "gone" not in m._configured and not m._warned   # noqa: SLF001


def test_the_prune_is_throttled_and_the_cap_survives_a_restart(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = use_clock(env)
    m = env.container.metrics
    names = engines(env.node, 5)
    for n in names:
        m.ingest("local", push(container=n, engine="vllm"))
    m.prune()
    m._keys.clear()                                                           # noqa: SLF001 - a restart forgets memory
    m._adopted = False                                                        # noqa: SLF001
    monkeypatch.setattr(mm, "MAX_KEYS_PER_HOST", 2)
    m.prune(clock.t + 1)                                                      # inside PRUNE_EVERY_S: does nothing
    assert not m._keys                                                        # noqa: SLF001
    m.prune(clock.t + mm.PRUNE_EVERY_S + 1)
    assert len(m._keys["local"]) == 2 and len(rollup_ids(env)) == 2           # noqa: SLF001
