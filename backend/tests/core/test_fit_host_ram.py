"""Host-RAM fit verdicts (pure) and the fit endpoint's host-aware wiring."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from engine_console.domain.models import HostMemInfo, SettingsPatch
from engine_console.services.fit import estimate_host_ram

from .conftest import QWEN_CFG, Env
from .fakes import FakeMultiAdapter

P = "/api/v1"


def mem(total: float = 100, avail: float = 50, shmem: float = 0) -> HostMemInfo:
    return HostMemInfo(total_gib=total, used_gib=total - avail, available_gib=avail, free_gib=1, cached_gib=1, shmem_gib=shmem,
                       swap_total_gib=0, swap_used_gib=0, updated_at=0)


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


def seed_hub(env: Env) -> None:
    env.hub.add("Qwen/Qwen3-32B", {"config.json": b"{}"}, {"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 2,
                                                          "num_key_value_heads": 1}, params={"BF16": 1_000_000})


def test_fit_endpoint_includes_host_ram_of_the_chosen_host(client: TestClient, env: Env) -> None:
    seed_hub(env)
    body = {"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 512}}
    h = client.post(f"{P}/fit", json=body).json()["host_ram"]
    assert h["verdict"] == "ok" and h["available_gib"] == 54.0 and h["breakdown"] == {"engine_process": 4.0}
    small = env.add_host("small", "Small box", ram_gib=16.0)
    small.mem.update(total=16.0, available=5.0)                  # the same model against another machine's RAM
    r = client.post(f"{P}/fit", json={**body, "host_id": "small"}).json()
    assert r["host_ram"]["available_gib"] == 5.0 and r["host_ram"]["verdict"] == "ok"       # 4 GiB needed of 5 available
    small.mem.update(available=3.0)
    assert client.post(f"{P}/fit", json={**body, "host_id": "small"}).json()["host_ram"]["verdict"] == "wont_fit"
    env.node.state = "down"
    r2 = client.post(f"{P}/fit", json=body)
    assert r2.status_code == 503 and r2.json()["code"] == "node_unavailable"                 # no numbers from a link that is gone


def test_fit_uses_the_hosts_gpus_and_a_library_copy_when_a_model_is_given(client: TestClient, env: Env) -> None:
    env.seed_local_model()                                       # a library copy: its facts come from the host, not Hugging Face
    env.node.set_gpu_free([90.0, 12.0])
    body = {"engine": "fake", "model": {"format": "hf", "repo_id": "Qwen/Qwen3-32B"}, "params": {"max_model_len": 4096}, "gpu_ids": [1]}
    r = client.post(f"{P}/fit", json=body).json()
    assert r["verdict"] == "wont_fit" and r["per_gpu"][0]["uuid"] == "GPU-uuid-1" and r["per_gpu"][0]["free_gib"] == 12.0
    assert not env.hub.calls                                     # no network: the library had it
    r = client.post(f"{P}/fit", json={**body, "gpu_ids": [0]}).json()
    assert r["verdict"] in ("fits", "tight") and r["per_gpu"][0]["free_gib"] == 90.0


def test_fit_for_a_format_the_engine_cannot_load_is_422(client: TestClient, env: Env) -> None:
    r = client.post(f"{P}/fit", json={"engine": "fake", "model": {"format": "gguf", "root_id": "gguf", "rel": "a/m.gguf"}})
    assert r.status_code == 422 and r.json()["code"] == "format_unsupported"


def test_fit_for_a_model_with_no_readable_facts_says_how_to_fix_it(client: TestClient, env: Env) -> None:
    env.container.adapters.register(FakeMultiAdapter())
    r = client.post(f"{P}/fit", json={"engine": "fakecpp", "model": {"format": "gguf", "root_id": "gguf", "rel": "a/m.gguf"}})
    assert r.status_code == 422 and r.json()["code"] == "model_info_unavailable" and "rescan" in r.json()["detail"]


def test_fit_defaults_the_gpu_choice_per_host_ignoring_default_indices_the_host_does_not_have(client: TestClient, env: Env) -> None:
    seed_hub(env)
    env.container.settings.update(SettingsPatch(default_gpu_ids=[5]))
    r = client.post(f"{P}/fit", json={"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 512}}).json()
    assert [g["gpu_id"] for g in r["per_gpu"]] == [0]            # index 5 does not exist on this host: the best GPU is chosen instead


def test_qwen_config_fixture_is_still_the_dense_gqa_model() -> None:
    assert QWEN_CFG["num_key_value_heads"] == 8
