"""Unit tests for the smaller pure/edge pieces: hub parsing, egress allowlist, store, registry, helpers.

The v1 tests of docker argv building, nvidia-smi parsing and the local-filesystem downloader are gone with the code they
tested: docker and GPUs are the node's business now (tests/core/fakefleet.py, test_lifecycle_fleet.py) and downloads go
through nodes (the download service's own tests)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
import respx

from engine_console.adapters import AdapterRegistry, load_default_adapters
from engine_console.domain.errors import EgressDenied, GatedModel, NotFound
from engine_console.domain.ports import HubModel, RepoFile
from engine_console.services.common import EventBus, percentile, redact, scrub
from engine_console.services.hf import build_model_info, detect_quantization
from engine_console.services.hf_http import HfHttpClient, host_allowed
from engine_console.services.store import Store, paginate

from .conftest import QWEN_CFG
from .fakes import FakeAdapter


# ---- hub parsing ---------------------------------------------------------------------------------
def hub(cfg: dict[str, object] | None, files: list[RepoFile] | None = None, **kw: object) -> HubModel:
    return HubModel(repo_id="a/b", sha="s", gated=False, license=None, pipeline_tag=None, library_name=None, tags=[],
                    files=files or [], config=cfg, **kw)  # type: ignore[arg-type]


def test_build_model_info_dense_qwen() -> None:
    m = build_model_info(hub(QWEN_CFG, [RepoFile("a.safetensors", 10), RepoFile("b.safetensors", 5), RepoFile("x.md", 999)],
                             safetensors_total=32_000_000_000))
    assert (m.num_layers, m.num_kv_heads, m.head_dim, m.hidden_size) == (64, 8, 128, 5120)
    assert m.weight_bytes == 15 and m.num_params == 32_000_000_000 and not m.is_moe and m.dtype == "bfloat16"


def test_build_model_info_moe_mla_nested_text_config_and_missing() -> None:
    moe = build_model_info(hub({"num_experts": 128, "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4}))
    assert moe.is_moe and moe.head_dim == 16 and moe.num_kv_heads is None    # never invents kv heads
    mla = build_model_info(hub({"kv_lora_rank": 512, "hidden_size": 64, "num_attention_heads": 4, "n_routed_experts": 8}))
    assert mla.kv_lora_rank == 512 and mla.head_dim is None and mla.is_moe
    nested = build_model_info(hub({"architectures": ["VL"], "text_config": {"num_hidden_layers": 7, "hidden_size": 8,
                                                                            "num_attention_heads": 2, "num_key_value_heads": 1}}))
    assert nested.num_layers == 7 and nested.num_kv_heads == 1 and nested.architectures == ["VL"]
    empty = build_model_info(hub(None))
    assert empty.num_layers is None and empty.raw_config == {}


def test_weight_bytes_fall_back_to_safetensors_metadata() -> None:
    m = build_model_info(hub({}, safetensors_params={"BF16": 10, "F8_E4M3": 4}))
    assert m.weight_bytes == 24


@pytest.mark.parametrize(("cfg", "expected"), [
    ({"quantization_config": {"quant_method": "fp8"}}, "fp8"),
    ({"quantization_config": {"quant_method": "mxfp4"}}, "mxfp4"),
    ({"quantization_config": {"quant_method": "compressed-tensors",
                              "config_groups": {"g": {"weights": {"num_bits": 4, "type": "float"}}}}}, "nvfp4"),
    ({"quantization_config": {"quant_method": "compressed-tensors",
                              "config_groups": {"g": {"weights": {"num_bits": 8, "type": "float"}}}}}, "fp8"),
    ({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}, "nvfp4"),
    ({"quantization_config": {"quant_method": "awq"}}, "awq"),
    ({"quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True}}, "bnb4"),
    ({}, None),
])
def test_detect_quantization(cfg: dict[str, object], expected: str | None) -> None:
    assert detect_quantization(cfg) == expected


def test_detect_quantization_from_name_and_tags() -> None:
    assert detect_quantization({}, "nvidia/Llama-NVFP4") == "nvfp4"
    assert detect_quantization({}, "x/y", ["gptq"]) == "gptq"


# ---- egress allowlist (ADR 8) ------------------------------------------------------------------------------
@pytest.mark.parametrize(("host", "ok"), [("huggingface.co", True), ("cdn-lfs-us-1.huggingface.co", True),
                                          ("cas-bridge.xethub.hf.co", True), ("x.hf.co", True),
                                          ("evil.com", False), ("huggingface.co.evil.com", False), ("nothf.co", False)])
def test_host_allowed(host: str, ok: bool) -> None:
    assert host_allowed(host, ["huggingface.co", "cdn-lfs*.huggingface.co", "cas-bridge.xethub.hf.co", "*.hf.co"]) is ok


@respx.mock
async def test_http_client_blocks_redirect_off_allowlist_and_never_puts_token_in_url() -> None:
    c = HfHttpClient("https://huggingface.co", ["huggingface.co"], lambda: "hf_secrettokenvalue")
    respx.get("https://huggingface.co/x/y/resolve/main/f").mock(
        return_value=httpx.Response(302, headers={"location": "https://evil.example/f"}))
    with pytest.raises(EgressDenied):
        await c.open_file("x/y", "main", "f", 0)
    route = respx.get("https://huggingface.co/api/models").mock(return_value=httpx.Response(200, json=[]))
    await c.search(q="q", task=None, library=None, quant=None, sort="downloads", limit=3)
    req = route.calls.last.request
    assert "secrettoken" not in str(req.url) and req.headers["authorization"] == "Bearer hf_secrettokenvalue"
    await c.aclose()


@respx.mock
async def test_http_client_maps_statuses_and_parses_model() -> None:
    c = HfHttpClient("https://huggingface.co", ["huggingface.co"], lambda: None)
    respx.get("https://huggingface.co/api/models/a/b/revision/main").mock(return_value=httpx.Response(200, json={
        "id": "a/b", "sha": "abc", "gated": "manual", "tags": ["license:mit"], "safetensors": {"total": 5, "parameters": {"BF16": 5}},
        "siblings": [{"rfilename": "m.safetensors", "size": 10, "lfs": {"sha256": "ff", "size": 10}}]}))
    respx.get("https://huggingface.co/a/b/resolve/abc/config.json").mock(return_value=httpx.Response(401))
    m = await c.model("a", "main") if False else await c.model("a/b", None)
    assert m.gated and m.license == "mit" and m.config is None and m.files[0].sha256 == "ff" and m.sha == "abc"
    respx.get("https://huggingface.co/api/models/no/pe/revision/main").mock(return_value=httpx.Response(404))
    with pytest.raises(NotFound):
        await c.model("no/pe", None)
    respx.get("https://huggingface.co/api/models/gt/ed/revision/main").mock(return_value=httpx.Response(403))
    with pytest.raises(GatedModel) as ei:
        await c.model("gt/ed", None)
    assert "HF_TOKEN" in ei.value.detail
    await c.aclose()


# ---- store ------------------------------------------------------------------------------------------------------------
def test_migrations_idempotent_and_audit_append_only(tmp_path: Path) -> None:
    s = Store(tmp_path / "x.db")
    assert s.migrate() == []
    names = [r["name"] for r in s.all("SELECT name FROM schema_migrations ORDER BY name")]
    assert names[0] == "0001_init.sql" and "0003_fleet.sql" in names
    assert oct((tmp_path / "x.db").stat().st_mode & 0o777) == "0o600"
    s.execute("INSERT INTO audit(ts,actor,role,method,path,status,params) VALUES(1,'a','admin','POST','/x',200,'{}')")
    import sqlite3
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        s.execute("DELETE FROM audit")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        s.execute("UPDATE audit SET actor='b'")


def test_paginate_helper() -> None:
    assert paginate(10, None) == (10, 0) and paginate(10, "20") == (10, 20)
    assert paginate(10, "junk") == (10, 0) and paginate(9999, "-5") == (500, 0)


# ---- registry / helpers ---------------------------------------------------------------------------------------------------------
def test_registry_tolerates_missing_and_broken_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "broken_adapter.py").write_text("raise RuntimeError('boom')\n")
    (tmp_path / "needs_missing.py").write_text("import not_a_real_module_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    reg = load_default_adapters(["engine_console.adapters.absent", "broken_adapter", "needs_missing"])
    assert reg.list() == []
    reg2 = AdapterRegistry([FakeAdapter()])
    assert "fake" in reg2 and reg2.get("fake").display_name == "Fake Engine"
    with pytest.raises(NotFound):
        reg2.get("nope")


def test_default_registry_never_raises() -> None:
    reg = load_default_adapters()
    assert all(a.id for a in reg.list())


def test_redact_and_scrub() -> None:
    r = redact({"api_key": "x", "nested": {"HF_TOKEN": "y", "note": "use hf_abcdefgh12345 now"}, "big": "a" * 900})
    assert r["api_key"] == "[redacted]" and r["nested"]["HF_TOKEN"] == "[redacted]"
    assert "hf_abcdefgh" not in r["nested"]["note"] and "more chars" in r["big"]
    assert scrub("Authorization: Bearer abcdefghijkl") == "Authorization: [redacted]"


def test_percentile() -> None:
    assert percentile([], 50) is None and percentile([1.0], 95) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


async def test_event_bus_drops_for_slow_subscribers_without_blocking() -> None:
    bus = EventBus()
    agen = bus.subscribe("t")
    task = asyncio.ensure_future(agen.__anext__())
    await asyncio.sleep(0)
    for i in range(1000):
        bus.publish("t", i)     # must never raise or block, even past queue capacity
    assert await task == 0
    await agen.aclose()


def test_the_fleet_migration_adds_host_model_and_image_columns_with_local_as_the_default(tmp_path: Path) -> None:
    s = Store(tmp_path / "x.db")
    cols = {r["name"]: r for r in s.all("PRAGMA table_info(instances)")}
    assert {"host_id", "model", "image_req", "container_port"} <= set(cols)
    assert cols["host_id"]["notnull"] == 1 and cols["host_id"]["dflt_value"] == "'local'" and cols["model"]["notnull"] == 0
    # a pre-fleet row (no host_id given) lands on the console's own machine and reads back as a Hugging Face model
    s.execute("INSERT INTO instances(id,name,engine,repo_id,params,gpu_ids,gpu_uuids,container_name,image,state,created_at) "
              "VALUES('i','n','fake','Qwen/Qwen3-32B','{}','[0]','[]','c','img','ready',1)")
    row = s.one("SELECT host_id, model FROM instances WHERE id='i'")
    assert row is not None and row["host_id"] == "local" and row["model"] is None


def test_an_upgrade_from_the_v1_schema_keeps_existing_instances(tmp_path: Path) -> None:
    """Apply only the v1 migrations, add a row, then let the Store apply the rest: nothing is lost, the row gains host 'local'."""
    import sqlite3
    from importlib import resources
    db = sqlite3.connect(tmp_path / "old.db")
    db.execute("CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, applied_at REAL)")
    for f in sorted(resources.files("engine_console.migrations").iterdir(), key=lambda x: x.name):
        if f.name in ("0001_init.sql", "0002_container_port.sql"):
            db.executescript(f.read_text())
            db.execute("INSERT INTO schema_migrations VALUES (?, 1)", (f.name,))
    db.execute("INSERT INTO instances(id,name,engine,repo_id,params,gpu_ids,gpu_uuids,container_name,image,state,created_at) "
               "VALUES('old','n','vllm','a/b','{}','[0]','[]','ec-x','img:1','ready',1)")
    db.commit()
    db.close()
    s = Store(tmp_path / "old.db")
    row = s.one("SELECT host_id, model, repo_id FROM instances WHERE id='old'")
    assert row is not None and (row["host_id"], row["model"], row["repo_id"]) == ("local", None, "a/b")
