"""Unit tests for the smaller pure/edge pieces: hub parsing, egress allowlist, downloads, store, docker argv."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest
import respx

from engine_console.adapters import AdapterRegistry, load_default_adapters
from engine_console.domain.errors import EgressDenied, GatedModel, NotFound
from engine_console.domain.models import DownloadRequest
from engine_console.domain.ports import HubModel, RepoFile
from engine_console.services.common import EventBus, percentile, redact, scrub
from engine_console.services.docker import ContainerSpec, build_run_args, command_snippets, gpus_arg
from engine_console.services.hardware import HardwareService, parse_smi_csv
from engine_console.services.hf import build_model_info, detect_quantization
from engine_console.services.hf_http import HfHttpClient, host_allowed
from engine_console.services.store import Store, paginate

from .conftest import QWEN_CFG, Env
from .fakes import BrokenProbe, FakeAdapter, FakeProbe


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


# ---- downloads -----------------------------------------------------------------------------------------------------
async def dl(env: Env, repo: str = "org/m", **kw: object) -> str:
    d = await env.container.downloads.create(DownloadRequest(repo_id=repo, **kw))  # type: ignore[arg-type]
    await env.container.downloads.wait(d.id)
    return d.id


def add_repo(env: Env, payload: bytes = b"0123456789" * 5) -> None:
    env.hub.add("org/m", {"config.json": b"{}", "w.safetensors": payload, "notes.txt": b"n"}, {"model_type": "x"})


async def test_download_layout_and_hash_verified(env: Env) -> None:
    add_repo(env)
    did = await dl(env)
    d = env.container.downloads.get(did)
    assert d.state == "completed" and d.total_bytes == 2 + 50 + 1 and d.files_done == 3
    repo = env.cfg.hf_cache_dir / "hub" / "models--org--m"
    link = repo / "snapshots" / "c0ffee00000000000000000000000000000000ff" / "w.safetensors"
    assert link.is_symlink() and link.read_bytes() == b"0123456789" * 5
    assert (repo / "refs" / "main").read_text() == "c0ffee00000000000000000000000000000000ff"
    assert not list((repo / "blobs").glob("*.incomplete"))
    assert env.container.downloads.list_local()[0].repo_id == "org/m"


async def test_download_allow_patterns(env: Env) -> None:
    add_repo(env)
    did = await dl(env, allow_patterns=["*.json"])
    assert env.container.downloads.get(did).files_total == 1


async def test_download_hash_mismatch_fails_and_discards_partial(env: Env) -> None:
    add_repo(env)
    env.hub.corrupt.add("w.safetensors")
    did = await dl(env)
    d = env.container.downloads.get(did)
    assert d.state == "failed" and d.error_code == "hash_mismatch"
    assert not list((env.cfg.hf_cache_dir / "hub" / "models--org--m" / "blobs").glob("*.incomplete"))


async def test_download_resumes_from_partial_and_survives_ignored_range(env: Env) -> None:
    payload = b"abcdefghij" * 10
    add_repo(env, payload)
    blobs = env.cfg.hf_cache_dir / "hub" / "models--org--m" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / (hashlib.sha256(payload).hexdigest() + ".incomplete")).write_bytes(payload[:40])
    did = await dl(env)
    assert env.container.downloads.get(did).state == "completed"
    assert (env.cfg.hf_cache_dir / "hub" / "models--org--m" / "snapshots" / "c0ffee00000000000000000000000000000000ff" / "w.safetensors").read_bytes() == payload
    # server that ignores Range: restart the file, still correct
    env2_hub = env.hub
    env2_hub.ignore_range = True
    env.container.downloads.delete_local("org/m")
    (blobs).mkdir(parents=True)
    (blobs / (hashlib.sha256(payload).hexdigest() + ".incomplete")).write_bytes(payload[:40])
    did2 = await dl(env)
    assert env.container.downloads.get(did2).state == "completed"


async def test_download_gated_gives_clear_error(env: Env) -> None:
    add_repo(env)
    env.hub.gated.add("org/m")
    d = env.container.downloads.get(await dl(env))
    assert d.state == "failed" and d.error_code == "hf_gated" and "gated" in (d.error or "")


async def test_download_disk_precheck(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    add_repo(env)
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(100, 99, 10))  # type: ignore[attr-defined]
    d = env.container.downloads.get(await dl(env))
    assert d.state == "failed" and d.error_code == "insufficient_disk"


async def test_download_pause_resume_cancel(env: Env) -> None:
    add_repo(env, b"z" * 4000)
    gate = asyncio.Event()
    orig = env.hub.open_file

    async def slow(repo: str, rev: str, path: str, start: int):  # type: ignore[no-untyped-def]
        s = await orig(repo, rev, path, start)
        inner = s.chunks

        async def chunks():  # type: ignore[no-untyped-def]
            async for ch in inner():
                await gate.wait()
                yield ch
        s.chunks = chunks  # type: ignore[method-assign]
        return s

    env.hub.open_file = slow  # type: ignore[method-assign]
    svc = env.container.downloads
    d = await svc.create(DownloadRequest(repo_id="org/m"))
    await asyncio.sleep(0.05)
    assert (await svc.pause(d.id)).state == "paused"
    gate.set()
    assert (await svc.resume(d.id)).state in ("queued", "running")
    await svc.wait(d.id)
    assert svc.get(d.id).state == "completed"
    svc.delete_local("org/m")
    gate.clear()
    d2 = await svc.create(DownloadRequest(repo_id="org/m"))
    await asyncio.sleep(0.02)
    assert (await svc.cancel(d2.id)).state == "cancelled"


async def test_recover_parks_running_downloads(env: Env) -> None:
    env.container.store.execute("INSERT INTO downloads(id,repo_id,state,created_at,updated_at) VALUES('d','a/b','running',1,1)")
    env.container.downloads.recover()
    assert env.container.downloads.get("d").state == "paused"


def test_delete_local_refuses_in_use_and_traversal(env: Env) -> None:
    env.seed_local_model()
    env.container.store.execute(
        "INSERT INTO instances(id,name,engine,repo_id,params,gpu_ids,gpu_uuids,container_name,image,state,created_at) "
        "VALUES('i','n','fake','Qwen/Qwen3-32B','{}','[0]','[]','c','img','ready',1)")
    from engine_console.domain.errors import BadRequest, Conflict
    with pytest.raises(Conflict):
        env.container.downloads.delete_local("Qwen/Qwen3-32B")
    with pytest.raises(BadRequest):
        env.container.downloads.delete_local("../../etc")


# ---- store ------------------------------------------------------------------------------------------------------------
def test_migrations_idempotent_and_audit_append_only(tmp_path: Path) -> None:
    s = Store(tmp_path / "x.db")
    assert s.migrate() == []
    assert s.one("SELECT name FROM schema_migrations")["name"] == "0001_init.sql"
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


# ---- docker ---------------------------------------------------------------------------------------------------------------
def spec(**kw: object) -> ContainerSpec:
    base = dict(name="ec-x", image="img:1", argv=["--a", "1"], host_port=18001, container_port=8000,
                gpu_uuids=["GPU-1"], network="ai-lab", mounts=[("/fast/models/hf", "/root/.cache/huggingface", False)],
                env={"A": "b"}, env_passthrough=["HF_TOKEN"], labels={"ai-lab.console.instance": "i1"})
    return ContainerSpec(**{**base, **kw})  # type: ignore[arg-type]


def test_gpus_arg_quotes_multiple_devices() -> None:
    assert gpus_arg(["GPU-a"]) == "device=GPU-a"
    assert gpus_arg(["GPU-a", "GPU-b"]) == '"device=GPU-a,GPU-b"'


def test_build_run_args_exact_shape() -> None:
    a = build_run_args(spec())
    assert a[:2] == ["run", "-d"] and a[-3:] == ["img:1", "--a", "1"]
    assert "127.0.0.1:18001:8000" in a and "no-new-privileges:true" in a
    assert a[a.index("--ipc") + 1] == "host" and a[a.index("--shm-size") + 1] == "16g"
    assert "ai-lab.console=1" in a and "-e" in a and a[a.index("HF_TOKEN") - 1] == "-e"
    assert not any("shell" in x for x in a)


def test_command_snippets_have_no_secret_values() -> None:
    out = command_snippets(spec(env_passthrough=["VLLM_API_KEY", "HF_TOKEN"]))
    blob = json.dumps(out)
    assert "-e VLLM_API_KEY" in out["docker_run"] and "${VLLM_API_KEY}" in out["compose_yaml"]
    assert "external: true" in out["compose_yaml"] and "GPU-1" in out["compose_yaml"] and "hf_" not in blob


# ---- hardware / registry / helpers ------------------------------------------------------------------------------------------
def test_smi_csv_parsing_and_probe_fallback() -> None:
    g = parse_smi_csv("0, GPU-abc, NVIDIA RTX PRO 6000, 97887, 90000, 12, 41, 88.5, [N/A], 12.0\n")
    assert g[0].uuid == "GPU-abc" and g[0].total_gib == pytest.approx(97887 / 1024) and g[0].fan_pct is None
    rep = HardwareService([BrokenProbe(), FakeProbe(1)]).report()
    assert rep.source == "fake" and len(rep.gpus) == 1
    none = HardwareService([BrokenProbe()]).report()
    assert none.gpus == [] and none.source == "none"
    hw = HardwareService([FakeProbe(2)]).hardware([1])
    assert hw.gpu_ids == [1] and hw.compute_capability == "12.0"


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


# ---- lifecycle extras: adoption, startup timeout, container vanishing ---------------------------------------------------
async def test_adopt_reattaches_labelled_containers(env: Env) -> None:
    env.runner.containers["ec-old-abc123"] = {"running": True, "exit": 0, "labels": {
        "ai-lab.console": "1", "ai-lab.console.instance": "inst_old", "ai-lab.console.engine": "fake",
        "ai-lab.console.model": "Qwen/Qwen3-32B", "ai-lab.console.port": "18007", "ai-lab.console.name": "old",
        "ai-lab.console.gpus": "GPU-uuid-1"}}
    env.runner.containers["ec-stranger"] = {"running": True, "exit": 0, "labels": {"ai-lab.console": "1"}}   # no instance label
    assert await env.container.lifecycle.adopt() == 1
    inst = env.container.lifecycle.get("inst_old")
    assert (inst.state, inst.port, inst.gpu_ids, inst.name) == ("loading", 18007, [1], "old")
    assert await env.container.lifecycle.adopt() == 0                       # idempotent
    env.runner.containers["ec-old-abc123"]["running"] = False
    await env.container.lifecycle.adopt()
    assert env.container.lifecycle.get("inst_old").state == "stopped"


async def test_startup_timeout_and_disappeared_container(env: Env) -> None:
    from engine_console.domain.models import InstanceCreate
    env.seed_local_model()
    life = env.container.lifecycle
    inst = await life.create(InstanceCreate(engine="fake", repo_id="Qwen/Qwen3-32B", params={"max_model_len": 4096}))
    env.container.store.execute("UPDATE instances SET started_at=1 WHERE id=?", (inst.id,))
    await life.tick()
    got = life.get(inst.id)
    assert got.state == "failed" and got.error == "startup timed out"
    inst2 = await life.create(InstanceCreate(engine="fake", repo_id="Qwen/Qwen3-32B", name="two", params={"max_model_len": 4096}))
    env.runner.containers.pop(inst2.container_name)
    await life.tick()
    assert life.get(inst2.id).error == "container disappeared"
    assert life.uses_repo("Qwen/Qwen3-32B") is False
