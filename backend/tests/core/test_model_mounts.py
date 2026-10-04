"""What each model format mounts into the engine's container, and what the launch context the adapter receives says."""
from __future__ import annotations

from typing import Any

import pytest

from engine_console.adapters.base import EngineAdapter, LaunchContext, LaunchSpec
from engine_console.domain.errors import Conflict, Unprocessable
from engine_console.domain.models import InstanceCreate
from engine_console.domain.refs import ModelRef
from engine_console.services.runspec import (
    check_ref,
    display_of,
    engine_labels,
    hf_layout,
    hf_mounts,
    model_mounts,
    split_env,
)

from .conftest import Env
from .fakes import FakeMultiAdapter

HF = "/root/.cache/huggingface"


def gguf(rel: str, root: str = "gguf") -> ModelRef:
    return ModelRef(format="gguf", root_id=root, rel=rel)


async def start(env: Env, ref: ModelRef, *, engine: str = "fakecpp", host: str = "local", **kw: Any) -> Any:
    if FakeMultiAdapter.id not in env.container.adapters:
        env.container.adapters.register(FakeMultiAdapter())
    return await env.container.lifecycle.create(InstanceCreate(engine=engine, model=ref, host_id=host, **kw))


# ---- pure planning ----------------------------------------------------------------------------------------------------------------
def test_hf_layouts() -> None:
    assert hf_layout("hf-home", "models--a--b") == "home" and hf_layout("hf-hub", "hub/x") == "hub"
    assert hf_layout("auto", "hub/models--a--b") == "home" and hf_layout("auto", "models--a--b") == "hub"
    assert hf_layout(None, "anything") == "home"
    assert hf_mounts("hf", "hf-home", "hub/models--a--b", HF, True) == [{"root": "hf", "rel": "", "dest": HF, "ro": True}]
    assert hf_mounts("cache", "hf-hub", "models--a--b", HF + "/", True)[0]["dest"] == HF + "/hub"       # no double slash
    assert hf_mounts("cache", "auto", "models--a--b", HF, False)[0]["ro"] is False


def test_gguf_mounts_the_directory_of_the_file_so_split_sets_are_whole() -> None:
    mounts, path = model_mounts(gguf("unsloth/m/m-Q4_K_M-00001-of-00003.gguf"), copy_rel=None, ollama_root=None)
    assert mounts == [{"root": "gguf", "rel": "unsloth/m", "dest": "/models", "ro": True}]
    assert path == "/models/m-Q4_K_M-00001-of-00003.gguf"
    top, p2 = model_mounts(gguf("model.gguf"), copy_rel=None, ollama_root=None)          # a file at the root's top level
    assert top == [{"root": "gguf", "rel": "", "dest": "/models", "ro": True}] and p2 == "/models/model.gguf"


def test_plain_mounts_the_model_directory_at_models() -> None:
    ref = ModelRef(format="plain", root_id="plain", rel="org/model")
    assert model_mounts(ref, copy_rel=None, ollama_root=None) == ([{"root": "plain", "rel": "org/model", "dest": "/models", "ro": True}],
                                                                 "/models")
    root_is_model = ModelRef(format="plain", root_id="plain", rel="")
    assert model_mounts(root_is_model, copy_rel=None, ollama_root=None)[0][0]["rel"] == ""


@pytest.mark.parametrize(("copy_rel", "root_path", "expected_rel"), [
    ("manifests/registry.ollama.ai/library/llama3/8b", "/data/ollama/models", ""),      # the root itself holds manifests/
    ("models/manifests/registry.ollama.ai/library/llama3/8b", "/data/ollama", "models"),  # an Ollama home
    (None, "/data/ollama/models", ""), (None, "/data/ollama", "models")])                 # a bare server: judged by the root's name
def test_ollama_mounts_the_models_dir_read_only(copy_rel: str | None, root_path: str, expected_rel: str) -> None:
    ref = ModelRef(format="ollama", name="llama3:8b")
    mounts, path = model_mounts(ref, copy_rel=copy_rel, ollama_root=("ollama", root_path))
    assert mounts == [{"root": "ollama", "rel": expected_rel, "dest": "/root/.ollama/models", "ro": True}] and path is None
    assert model_mounts(ref, copy_rel=None, ollama_root=None) == ([], None)               # no Ollama root: nothing to mount


# ---- refs --------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ref", [
    ModelRef(format="hf", repo_id="a--b/c"), ModelRef(format="hf"), ModelRef(format="hf", repo_id="../x/y"),
    ModelRef(format="gguf", root_id="g", rel="/abs/m.gguf"), ModelRef(format="gguf", root_id="g", rel="../m.gguf"),
    ModelRef(format="gguf", root_id="g", rel="a//m.gguf"), ModelRef(format="gguf", root_id="g", rel="m.bin"),
    ModelRef(format="gguf", rel="m.gguf"), ModelRef(format="gguf", root_id="a b", rel="m.gguf"),
    ModelRef(format="gguf", root_id="g"), ModelRef(format="plain", root_id="g", rel="a/../b"),
    ModelRef(format="plain", rel="a"), ModelRef(format="ollama", name="llama3; rm -rf /"),
    ModelRef(format="ollama", name="x" * 200), ModelRef(format="gguf", root_id="g", rel="a\\b.gguf")])
def test_hostile_or_incomplete_refs_are_422_invalid_model(ref: ModelRef) -> None:
    with pytest.raises(Unprocessable) as ei:
        check_ref(ref)
    assert ei.value.code == "invalid_model"


@pytest.mark.parametrize("ref", [
    ModelRef(format="hf", repo_id="Qwen/Qwen3-32B"), gguf("a/b/model-Q4.gguf"), ModelRef(format="plain", root_id="p", rel="org/m"),
    ModelRef(format="plain", root_id="p", rel=""), ModelRef(format="ollama", name="llama3:8b"),
    ModelRef(format="ollama", name="hf.co/org/repo:Q4_K_M"), ModelRef(format="ollama"), ModelRef(format="ollama", name="")])
def test_valid_refs_pass(ref: ModelRef) -> None:
    assert check_ref(ref) is ref


def test_display_names_are_what_chat_sends_as_model() -> None:
    assert display_of(ModelRef(format="hf", repo_id="Qwen/Qwen3-32B")) == "Qwen/Qwen3-32B"
    assert display_of(gguf("unsloth/m/Model-Q4_K_M.gguf")) == "Model-Q4_K_M"
    assert display_of(gguf("m/Model-Q4_K_M-00001-of-00004.gguf")) == "Model-Q4_K_M"          # the shard suffix is not part of the name
    assert display_of(ModelRef(format="plain", root_id="p", rel="org/my-model")) == "my-model"
    assert display_of(ModelRef(format="plain", root_id="p", rel="")) == "p"
    assert display_of(ModelRef(format="ollama", name="llama3:8b")) == "llama3:8b" and display_of(ModelRef(format="ollama")) == "ollama"


def test_secret_looking_variables_travel_as_secret_env_and_offline_is_default() -> None:
    env, secret = split_env({"HF_HOME": HF, "VLLM_API_KEY": "s", "MY_TOKEN": "t", "PASSWORD_FILE": "p", "X": "1"}, offline=True)
    assert env == {"HF_HOME": HF, "X": "1", "HF_HUB_OFFLINE": "1"} and secret == {"VLLM_API_KEY": "s", "MY_TOKEN": "t", "PASSWORD_FILE": "p"}
    env2, _ = split_env({"HF_HUB_OFFLINE": "0"}, offline=True)
    assert env2["HF_HUB_OFFLINE"] == "0"                                                         # an adapter's own choice wins
    assert "HF_HUB_OFFLINE" not in split_env({}, offline=False)[0]


def test_labels_leave_out_what_would_not_fit_instead_of_truncating() -> None:
    ref = gguf("a/" + "d" * 300 + ".gguf")
    labels = engine_labels(instance="inst_1", engine="fakecpp", display="m", ref=ref, port=18000, name="n", gpu_ids=[0, 1],
                           gpu_uuids=["GPU-" + "u" * 40] * 8)
    assert "engine-console.rel" not in labels and labels["engine-console.root"] == "gguf" and labels["engine-console.format"] == "gguf"
    assert "engine-console.gpus" not in labels and labels["engine-console.gpu_ids"] == "0,1"      # 8 UUIDs are over 256 characters
    assert all(len(v) <= 256 for v in labels.values()) and labels["engine-console.port"] == "18000"


# ---- through the lifecycle --------------------------------------------------------------------------------------------------------------
async def test_gguf_launch_mounts_the_directory_and_passes_the_file_as_model_path(env: Env) -> None:
    ref = gguf("unsloth/m/m-Q4.gguf")
    env.library.add("local", ref, root_id="gguf", rel="unsloth/m/m-Q4.gguf", size_bytes=5 * 1024**3)
    inst = await start(env, ref)
    spec = env.node.run_specs[0]
    assert spec["mounts"] == [{"root": "gguf", "rel": "unsloth/m", "dest": "/models", "ro": True}]      # no HF cache for gguf
    assert spec["argv"][:4] == ["--model", "/models/m-Q4.gguf", "--alias", "m-Q4"] and spec["env"]["FAKE_FORMAT"] == "gguf"
    assert inst.model == ref and inst.repo_id == "m-Q4" and inst.name == "m-Q4"
    assert spec["labels"]["engine-console.format"] == "gguf" and spec["labels"]["engine-console.rel"] == "unsloth/m/m-Q4.gguf"
    assert inst.model_location is not None and inst.model_location.path == "/data/gguf/unsloth/m/m-Q4.gguf"
    assert env.node.containers[inst.container_name]["mounts"][0]["source"] == "/data/gguf/unsloth/m"


async def test_gguf_at_the_top_of_a_root_mounts_the_root_itself(env: Env) -> None:
    ref = gguf("top.gguf")
    env.library.add("local", ref, root_id="gguf", rel="top.gguf")
    await start(env, ref)
    assert env.node.run_specs[0]["mounts"] == [{"root": "gguf", "rel": "", "dest": "/models", "ro": True}]


async def test_plain_model_directory_launch(env: Env) -> None:
    ref = ModelRef(format="plain", root_id="gguf", rel="org/my-model")
    env.library.add("local", ref, root_id="gguf", rel="org/my-model")
    await start(env, ref)
    spec = env.node.run_specs[0]
    assert spec["mounts"] == [{"root": "gguf", "rel": "org/my-model", "dest": "/models", "ro": True}]
    assert spec["argv"][:2] == ["--model", "/models"]


@pytest.mark.parametrize(("copy_rel", "root_path", "rel"), [
    ("manifests/registry.ollama.ai/library/llama3/8b", "/data/ollama/models", ""),
    ("models/manifests/registry.ollama.ai/library/llama3/8b", "/data/ollama", "models")])
async def test_ollama_launch_mounts_the_stores_models_dir_read_only(env: Env, copy_rel: str, root_path: str, rel: str) -> None:
    ref = ModelRef(format="ollama", name="llama3:8b")
    env.node.policy["model_roots"][2]["path"] = root_path
    env.library.add("local", ref, root_id="ollama", rel=copy_rel)
    inst = await start(env, ref)
    spec = env.node.run_specs[0]
    assert spec["mounts"] == [{"root": "ollama", "rel": rel, "dest": "/root/.ollama/models", "ro": True}]
    assert spec["argv"][:2] == ["--model", "llama3:8b"] and inst.repo_id == "llama3:8b"


async def test_a_named_ollama_model_must_be_in_the_library_but_a_bare_server_need_not(env: Env) -> None:
    with pytest.raises(Conflict) as ei:
        await start(env, ModelRef(format="ollama", name="llama3:8b"))
    assert {c["code"] for c in ei.value.extra["preflight"]["checks"]} >= {"model_missing"}
    assert not env.node.ops("docker.run")
    inst = await start(env, ModelRef(format="ollama"))
    assert inst.repo_id == "ollama" and env.node.run_specs[0]["mounts"] == [
        {"root": "ollama", "rel": "models", "dest": "/root/.ollama/models", "ro": True}]


async def test_an_hf_root_in_hub_layout_is_mounted_under_the_hub_folder(env: Env) -> None:
    env.node.policy["model_roots"][0].update(kind="hf-hub", path="/data/hub-cache")
    env.library.add_hf("local", "Qwen/Qwen3-32B", {"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 2,
                                                     "num_key_value_heads": 1}, rel="models--Qwen--Qwen3-32B")
    inst = await env.container.lifecycle.create(InstanceCreate(engine="fake", repo_id="Qwen/Qwen3-32B", params={"max_model_len": 512}))
    assert env.node.run_specs[0]["mounts"] == [{"root": "hf", "rel": "", "dest": HF + "/hub", "ro": True}]
    assert env.node.containers[inst.container_name]["mounts"][0]["source"] == "/data/hub-cache"


async def test_the_hf_mount_is_writable_only_when_configured_and_the_root_is_writable(env: Env) -> None:
    env.cfg.hf_cache_readonly = False
    env.seed_local_model()
    await env.container.lifecycle.create(InstanceCreate(engine="fake", repo_id="Qwen/Qwen3-32B", params={"max_model_len": 4096}))
    assert env.node.run_specs[0]["mounts"][0]["ro"] is False
    env.node.policy["model_roots"][0]["writable"] = False
    await env.container.lifecycle.create(InstanceCreate(engine="fake", repo_id="Qwen/Qwen3-32B", name="b", params={"max_model_len": 4096}))
    assert env.node.run_specs[1]["mounts"][0]["ro"] is True


async def test_an_adapter_that_wants_the_hf_cache_next_to_a_gguf_gets_it_when_the_host_has_one(env: Env) -> None:
    class WantsCache(FakeMultiAdapter):
        id = "wantscache"

        def build_launch_ctx(self, ctx: LaunchContext) -> LaunchSpec:
            return super().build_launch_ctx(ctx).model_copy(update={"needs_hf_cache": True})

    env.container.adapters.register(WantsCache())
    ref = gguf("a/m.gguf")
    env.library.add("local", ref, root_id="gguf", rel="a/m.gguf")
    await start(env, ref, engine="wantscache")
    assert [m["dest"] for m in env.node.run_specs[0]["mounts"]] == ["/models", HF]
    env.node.policy["model_roots"] = [r for r in env.node.policy["model_roots"] if r["kind"] != "hf-home"]
    await start(env, ref, engine="wantscache", name="two")
    assert [m["dest"] for m in env.node.run_specs[1]["mounts"]] == ["/models"]                # no HF root on this host: just skip it


async def test_a_model_on_shared_storage_is_loaded_from_the_copy_the_library_finds_for_that_host(env: Env) -> None:
    ref = gguf("a/m.gguf", root="shared")
    env.node.policy["model_roots"].append({"id": "shared", "path": "/mnt/shared", "kind": "gguf", "writable": False, "label": "Shared"})
    env.library.add("local", ref, root_id="shared", rel="a/m.gguf")
    await start(env, ref)
    assert env.node.run_specs[0]["mounts"] == [{"root": "shared", "rel": "a", "dest": "/models", "ro": True}]


async def test_an_adapter_that_cannot_load_the_format_is_refused_at_build_time_too(env: Env) -> None:
    ref = gguf("a/m.gguf")
    env.library.add("local", ref, root_id="gguf", rel="a/m.gguf")
    with pytest.raises(Conflict) as ei:
        await start(env, ref, engine="fake")
    assert {c["code"] for c in ei.value.extra["preflight"]["checks"]} >= {"format_unsupported"}
    assert not env.node.ops("docker.run")


async def test_hostile_model_refs_are_422_before_any_check_or_call(env: Env) -> None:
    env.container.adapters.register(FakeMultiAdapter())
    for ref in (gguf("../../etc/passwd.gguf"), ModelRef(format="plain", root_id="x y", rel="a")):
        with pytest.raises(Unprocessable):
            await env.container.lifecycle.preflight(InstanceCreate(engine="fakecpp", model=ref))
    assert not env.node.calls


def test_adapters_are_asked_through_the_v2_entry_point_with_a_complete_context(env: Env) -> None:
    seen: list[LaunchContext] = []

    class Spy(FakeMultiAdapter):
        id = "spy"

        def build_launch_ctx(self, ctx: LaunchContext) -> LaunchSpec:
            seen.append(ctx)
            return super().build_launch_ctx(ctx)

    assert isinstance(Spy(), EngineAdapter)
    import asyncio
    env.container.adapters.register(Spy())
    ref = gguf("a/m.gguf")
    env.library.add("local", ref, root_id="gguf", rel="a/m.gguf")
    asyncio.run(env.container.lifecycle.create(InstanceCreate(engine="spy", model=ref, params={"max_model_len": 4096})))
    ctx = seen[-1]
    assert (ctx.model_format, ctx.model_path, ctx.served_name, ctx.hf_cache_container_path) == ("gguf", "/models/m.gguf", "m", HF)
    assert ctx.hw.host_id == "local" and ctx.hw.arch == "x86_64" and ctx.params == {"max_model_len": 4096}


async def test_generated_config_files_travel_in_the_run_spec_and_are_validated_against_the_nodes_limits(env: Env) -> None:
    class WithFiles(FakeMultiAdapter):
        id = "withfiles"

        def build_launch_ctx(self, ctx: LaunchContext) -> LaunchSpec:
            return super().build_launch_ctx(ctx).model_copy(update={"files": {"/etc/engine/models.yaml": "models: []\n"}})

    env.container.adapters.register(WithFiles())
    ref = gguf("a/m.gguf")
    env.library.add("local", ref, root_id="gguf", rel="a/m.gguf")
    await start(env, ref, engine="withfiles")
    assert env.node.run_specs[0]["files"] == {"/etc/engine/models.yaml": "models: []\n"}

    class TooMany(FakeMultiAdapter):
        id = "toomany"

        def build_launch_ctx(self, ctx: LaunchContext) -> LaunchSpec:
            return super().build_launch_ctx(ctx).model_copy(update={"files": {f"/etc/f{i}": "x" for i in range(5)}})

    env.container.adapters.register(TooMany())
    with pytest.raises(Conflict) as ei:
        await start(env, ref, engine="toomany", name="two")
    assert {c["code"] for c in ei.value.extra["preflight"]["checks"]} >= {"invalid_launch"}
    assert len(env.node.run_specs) == 1
