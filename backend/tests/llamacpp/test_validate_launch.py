"""validate() and the launch (golden argv/env) of the llama.cpp adapter."""
from __future__ import annotations

from typing import Any

import pytest

from engine_console.adapters.base import MODEL_MOUNT, Hardware, LaunchContext, LaunchSpec
from engine_console.adapters.llamacpp import LlamaCppAdapter
from engine_console.adapters.llamacpp_params import CHAT_TEMPLATE_PATH, DEFAULT_IMAGE

MODEL = f"{MODEL_MOUNT}/demo-Q4_K_M.gguf"


def launch(adapter: LlamaCppAdapter, hw: Hardware, params: dict[str, Any] | None = None, *, path: str = MODEL,
           served: str = "demo", fmt: str = "gguf") -> LaunchSpec:
    ctx = LaunchContext(model="example-org/chat-model-GGUF:Q4_K_M", model_format=fmt,  # type: ignore[arg-type]
                        model_path=path, served_name=served, hf_cache_container_path="/hf",
                        params=params or {}, hw=hw)
    return adapter.build_launch_ctx(ctx)


# ------------------------------------------------------------------ validate
def test_presets_validate(adapter: LlamaCppAdapter) -> None:
    pre = adapter.presets()
    assert {"balanced", "max-throughput", "low-latency", "long-context", "tool-agent", "reasoning",
            "big-model-offload", "moe-experts-on-cpu"} == set(pre)
    for name, p in pre.items():
        assert adapter.validate(p) == [], name


def test_validate_unknown_and_none(adapter: LlamaCppAdapter) -> None:
    assert adapter.validate({"nope": 1}) == ["unknown parameter 'nope'"]
    assert adapter.validate({"ctx_size": None, "flash_attn": None}) == []
    assert adapter.validate({}) == []


@pytest.mark.parametrize("params", [
    {"ctx_size": -1}, {"ctx_size": 1.5}, {"ctx_size": True}, {"ctx_size": "8192"}, {"ctx_size": float("nan")},
    {"parallel": -2}, {"fit_target": -1}, {"top_p": 1.5}, {"min_p": -0.1}, {"log_verbosity": 6}, {"temperature": -1},
    {"flash_attn": "maybe"}, {"cache_type_v": "fp7"}, {"split_mode": "diagonal"}, {"load_mode": "mlock+dio"},
    {"jinja": 1}, {"kv_offload": "yes"}, {"cpu_moe": "true"},
    {"n_gpu_layers": -1}, {"n_gpu_layers": "most"}, {"n_gpu_layers": 1.5}, {"n_gpu_layers": True},
    {"spec_draft_ngl": "some"}, {"spec_type": "magic"},
    {"tensor_split": "3;1"}, {"tensor_split": "a,b"}, {"device": "CUDA0;rm"}, {"cors_origins": "a b"},
    {"api_key": "has space"}, {"api_key": ""}, {"api_key": "x" * 600},
    {"chat_template_kwargs": "{not json"}, {"chat_template_kwargs": "[1]"},
    {"served_model_name": []}, {"served_model_name": ["a,b"]}, {"served_model_name": ["-x"]}, {"served_model_name": "a"},
    {"override_tensor": "exps"}, {"override_kv": "tokenizer.x=bool"}, {"n_cpu_moe": -1},
    {"mmproj": "/etc/passwd"}, {"mmproj": "/models/../etc/passwd"}, {"mmproj": "mmproj.gguf"}, {"mmproj": "/models/readme.txt"},
    {"lora": "/models/a.gguf,/tmp/b.gguf"}, {"lora_scaled": "/models/a.gguf"}, {"lora_scaled": "/models/a.gguf:x"},
    {"spec_draft_model": "-m"}, {"spec_draft_model": "/models/a\x00.gguf"},
    {"chat_template": "x" * 4001}, {"chat_template_jinja": "x" * 130_000}, {"chat_template_jinja": "a\x00b"},
    {"override_tensor": "-x=CPU"}, {"cache_reuse": 1.5},
])
def test_validate_rejects(adapter: LlamaCppAdapter, params: dict[str, Any]) -> None:
    assert adapter.validate(params), params


def test_validate_rejects_both_template_forms(adapter: LlamaCppAdapter) -> None:
    errs = adapter.validate({"chat_template": "chatml", "chat_template_jinja": "{{ x }}"})
    assert errs == ["chat_template and chat_template_jinja are alternatives: set only one"]


def test_validate_accepts_valid_values(adapter: LlamaCppAdapter) -> None:
    ok = {
        "n_gpu_layers": "all", "spec_draft_ngl": 12, "ctx_size": 0, "parallel": -1, "tensor_split": "0.7,0.3",
        "device": "CUDA0,CUDA1", "cors_origins": "https://chat.example.org,localhost", "api_key": "k-1.2_3",
        "chat_template_kwargs": '{"enable_thinking": false}', "served_model_name": ["gpt-4o", "org/model:q4"],
        "override_tensor": r"\.ffn_.*_exps\.=CPU,blk\.1\.=CUDA0", "override_kv": "tokenizer.ggml.add_bos_token=bool:false",
        "mmproj": f"{MODEL_MOUNT}/mmproj.gguf", "lora": f"{MODEL_MOUNT}/a.gguf,{MODEL_MOUNT}/b.gguf",
        "lora_scaled": f"{MODEL_MOUNT}/a.gguf:0.5,{MODEL_MOUNT}/b.gguf:-1", "spec_draft_model": f"{MODEL_MOUNT}/snapshots/abc/d.gguf",
        "chat_template_jinja": "{% for m in messages %}{{ m.content }}{% endfor %}\n", "temperature": 0, "top_p": 1.0,
        "fit_target": 0, "log_verbosity": 5, "reasoning_budget": -1, "cache_ram": -1, "sleep_idle_seconds": 600,
        "cache_type_k": "iq4_nl", "numa": "isolate", "pooling": "rank",
    }
    assert adapter.validate(ok) == []
    assert adapter.validate({"chat_template": "chatml"}) == []


# ------------------------------------------------------------------ launch: golden argv
def test_golden_balanced(adapter: LlamaCppAdapter, hw: Hardware) -> None:
    spec = launch(adapter, hw, adapter.presets()["balanced"])
    assert spec.image == DEFAULT_IMAGE
    assert spec.argv == [
        "-m", MODEL, "--host", "0.0.0.0", "--port", "8080", "--alias", "demo", "--metrics", "--offline",
        "--ctx-size", "16384", "--parallel", "1", "--flash-attn", "on", "--cors-origins", "localhost"]
    assert spec.env == {} and spec.files == {}
    assert (spec.container_port, spec.health_path, spec.metrics_path, spec.models_path) == (8080, "/health", "/metrics", "/v1/models")
    assert spec.needs_hf_cache is False and spec.ipc_host is False and spec.shm_size == "1g"


def test_golden_complex(adapter: LlamaCppAdapter, hw: Hardware) -> None:
    params = {
        "n_gpu_layers": "all", "kv_offload": False, "cache_type_k": "q8_0", "cache_type_v": "q8_0", "cpu_moe": True,
        "split_mode": "layer", "tensor_split": "3,1", "main_gpu": 0, "ctx_size": 32768, "rope_scaling": "yarn",
        "rope_scale": 4, "context_shift": True, "parallel": 4, "kv_unified": False, "cont_batching": True,
        "flash_attn": "on", "threads": 16, "batch_size": 4096, "jinja": True, "chat_template": "chatml",
        "chat_template_kwargs": {"enable_thinking": False}, "reasoning": "off", "reasoning_format": "deepseek",
        "spec_type": "ngram-mod", "served_model_name": ["alias-a", "demo"], "api_key": "secret-value",
        "cors_origins": "https://chat.example.org", "ui": False, "temperature": 0.6, "top_k": 20, "top_p": 0.95,
        "min_p": 0.0, "seed": 42,
    }
    spec = launch(adapter, hw, params)
    assert spec.argv == [
        "-m", MODEL, "--host", "0.0.0.0", "--port", "8080", "--alias", "demo,alias-a", "--metrics", "--offline",
        "--n-gpu-layers", "all", "--no-kv-offload", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--cpu-moe",
        "--split-mode", "layer", "--tensor-split", "3,1", "--main-gpu", "0",
        "--ctx-size", "32768", "--rope-scaling", "yarn", "--rope-scale", "4", "--context-shift",
        "--parallel", "4", "--no-kv-unified", "--cont-batching",
        "--flash-attn", "on", "--threads", "16", "--batch-size", "4096",
        "--jinja", "--chat-template", "chatml", "--chat-template-kwargs", '{"enable_thinking":false}',
        "--reasoning", "off", "--reasoning-format", "deepseek", "--spec-type", "ngram-mod",
        "--cors-origins", "https://chat.example.org", "--no-ui",
        "--temp", "0.6", "--top-k", "20", "--top-p", "0.95", "--min-p", "0", "--seed", "42"]
    assert spec.env == {"LLAMA_API_KEY": "secret-value"}


def test_every_preset_builds(adapter: LlamaCppAdapter, hw: Hardware) -> None:
    for name, params in adapter.presets().items():
        argv = launch(adapter, hw, params).argv
        assert argv[:2] == ["-m", MODEL] and "--metrics" in argv and "--offline" in argv, name


def test_metrics_offline_and_loopback_hardening_are_always_present(adapter: LlamaCppAdapter, hw: Hardware) -> None:
    argv = launch(adapter, hw, {}).argv
    assert argv.count("--metrics") == 1 and argv.count("--offline") == 1
    assert argv[argv.index("--host") + 1] == "0.0.0.0" and argv[argv.index("--port") + 1] == "8080"
    assert argv[argv.index("--cors-origins") + 1] == "localhost"          # measured: the engine's default reflects any origin
    explicit = launch(adapter, hw, {"cors_origins": "*"}).argv
    assert explicit.count("--cors-origins") == 1 and explicit[explicit.index("--cors-origins") + 1] == "*"


@pytest.mark.parametrize("path", [
    f"{MODEL_MOUNT}/a.gguf", f"{MODEL_MOUNT}/qwen/Qwen3-30B-A3B-Q4_K_M-00001-of-00003.gguf",
    f"{MODEL_MOUNT}/snapshots/0123456789abcdef/sub dir/model.Q8_0.gguf"])
def test_model_paths_pass_through(adapter: LlamaCppAdapter, hw: Hardware, path: str) -> None:
    """Loose files, the first shard of a split set, and files inside an HF-cache repo (`/models/snapshots/<rev>/...`)."""
    argv = launch(adapter, hw, {}, path=path).argv
    assert argv[:2] == ["-m", path]


@pytest.mark.parametrize("path", [None, "", "/models", "/models/", "/etc/passwd", "models/a.gguf", f"{MODEL_MOUNT}/../etc/x.gguf",
                                  "-m", f"{MODEL_MOUNT}/a\n.gguf", "/models2/a.gguf"])
def test_bad_model_paths_are_refused(adapter: LlamaCppAdapter, hw: Hardware, path: str | None) -> None:
    ctx = LaunchContext(model="m", model_format="gguf", model_path=path, served_name="s", hf_cache_container_path="/hf", params={}, hw=hw)
    with pytest.raises(ValueError, match="model_path"):
        adapter.build_launch_ctx(ctx)


@pytest.mark.parametrize("fmt", ["hf", "plain", "ollama"])
def test_only_gguf_is_loadable(adapter: LlamaCppAdapter, hw: Hardware, fmt: str) -> None:
    with pytest.raises(ValueError, match="GGUF files only"):
        launch(adapter, hw, {}, fmt=fmt)


def test_invalid_params_raise(adapter: LlamaCppAdapter, hw: Hardware) -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        launch(adapter, hw, {"bogus": 1})
    with pytest.raises(ValueError, match="ctx_size"):
        launch(adapter, hw, {"ctx_size": -5})


@pytest.mark.parametrize("served", ["", "  ", "a,b", "-x", "a\nb"])
def test_bad_served_names_are_refused(adapter: LlamaCppAdapter, hw: Hardware, served: str) -> None:
    with pytest.raises(ValueError, match="served name"):
        launch(adapter, hw, {}, served=served)


def test_aliases_put_the_console_name_first_and_drop_duplicates(adapter: LlamaCppAdapter, hw: Hardware) -> None:
    argv = launch(adapter, hw, {"served_model_name": ["extra", "demo", "extra2"]}).argv
    assert argv[argv.index("--alias") + 1] == "demo,extra,extra2"


def test_api_key_travels_in_the_environment_never_in_argv(adapter: LlamaCppAdapter, hw: Hardware) -> None:
    spec = launch(adapter, hw, {"api_key": "s3cr3t-token-value"})
    assert spec.env == {"LLAMA_API_KEY": "s3cr3t-token-value"}
    assert "s3cr3t-token-value" not in " ".join(spec.argv)
    assert "--api-key" not in spec.argv


def test_bools_follow_the_negation_table(adapter: LlamaCppAdapter, hw: Hardware) -> None:
    on = launch(adapter, hw, {"jinja": True, "warmup": True, "swa_full": True, "embedding": True}).argv
    assert {"--jinja", "--warmup", "--swa-full", "--embedding"} <= set(on)
    off = launch(adapter, hw, {"jinja": False, "warmup": False, "swa_full": False, "embedding": False,
                               "cpu_moe": False, "kv_unified": False}).argv
    assert {"--no-jinja", "--no-warmup", "--no-kv-unified"} <= set(off)
    assert not {"--swa-full", "--embedding", "--cpu-moe", "--no-swa-full", "--no-embedding", "--no-cpu-moe"} & set(off)


def test_floats_render_compactly(adapter: LlamaCppAdapter, hw: Hardware) -> None:
    argv = launch(adapter, hw, {"temperature": 0.6, "rope_scale": 2.5, "min_p": 0.05, "slot_prompt_similarity": 0.1}).argv
    for flag, value in (("--temp", "0.6"), ("--rope-scale", "2.5"), ("--min-p", "0.05"), ("--slot-prompt-similarity", "0.1")):
        assert argv[argv.index(flag) + 1] == value


# ------------------------------------------------------------------ chat templates
def test_custom_template_goes_through_a_generated_file(adapter: LlamaCppAdapter, hw: Hardware) -> None:
    text = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
    spec = launch(adapter, hw, {"chat_template_jinja": text})
    assert spec.files == {CHAT_TEMPLATE_PATH: text}
    argv = spec.argv
    assert argv[argv.index("--chat-template-file") + 1] == CHAT_TEMPLATE_PATH
    assert text not in argv
    assert argv.index("--jinja") < argv.index("--chat-template-file")          # the help: --jinja must come first


def test_inline_custom_template_gets_jinja_first_but_a_builtin_name_does_not(adapter: LlamaCppAdapter, hw: Hardware) -> None:
    custom = launch(adapter, hw, {"chat_template": "{{ messages }}"}).argv
    assert custom.index("--jinja") < custom.index("--chat-template")
    builtin = launch(adapter, hw, {"chat_template": "chatml"}).argv
    assert "--jinja" not in builtin
    explicit_off = launch(adapter, hw, {"chat_template_jinja": "{{ x }}", "jinja": False}).argv
    assert "--no-jinja" in explicit_off and "--jinja" not in explicit_off


# ------------------------------------------------------------------ v1 bridge
def test_v1_build_launch_bridges_to_the_context_path(adapter: LlamaCppAdapter, hw: Hardware) -> None:
    spec = adapter.build_launch(MODEL, {"ctx_size": 4096}, hw, served_name="demo", hf_cache_container_path="/hf")
    assert spec.argv[:2] == ["-m", MODEL] and "--ctx-size" in spec.argv
    with pytest.raises(ValueError, match="model_path"):
        adapter.build_launch("example-org/model", {}, hw, served_name="demo", hf_cache_container_path="/hf")
