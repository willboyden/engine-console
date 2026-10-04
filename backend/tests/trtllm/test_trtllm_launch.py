"""Validation and launch building: argv, the generated engine-options YAML, env, and LaunchSpec fields."""
from __future__ import annotations

import json
import re
from typing import Any

import pytest
import yaml

from engine_console.adapters.base import Hardware, LaunchContext
from engine_console.adapters.trtllm import TrtllmAdapter
from engine_console.adapters.trtllm_params import EXTRA_OPTIONS_PATH, TORCH_LLM_ARGS_FIELDS, YAML_NESTED

from .conftest import read_json
from .samples import SAMPLES, SCENARIOS, sample_params

SCHEMA = read_json("llm_args_1.2.1.json")
HF = "/root/.cache/huggingface"


def ctx(params: dict[str, Any], hw: Hardware, model: str = "example-org/chat-model-36B", **kw: Any) -> LaunchContext:
    return LaunchContext(model=model, served_name="chat", hf_cache_container_path=HF, params=params, hw=hw, **kw)


def yaml_of(spec: Any) -> dict[str, Any]:
    out = yaml.safe_load(spec.files[EXTRA_OPTIONS_PATH])
    assert isinstance(out, dict)
    return out


def _type_problem(value: Any, type_str: str) -> str | None:
    """A light check of one YAML value against the image's annotation string (None = fine or not checkable)."""
    if value is None:
        return None if ("Optional" in type_str or "NoneType" in type_str) else "null for a non-optional field"
    literal = re.search(r"Literal\[(.*?)\]", type_str)
    if literal:
        allowed = re.findall(r"'([^']+)'", literal.group(1))
        return None if value in allowed else f"{value!r} not in {allowed}"
    if "List[int]" in type_str:
        ok = isinstance(value, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
        return None if ok else f"{value!r} is not a list of ints"
    if "List[str]" in type_str:
        return None if isinstance(value, list) and all(isinstance(x, str) for x in value) else "not a list of strings"
    if "bool" in type_str and "int" not in type_str:
        return None if isinstance(value, bool) else f"{value!r} is not a bool"
    if "float" in type_str:
        return None if isinstance(value, (int, float)) and not isinstance(value, bool) else f"{value!r} is not a number"
    if "int" in type_str:
        return None if isinstance(value, int) and not isinstance(value, bool) else f"{value!r} is not an int"
    if "str" in type_str and "Union" not in type_str:
        return None if isinstance(value, str) else f"{value!r} is not a string"
    return None


def schema_problems(tree: dict[str, Any]) -> list[str]:
    """Every key of a generated options file must exist in the image's TorchLlmArgs (nested blocks too), with a type
    that fits. This is the offline twin of the one-off run through the real pydantic models."""
    out: list[str] = []
    top = SCHEMA["classes"]["TorchLlmArgs"]
    for name, value in tree.items():
        if name not in top:
            out.append(f"unknown top-level key {name}")
            continue
        cls = YAML_NESTED.get(name)
        if cls and isinstance(value, dict):
            fields = SCHEMA["classes"][cls]
            for sub, subval in value.items():
                if sub not in fields:
                    out.append(f"unknown key {name}.{sub}")
                elif (problem := _type_problem(subval, fields[sub]["type"])):
                    out.append(f"{name}.{sub}: {problem}")
        elif name == "speculative_config":
            decoding = {"MTP": "MTPDecodingConfig", "Eagle": "EagleDecodingConfig", "NGram": "NGramDecodingConfig",
                        "DraftTarget": "DraftTargetDecodingConfig", "AUTO": "DecodingBaseConfig"}
            fields = SCHEMA["classes"][decoding[value["decoding_type"]]]
            out += [f"unknown key speculative_config.{k}" for k in value if k != "decoding_type" and k not in fields]
        elif (problem := _type_problem(value, top[name]["type"])) and not cls:
            out.append(f"{name}: {problem}")
    return out


# --------------------------------------------------------------- validation
def test_presets_validate(adapter: TrtllmAdapter) -> None:
    pre = adapter.presets()
    assert {"balanced", "max-throughput", "low-latency", "long-context", "tool-agent", "reasoning"} <= set(pre)
    for name, params in pre.items():
        assert adapter.validate(params) == [], name


def test_every_sample_and_scenario_validates(adapter: TrtllmAdapter) -> None:
    for key in SAMPLES:
        assert adapter.validate(sample_params(key)) == [], key
    for name, params in SCENARIOS.items():
        assert adapter.validate(params) == [], name


def test_validate_value_errors(adapter: TrtllmAdapter) -> None:
    assert adapter.validate({"nope": 1}) == ["unknown parameter 'nope'"]
    for bad in ({"free_gpu_memory_fraction": 1.0}, {"free_gpu_memory_fraction": "0.9"}, {"max_seq_len": True},
                {"max_seq_len": 1000.5}, {"kv_cache_dtype": "fp4"}, {"enable_block_reuse": 1},
                {"tokens_per_block": 48}, {"tokens_per_block": 1}, {"cuda_graph_batch_sizes": ["1", "0"]},
                {"cuda_graph_batch_sizes": []}, {"kv_max_attention_window": ["a"]},
                {"nvfp4_gemm_backends": ["magic"]}, {"image": "nvcr.io/nvidia/tensorrt-llm/release:latest"},
                {"image": "nvcr.io/nvidia/tensorrt-llm/release"}, {"revision": "-bad"}, {"tokenizer": "a b"},
                {"chat_template": "-x"}, {"chat_template": "x" * 4001}, {"chat_template": "plainword"},
                {"chat_template": "relative/path.jinja"}, {"media_io_kwargs": "{not json"},
                {"media_io_kwargs": "[1]"}, {"speculative_config": {"decoding_type": "Medusa"}},
                {"speculative_config": {}}, {"orchestrator_type": "ray"}, {"custom_tokenizer": "a.b.C"},
                {"tool_parser": "hermes"}, {"reasoning_parser": "gptoss"}, {"attention_backend": "NOPE"},
                {"tensor_parallel_size": 0}, {"batch_wait_max_tokens_ratio": 1.5}):
        assert adapter.validate(bad), bad


def test_json_parameters_respect_the_nodes_size_limits(adapter: TrtllmAdapter) -> None:
    """ecnode.proto: one argv entry is at most 4096 characters, one generated file at most 128 KiB."""
    big = {"video": {"note": "x" * 3600}}
    assert "at most 3500" in adapter.validate({"media_io_kwargs": big})[0]
    assert adapter.validate({"media_io_kwargs": {"video": {"note": "x" * 3000}}}) == []
    assert "at most 50000" in adapter.validate({"extra_llm_options": {"enable_min_latency": "x" * 60000}})[0]
    assert "at most 50000" in adapter.validate(
        {"speculative_config": {"decoding_type": "NGram", "note": "x" * 60000}})[0]


def test_chat_template_is_a_path_or_a_template(adapter: TrtllmAdapter) -> None:
    """load_chat_template opens the value as a file; only { } or a newline mark it as the template text."""
    assert adapter.validate({"chat_template": "/models/chat_template.jinja"}) == []
    assert adapter.validate({"chat_template": "{{ messages[0].content }}"}) == []
    assert adapter.validate({"chat_template": "line one\nline two"}) == []
    assert "file name" in adapter.validate({"chat_template": "template.jinja"})[0]


def test_validate_accepts_none_and_json_strings(adapter: TrtllmAdapter) -> None:
    assert adapter.validate({"max_seq_len": None, "tool_parser": None}) == []
    assert adapter.validate({"media_io_kwargs": '{"video": {"num_frames": 4}}'}) == []
    assert adapter.validate({"speculative_config": '{"decoding_type": "AUTO"}'}) == []


def test_validate_relations(adapter: TrtllmAdapter) -> None:
    errs = adapter.validate({"enable_cuda_graphs": False, "cuda_graph_padding": True})
    assert any("enable_cuda_graphs" in e for e in errs)
    assert adapter.validate({"cuda_graph_batch_sizes": ["1", "2"], "cuda_graph_max_batch_size": 8})
    assert adapter.validate({"cuda_graph_batch_sizes": ["1", "2"], "cuda_graph_max_batch_size": 0}) == []
    assert adapter.validate({"torch_compile_inductor": True})
    assert adapter.validate({"torch_compile_inductor": True, "enable_torch_compile": True}) == []
    assert adapter.validate({"spec_max_draft_len": 3})                                    # needs spec_decoding_type
    assert adapter.validate({"spec_decoding_type": "Eagle", "spec_max_draft_len": 3})     # needs a draft model
    assert adapter.validate({"spec_decoding_type": "NGram", "spec_max_draft_len": 3})     # needs the match length
    assert adapter.validate({"spec_decoding_type": "MTP", "spec_max_draft_len": 3})       # engine overwrites it
    assert adapter.validate({"spec_decoding_type": "MTP", "spec_ngram_max_matching_size": 3})
    assert adapter.validate({"spec_decoding_type": "AUTO", "spec_max_concurrency": 4})
    assert adapter.validate({"spec_decoding_type": "AUTO"}) == []
    assert adapter.validate({"speculative_config": {"decoding_type": "AUTO"}, "spec_decoding_type": "AUTO"})


def test_validate_moe_parallel_sizes(adapter: TrtllmAdapter) -> None:
    ok = {"tensor_parallel_size": 4, "moe_expert_parallel_size": 2, "moe_tensor_parallel_size": 2}
    assert adapter.validate(ok) == []
    assert adapter.validate({**ok, "moe_tensor_parallel_size": 4})                           # 2 x 4 != 4
    assert adapter.validate({"tensor_parallel_size": 4, "moe_expert_parallel_size": 3})       # 3 does not divide 4
    assert adapter.validate({"tensor_parallel_size": 4, "moe_expert_parallel_size": 4}) == []
    assert adapter.validate({"tensor_parallel_size": 4, "moe_cluster_parallel_size": 2,
                             "moe_expert_parallel_size": 2})                                # cluster > 1 needs ep 1
    assert adapter.validate({"tensor_parallel_size": 4, "moe_cluster_parallel_size": 2}) == []
    assert adapter.validate({"moe_expert_parallel_size": 2})                                 # tp defaults to 1


def test_validate_extra_options(adapter: TrtllmAdapter) -> None:
    assert adapter.validate({"extra_llm_options": {"garbage_collection_gen0_threshold": 5}}) == []
    for bad, why in (({"model": "x"}, "not allowed"), ({"env_overrides": {"A": "1"}}, "not allowed"),
                     ({"kv_cache_free_gpu_memory_fraction": 0.8}, "refuses unknown"),
                     ({"kv_cache_config": {"dtype": "fp8"}}, "own parameter"),
                     ({"max_batch_size": 4}, "own parameter"),
                     ({"enable_iter_perf_stats": True}, "own parameter"),
                     ({"max_stats_len": 50}, "own parameter")):
        errs = adapter.validate({"extra_llm_options": bad})
        assert errs and why in errs[0], (bad, errs)


# --------------------------------------------------------------- golden launches
def test_golden_empty_launch(adapter: TrtllmAdapter, hw: Hardware) -> None:
    spec = adapter.build_launch_ctx(ctx({}, hw))
    assert spec.image == "nvcr.io/nvidia/tensorrt-llm/release:1.2.1"
    assert spec.argv == [
        "trtllm-serve", "serve", "example-org/chat-model-36B", "--backend", "pytorch", "--host", "0.0.0.0",
        "--port", "8000", "--extra_llm_api_options", "/config/extra-llm-api-options.yaml"]
    assert spec.files == {EXTRA_OPTIONS_PATH: (
        "# Generated by Engine Console for `trtllm-serve --extra_llm_api_options`.\n"
        "# Edit the launch parameters, not this file.\n"
        "enable_iter_perf_stats: true\n"
        "max_stats_len: 100\n")}
    assert spec.env == {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
                        "PYTHONUNBUFFERED": "1", "HF_HOME": HF}
    assert (spec.container_port, spec.health_path, spec.metrics_path, spec.models_path) == (
        8000, "/health", "/metrics", "/v1/models")
    assert spec.ipc_host is True and spec.shm_size == "16g" and spec.needs_hf_cache is True


def test_no_file_and_no_flag_when_nothing_needs_it(adapter: TrtllmAdapter, hw: Hardware) -> None:
    spec = adapter.build_launch_ctx(ctx({"enable_iter_perf_stats": False, "max_batch_size": 8}, hw))
    assert spec.files == {} and "--extra_llm_api_options" not in spec.argv
    assert spec.argv[-2:] == ["--max_batch_size", "8"]


def test_golden_tensor_parallel_fp8_launch(adapter: TrtllmAdapter, hw: Hardware) -> None:
    params = {"tensor_parallel_size": 2, "free_gpu_memory_fraction": 0.85, "max_batch_size": 64,
              "max_num_tokens": 8192, "enable_chunked_prefill": True, "kv_cache_dtype": "fp8",
              "enable_block_reuse": True, "tool_parser": "qwen3", "reasoning_parser": "qwen3",
              "cuda_graph_padding": True, "nccl_p2p_disable": True, "trust_remote_code": False}
    spec = adapter.build_launch_ctx(ctx(params, hw))
    assert spec.argv == [
        "trtllm-serve", "serve", "example-org/chat-model-36B", "--backend", "pytorch", "--host", "0.0.0.0",
        "--port", "8000", "--free_gpu_memory_fraction", "0.85", "--tensor_parallel_size", "2",
        "--max_batch_size", "64", "--max_num_tokens", "8192", "--enable_chunked_prefill",
        "--tool_parser", "qwen3", "--reasoning_parser", "qwen3",
        "--extra_llm_api_options", "/config/extra-llm-api-options.yaml"]
    assert "--trust_remote_code" not in spec.argv                       # a false flag is simply absent
    assert yaml_of(spec) == {"enable_iter_perf_stats": True, "max_stats_len": 100,
                             "cuda_graph_config": {"enable_padding": True},
                             "kv_cache_config": {"dtype": "fp8", "enable_block_reuse": True}}
    assert spec.env["NCCL_P2P_DISABLE"] == "1"
    assert spec.files[EXTRA_OPTIONS_PATH].endswith(
        "cuda_graph_config:\n  enable_padding: true\nenable_iter_perf_stats: true\nkv_cache_config:\n"
        "  dtype: fp8\n  enable_block_reuse: true\nmax_stats_len: 100\n")  # keys sorted: stable diffs


def test_plain_model_folder_launch(adapter: TrtllmAdapter, hw1: Hardware) -> None:
    spec = adapter.build_launch_ctx(ctx({}, hw1, model="", model_format="plain", model_path="/models"))
    assert spec.argv[:3] == ["trtllm-serve", "serve", "/models"]
    assert spec.needs_hf_cache is False and "HF_HOME" not in spec.env and spec.env["HF_HUB_OFFLINE"] == "1"
    with pytest.raises(ValueError, match="model reference"):
        adapter.build_launch_ctx(ctx({}, hw1, model="x", model_format="plain", model_path=None))


def test_unsupported_formats_and_hostile_model_refs(adapter: TrtllmAdapter, hw: Hardware) -> None:
    for fmt in ("gguf", "ollama"):
        with pytest.raises(ValueError, match="not"):
            adapter.build_launch_ctx(ctx({}, hw, model_format=fmt))  # type: ignore[arg-type]
    for bad in ("--evil", "-x", "a b", "a\nb", ""):
        with pytest.raises(ValueError, match="model reference"):
            adapter.build_launch_ctx(ctx({}, hw, model=bad))


def test_build_rejects_invalid_params(adapter: TrtllmAdapter, hw: Hardware) -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        adapter.build_launch_ctx(ctx({"bogus": 1}, hw))


def test_build_launch_bridge_uses_the_hf_path(adapter: TrtllmAdapter, hw: Hardware) -> None:
    spec = adapter.build_launch("example-org/m", {"max_seq_len": 4096}, hw, served_name="s",
                                hf_cache_container_path="/hf")
    assert spec.argv[:3] == ["trtllm-serve", "serve", "example-org/m"] and "--max_seq_len" in spec.argv
    assert spec.env["HF_HOME"] == "/hf"


def test_image_override_and_types(adapter: TrtllmAdapter, hw: Hardware) -> None:
    spec = adapter.build_launch_ctx(ctx({"image": "nvcr.io/nvidia/tensorrt-llm/release:1.3.0"}, hw))
    assert spec.image == "nvcr.io/nvidia/tensorrt-llm/release:1.3.0" and "--image" not in " ".join(spec.argv)


def test_values_are_typed_in_cli_and_yaml(adapter: TrtllmAdapter, hw: Hardware) -> None:
    params = {"free_gpu_memory_fraction": 0.5, "kv_cache_max_gib": 12.5, "kv_host_cache_gib": 4,
              "tokens_per_block": 64, "stream_interval": 4, "batch_wait_timeout_ms": 5,
              "batch_wait_max_tokens_ratio": 0.25, "enable_partial_reuse": False, "enable_autotuner": False,
              "cuda_graph_batch_sizes": ["8", "4", "1"], "kv_max_attention_window": [4096, "512"],
              "nvfp4_gemm_backends": ["cutlass"], "media_io_kwargs": {"video": {"num_frames": 8}},
              "otlp_traces_endpoint": "http://otel-collector:4317"}
    spec = adapter.build_launch_ctx(ctx(params, hw))
    i = spec.argv.index("--media_io_kwargs")
    assert spec.argv[i + 1] == '{"video":{"num_frames":8}}'                       # compact JSON, one argv entry
    assert spec.argv[spec.argv.index("--free_gpu_memory_fraction") + 1] == "0.5"
    tree = yaml_of(spec)
    assert tree["kv_cache_config"] == {"max_gpu_total_bytes": 13421772800, "host_cache_size": 4294967296,
                                       "tokens_per_block": 64, "enable_partial_reuse": False,
                                       "max_attention_window": [4096, 512]}
    assert tree["cuda_graph_config"] == {"batch_sizes": [8, 4, 1]}                  # ints, order kept
    assert tree["stream_interval"] == 4 and isinstance(tree["stream_interval"], int)
    assert tree["batch_wait_timeout_ms"] == 5.0 and isinstance(tree["batch_wait_timeout_ms"], float)
    assert tree["enable_autotuner"] is False and tree["nvfp4_gemm_config"] == {"allowed_backends": ["cutlass"]}
    assert schema_problems(tree) == []


def test_cuda_graphs_off_is_an_explicit_null(adapter: TrtllmAdapter, hw: Hardware) -> None:
    spec = adapter.build_launch_ctx(ctx({"enable_cuda_graphs": False}, hw))
    text = spec.files[EXTRA_OPTIONS_PATH]
    assert "cuda_graph_config: null\n" in text and yaml_of(spec)["cuda_graph_config"] is None
    on = adapter.build_launch_ctx(ctx({"enable_cuda_graphs": True}, hw))
    assert "cuda_graph_config" not in on.files[EXTRA_OPTIONS_PATH]               # the engine default is already on


def test_torch_compile_block(adapter: TrtllmAdapter, hw: Hardware) -> None:
    assert yaml_of(adapter.build_launch_ctx(ctx({"enable_torch_compile": True}, hw)))["torch_compile_config"] == {}
    both = yaml_of(adapter.build_launch_ctx(ctx({"enable_torch_compile": True, "torch_compile_inductor": True}, hw)))
    assert both["torch_compile_config"] == {"enable_inductor": True}
    assert "torch_compile_config" not in yaml_of(adapter.build_launch_ctx(ctx({"enable_torch_compile": False}, hw)))


def test_scheduler_block_restates_the_stock_defaults(adapter: TrtllmAdapter, hw: Hardware) -> None:
    only_chunking = yaml_of(adapter.build_launch_ctx(ctx({"context_chunking_policy": "EQUAL_PROGRESS"}, hw)))
    assert only_chunking["scheduler_config"] == {
        "capacity_scheduler_policy": "GUARANTEED_NO_EVICT", "context_chunking_policy": "EQUAL_PROGRESS",
        "dynamic_batch_config": {"enable_batch_size_tuning": True, "enable_max_num_tokens_tuning": False,
                                 "dynamic_batch_moving_average_window": 128}}
    overridden = yaml_of(adapter.build_launch_ctx(ctx({"capacity_scheduler_policy": "MAX_UTILIZATION"}, hw)))
    assert overridden["scheduler_config"]["capacity_scheduler_policy"] == "MAX_UTILIZATION"
    assert "scheduler_config" not in yaml_of(adapter.build_launch_ctx(ctx({}, hw)))


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenarios_build_and_match_the_schema(adapter: TrtllmAdapter, hw: Hardware, name: str) -> None:
    spec = adapter.build_launch_ctx(ctx(SCENARIOS[name], hw))
    assert spec.argv[:2] == ["trtllm-serve", "serve"]
    if spec.files:
        assert schema_problems(yaml_of(spec)) == [], name


@pytest.mark.parametrize("key", sorted(SAMPLES))
def test_each_sample_builds_and_matches_the_schema(adapter: TrtllmAdapter, hw: Hardware, key: str) -> None:
    spec = adapter.build_launch_ctx(ctx(sample_params(key), hw))
    tree = yaml_of(spec) if spec.files else {}
    assert schema_problems(tree) == [], key
    assert set(tree) <= set(TORCH_LLM_ARGS_FIELDS)
    flag = {p.key: p.flag for p in adapter.param_catalog()}[key]
    if flag.startswith("--"):
        assert flag in spec.argv, key
    elif flag.startswith("env:"):
        assert flag[4:] in spec.env, key
    elif flag == "yaml:*":
        assert set(SAMPLES[key]) <= set(tree), key
    elif flag == "@image":
        assert spec.image == SAMPLES[key]
    elif key == "enable_iter_perf_stats":
        assert spec.files == {}                          # false is the engine default: nothing to write
    else:
        assert flag[5:].split(".")[0] in tree, key


def test_speculative_blocks(adapter: TrtllmAdapter, hw: Hardware) -> None:
    tree = yaml_of(adapter.build_launch_ctx(ctx(SCENARIOS["eagle3"], hw)))
    assert tree["speculative_config"] == {"decoding_type": "Eagle", "speculative_model": "example-org/eagle3-head",
                                          "max_draft_len": 3, "eagle3_one_model": True}
    tree = yaml_of(adapter.build_launch_ctx(ctx(SCENARIOS["mtp"], hw)))
    assert tree["speculative_config"] == {"decoding_type": "MTP", "num_nextn_predict_layers": 2,
                                          "max_concurrency": 16}
    tree = yaml_of(adapter.build_launch_ctx(ctx(SCENARIOS["spec_json"], hw)))
    assert tree["speculative_config"] == {"decoding_type": "NGram", "max_draft_len": 2, "max_matching_ngram_size": 2}


def test_extra_options_are_merged_not_overriding(adapter: TrtllmAdapter, hw: Hardware) -> None:
    tree = yaml_of(adapter.build_launch_ctx(ctx(SCENARIOS["extra"], hw)))
    assert tree["garbage_collection_gen0_threshold"] == 10000 and tree["enable_min_latency"] is True
    assert tree["enable_iter_perf_stats"] is True and tree["max_stats_len"] == 100


def test_iteration_statistics_default_and_history(adapter: TrtllmAdapter, hw: Hardware) -> None:
    """The console turns the statistics on and keeps a short history; both are the user's to change."""
    assert yaml_of(adapter.build_launch_ctx(ctx({}, hw))) == {"enable_iter_perf_stats": True, "max_stats_len": 100}
    assert yaml_of(adapter.build_launch_ctx(ctx({"iter_stats_max_entries": 250}, hw))) == {
        "enable_iter_perf_stats": True, "max_stats_len": 250}
    assert yaml_of(adapter.build_launch_ctx(ctx({"enable_iter_perf_stats": True}, hw))) == {
        "enable_iter_perf_stats": True, "max_stats_len": 100}
    off = adapter.build_launch_ctx(ctx({"enable_iter_perf_stats": False}, hw))
    assert off.files == {}                                       # nothing to write: false is the engine default
    kept = adapter.build_launch_ctx(ctx({"enable_iter_perf_stats": False, "iter_stats_max_entries": 50}, hw))
    assert yaml_of(kept) == {"max_stats_len": 50}


def test_history_keeps_one_scrape_below_the_node_cap(adapter: TrtllmAdapter) -> None:
    """Why 100: the node (ecnode.proto.MAX_METRICS_TEXT) drops a scrape body above 512 KiB, and the stock history of
    1000 iteration objects is larger than that."""
    from engine_console.adapters import trtllm as module
    one = len(json.dumps(read_json("iter_stats_1.2.1.json")["with_spec_and_requests"], separators=(",", ":")))
    assert 800 < one < 3000
    assert module._STATS_HISTORY * one < 512 * 1024 // 2
    assert 1000 * one > 512 * 1024 // 2 or one < 1100            # the engine default would sit near or past the cap


def test_no_secret_reaches_argv_env_or_files(adapter: TrtllmAdapter, hw: Hardware) -> None:
    """The engine has no key or token option; nothing secret-shaped may appear anywhere in a launch."""
    for key in SAMPLES:
        spec = adapter.build_launch_ctx(ctx(sample_params(key), hw))
        blob = " ".join(spec.argv) + " " + " ".join(f"{k}={v}" for k, v in spec.env.items()) + " ".join(spec.files.values())
        assert not any(name in spec.env for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "API_KEY", "TRTLLM_API_KEY"))
        assert not re.search(r"(token|secret|password|api[_-]?key)\s*[:=]", blob, re.I), key
    names = {p.key for p in adapter.param_catalog()}
    assert not any(re.search(r"token(?!izer|s)|secret|password|api_?key", n) for n in names - {
        "max_num_tokens", "kv_cache_max_tokens", "batch_wait_max_tokens_ratio", "moe_max_num_tokens"}), \
        "a secret-looking parameter was added: secrets go through env by name, never argv"
