"""Catalog sanity against fixtures captured from the pinned image (fixtures/README.md says how they were made)."""
from __future__ import annotations

import re
from typing import Any

from engine_console.adapters.trtllm import TrtllmAdapter
from engine_console.adapters.trtllm_params import (
    DEFAULT_IMAGE,
    DOCS_BASE,
    EXCLUDED_SERVE_FLAGS,
    IMAGE_DIGEST,
    SOURCE_COMMIT,
    SPEC_TYPES,
    SUPPORTED_ARCHITECTURES,
    TORCH_LLM_ARGS_FIELDS,
    YAML_EXTRA_FLAG,
    YAML_NESTED,
    YAML_PREFIX,
)

from .conftest import read, read_json
from .samples import COMPANIONS, SAMPLES, SCENARIOS

SCHEMA = read_json("llm_args_1.2.1.json")
FLAG_RE = re.compile(r"--[a-z0-9_]+|yaml:[a-z0-9_]+(?:\.[a-z0-9_]+)*|yaml:\*|env:[A-Z0-9_]+|@image")


def _literal_values(type_str: str) -> list[str] | None:
    m = re.search(r"Literal\[(.*?)\]", type_str)
    return re.findall(r"'([^']+)'", m.group(1)) if m else None


def _class_for_path(path: list[str]) -> tuple[dict[str, Any], str]:
    """The fixture field table that holds the last path element, and that element's name."""
    fields = SCHEMA["classes"]["TorchLlmArgs"]
    for part in path[:-1]:
        cls = YAML_NESTED.get(part)
        assert cls is not None, f"{part} is not a nested config class in YAML_NESTED"
        fields = SCHEMA["classes"][cls]
    return fields, path[-1]


def _long_options_in_help(text: str) -> set[str]:
    opts: set[str] = set()
    for line in text.splitlines():
        if line.startswith("  --") or line.startswith("  -"):
            opts.update(re.findall(r"--[a-z0-9_]+", line.split("  ", 2)[1] if line.startswith("  ") else line))
    return opts


def test_catalog_size_and_uniqueness(adapter: TrtllmAdapter) -> None:
    cat = adapter.param_catalog()
    assert len(cat) >= 60
    assert len({p.key for p in cat}) == len(cat)
    assert len({p.flag for p in cat}) == len(cat)


def test_every_param_is_well_formed(adapter: TrtllmAdapter) -> None:
    for p in adapter.param_catalog():
        assert re.fullmatch(r"[a-z][a-z0-9_]*", p.key), p.key
        assert FLAG_RE.fullmatch(p.flag), p.flag
        assert p.help.strip() and p.label.strip()
        assert p.docs_url and p.docs_url.startswith(f"https://github.com/NVIDIA/TensorRT-LLM/blob/{SOURCE_COMMIT}/"), p.key
        if p.type == "enum":
            assert p.choices, p.key
            if p.default is not None:
                assert str(p.default) in p.choices, p.key
        else:
            assert p.choices is None, p.key
        if p.min is not None and p.max is not None:
            assert p.min <= p.max
        if p.type in ("int", "float") and p.default is not None and p.min is not None:
            assert p.default >= p.min, p.key
        if p.type == "bool" and p.flag.startswith(("yaml:", "env:")):
            assert p.bool_style == "value", p.key       # explicit true/false is emitted
        assert len(p.help) <= 400, p.key


def test_required_params_present(adapter: TrtllmAdapter) -> None:
    keys = {p.key for p in adapter.param_catalog()}
    needed = {"free_gpu_memory_fraction", "kv_cache_dtype", "enable_block_reuse", "max_seq_len", "max_batch_size",
              "max_num_tokens", "tensor_parallel_size", "pipeline_parallel_size", "moe_expert_parallel_size",
              "enable_chunked_prefill", "enable_cuda_graphs", "attention_backend", "tool_parser",
              "reasoning_parser", "chat_template", "spec_decoding_type", "speculative_config",
              "guided_decoding_backend", "enable_iter_perf_stats", "trust_remote_code", "revision", "image",
              "extra_llm_options", "moe_backend", "log_level"}
    assert needed <= keys


def test_all_groups_used(adapter: TrtllmAdapter) -> None:
    groups = {p.group for p in adapter.param_catalog()}
    assert groups == {"memory", "parallelism", "context", "quantization", "scheduling", "tools_reasoning",
                      "performance", "speculative", "network", "advanced"}


def test_memory_params_flagged(adapter: TrtllmAdapter) -> None:
    by = {p.key: p for p in adapter.param_catalog()}
    for k in ("free_gpu_memory_fraction", "kv_cache_dtype", "max_seq_len", "max_num_tokens", "max_batch_size",
              "tensor_parallel_size", "pipeline_parallel_size", "kv_cache_max_gib", "kv_host_cache_gib",
              "enable_cuda_graphs", "spec_decoding_type"):
        assert by[k].affects_memory, k


def test_samples_cover_catalog(adapter: TrtllmAdapter) -> None:
    keys = {p.key for p in adapter.param_catalog()}
    assert set(SAMPLES) == keys, f"missing samples {sorted(keys - set(SAMPLES))}, stale {sorted(set(SAMPLES) - keys)}"
    assert set(COMPANIONS) <= keys
    for name, params in SCENARIOS.items():
        assert set(params) <= keys, name


def test_default_image_pinned(adapter: TrtllmAdapter) -> None:
    assert adapter.default_image == DEFAULT_IMAGE == "nvcr.io/nvidia/tensorrt-llm/release:1.2.1"
    assert not adapter.default_image.endswith(":latest")
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", IMAGE_DIGEST)
    assert re.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT)
    assert SCHEMA["tensorrt_llm_version"] == "1.2.1"
    assert DOCS_BASE.startswith("https://github.com/NVIDIA/TensorRT-LLM/blob/")


def test_adapter_identity(adapter: TrtllmAdapter) -> None:
    assert adapter.id == "trtllm" and adapter.default_port == 8000
    assert adapter.model_formats == frozenset({"hf", "plain"})
    assert adapter.multi_model is False and adapter.images_by_arch == {}
    assert adapter.metric_prefixes == ("trtllm_", "[")


def test_catalog_returns_copies(adapter: TrtllmAdapter) -> None:
    adapter.param_catalog()[0].label = "mutated"
    assert adapter.param_catalog()[0].label != "mutated"


# --------------------------------------------------------------- against the image's click options
def test_cli_flags_exist_in_the_serve_command(adapter: TrtllmAdapter) -> None:
    options = {o for p in SCHEMA["serve_params"] if p["kind"] == "Option" for o in p["opts"]}
    for p in adapter.param_catalog():
        if p.flag.startswith("--"):
            assert p.flag in options, f"{p.key}: {p.flag} is not an option of `trtllm-serve serve`"


def test_help_text_fixture_agrees_with_the_click_dump() -> None:
    help_text = read("serve_help_1.2.1.txt")
    dumped = {o for p in SCHEMA["serve_params"] if p["kind"] == "Option" for o in p["opts"] if o.startswith("--")}
    in_help = set(re.findall(r"--[a-z0-9_]+", help_text))
    assert dumped <= in_help | {"--help"}, sorted(dumped - in_help)
    assert "Usage: trtllm-serve serve [OPTIONS] MODEL" in help_text
    assert {"disaggregated", "mm_embedding_serve", "serve"} <= set(re.findall(r"^  (\w+)\s{2,}", read("top_help_1.2.1.txt"), re.M))


def test_every_serve_option_is_exposed_or_excluded(adapter: TrtllmAdapter) -> None:
    exposed = {p.flag for p in adapter.param_catalog() if p.flag.startswith("--")}
    for p in SCHEMA["serve_params"]:
        if p["kind"] != "Option" or "--help" in p["opts"]:
            continue
        names = {o for o in p["opts"] if o.startswith("--")}
        assert names & (exposed | set(EXCLUDED_SERVE_FLAGS)), f"{sorted(names)} is neither exposed nor excluded"
    options = {o for p in SCHEMA["serve_params"] for o in p["opts"]}
    assert set(EXCLUDED_SERVE_FLAGS) <= options, "an excluded flag no longer exists in the image"
    # the flag serve() drops must not be what the catalog uses; a flag only the TensorRT backend reads is not offered
    assert "--moe_cluster_parallel_size" in EXCLUDED_SERVE_FLAGS
    assert "--fail_fast_on_attention_window_too_large" in EXCLUDED_SERVE_FLAGS
    assert "fail_fast_on_attention_window_too_large" not in {p.key for p in adapter.param_catalog()}
    by = {p.key: p for p in adapter.param_catalog()}
    assert by["moe_cluster_parallel_size"].flag == "yaml:moe_cluster_parallel_size"


def test_cli_defaults_and_choices_match_the_image(adapter: TrtllmAdapter) -> None:
    click = {p["name"]: p for p in SCHEMA["serve_params"]}
    by = {p.key: p for p in adapter.param_catalog()}
    for key, name in (("free_gpu_memory_fraction", "free_gpu_memory_fraction"), ("max_batch_size", "max_batch_size"),
                      ("max_num_tokens", "max_num_tokens"), ("max_beam_width", "max_beam_width"),
                      ("tensor_parallel_size", "tensor_parallel_size"), ("pipeline_parallel_size", "pipeline_parallel_size"),
                      ("context_parallel_size", "context_parallel_size"), ("log_level", "log_level"),
                      ("num_postprocess_workers", "num_postprocess_workers")):
        assert by[key].default == click[name]["default"], key
    assert SCHEMA["build_config_defaults"]["max_batch_size"] == by["max_batch_size"].default
    assert SCHEMA["build_config_defaults"]["max_num_tokens"] == by["max_num_tokens"].default
    for key in ("log_level", "reasoning_parser", "tool_parser"):
        assert by[key].choices == click[key]["choices"], key
    assert SCHEMA["reasoning_parsers"] == sorted(by["reasoning_parser"].choices or [])
    assert SCHEMA["tool_parsers"] == sorted(by["tool_parser"].choices or [])
    for key in ("trust_remote_code", "enable_chunked_prefill"):
        assert click[key]["is_flag"] is True and by[key].type == "bool", key


# --------------------------------------------------------------- against the image's YAML schema
def test_top_level_fields_match_the_image() -> None:
    assert list(TORCH_LLM_ARGS_FIELDS) == sorted(SCHEMA["classes"]["TorchLlmArgs"])
    assert len(TORCH_LLM_ARGS_FIELDS) == 83


def test_architecture_registry_matches_the_image() -> None:
    assert sorted(SUPPORTED_ARCHITECTURES) == SCHEMA["architectures"]
    assert {"LlamaForCausalLM", "Qwen3ForCausalLM", "GptOssForCausalLM"} <= SUPPORTED_ARCHITECTURES


def test_yaml_paths_resolve_in_the_image_schema(adapter: TrtllmAdapter) -> None:
    decoding = {f for c in ("MTPDecodingConfig", "EagleDecodingConfig", "NGramDecodingConfig",
                            "DraftTargetDecodingConfig", "DecodingBaseConfig") for f in SCHEMA["classes"][c]}
    decoding.add("decoding_type")                 # the dispatch key of DecodingBaseConfig.from_dict
    for p in adapter.param_catalog():
        if not p.flag.startswith(YAML_PREFIX) or p.flag == YAML_EXTRA_FLAG:
            continue
        path = p.flag[len(YAML_PREFIX):].split(".")
        assert path[0] in TORCH_LLM_ARGS_FIELDS, f"{p.key}: top-level '{path[0]}' is not a TorchLlmArgs field"
        if path[0] == "speculative_config" and len(path) == 2:
            assert path[1] in decoding, f"{p.key}: '{path[1]}' is not a speculative config field"
            continue
        assert len(path) <= 2, p.key
        fields, leaf = _class_for_path(path)
        assert leaf in fields, f"{p.key}: '{leaf}' is not a field of the config class for {path[:-1]}"


def test_yaml_defaults_and_choices_match_the_image(adapter: TrtllmAdapter) -> None:
    special = {"enable_cuda_graphs", "enable_torch_compile", "speculative_config", "extra_llm_options",
               "kv_cache_max_gib", "kv_host_cache_gib"}
    for p in adapter.param_catalog():
        if not p.flag.startswith(YAML_PREFIX) or p.flag == YAML_EXTRA_FLAG or p.key in special:
            continue
        path = p.flag[len(YAML_PREFIX):].split(".")
        if path[0] == "speculative_config":
            continue
        fields, leaf = _class_for_path(path)
        entry = fields[leaf]
        if p.key == "enable_iter_perf_stats":
            assert entry["default"] is False and p.default is False   # the engine's default; the console's is applied at build
            continue
        if p.default is not None:
            assert entry["default"] == p.default, f"{p.key}: image default {entry['default']!r}, catalog {p.default!r}"
        literal = _literal_values(entry["type"])
        if literal and p.type == "enum":
            assert set(p.choices or []) <= set(literal), f"{p.key}: choices outside the schema Literal {literal}"
    # enums that are not Literal types in the schema
    by = {p.key: p for p in adapter.param_catalog()}
    assert by["capacity_scheduler_policy"].choices and set(by["capacity_scheduler_policy"].choices) == set(
        SCHEMA["enums"]["CapacitySchedulerPolicy"])
    assert set(by["context_chunking_policy"].choices or []) == set(SCHEMA["enums"]["ContextChunkingPolicy"])
    assert set(by["sampler_type"].choices or []) == set(SCHEMA["enums"]["SamplerType"])
    assert set(by["moe_backend"].choices or []) == set(_literal_values(SCHEMA["classes"]["MoeConfig"]["backend"]["type"]) or [])
    assert set(by["allreduce_strategy"].choices or []) == set(
        _literal_values(SCHEMA["classes"]["TorchLlmArgs"]["allreduce_strategy"]["type"]) or [])
    assert set(by["guided_decoding_backend"].choices or []) == set(
        _literal_values(SCHEMA["classes"]["TorchLlmArgs"]["guided_decoding_backend"]["type"]) or [])
    assert set(by["spec_decoding_type"].choices or []) == set(SPEC_TYPES)


def test_cuda_graph_and_scheduler_defaults_match_the_image() -> None:
    graphs = SCHEMA["classes"]["CudaGraphConfig"]
    assert graphs["max_batch_size"]["default"] == 0 and graphs["enable_padding"]["default"] is False
    sched = SCHEMA["classes"]["SchedulerConfig"]
    assert sched["capacity_scheduler_policy"]["default"] == "GUARANTEED_NO_EVICT"
    # the stock CLI restates this dynamic batch config (serve.get_llm_args); the adapter copies it
    from engine_console.adapters import trtllm as module
    assert module._SCHEDULER_DEFAULTS["capacity_scheduler_policy"] == "GUARANTEED_NO_EVICT"
    assert set(module._SCHEDULER_DEFAULTS["dynamic_batch_config"]) == set(SCHEMA["classes"]["DynamicBatchConfig"])
