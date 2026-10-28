"""Golden vectors for the Rust port of the vLLM and SGLang adapters, produced by the Python reference.

  "$(node backend/parity/lib/python-release.mjs)/backend/.venv/bin/python" -I backend/parity/gen/ws_a1_adapter_goldens.py
(the Python reference: gen/pyref.py)

Writes backend/crates/ec-hub/tests/data/ws_a1/{vllm,sglang,py}.json. Every vector is an input plus what the Python code
returned (or the exception it raised); the Rust tests replay all of them. Inputs are seeded pseudo-random draws over the
catalog, the model and hardware space, Prometheus text and log lines (property-test style, with Python as the oracle),
plus every committed fixture. Hermetic: no network, no docker, nothing outside this repo is read.

Encoding: a float that JSON cannot carry (NaN, +-inf) is written as {"__float__": "nan"|"inf"|"-inf"}; FloatMap outputs
(dict[str, float]) are lists of [key, repr(value)] so the Rust side compares them bit for bit.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyref import DATA, tree  # noqa: E402

from engine_console.adapters.base import Hardware, LaunchContext, ModelInfo  # noqa: E402
from engine_console.adapters.sglang import SglangAdapter
from engine_console.adapters.sglang_params import build_catalog as sg_catalog
from engine_console.adapters.vllm import VllmAdapter
from engine_console.adapters.vllm_params import build_catalog as vl_catalog

OUT = DATA / "ws_a1"
FIX = tree() / "backend" / "tests"
REPO = DATA.parents[4]
SEED = 20261027


def enc(v: Any) -> Any:
    """JSON-safe copy: non-finite floats become {"__float__": ...}."""
    if isinstance(v, float) and not math.isfinite(v):
        return {"__float__": "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")}
    if isinstance(v, dict):
        return {k: enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [enc(x) for x in v]
    return v


def floatmap(d: dict[str, float]) -> list[list[str]]:
    return [[k, repr(float(v))] for k, v in d.items()]


def has_nonfinite(v: Any) -> bool:
    """A value JSON (as serde_json reads it) cannot carry: NaN, +-inf, an int beyond the float range."""
    if isinstance(v, float):
        return not math.isfinite(v)
    if isinstance(v, int) and not isinstance(v, bool):
        return abs(v) >= 2 ** 1024
    if isinstance(v, dict):
        return any(has_nonfinite(x) for x in v.values())
    if isinstance(v, list):
        return any(has_nonfinite(x) for x in v)
    return False


def valid(adapter, k: str, v: Any) -> bool:
    """`adapter.validate({k: v}) == []`, False where validate raises."""
    try:
        return not adapter.validate({k: v})
    except Exception:  # noqa: BLE001
        return False


def call(fn, *a, **kw) -> tuple[str, Any]:
    try:
        return "ok", fn(*a, **kw)
    except Exception as e:  # noqa: BLE001 - the exception type is the golden value
        return "raise", e


# ── random inputs ─────────────────────────────────────────────────────────────────────────────────────────
R = random.Random(SEED)
WEIRD_STR = ["", " ", "-x", "--evil", "a\x00b", "a\nb", " ", "\x1c", "ok", "1", "-1", "main", "/", "/models",
             "ünïcödé", "x" * 40, "a b", "Ünï", "퟿"]
JSON_TEXT = ['{"a": 1}', "[1, 2]", "{bad", "[1,]", '{"a":1,}', "5", '"s"', "NaN", '{"a": NaN, "b": Infinity}',
             '{"method": "mtp"}', '{"model": "org/draft"}', '{"num_speculative_tokens": 2}', '{"a": 1e400}',
             '{"a": -0, "b": -0.0, "c": 1.5e-7, "d": 12345678901234567890123}', '﻿{}', '"\\ud800"',
             '{"a": "\\ud83d\\ude00\\u00e9\\u007f"}', '{"a":1,"a":2,"b":3}', "[" * 30 + "]" * 30, "", " ", "{",
             '{"a" 1}', '"\\x"', '"\\u12"', '"a', '"\x01"', "1 2", "[1 2]", "tru", "-", "01", "1.", "1e",
             '{"text_config": {"rope_parameters": {"factor": 4.0}}}', '{"k": [true, false, null]}']


def rand_number() -> Any:
    c = R.random()
    if c < 0.35:
        return R.choice([0, 1, 2, 3, 4, 8, 16, 32, 64, 65, 100, 128, 256, 4096, 65536, 262144, -1, -5, 7, 36, 10**12,
                         2**53 + 1, 10**19])
    if c < 0.7:
        return R.choice([0.0, 0.05, 0.1, 0.5, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99, 1.0, 1.5, 2.0, 4.0, 0.15, 1e-5,
                         1234567.0, 2.5, -0.0, 1e20, 12.25, 0.125, 100.5, 8192.0])
    return round(R.uniform(-10, 300000), R.choice([0, 1, 3, 6]))


def rand_value(spec) -> Any:
    t = spec.type
    c = R.random()
    if c < 0.08:
        return None
    if c < 0.16:
        return R.choice([True, False, "x", [1], {"a": 1}, 3, 2.5, "", [], {}])
    if t == "bool":
        return R.choice([True, False, True, False, 1, 0, "true"])
    if t in ("int", "float"):
        if R.random() < 0.3 and (spec.min is not None or spec.max is not None):
            lo = spec.min if spec.min is not None else 0
            hi = spec.max if spec.max is not None else lo + 1000
            v = R.uniform(lo - 1, hi + 1)
            return int(v) if t == "int" and R.random() < 0.7 else v
        return rand_number()
    if t == "enum":
        if R.random() < 0.75 and spec.choices:
            ch = R.choice(spec.choices)
            return int(ch) if ch.isdigit() and R.random() < 0.3 else ch
        return R.choice(["bogus", "AUTO", 16, "fp7", "gguf", "quark", "marlin", "cutlass_mla"])
    if t == "string":
        return R.choice(WEIRD_STR + ["org/model", "localhost:4317", "/tmp/x", "auto", "vllm", "abc"])
    if t == "string_list":
        k = R.random()
        if k < 0.1:
            return []
        if k < 0.2:
            return "not-a-list"
        items = []
        for _ in range(R.randint(1, 4)):
            items.append(R.choice(["a", "b=/x", "--x", "-1", " ", "", "a.example", "sliding_window", 1, 2, 0, -3,
                                   4, 2.5, True, "٣", "²", "007", "x=/y", "8"]))
        return items
    if t == "json":
        k = R.random()
        if k < 0.5:
            return R.choice(JSON_TEXT)
        if k < 0.8:
            return R.choice([{"method": "mtp", "num_speculative_tokens": 2}, {"a": 1}, [1, 2], {"k": "ü "},
                             {"model": "x"}, {"num_speculative_tokens": 3}, {}, [], {"a": {"b": [1.5, None, True]}}])
        return R.choice([5, 2.5, True])
    return None


def rand_params(catalog, retired: list[str], extra_keys: list[str], focus: dict[str, list[Any]]) -> dict[str, Any]:
    p: dict[str, Any] = {}
    for _ in range(R.randint(0, 5)):
        spec = R.choice(catalog)
        p[spec.key] = rand_value(spec)
    for _ in range(R.randint(0, 4)):
        k = R.choice(list(focus))
        p[k] = R.choice(focus[k])
    if R.random() < 0.08:
        p[R.choice(retired + ["nope", "bogus"])] = R.choice(["x", 1, None])
    if R.random() < 0.05:
        p[R.choice(extra_keys)] = 1
    return p


REPOS = ["Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3-8B", "Qwen/Qwen3-32B", "Qwen/Qwen3.8-27B", "Qwen/Qwen3-Coder-30B",
         "Qwen/Qwen3-Next-80B-A3B", "Qwen/Qwen2.5-7B-Instruct", "Qwen/QwQ-32B", "openai/gpt-oss-120b",
         "openai/gpt-oss-20b", "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4", "nvidia/x-30B-A3B-NVFP4",
         "meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-4-Scout", "mistralai/Devstral-Small",
         "deepseek-ai/DeepSeek-V3.2", "deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-V4",
         "zai-org/GLM-4.7-Flash", "zai-org/GLM-4.5-Air", "zai-org/glm-5", "moonshotai/Kimi-K2-Thinking",
         "moonshotai/kımı-k3", "MiniMaxAI/MiniMax-M2", "poolside/Laguna-S-2.1-NVFP4", "ibm/granite-4.0",
         "google/gemma-4-31B", "allenai/Olmo-3-7B", "NousResearch/Hermes-4.3-36B", "acme/moe-mxfp4",
         "unsloth/Qwen3.8-27B-NVFP4", "example-org/chat-model-36B", "a/b", "x/İnstruct-2507", "qwen/qwen3_5-mtp",
         "QWEN/QWEN3", "qwen/qwen3.", "qwen/qwen3_x", "qwen/qwen3٣", "QwEn/QwEn3-ſ", "org/Qwen3-Coder-Instruct-2507"]
ARCHS = [[], ["LlamaForCausalLM"], ["NemotronHForCausalLM"], ["Qwen3NextForCausalLM"], ["GptOssForCausalLM"],
         ["Qwen3MoeForCausalLM"], ["JambaForCausalLM"], ["Qwen3_5ForCausalLM", "X"]]


def rand_model() -> ModelInfo:
    return ModelInfo(
        repo_id=R.choice(REPOS), architectures=R.choice(ARCHS),
        model_type=R.choice([None, "llama", "qwen3_next", "nemotron_h", "gpt_oss", "glm4", "deepseek_v3", "mamba"]),
        num_params=R.choice([None, 36_000_000_000]), is_moe=R.random() < 0.4,
        weight_bytes=R.choice([None, 0, 40 * 1024**3, 72 * 1024**3, int(90 * 1024**3), 5 * 1024**3, 10**15]),
        quantization=R.choice([None, "", "nvfp4", "fp8", "mxfp4", "awq", "NONE", "fp4", "mxfp8", "fp8_block",
                               "gptq", "modelopt_fp4", "NVFP4"]),
        num_attention_heads=R.choice([None, 0, 64, 36, 32, 40, 28]), num_kv_heads=R.choice([None, 0, 8, 4, 3, 2, 1]),
        max_position_embeddings=R.choice([None, 0, 32768, 131072, 262144, 4096]),
        kv_lora_rank=R.choice([None, 0, 512]), gated=R.random() < 0.2,
        pipeline_tag=R.choice([None, "text-generation", "image-text-to-text", "any-to-any", "feature-extraction"]))


def rand_hw() -> Hardware:
    n = R.choice([0, 1, 1, 2, 2, 4, 8])
    totals = [R.choice([95.6, 95.0, 48.0, 24.0, 80.0, 95.6, 95.0, 0.0 if R.random() < 0.2 else 95.6])
              for _ in range(n)]
    free = [round(min(t, R.uniform(0, t + 1)), 2) if R.random() < 0.6 else t - 1.6 for t in totals]
    return Hardware(gpu_ids=list(range(n)), gpu_uuids=[f"GPU-{i}" for i in range(n)], gpu_names=["GPU"] * n,
                    gpu_total_gib=totals, gpu_free_gib=free,
                    compute_capability=R.choice(["12.0", "12.0", "9.0", "10.0", "8.9", " 12.0 ", "unknown", "١٢.٠",
                                                 "12", "12.1", "120"]),
                    host_ram_gib=246.0, arch=R.choice(["x86_64", "aarch64"]))


# ── engine vectors ────────────────────────────────────────────────────────────────────────────────────────────
def dump_spec(s) -> dict[str, Any]:
    return enc(s.model_dump())


def engine_vectors(name: str, make, catalog_fn, focus, n_validate: int, n_compat: int) -> dict[str, Any]:
    a = make()
    cat = catalog_fn()
    retired = ["image", "nvfp4_gemm_backend", "flashinfer_moe_backend"] if name == "vllm" else ["api_key"]
    out: dict[str, Any] = {
        "identity": {"id": a.id, "display_name": a.display_name, "default_image": a.default_image,
                     "default_port": a.default_port, "model_formats": sorted(a.model_formats),
                     "images_by_arch": dict(a.images_by_arch), "multi_model": a.multi_model,
                     "metric_prefixes": list(getattr(a, "metric_prefixes", ())),
                     "preset_info": {k: v.model_dump() for k, v in a.preset_info().items()},
                     "image_for": {arch: a.image_for(arch) for arch in ("x86_64", "aarch64", "riscv")}},
        "catalog": [s.model_dump() for s in a.param_catalog()],
        "presets": [[k, v] for k, v in a.presets().items()],
    }
    extra = [s.key for s in cat]
    # validate
    vv = []
    for _ in range(n_validate):
        p = rand_params(cat, retired, extra, focus)
        st, res = call(a.validate, p)
        vv.append({"params": p, "errors": res} if st == "ok" else {"params": p, "raises": type(res).__name__})
    for pre in a.presets().values():
        vv.append({"params": pre, "errors": a.validate(pre)})
    out["validate"] = vv
    # launch
    lv = []
    for i in range(n_validate // 2):
        p = rand_params(cat, retired, extra, focus) if R.random() < 0.6 else dict(R.choice(list(a.presets().values())))
        if R.random() < 0.5:   # mostly valid params, so argv rendering is exercised
            p = {k: v for k, v in p.items() if valid(a, k, v)}
        hw = rand_hw()
        fmt = R.choice(["hf", "hf", "plain", "gguf", "ollama"])
        ctx = LaunchContext(model=R.choice(["org/m", "Qwen/Qwen3.6-35B-A3B", "--evil", "", "my-dir", "ü/ñ"]),
                            model_format=fmt, model_path=R.choice([None, "/models", "", "/models/x.gguf", "-x"]),
                            served_name=R.choice(["chat", "s", "qwen", "ünï", "a b"]),
                            hf_cache_container_path=R.choice(["/root/.cache/huggingface", "/hf"]), params=p, hw=hw)
        st, res = call(a.build_launch_ctx, ctx)
        row: dict[str, Any] = {"ctx": enc(ctx.model_dump())}
        if st == "ok":
            row["spec"] = res.model_dump()
        elif isinstance(res, ValueError):
            row["error"] = str(res)
        else:
            row["raises"] = type(res).__name__
        lv.append(row)
    out["launch"] = lv
    # compatibility, memory_model, host_memory_gib
    cv, mv, hv = [], [], []
    scenarios = SCENARIOS[name]
    for j in range(n_compat):
        if j < 4 * len(scenarios):          # every hand-made scenario, against four random models/cards
            p = dict(scenarios[j % len(scenarios)])
        else:
            p = rand_params(cat, retired, extra, focus)
            if R.random() < 0.7:
                p = {k: v for k, v in p.items() if valid(a, k, v)}
        m, hw = rand_model(), rand_hw()
        st, res = call(a.compatibility, m, p, hw)
        row = {"model": enc(m.model_dump(exclude_defaults=True)), "params": p,
               "hw": enc(hw.model_dump(exclude_defaults=True))}
        cv.append({**row, "out": [c.model_dump() for c in res]} if st == "ok" else {**row, "raises": type(res).__name__})
        # memory_model / host_memory_gib run on the same inputs: their rows name the compat row by index
        i = len(cv) - 1
        st, res = call(a.memory_model, m, p)
        mv.append({"i": i, **({"out": floatmap(res)} if st == "ok" else {"raises": type(res).__name__})})
        st, res = call(a.host_memory_gib, m, p, hw)
        hv.append({"i": i, **({"out": floatmap(res)} if st == "ok" else {"raises": type(res).__name__})})
    out["compat"], out["memory"], out["host_memory"] = cv, mv, hv
    return out


VLLM_FOCUS: dict[str, list[Any]] = {
    "tensor_parallel_size": [1, 2, 4, 8, 3, 2.0], "pipeline_parallel_size": [1, 2], "data_parallel_size": [1, 2],
    "kv_cache_dtype": ["fp8", "fp8_e4m3", "auto", "int4_per_token_head", "nvfp4", "fp8_per_token_head", "bfloat16"],
    "attention_backend": ["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "CUTLASS_MLA", "FLASHMLA", "TRITON_MLA",
                          "FLASHINFER_MLA_SPARSE_SM120", "TURBOQUANT", "auto", "HPC_ATTN", "FLASH_ATTN_DIFFKV"],
    "moe_backend": ["auto", "aiter", "hpc", "deep_gemm", "flashinfer_b12x", "cutlass", "flydsl", "marlin"],
    "linear_backend": ["xpu", "aiter", "auto", "xpu_woq", "marlin"],
    "quantization": ["fp8", "awq", "mxfp4", "modelopt_fp4"], "gpu_memory_utilization": [0.9, 0.05, 0.97, 1, 0.3, 0.92],
    "max_model_len": [262144, 65536, 4096, 10**7, 131072.0], "hf_overrides": [{"a": 1}, '{"a": 1}'],
    "allow_long_max_model_len": [True, False], "kv_cache_memory_bytes": [7_000_000_000, 0],
    "max_num_seqs": [256, 8, 0], "enforce_eager": [True, False], "cpu_offload_gb": [4, 0, 2.5],
    "trust_remote_code": [True], "enable_sleep_mode": [True], "tool_call_parser": ["hermes", "qwen3_xml", "openai",
                                                                                  "qwen3_coder", "glm47"],
    "enable_auto_tool_choice": [True, False], "reasoning_parser": ["qwen3", "deepseek_r1"],
    "enable_expert_parallel": [True], "prefix_caching_hash_algo": ["xxhash", "xxhash_cbor", "sha256_cbor"],
    "speculative_config": [{"method": "mtp"}, '{"num_speculative_tokens": 2}', {"model": "x"}],
    "spec_method": ["mtp"], "spec_model": ["org/draft"], "spec_tokens": [3], "max_cpu_loras": [1, 4, 2.0],
    "max_loras": [2, 1], "allowed_local_media_path": ["/", "/data"], "lora_modules": [["a=/x"]],
    "enable_lora": [True, False], "kv_offloading_size": [8, 0, 1.5], "served_model_name": [["alias", "chat"], [5]],
    "cudagraph_capture_sizes": [[1, 2, 4], ["٣"], ["²"], [0]], "revision": ["main"], "api_key": ["sekret"],
    "logging_level": ["DEBUG"], "max_lora_rank": [16, "16", 16.0],
}
SGLANG_FOCUS: dict[str, list[Any]] = {
    "tp_size": [1, 2, 4, None, 2.0, 3], "dp_size": [1, 2, None], "pp_size": [1, 2], "ep_size": [1, 2, 3, None],
    "enable_dp_attention": [True, False], "mem_fraction_static": [0.85, 0.95, 0.5, None, 1],
    "quantization": ["fp8", "modelopt_fp4", "nvfp4_online", "mxfp4", "modelopt", "awq"],
    "attention_backend": ["fa3", "flashmla", "trtllm_mla", "fa4", "trtllm_mha", "flashinfer", "triton"],
    "prefill_attention_backend": ["trtllm_mha", "fa3"], "decode_attention_backend": ["flashmla", "cutedsl_mla"],
    "kv_cache_dtype": ["fp8_e4m3", "fp8_e5m2", "nvfp4", "fp4_e2m1", "auto"],
    "linear_attn_decode_backend": ["flashinfer", "triton"], "linear_attn_backend": ["flashinfer", "triton"],
    "mamba_ssm_dtype": ["float32", "bfloat16", None], "tool_call_parser": ["qwen", "qwen3_coder", "hermes", "auto",
                                                                          "qwen25", "gpt-oss", "glm47"],
    "context_length": [65536, 1048576, 4096.0], "allow_longer_context_len": [True, False],
    "chunked_prefill_size": [-1, 8192], "enable_mixed_chunk": [True], "enable_torch_compile": [True],
    "cpu_offload_gb": [2, 0, 8], "enable_metrics": [False, True], "log_requests": [True], "trust_remote_code": [True],
    "speculative_algorithm": ["EAGLE", "EAGLE3", "NEXTN", None, "STANDALONE"],
    "speculative_draft_model_path": ["org/draft"], "speculative_num_steps": [3],
    "enable_hierarchical_cache": [True, False], "disable_radix_cache": [True], "hicache_size": [16, 0],
    "hicache_ratio": [9.0, 2.0], "enable_trace": [True], "otlp_traces_endpoint": ["localhost:4317"],
    "served_model_name": ["mine", 5], "max_running_requests": [8, 0], "schedule_conservativeness": [1, 1.0],
    "json_model_override_args": [{"k": 1}, "{bad", '{"rope": {"factor": 4.0}}'], "lora_paths": [["a=/x", "b=/y"]],
    "default_chat_template_kwargs": [{"enable_thinking": True}], "revision": ["main"], "api_key": ["s3cret"],
    "watchdog_timeout": [1800, 0], "page_size": [64]}


# Parameter sets that reach every compatibility rule (the random draws rarely combine two specific keys).
SCENARIOS: dict[str, list[dict[str, Any]]] = {
    "vllm": [
        {}, {"kv_cache_dtype": "int4_per_token_head", "attention_backend": "FLASHINFER"},
        {"kv_cache_dtype": "fp8", "attention_backend": "FLASH_ATTN"}, {"kv_cache_dtype": "nvfp4", "attention_backend":
                                                                       "TRITON_ATTN"},
        {"kv_cache_dtype": "fp8", "attention_backend": "FLASH_ATTN_DIFFKV"}, {"max_model_len": 262144, "hf_overrides":
                                                                             {"a": 1}},
        {"max_model_len": 262144}, {"max_model_len": 262144, "allow_long_max_model_len": True},
        {"tensor_parallel_size": 2}, {"tensor_parallel_size": 8}, {"tensor_parallel_size": 3},
        {"tensor_parallel_size": 2, "data_parallel_size": 2}, {"moe_backend": "deep_gemm"},
        {"moe_backend": "hpc"}, {"linear_backend": "xpu_woq"}, {"attention_backend": "FLASHMLA"},
        {"tool_call_parser": "hermes", "enable_auto_tool_choice": True}, {"tool_call_parser": "qwen3_xml"},
        {"reasoning_parser": "qwen3"}, {"quantization": "fp8"}, {"gpu_memory_utilization": 0.97},
        {"gpu_memory_utilization": 0.3, "kv_cache_memory_bytes": 7_000_000_000}, {"enable_expert_parallel": True},
        {"prefix_caching_hash_algo": "xxhash"}, {"enforce_eager": True, "cpu_offload_gb": 4, "trust_remote_code": True,
                                                 "enable_sleep_mode": True}],
    "sglang": [
        {}, {"enable_dp_attention": True, "tp_size": 2, "dp_size": 1}, {"enable_dp_attention": True, "tp_size": 2,
                                                                       "dp_size": 2},
        {"quantization": "modelopt"}, {"quantization": "modelopt_fp4"}, {"quantization": "fp8"},
        {"quantization": "mxfp4"}, {"quantization": "nvfp4_online"}, {"chunked_prefill_size": -1,
                                                                      "enable_mixed_chunk": True},
        {"attention_backend": "trtllm_mha", "kv_cache_dtype": "fp8_e4m3"}, {"attention_backend": "fa3"},
        {"decode_attention_backend": "flashmla", "prefill_attention_backend": "fa4"},
        {"linear_attn_decode_backend": "flashinfer", "mamba_ssm_dtype": "float32"}, {"linear_attn_backend":
                                                                                    "flashinfer"},
        {"tp_size": 2}, {"tp_size": 4}, {"ep_size": 2, "tp_size": 2}, {"mem_fraction_static": 0.95},
        {"context_length": 1048576}, {"context_length": 1048576, "allow_longer_context_len": True},
        {"tool_call_parser": "qwen"}, {"tool_call_parser": "auto"}, {"kv_cache_dtype": "nvfp4"},
        {"enable_torch_compile": True, "cpu_offload_gb": 2, "enable_metrics": False, "log_requests": True,
         "trust_remote_code": True}],
}


# ── metrics text ──────────────────────────────────────────────────────────────────────────────────────────────
VLLM_NAMES = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
              "vllm:gpu_cache_usage_perc", "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
              "vllm:gpu_prefix_cache_hits_total", "vllm:gpu_prefix_cache_queries", "vllm:gpu_prefix_cache_hit_rate",
              "vllm:cpu_prefix_cache_hit_rate", "vllm:prompt_tokens_total", "vllm:generation_tokens",
              "vllm:num_preemptions_total", "vllm:spec_decode_num_accepted_tokens_total",
              "vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_draft_acceptance_rate",
              "process_cpu_seconds_total", "vllm:other"]
VLLM_HISTS = ["vllm:time_to_first_token_seconds", "vllm:inter_token_latency_seconds",
              "vllm:time_per_output_token_seconds", "vllm:e2e_request_latency_seconds"]
SG_NAMES = ["sglang:num_running_reqs", "sglang_num_running_reqs", "sglang:num_queue_reqs", "sglang:token_usage",
            "sglang:full_token_usage", "sglang:kv_used_tokens", "sglang:max_total_num_tokens", "sglang:cache_hit_rate",
            "sglang:prompt_tokens_total", "sglang_prompt_tokens", "sglang:generation_tokens_total",
            "sglang:gen_throughput", "sglang:num_retracted_requests_total", "sglang:num_retracted_reqs",
            "sglang:spec_accept_rate", "sglang:spec_accept_length", "sglang:x_sum", "sglang:x_count",
            "sglang:x_created", "other_metric"]
SG_HISTS = ["sglang:time_to_first_token_seconds", "sglang:inter_token_latency_seconds",
            "sglang:e2e_request_latency_seconds", "sglang_e2e_request_latency_seconds_total"]
VALUES = ["0", "1", "2.5", "0.42", "1e3", "NaN", "+Inf", "-Inf", "inf", "1_000", "١٢", "abc", "-0", "7", "0.0",
          "120000.0", "3.0", "1.5", "nan", " 4", "0.75", "100", "1.7976931348623157e308"]
LES = ["0.1", "0.5", "1.0", "+Inf", "Inf", "inf", "0.25", "5.0", "0.05", "1e-2", "-0", "0", "10", "2.5"]
WEIRD_LES = ["NaN", "abc", "", " 1 ", "1_0", "٣", "-inf"]
SEPS = ["\n", "\n", "\n", "\r\n", "\x0b", " ", "\x1c"]


def prom_line(names: list[str], hists: list[str], weird: bool) -> str:
    c = R.random()
    if c < 0.08:
        return R.choice(["# HELP x y", "# TYPE x gauge", "", "garbage line", "{}", "x{a=\"b\"}", "  "])
    labels = []
    for _ in range(R.randint(0, 3)):
        labels.append(R.choice(['model_name="m"', 'model_name="z"', 'engine="0"', 'engine="1"', 'tp_rank="0"',
                                'tp_rank="1"', 'dp_rank="0"', 'l="q\\"x"', 'path="a\\\\b"', 'model_name="m2"',
                                'is_streaming="true"']))
    ws = R.choice([" ", " ", "  ", "\t", " ", "\x1f"])
    if c < 0.45:
        h = R.choice(hists)
        le = R.choice(LES + (WEIRD_LES if weird else []))
        lab = ",".join([f'le="{le}"'] + labels) if R.random() < 0.9 else ",".join(labels)
        val = R.choice(["0", "10", "50", "90", "100", "100.0", "5", "1e2", str(R.randint(0, 200))])
        return f"{h}_bucket{{{lab}}}{ws}{val}"
    name = R.choice(names)
    lab = "{" + ",".join(labels) + "}" if labels or R.random() < 0.2 else ""
    ts = R.choice(["", "", "", " 1712345", " -5", " abc"])
    return f"{name}{lab}{ws}{R.choice(VALUES)}{ts}"


def prom_text(names, hists) -> str:
    weird = R.random() < 0.15
    lines = [prom_line(names, hists, weird) for _ in range(R.randint(0, 25))]
    out = ""
    for ln in lines:
        out += ln + R.choice(SEPS)
    return out if R.random() < 0.9 else out.rstrip()


# ── log lines ─────────────────────────────────────────────────────────────────────────────────────────────────
NUMS = ["0", "3", "47", "100", "534,714", "1,2", ",", "22.13", "3.962537", "1.2.3", ".", "٤٢", "²", "9" * 25,
        "9" * 400, "0.59", "15.73", "2.04", "262,144", "70000", "316", "1024", "007"]


def n() -> str:
    return R.choice(NUMS)


def v() -> str:
    return R.choice(["0.0", "5.4", "110.6", "91.2", "2.8", "100.0", "0.5"])


def i() -> str:
    return R.choice(["0", "1", "4", "2", "17"])


def vllm_line() -> str:
    t = [
        lambda: f"(EngineCore pid=1) INFO GPU KV cache size: {n()} tokens",
        lambda: f"Maximum concurrency for {n()} tokens per request: {n()}x",
        lambda: f"[gpu_worker.py] Available KV cache memory: {n()} GiB",
        lambda: f"Initial free memory 56.57 GiB, reserved {n()} GiB memory for KV Cache as specified",
        lambda: f"Model loading took {n()} GiB{R.choice([' memory', ''])} and {n()} seconds",
        lambda: "Starting to load model x...",
        lambda: f"Loading safetensors checkpoint shards: {R.choice([' ', '  ', ''])}{n()}% Completed | 1/2",
        lambda: f"Loading pt checkpoint shards:  {n()}% | Loading checkpoint shards: {n()}%",
        lambda: "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): " + "".join(
            f"\r{R.choice(['Capturing CUDA graphs (decode, FULL): ', ''])} {n()}%|x| 1/43 [00:00]" for _ in range(3)),
        lambda: "Capturing CUDA graphs (decode)",
        lambda: f"Graph capturing finished in {n()} secs, took {n()} GiB",
        lambda: f"init engine (profile, create kv cache, warmup model) took {n()} s (compilation: 2.30 s)",
        lambda: R.choice(["torch.compile took 2.24 s", "Dynamo bytecode transform time: 3 s",
                          "Compiling a graph for dynamic shape", "Directly load AOT compilation from x"]),
        lambda: (f"Engine 000: Avg prompt throughput: {v()} tokens/s, Avg generation throughput: {v()} tokens/s, "
                 f"Running: {i()} reqs, Waiting: {i()} reqs" + R.choice(["", f", Deferred: {i()} reqs"])
                 + R.choice(["", f", Preemptions: {i()}"]) + f", GPU KV cache usage: {v()}%"
                 + R.choice(["", f", Prefix cache hit rate: {v()}%"])),
        lambda: (f"Engine 000: Avg prompt throughput: {n()} tokens/s, Avg generation throughput: {n()} tokens/s, "
                 f"Running: {n()} req{R.choice(['', 's'])}, Waiting: {n()} reqs"
                 + R.choice(["", f", Deferred: {n()} reqs"]) + R.choice(["", f", Preemptions: {n()}"])
                 + f", GPU KV cache usage: {n()}%" + R.choice(["", f", Prefix cache hit rate: {n()}%"])),
        lambda: R.choice(["torch.OutOfMemoryError: CUDA out of memory. Tried", "x OutOfMemoryError y" * 40]),
        lambda: f"ValueError: Based on the available memory, the estimated maximum model length is {n()}. Try",
        lambda: R.choice(["9.6 GiB KV cache is needed, which is larger than the available KV cache memory (2 GiB)",
                          "larger than the maximum number of tokens that can be stored in KV cache",
                          "No available memory for the cache blocks."]),
        lambda: f"max_num_seqs ({n()}) exceeds available Mamba cache blocks ({n()}).",
        lambda: f"Selected backend {R.choice(['FLASH_ATTN', 'Ünï_2', 'x²', 'a-b'])} is not valid for this configuration",
        lambda: R.choice(["RuntimeError: No CUDA runtime is found", "Failed to infer device type"]),
        lambda: "Free memory on device cuda:0 (35/95 GiB) on startup is less than desired GPU memory utilization (0.9)",
        lambda: "(APIServer pid=1) INFO:     Application startup complete.",
        lambda: R.choice(["INFO random unrelated line", "", "   ", "\x1b[32mINFO\x1b[0m nothing"]),
    ]
    s = R.choice(t)()
    if R.random() < 0.3:
        s = f"\x1b[{R.choice(['32', '1;31', '0', ''])}m" + s + "\x1b[0m"
    if R.random() < 0.1:
        s = "  " + s + "  "
    return s


def sglang_line() -> str:
    t = [
        lambda: f"[2026] Load weight begin. avail mem={n()} GB",
        lambda: f"Loading safetensors checkpoint shards: {n()}% Completed | {n()}/{n()} [00:10<00:10]",
        lambda: f"Multi-thread loading shards:  {n()}%|██▌    | {n()}/{n()} [00:02<00:06]",
        lambda: f"LOADING CHECKPOINT {n()}%|x| 1/4",
        lambda: (f"Load weight end. {R.choice(['elapsed=' + n() + ' s, ', ''])}type=Qwen, dtype=torch.bfloat16, "
                 f"avail mem={n()} GB, mem usage={n()} GB."),
        lambda: f"KV Cache is allocated. dtype: torch.bfloat16, #tokens: {n()}, K size: 20.00 GB",
        lambda: f"Full KV Cache VA upper bound. #tokens: {n()}, K size: 1.00 GB",
        lambda: (f"max_total_num_tokens={n()}, chunked_prefill_size=8192, "
                 + R.choice([f"max_running_requests={n()}, ", ""]) + R.choice([f"context_len={n()}, ", ""])
                 + "available_gpu_mem=9.5 GB"),
        lambda: f"Capture {R.choice(['decode CUDA', 'prefill CUDA', 'cuda', 'x y cuda'])} graph begin. avail mem=9.5 GB",
        lambda: f"Capturing batches (bs={n()} avail_mem=9.10 GB):  {n()}%|████      | 4/10 [00:03<00:04]",
        lambda: f"Capturing num tokens (num_tokens={n()} avail_mem=9.00 GB)",
        lambda: f"Capture decode CUDA graph end. elapsed=12.31 s, mem usage={n()} GB, avail mem=8.30 GB.",
        lambda: "Capture cuda graph end. Time elapsed: 12.31 s.",
        lambda: "INFO:     The server is fired up and ready to roll!",
        lambda: R.choice(["torch.OutOfMemoryError: CUDA out of memory.", "The memory capacity is unbalanced. x"]),
        lambda: f"ValueError: Raise --mem-fraction-static above {n()} (minimum viable)",
        lambda: "AssertionError: FlashAttention v3 Backend requires SM>=80 and SM<=90.",
        lambda: "Capture cuda graph failed: boom\nPossible solutions",
        lambda: R.choice(["Compiling a graph", "torch.compile done", "Compile x", "Prefill batch, #new-seq: 1", "",
                          "\r", "a\rb\r", "x\r\r"]),
        lambda: (f"Loading safetensors checkpoint shards:   0% Completed | 0/4\rLoading safetensors checkpoint "
                 f"shards:  {n()}% Completed | 3/4"),
    ]
    s = R.choice(t)()
    if R.random() < 0.3:
        s = "\x1b[32m" + s + "\x1b[0m"
    if R.random() < 0.1:
        s = s + "\r"
    return s


def log_vectors(parse, gen, fixtures: list[str], count: int) -> list[dict[str, Any]]:
    out = []
    lines = [x for f in fixtures for x in (FIX / f).read_text().splitlines()] + [(FIX / f).read_text()
                                                                                  for f in fixtures]
    lines += [gen() for _ in range(count)]
    for line in lines:
        st, res = call(parse, line)
        if st == "raise":
            out.append({"line": line, "raises": type(res).__name__})
        elif res is not None and has_nonfinite(res):
            out.append({"line": line, "nonfinite": True})
        else:
            out.append({"line": line, "out": res})
    return out


def metrics_vectors(parse_new, names, hists, fixtures: list[str], count: int) -> list[dict[str, Any]]:
    texts = [(FIX / f).read_text() for f in fixtures] + [prom_text(names, hists) for _ in range(count)]
    out = []
    for t in texts:
        st, res = call(parse_new(), t)
        out.append({"text": t, **({"out": floatmap(res)} if st == "ok" else {"raises": type(res).__name__})})
    return out


def vllm_rate_vectors(count: int) -> list[dict[str, Any]]:
    out = []
    for _ in range(count):
        t = [100.0]
        a = VllmAdapter(clock=lambda: t[0])
        steps = []
        for _ in range(R.randint(2, 4)):
            t[0] += R.choice([0.0, 0.1, 0.2, 2.0, 2.5, -1.0, 1e-3])
            text = prom_text(["vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:prompt_tokens",
                              "vllm:generation_tokens", "vllm:num_requests_running"], VLLM_HISTS)
            st, res = call(a.parse_metrics, text)
            steps.append({"t": t[0], "text": text,
                          **({"out": floatmap(res)} if st == "ok" else {"raises": type(res).__name__})})
        out.append({"steps": steps})
    return out


# ── python semantics vectors ──────────────────────────────────────────────────────────────────────────────────
def py_vectors() -> dict[str, Any]:
    from engine_console.adapters import vllm_parse
    sums = []
    for _ in range(400):
        vals = [R.choice([R.uniform(-1e6, 1e6), R.uniform(-1, 1), 0.1, 1e100, -1e100, 1e-300, -0.0, 0.0, 3.0,
                          float("inf"), float("-inf"), 1e308, R.random() * 10 ** R.randint(-20, 20)])
                for _ in range(R.randint(1, 12))]
        sums.append([[repr(v) for v in vals], repr(sum(vals))])
    floats = []
    for s in ["1", " 1_000.5 ", "١٢", "-iNfInItY", "nan", "", ".", "1.2.3", "1__0", "_1", "1_", "0x10", "e5", "1e",
              "abc", "1_.5", "1.", ".5", "1.e5", "+1.5", "- 1", "1e400", "-0", " 1 ", "\x1c2\x1f", "1e-400",
              "²", "1_0e1_0", "INF", "+nan", "1,000", "٣.٥", "0001.5000"]:
        try:
            floats.append([s, repr(float(s))])
        except ValueError:
            floats.append([s, None])
    ints = []
    for s in ["7", " 007 ", "-0", "+5", "1_000", "1__0", "_1", "1_", "٣", "²", "1.0", "", " ", "- 1",
              "123456789012345678901234567890", " 9 "]:
        try:
            ints.append([s, str(int(s))])
        except ValueError:
            ints.append([s, None])
    loads = []
    for s in JSON_TEXT + ['{"a": [1, {"b": "\\n\\t\\"\\\\\\/"}], "c": 1E5, "d": -1.5e+3, "e": 0}', "[1,2,3,]", "{]",
                          '{"a":}', '{"a" :1 , "b":[ ] }', "nulls", "null", "true", "false", "[NaN, -Infinity]",
                          '"\\ud800\\udc00"', '"\\ud800\\uZZZZ"', '"\\ud800\\u0041"', '"\\ud800"x', '"abc\\', '"\\u00e9"',
                          '"' + "é \U0001f600" + '"', "[[]]", "{}", "[]", '{"":1}', "-0.0", "1.5", "-"]:
        try:
            v = json.loads(s)
            dumped = json.dumps(v, separators=(",", ":"))
            loads.append({"text": s, "dumps": dumped, "container": isinstance(v, (dict, list))})
        except json.JSONDecodeError as e:
            loads.append({"text": s, "msg": e.msg})
    rounds = []
    for _ in range(300):
        x = R.choice([R.uniform(-200, 200), R.random(), 1.03125, 0.00005, 2.5, 0.125, 1e20, -1e-10, R.random() * 1e-3])
        rounds.append([repr(x), repr(round(x, 4)), format(x, ".1f"), format(x, ".2f"), format(x, "g")])
    sorts = []
    for _ in range(200):
        keys = list({R.choice([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, math.inf, -1.0, 0.0, 3.0, 7.5]) for _ in
                     range(R.randint(1, 10))})
        R.shuffle(keys)
        items = [(k, i) for i, k in enumerate(keys)]
        for _ in range(R.randint(0, 3)):
            items.insert(R.randint(0, len(items)), (float("nan"), -1))
        items = [(k, i) for i, (k, _) in enumerate(items)]      # payload = position
        order = [i for _, i in sorted(items)]
        sorts.append({"keys": [repr(k) for k, _ in items], "order": order})
    samples = []
    for _ in range(150):
        t = prom_text(VLLM_NAMES + SG_NAMES, VLLM_HISTS)
        samples.append({"text": t, "out": [[nm, lab, repr(v)] for nm, lab, v in vllm_parse.parse_samples(t)]})
    quant = []
    for _ in range(300):
        les = sorted({R.choice([0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]) for _ in range(R.randint(1, 6))})
        cum, c = [], 0.0
        for le in les:
            c += R.choice([0.0, 1.0, 5.0, 10.0, 0.5])
            cum.append((le, c))
        if R.random() < 0.7:
            cum.append((math.inf, c + R.choice([0.0, 2.0])))
        q = R.choice([0.5, 0.95, 0.99, 0.0, 1.0])
        from engine_console.adapters import sglang_parsers
        quant.append({"q": q, "buckets": [[repr(a), repr(b)] for a, b in cum],
                      "vllm": None if (r := vllm_parse.histogram_quantile(q, cum)) is None else repr(r),
                      "sglang": None if (r2 := sglang_parsers.histogram_quantile(q, cum)) is None else repr(r2)})
    return {"sum": sums, "float": floats, "int": ints, "loads": loads, "round": rounds, "sorted": sorts,
            "parse_samples": samples, "quantile": quant}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    vl = engine_vectors("vllm", VllmAdapter, vl_catalog, VLLM_FOCUS, 900, 600)
    vl["metrics"] = metrics_vectors(lambda: VllmAdapter().parse_metrics, VLLM_NAMES, VLLM_HISTS,
                                    ["vllm/fixtures/metrics_v0.27.1.prom"], 300)
    vl["rates"] = vllm_rate_vectors(80)
    vl["log"] = log_vectors(VllmAdapter().parse_startup_log, vllm_line,
                            ["vllm/fixtures/startup_v0.27.1.log", "vllm/fixtures/tqdm_line.log"], 1500)
    sg = engine_vectors("sglang", SglangAdapter, sg_catalog, SGLANG_FOCUS, 900, 600)
    sg["metrics"] = metrics_vectors(lambda: SglangAdapter().parse_metrics, SG_NAMES, SG_HISTS,
                                    ["sglang/fixtures/metrics_v0.5.20.prom"], 300)
    sg["log"] = log_vectors(SglangAdapter().parse_startup_log, sglang_line, [], 1500)
    for name, doc in (("vllm", vl), ("sglang", sg), ("py", py_vectors())):
        text = json.dumps(enc(doc), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        (OUT / f"{name}.json").write_text(text + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}/{name}.json ({len(text) // 1024} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
