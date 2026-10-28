"""Golden vectors for the TensorRT-LLM, llama.cpp and Ollama adapters of the Rust port.

Runs the Python adapters (the reference implementation) over a large, seeded set of inputs and records exactly what
they return, so the Rust tests (backend/crates/ec-hub/tests/test_ws_a2_goldens.rs) can demand the same output:

  catalogs.json         identity, parameter catalog (ParamSpec dumps) and presets of each engine, in order
  <engine>_cases.json   validate(), build_launch_ctx(), compatibility(), memory_model(), host_memory_gib() results
  parse_cases.json      parse_startup_log() / parse_metrics() of each engine over fixture lines and generated ones
  py_cases.json         the Python behaviours the adapters lean on: json.loads messages, yaml.safe_dump text,
                        float(), str.isdigit()/int(), datetime.strptime of the trtllm timestamp

Offline and hermetic: imports only engine_console.adapters (no services, no network, no docker). Deterministic: one
seeded random.Random per file. Run from the repository root with the Python reference (gen/pyref.py):

    env -u LITELLM_API_KEY -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u HF_TOKEN \
        "$(node backend/parity/lib/python-release.mjs)/backend/.venv/bin/python" -I backend/parity/gen/goldens_ws_a2.py

Encoding: a non-finite float is written as {"$f": "nan" | "inf" | "-inf"} (JSON has no NaN); an exception as
{"$raise": "<TypeName>", "msg": "<str(e)>"}. Every file is a JSON object whose list members hold one case per line.
"""
from __future__ import annotations

import json
import math
import random
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyref import DATA, tree  # noqa: E402

from engine_console.adapters.base import Hardware, LaunchContext, ModelInfo  # noqa: E402
from engine_console.adapters.llamacpp import LlamaCppAdapter  # noqa: E402
from engine_console.adapters.ollama import OllamaAdapter  # noqa: E402
from engine_console.adapters.trtllm import TrtllmAdapter  # noqa: E402

OUT = DATA / "ws_a2"
FIX = tree() / "backend" / "tests"
ADAPTERS = {"trtllm": TrtllmAdapter(), "llamacpp": LlamaCppAdapter(), "ollama": OllamaAdapter()}


# ------------------------------------------------------------------ encoding
def enc(v: Any) -> Any:
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        if math.isnan(v):
            return {"$f": "nan"}
        if math.isinf(v):
            return {"$f": "inf" if v > 0 else "-inf"}
        return v
    if isinstance(v, dict):
        return {str(k): enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [enc(x) for x in v]
    if hasattr(v, "model_dump"):
        return enc(v.model_dump())
    raise TypeError(type(v).__name__)


def call(fn: Any, *a: Any, **kw: Any) -> Any:
    try:
        return enc(fn(*a, **kw))
    except Exception as e:  # noqa: BLE001 - recorded, the Rust side decides what it means
        return {"$raise": type(e).__name__, "msg": str(e)}


def write(name: str, doc: dict[str, Any]) -> None:
    """One case per line inside each list member, so a diff shows which case changed."""
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for key, val in doc.items():
        if isinstance(val, list):
            body = ",\n".join("  " + json.dumps(x, ensure_ascii=False, allow_nan=False) for x in val)
            parts.append(f"{json.dumps(key)}: [\n{body}\n]")
        else:
            parts.append(f"{json.dumps(key)}: {json.dumps(val, ensure_ascii=False, allow_nan=False)}")
    text = "{\n" + ",\n".join(parts) + "\n}\n"
    (OUT / name).write_text(text, encoding="utf-8")
    print(f"{name}: {len(text) // 1024} KiB", file=sys.stderr)


# ------------------------------------------------------------------ catalogs
def catalogs() -> None:
    out = {}
    for eid, a in ADAPTERS.items():
        out[eid] = {
            "id": a.id, "display_name": a.display_name, "default_image": a.default_image, "default_port": a.default_port,
            "model_formats": sorted(a.model_formats), "images_by_arch": dict(a.images_by_arch),
            "multi_model": a.multi_model, "metric_prefixes": list(a.metric_prefixes),
            "catalog": [p.model_dump(mode="json") for p in a.param_catalog()],
            "presets": [[n, p] for n, p in a.presets().items()],
            "preset_info": {n: e.model_dump() for n, e in a.preset_info().items()},
            "image_for": {arch: a.image_for(arch) for arch in ("x86_64", "aarch64", "riscv64")},
        }
    write("catalogs.json", {"engines": [out[e] for e in ADAPTERS]})


# ------------------------------------------------------------------ value pools
STR_NOISE = ["", "   ", "-x", "a b", "a\nb", "x\n", "a\tb", "a\x00b", "a\x1fb", "a\x7fb", "x" * 4001, "模型", "/abs/path",
             "{x}", "{{ m }}", "line\nline", "ok", "ok-1.2_3", "--evil", "a,b", "a:b", "a=b", " x", " ",
             "x y", "٣", "12", "0", "auto", "all", "none", "/models/a.gguf", "/models/../x.gguf",
             "/models/a.GGUF", "/models/a.gguf:0.5", "exps=CPU", "k=int:1", "CUDA0,CUDA1", "3,1", "0.5,0.5",
             "https://chat.example.org", "localhost", "*", "1h30m", "-1", "5m", "0.5h", "300µs", "cuda_v13",
             "nvcr.io/nvidia/tensorrt-llm/release:1.3.0", "nvcr.io/nvidia/tensorrt-llm/release:1.2.1\n",
             "repo:latest", "repo", "localhost:5000/team/img:tag", "example-org/model\n", "x" * 600]


def num_pool(spec: Any) -> list[Any]:
    vals: list[Any] = [0, 1, -1, 2, 3, 4, 7, 8, 48, 64, 1024, 5.0, 1.5, 0.5, 0.25, -0.0, 0.0, True, False, "8", None,
                       10 ** 12, -(10 ** 12), 2 ** 53 + 1, 1e20, 1e-7, 1e300, 123456.789, [1], {"a": 1}]
    for v in (spec.min, spec.max, spec.default):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals += [v, v - 1, v + 1, v + 0.5, v - 0.001, int(v)]
    return vals


def pool(spec: Any, extra: dict[str, list[Any]]) -> list[Any]:
    t = spec.type
    vals: list[Any] = list(extra.get(spec.key, []))
    if t == "bool":
        vals += [True, False, 1, 0, "true", None, "False"]
    elif t in ("int", "float"):
        vals += num_pool(spec)
    elif t == "enum":
        ch = spec.choices or []
        vals += list(ch) + ["nope", ch[0].upper(), ch[0].lower(), " " + ch[0], 5, True, "", None]
    elif t == "string":
        vals += STR_NOISE + [5, 1.5, True, None, ["a"]]
    elif t == "string_list":
        vals += [[], ["a"], ["1", "2"], [1, 2], ["a b"], "a", [True], [""], ["-x"], ["0"], ["٣"], [1.5], ["4096", "512"],
                 [4096, "512"], ["cutlass"], ["cutlass", "cuda_core"], ["magic"], ["a,b"], ["https://x.example", "app://*"],
                 ["gpt-4o", "org/model:q4"], [" a"], ["a\x01"], None, [-1], ["-1"], ["08"], ["²"], ["1", "³"],
                 ["9" * 4400]]
    elif t == "json":
        vals += [{}, {"a": 1}, '{"a": 1}', "{bad", "[1]", [1], 5, None, "NaN", '{"a": NaN}', '{"a": 1e400}',
                 {"decoding_type": "MTP"}, '{"decoding_type": "NGram", "max_draft_len": 2}',
                 {"decoding_type": "Medusa"}, {"video": {"num_frames": 8}}, '{"enable_thinking": false}',
                 {"enable_min_latency": True}, {"model": "x"}, {"max_batch_size": 4}, {"kv_cache_config": {}},
                 {"garbage_collection_gen0_threshold": 10000}, {"bogus_key": 1}, '{"a":1,"a":2}', "",
                 '{"k": "\\u00e9 é"}', {"nested": {"deep": [1, 2.5, None, True, "s"]}}, '{"a": [1, {"b": null}]}',
                 '{"a":1,}', "[1,]", '"just a string"', '{"big": 123456789012345678901234567890}',
                 {"x": "y" * 3600}, {"enable_iter_perf_stats": True}, {"env_overrides": {}}]
    return vals


TRT_EXTRA: dict[str, list[Any]] = {}
LLAMA_EXTRA: dict[str, list[Any]] = {
    "n_gpu_layers": ["all", "auto", "12", 12, 0, "0", -1, "most", 1.5, True, "٣", "²", "①", "9" * 4400],
    "spec_draft_ngl": ["all", "auto", "12", 12, "some"],
    "chat_template": ["chatml", "llama3", "{{ messages }}", "custom-name"],
    "chat_template_jinja": ["{% for m in messages %}{{ m.content }}{% endfor %}\n", "{{ x }}", "x" * 130_000],
    "lora": ["/models/a.gguf,/models/b.gguf", "/models/a.gguf,/tmp/b.gguf"],
    "lora_scaled": ["/models/a.gguf:0.5,/models/b.gguf:-1", "/models/a.gguf", "/models/a.gguf:x"],
    "override_kv": ["tokenizer.ggml.add_bos_token=bool:false", "tokenizer.x=bool"],
    "served_model_name": [["extra", "demo", "extra2"], ["demo"]],
}
OLLAMA_EXTRA: dict[str, list[Any]] = {
    "keep_alive": ["5m", "24h", "1h30m", "90s", "0", "-1", "-1m", 3600, "0.5h", "300ms", " 10m ", "soon", "m5",
                   "5m;rm", -5, 1.5, "1.5", "-5m"],
    "load_timeout": ["30m", 600, "-1", "-5m", "later", 0, -1],
    "llm_library": ["cuda_v13", "vulkan", "cpu", "cpu_avx2", "CUDA", "../x", "cuda v13"],
}
EXTRAS = {"trtllm": TRT_EXTRA, "llamacpp": LLAMA_EXTRA, "ollama": OLLAMA_EXTRA}


def samples_for(eid: str) -> list[dict[str, Any]]:
    """The param sets the Python tests use (each one valid on its own)."""
    sys.path.insert(0, str(FIX.parent))
    if eid == "trtllm":
        from tests.trtllm.samples import SAMPLES, SCENARIOS, sample_params
        return [sample_params(k) for k in SAMPLES] + list(SCENARIOS.values())
    if eid == "llamacpp":
        from tests.llamacpp.acceptance_cases import values_for
        return [{p.key: v} for p in ADAPTERS[eid].param_catalog() for v in values_for(p)]
    from tests.ollama.live_cases import ADAPTER_PARAMS
    return [ADAPTER_PARAMS]


def _valid(a: Any, params: dict[str, Any]) -> bool:
    try:
        return not a.validate(params)
    except Exception:  # noqa: BLE001 - a value Python's validate() raises on is not a valid pick
        return False


def random_params(rng: random.Random, a: Any, extra: dict[str, list[Any]], valid_bias: float, n: int) -> dict[str, Any]:
    cat = a.param_catalog()
    out: dict[str, Any] = {}
    for spec in rng.sample(cat, min(n, len(cat))):
        candidates = pool(spec, extra)
        if rng.random() < valid_bias:
            good = [v for v in candidates if v is not None and _valid(a, {spec.key: v})]
            if good:
                out[spec.key] = rng.choice(good)
                continue
        out[spec.key] = rng.choice(candidates)
    return out


# ------------------------------------------------------------------ models and hardware
def hw_pool() -> list[Hardware]:
    def hw(n: int, cc: str = "12.0", free: float = 90.0, total: float = 95.6, ram: float = 246.0, arch: str = "x86_64",
           frees: list[float] | None = None) -> Hardware:
        return Hardware(gpu_ids=list(range(n)), gpu_uuids=[f"GPU-{i}" for i in range(n)], gpu_names=["RTX"] * n,
                        gpu_total_gib=[total] * n, gpu_free_gib=frees if frees is not None else [free] * n,
                        compute_capability=cc, host_ram_gib=ram, arch=arch)
    return [hw(1), hw(2), hw(2, frees=[90.0, 40.0]), hw(4), hw(8, ram=512.0), hw(1, cc="9.0"), hw(1, cc="12.1"),
            hw(1, cc="8.9", free=14.0, total=16.0, ram=64.0), hw(1, arch="aarch64"), hw(1, free=30.0), hw(0),
            hw(1, cc=""), hw(1, cc="x.y"), hw(1, ram=0.0), hw(2, free=10.0), hw(1, total=32.0, free=30.0)]


def model_pool(rng: random.Random) -> list[ModelInfo]:
    gib = 1024 ** 3
    fixed = [
        ModelInfo(repo_id="example-org/chat-model-36B", architectures=["LlamaForCausalLM"], model_type="llama",
                  num_params=36_000_000_000, weight_bytes=36 * gib, num_attention_heads=64, num_kv_heads=8,
                  max_position_embeddings=131072, pipeline_tag="text-generation"),
        ModelInfo(repo_id="example-org/chat-model-32B-GGUF", architectures=["qwen3"], model_type="qwen3",
                  num_params=32_000_000_000, weight_bytes=int(18.6 * gib), quantization="q4_k_m", hidden_size=5120,
                  num_layers=64, num_attention_heads=64, num_kv_heads=8, head_dim=128, max_position_embeddings=40960),
        ModelInfo(repo_id="example-org/moe-30B-A3B-GGUF", architectures=["qwen3moe"], model_type="qwen3moe",
                  num_params=30_500_000_000, num_active_params=3_300_000_000, is_moe=True,
                  weight_bytes=int(17.5 * gib), quantization="q4_k_m", hidden_size=2048, num_layers=48,
                  num_attention_heads=32, num_kv_heads=4, head_dim=128, max_position_embeddings=262144,
                  raw_config={"qwen3moe.expert_count": 128, "qwen3moe.expert_used_count": 8}),
        ModelInfo(repo_id="llama3.1:8b", architectures=["llama"], model_type="llama", num_params=8_000_000_000,
                  weight_bytes=int(4.6 * gib), quantization="q4_k_m", hidden_size=4096, num_layers=32,
                  num_attention_heads=32, num_kv_heads=8, head_dim=128, max_position_embeddings=131072),
        ModelInfo(repo_id="mystery:latest", weight_bytes=int(8 * gib)),
        ModelInfo(repo_id="a/b"),
        ModelInfo(repo_id="Qwen/Qwen3-32B", model_type="qwen3", pipeline_tag="text-generation"),
        ModelInfo(repo_id="Qwen/Qwen3-Coder-30B", model_type="qwen3_moe", pipeline_tag="text-generation"),
        ModelInfo(repo_id="moonshotai/Kimi-K2-Instruct", pipeline_tag="text-generation"),
        ModelInfo(repo_id="deepseek-ai/DeepSeek-R1-0528", architectures=["DeepseekV3ForCausalLM"]),
        ModelInfo(repo_id="Qwen/Qwen2.5-VL-7B", architectures=["Qwen2_5_VLForConditionalGeneration"],
                  pipeline_tag="image-text-to-text"),
        ModelInfo(repo_id="openai/gpt-oss-120b", quantization="mxfp4", architectures=["GptOssForCausalLM"]),
        ModelInfo(repo_id="a/b-FP8", quantization="fp8", raw_config={"quantization_config": {
            "quant_method": "fp8", "weight_block_size": [128, 128]}}),
        ModelInfo(repo_id="a/b", raw_config={"hidden_size": 64}),
        ModelInfo(repo_id="a/b", raw_config={"max_position_embeddings": 131072,
                                             "rope_scaling": {"type": "linear", "factor": 2.0}}),
        ModelInfo(repo_id="nvidia/Nemotron-3-Nano", architectures=["NemotronHForCausalLM"]),
        ModelInfo(repo_id="Qwen/Qwen3-Embedding-8B", pipeline_tag="feature-extraction"),
    ]
    repos = ["Qwen/Qwen3-8B", "Qwen/Qwen3-Coder-480B", "Qwen/Qwen3-30B-Instruct-2507", "Qwen/Qwen2.5-7B",
             "moonshotai/Kimi-K2", "deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3.2-Exp", "deepseek-ai/DeepSeek-V3",
             "nvidia/nemotron-nano-3-8b", "meta-llama/Llama-3.3-70B", "mistralai/Pixtral-12B", "acme/moe-mxfp4",
             "openai/gpt-oss-20b", "llava-hf/llava-1.5", "x/omni-7b", "google/gemma-3-27b"]
    archs = ["LlamaForCausalLM", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "WeirdModel", "Qwen3_5MoeForConditionalGeneration",
             "MllamaForConditionalGeneration", "GptOssForCausalLM", "PixtralForConditionalGeneration", "Phi4MMForCausalLM",
             "DeepseekV3ForCausalLM", "llama", "qwen3moe"]
    quants = [None, "fp8", "nvfp4", "awq", "gptq", "mxfp4", "bitsandbytes", "bnb", "gguf", "exl2", "hqq", "q4_k_m", "FP8"]
    tags = [None, "text-generation", "image-text-to-text", "any-to-any", "video-text-to-text", "feature-extraction"]
    out = list(fixed)
    for _ in range(40):
        layers = rng.choice([None, 24, 32, 48, 64, 80])
        raw: dict[str, Any] = {}
        r = rng.random()
        if r < 0.15:
            raw = {"max_position_embeddings": rng.choice([4096, 32768, 131072]),
                   "rope_scaling": rng.choice([None, {"type": "linear", "factor": 2.0}, {"rope_type": "yarn", "factor": 4.0},
                                               {"type": "llama3", "factor": 8.0}, {"factor": 0}, {"type": "dynamic", "factor": 1.5}])}
        elif r < 0.25:
            raw = {"text_config": {"max_position_embeddings": rng.choice([8192, 65536])}}
        elif r < 0.35:
            raw = {"quantization_config": {"quant_method": "fp8", "weight_block_size": rng.choice([None, [128, 128]])}}
        elif r < 0.45:
            raw = {"llama.expert_count": rng.choice([8, 64, 128]), "llama.expert_used_count": rng.choice([1, 2, 8]),
                   "num_experts": True}
        params = rng.choice([None, 7_000_000_000, 30_000_000_000, 120_000_000_000])
        out.append(ModelInfo(
            repo_id=rng.choice(repos), architectures=rng.sample(archs, rng.choice([0, 1, 1, 2])),
            model_type=rng.choice([None, "llama", "qwen3", "qwen3_moe", "deepseek_v3"]),
            num_params=params, num_active_params=rng.choice([None, 3_000_000_000]) if params else None,
            is_moe=rng.random() < 0.3, weight_bytes=rng.choice([None, 4 * gib, 36 * gib, 90 * gib, 140 * gib, 250 * gib]),
            quantization=rng.choice(quants), num_layers=layers, num_attention_heads=rng.choice([None, 32, 36, 64]),
            num_kv_heads=rng.choice([None, 4, 8]) if layers else None, head_dim=rng.choice([None, 128]) if layers else None,
            max_position_embeddings=rng.choice([None, 4096, 40960, 131072, 262144]), pipeline_tag=rng.choice(tags),
            raw_config=raw))
    return out


# ------------------------------------------------------------------ per-engine cases
def ctx_variants(eid: str, rng: random.Random) -> list[dict[str, Any]]:
    """LaunchContext fields (besides params and hw) to try."""
    if eid == "trtllm":
        return [{"model": "example-org/chat-model-36B"}, {"model": "", "model_format": "plain", "model_path": "/models"},
                {"model": "x", "model_format": "plain", "model_path": None}, {"model": "m", "model_format": "gguf"},
                {"model": "--evil"}, {"model": "a b"}, {"model": "a\nb"}, {"model": ""}, {"model": "example-org/m\n"},
                {"model": "org/模型"}, {"model": "m", "model_format": "ollama"}]
    if eid == "llamacpp":
        return [{"model": "m", "model_format": "gguf", "model_path": "/models/demo-Q4_K_M.gguf", "served_name": "demo"},
                {"model": "m", "model_format": "gguf", "model_path": "/models/snapshots/0123/sub dir/model.Q8_0.gguf",
                 "served_name": "demo"},
                {"model": "m", "model_format": "gguf", "model_path": None}, {"model": "m", "model_format": "gguf",
                                                                           "model_path": "/models/../etc/x.gguf"},
                {"model": "m", "model_format": "gguf", "model_path": "/models/a\n.gguf"},
                {"model": "m", "model_format": "gguf", "model_path": "/models2/a.gguf"},
                {"model": "m", "model_format": "gguf", "model_path": "/models/a.gguf", "served_name": " a,b "},
                {"model": "m", "model_format": "gguf", "model_path": "/models/a.gguf", "served_name": "  x  "},
                {"model": "m", "model_format": "gguf", "model_path": "/models/a.gguf", "served_name": "-x"},
                {"model": "m", "model_format": "hf", "model_path": "/models/a.gguf"},
                {"model": "m", "model_format": "plain", "model_path": "/models/a.gguf"}]
    return [{"model": "llama3.1:8b", "model_format": "ollama"}, {"model": "", "model_format": "ollama"},
            {"model": "hf.co/org/repo-GGUF:Q4_K_M", "model_format": "ollama"},
            {"model": "a b", "model_format": "ollama"}, {"model": "x" * 300, "model_format": "ollama"},
            {"model": "gemma3", "model_format": "gguf"}, {"model": "llama3", "model_format": "hf"}]


def engine_cases(eid: str) -> None:
    a = ADAPTERS[eid]
    rng = random.Random(f"ws-a2-{eid}")
    extra = EXTRAS[eid]
    validate: list[Any] = []
    seen: set[str] = set()

    def add_validate(params: dict[str, Any]) -> None:
        key = repr(sorted(params.items(), key=lambda kv: kv[0]))
        if key in seen:
            return
        seen.add(key)
        validate.append({"params": enc(params), "out": call(a.validate, params)})

    for spec in a.param_catalog():
        for v in pool(spec, extra):
            add_validate({spec.key: v})
    for p in samples_for(eid):
        add_validate(p)
    for _ in range(900):
        add_validate(random_params(rng, a, extra, rng.choice([0.5, 0.9, 1.0]), rng.choice([1, 2, 3, 4, 6])))
    add_validate({"nope": 1, "also_nope": None})

    hws = hw_pool()
    launch: list[Any] = []
    ctxs = ctx_variants(eid, rng)
    param_sets = [{}] + [n_p for _, n_p in a.presets().items()] + samples_for(eid)
    param_sets += [random_params(rng, a, extra, 1.0, rng.choice([1, 2, 3, 5, 8])) for _ in range(250)]
    param_sets += [random_params(rng, a, extra, 0.7, rng.choice([1, 2, 3])) for _ in range(60)]
    def launch_case(params: dict[str, Any], cv: dict[str, Any], hw_i: int, hf: str) -> dict[str, Any]:
        fields = {"model": "m", "served_name": "demo", "hf_cache_container_path": hf} | cv
        ctx = LaunchContext(params=params, hw=hws[hw_i], **fields)
        dumped = enc(ctx.model_dump())
        del dumped["hw"]
        return {"ctx": dumped, "hw": hw_i, "out": call(a.build_launch_ctx, ctx)}

    for i, params in enumerate(param_sets):
        cv = ctxs[0] if i % 7 else ctxs[(i // 7) % len(ctxs)]
        launch.append(launch_case(params, cv, i % len(hws), "/root/.cache/huggingface"))
    for cv in ctxs:
        launch.append(launch_case({}, cv, 0, "/hf"))
    v1 = [{"model": m, "served_name": s, "hf": "/hf"} for m, s in
          (("example-org/m", "s"), ("/models/a.gguf", "demo"), ("llama3.1:8b", "x"), ("org/Model with spaces", "x"))]
    bridge = [{"args": c, "out": call(a.build_launch, c["model"], {}, hws[0], served_name=c["served_name"],
                                      hf_cache_container_path=c["hf"])} for c in v1]

    models = model_pool(rng)
    compat: list[Any] = []
    memory: list[Any] = []
    for i in range(700):
        params = random_params(rng, a, extra, 1.0, rng.choice([0, 1, 2, 3, 4, 6]))
        if i % 8 == 0:
            # unvalidated: trtllm must never raise; the others raise where Python does (int("abc"), "x" < 2, ...)
            params = random_params(rng, a, extra, 0.0, rng.choice([1, 2, 3]))
        mi = i % len(models) if i % 3 else rng.randrange(len(models))
        hi = (i * 7) % len(hws)
        m, hw = models[mi], hws[hi]
        compat.append({"model": mi, "params": enc(params), "hw": hi, "out": call(a.compatibility, m, params, hw)})
        if i % 2 == 0:
            memory.append({"model": mi, "params": enc(params), "hw": hi, "memory_model": call(a.memory_model, m, params),
                           "host_memory_gib": call(a.host_memory_gib, m, params, hw)})
    write(f"{eid}_cases.json", {"models": [enc(m.model_dump()) for m in models],
                                "hws": [enc(h.model_dump()) for h in hws], "validate": validate, "launch": launch,
                                "bridge": bridge, "compat": compat, "memory": memory})


# ------------------------------------------------------------------ parsers
def mutate(rng: random.Random, line: str) -> str:
    ops = rng.randrange(6)
    if ops == 0 and line:
        i = rng.randrange(len(line))
        return line[:i] + line[i + 1:]
    if ops == 1:
        return "\x1b[0;32m" + line + "\x1b[0m"
    if ops == 2:
        return "  " + line + "  "
    if ops == 3 and line:
        i = rng.randrange(len(line))
        return line[:i] + rng.choice(["9", ".", " ", "x", "\t", "é"]) + line[i:]
    if ops == 4:
        return line.upper()
    return line + " " + "x" * rng.randrange(400)


def parse_cases() -> None:
    rng = random.Random("ws-a2-parse")
    out: dict[str, Any] = {}
    sources = {
        "trtllm": [FIX / "trtllm/fixtures/startup_1.2.1.log"],
        "llamacpp": [FIX / "llamacpp/fixtures/startup_synthetic.log", FIX / "llamacpp/fixtures/startup_no_model_b10156.log",
                     FIX / "llamacpp/fixtures/router_mode_probe_b10156.log", FIX / "llamacpp/fixtures/api_key_probe_b10156.log"],
        "ollama": [FIX / "ollama/fixtures/serve_bare_0.30.11.log", FIX / "ollama/fixtures/serve_readonly_no_blobs_0.30.11.log"],
    }
    extra_lines = {
        "trtllm": [
            "RuntimeError: Executor creation failed due to insufficient GPU memory.", "Total GPU memory (GiB): 94.99",
            "Free GPU memory before component creation attempt (GiB): 12.31",
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
            "ValueError: Unknown architecture for AutoModelForCausalLM: FooForCausalLM",
            "NotImplementedError: TRTLLMGenFusedMoE does not support SM120 and above.",
            "ValueError: No GPU devices found on the node",
            "ImportError: libcuda.so.1: cannot open shared object file: No such file or directory",
            "RuntimeError: Failed to bind socket to 0.0.0.0:8000: [Errno 98] Address already in use",
            "RuntimeError: Failed, NCCL error /app/x.cpp:12 'unhandled cuda error (run with NCCL_DEBUG=INFO for details)'",
            "AssertionError: Impossible to fit in any sequence in kvCache", "Impossibe to fit in any sequence",
            "Cannot find an appropriate cached snapshot folder for the specified revision",
            "RuntimeError: No weight files found in /x.", "[TensorRT-LLM][INFO] [MemUsageChange] Allocated 64.80 GiB for max "
            "tokens in paged KV cache (528384).", "\rLoading safetensors weights in parallel:   0%|          | 0/4"
            "\rLoading safetensors weights in parallel:  75%|███████▌  | 3/4", "Loading bin weights in parallel: 50%|",
            "Downloading model x from HuggingFace", "Prefetching 15.2GB checkpoint files", "Running torch.compile warmup",
            "Running piecewise CUDA graph warmup", "[Autotuner] Autotuning process starts", "unhandled cuda error NCCL",
            "x" * 1000 + " out of memory", "", "   ", "INFO:     Application startup complete.",
            "iter = 154, global_rank = 0, rank = 0, currank_total_requests = 3/3, host_step_time = 9.8ms, "
            "prev_device_step_time = N/A, timestamp = 2026-10-04 18:20:34, num_scheduled_requests: 3, states = "
            "{'num_ctx_requests': 0, 'num_ctx_tokens': 0, 'num_generation_tokens': 3}",
            "Specified max_seq_len=262144 is larger than what the model can support (131072)",
            "Adjusted max_seq_len to 98304", "max_seq_len is not specified, using inferred value 131072",
            "Use 1.5.5 GB for model weights", "Model init total -- 21.34s", "Model init total -- .s",
            "max_seq_len is not specified, using inferred value " + "7" * 4400],
        "llamacpp": [
            "0.01.0 I load_tensors: offloaded 0/33 layers to GPU", "main: model loaded", "srv  update_slots: all slots are idle",
            "llama_context: n_seq_max     = 4", "........................................", "." * 120, "..", "." * 9 + "x",
            "warning: failed to mlock 8192-byte buffer", "common_fit_params: failed to fit params to free device memory: no",
            "common_fit_params: context size reduced from 8192 to 4096", "\x1b[0;32mmain: model loaded\x1b[0m",
            "alloc_tensor_range: failed to allocate CUDA0 buffer of size 105226698752", "CUDA error: out of memory",
            "srv  operator(): couldn't bind HTTP server socket", "E failed to load model " + "x" * 1000,
            "llama_kv_cache: size = 3072.00 MiB ( 65536 cells,  48 layers,  4/4 seqs)",
            "llama_kv_cache:      CUDA0 KV buffer size =  3072.00 MiB", "load_tensors:        CUDA0 model buffer size = 17408.00 MiB",
            "ggml_cuda_init: found 2 CUDA devices:", "  Device 1: NVIDIA X, compute capability 12.0, VMM: yes, VRAM: 97887 MiB",
            "srv  load_model: initializing slots, n_slots = 4", "slot   load_model: id  0 | task -1 | new slot, n_ctx_slot = 1",
            "n_parallel is set to auto, using n_parallel = 4 and kv_unified = false", "llama_context: n_ctx         = 65536",
            "print_info: n_ctx_train      = 262144", "common_fit_params: projected to use 21000 MiB of device memory vs. "
            "90000 MiB of free device memory", "build 10156 (91f8c9c5f) with cc", "server is entering sleeping state",
            "server is exiting sleeping state", "listening on http://0.0.0.0:8080", "   0.00.007.493 I   ", "", "model loaded  ",
            "1.23.456.789 W srv x: model loaded", "exiting due to model loading error",
            "print_info: n_ctx_train      = " + "8" * 4400, "load_tensors: CUDA0 model buffer size = 1.2.3 MiB"],
        "ollama": [
            'time=x level=INFO source=types.go:50 msg="inference compute" id=GPU-aaaa library=CUDA compute=12.0 '
            'name=CUDA0 total="95.0 GiB" available="94.0 GiB"',
            'time=x level=INFO msg="vram-based default context" total_vram="512 MiB" default_num_ctx=4096',
            'time=x level=INFO msg="vram-based default context" total_vram="2 TiB" default_num_ctx=262144',
            'time=x level=INFO msg="vram-based default context" total_vram="7 B" default_num_ctx=1',
            'time=x level=INFO msg="vram-based default context" total_vram="x" default_num_ctx=1',
            'time=x level=INFO msg="inference compute" id=cpu library=cpu total="1 KiB" available=7',
            "load_tensors: offloaded 33/33 layers to GPU", 'time=x level=INFO msg="llama runner started in 4.52 seconds"',
            'time=x level=INFO msg="Ollama cloud disabled: false"', 'time=x level=INFO msg=Listening on 1:2 (version 3)',
            'time=x level=ERROR source=sched.go:1 msg="cudaMalloc failed: out of memory"',
            "Error: listen tcp 0.0.0.0:11434: bind: address already in use", 'time=x level=ERROR msg="something broke"',
            "Error: unexpected failure", "Error: " + "x" * 1000, "plain text", 'time=x level=WARN msg="slow"',
            'msg="a \\"quoted\\" text" level=INFO', 'level=ERROR', 'msg=inference compute', 'msg="inference compute"',
            '[GIN] 2026/10/04 - 13:53:18 | 200 | 139.641µs | 127.0.0.1 | GET "/api/version"', "", "   ",
            'time=x msg="inference compute" id=g library=CUDA total="1.2.3 GiB" available="1 GiB"',
            "load_tensors: offloaded " + "4" * 4400 + "/33 layers to GPU"],
    }
    for eid, files in sources.items():
        a = ADAPTERS[eid]
        lines: list[str] = []
        for f in files:
            lines += f.read_text(encoding="utf-8").replace("\r", "\n").splitlines()
        lines += extra_lines[eid]
        base = list(lines)
        for _ in range(600):
            lines.append(mutate(rng, rng.choice(base)))
        log = [{"line": ln, "out": call(a.parse_startup_log, ln)} for ln in lines]
        out[f"{eid}_log"] = log
    # metrics
    trt_stats = json.loads((FIX / "trtllm/fixtures/iter_stats_1.2.1.json").read_text(encoding="utf-8"))
    prom = (FIX / "trtllm/fixtures/collector_1.2.1.prom").read_text(encoding="utf-8")
    trt_texts = ["", "   ", "[]", "{}", "null", "42", '"x"', "not json", "[1, 2]", "# HELP x\n", "process_cpu_seconds_total 3\n",
                 '[{"numActiveRequests": "many", "kvCacheStats": {"maxNumBlocks": 0}}]',
                 '[{"numActiveRequests": NaN, "numQueuedRequests": 1}]', prom, prom[: len(prom) // 2],
                 '[{"numActiveRequests": 1' + "0" * 400 + '}]', '[{"numQueuedRequests": ' + "1" * 4400 + "}]",
                 'trtllm_a{l="q\\"x"} 1.5\nbad line\ntrtllm_b 2 1712345\ntrtllm_c{} NaN\n']
    plain, spec = trt_stats["plain"], trt_stats["with_spec_and_requests"]
    for _ in range(220):
        objs = []
        for _ in range(rng.choice([1, 1, 2, 3, 5])):
            o = json.loads(json.dumps(rng.choice([plain, spec])))
            if rng.random() < 0.5:
                o["iterLatencyMS"] = rng.choice([0, 0.5, 12.5, 20.0, 40.0, -1, None, "x", 1e-3])
            if rng.random() < 0.4:
                o["inflightBatchingStats"]["numContextRequests"] = rng.choice([0, 1, None])
                o["inflightBatchingStats"]["numGenRequests"] = rng.choice([0, 1, 4, 2.5, True])
            if rng.random() < 0.3:
                o["timestamp"] = rng.choice(["10-04-2026 09:00:00.000000", "10-04-2026 09:00:01.5", "1-4-2026 9:0:1.25",
                                             "13-04-2026 09:00:00.0", "02-30-2026 09:00:00.0", "not a time", 5,
                                             "10-04-2026 09:00:00.0000001", "10-04-2026  09:00:00.1", "12-31-2026 23:59:59.999999",
                                             "01-01-2027 00:00:00.000000"])
            if rng.random() < 0.3:
                o["kvCacheStats"] = rng.choice([None, {}, {"usedNumBlocks": 5, "maxNumBlocks": 4},
                                                {"reusedBlocks": 4, "missedBlocks": 3121},
                                                {"usedNumBlocks": 1, "maxNumBlocks": 10, "cacheHitRate": 1.7},
                                                {"reusedBlocks": 0, "missedBlocks": 0, "cacheHitRate": None}])
            if rng.random() < 0.3:
                o["specDecodingStats"] = rng.choice([None, {}, {"numDraftTokens": 8, "numAcceptedTokens": 5},
                                                     {"numDraftTokens": 0}, {"numDraftTokens": 3, "numAcceptedTokens": None}])
            if rng.random() < 0.2:
                o["numActiveRequests"] = rng.choice([None, True, 7, 2.5, "3"])
            objs.append(o)
        text = json.dumps(objs, separators=rng.choice([(",", ":"), (", ", ": ")]))
        r = rng.random()
        if r < 0.2:
            text = text[: rng.randrange(1, len(text))]
        elif r < 0.3:
            text = "".join(json.dumps(o) for o in objs)
        elif r < 0.35:
            text = json.dumps(objs[0])
        trt_texts.append(text)
    for _ in range(150):
        lines = []
        for _ in range(rng.randrange(1, 12)):
            name = rng.choice(["trtllm_time_to_first_token_seconds", "trtllm_e2e_request_latency_seconds",
                               "trtllm_time_per_output_token_seconds", "trtllm_request_success_total", "other_x"])
            le = rng.choice(["0.1", "0.5", "1.0", "+Inf", "Inf", "x", "2.5", "1e1"])
            val = rng.choice(["0", "1", "3", "10", "NaN", "1e3", "2.5", "+Inf", "1_0", "x"])
            lab = rng.choice([f'le="{le}"', f'model="m",le="{le}"', ""])
            lines.append(f"{name}_bucket{{{lab}}} {val}" if rng.random() < 0.8 else f"{name}_sum {val}")
        trt_texts.append("\n".join(lines) + "\n")
    out["trtllm_metrics"] = [{"text": t, "out": call(ADAPTERS["trtllm"].parse_metrics, t)} for t in trt_texts]

    llama_prom = (FIX / "llamacpp/fixtures/metrics_synthetic.prom").read_text(encoding="utf-8")
    llama_texts = ["", llama_prom, "llamacpp:kv_cache_usage_ratio 0.25\nllamacpp:kv_cache_tokens 1024\n",
                   "process_cpu_seconds_total 3\nvllm:num_requests_running 2\n", "llamacpp:requests_deferred 7\n",
                   "# HELP llamacpp:requests_processing x\n", '# HELP x\nllamacpp:a 1.5\nbad line\nllamacpp:b 2 1712345\n'
                   'llamacpp:c NaN\nllamacpp:d{l="q\\"x"} 4\nllamacpp:e abc\n']
    names = ["requests_processing", "requests_deferred", "prompt_tokens_total", "prompt_tokens", "tokens_predicted_total",
             "tokens_predicted", "prompt_tokens_seconds", "predicted_tokens_seconds", "kv_cache_usage_ratio",
             "kv_cache_usage_ratio_total", "n_decode_total", "other"]
    for _ in range(250):
        lines = []
        for _ in range(rng.randrange(1, 10)):
            n = rng.choice(names)
            lab = rng.choice(["", '{slot="0"}', '{model="a"}', "{}"])
            val = rng.choice(["0", "1", "2.5", "NaN", "+Inf", "-Inf", "1e3", "x", "1_000", "٣", " 7"])
            pre = rng.choice(["llamacpp:", "llamacpp:", "vllm:", ""])
            lines.append(f"{pre}{n}{lab} {val}" + rng.choice(["", " 1712345", "  "]))
        llama_texts.append("\n".join(lines))
    out["llamacpp_metrics"] = [{"text": t, "out": call(ADAPTERS["llamacpp"].parse_metrics, t)} for t in llama_texts]
    write("parse_cases.json", out)


# ------------------------------------------------------------------ python behaviours
def py_cases() -> None:
    rng = random.Random("ws-a2-py")
    loads: list[Any] = []
    seeds = ['', ' ', '{', '{not json', '{"a"', '{"a" 1}', '{"a":', '{"a":1', '{"a":1,', '{"a":1,}', '[1,]', '[1', '[1 2]',
             '"abc', '"a\x01"', '"\\x"', '"\\u12"', 'nul', 'tru', '{"a":1}x', '{} {}', '﻿{}', '01', '-', '1.', '1e',
             'NaN', '-Infinity', '[NaN]', '{"a": Infinity}', '{1:2}', "{'a':1}", '[', ']', '{"a":1 "b":2}', '[1,,2]',
             '{,}', '{"a":1,,}', '[-]', '"\t"', '1e400', '[1e400]', '  {"a":1}  ', '{"a":1}\n', ' {}', '{"a":-0}',
             '-0', '-0.0', '1E5', '123456789012345678901234567890', '"\\', '"\\u', '"\\ud800\\u12"', '"\\ud800\\uzzzz"',
             '"\\ud800\\udc00"', '[1,]  ', '{"a":1 ,}', '[ ]', '{ }', '[1,\n]', '{"a"  :  1  ,  "b":2}', '{"a":1}}', '[}',
             '{]', '{"a":]', '"\\/"', '"\x7f"', '" "', '1.5e+3', '0.0e-1', '1e+', '00', '2.', '.5', '+1', 'true1',
             'nullx', 'Infinityx', '-NaN', '- 1', 'Nan', '{"a":1,"a":2,"b":3}', '{"k": "\\u00e9 é \\n \\" \\\\"}',
             '[true, false, null, 1, -2.5e-3, "s"]', '{"a": {"b": {"c": [1, [2, [3]]]}}}', '[1e-400]', '[-1e400]',
             '{"a" :1}', '\t\n\r {"a":1}', '\x0c{}', '[1,\t2]', '"\\b\\f\\n\\r\\t"', '"\\uD83D\\uDE00"', '[0e0, 0E+0]',
             '[' + "1" * 4400 + ']', '{"a": ' + "2" * 4301 + '}']
    alphabet = '{}[]:,"\\ 0123456789.eE+-abtrufnlsNIyé\x01'
    for s in seeds:
        loads.append(s)
    for _ in range(1500):
        base = rng.choice(seeds[60:] + ['{"decoding_type": "MTP", "n": 2}', '{"video": {"num_frames": 8}}'])
        s = list(base)
        for _ in range(rng.randrange(1, 4)):
            op = rng.randrange(3)
            if op == 0 and s:
                del s[rng.randrange(len(s))]
            elif op == 1:
                s.insert(rng.randrange(len(s) + 1), rng.choice(alphabet))
            elif s:
                s[rng.randrange(len(s))] = rng.choice(alphabet)
        loads.append("".join(s))

    def load_case(s: str) -> Any:
        try:
            v = json.loads(s)
        except json.JSONDecodeError as e:
            return {"err": e.msg}
        except Exception as e:  # noqa: BLE001
            return {"$raise": type(e).__name__}
        if _has_surrogate(v):
            return {"surrogate": True}
        return {"ok": enc(v), "dumps": _dumps_or_none(v)}

    py_loads = [{"s": s, "out": load_case(s)} for s in dict.fromkeys(loads)]

    trees: list[Any] = []
    words = ["a", "b", "key", "true", "True", "yes", "no", "on", "off", "null", "~", "", " ", "-", "- x", "? x", ":", "a: b",
             "#c", "a #c", "a#c", "x\ny", "\n", "  lead", "trail  ", "é", "模型", "\x01", "\x85", "'q'", '"q"', "1", "01",
             "1.5", "1e3", ".5", "0x1F", "0o17", "0b10", "1_000", "+1", "-1", ".inf", ".nan", "2026-10-04",
             "2026-10-4 1:02:03", "<<", "=", "!tag", "&a", "*a", "%x", "@x", "`x", "|", ">", "a,b", "[a]", "{a}", "---",
             "...", "x" * 90, " ".join(["word"] * 30), "\t", "a\tb", "line one\nline two\n", "ends with space ",
             "﻿", " ", "퟿", "", "🙂", "a" * 130, "k " * 70, "with 'single' quote", "back\\slash",
             "multi\n\nbreak", " \n", "x\n ", "-x", "/config/extra-llm-api-options.yaml", "GUARANTEED_NO_EVICT",
             "example-org/draft-model-1B", "http://otel:4317", "10:30", "1:2:3", "12:00:00", "y", "n", "Y", "N", "NULL"]

    def rand_tree(depth: int) -> Any:
        r = rng.random()
        if depth > 3 or r < 0.45:
            return rng.choice([rng.choice(words), rng.choice(words), rng.randrange(-5, 10 ** 6), 10 ** 25,
                               rng.choice([0.1, 1.5, 1e16, 1e17, 1e-5, 2.5e-7, 123456789.125, -0.0, 0.0, 1e300,
                                           float("inf"), float("-inf"), 3.0, 1e22]),
                               True, False, None])
        if r < 0.7:
            return [rand_tree(depth + 1) for _ in range(rng.randrange(0, 4))]
        return {rng.choice(words): rand_tree(depth + 1) for _ in range(rng.randrange(0, 5))}

    for _ in range(1200):
        t = {rng.choice(words): rand_tree(1) for _ in range(rng.randrange(1, 6))}
        trees.append({"tree": enc(t), "text": yaml.safe_dump(t, sort_keys=True, default_flow_style=False)})
    for w in words:
        t = {w: w, "k": [w, {w: [w]}], "n": {"m": w}}
        trees.append({"tree": enc(t), "text": yaml.safe_dump(t, sort_keys=True, default_flow_style=False)})

    floats = ["1", "1.5", "-2", "+3", "1e3", "1E-3", "NaN", "nan", "-nan", "+Inf", "Inf", "inf", "-inf", "infinity",
              "-Infinity", "1_000", "1__0", "_1", "1_", "1_0.5", "0x10", "", " ", " 7 ", " 7", "٣", "١٫٥", "1.5.5",
              "abc", ".5", "5.", "1e", "1e+", "0001", "1e400", "-0", "1_0e1_0", "²", "１２"]
    py_float = []
    for s in floats:
        try:
            py_float.append({"s": s, "out": enc(float(s))})
        except ValueError:
            py_float.append({"s": s, "out": None})

    digits = ["12", "0", "", "٣", "١٢", "²", "①", "a", "1a", " 1", "𝟎𝟏", "০৯", "１２", "-1", "1.5", "߀"]
    isdigit = []
    for s in digits:
        try:
            iv = int(s)
        except ValueError:
            iv = None
        isdigit.append({"s": s, "isdigit": s.isdigit(), "int": iv})
    nd_zeros = sorted({cp for cp in range(0x110000)
                       if unicodedata.category(chr(cp)) == "Nd" and unicodedata.decimal(chr(cp), -1) == 0})

    stamps = ["10-04-2026 09:00:00.000000", "10-04-2026 09:00:01.5", "1-4-2026 9:0:1.25", "13-04-2026 09:00:00.0",
              "02-30-2026 09:00:00.0", "02-29-2028 09:00:00.0", "02-29-2026 09:00:00.0", "not a time",
              "10-04-2026 09:00:00.0000001", "10-04-2026  09:00:00.1", "12-31-2026 23:59:59.999999", "01-01-0000 00:00:00.0",
              "10-04-2026 09:00:60.0", "10-04-2026 24:00:00.0", "10-04-2026 09:00:00", "10-04-26 09:00:00.0",
              " 1-04-2026 09:00:00.0", "10- 4-2026 09:00:00.0", "10-04-2026T09:00:00.0", "10-04-2026\t09:00:00.0",
              "10-04-2026 09:00:00.0 ", "١٠-04-2026 09:00:00.0", "10-04-2026 09:00:00.", "10-04-2026 9:5:7.000001"]
    strptime = []
    epoch = datetime(2000, 1, 1)
    for s in stamps:
        try:
            d = datetime.strptime(s, "%m-%d-%Y %H:%M:%S.%f")
            strptime.append({"s": s, "us": round((d - epoch).total_seconds() * 1e6)})
        except ValueError:
            strptime.append({"s": s, "us": None})
    write("py_cases.json", {"json_loads": py_loads, "yaml": trees, "float": py_float, "isdigit": isdigit,
                            "nd_zeros": nd_zeros, "strptime": strptime})


def _has_surrogate(v: Any) -> bool:
    if isinstance(v, str):
        return any(0xD800 <= ord(c) <= 0xDFFF for c in v)
    if isinstance(v, list):
        return any(_has_surrogate(x) for x in v)
    if isinstance(v, dict):
        return any(_has_surrogate(k) or _has_surrogate(x) for k, x in v.items())
    return False


def _dumps_or_none(v: Any) -> Any:
    return {"compact": json.dumps(v, separators=(",", ":")),
            "strict_ok": _strict_ok(v)}


def _strict_ok(v: Any) -> bool:
    try:
        json.dumps(v, allow_nan=False)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    catalogs()
    for engine in ADAPTERS:
        engine_cases(engine)
    parse_cases()
    py_cases()
