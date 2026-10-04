"""The inputs of the live parser check (fixtures/parser_acceptance_b10156.json).

For every parameter of the catalog (every enum choice, every bool both ways when it has a negation, boundary integers) and
for every preset, the adapter's real launch argv was given to the pinned `llama-server` in a throwaway container with no
network and no GPU and a model path that does not exist. The server parses its command line first and only then tries to
load the file, so reaching "loading model '...'" without an `error:` line proves the parser accepted every flag and value.
No model was loaded. `test_acceptance.py` checks that the adapter still produces exactly the recorded argv; if it does not,
repeat the check.
"""
from __future__ import annotations

from typing import Any

from engine_console.adapters.base import Hardware, LaunchContext
from engine_console.adapters.llamacpp import LlamaCppAdapter
from engine_console.adapters.llamacpp_params import NEGATABLE_KEYS

MODEL_PATH = "/models/does-not-exist.gguf"
TEMPLATE_TEXT = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
HW = Hardware(gpu_ids=[0], gpu_uuids=["GPU-aaaa"], gpu_names=["RTX PRO 6000 Blackwell"], gpu_total_gib=[95.6],
              gpu_free_gib=[90.0], compute_capability="12.0", host_ram_gib=246.0)

STRING_VALUES: dict[str, list[str]] = {
    "n_gpu_layers": ["all", "auto", "12"], "spec_draft_ngl": ["all", "auto", "12"], "tensor_split": ["3,1"],
    "device": ["none"], "override_tensor": ["exps=CPU"], "override_kv": ["tokenizer.ggml.add_bos_token=bool:false"],
    "mmproj": ["/models/mmproj.gguf"], "lora": ["/models/a.gguf,/models/b.gguf"], "lora_scaled": ["/models/a.gguf:0.5"],
    "spec_draft_model": ["/models/draft.gguf"], "chat_template": ["chatml", "llama3"],
    "chat_template_jinja": [TEMPLATE_TEXT], "cors_origins": ["localhost", "*", "https://chat.example.org"],
    "api_key": ["probe-key-not-a-secret"],
}
LIST_VALUES: dict[str, list[list[str]]] = {"served_model_name": [["extra-a", "extra-b"]]}
JSON_VALUES: dict[str, list[Any]] = {"chat_template_kwargs": [{"enable_thinking": False}]}


def values_for(spec: Any) -> list[Any]:
    """Representative values of one parameter: all choices, both bool states, the default and the minimum."""
    if spec.type == "bool":
        return [True, False] if spec.key in NEGATABLE_KEYS else [True]
    if spec.type == "enum":
        return list(spec.choices)
    if spec.type == "string":
        return STRING_VALUES[spec.key]
    if spec.type == "string_list":
        return LIST_VALUES[spec.key]
    if spec.type == "json":
        return JSON_VALUES[spec.key]
    base = spec.default if spec.default is not None else (spec.min if spec.min is not None else 1)
    out: list[Any] = [base]
    if spec.min is not None and spec.min != base:
        out.append(int(spec.min) if spec.type == "int" else spec.min)
    return out


# Negative controls: each of these must be REJECTED by the parser, which shows that the check can tell the difference
# (and that an `error:` line is printed even at verbosity 0, the one case whose "loading model" line is silenced).
NEGATIVE_CASES: list[tuple[str, list[str]]] = [
    ("unknown flag", ["-m", MODEL_PATH, "--no-such-flag"]),
    ("bad enum --fit", ["-m", MODEL_PATH, "--fit", "maybe"]),
    ("bad enum --cache-type-k", ["-m", MODEL_PATH, "--cache-type-k", "fp7"]),
    ("bad enum --split-mode", ["-m", MODEL_PATH, "--split-mode", "diagonal"]),
    ("bad enum --flash-attn", ["-m", MODEL_PATH, "--flash-attn", "maybe"]),
    ("bad enum --spec-type", ["-m", MODEL_PATH, "--spec-type", "magic"]),
    ("missing value", ["-m", MODEL_PATH, "--ctx-size"]),
    ("bad enum at verbosity 0", ["-m", MODEL_PATH, "--verbosity", "0", "--fit", "maybe"]),
]


def cases(adapter: LlamaCppAdapter) -> list[dict[str, Any]]:
    """name, argv and env of every case, exactly as the adapter builds them today."""
    out: list[dict[str, Any]] = []

    def add(name: str, params: dict[str, Any]) -> None:
        spec = adapter.build_launch_ctx(LaunchContext(model="m", model_format="gguf", model_path=MODEL_PATH, served_name="demo",
                                                      hf_cache_container_path="/hf", params=params, hw=HW))
        out.append({"name": name, "argv": spec.argv, "env": spec.env, "files": sorted(spec.files)})

    add("baseline", {})
    for p in adapter.param_catalog():
        for v in values_for(p):
            add(f"{p.key}={v!r}", {p.key: v})
    for name, params in adapter.presets().items():
        add(f"preset:{name}", params)
    return out
