"""Ollama parameter catalog: data only (the adapter lives in `ollama.py`).

Ollama is configured through environment variables, so every parameter here has the pseudo-flag ``env:NAME`` (the same
convention as `vllm_params.py`): the value is set as environment variable NAME in the container and never reaches argv.

Provenance. The variables, and what each one means, come from `ollama serve --help` of the pinned image (tag
`DEFAULT_IMAGE`, repository digest `IMAGE_DIGEST`), run in a throwaway container with no network and no GPU; the output is
committed as `tests/ollama/fixtures/ollama-serve-help.txt`. The help lists 21 variables:

    exposed here (17)   OLLAMA_CONTEXT_LENGTH  OLLAMA_NUM_PARALLEL  OLLAMA_MAX_LOADED_MODELS  OLLAMA_KEEP_ALIVE
                        OLLAMA_KV_CACHE_TYPE  OLLAMA_FLASH_ATTENTION  OLLAMA_GPU_OVERHEAD  OLLAMA_SCHED_SPREAD
                        OLLAMA_MAX_QUEUE  OLLAMA_LOAD_TIMEOUT  OLLAMA_ORIGINS  OLLAMA_LLM_LIBRARY  OLLAMA_IGPU_ENABLE
                        LLAMA_ARG_FIT  LLAMA_ARG_FIT_TARGET  OLLAMA_MAX_TRANSFER_STREAMS  OLLAMA_DEBUG
    owned by the adapter, never a parameter (`RESERVED_ENV`, 4)   OLLAMA_HOST  OLLAMA_MODELS  OLLAMA_NO_CLOUD  OLLAMA_NOPRUNE

Defaults were cross-checked against the "server config" log line a bare `ollama serve` prints at start-up (the effective
value of every variable; captured in `tests/ollama/fixtures/serve_bare_0.30.11.log`). The binary knows a few more
variables than the help lists (for example OLLAMA_VULKAN, OLLAMA_REMOTES, OLLAMA_NOHISTORY, OLLAMA_EDITOR); they are not
exposed. Accepted values that the help does not state (the three KV cache types, the keep-alive forms, the context
tiers by GPU memory) were read from the official docs pages named in `DOCS`.
"""
from __future__ import annotations

from typing import Any

from .base import LearnLevel, ParamGroup, ParamSpec, ParamType

VERSION = "0.30.11"
DEFAULT_IMAGE = f"ollama/ollama:{VERSION}"
# What the tag resolved to on the machine used for this catalog (image created 2026-06-26).
IMAGE_DIGEST = "sha256:c484b703176aa19dfc0a54cbfb60ab8094b38faa04283fb77eba1d33319e5eca"
CONTAINER_PORT = 11434
MODELS_DIR = "/root/.ollama/models"       # where the core mounts the host's Ollama models directory, read-only
DOCS = "https://docs.ollama.com/faq"
CONTEXT_DOCS = "https://docs.ollama.com/context-length"

KV_CACHE_TYPES = ["f16", "q8_0", "q4_0"]
# Bytes per KV element: ggml block layouts (a half-precision scale plus packed values per 32 elements), from memory of
# ggml's type table; the docs say q8_0 is about half and q4_0 about a quarter of f16.
KV_BYTES: dict[str, float] = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}

# bool params that are sent only when true. The pinned server echoes OLLAMA_IGPU_ENABLE as the raw string it was given
# ("0" and "false" show up verbatim in its "server config" line, unlike the real booleans), so there is no telling that
# a false value would be read as off; leaving the variable unset is the only safe "off".
OMIT_WHEN_FALSE = frozenset({"igpu_enable"})

# Variables the adapter sets itself on every launch, with the reason. They are in the help but are not parameters.
RESERVED_ENV: dict[str, str] = {
    "OLLAMA_HOST": "the listen address is fixed by the launch (container port 11434, published by the gateway)",
    "OLLAMA_MODELS": "the host's models directory is mounted read-only at a fixed path",
    "OLLAMA_NO_CLOUD": "always 1: cloud inference and web search are off (deny-by-default egress)",
    "OLLAMA_NOPRUNE": "always 1: the store is mounted read-only, so start-up must never try to prune blobs",
}


def _p(key: str, name: str, label: str, help: str, type: ParamType, group: ParamGroup, *,
       default: Any = None, choices: list[str] | None = None, min: float | None = None, adv: bool = False,
       mem: bool = False, unit: str | None = None, expert: bool = False, docs: str = DOCS) -> ParamSpec:
    level: LearnLevel | None = "expert" if expert else None
    return ParamSpec(key=key, flag=f"env:{name}", label=label, help=help, type=type, group=group, default=default,
                     choices=choices, min=min, advanced=adv or expert, affects_memory=mem, requires_restart=True,
                     docs_url=docs, level=level, unit=unit)


def build_catalog() -> list[ParamSpec]:
    c: list[ParamSpec] = []
    a = c.append
    # ---------------- context ----------------
    a(_p("context_length", "OLLAMA_CONTEXT_LENGTH", "Context length (tokens)",
         "Tokens of conversation memory for each request slot. 0 lets Ollama choose from the GPU memory: 4k below "
         "24 GiB, 32k from 24 GiB, 256k from 48 GiB.",
         "int", "context", default=0, min=0, mem=True, unit="tokens", docs=CONTEXT_DOCS))
    # ---------------- memory ----------------
    a(_p("keep_alive", "OLLAMA_KEEP_ALIVE", "Keep-alive time",
         "How long an idle model stays loaded: 5m, 24h, a number of seconds, 0 to unload at once, or -1 to keep it forever.",
         "string", "memory", default="5m"))
    a(_p("max_loaded_models", "OLLAMA_MAX_LOADED_MODELS", "Models kept loaded",
         "Most models that stay loaded at once, per GPU. 0 lets Ollama decide (3 per GPU, or 3 for CPU-only).",
         "int", "memory", default=0, min=0, adv=True, mem=True, unit="models"))
    a(_p("kv_cache_type", "OLLAMA_KV_CACHE_TYPE", "KV cache type",
         "Number format of the KV cache for every model: f16 (default), q8_0 (about half the memory) or q4_0 (about a quarter). "
         "The quantized types need flash attention.",
         "enum", "memory", default="f16", choices=KV_CACHE_TYPES, adv=True, mem=True))
    a(_p("gpu_overhead", "OLLAMA_GPU_OVERHEAD", "Reserved GPU memory (bytes per GPU)",
         "Memory Ollama leaves free on each GPU for other programs.",
         "int", "memory", default=0, min=0, adv=True, mem=True, unit="bytes"))
    a(_p("fit", "LLAMA_ARG_FIT", "llama.cpp automatic fit",
         "Let the llama.cpp engine inside Ollama shrink unset memory options so the model fits (on or off).",
         "enum", "memory", default="on", choices=["on", "off"], expert=True, mem=True))
    a(_p("fit_target", "LLAMA_ARG_FIT_TARGET", "Fit safety margin (MiB per GPU)",
         "Free GPU memory the llama.cpp automatic fit leaves over on each GPU.",
         "int", "memory", min=0, expert=True, mem=True, unit="MiB"))
    # ---------------- scheduling ----------------
    a(_p("num_parallel", "OLLAMA_NUM_PARALLEL", "Parallel requests per model",
         "How many requests one loaded model answers at the same time. The context memory is multiplied by this number.",
         "int", "scheduling", default=1, min=1, mem=True, unit="requests"))
    a(_p("max_queue", "OLLAMA_MAX_QUEUE", "Request queue length",
         "Most requests that may wait; further requests are refused with an error.",
         "int", "scheduling", default=512, min=1, adv=True, unit="requests"))
    a(_p("load_timeout", "OLLAMA_LOAD_TIMEOUT", "Model load timeout",
         "How long a model load may stall before Ollama gives up, for example 5m or 30m.",
         "string", "scheduling", default="5m", adv=True))
    # ---------------- parallelism ----------------
    a(_p("sched_spread", "OLLAMA_SCHED_SPREAD", "Spread models over all GPUs",
         "Always split a model across every GPU. By default Ollama uses as few GPUs as the model needs.",
         "bool", "parallelism", default=False, adv=True, mem=True))
    # ---------------- performance ----------------
    a(_p("flash_attention", "OLLAMA_FLASH_ATTENTION", "Flash attention",
         "Force flash attention on (true) or off (false). Unset lets Ollama decide for each model.",
         "bool", "performance", adv=True, mem=True))
    a(_p("llm_library", "OLLAMA_LLM_LIBRARY", "Compute library",
         "Choose the compute library instead of auto-detection. The pinned image has cuda_v12, cuda_v13 and vulkan.",
         "string", "performance", expert=True))
    a(_p("igpu_enable", "OLLAMA_IGPU_ENABLE", "Use integrated GPUs",
         "Let Ollama use integrated graphics as well. Only 'on' is sent; 'off' leaves the variable unset.",
         "bool", "performance", expert=True))
    # ---------------- network ----------------
    a(_p("origins", "OLLAMA_ORIGINS", "Extra browser origins",
         "Web origins allowed to call the API from a browser, added to the built-in list (localhost and a few desktop-app origins).",
         "string_list", "network", adv=True))
    a(_p("max_transfer_streams", "OLLAMA_MAX_TRANSFER_STREAMS", "Model transfer streams",
         "Parallel streams for pulling and pushing safetensors models. No effect here: the store is read-only and the engine is offline.",
         "int", "network", default=4, min=1, expert=True))
    # ---------------- advanced ----------------
    a(_p("debug", "OLLAMA_DEBUG", "Debug logging",
         "Show additional debug information in the engine log.",
         "bool", "advanced", default=False, adv=True))
    return c
