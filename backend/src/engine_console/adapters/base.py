"""Engine port — the single seam between the engine-agnostic core and a concrete engine.

The core (lifecycle, fit estimator, API, UI) depends ONLY on this module. Adding an engine
means implementing `EngineAdapter` and registering it in `adapters/__init__.py`; no core or
frontend change is required because the UI renders parameter forms from `param_catalog()`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

ParamType = Literal["int", "float", "bool", "enum", "string", "string_list", "json"]
ParamGroup = Literal["memory", "parallelism", "context", "quantization", "scheduling",
                     "tools_reasoning", "performance", "speculative", "network", "advanced"]
# Contract v2 (docs/ARCHITECTURE.md §10): additive only. Model formats an engine can load:
#   hf     a Hugging Face repo in an HF cache root (safetensors + config.json), referenced by repo id
#   gguf   a .gguf file (or the first shard of a split set) in a model root
#   ollama a model inside an Ollama server's own store (name:tag)
#   plain  a model directory (config.json + weights) outside any HF cache
ModelFormat = Literal["hf", "gguf", "ollama", "plain"]
LearnLevel = Literal["basic", "advanced", "expert"]


class Explain(BaseModel):
    """Plain-language help for one parameter, metric or concept ("learn mode", contract v2). Written for someone
    who has never run an inference engine. Authored in learn/*.json, never in code; see learn/README.md."""
    summary: str                     # one plain sentence, <= 160 chars: what it is
    details: str = ""                # <= 900 chars: how it works, with an everyday analogy where it helps
    when_to_change: str = ""         # <= 400 chars: when a beginner should touch it (or "leave it alone unless...")
    risks: str = ""                  # <= 400 chars: what goes wrong if it is set badly
    example: str = ""                # <= 200 chars: a concrete value and its effect
    terms: list[str] = []            # glossary keys the text relies on (each must exist in learn/glossary.json)
    level: LearnLevel = "advanced"


class ParamSpec(BaseModel):
    """One tunable engine parameter, described enough to render a form + validate + emit a CLI flag."""
    key: str                       # stable id used in profiles/API, e.g. "max_model_len"
    flag: str                      # CLI flag, e.g. "--max-model-len"
    label: str
    help: str                      # one-to-two sentences; shown as tooltip
    type: ParamType
    group: ParamGroup
    default: Any = None            # engine default (None = flag omitted unless user sets it)
    choices: list[str] | None = None
    min: float | None = None
    max: float | None = None
    advanced: bool = False         # hidden behind "Show advanced"
    affects_memory: bool = False   # UI re-runs the fit estimate when this changes
    requires_restart: bool = True
    bool_style: Literal["flag", "value"] = "flag"   # "flag": --x when true; "value": --x true|false
    docs_url: str | None = None
    # --- contract v2 (additive) ---
    explain: Explain | None = None   # merged in from learn/params/<engine>.json when catalogs are served
    level: LearnLevel | None = None  # None: "advanced" if `advanced` else "basic"
    unit: str | None = None          # display unit: "GiB", "tokens", "%", "s", "requests"


class ModelInfo(BaseModel):
    """Facts about a HF model needed for fit estimation + compatibility rules (built by core from the HF API)."""
    repo_id: str
    revision: str | None = None
    architectures: list[str] = []
    model_type: str | None = None
    num_params: int | None = None                 # total parameters
    num_active_params: int | None = None          # MoE active params, if known
    is_moe: bool = False
    weight_bytes: int | None = None               # sum of safetensors on disk (what must fit in VRAM)
    quantization: str | None = None               # e.g. "fp8", "nvfp4", "awq", "gptq", "mxfp4", None
    dtype: str | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    num_attention_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    max_position_embeddings: int | None = None
    sliding_window: int | None = None
    kv_lora_rank: int | None = None               # MLA models (DeepSeek-style) if present
    gated: bool = False
    license: str | None = None
    pipeline_tag: str | None = None
    raw_config: dict[str, Any] = Field(default_factory=dict)


class Hardware(BaseModel):
    gpu_ids: list[int]             # indices the instance may use
    gpu_uuids: list[str]
    gpu_names: list[str]
    gpu_total_gib: list[float]
    gpu_free_gib: list[float]
    compute_capability: str        # "12.0"
    host_ram_gib: float
    # --- contract v2 (additive): which machine, and what kind ---
    host_id: str = "local"
    arch: str = "x86_64"           # platform.machine() of the host: "x86_64" | "aarch64"
    unified_memory: bool = False   # GPU memory IS host RAM (e.g. GB10): VRAM and RAM budgets are the same pool


class LaunchSpec(BaseModel):
    """What the lifecycle service needs to start an engine container. Produced by the adapter."""
    image: str
    argv: list[str]                # engine CLI args AFTER the image entrypoint (no shell)
    env: dict[str, str] = {}
    container_port: int
    health_path: str
    metrics_path: str
    models_path: str = "/v1/models"
    # Container-level flags argv can't carry. Defaults suit both engines (NCCL/TP needs host IPC + big shm).
    shm_size: str = "16g"
    ipc_host: bool = True
    # --- contract v2 (additive) ---
    files: dict[str, str] = {}     # container path -> content: generated config files, written by the node, mounted ro
    needs_hf_cache: bool = True    # mount the host's HF cache root at hf_cache_container_path (read-only)


class LaunchContext(BaseModel):
    """Contract v2: everything an adapter needs to build one launch. The core resolves the model first:
    - format "hf": `model` is the repo id; the HF cache root is mounted at `hf_cache_container_path`.
    - format "gguf": the directory holding the file (all shards of a split set) is mounted read-only at
      MODEL_MOUNT; `model_path` is MODEL_MOUNT + "/" + the file name (the first shard for split sets).
    - format "plain": the model directory is mounted read-only at MODEL_MOUNT; `model_path` == MODEL_MOUNT.
    - format "ollama": nothing is mounted by the core; `model` is the Ollama name:tag (or "" for a bare server)."""
    model: str
    model_format: ModelFormat = "hf"
    model_path: str | None = None
    served_name: str
    hf_cache_container_path: str
    params: dict[str, Any] = {}
    hw: Hardware


MODEL_MOUNT = "/models"


class Compat(BaseModel):
    level: Literal["ok", "warn", "block"]
    code: str                      # machine id, e.g. "nvfp4_moe_sm120"
    message: str                   # human sentence with the fix ("use FP8 or SGLang")


class EngineAdapter(ABC):
    id: str                        # "vllm" | "sglang"
    display_name: str
    default_image: str             # pinned tag (never :latest)
    default_port: int = 8000
    # --- contract v2 (additive; defaults keep v1 adapters working unchanged) ---
    model_formats: frozenset[str] = frozenset({"hf"})   # ModelFormat values this engine loads
    images_by_arch: dict[str, str] = {}                  # e.g. {"aarch64": "<arm64 image>"}; default_image otherwise
    multi_model: bool = False                            # one server, many models loaded on demand (Ollama)

    def image_for(self, arch: str) -> str:
        """The pinned default image for a host architecture ("x86_64" | "aarch64")."""
        return self.images_by_arch.get(arch, self.default_image)

    def build_launch_ctx(self, ctx: LaunchContext) -> LaunchSpec:
        """Contract v2 entry point. v1 adapters inherit this bridge to `build_launch`; adapters for engines that
        load gguf/plain/ollama models override it (and may implement `build_launch` by building a context)."""
        return self.build_launch(ctx.model, ctx.params, ctx.hw, served_name=ctx.served_name,
                                 hf_cache_container_path=ctx.hf_cache_container_path)

    def preset_info(self) -> dict[str, Explain]:
        """Optional: a plain-language description per preset name (what it is for, what it trades off)."""
        return {}

    @abstractmethod
    def param_catalog(self) -> list[ParamSpec]: ...

    @abstractmethod
    def presets(self) -> dict[str, dict[str, Any]]:
        """Named param bundles: 'balanced', 'max-throughput', 'low-latency', 'long-context', 'tool-agent'."""

    @abstractmethod
    def validate(self, params: dict[str, Any]) -> list[str]:
        """Return human-readable errors (empty = valid). Unknown keys are errors."""

    @abstractmethod
    def build_launch(self, model: str, params: dict[str, Any], hw: Hardware, *,
                     served_name: str, hf_cache_container_path: str) -> LaunchSpec: ...

    @abstractmethod
    def compatibility(self, model: ModelInfo, params: dict[str, Any], hw: Hardware) -> list[Compat]:
        """Engine/hardware rules (quantisation/kernel support per GPU generation), e.g. NVFP4-MoE on sm_120."""

    @abstractmethod
    def memory_model(self, model: ModelInfo, params: dict[str, Any]) -> dict[str, float]:
        """Engine-specific knobs for the shared fit estimator, all optional:
        {'mem_fraction': 0.92, 'kv_bytes_per_elem': 2|1, 'overhead_gib': float,
         'max_len': int, 'max_seqs': int, 'tp': int, 'cuda_graph_gib': float}"""

    def host_memory_gib(self, model: ModelInfo, params: dict[str, Any], hw: Hardware) -> dict[str, float]:
        """Optional (non-abstract, default `{}`): host RAM the engine will pin/allocate on top of its baseline, in GiB,
        as {component: gib}. A `float('nan')` value means "needed but cannot be computed" (the fit reports
        `unknown` instead of guessing)."""
        return {}

    @abstractmethod
    def parse_metrics(self, prometheus_text: str) -> dict[str, float]:
        """Normalize engine Prometheus text to the canonical keys in docs/ARCHITECTURE.md §Metrics."""

    @abstractmethod
    def parse_startup_log(self, line: str) -> dict[str, Any] | None:
        """Extract structured progress from an engine log line, e.g.
        {'phase': 'loading_weights'|'compiling'|'capturing_graphs'|'ready', 'pct': float?, 'kv_cache_tokens': int?}"""
