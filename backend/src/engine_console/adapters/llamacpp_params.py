"""llama.cpp (`llama-server`) parameter catalog: data only, no logic (the adapter lives in `llamacpp.py`).

Provenance. Names, defaults and choices were read from `llama-server --help` of the pinned image (`DEFAULT_IMAGE`,
llama.cpp build `BUILD`), run in a throwaway container with no network and no GPU (this binary prints its help without
one). The captured text is committed as `tests/llamacpp/fixtures/llama-server-help.txt`, and the tests check every flag,
every negation (`--no-x`) and every literal default of this catalog against it. Log and metric strings were read from the
same image's shared libraries; the metric names were also compared with the server README at the same build tag.

What is deliberately NOT here:
  * `--host`, `--port`, `--alias` (the launch sets them), `--metrics` and `--offline` (always on), `-m/--model` (the core
    resolves it) and every model-download flag (`-hf`, `-mu`, `-dr`, ...): engines run offline.
  * `--tools`, `--agent`, `--mcp-servers-*`, `--ui-mcp-proxy`, `--props`, `--models-dir`/`--models-preset` (router mode),
    `--ssl-*` and `--api-key-file`. The first group lets a prompt run shell commands or reach other servers (the help
    itself says "do not enable in untrusted environments"); the rest change what the server exposes or how it is
    reached, which the console owns.
  * Deprecated spellings. `--mlock`, `--mmap/--no-mmap` and `--direct-io` are marked DEPRECATED in the pinned help in
    favour of `--load-mode`, so only `load_mode` exists here.
Param keys must not contain "key", "token", "secret" or "password" on a string-typed param: the core treats those as
secrets (see `services/secrets_store.py`). `api_key` is the one on purpose and travels in the environment
(`env:LLAMA_API_KEY`); the integer `image_*_tokens` params are fine because only string values are treated as secrets.

Pseudo-flag convention shared with the other adapters: ``env:NAME`` means the value is set as environment variable NAME
in the container (never put on argv).
"""
from __future__ import annotations

from typing import Any

from .base import LearnLevel, ParamGroup, ParamSpec, ParamType

BUILD = "b10156"
COMMIT = "91f8c9c5f"             # `llama-server --version` prints "version: 10156 (91f8c9c5f)"
IMAGE_REPO = "ghcr.io/ggml-org/llama.cpp"
# The upstream tag `server-cuda` moves with every release; the repository digest below is what that tag resolved to on
# the machine used for this catalog (image created 2026-07-28, CUDA 12.8.1, `sm_120a` kernels present in libggml-cuda).
IMAGE_DIGEST = "sha256:4c0ece468721077a8bff8c114f8b0a92a8604ed310c91939fadfe37595a53b9e"
DEFAULT_IMAGE = f"{IMAGE_REPO}@{IMAGE_DIGEST}"
DOCS_URL = f"https://github.com/ggml-org/llama.cpp/blob/{BUILD}/tools/server/README.md"
CONTAINER_PORT = 8080            # the image's own healthcheck probes this port
CHAT_TEMPLATE_PATH = "/etc/engine-console/chat-template.jinja"
MAX_TEMPLATE_BYTES = 120_000     # the node accepts at most 128 KiB per generated file

CACHE_TYPES = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
# Bytes per KV element. The 1-byte-and-below types are ggml block layouts (a half-precision scale plus the packed values
# per 32 elements), written from memory of ggml's type table and not measured here.
CACHE_TYPE_BYTES: dict[str, float] = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32, "iq4_nl": 18 / 32,
    "q5_0": 22 / 32, "q5_1": 24 / 32,
}
SPEC_TYPES = ["none", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash", "ngram-simple", "ngram-map-k",
              "ngram-map-k4v", "ngram-mod", "ngram-cache"]
LOAD_MODES = ["none", "mmap", "mlock", "mmap+mlock", "dio"]
# Built-in chat templates listed by the pinned help for --chat-template.
CHAT_TEMPLATES = ("bailing", "bailing-think", "bailing2", "chatglm3", "chatglm4", "chatml", "command-r", "deepseek",
                  "deepseek-ocr", "deepseek2", "deepseek3", "exaone-moe", "exaone3", "exaone4", "falcon3", "gemma",
                  "gigachat", "glmedge", "gpt-oss", "granite", "granite-4.0", "granite-4.1", "grok-2", "hunyuan-dense",
                  "hunyuan-moe", "hunyuan-vl", "kimi-k2", "llama2", "llama2-sys", "llama2-sys-bos", "llama2-sys-strip",
                  "llama3", "llama4", "megrez", "minicpm", "mistral-v1", "mistral-v3", "mistral-v3-tekken",
                  "mistral-v7", "mistral-v7-tekken", "monarch", "openchat", "orion", "pangu-embedded", "phi3", "phi4",
                  "rwkv-world", "seed_oss", "smolvlm", "solar-open", "vicuna", "vicuna-orca", "yandex", "zephyr")

# bool params: false is emitted as `--no-<flag>` (the pinned help lists every one of these negations). A bool outside
# this set emits nothing when false.
NEGATABLE_KEYS = frozenset({
    "kv_offload", "mmproj_offload", "context_shift", "kv_unified", "cont_batching", "cache_prompt", "cache_idle_slots",
    "warmup", "jinja", "reasoning_preserve", "skip_chat_parsing", "prefill_assistant", "ui", "slots",
})
# string params that hold a path INSIDE the container; the adapter only accepts files under the model mount.
PATH_KEYS = frozenset({"mmproj", "lora", "lora_scaled", "spec_draft_model"})
# keys whose value is "auto", "all" or a layer count
LAYER_KEYS = frozenset({"n_gpu_layers", "spec_draft_ngl"})


def _p(key: str, flag: str, label: str, help: str, type: ParamType, group: ParamGroup, *,
       default: Any = None, choices: list[str] | None = None, min: float | None = None, max: float | None = None,
       adv: bool = False, mem: bool = False, unit: str | None = None, expert: bool = False) -> ParamSpec:
    level: LearnLevel | None = "expert" if expert else None
    return ParamSpec(key=key, flag=flag, label=label, help=help, type=type, group=group, default=default,
                     choices=choices, min=min, max=max, advanced=adv or expert, affects_memory=mem,
                     requires_restart=True, docs_url=DOCS_URL, level=level, unit=unit)


def build_catalog() -> list[ParamSpec]:
    c: list[ParamSpec] = []
    a = c.append
    # ---------------- memory ----------------
    a(_p("n_gpu_layers", "--n-gpu-layers", "GPU layers",
         "How many of the model's layers to keep in GPU memory: a number, 'all', or 'auto' (the engine picks what fits). "
         "Layers that stay out run on the CPU, which is far slower.",
         "string", "memory", default="auto", mem=True))
    a(_p("fit", "--fit", "Automatic fit",
         "When on, start-up shrinks every memory setting you left unset (context size, GPU layers) until the model fits "
         "in the free GPU memory. Settings you give explicitly are never changed.",
         "enum", "memory", default="on", choices=["on", "off"], adv=True, mem=True))
    a(_p("fit_target", "--fit-target", "Fit safety margin (MiB per GPU)",
         "Free GPU memory the automatic fit must leave over on every GPU.",
         "int", "memory", default=1024, min=0, adv=True, mem=True, unit="MiB"))
    a(_p("fit_ctx", "--fit-ctx", "Smallest context the fit may pick",
         "The automatic fit will not shrink the context below this many tokens.",
         "int", "memory", default=4096, min=1, adv=True, mem=True, unit="tokens"))
    a(_p("kv_offload", "--kv-offload", "KV cache on the GPU",
         "Keep the KV cache (the running conversation memory) in GPU memory. Off moves it to host RAM, which saves "
         "GPU memory and slows every token.",
         "bool", "memory", default=True, adv=True, mem=True))
    a(_p("cache_type_k", "--cache-type-k", "KV cache type (keys)",
         "Number format of the key half of the KV cache. q8_0 uses about half the memory of f16 with little quality loss.",
         "enum", "memory", default="f16", choices=CACHE_TYPES, adv=True, mem=True))
    a(_p("cache_type_v", "--cache-type-v", "KV cache type (values)",
         "Number format of the value half of the KV cache. Anything other than f16, bf16 or f32 needs flash attention.",
         "enum", "memory", default="f16", choices=CACHE_TYPES, adv=True, mem=True))
    a(_p("swa_full", "--swa-full", "Full-size sliding-window cache",
         "Keep a full-length KV cache even for sliding-window layers. Uses more memory; helps prompt reuse on some models.",
         "bool", "memory", default=False, adv=True, mem=True))
    a(_p("cache_ram", "--cache-ram", "Prompt cache size (MiB of RAM)",
         "Host RAM the server may use to remember earlier prompts so repeated prefixes are not recomputed. "
         "-1 removes the limit and 0 turns the cache off.",
         "int", "memory", default=8192, min=-1, adv=True, mem=True, unit="MiB"))
    a(_p("load_mode", "--load-mode", "Model loading mode",
         "How the model file is read: mmap maps it (default), none reads it into RAM, mlock pins it in RAM, "
         "mmap+mlock does both, dio uses direct I/O.",
         "enum", "memory", default="mmap", choices=LOAD_MODES, adv=True, mem=True))
    a(_p("cpu_moe", "--cpu-moe", "All experts on the CPU",
         "Keep every mixture-of-experts weight in host RAM while attention and shared layers stay on the GPU. "
         "Lets a very large MoE model run on a small GPU.",
         "bool", "memory", default=False, adv=True, mem=True))
    a(_p("n_cpu_moe", "--n-cpu-moe", "Experts of the first N layers on the CPU",
         "Keep the expert weights of the first N layers in host RAM. A middle path between all on the GPU and --cpu-moe.",
         "int", "memory", min=0, adv=True, mem=True, unit="layers"))
    a(_p("override_tensor", "--override-tensor", "Tensor placement overrides",
         "Comma-separated pattern=buffer rules that force named tensors onto a chosen device, for example "
         "exps=CPU. Very detailed control; the engine warns that CPU overrides combine badly with mmap.",
         "string", "memory", expert=True, mem=True))
    a(_p("mmproj_offload", "--mmproj-offload", "Vision projector on the GPU",
         "Run the image (multimodal projector) part of a vision model on the GPU.",
         "bool", "memory", default=True, adv=True, mem=True))
    # ---------------- parallelism (GPU placement) ----------------
    a(_p("split_mode", "--split-mode", "Multi-GPU split mode",
         "How a model is spread over several GPUs: none (one GPU), layer (default, layers and KV split in a pipeline), "
         "row, or tensor (experimental).",
         "enum", "parallelism", default="layer", choices=["none", "layer", "row", "tensor"], adv=True, mem=True))
    a(_p("tensor_split", "--tensor-split", "GPU split proportions",
         "Share of the model each GPU gets, comma-separated, for example 3,1 puts three quarters on the first GPU.",
         "string", "parallelism", adv=True, mem=True))
    a(_p("main_gpu", "--main-gpu", "Main GPU",
         "Index of the GPU used when the split mode is none, and for intermediate results with the row split.",
         "int", "parallelism", default=0, min=0, adv=True, mem=True))
    a(_p("device", "--device", "Devices to use",
         "Comma-separated device names to offload to, or none to keep everything on the CPU. Run with --list-devices to see names.",
         "string", "parallelism", expert=True, mem=True))
    # ---------------- context ----------------
    a(_p("ctx_size", "--ctx-size", "Context size (tokens)",
         "Total tokens of conversation memory the server allocates; 0 uses the model's trained maximum. "
         "With several slots the budget is shared between them.",
         "int", "context", default=0, min=0, mem=True, unit="tokens"))
    a(_p("rope_scaling", "--rope-scaling", "RoPE scaling method",
         "How positions are stretched to reach a context longer than the model was trained on: none, linear or yarn.",
         "enum", "context", choices=["none", "linear", "yarn"], adv=True))
    a(_p("rope_scale", "--rope-scale", "RoPE scale factor",
         "Context expansion factor: 4 stretches a 32k-trained model towards 128k.",
         "float", "context", min=0.01, adv=True))
    a(_p("rope_freq_base", "--rope-freq-base", "RoPE base frequency",
         "Base frequency for NTK-aware scaling; the default is read from the model.",
         "float", "context", min=1, expert=True))
    a(_p("yarn_orig_ctx", "--yarn-orig-ctx", "YaRN original context",
         "The context length the model was trained on, for YaRN scaling; 0 uses the value stored in the model.",
         "int", "context", default=0, min=0, expert=True, unit="tokens"))
    a(_p("context_shift", "--context-shift", "Context shift",
         "When a conversation outgrows its slot, drop the oldest tokens and keep going instead of stopping.",
         "bool", "context", default=False, adv=True))
    a(_p("image_min_tokens", "--image-min-tokens", "Minimum tokens per image",
         "Lower bound of tokens one image takes, for vision models with dynamic resolution; the default is read from the model.",
         "int", "context", min=1, adv=True, mem=True, unit="tokens"))
    a(_p("image_max_tokens", "--image-max-tokens", "Maximum tokens per image",
         "Upper bound of tokens one image takes, for vision models with dynamic resolution; the default is read from the model.",
         "int", "context", min=1, adv=True, mem=True, unit="tokens"))
    # ---------------- model files (quantization group: loading) ----------------
    a(_p("mmproj", "--mmproj", "Vision projector file",
         "Path of the multimodal projector GGUF, inside the model folder (/models/<file>), for vision models.",
         "string", "quantization", adv=True, mem=True))
    a(_p("lora", "--lora", "LoRA adapters",
         "Comma-separated LoRA adapter GGUF files inside the model folder (/models/<file>), applied on top of the model.",
         "string", "quantization", adv=True, mem=True))
    a(_p("lora_scaled", "--lora-scaled", "LoRA adapters with scales",
         "Comma-separated file:scale pairs, for example /models/a.gguf:0.5, applied on top of the model.",
         "string", "quantization", adv=True, mem=True))
    a(_p("lora_init_without_apply", "--lora-init-without-apply", "Load LoRA without applying",
         "Load the adapters but leave them switched off until a request turns them on.",
         "bool", "quantization", default=False, expert=True))
    a(_p("override_kv", "--override-kv", "Override model metadata",
         "Comma-separated KEY=TYPE:VALUE entries (types int, float, bool, str) that replace metadata stored in the model file.",
         "string", "quantization", expert=True))
    # ---------------- scheduling ----------------
    a(_p("parallel", "--parallel", "Slots (parallel requests)",
         "Number of conversations served at the same time. -1 lets the engine pick (4 slots sharing one KV buffer).",
         "int", "scheduling", default=-1, min=-1, mem=True, unit="slots"))
    a(_p("kv_unified", "--kv-unified", "Unified KV buffer",
         "Let all slots share one KV buffer instead of splitting the context evenly; on by default only when the slot count is auto.",
         "bool", "scheduling", adv=True, mem=True))
    a(_p("cont_batching", "--cont-batching", "Continuous batching",
         "Let new requests join a batch that is already running instead of waiting for it to finish.",
         "bool", "scheduling", default=True, adv=True))
    a(_p("cache_prompt", "--cache-prompt", "Prompt caching",
         "Reuse the part of a slot's earlier prompt that matches the new request so only the new part is processed.",
         "bool", "scheduling", default=True, adv=True))
    a(_p("cache_reuse", "--cache-reuse", "Prompt reuse chunk size",
         "Minimum length of a matching chunk that may be reused from the cache by shifting it; 0 turns this off.",
         "int", "scheduling", default=0, min=0, adv=True, unit="tokens"))
    a(_p("cache_idle_slots", "--cache-idle-slots", "Save idle slots to the prompt cache",
         "When a new task starts, store the idle slots' conversations in the prompt cache and free them. Needs the prompt cache.",
         "bool", "scheduling", default=True, adv=True))
    a(_p("slot_prompt_similarity", "--slot-prompt-similarity", "Slot prompt similarity",
         "How closely a request must match a slot's earlier prompt for that slot to be reused; 0 disables the check.",
         "float", "scheduling", default=0.1, min=0, max=1, adv=True))
    a(_p("ctx_checkpoints", "--ctx-checkpoints", "Context checkpoints per slot",
         "Maximum snapshots of a conversation kept per slot, used to rewind models that cannot drop recent tokens.",
         "int", "scheduling", default=32, min=0, adv=True))
    a(_p("sleep_idle_seconds", "--sleep-idle-seconds", "Sleep after idle (s)",
         "Seconds without requests after which the server sleeps and frees the model; -1 turns it off.",
         "int", "scheduling", default=-1, min=-1, adv=True, unit="s"))
    a(_p("warmup", "--warmup", "Warm-up run",
         "Run one empty generation at start-up so the first real request is not slow.",
         "bool", "scheduling", default=True, adv=True))
    # ---------------- performance ----------------
    a(_p("flash_attn", "--flash-attn", "Flash attention",
         "Use the memory-efficient attention kernels: on, off, or auto (default; on whenever the GPU backend supports the model).",
         "enum", "performance", default="auto", choices=["on", "off", "auto"], mem=True))
    a(_p("threads", "--threads", "CPU threads",
         "Threads used while generating tokens on the CPU; -1 picks automatically. Only matters when layers run on the CPU.",
         "int", "performance", default=-1, min=-1, adv=True, unit="threads"))
    a(_p("threads_batch", "--threads-batch", "CPU threads (prompt processing)",
         "Threads used while processing prompts; by default the same as threads.",
         "int", "performance", min=-1, adv=True, unit="threads"))
    a(_p("batch_size", "--batch-size", "Logical batch size",
         "Largest number of tokens handed to the model in one step. Larger speeds up long prompts and uses more memory.",
         "int", "performance", default=2048, min=1, adv=True, mem=True, unit="tokens"))
    a(_p("ubatch_size", "--ubatch-size", "Physical batch size",
         "Largest number of tokens the GPU computes at once; sets the size of the compute buffers.",
         "int", "performance", default=512, min=1, adv=True, mem=True, unit="tokens"))
    a(_p("numa", "--numa", "NUMA strategy",
         "Placement hint for multi-socket CPUs: distribute, isolate or numactl. Leave unset on single-socket machines.",
         "enum", "performance", choices=["distribute", "isolate", "numactl"], expert=True))
    # ---------------- tools & reasoning ----------------
    a(_p("jinja", "--jinja", "Jinja chat templates",
         "Use the full chat template engine. Needed for tool calls and for splitting reasoning from the answer.",
         "bool", "tools_reasoning", default=True, adv=True))
    a(_p("chat_template", "--chat-template", "Built-in chat template",
         "Name of a built-in template (for example chatml or llama3) or an inline Jinja template; replaces the one stored in the model.",
         "string", "tools_reasoning", adv=True))
    a(_p("chat_template_jinja", "--chat-template-file", "Custom chat template (Jinja)",
         "Full Jinja template text. The console writes it to a file inside the container; it replaces the model's own template.",
         "string", "tools_reasoning", adv=True))
    a(_p("chat_template_kwargs", "--chat-template-kwargs", "Chat template arguments (JSON)",
         'Extra arguments handed to the template for every request, for example {"enable_thinking": false}.',
         "json", "tools_reasoning", adv=True))
    a(_p("reasoning", "--reasoning", "Reasoning mode",
         "Turn the model's thinking on, off, or auto (detect from the chat template).",
         "enum", "tools_reasoning", default="auto", choices=["on", "off", "auto"]))
    a(_p("reasoning_format", "--reasoning-format", "Reasoning output format",
         "Where thinking text goes: none leaves it in the answer, deepseek puts it in reasoning_content, "
         "deepseek-legacy does both.",
         "enum", "tools_reasoning", choices=["none", "deepseek", "deepseek-legacy"], adv=True))
    a(_p("reasoning_budget", "--reasoning-budget", "Reasoning budget (tokens)",
         "Cap on thinking tokens: -1 unlimited, 0 ends thinking immediately, a positive number is the budget.",
         "int", "tools_reasoning", default=-1, min=-1, adv=True, unit="tokens"))
    a(_p("reasoning_preserve", "--reasoning-preserve", "Keep reasoning in the history",
         "Keep earlier thinking in the conversation history for templates that support it, not only in the latest message.",
         "bool", "tools_reasoning", adv=True))
    a(_p("skip_chat_parsing", "--skip-chat-parsing", "Skip chat output parsing",
         "Return everything, including reasoning and tool calls, as plain content.",
         "bool", "tools_reasoning", default=False, expert=True))
    a(_p("prefill_assistant", "--prefill-assistant", "Prefill the assistant reply",
         "When the last message is from the assistant, continue it instead of treating it as a finished message.",
         "bool", "tools_reasoning", default=True, expert=True))
    # ---------------- speculative decoding ----------------
    a(_p("spec_type", "--spec-type", "Speculative decoding type",
         "How draft tokens are produced: none, a draft model (draft-simple, draft-eagle3, draft-dflash), the model's own "
         "MTP layers (draft-mtp), or n-gram lookup from the context.",
         "enum", "speculative", default="none", choices=SPEC_TYPES, adv=True, mem=True))
    a(_p("spec_draft_model", "--spec-draft-model", "Draft model file",
         "Path of the draft model GGUF, inside the model folder (/models/<file>).",
         "string", "speculative", adv=True, mem=True))
    a(_p("spec_draft_n_max", "--spec-draft-n-max", "Draft tokens per step (max)",
         "Most tokens the draft proposes before the main model checks them.",
         "int", "speculative", default=3, min=0, adv=True, unit="tokens"))
    a(_p("spec_draft_n_min", "--spec-draft-n-min", "Draft tokens per step (min)",
         "Fewest draft tokens worth checking; shorter drafts are skipped.",
         "int", "speculative", default=0, min=0, expert=True, unit="tokens"))
    a(_p("spec_draft_p_min", "--spec-draft-p-min", "Draft confidence floor",
         "Stop drafting when the draft model's confidence in the next token falls below this probability.",
         "float", "speculative", default=0.0, min=0, max=1, expert=True))
    a(_p("spec_draft_ngl", "--spec-draft-ngl", "Draft model GPU layers",
         "How many layers of the draft model go to the GPU: a number, 'all' or 'auto'.",
         "string", "speculative", default="auto", adv=True, mem=True))
    # ---------------- network ----------------
    a(_p("served_model_name", "--alias", "Extra model names",
         "Extra names the API answers to. The console always serves its own name first.",
         "string_list", "network", adv=True))
    a(_p("api_key", "env:LLAMA_API_KEY", "API key",
         "Require this key (sent as a Bearer token) on the API. Passed through the LLAMA_API_KEY environment variable, "
         "not the command line. The metrics page then needs the key too.",
         "string", "network"))
    a(_p("cors_origins", "--cors-origins", "Allowed browser origins",
         "Which web pages may call the API from a browser. The console sets localhost; the engine's own default (*) "
         "lets any website read answers from a loopback port.",
         "string", "network", default="localhost", adv=True))
    a(_p("ui", "--ui", "Built-in web page",
         "Serve llama.cpp's own chat page on the same port.",
         "bool", "network", default=True, adv=True))
    a(_p("slots", "--slots", "Slots monitoring endpoint",
         "Expose /slots, which lists what each slot is doing.",
         "bool", "network", default=True, adv=True))
    a(_p("timeout", "--timeout", "Server read/write timeout (s)",
         "Seconds the HTTP server waits on a stalled connection.",
         "int", "network", default=3600, min=1, adv=True, unit="s"))
    a(_p("log_verbosity", "--verbosity", "Log verbosity",
         "How much the server logs: 0 generic, 1 errors, 2 warnings, 3 info, 4 trace, 5 debug.",
         "int", "network", default=3, min=0, max=5, adv=True))
    # ---------------- advanced: server mode and sampling defaults ----------------
    a(_p("embedding", "--embedding", "Embeddings-only server",
         "Serve only the embeddings endpoint; use with a dedicated embedding model.",
         "bool", "advanced", default=False, adv=True))
    a(_p("pooling", "--pooling", "Embedding pooling",
         "How token vectors are combined into one embedding; the model's own default is used when unset.",
         "enum", "advanced", choices=["none", "mean", "cls", "last", "rank"], adv=True))
    a(_p("rerank", "--rerank", "Reranking endpoint",
         "Enable the reranking endpoint; use with a reranker model.",
         "bool", "advanced", default=False, adv=True))
    a(_p("temperature", "--temp", "Default temperature",
         "Randomness of sampling when a request does not set it: low is focused, high is varied.",
         "float", "advanced", default=0.8, min=0, adv=True))
    a(_p("top_k", "--top-k", "Default top-k",
         "Sample only from the k most likely tokens when a request does not set it; 0 turns it off.",
         "int", "advanced", default=40, min=0, adv=True))
    a(_p("top_p", "--top-p", "Default top-p",
         "Sample only from the smallest set of tokens whose probabilities add up to p; 1.0 turns it off.",
         "float", "advanced", default=0.95, min=0, max=1, adv=True))
    a(_p("min_p", "--min-p", "Default min-p",
         "Drop tokens less likely than p times the best token; 0 turns it off.",
         "float", "advanced", default=0.05, min=0, max=1, adv=True))
    a(_p("repeat_penalty", "--repeat-penalty", "Default repeat penalty",
         "Penalty for tokens that appeared in the recent text; 1.0 turns it off.",
         "float", "advanced", default=1.0, min=0, adv=True))
    a(_p("presence_penalty", "--presence-penalty", "Default presence penalty",
         "Flat penalty for any token that has already appeared; 0 turns it off.",
         "float", "advanced", default=0.0, adv=True))
    a(_p("frequency_penalty", "--frequency-penalty", "Default frequency penalty",
         "Penalty that grows with how often a token has already appeared; 0 turns it off.",
         "float", "advanced", default=0.0, adv=True))
    a(_p("seed", "--seed", "Random seed",
         "Fixed seed for repeatable sampling; -1 picks a random one.",
         "int", "advanced", default=-1, adv=True))
    return c
