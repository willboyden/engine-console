//! Python: backend/src/engine_console/adapters/llamacpp_params.py.
//!
//! llama.cpp (`llama-server`) parameter catalog: data only (the adapter lives in `llamacpp.rs`). Names, defaults and
//! choices were read from `llama-server --help` of the pinned image (build `BUILD`); the captured help is a test
//! fixture (tests/data/llamacpp/llama-server-help.txt) and the catalog tests check every flag against it.
//!
//! Not here on purpose: `--host`, `--port`, `--alias`, `--metrics`, `--offline`, `-m`, every download flag, the tool /
//! agent / MCP flags, router mode, `--ssl-*` and `--api-key-file`; deprecated spellings (`--mlock`, `--mmap`,
//! `--direct-io`: only `load_mode`). `env:NAME` means the value is set as environment variable NAME (never argv).
//!
//! The body of [`build_catalog`] is generated from the Python calls by backend/parity/gen/catalog_ws_a2.py.

use serde_json::Value;

use super::base::ParamSpec;

pub const BUILD: &str = "b10156";
/// `llama-server --version` prints "version: 10156 (91f8c9c5f)"
pub const COMMIT: &str = "91f8c9c5f";
pub const IMAGE_REPO: &str = "ghcr.io/ggml-org/llama.cpp";
/// What the moving `server-cuda` tag resolved to when the catalog was read (CUDA 12.8.1, `sm_120a` kernels present).
pub const IMAGE_DIGEST: &str =
    "sha256:4c0ece468721077a8bff8c114f8b0a92a8604ed310c91939fadfe37595a53b9e";
pub const DEFAULT_IMAGE: &str = "ghcr.io/ggml-org/llama.cpp@sha256:4c0ece468721077a8bff8c114f8b0a92a8604ed310c91939fadfe37595a53b9e";
pub const DOCS_URL: &str =
    "https://github.com/ggml-org/llama.cpp/blob/b10156/tools/server/README.md";
/// the image's own healthcheck probes this port
pub const CONTAINER_PORT: i64 = 8080;
pub const CHAT_TEMPLATE_PATH: &str = "/etc/engine-console/chat-template.jinja";
/// the node accepts at most 128 KiB per generated file
pub const MAX_TEMPLATE_BYTES: usize = 120_000;

pub const CACHE_TYPES: &[&str] = &[
    "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
];
/// Bytes per KV element (ggml block layouts; recalled, not measured).
pub const CACHE_TYPE_BYTES: &[(&str, f64)] = &[
    ("f32", 4.0),
    ("f16", 2.0),
    ("bf16", 2.0),
    ("q8_0", 34.0 / 32.0),
    ("q4_0", 18.0 / 32.0),
    ("q4_1", 20.0 / 32.0),
    ("iq4_nl", 18.0 / 32.0),
    ("q5_0", 22.0 / 32.0),
    ("q5_1", 24.0 / 32.0),
];
pub const SPEC_TYPES: &[&str] = &[
    "none",
    "draft-simple",
    "draft-eagle3",
    "draft-mtp",
    "draft-dflash",
    "ngram-simple",
    "ngram-map-k",
    "ngram-map-k4v",
    "ngram-mod",
    "ngram-cache",
];
pub const LOAD_MODES: &[&str] = &["none", "mmap", "mlock", "mmap+mlock", "dio"];
/// Built-in chat templates listed by the pinned help for --chat-template.
pub const CHAT_TEMPLATES: &[&str] = &[
    "bailing",
    "bailing-think",
    "bailing2",
    "chatglm3",
    "chatglm4",
    "chatml",
    "command-r",
    "deepseek",
    "deepseek-ocr",
    "deepseek2",
    "deepseek3",
    "exaone-moe",
    "exaone3",
    "exaone4",
    "falcon3",
    "gemma",
    "gigachat",
    "glmedge",
    "gpt-oss",
    "granite",
    "granite-4.0",
    "granite-4.1",
    "grok-2",
    "hunyuan-dense",
    "hunyuan-moe",
    "hunyuan-vl",
    "kimi-k2",
    "llama2",
    "llama2-sys",
    "llama2-sys-bos",
    "llama2-sys-strip",
    "llama3",
    "llama4",
    "megrez",
    "minicpm",
    "mistral-v1",
    "mistral-v3",
    "mistral-v3-tekken",
    "mistral-v7",
    "mistral-v7-tekken",
    "monarch",
    "openchat",
    "orion",
    "pangu-embedded",
    "phi3",
    "phi4",
    "rwkv-world",
    "seed_oss",
    "smolvlm",
    "solar-open",
    "vicuna",
    "vicuna-orca",
    "yandex",
    "zephyr",
];

/// bool params whose false is emitted as `--no-<flag>`; a bool outside this set emits nothing when false.
pub const NEGATABLE_KEYS: &[&str] = &[
    "kv_offload",
    "mmproj_offload",
    "context_shift",
    "kv_unified",
    "cont_batching",
    "cache_prompt",
    "cache_idle_slots",
    "warmup",
    "jinja",
    "reasoning_preserve",
    "skip_chat_parsing",
    "prefill_assistant",
    "ui",
    "slots",
];
/// string params that hold a path INSIDE the container; only files under the model mount are accepted.
pub const PATH_KEYS: &[&str] = &["mmproj", "lora", "lora_scaled", "spec_draft_model"];
/// keys whose value is "auto", "all" or a layer count
pub const LAYER_KEYS: &[&str] = &["n_gpu_layers", "spec_draft_ngl"];

/// `CACHE_TYPE_BYTES.get(name)`.
pub fn cache_type_bytes(name: &str) -> Option<f64> {
    CACHE_TYPE_BYTES
        .iter()
        .find(|(k, _)| *k == name)
        .map(|(_, v)| *v)
}

/// The `_p()` helper.
pub struct P {
    spec: ParamSpec,
    expert: bool,
}

fn p(key: &str, flag: &str, label: &str, help: &str, type_: &str, group: &str) -> P {
    P {
        spec: ParamSpec {
            key: key.into(),
            flag: flag.into(),
            label: label.into(),
            help: help.into(),
            type_: type_.into(),
            group: group.into(),
            default: Value::Null,
            choices: None,
            min: None,
            max: None,
            advanced: false,
            affects_memory: false,
            requires_restart: true,
            bool_style: "flag".into(),
            docs_url: Some(DOCS_URL.into()),
            explain: None,
            level: None,
            unit: None,
        },
        expert: false,
    }
}

impl P {
    fn default(mut self, v: Value) -> Self {
        self.spec.default = v;
        self
    }
    fn choices(mut self, c: &[&str]) -> Self {
        self.spec.choices = Some(c.iter().map(|s| (*s).to_string()).collect());
        self
    }
    fn min(mut self, v: f64) -> Self {
        self.spec.min = Some(v);
        self
    }
    fn max(mut self, v: f64) -> Self {
        self.spec.max = Some(v);
        self
    }
    fn adv(mut self) -> Self {
        self.spec.advanced = true;
        self
    }
    fn mem(mut self) -> Self {
        self.spec.affects_memory = true;
        self
    }
    fn unit(mut self, u: &str) -> Self {
        self.spec.unit = Some(u.into());
        self
    }
    fn expert(mut self) -> Self {
        self.expert = true;
        self
    }
    /// `_p`: `expert` sets level "expert" and implies advanced.
    fn build(mut self) -> ParamSpec {
        if self.expert {
            self.spec.level = Some("expert".into());
            self.spec.advanced = true;
        }
        self.spec
    }
}

// ---- generated by backend/parity/gen/catalog_ws_a2.py: begin ----
pub fn build_catalog() -> Vec<ParamSpec> {
    vec![
        p("n_gpu_layers", "--n-gpu-layers", "GPU layers", "How many of the model's layers to keep in GPU memory: a number, 'all', or 'auto' (the engine picks what fits). Layers that stay out run on the CPU, which is far slower.", "string", "memory").default(Value::from("auto")).mem().build(),
        p("fit", "--fit", "Automatic fit", "When on, start-up shrinks every memory setting you left unset (context size, GPU layers) until the model fits in the free GPU memory. Settings you give explicitly are never changed.", "enum", "memory").default(Value::from("on")).choices(&["on", "off"]).adv().mem().build(),
        p("fit_target", "--fit-target", "Fit safety margin (MiB per GPU)", "Free GPU memory the automatic fit must leave over on every GPU.", "int", "memory").default(Value::from(1024_i64)).min(0.0).adv().mem().unit("MiB").build(),
        p("fit_ctx", "--fit-ctx", "Smallest context the fit may pick", "The automatic fit will not shrink the context below this many tokens.", "int", "memory").default(Value::from(4096_i64)).min(1.0).adv().mem().unit("tokens").build(),
        p("kv_offload", "--kv-offload", "KV cache on the GPU", "Keep the KV cache (the running conversation memory) in GPU memory. Off moves it to host RAM, which saves GPU memory and slows every token.", "bool", "memory").default(Value::Bool(true)).adv().mem().build(),
        p("cache_type_k", "--cache-type-k", "KV cache type (keys)", "Number format of the key half of the KV cache. q8_0 uses about half the memory of f16 with little quality loss.", "enum", "memory").default(Value::from("f16")).choices(CACHE_TYPES).adv().mem().build(),
        p("cache_type_v", "--cache-type-v", "KV cache type (values)", "Number format of the value half of the KV cache. Anything other than f16, bf16 or f32 needs flash attention.", "enum", "memory").default(Value::from("f16")).choices(CACHE_TYPES).adv().mem().build(),
        p("swa_full", "--swa-full", "Full-size sliding-window cache", "Keep a full-length KV cache even for sliding-window layers. Uses more memory; helps prompt reuse on some models.", "bool", "memory").default(Value::Bool(false)).adv().mem().build(),
        p("cache_ram", "--cache-ram", "Prompt cache size (MiB of RAM)", "Host RAM the server may use to remember earlier prompts so repeated prefixes are not recomputed. -1 removes the limit and 0 turns the cache off.", "int", "memory").default(Value::from(8192_i64)).min(-1.0).adv().mem().unit("MiB").build(),
        p("load_mode", "--load-mode", "Model loading mode", "How the model file is read: mmap maps it (default), none reads it into RAM, mlock pins it in RAM, mmap+mlock does both, dio uses direct I/O.", "enum", "memory").default(Value::from("mmap")).choices(LOAD_MODES).adv().mem().build(),
        p("cpu_moe", "--cpu-moe", "All experts on the CPU", "Keep every mixture-of-experts weight in host RAM while attention and shared layers stay on the GPU. Lets a very large MoE model run on a small GPU.", "bool", "memory").default(Value::Bool(false)).adv().mem().build(),
        p("n_cpu_moe", "--n-cpu-moe", "Experts of the first N layers on the CPU", "Keep the expert weights of the first N layers in host RAM. A middle path between all on the GPU and --cpu-moe.", "int", "memory").min(0.0).adv().mem().unit("layers").build(),
        p("override_tensor", "--override-tensor", "Tensor placement overrides", "Comma-separated pattern=buffer rules that force named tensors onto a chosen device, for example exps=CPU. Very detailed control; the engine warns that CPU overrides combine badly with mmap.", "string", "memory").expert().mem().build(),
        p("mmproj_offload", "--mmproj-offload", "Vision projector on the GPU", "Run the image (multimodal projector) part of a vision model on the GPU.", "bool", "memory").default(Value::Bool(true)).adv().mem().build(),
        p("split_mode", "--split-mode", "Multi-GPU split mode", "How a model is spread over several GPUs: none (one GPU), layer (default, layers and KV split in a pipeline), row, or tensor (experimental).", "enum", "parallelism").default(Value::from("layer")).choices(&["none", "layer", "row", "tensor"]).adv().mem().build(),
        p("tensor_split", "--tensor-split", "GPU split proportions", "Share of the model each GPU gets, comma-separated, for example 3,1 puts three quarters on the first GPU.", "string", "parallelism").adv().mem().build(),
        p("main_gpu", "--main-gpu", "Main GPU", "Index of the GPU used when the split mode is none, and for intermediate results with the row split.", "int", "parallelism").default(Value::from(0_i64)).min(0.0).adv().mem().build(),
        p("device", "--device", "Devices to use", "Comma-separated device names to offload to, or none to keep everything on the CPU. Run with --list-devices to see names.", "string", "parallelism").expert().mem().build(),
        p("ctx_size", "--ctx-size", "Context size (tokens)", "Total tokens of conversation memory the server allocates; 0 uses the model's trained maximum. With several slots the budget is shared between them.", "int", "context").default(Value::from(0_i64)).min(0.0).mem().unit("tokens").build(),
        p("rope_scaling", "--rope-scaling", "RoPE scaling method", "How positions are stretched to reach a context longer than the model was trained on: none, linear or yarn.", "enum", "context").choices(&["none", "linear", "yarn"]).adv().build(),
        p("rope_scale", "--rope-scale", "RoPE scale factor", "Context expansion factor: 4 stretches a 32k-trained model towards 128k.", "float", "context").min(0.01).adv().build(),
        p("rope_freq_base", "--rope-freq-base", "RoPE base frequency", "Base frequency for NTK-aware scaling; the default is read from the model.", "float", "context").min(1.0).expert().build(),
        p("yarn_orig_ctx", "--yarn-orig-ctx", "YaRN original context", "The context length the model was trained on, for YaRN scaling; 0 uses the value stored in the model.", "int", "context").default(Value::from(0_i64)).min(0.0).expert().unit("tokens").build(),
        p("context_shift", "--context-shift", "Context shift", "When a conversation outgrows its slot, drop the oldest tokens and keep going instead of stopping.", "bool", "context").default(Value::Bool(false)).adv().build(),
        p("image_min_tokens", "--image-min-tokens", "Minimum tokens per image", "Lower bound of tokens one image takes, for vision models with dynamic resolution; the default is read from the model.", "int", "context").min(1.0).adv().mem().unit("tokens").build(),
        p("image_max_tokens", "--image-max-tokens", "Maximum tokens per image", "Upper bound of tokens one image takes, for vision models with dynamic resolution; the default is read from the model.", "int", "context").min(1.0).adv().mem().unit("tokens").build(),
        p("mmproj", "--mmproj", "Vision projector file", "Path of the multimodal projector GGUF, inside the model folder (/models/<file>), for vision models.", "string", "quantization").adv().mem().build(),
        p("lora", "--lora", "LoRA adapters", "Comma-separated LoRA adapter GGUF files inside the model folder (/models/<file>), applied on top of the model.", "string", "quantization").adv().mem().build(),
        p("lora_scaled", "--lora-scaled", "LoRA adapters with scales", "Comma-separated file:scale pairs, for example /models/a.gguf:0.5, applied on top of the model.", "string", "quantization").adv().mem().build(),
        p("lora_init_without_apply", "--lora-init-without-apply", "Load LoRA without applying", "Load the adapters but leave them switched off until a request turns them on.", "bool", "quantization").default(Value::Bool(false)).expert().build(),
        p("override_kv", "--override-kv", "Override model metadata", "Comma-separated KEY=TYPE:VALUE entries (types int, float, bool, str) that replace metadata stored in the model file.", "string", "quantization").expert().build(),
        p("parallel", "--parallel", "Slots (parallel requests)", "Number of conversations served at the same time. -1 lets the engine pick (4 slots sharing one KV buffer).", "int", "scheduling").default(Value::from(-1_i64)).min(-1.0).mem().unit("slots").build(),
        p("kv_unified", "--kv-unified", "Unified KV buffer", "Let all slots share one KV buffer instead of splitting the context evenly; on by default only when the slot count is auto.", "bool", "scheduling").adv().mem().build(),
        p("cont_batching", "--cont-batching", "Continuous batching", "Let new requests join a batch that is already running instead of waiting for it to finish.", "bool", "scheduling").default(Value::Bool(true)).adv().build(),
        p("cache_prompt", "--cache-prompt", "Prompt caching", "Reuse the part of a slot's earlier prompt that matches the new request so only the new part is processed.", "bool", "scheduling").default(Value::Bool(true)).adv().build(),
        p("cache_reuse", "--cache-reuse", "Prompt reuse chunk size", "Minimum length of a matching chunk that may be reused from the cache by shifting it; 0 turns this off.", "int", "scheduling").default(Value::from(0_i64)).min(0.0).adv().unit("tokens").build(),
        p("cache_idle_slots", "--cache-idle-slots", "Save idle slots to the prompt cache", "When a new task starts, store the idle slots' conversations in the prompt cache and free them. Needs the prompt cache.", "bool", "scheduling").default(Value::Bool(true)).adv().build(),
        p("slot_prompt_similarity", "--slot-prompt-similarity", "Slot prompt similarity", "How closely a request must match a slot's earlier prompt for that slot to be reused; 0 disables the check.", "float", "scheduling").default(Value::from(0.1)).min(0.0).max(1.0).adv().build(),
        p("ctx_checkpoints", "--ctx-checkpoints", "Context checkpoints per slot", "Maximum snapshots of a conversation kept per slot, used to rewind models that cannot drop recent tokens.", "int", "scheduling").default(Value::from(32_i64)).min(0.0).adv().build(),
        p("sleep_idle_seconds", "--sleep-idle-seconds", "Sleep after idle (s)", "Seconds without requests after which the server sleeps and frees the model; -1 turns it off.", "int", "scheduling").default(Value::from(-1_i64)).min(-1.0).adv().unit("s").build(),
        p("warmup", "--warmup", "Warm-up run", "Run one empty generation at start-up so the first real request is not slow.", "bool", "scheduling").default(Value::Bool(true)).adv().build(),
        p("flash_attn", "--flash-attn", "Flash attention", "Use the memory-efficient attention kernels: on, off, or auto (default; on whenever the GPU backend supports the model).", "enum", "performance").default(Value::from("auto")).choices(&["on", "off", "auto"]).mem().build(),
        p("threads", "--threads", "CPU threads", "Threads used while generating tokens on the CPU; -1 picks automatically. Only matters when layers run on the CPU.", "int", "performance").default(Value::from(-1_i64)).min(-1.0).adv().unit("threads").build(),
        p("threads_batch", "--threads-batch", "CPU threads (prompt processing)", "Threads used while processing prompts; by default the same as threads.", "int", "performance").min(-1.0).adv().unit("threads").build(),
        p("batch_size", "--batch-size", "Logical batch size", "Largest number of tokens handed to the model in one step. Larger speeds up long prompts and uses more memory.", "int", "performance").default(Value::from(2048_i64)).min(1.0).adv().mem().unit("tokens").build(),
        p("ubatch_size", "--ubatch-size", "Physical batch size", "Largest number of tokens the GPU computes at once; sets the size of the compute buffers.", "int", "performance").default(Value::from(512_i64)).min(1.0).adv().mem().unit("tokens").build(),
        p("numa", "--numa", "NUMA strategy", "Placement hint for multi-socket CPUs: distribute, isolate or numactl. Leave unset on single-socket machines.", "enum", "performance").choices(&["distribute", "isolate", "numactl"]).expert().build(),
        p("jinja", "--jinja", "Jinja chat templates", "Use the full chat template engine. Needed for tool calls and for splitting reasoning from the answer.", "bool", "tools_reasoning").default(Value::Bool(true)).adv().build(),
        p("chat_template", "--chat-template", "Built-in chat template", "Name of a built-in template (for example chatml or llama3) or an inline Jinja template; replaces the one stored in the model.", "string", "tools_reasoning").adv().build(),
        p("chat_template_jinja", "--chat-template-file", "Custom chat template (Jinja)", "Full Jinja template text. The console writes it to a file inside the container; it replaces the model's own template.", "string", "tools_reasoning").adv().build(),
        p("chat_template_kwargs", "--chat-template-kwargs", "Chat template arguments (JSON)", "Extra arguments handed to the template for every request, for example {\"enable_thinking\": false}.", "json", "tools_reasoning").adv().build(),
        p("reasoning", "--reasoning", "Reasoning mode", "Turn the model's thinking on, off, or auto (detect from the chat template).", "enum", "tools_reasoning").default(Value::from("auto")).choices(&["on", "off", "auto"]).build(),
        p("reasoning_format", "--reasoning-format", "Reasoning output format", "Where thinking text goes: none leaves it in the answer, deepseek puts it in reasoning_content, deepseek-legacy does both.", "enum", "tools_reasoning").choices(&["none", "deepseek", "deepseek-legacy"]).adv().build(),
        p("reasoning_budget", "--reasoning-budget", "Reasoning budget (tokens)", "Cap on thinking tokens: -1 unlimited, 0 ends thinking immediately, a positive number is the budget.", "int", "tools_reasoning").default(Value::from(-1_i64)).min(-1.0).adv().unit("tokens").build(),
        p("reasoning_preserve", "--reasoning-preserve", "Keep reasoning in the history", "Keep earlier thinking in the conversation history for templates that support it, not only in the latest message.", "bool", "tools_reasoning").adv().build(),
        p("skip_chat_parsing", "--skip-chat-parsing", "Skip chat output parsing", "Return everything, including reasoning and tool calls, as plain content.", "bool", "tools_reasoning").default(Value::Bool(false)).expert().build(),
        p("prefill_assistant", "--prefill-assistant", "Prefill the assistant reply", "When the last message is from the assistant, continue it instead of treating it as a finished message.", "bool", "tools_reasoning").default(Value::Bool(true)).expert().build(),
        p("spec_type", "--spec-type", "Speculative decoding type", "How draft tokens are produced: none, a draft model (draft-simple, draft-eagle3, draft-dflash), the model's own MTP layers (draft-mtp), or n-gram lookup from the context.", "enum", "speculative").default(Value::from("none")).choices(SPEC_TYPES).adv().mem().build(),
        p("spec_draft_model", "--spec-draft-model", "Draft model file", "Path of the draft model GGUF, inside the model folder (/models/<file>).", "string", "speculative").adv().mem().build(),
        p("spec_draft_n_max", "--spec-draft-n-max", "Draft tokens per step (max)", "Most tokens the draft proposes before the main model checks them.", "int", "speculative").default(Value::from(3_i64)).min(0.0).adv().unit("tokens").build(),
        p("spec_draft_n_min", "--spec-draft-n-min", "Draft tokens per step (min)", "Fewest draft tokens worth checking; shorter drafts are skipped.", "int", "speculative").default(Value::from(0_i64)).min(0.0).expert().unit("tokens").build(),
        p("spec_draft_p_min", "--spec-draft-p-min", "Draft confidence floor", "Stop drafting when the draft model's confidence in the next token falls below this probability.", "float", "speculative").default(Value::from(0.0)).min(0.0).max(1.0).expert().build(),
        p("spec_draft_ngl", "--spec-draft-ngl", "Draft model GPU layers", "How many layers of the draft model go to the GPU: a number, 'all' or 'auto'.", "string", "speculative").default(Value::from("auto")).adv().mem().build(),
        p("served_model_name", "--alias", "Extra model names", "Extra names the API answers to. The console always serves its own name first.", "string_list", "network").adv().build(),
        p("api_key", "env:LLAMA_API_KEY", "API key", "Require this key (sent as a Bearer token) on the API. Passed through the LLAMA_API_KEY environment variable, not the command line. The metrics page then needs the key too.", "string", "network").build(),
        p("cors_origins", "--cors-origins", "Allowed browser origins", "Which web pages may call the API from a browser. The console sets localhost; the engine's own default (*) lets any website read answers from a loopback port.", "string", "network").default(Value::from("localhost")).adv().build(),
        p("ui", "--ui", "Built-in web page", "Serve llama.cpp's own chat page on the same port.", "bool", "network").default(Value::Bool(true)).adv().build(),
        p("slots", "--slots", "Slots monitoring endpoint", "Expose /slots, which lists what each slot is doing.", "bool", "network").default(Value::Bool(true)).adv().build(),
        p("timeout", "--timeout", "Server read/write timeout (s)", "Seconds the HTTP server waits on a stalled connection.", "int", "network").default(Value::from(3600_i64)).min(1.0).adv().unit("s").build(),
        p("log_verbosity", "--verbosity", "Log verbosity", "How much the server logs: 0 generic, 1 errors, 2 warnings, 3 info, 4 trace, 5 debug.", "int", "network").default(Value::from(3_i64)).min(0.0).max(5.0).adv().build(),
        p("embedding", "--embedding", "Embeddings-only server", "Serve only the embeddings endpoint; use with a dedicated embedding model.", "bool", "advanced").default(Value::Bool(false)).adv().build(),
        p("pooling", "--pooling", "Embedding pooling", "How token vectors are combined into one embedding; the model's own default is used when unset.", "enum", "advanced").choices(&["none", "mean", "cls", "last", "rank"]).adv().build(),
        p("rerank", "--rerank", "Reranking endpoint", "Enable the reranking endpoint; use with a reranker model.", "bool", "advanced").default(Value::Bool(false)).adv().build(),
        p("temperature", "--temp", "Default temperature", "Randomness of sampling when a request does not set it: low is focused, high is varied.", "float", "advanced").default(Value::from(0.8)).min(0.0).adv().build(),
        p("top_k", "--top-k", "Default top-k", "Sample only from the k most likely tokens when a request does not set it; 0 turns it off.", "int", "advanced").default(Value::from(40_i64)).min(0.0).adv().build(),
        p("top_p", "--top-p", "Default top-p", "Sample only from the smallest set of tokens whose probabilities add up to p; 1.0 turns it off.", "float", "advanced").default(Value::from(0.95)).min(0.0).max(1.0).adv().build(),
        p("min_p", "--min-p", "Default min-p", "Drop tokens less likely than p times the best token; 0 turns it off.", "float", "advanced").default(Value::from(0.05)).min(0.0).max(1.0).adv().build(),
        p("repeat_penalty", "--repeat-penalty", "Default repeat penalty", "Penalty for tokens that appeared in the recent text; 1.0 turns it off.", "float", "advanced").default(Value::from(1.0)).min(0.0).adv().build(),
        p("presence_penalty", "--presence-penalty", "Default presence penalty", "Flat penalty for any token that has already appeared; 0 turns it off.", "float", "advanced").default(Value::from(0.0)).adv().build(),
        p("frequency_penalty", "--frequency-penalty", "Default frequency penalty", "Penalty that grows with how often a token has already appeared; 0 turns it off.", "float", "advanced").default(Value::from(0.0)).adv().build(),
        p("seed", "--seed", "Random seed", "Fixed seed for repeatable sampling; -1 picks a random one.", "int", "advanced").default(Value::from(-1_i64)).adv().build(),
    ]
}
// ---- generated by backend/parity/gen/catalog_ws_a2.py: end ----

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_compose_like_python() {
        assert_eq!(DEFAULT_IMAGE, format!("{IMAGE_REPO}@{IMAGE_DIGEST}"));
        assert_eq!(
            DOCS_URL,
            format!("https://github.com/ggml-org/llama.cpp/blob/{BUILD}/tools/server/README.md")
        );
        assert_eq!(cache_type_bytes("q8_0"), Some(1.0625));
        assert_eq!(cache_type_bytes("nope"), None);
        let _ = COMMIT;
    }
}
