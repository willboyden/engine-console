import fs from 'node:fs';
import path from 'node:path';
// Mock learn-mode content for GET /api/v1/learn: a sample in the shape of backend domain/learn.rs (LearnBundle). The real
// content lives in backend learn/*.json and is written separately; this exists so the UI and its tests have something solid
// to render. The wording is general and plain; it is not a verified statement about any particular engine version.
// Limits mirror backend learn/README.md: summary/short <= 160, details <= 900, when_to_change/risks <= 400, example <= 200,
// long <= 1500. tests/learn.test.mjs checks them, that every referenced term exists, and that every mock param is covered.
const T = (title, short, long = '', see_also = []) => ({ title, short, long, see_also });
const E = (summary, details = '', when_to_change = '', risks = '', example = '', terms = [], level = 'advanced') =>
  ({ summary, details, when_to_change, risks, example, terms, level });

// ───────────────────────────────── glossary ─────────────────────────────────
export const glossary = {
  vram: T('VRAM (GPU memory)', "The memory on the graphics card, separate from the computer's own RAM.",
    "A model's weights and its working memory must sit in VRAM to run fast. Cards for AI work have 24 to 96 GB. When VRAM is full the engine cannot load another model or serve more conversations, however much ordinary RAM is free.", ['gpu', 'weights', 'kv_cache', 'unified_memory']),
  gpu: T('GPU', 'The graphics card that does the heavy arithmetic for a model.',
    'A GPU runs thousands of small calculations at once, which is exactly what a language model needs. A machine may have none, one or several; each has its own VRAM, temperature and power draw.', ['vram', 'compute_capability']),
  unified_memory: T('Unified memory', 'Memory that the GPU and the processor share, so there is one pool instead of two.',
    'Some compact AI machines have no separate VRAM: the GPU uses the same RAM as everything else. Models can then use most of the RAM, but the engine, the operating system and the page cache all compete for the same space.', ['vram', 'host_ram']),
  host_ram: T('Host RAM', "The computer's ordinary memory, used by programs, the system and file caching.",
    'Engines keep some data here too: the program itself, tokenizers, and any weights you offload from the GPU. If RAM runs short the system starts using swap, which makes everything slow.', ['swap', 'shared_memory', 'page_cache']),
  swap: T('Swap', 'Disk space used as overflow when RAM is full. It is far slower than RAM.',
    'Heavy swap use is a warning sign: engines can stall for seconds at a time while the system moves data to and from disk. Free some RAM or stop an engine instead of relying on swap.', ['host_ram']),
  shared_memory: T('Shared memory (shm)', 'RAM that several processes can read and write together; the system cannot take it back.',
    'Engines that use several GPUs, and many containers, put buffers here. It counts as used memory even though it can look like cache, and it is not freed under pressure.', ['host_ram', 'page_cache']),
  page_cache: T('Page cache', "Spare RAM the system uses to remember recently read files, so the next read is fast.",
    'Loading a big model fills the page cache with its files. That memory is handed back when programs need it, so a large cache is normal and not a problem.', ['host_ram']),
  model: T('Model', 'The trained neural network that reads text in and writes text out.',
    'A model is a large set of numbers (its weights) plus a small description of how to use them. Bigger models are usually smarter but need more memory and run slower.', ['weights', 'parameters']),
  weights: T('Weights', 'The numbers that make up a trained model; they must be loaded into memory to run it.',
    'The size of the weights decides how much memory a model needs just to start. A model with 30 billion parameters at 8 bits per number needs about 30 GB for its weights alone.', ['parameters', 'quantization', 'vram']),
  parameters: T('Parameters', "How many numbers a model has, such as 8B for eight billion; a rough measure of its size.",
    'More parameters generally means a more capable model and more memory. Models marked A3B or similar are mixtures of experts: they have many parameters in total but use only a few at a time.', ['moe', 'weights']),
  token: T('Token', 'A piece of text the model reads or writes, roughly three quarters of an English word.',
    'Models do not see letters or words but tokens. About 1,000 tokens is 750 English words or one and a half pages. Speed, limits and costs are all counted in tokens.', ['context_window', 'throughput']),
  context_window: T('Context window', 'The most text, in tokens, that the model can consider at once: your input plus its answer.',
    'Everything in a conversation has to fit in the window. Longer windows let you paste longer documents but use much more memory for the KV cache.', ['token', 'kv_cache']),
  prompt: T('Prompt', 'The text you send to the model, including any earlier conversation and instructions.',
    'The engine first reads the whole prompt (prefill) and then writes the answer token by token (decode).', ['prefill', 'decode']),
  prefill: T('Prefill', 'The first stage of a request: the model reads the whole prompt in one go.',
    'Prefill is fast per token but a long prompt still takes a moment, which is why the first word of an answer can take time to appear.', ['decode', 'ttft']),
  decode: T('Decode', 'The second stage: the model writes the answer one token at a time.',
    'Each new token needs a pass through the whole model, so decode speed (tokens per second) is what you feel as "typing speed".', ['prefill', 'throughput']),
  kv_cache: T('KV cache', "The model's working memory for the conversations it is serving.",
    'It stores what the model has already read so it does not repeat the work for every new token. It grows with the length of each conversation and the number of conversations at once, and it is usually the biggest use of VRAM after the weights.', ['vram', 'context_window', 'prefix_cache']),
  prefix_cache: T('Prefix cache', 'Reuse of the work done on the start of a prompt when many requests begin the same way.',
    'If every request starts with the same instructions, the engine can skip reading them again. This lowers the time to first token and is most useful for agents and chat with a long system prompt.', ['kv_cache', 'ttft']),
  throughput: T('Throughput', 'How many tokens per second the engine produces in total, across all users.',
    'Throughput rises when many requests are batched together, while the speed one person sees may fall a little.', ['concurrency', 'latency']),
  latency: T('Latency', 'How long you wait for a response.',
    'Two numbers matter most: how long until the first word appears, and how fast the rest streams.', ['ttft', 'itl']),
  ttft: T('Time to first token (TTFT)', 'How long it takes from sending a request until the first word of the answer arrives.',
    'It is dominated by reading the prompt, so it grows with prompt length and with how busy the engine is.', ['prefill', 'latency']),
  itl: T('Inter-token latency (ITL)', 'The gap between one streamed token and the next; the "typing speed" of the answer.',
    'Lower is smoother. It rises as more requests share the engine.', ['decode', 'latency']),
  concurrency: T('Concurrency', 'How many requests are being served at the same moment.',
    'More concurrency raises total throughput and the memory needed for the KV cache. The engines have a limit; extra requests wait in a queue.', ['kv_cache', 'throughput', 'batch']),
  batch: T('Batch', 'A group of requests the engine processes together in one step.',
    'Batching is why one GPU can serve many people: a step costs about the same for one request or for dozens.', ['concurrency', 'throughput']),
  quantization: T('Quantization', 'Storing the weights with fewer bits per number so the model is smaller and faster.',
    'An 8-bit model needs half the memory of a 16-bit one, at a small cost in accuracy. Quantized variants of the same model are named after the format, such as FP8 or Q4_K_M.', ['weights', 'fp8', 'gguf']),
  fp8: T('FP8', 'An 8-bit number format that halves memory compared with 16-bit with little loss of quality.',
    'Recent NVIDIA GPUs run FP8 natively, so it is usually the best first choice for large models.', ['quantization']),
  nvfp4: T('NVFP4 (4-bit)', 'A 4-bit number format for the newest GPUs; very small, with a larger quality trade-off.',
    'It works best on dense models. Some engines are slow with 4-bit mixture-of-experts models on certain GPUs, so check the warnings the console shows.', ['quantization', 'moe']),
  gguf: T('GGUF', 'A single-file model format used by llama.cpp, with the quantization named in the file name.',
    'Names like Q4_K_M mean about 4 bits per weight. One repository usually offers many quantizations of the same model: you pick one file group.', ['quantization', 'model_root']),
  safetensors: T('Safetensors', 'The common safe file format for model weights on Hugging Face.',
    'A Hugging Face model is a folder of safetensors files plus small configuration files; engines such as vLLM and SGLang load that folder directly.', ['hf_cache']),
  hf_cache: T('Hugging Face cache', 'The folder where downloaded Hugging Face models are kept so they are fetched only once.',
    'Each machine has its own. The console can download into it and show what is stored, which drive it is on, and how much space is left.', ['model_root', 'safetensors']),
  tensor_parallelism: T('Tensor parallelism', 'Splitting each layer of a model across several GPUs so they work on it together.',
    'It lets a model that is too big for one card run, and it can speed up answers. The GPUs exchange data constantly, so it needs a fast connection between them.', ['pipeline_parallel', 'expert_parallel', 'gpu']),
  pipeline_parallel: T('Pipeline parallelism', 'Putting different layers of a model on different GPUs, like stations on an assembly line.',
    'It needs less communication than tensor parallelism but leaves cards idle part of the time, so it is mainly used across machines.', ['tensor_parallelism']),
  expert_parallel: T('Expert parallelism', 'Spreading the experts of a mixture-of-experts model across GPUs.',
    'Each GPU holds some of the experts and receives only the tokens that need them.', ['moe', 'tensor_parallelism']),
  moe: T('Mixture of experts (MoE)', 'A model made of many small specialist networks of which only a few run for each token.',
    'A 35B-A3B model has 35 billion parameters in total but uses about 3 billion for each token, so it is fast for its size, but all of it must still fit in memory.', ['parameters', 'expert_parallel']),
  cuda_graph: T('CUDA graph', 'A recording of the GPU work for one step that the engine replays instead of rebuilding it.',
    'It makes decoding faster at the price of some extra memory and a longer start-up while the recordings are made.', ['decode']),
  speculative_decoding: T('Speculative decoding', 'A speed-up where a small helper guesses several tokens and the big model checks them at once.',
    'When the guesses are right the answer arrives faster with the same text. When they are wrong little is lost.', ['decode', 'throughput']),
  chunked_prefill: T('Chunked prefill', 'Reading a long prompt in slices so that other users are not stalled while it is processed.',
    'The engine mixes slices of the long prompt with other requests in each step, which keeps streaming smooth for everyone.', ['prefill', 'itl']),
  engine: T('Engine', 'The program that loads a model and answers requests, such as vLLM, SGLang, llama.cpp or Ollama.',
    'Engines differ in the file formats they load, the hardware they favour and the settings they offer. The console starts them as containers.', ['container', 'image']),
  container: T('Container', 'A sealed, repeatable environment in which an engine runs, separate from the rest of the machine.',
    'The console starts every engine in its own container, connected only to a private network.', ['image', 'rootless_docker']),
  image: T('Container image', 'The packaged software a container is started from, such as one engine version.',
    'Each machine may only run images its own policy allows, and the image must already be on that machine unless pulling is permitted.', ['container', 'policy']),
  gateway: T('Gateway', 'A tiny program in front of each engine that is the only way to reach it.',
    'The engine itself has no open ports; the gateway listens on the machine and forwards requests, which keeps engines isolated.', ['container']),
  node: T('Node', 'A small helper program on each machine that does the actual work for the console.',
    'It reads hardware, runs containers and moves files on that machine. The console never runs commands on another machine itself.', ['host', 'policy', 'ssh_key']),
  host: T('Machine (host)', 'One computer the console manages, such as this workstation or a server on your network.',
    'The console connects to each machine through its node. A machine can be up, stale, down or other states shown by its link badge.', ['node', 'policy']),
  policy: T('Node policy', "A file on each machine that sets what the console is allowed to do there.",
    'It is owned by an administrator on that machine and the console can read it but never change it. It sets the mode, allowed images, model folders and whether files may be deleted.', ['node', 'image']),
  rootless_docker: T('Rootless Docker', 'A way of running containers without administrator rights on the machine.',
    'It limits what a container could do if something went wrong. With ordinary (rootful) Docker, being allowed to start containers is close to being an administrator.', ['container']),
  nfs: T('NFS (network storage)', 'A way of mounting a folder from another machine so it looks like a local one.',
    'Several machines can read the same model files without copying them. Reads go over the network, so loading is slower than from a local disk.', ['model_root']),
  model_root: T('Model root', 'A folder on a machine where model files are kept, as listed in its policy.',
    'Roots have a kind (Hugging Face cache, GGUF files, plain folders...) and may be read-only. The console only reads and writes inside roots.', ['hf_cache', 'nfs']),
  preset: T('Preset', 'A ready-made bundle of settings for a goal, such as low latency or long documents.',
    'Pick a preset when you know what you want to do but not which settings to change; you can adjust any value afterwards.', []),
  tool_calling: T('Tool calling', 'Letting the model ask your program to run a function, such as a search, and use the result.',
    'The engine needs the right parser for the model family so it can recognise those requests.', ['reasoning']),
  reasoning: T('Reasoning model', 'A model that thinks step by step before it answers.',
    'The thinking can be shown separately from the answer if the engine has the matching parser. Reasoning uses more tokens and takes longer.', ['tool_calling']),
  offload: T('CPU offload', 'Keeping part of the model in ordinary RAM and moving it to the GPU only when needed.',
    'It lets a model run that is slightly too big for the card, but it is much slower and uses host RAM.', ['host_ram', 'vram']),
  preemption: T('Preemption', 'The engine pausing a request to free memory for others, then redoing its work later.',
    'A rising count means the KV cache is too small for the load.', ['kv_cache', 'concurrency']),
  ssh_key: T('Hub key', "The credential the console uses to reach other machines over SSH.",
    'On each machine it should be limited to starting the node and nothing else. If a machine shows the unrestricted warning, the key there can do more than that.', ['node']),
  compute_capability: T('Compute capability', "A GPU generation number such as 12.0 that engines check for supported features.",
    'Some number formats and kernels only exist for certain generations; the console warns when a combination is known to be slow or unsupported.', ['gpu', 'nvfp4']),
  embedding: T('Embedding model', 'A model that turns text into a list of numbers for search, rather than writing text.',
    'Embedding models are small and answer in a single step, so they are not used with chat settings.', ['model']),
};

// ───────────────────────────────── metrics ─────────────────────────────────
export const metrics = {
  requests_running: E('How many requests the engine is working on right now.', 'Each running request is a conversation being answered. More running requests use more KV cache.', '', 'If it sits at the engine limit while others wait, raise the limit or add capacity.', 'A value of 6 means six people are being answered at once.', ['concurrency', 'kv_cache'], 'basic'),
  requests_waiting: E('How many requests are queued because the engine is full.', 'A request waits when the engine has reached its limit on parallel requests or has no KV cache left.', 'If it is often above zero, users are waiting for answers: allow more parallel requests or use a bigger card.', 'A long queue means slow first words and may end in timeouts.', '', ['concurrency', 'kv_cache'], 'basic'),
  kv_cache_usage_pct: E('How full the model\'s working memory is, in percent.', 'The KV cache holds every active conversation. At 100% the engine has to pause or drop work.', 'If it stays above 90%, reduce the context length or the number of parallel requests.', 'Above 95% the engine may preempt requests, and answers slow down.', '', ['kv_cache', 'preemption'], 'basic'),
  prefix_cache_hit_pct: E('How often the start of a prompt was already remembered, in percent.', 'A high value means many requests share the same beginning, so the engine skips repeated work.', '', '', '80% means four in five prompt tokens came from the cache.', ['prefix_cache'], 'advanced'),
  prompt_tokens_total: E('The total number of prompt tokens read since the engine started.', 'It only counts up. Divide changes over time to get the reading speed.', '', '', '', ['token', 'prefill'], 'advanced'),
  generation_tokens_total: E('The total number of tokens written since the engine started.', 'It only counts up and is the best measure of how much work an engine has done.', '', '', '', ['token', 'decode'], 'advanced'),
  ttft_p50_s: E('The typical wait, in seconds, until the first word of an answer appears.', 'Half of requests are faster than this value (the median).', '', 'If it grows with load, the engine is busy reading prompts.', '0.12 s feels instant; 3 s feels slow.', ['ttft', 'prefill'], 'basic'),
  ttft_p95_s: E('How long the slowest one in twenty requests waits for the first word.', 'It shows the worst typical experience rather than the average.', '', '', '', ['ttft'], 'advanced'),
  itl_p50_s: E('The typical gap in seconds between streamed tokens.', 'The reciprocal is the typing speed: 0.02 s between tokens is 50 tokens per second.', '', '', '', ['itl', 'decode'], 'advanced'),
  e2e_p50_s: E('The typical time from sending a request to receiving the whole answer.', 'It adds the wait for the first word to the time spent writing.', '', '', '', ['latency'], 'advanced'),
  e2e_p95_s: E('How long the slowest one in twenty requests takes to finish.', 'Useful for setting timeouts in the programs that call the engine.', '', '', '', ['latency'], 'advanced'),
  prompt_tps: E('How fast the engine reads prompts, in tokens per second.', 'Reading is much faster than writing because the model can read many tokens in parallel.', '', '', '', ['prefill', 'throughput'], 'advanced'),
  generation_tps: E('How fast the engine writes answers, in tokens per second across all users.', 'With many users this is the total; each one gets a share of it.', '', '', '100 tokens per second is about 75 words per second.', ['decode', 'throughput'], 'basic'),
  preemptions_total: E('How many times a request was paused to make room for others.', 'It only counts up. Preemption wastes work because the paused request must be redone.', 'Rising quickly means the KV cache is too small: lower the context length or concurrency.', '', '', ['preemption', 'kv_cache'], 'advanced'),
  spec_decode_accept_pct: E('How many of the helper\'s guessed tokens the big model accepted, in percent.', 'A high value means speculative decoding is paying off.', '', '', '', ['speculative_decoding'], 'expert'),
  vram_used_gib: E('How much of the GPU memory is in use, in GiB.', 'It includes the weights, the KV cache and working memory of every program on that card, not only this console\'s engines.', '', 'Near full, new engines cannot start on that card.', '', ['vram'], 'basic'),
  gpu_util_pct: E('How busy the GPU\'s compute units are, in percent.', 'It measures activity, not efficiency: a decode-bound engine may show high use while moving little data.', '', '', '', ['gpu'], 'basic'),
  gpu_temp_c: E('The GPU temperature in degrees Celsius.', 'GPUs slow themselves down when they get too hot. Sustained temperatures above the mid 80s deserve a look at airflow.', '', '', '', ['gpu'], 'basic'),
  gpu_power_w: E('How much electrical power the GPU is drawing, in watts.', 'It rises with load and falls when idle.', '', '', '', ['gpu'], 'advanced'),
  host_ram_gib: E('How much of the machine\'s RAM this engine holds, in GiB.', 'It is read from the container\'s own accounting, so it adds up cleanly across engines.', '', '', '', ['host_ram'], 'basic'),
  host_ram_anon_gib: E('RAM the engine program itself allocated, in GiB.', 'Its working data: not backed by a file and not reclaimable while the engine runs.', '', '', '', ['host_ram'], 'advanced'),
  host_ram_cache_gib: E('RAM the engine uses to cache files, in GiB.', 'Mostly model files read from disk. The system can take it back when it needs the space.', '', '', '', ['page_cache'], 'advanced'),
  host_ram_shmem_gib: E('Shared memory the engine holds, in GiB.', 'It looks like cache but the system cannot reclaim it, so treat it as used.', '', '', '', ['shared_memory'], 'advanced'),
  ram_used_gib: E('The machine\'s RAM in use, in GiB.', 'Total minus what is available: it counts what programs hold, not the reclaimable cache.', '', '', '', ['host_ram'], 'basic'),
  ram_available_gib: E('RAM that programs could still use without swapping, in GiB.', 'It counts free memory plus cache the system can give back.', 'Keep an eye on it before loading a big model.', 'Low available RAM leads to swap and stalls.', '', ['host_ram', 'swap'], 'basic'),
  shmem_gib: E('Shared memory in use on the machine, in GiB.', 'The system cannot reclaim it, so large values reduce the memory left for everything else.', '', '', '', ['shared_memory'], 'advanced'),
  swap_used_gib: E('Disk swap in use, in GiB.', 'Any sustained use is a sign that RAM is short.', '', 'Heavy swapping makes engines stall.', '', ['swap'], 'advanced'),
  disk_free_gib: E('Free space on a drive, in GiB.', 'Model downloads and copies need room for the whole model.', '', 'A full drive makes downloads fail partway.', '', ['model_root'], 'basic'),
};

// ───────────────────────────────── concepts ─────────────────────────────────
const states = {
  'state.stopped': E('The engine is not running.', 'Nothing is loaded and no memory is held. Starting it again loads the model from disk.', '', '', '', ['engine'], 'basic'),
  'state.starting': E('The engine\'s container is starting up.', 'This stage is short: the container and engine program are launching.', '', '', '', ['container'], 'basic'),
  'state.loading': E('The model is being loaded into memory and prepared.', 'Big models take minutes: weights are read from disk, then the engine builds its caches and may record CUDA graphs.', 'Wait. The progress bar shows the current step.', 'If it never finishes, open the log: the usual causes are too little memory or a wrong setting.', '', ['weights', 'cuda_graph'], 'basic'),
  'state.ready': E('The engine is running and can answer requests.', 'The model is loaded and its health check passes. You can chat with it or send requests through its address.', '', '', '', ['engine'], 'basic'),
  'state.stopping': E('The engine is shutting down.', 'Its memory is released as soon as the container has stopped.', '', '', '', [], 'basic'),
  'state.failed': E('The engine stopped with an error.', 'The log shows the last lines before the failure. Common causes: not enough memory, an unsupported setting, or a model file that is missing.', 'Read the log, change the setting it names, and start again.', '', '', ['vram'], 'basic'),
  'state.auth_required': E('The engine is running but needs an API key that the console does not have.', 'It answers with "unauthorized", so the console can see it exists but cannot read its details or metrics.', '', '', '', ['engine'], 'basic'),
  'state.unreachable': E('The console cannot reach this engine right now.', 'It did not answer the last checks. It may be starting, overloaded, or stopped without the console noticing.', 'Wait a moment: the console keeps checking.', '', '', ['engine'], 'basic'),
};
const links = {
  'link.connecting': E('The console is opening its connection to this machine.', 'This takes a few seconds. If it stays here, the machine may be off or unreachable.', '', '', '', ['node'], 'basic'),
  'link.up': E('Connected: the machine is reporting normally.', 'The node answers and its readings are fresh.', '', '', '', ['node'], 'basic'),
  'link.stale': E('The connection exists but no new readings have arrived for a few seconds.', 'The numbers shown are the last ones received. The machine may be busy or the network slow.', 'Wait; if it persists, press Reconnect.', 'Do not trust live numbers while a machine is stale.', '', ['node'], 'basic'),
  'link.down': E('The console cannot reach this machine.', 'The message says why: login refused, host key problem, node not installed, or no route. The console retries automatically and the countdown shows when.', 'Fix the cause shown, then press Reconnect to skip the wait.', '', '', ['node', 'ssh_key'], 'basic'),
  'link.incompatible': E('The node on this machine speaks a different protocol version than the console.', 'The two versions cannot understand each other, so nothing is shown from that machine.', 'Update the node (or the console) so that both match.', '', '', ['node'], 'basic'),
  'link.disabled': E('This machine is turned off in the console\'s configuration.', 'It is listed but the console does not try to connect.', 'Set it to enabled in the hosts file and restart the console.', '', '', ['host'], 'basic'),
};
const managements = {
  'management.managed': E('The console started this engine and can start, stop, change and remove it.', 'It carries the console\'s label and runs on the private engine network.', '', '', '', ['container', 'engine'], 'basic'),
  'management.script': E('An engine that your own start script runs; the console uses that script\'s actions to start and stop it.', 'Scripts often encode rules such as "start the big model before the small one". The console never works around them.', 'Use the script\'s action buttons.', '', '', ['policy', 'engine'], 'basic'),
  'management.external': E('An engine started by something else. The console can watch it, and stop or start it only if the machine allows.', 'It was found by looking at running containers and open ports.', '', 'Settings shown are a best-effort read of its command line.', '', ['policy', 'engine'], 'basic'),
};
const verdicts = {
  'verdict.fits': E('Estimated to fit with comfortable room to spare.', 'The estimate adds the weights, the working memory for the chosen context and users, and some overhead, and compares it with the memory budget. Fits means at most 90% of the budget.', '', '', '', ['weights', 'kv_cache'], 'basic'),
  'verdict.tight': E('Estimated to fit, but with little room: it may fail or be unstable.', 'Between 90% and 100% of the budget. A long conversation or one more user could push it over.', 'Lower the context length or the number of parallel requests to get more room.', '', '', ['kv_cache', 'vram'], 'basic'),
  'verdict.wont_fit': E('Estimated not to fit in the memory available.', 'The model, its working memory and overhead need more than the budget. Starting it would most likely fail.', 'Choose a smaller or more compressed model, a shorter context, or a machine with more memory.', '', '', ['quantization', 'vram'], 'basic'),
  'verdict.unknown': E('There is not enough information to estimate the fit.', 'The model\'s details could not be read, so the console will not guess.', '', '', '', [], 'basic'),
};
const checks = {
  'check.host_down': E('The machine must be connected before anything can be started on it.', 'The console reaches every machine through its node. While the link is down, stale or connecting, nothing can be launched there.', 'Fix the connection shown on the machine, then try again.', '', '', ['host', 'node'], 'basic'),
  'check.monitor_only': E('The machine is in monitor mode, so the console may look but not change anything.', 'An administrator on that machine switches it to manage mode in its policy file. The console cannot do that for them.', '', '', '', ['policy'], 'basic'),
  'check.invalid_params': E('One or more settings are not valid for this engine.', 'The message names the setting and what is wrong, such as a number out of range or an unknown option.', 'Fix the setting named in the message.', '', '', ['engine'], 'basic'),
  'check.format_unsupported': E('This engine cannot load this kind of model file.', 'Each engine understands certain formats: llama.cpp loads GGUF files, Ollama its own store, vLLM and SGLang Hugging Face folders.', 'Pick an engine that supports the model\'s format.', '', '', ['gguf', 'engine'], 'basic'),
  'check.model_missing': E('The model must be stored on that machine, or on storage it can read.', 'Copy the model there from the Library, or download it onto that machine first.', '', '', '', ['model_root', 'nfs'], 'basic'),
  'check.fit_unavailable': E('The console could not estimate whether the model fits.', 'The model\'s details could not be read, so it will not guess.', 'You can still try; watch the log if it fails.', '', '', ['vram'], 'advanced'),
  'check.fit': E('The model and settings should fit in memory.', 'This repeats the fit estimate: weights, working memory and overhead against the memory budget.', '', '', '', ['vram', 'kv_cache'], 'basic'),
  'check.trust_remote_code': E('This setting lets code from the model repository run inside the engine.', 'Some new models need their own code. Only enable it for sources you trust.', '', 'Untrusted code runs in the engine container.', '', ['container'], 'advanced'),
  'check.port': E('A free network port was found for the engine.', 'The console picks one from the machine\'s allowed range and puts a gateway on it.', '', '', '', ['gateway'], 'advanced'),
  'check.no_free_port': E('There is no free port left in the machine\'s allowed range.', 'Each engine needs one. Stop an engine to free its port.', 'Stop an engine you no longer need.', '', '', ['gateway'], 'basic'),
  'check.no_gpu': E('The machine reports no GPU.', 'Engines here need a graphics card. The driver may be missing or not loaded.', '', '', '', ['gpu'], 'basic'),
  'check.gpu_unknown': E('You chose a GPU the machine does not have.', 'GPU numbers are per machine, starting at 0.', 'Choose from the GPUs listed for that machine.', '', '', ['gpu'], 'basic'),
  'check.gpu_shared': E('Another engine already uses the chosen GPU.', 'Two engines can share a card if both fit; free memory is the limit.', 'Choose a different GPU or stop the other engine if the fit check fails.', '', '', ['gpu', 'vram'], 'basic'),
  'check.image_not_allowed': E('This machine\'s policy does not allow the engine image.', 'Each machine lists the images it will run. Pick one that is allowed, or ask the administrator to add it.', 'Choose another image.', '', '', ['image', 'policy'], 'basic'),
  'check.image_check_failed': E('The console could not check whether the image is on the machine.', 'The machine did not answer the question. Try again.', '', '', '', ['image'], 'advanced'),
  'check.image_missing': E('The engine image is not on the machine yet.', 'The console never downloads images itself. Pull it on that machine, then launch again.', 'Pull the image on that machine.', '', '', ['image'], 'basic'),
  'check.gateway_missing': E('The machine\'s policy names no gateway image.', 'Every engine gets a small gateway in front of it. Without one the engine cannot be reached.', '', '', '', ['gateway', 'policy'], 'advanced'),
  'check.tp_gt_gpus': E('Tensor parallelism cannot use more GPUs than you selected.', 'A value of 2 needs two GPUs.', 'Select more GPUs or lower the value.', '', '', ['tensor_parallelism'], 'basic'),
  'check.nvfp4_moe_sm120': E('4-bit mixture-of-experts models are slow with this engine on this GPU generation.', 'The engine falls back to a slower kernel for them.', 'Use an FP8 version of the model, or another engine.', '', '', ['nvfp4', 'moe'], 'advanced'),
  'check.mxfp4_moe_sm120': E('This 4-bit format is not supported for this model on this GPU generation.', 'The engine stops with an error when it meets it.', 'Choose a different model format.', '', '', ['quantization', 'moe'], 'advanced'),
  'check.gated': E('This model needs a Hugging Face account and an accepted licence.', 'Downloads fail until the licence is accepted and a token is set on the console.', '', '', '', [], 'advanced'),
};
const formats = {
  'format.hf': E('A Hugging Face model folder, found by its repository name such as org/model.', 'The weights are safetensors files in a Hugging Face cache; vLLM and SGLang load them directly.', '', '', '', ['hf_cache', 'safetensors'], 'basic'),
  'format.gguf': E('A single-file model, used by llama.cpp.', 'The quantization is part of the file name, and one repository often has several.', '', '', '', ['gguf', 'quantization'], 'basic'),
  'format.ollama': E('A model stored inside an Ollama server, named like name:tag.', 'Ollama keeps its own store; the console lists what it reports.', '', '', '', ['engine'], 'basic'),
  'format.plain': E('A model folder with weights and a config.json that is not in a Hugging Face cache.', 'For example your own fine-tune. Engines load it from the folder.', '', '', '', ['model_root'], 'basic'),
  'format.other': E('A model layout the console does not recognise.', 'It is listed by size only and cannot be launched from the console.', '', '', '', [], 'advanced'),
};
const roots = {
  'root.auto': E('The console works out what kind of folder this is.', '', '', '', '', ['model_root'], 'advanced'),
  'root.hf-home': E('A Hugging Face home folder; downloads go into its hub subfolder.', '', '', '', '', ['hf_cache'], 'basic'),
  'root.hf-hub': E('A Hugging Face hub cache folder (models--org--name entries).', '', '', '', '', ['hf_cache'], 'basic'),
  'root.plain': E('A folder where each model sits in its own subfolder.', '', '', '', '', ['model_root'], 'basic'),
  'root.gguf': E('A folder of GGUF files.', '', '', '', '', ['gguf'], 'basic'),
  'root.ollama': E('The folder where an Ollama server keeps its models.', 'Normally only the Ollama server writes here.', '', '', '', ['engine'], 'advanced'),
  'root.other': E('A folder of another kind; only its size is shown.', '', '', '', '', [], 'advanced'),
};
const alerts = {
  'alert.host_ram_low': E('The machine is running low on free RAM.', 'It warns below 10% available and turns critical below 5%. Starting more engines now risks swapping.', 'Stop an engine you do not need, or pick a smaller model.', '', '', ['host_ram'], 'basic'),
  'alert.swap_heavy': E('A lot of swap is in use.', 'The system has moved a large part of memory to disk, which makes engines stall.', 'Free memory by stopping something.', '', '', ['swap'], 'basic'),
  'alert.shmem_high': E('A large share of RAM is shared memory that cannot be reclaimed.', 'It reduces the memory left for new engines even though it can look like cache.', '', '', '', ['shared_memory'], 'advanced'),
};
const pages = {
  'page.dashboard': E('Every machine at a glance: its GPUs, memory, drives and the engines running on it.', 'Each card is one machine. Bars are split by engine so you can see who is using what.', '', '', '', ['host', 'vram'], 'basic'),
  'page.host': E('Everything about one machine: connection, policy, hardware, drives, images and scripts.', '', '', '', '', ['node', 'policy'], 'basic'),
  'page.library': E('All models stored on your machines, with where each copy lives.', 'Launch a model from here, copy it to another machine, or delete a copy.', '', '', '', ['model_root', 'nfs'], 'basic'),
  'page.discover': E('Search Hugging Face and download a model onto a machine of your choice.', '', '', '', '', ['hf_cache', 'gguf'], 'basic'),
  'page.downloads': E('Progress of model downloads and copies between machines.', '', '', '', '', ['model_root'], 'basic'),
  'page.launch': E('Start an engine with full control over every setting.', 'The guided launch walks you through it step by step.', '', '', '', ['engine', 'preset'], 'basic'),
  'page.wizard': E('A step-by-step way to start a model: choose a model, a machine and a goal.', '', '', '', '', ['preset'], 'basic'),
  'page.hosts': E('All the machines the console manages or watches, with their connection state and what they hold.', '', '', '', '', ['host'], 'basic'),
  'page.instance': E('One engine in detail: its state, settings, logs, live metrics and the buttons that apply to it.', '', '', '', '', ['engine'], 'basic'),
  'page.instances': E('The engines on all machines: running, stopped or monitored.', '', '', '', '', ['engine'], 'basic'),
  'page.metrics': E('Charts of speed, delay, queue and memory use over time.', '', '', '', '', ['throughput', 'latency'], 'basic'),
  'page.bench': E('Measure how fast an engine really is, and compare runs.', '', '', '', '', ['throughput', 'ttft'], 'basic'),
  'page.chat': E('Talk to a running engine to try it out.', '', '', '', '', ['model'], 'basic'),
  'page.arena': E('Put two engines side by side and vote on the better answer.', '', '', '', '', ['model'], 'basic'),
  'page.usage': E('How much each model has been used.', '', '', '', '', ['token'], 'basic'),
  'page.learn': E('Plain-language explanations of every term the console uses.', '', '', '', '', [], 'basic'),
  'page.settings': E('Console preferences: beginner mode, theme, keys and image choices.', '', '', '', '', [], 'basic'),
};
const fields = {
  'field.host': E('The machine the engine will run on.', 'Each machine has its own GPUs, memory and models, so this decides what is possible.', '', '', '', ['host'], 'basic'),
  'field.gpus': E('Which of the machine\'s graphics cards the engine may use.', 'The free memory on each card is shown. Choose more than one for models that need to be split.', '', 'Cards already used by another engine may not have room.', '', ['gpu', 'tensor_parallelism'], 'basic'),
  'field.root': E('The folder on the machine where the files will be stored.', 'Pick one with enough free space; the free space of its drive is shown.', '', '', '', ['model_root'], 'basic'),
  'field.image': E('Which packaged version of the engine to run.', 'The default is the one pinned for that machine. Only images the machine allows are listed.', 'Leave it on the default unless you need another version.', '', '', ['image', 'policy'], 'advanced'),
  'field.engine': E('The program that will run the model.', 'Only engines that can load this model\'s format are offered.', '', '', '', ['engine'], 'basic'),
  'field.model': E('The model to run, from this machine\'s library.', 'Models that are not on the chosen machine are not listed: copy them there first.', '', '', '', ['model', 'model_root'], 'basic'),
  'field.preset': E('A ready-made group of settings for a goal.', '', '', '', '', ['preset'], 'basic'),
  'field.concurrency': E('How many requests at the same moment the memory estimate should plan for.', 'It changes the size of the KV cache that is reserved.', 'Set it to the number of people or programs that will use the engine at once.', '', 'Four people chatting: 4.', ['concurrency', 'kv_cache'], 'basic'),
  'field.name': E('A label for this engine so you can recognise it.', 'Optional. Letters, digits, spaces, dots and dashes.', '', '', '', [], 'basic'),
  'field.ttl': E('Stop the engine automatically after this many idle minutes.', 'Frees the GPU when nobody has used the engine for a while. Zero means never.', '', '', '', ['vram'], 'advanced'),
  'field.pin': E('Keep this engine from being stopped automatically.', 'Pinned engines ignore the idle timer.', '', '', '', [], 'advanced'),
  'field.quant': E('Which compressed version of a GGUF model to download.', 'Smaller files are faster and use less memory but lose a little accuracy.', '', '', 'Q4_K_M is a popular middle ground.', ['gguf', 'quantization'], 'basic'),
  'field.policy_mode': E('Whether the console may change anything on this machine.', 'Monitor mode is read-only. Manage mode lets the console start and stop engines, within the policy.', '', '', '', ['policy'], 'basic'),
  'field.machine_filter': E('Show only models on, or reachable from, one machine.', '', '', '', '', ['host'], 'basic'),
  'field.format_filter': E('Show only one kind of model file.', '', '', '', '', [], 'basic'),
  'field.target_host': E('The machine that will receive a download or a copy.', 'Files are written into one of that machine\'s writable model folders; its free disk space decides whether they fit.', '', '', '', ['host', 'model_root'], 'basic'),
  'field.beginner_mode': E('A setting that shows short explanations next to everything and guides launching step by step.', 'With it on, each summary is shown beside its setting instead of behind the info button, and launching starts as a guided wizard.', 'Turn it off once the screens feel familiar.', '', '', [], 'basic'),
  'field.profile': E('A saved set of settings for a model and an engine, which you can load again, compare and share.', '', '', '', '', ['engine'], 'basic'),
};
export const concepts = { ...states, ...links, ...managements, ...verdicts, ...checks, ...formats, ...roots, ...alerts, ...pages, ...fields };

// ───────────────────────────────── params ─────────────────────────────────
// Shared wording for parameters that more than one engine has.
const quantizationE = E('Force a quantization method for the weights instead of the one the model declares.', 'Models usually announce their own format and the engine detects it. Setting it here overrides that, for example to quantize a full-precision model on the fly.', 'Leave it unset unless the engine fails to detect the format or you want to compress a full-size model.', 'A method the model was not made for fails at start-up or gives poor answers.', 'fp8 stores each weight in 8 bits, about half the memory of 16 bits.', ['quantization', 'fp8']);
const kvDtypeE = E('The number format used to store the KV cache.', 'Using FP8 instead of 16 bits halves the memory each conversation needs, so the same card holds twice the context or users, at a small cost in precision.', 'Try FP8 when you need longer context or more users and memory is tight.', 'Very long answers may drift slightly in quality.', 'fp8 doubles the number of tokens that fit in the same cache.', ['kv_cache', 'fp8']);
const cpuOffloadE = E('How many GB of the model to keep in ordinary RAM instead of on the GPU.', 'Weights kept in RAM are copied to the card as each step needs them, so a model slightly too big for the card can run, much more slowly.', 'Only when a model does not fit and you accept slow answers.', 'Every step waits on the transfer, so speed drops sharply, and RAM must be free.', '16 offloads 16 GB per GPU to RAM.', ['offload', 'host_ram']);
const toolParserE = E('The reader that recognises the model\'s tool calls in its output.', 'Each model family writes tool calls in its own format. The matching parser lets programs that use functions work with the engine.', 'Set it, with auto tool choice, when an agent or app needs the model to call tools.', 'The wrong parser makes tool calls come back as plain text.', 'hermes for Hermes-style models.', ['tool_calling'], 'basic');
const reasoningParserE = E('Separates the model\'s thinking from its final answer.', 'Reasoning models write their thinking first. With the right parser the engine puts it in its own field so apps can hide or show it.', 'Set it for reasoning models if you want clean answers.', 'The wrong parser can cut text out of the answer.', 'qwen3 for Qwen3 family models.', ['reasoning'], 'basic');
const trustE = E('Allow code that ships with the model repository to run.', 'Some new models need their own code to be loaded. Running it means trusting whoever published it.', 'Only when a model refuses to load without it and you trust its source.', 'Untrusted code runs inside the engine container with access to its files and memory.', '', ['container']);
const chatTemplateE = E('Replace the formatting template that turns a conversation into the text the model reads.', 'Every chat model expects its messages wrapped in a specific format. The model normally carries the right one.', 'Only for models that ship without a template, or to test a changed one.', 'A wrong template makes answers confused or breaks tool calls.', '', ['model']);
const apiKeyE = E('A password clients must send to use this engine.', 'It is passed to the engine by name through its environment, so the value never appears in the command line or the console\'s logs.', 'Set it if the engine\'s address is reachable by others on the network.', 'Programs that do not know the key will be refused.', '', ['engine']);

const vllm = {
  max_model_len: E('The longest conversation, in tokens, that this engine will accept: your question plus the answer.', 'Everything the model reads and writes in one conversation has to fit inside this window. About 1,000 tokens is 750 English words, or one and a half pages. The engine reserves memory for this length up front, so a bigger window leaves less room for other users at the same time.', 'Raise it when you paste long documents or run agents with long histories. Lower it to fit more users or a bigger model on the card.', 'Too high: the model may not fit, or the engine refuses to start. Too low: long inputs are rejected with an error.', '32768 holds about 50 pages of text; 131072 about 200 pages.', ['context_window', 'token', 'kv_cache'], 'basic'),
  gpu_memory_utilization: E('How much of each graphics card\'s memory this engine may claim.', 'At start-up the engine reserves this share of every GPU it uses: room for the weights, and the rest becomes the KV cache. It takes the whole share at once, even when idle, much like booking a whole room rather than a seat.', 'Lower it when another program or a second engine shares the card. Raise it a little when you need longer conversations or more users and the card is otherwise free.', 'Too high: start-up fails because the memory is already in use, or other programs run out of memory. Too low: only short conversations fit.', '0.85 on a 96 GB card reserves about 82 GB and leaves about 14 GB.', ['vram', 'kv_cache', 'weights'], 'basic'),
  tensor_parallel_size: E('How many graphics cards share the work of running one model.', 'Splitting a model across cards lets you run one that is too big for a single card, and it can speed up replies. The cards exchange data constantly, so it works best with a fast link between them.', 'Use 2 or more only when the model does not fit on one card. Leave it at 1 otherwise.', 'It cannot exceed the number of GPUs you selected, or the engine fails to start. Cards that are also used by other programs waste memory.', '2 on a machine with two 96 GB cards leaves room for a model about twice as large as one card holds.', ['tensor_parallelism', 'gpu'], 'basic'),
  pipeline_parallel_size: E('Splits the model into consecutive stages, each on its own card, like stations on an assembly line.', 'It needs less communication between cards than tensor parallelism but leaves them idle part of the time.', 'Rarely: when the cards are connected slowly, or across machines.', 'Lower throughput for single requests and extra complexity.', '', ['pipeline_parallel'], 'expert'),
  enable_expert_parallel: E('For mixture-of-experts models: place whole experts on different GPUs instead of splitting each one.', 'Each GPU holds some experts and handles only the tokens routed to them.', 'Large mixture-of-experts models on several GPUs, after measuring.', 'Uneven load between GPUs, and it does nothing for models that are not mixtures of experts.', '', ['moe', 'expert_parallel'], 'expert'),
  dtype: E('The number format for the parts of the model that are not already quantized.', 'Auto uses what the model was saved in. Half-precision formats use 2 bytes per weight; full precision uses 4.', 'Leave it on auto. Change it only to match a model that needs a specific format.', 'Float32 doubles memory and is almost never needed; a wrong choice can reduce accuracy.', 'bfloat16 keeps a wide range of values at 2 bytes each.', ['weights', 'quantization'], 'advanced'),
  quantization: quantizationE,
  kv_cache_dtype: kvDtypeE,
  max_num_seqs: E('The most conversations the engine will answer at the same moment.', 'Extra requests wait in a queue. Each running conversation reserves KV cache, so a higher limit needs more memory when many are active.', 'Raise it for many users at once; lower it for fewer, faster replies or to save memory.', 'Too high with long contexts uses all the KV cache and slows everyone; too low makes users wait.', '16 suits one team; 256 suits a busy service.', ['concurrency', 'batch', 'kv_cache'], 'basic'),
  max_num_batched_tokens: E('How many tokens the engine processes in one step across all requests.', 'Bigger steps raise total throughput but make each step slower, so a user may wait longer between tokens.', 'Tune it when measuring throughput against delay. Smaller favours smooth streaming, bigger favours bulk work.', 'Too small slows long prompts; too big makes streaming jerky.', '', ['batch', 'throughput', 'itl'], 'advanced'),
  enable_prefix_caching: E('Remember the start of prompts so repeated beginnings are not recomputed.', 'Agents and chats with a long fixed instruction send the same opening every time. With the cache on, the engine reads it once and reuses it.', 'Leave it on. Turn it off only to test or when requests never share a beginning.', 'It uses some memory that could hold more conversations.', '', ['prefix_cache', 'kv_cache'], 'basic'),
  enable_chunked_prefill: E('Process long prompts in slices mixed in with other requests.', 'A very long prompt can otherwise stall everybody for a moment. In slices, answers to others keep streaming smoothly.', 'Leave it on for shared use.', '', '', ['chunked_prefill', 'prefill'], 'advanced'),
  enforce_eager: E('Switch off the pre-recorded GPU steps (CUDA graphs) and run each step directly.', 'Recording saves time per token but costs start-up time and some memory. Without it the engine starts faster and uses less memory but decodes more slowly.', 'To save a little memory, to debug, or when recording fails.', 'Slower answers, often noticeably.', '', ['cuda_graph', 'decode'], 'advanced'),
  max_cudagraph_capture_size: E('The largest batch size that gets a pre-recorded GPU step.', 'Larger batches are recorded only up to this size; bigger ones run the normal way.', 'Lower it to cut start-up time and memory when you never serve large batches.', '', '', ['cuda_graph', 'batch'], 'expert'),
  cpu_offload_gb: cpuOffloadE,
  swap_space: E('RAM per GPU set aside to park paused conversations.', 'When the KV cache is full the engine can move a paused conversation here instead of throwing its work away.', '', 'Uses host RAM; too much can starve other programs.', '', ['kv_cache', 'preemption', 'host_ram'], 'expert'),
  block_size: E('How many tokens each KV cache block holds.', 'The cache is stored in fixed-size blocks, like pages. Different sizes trade bookkeeping against waste.', 'Almost never; the default suits the engine\'s kernels.', 'An unsupported size fails at start-up.', '', ['kv_cache', 'token'], 'expert'),
  enable_auto_tool_choice: E('Let the model decide on its own when to call a tool.', 'This is what makes agent-style apps work. It needs a tool-call parser that matches the model.', 'Turn it on for coding agents and apps that use tools.', 'Without the right parser, tool calls appear as plain text.', '', ['tool_calling'], 'basic'),
  tool_call_parser: toolParserE,
  reasoning_parser: reasoningParserE,
  speculative_config: E('Settings for speculative decoding: which helper to use and how many tokens it guesses.', 'A small helper proposes several tokens and the main model checks them at once. It is entered as a small JSON object.', 'When you want faster answers for a single user and have a suitable helper model.', 'A badly matched helper makes things slower and uses extra memory.', '{"method": "ngram", "num_speculative_tokens": 4}', ['speculative_decoding'], 'advanced'),
  limit_mm_per_prompt: E('How many images or videos one request may contain.', 'Models that understand pictures need memory for each one. A limit stops one request from using it all.', 'Raise it only for multi-image tasks; lower it to save memory.', '', '{"image": 4} allows four images per request.', ['kv_cache'], 'advanced'),
  trust_remote_code: trustE,
  served_model_name: E('The name the engine reports for its model.', 'Programs that call the engine use this name in their requests.', 'When an app expects a particular model name.', '', '', ['model'], 'advanced'),
  allowed_origins: E('Which web pages may call this engine directly from a browser.', 'It sets the CORS rule. Programs on servers are not affected by it.', '', 'Allowing everything lets any web page you open call the engine.', '', ['engine'], 'expert'),
  chat_template: chatTemplateE,
  seed: E('A starting number for the random choices the model makes while writing.', 'The same seed with the same input gives repeatable answers.', 'For tests that must repeat.', '', '', ['token'], 'expert'),
  hf_token: E('Your Hugging Face access token, for models that need a login.', 'It is passed to the container by name and never shown again; the console only knows that one is set.', 'Only for gated or private models that you may use.', 'Anyone with the token has your account\'s access.', '', ['hf_cache'], 'advanced'),
  api_key: apiKeyE,
  nccl_debug: E('How much detail the GPU-to-GPU communication library logs.', 'It is for diagnosing problems with several GPUs.', 'Only when debugging a multi-GPU start-up problem.', 'INFO and TRACE write a lot of text to the log.', '', ['tensor_parallelism'], 'expert'),
  generation_config: E('Where the default writing settings come from: the model\'s own file or the engine.', 'A model may ship its preferred temperature and other values. Auto uses them.', '', '', '', ['model'], 'expert'),
};

const sglang = {
  context_length: E('The longest conversation, in tokens, that this engine will accept: your question plus the answer.', 'Everything the model reads and writes in one conversation has to fit inside this window. About 1,000 tokens is 750 English words. The engine reserves memory for it, so a bigger window leaves less for other users.', 'Raise it for long documents and agents; lower it to fit more users or a bigger model.', 'Too high: the model may not fit. Too low: long inputs are rejected.', '32768 holds about 50 pages of text.', ['context_window', 'token', 'kv_cache'], 'basic'),
  mem_fraction_static: E('How much of each card\'s memory goes to the model and its conversation cache.', 'Everything the engine reserves for weights and the KV cache is taken at start-up. What is left is working space for the steps themselves.', 'Lower it if start-up runs out of memory or another program shares the card. Raise it a little for more users when the card is free.', 'Too high leaves too little working space and requests fail; too low leaves little room for conversations.', '0.88 reserves 88% of each card for weights and cache.', ['vram', 'kv_cache', 'weights'], 'basic'),
  tp_size: E('How many graphics cards share the work of running one model.', 'Splitting a model across cards lets you run one that is too big for a single card, and it can speed up replies. The cards exchange data constantly.', 'Use 2 or more only when the model does not fit on one card.', 'It cannot exceed the number of GPUs selected.', '2 on two 96 GB cards.', ['tensor_parallelism', 'gpu'], 'basic'),
  dp_size: E('Run several full copies of the model, each on its own group of GPUs, to serve more users.', 'Each copy needs its own memory for the weights. Requests are shared out between the copies.', 'When one copy fits on fewer GPUs and you want more total capacity.', 'Uses more memory in total.', '', ['throughput'], 'expert'),
  ep_size: E('For mixture-of-experts models: how many GPUs the experts are spread across.', 'Each GPU holds a share of the experts.', 'Large mixture-of-experts models on several GPUs.', '', '', ['moe', 'expert_parallel'], 'expert'),
  quantization: quantizationE,
  kv_cache_dtype: kvDtypeE,
  max_running_requests: E('The most requests the engine will work on at the same moment.', 'Extra requests wait in a queue. Each running one reserves cache, so a higher limit needs more memory when many are active.', 'Raise it for many users; lower it for fewer, faster replies.', 'Too high with long contexts can use all the cache and slow everyone.', '16 suits a small team.', ['concurrency', 'kv_cache'], 'basic'),
  max_total_tokens: E('A cap on the total number of tokens the cache can hold.', 'Normally the engine works this out from the memory available; setting it forces a smaller cache.', 'To leave memory for something else without changing the memory share.', 'Too small limits how many conversations fit.', '', ['kv_cache', 'token'], 'advanced'),
  chunked_prefill_size: E('How many prompt tokens are read in one slice.', 'A long prompt is read in slices so that other requests keep streaming between them.', 'Smaller values keep streaming smoother; larger ones read long prompts faster.', '', '', ['chunked_prefill', 'prefill'], 'advanced'),
  schedule_policy: E('The order in which waiting requests are started.', 'First come first served is fair. Other policies group similar prompts to reuse the cache.', '', '', '', ['prefix_cache'], 'advanced'),
  enable_mixed_chunk: E('Allow reading prompt slices and writing answers within the same step.', 'It can raise throughput under heavy mixed load.', 'Only after measuring.', 'May change delay behaviour.', '', ['chunked_prefill', 'batch'], 'expert'),
  attention_backend: E('Which kernel the engine uses for the attention calculation.', 'Different kernels suit different models and GPU generations.', 'When the default fails for a model or you have benchmarked a better one.', 'An unsupported choice fails at start-up.', '', ['compute_capability', 'gpu'], 'expert'),
  disable_radix_cache: E('Turn off remembering the start of prompts.', 'With it on, requests with the same beginning reuse earlier work.', 'Only for tests or when no requests ever share a beginning.', 'Slower first words for repeated prompts.', '', ['prefix_cache'], 'advanced'),
  disable_cuda_graph: E('Switch off the pre-recorded GPU steps and run each step directly.', 'Saves some memory and start-up time but decodes more slowly.', 'To save a little memory or debug.', 'Slower answers.', '', ['cuda_graph', 'decode'], 'advanced'),
  cuda_graph_max_bs: E('The largest batch that gets a pre-recorded GPU step.', 'Bigger batches run the normal way.', 'Lower it to save start-up time and memory.', '', '', ['cuda_graph', 'batch'], 'expert'),
  enable_torch_compile: E('Compile the model\'s code for extra speed.', 'It takes longer to start and may speed up each step.', 'After measuring on your model.', 'Longer start-up and occasional compile failures.', '', ['decode'], 'expert'),
  cpu_offload_gb: cpuOffloadE,
  tool_call_parser: toolParserE,
  reasoning_parser: reasoningParserE,
  speculative_algorithm: E('Which speculative decoding method to use for faster answers.', 'A cheap guesser proposes tokens that the main model verifies in one step.', 'For faster single-user answers when a draft model or method suits your model.', 'Wrong choices slow things down and cost memory.', '', ['speculative_decoding'], 'advanced'),
  speculative_num_steps: E('How many tokens ahead the helper guesses.', 'More steps pay off when the guesses are usually right.', '', '', '', ['speculative_decoding'], 'expert'),
  trust_remote_code: trustE,
  chat_template: chatTemplateE,
  random_seed: E('A starting number for the random choices made while writing.', 'The same seed with the same input gives repeatable answers.', 'For tests that must repeat.', '', '', ['token'], 'expert'),
  api_key: E('A password clients must send to use this engine.', 'This engine only accepts it as a command-line argument, which would reveal it in the process list, so the console refuses to launch with one.', 'Use an engine or a gateway that reads it from the environment.', '', '', ['engine']),
};

const llamacpp = {
  ctx_size: E('The total context, in tokens, shared between all parallel slots.', 'Each parallel slot gets an equal share, so two slots with 16384 tokens need 8192 each.', 'Raise it for long documents.', 'Large values use much more memory.', '', ['context_window', 'kv_cache'], 'basic'),
  n_gpu_layers: E('How many layers of the model to put on the GPU.', 'A large number moves the whole model onto the card. Layers left over run on the processor, which is slow.', 'Lower it only when the model does not fit in the card\'s memory.', 'Too few layers on the GPU makes answers slow.', '', ['gpu', 'offload'], 'basic'),
  parallel: E('How many conversations the server answers at once.', 'The context is divided between them.', 'Raise it for several users.', 'Each slot gets a smaller context.', '', ['concurrency'], 'basic'),
  flash_attn: E('Use the memory-saving attention kernel.', 'It lowers memory use for long contexts and is usually faster.', '', '', '', ['kv_cache']),
  cache_type_k: E('The number format used for the cached attention keys.', 'Smaller formats save memory at a small cost in quality.', '', '', '', ['kv_cache', 'quantization']),
  cache_type_v: E('The number format used for the cached attention values.', 'Smaller formats save memory at a small cost in quality.', '', '', '', ['kv_cache', 'quantization']),
  split_mode: E('How a model is divided between several GPUs.', 'Layer mode puts whole layers on each card; row mode splits each layer.', '', '', '', ['tensor_parallelism']),
  threads: E('How many processor threads help with the work.', 'Mostly relevant when part of the model runs on the processor.', '', '', '', []),
  batch_size: E('The largest group of prompt tokens read together.', '', '', '', '', ['batch', 'prefill'], 'expert'),
  ubatch_size: E('The physical chunk size inside a batch.', '', '', '', '', ['batch'], 'expert'),
  mlock: E('Keep the model in RAM so the system cannot move it to swap.', '', '', 'Needs enough free RAM.', '', ['host_ram', 'swap'], 'expert'),
  no_mmap: E('Read the whole model file into memory instead of mapping it.', '', '', 'Slower start and more RAM use.', '', ['host_ram'], 'expert'),
  jinja: E('Apply the chat template stored inside the model file.', 'Needed for tool calling with many models.', 'Turn it on for agents.', '', '', ['tool_calling']),
  alias: E('The name the server reports for its model.', '', '', '', '', ['model']),
};
const ollama = {
  num_parallel: E('How many requests each loaded model handles at once.', 'More parallel requests use more memory for each loaded model.', '', '', '', ['concurrency'], 'basic'),
  max_loaded_models: E('How many different models may stay in memory together.', 'Ollama loads models on demand and unloads idle ones.', 'Raise it when you switch between several models often and have the memory.', 'Memory fills up quickly with large models.', '', ['model', 'vram'], 'basic'),
  keep_alive: E('How long an idle model stays loaded.', 'A short time frees memory sooner; a long time makes the next answer start faster.', '', '', '5m keeps it five minutes; -1 keeps it forever.', ['vram'], 'basic'),
  context_length: E('The default context, in tokens, for models that do not set their own.', '', '', 'Large values use much more memory.', '', ['context_window'], 'basic'),
  flash_attention: E('Use the memory-saving attention kernel.', '', '', '', '', ['kv_cache']),
  kv_cache_type: E('The number format used for the conversation cache.', 'Smaller formats save memory at a small cost in quality.', '', '', '', ['kv_cache', 'quantization']),
  max_queue: E('How many requests may wait before the server answers busy.', '', '', '', '', ['concurrency'], 'expert'),
};
const trtllm = {
  tp_size: E('How many graphics cards share the work of running one model.', 'It lets a model that is too big for one card run.', 'Use more than 1 only when the model does not fit one card.', '', '', ['tensor_parallelism'], 'basic'),
  pp_size: E('Splits the model into consecutive stages, each on its own card.', '', 'Rarely: across machines or slow links.', '', '', ['pipeline_parallel'], 'expert'),
  ep_size: E('For mixture-of-experts models: how many GPUs the experts are spread across.', '', '', '', '', ['moe', 'expert_parallel'], 'expert'),
  max_batch_size: E('The largest number of requests in one step.', '', '', '', '', ['batch', 'concurrency'], 'basic'),
  max_seq_len: E('The longest sequence, prompt plus answer, in tokens.', '', 'Raise for long documents.', 'Large values reserve a lot of memory.', '', ['context_window'], 'basic'),
  kv_cache_free_gpu_memory_fraction: E('The share of the memory left after loading that the KV cache may use.', '', 'Lower it if other programs share the card.', '', '', ['kv_cache', 'vram'], 'basic'),
  max_num_tokens: E('The token budget for one scheduler step.', '', '', '', '', ['batch', 'throughput']),
  backend: E('Which runtime executes the model.', '', '', '', '', ['engine']),
  reasoning_parser: reasoningParserE,
  trust_remote_code: trustE,
};

export const params = { vllm, sglang, llamacpp, ollama, trtllm };

// ───────────────────────────────── presets ─────────────────────────────────
const lowLatency = E('Quick replies for one person chatting.', 'A shorter context and fewer parallel requests keep each step fast and leave memory free.', '', 'Long documents will not fit.', '', ['latency', 'context_window'], 'basic');
const balanced = E('Sensible settings for everyday use by one or a few people.', 'A medium context window and prefix caching, with the engine\'s defaults for the rest.', '', '', '', ['context_window', 'prefix_cache'], 'basic');
const throughput = E('Serve many users at once; each may wait a little longer.', 'A larger share of memory, many parallel requests and a compressed KV cache make the engine fit as much work as possible.', '', 'Single-user replies are not faster, and the context is shorter.', '', ['concurrency', 'throughput', 'kv_cache'], 'basic');
const longContext = E('Work with very long documents or conversations.', 'A long context window and a compressed cache, with few parallel requests so the memory goes to length.', '', 'Few users at once, and loading is slower.', '', ['context_window', 'kv_cache'], 'basic');
const toolAgent = E('Run a coding assistant or agent that calls tools.', 'Turns on tool calling with the matching parser and a long context, and keeps the prompt cache on for fast repeated instructions.', '', 'The parser must match the model family.', '', ['tool_calling', 'prefix_cache'], 'basic');
export const presets = {
  vllm: { 'low-latency': lowLatency, balanced, 'max-throughput': throughput, 'long-context': longContext, 'tool-agent': toolAgent },
  sglang: { 'low-latency': lowLatency, balanced, 'max-throughput': throughput, 'long-context': longContext, 'tool-agent': toolAgent },
  llamacpp: { 'low-latency': lowLatency, balanced, 'max-throughput': throughput, 'long-context': longContext, 'tool-agent': toolAgent },
  ollama: { 'low-latency': E('Quick replies for one person; the model stays loaded between questions.', '', '', '', '', ['latency'], 'basic'), balanced, 'long-context': longContext },
  trtllm: { 'low-latency': lowLatency, balanced, 'max-throughput': throughput, 'long-context': longContext },
};

export const learnBundle = () => ({ version: 1, glossary, metrics, concepts, params, presets });

/**
 * Dev only: assemble the bundle exactly as the hub does (backend learn/mod.rs: validate each file, fill the model defaults)
 * from a directory of the real content files, so screenshots and checks can run on the production wording instead of the mock's.
 */
export function bundleFromDir(dir) {
  const read = (file, section) => {
    if (!fs.existsSync(file)) return {};
    const doc = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (!doc || doc.version !== 1 || typeof doc[section] !== 'object') throw new Error(`${path.basename(file)}: expected {"version": 1, "${section}": {...}}`);
    return doc[section];
  };
  const explain = (e) => ({ details: '', when_to_change: '', risks: '', example: '', terms: [], level: 'advanced', ...e });
  const map = (o, f) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, f(v)]));
  const perEngine = (sub, section) => {
    const d = path.join(dir, sub);
    return fs.existsSync(d) ? Object.fromEntries(fs.readdirSync(d).filter((f) => f.endsWith('.json')).sort().map((f) => [f.replace(/\.json$/, ''), map(read(path.join(d, f), section), explain)])) : {};
  };
  return {
    version: 1,
    glossary: map(read(path.join(dir, 'glossary.json'), 'terms'), (t) => ({ long: '', see_also: [], ...t })),
    metrics: map(read(path.join(dir, 'metrics.json'), 'metrics'), explain),
    concepts: map(read(path.join(dir, 'concepts.json'), 'concepts'), explain),
    params: perEngine('params', 'params'),
    presets: perEngine('presets', 'presets'),
  };
}
