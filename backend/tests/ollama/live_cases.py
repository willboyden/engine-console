"""The inputs of the live environment check (fixtures/live_env_0.30.11.json): a bare `ollama serve` was started in a
throwaway container (no network, no GPU, no model, an empty read-only store) once with the adapter's launch environment for
ADAPTER_PARAMS and once per entry of FORMS, and the effective values from its "server config" line were recorded.
`test_live_env.py` checks that the adapter still produces exactly this environment; if it does not, repeat the check."""
from __future__ import annotations

from typing import Any

# One of every parameter, so every variable the adapter can set was exercised.
ADAPTER_PARAMS: dict[str, Any] = {
    "context_length": 32768, "num_parallel": 4, "max_loaded_models": 2, "keep_alive": "-1", "kv_cache_type": "q8_0",
    "flash_attention": True, "gpu_overhead": 2147483648, "sched_spread": True, "max_queue": 128, "load_timeout": "30m",
    "origins": ["https://chat.example.org", "http://localhost:3000"], "llm_library": "cuda_v13", "igpu_enable": True,
    "fit": "off", "fit_target": 2048, "max_transfer_streams": 8, "debug": True,
}

# Value forms the adapter's validation accepts or rejects, and what the server did with each.
FORMS: list[tuple[str, dict[str, str]]] = [
    ("bools_zero", {"OLLAMA_FLASH_ATTENTION": "0", "OLLAMA_SCHED_SPREAD": "0", "OLLAMA_DEBUG": "0"}),
    ("bools_one", {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_SCHED_SPREAD": "1", "OLLAMA_DEBUG": "1"}),
    ("bools_false", {"OLLAMA_FLASH_ATTENTION": "false", "OLLAMA_SCHED_SPREAD": "false", "OLLAMA_DEBUG": "false"}),
    ("igpu_zero", {"OLLAMA_IGPU_ENABLE": "0"}),
    ("igpu_false", {"OLLAMA_IGPU_ENABLE": "false"}),
    ("igpu_one", {"OLLAMA_IGPU_ENABLE": "1"}),
    ("keep_alive_seconds", {"OLLAMA_KEEP_ALIVE": "3600"}),
    ("keep_alive_zero", {"OLLAMA_KEEP_ALIVE": "0"}),
    ("keep_alive_minus_minutes", {"OLLAMA_KEEP_ALIVE": "-1m"}),
    ("keep_alive_compound", {"OLLAMA_KEEP_ALIVE": "1h30m"}),
    ("keep_alive_junk", {"OLLAMA_KEEP_ALIVE": "soon"}),
    ("load_timeout_seconds", {"OLLAMA_LOAD_TIMEOUT": "600"}),
    ("kv_type_junk", {"OLLAMA_KV_CACHE_TYPE": "bogus"}),
    ("parallel_zero", {"OLLAMA_NUM_PARALLEL": "0"}),
]
