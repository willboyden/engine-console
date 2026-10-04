"""Learn-mode content gate (docs/ARCHITECTURE.md section 10.7, engine_console/learn/README.md).

Every information button in the UI shows text from `engine_console/learn/*.json`. This module fails the build when the
UI would show nothing, something unreadable, or something that must not be public. It is offline: it reads JSON,
imports the adapter registry and walks the source tree with `ast`; it never touches docker, the network or a GPU.

Three groups of checks:

* coverage - a parameter, preset, metric, state, link, check, alert, page or field exists in code or in the contract
             but has no explanation, or a required glossary term is missing;
* quality  - the length limits of learn/README.md, empty or multi-sentence summaries, dangling glossary links,
             unlinked jargon, vague or promotional wording, unknown fields, duplicate keys, unsorted files;
* safety   - the repository is public: no personal paths, e-mail addresses, public IP addresses, tokens, or the
             name of the account that runs the tests.

Failure messages list every offender at once, so one run is enough to fix a file. `test_param_coverage` and
`test_preset_coverage` fail until every registered adapter has its `learn/params/<engine>.json` and
`learn/presets/<engine>.json`; the message names the missing keys. The second half of the module tests the checks
themselves on small synthetic inputs, so the gate cannot go quiet without a test noticing.
"""
from __future__ import annotations

import ast
import contextlib
import getpass
import ipaddress
import json
import re
import socket
import textwrap
import warnings
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, get_args

import pytest

import engine_console
from engine_console.adapters import DEFAULT_MODULES, load_default_adapters
from engine_console.adapters.base import Explain
from engine_console.domain.fleet import LibraryFormat, LinkState, Management
from engine_console.domain.learn import LearnBundle, Term
from engine_console.domain.models import (
    Confidence,
    Download,
    DownloadState,
    HostRamFit,
    InstanceState,
    Verdict,
)
from engine_console.learn import LEARN_DIR, load_bundle

SRC_ROOT = Path(engine_console.__file__).resolve().parent      # backend/src/engine_console
REPO_ROOT = SRC_ROOT.parents[2]                                 # backend/src/engine_console -> repository root

# ---- what must exist --------------------------------------------------------------------------------------------
# The glossary keys engine specialists and the UI refer to by name. Deleting or renaming one breaks them.
REQUIRED_TERMS: tuple[str, ...] = (
    "model", "weights", "parameters", "inference_engine", "serving", "gpu", "vram", "unified_memory", "host_ram",
    "shared_memory", "page_cache", "swap", "cgroup", "container", "docker_image", "image_tag", "gateway",
    "internal_network", "port", "endpoint", "openai_compatible_api", "token", "tokenizer", "context_window",
    "prompt", "prefill", "decode", "ttft", "itl", "throughput", "latency", "batch", "continuous_batching",
    "concurrency", "kv_cache", "paged_attention", "prefix_caching", "chunked_prefill", "quantization", "bf16",
    "fp16", "fp8", "nvfp4", "mxfp4", "int8", "int4", "awq", "gptq", "gguf", "gguf_quant_types", "safetensors",
    "hugging_face_hub", "hf_cache", "repo_id", "revision", "gated_model", "hf_token", "moe", "active_parameters",
    "dense_model", "tensor_parallelism", "pipeline_parallelism", "expert_parallelism", "data_parallelism", "nccl",
    "nvlink", "pcie", "peer_to_peer", "cuda", "cuda_graphs", "torch_compile", "compute_capability", "blackwell",
    "attention_backend", "flash_attention", "speculative_decoding", "draft_model", "mtp", "rope_scaling", "yarn",
    "sliding_window_attention", "gqa", "mla", "lora", "tool_calling", "reasoning_model", "chat_template",
    "trust_remote_code", "served_model_name", "cpu_offload", "kv_cache_offload", "gpu_layers_offload",
    "tensor_split", "mmap", "mlock", "cpu_threads", "numa", "fit_estimate", "preflight", "prometheus_metrics",
    "sampling", "temperature", "max_tokens", "embedding_model", "vision_language_model", "multimodal", "benchmark",
    "idle_ttl", "health_check", "warmup", "oom", "node_agent", "host", "nfs", "zfs", "model_root", "monitor_mode",
    "manage_mode", "restricted_ssh_key", "egress_proxy", "elo_rating", "drive", "mount", "docker_rootless",
    "management_class", "lifecycle_script",
)

# ARCHITECTURE section 5 (engine metrics) and the host / instance keys of sections 8 and 10.9.
CANONICAL_METRICS: tuple[str, ...] = (
    "requests_running", "requests_waiting", "kv_cache_usage_pct", "prefix_cache_hit_pct", "prompt_tokens_total",
    "generation_tokens_total", "ttft_p50_s", "ttft_p95_s", "itl_p50_s", "e2e_p50_s", "e2e_p95_s", "prompt_tps",
    "generation_tps", "preemptions_total", "spec_decode_accept_pct",
)
HOST_METRICS: tuple[str, ...] = (
    "host_ram_gib", "host_ram_anon_gib", "host_ram_cache_gib", "host_ram_shmem_gib", "ram_used_gib",
    "ram_available_gib", "shmem_gib", "swap_used_gib", "vram_used_gib", "gpu_util_pct", "gpu_temp_c", "gpu_power_w",
    "gpu_power_limit_w", "gpu_fan_pct", "cpu_util_pct", "load1", "disk_free_bytes",
    "disk_free_gib",       # the frontend asks for this spelling (frontend/js/learn-keys.js); same meaning, other unit
)

# Concept families (ARCHITECTURE 10.7). The sets below are the keys the UI can ask about; where the code owns the
# list (a Literal type, an emitted code) it is read from the code, not copied.
FAMILIES: tuple[str, ...] = ("state", "link", "management", "verdict", "confidence", "check", "format", "root",
                             "alert", "download", "mode", "policy", "page", "field")
CLOSED_FAMILIES = frozenset({"state", "link", "management", "verdict", "confidence", "format", "root", "download",
                             "mode", "check", "alert"})   # exactly the values of a type or the codes the code emits:
#                                                            an extra key explains something that no longer exists
# Preflight codes named by the contract (ARCHITECTURE 10.4 / 10.11). Most are also found by the scan below; listing
# them keeps them required even if a refactor hides the literal.
V2_CHECK_CODES: tuple[str, ...] = ("host_down", "monitor_only", "image_not_allowed", "image_missing", "model_missing",
                                   "format_unsupported", "port_in_use")
# Raised by the lifecycle service's launch build as ProblemError(code=...) and shown by preflight as
# `PreflightCheck(code=e.code, ...)`: no string literal sits in a PreflightCheck call, so the scan cannot see them.
# test_dynamic_check_codes_are_still_raised fails when one is no longer raised, so this list cannot rot.
DYNAMIC_CHECK_CODES: tuple[str, ...] = ("secret_in_argv", "invalid_launch")
V2_ALERT_CODES: tuple[str, ...] = ("gpu_hot", "vram_full")    # raised from live GPU stats, not by services/hostmem.py
ROOT_KINDS: tuple[str, ...] = ("auto", "hf-home", "hf-hub", "plain", "gguf", "ollama", "other")   # FLEET-PROTOCOL 6.1
MODES: tuple[str, ...] = ("monitor", "manage")
POLICY_KEYS: tuple[str, ...] = ("engine_images", "allow_pull", "allow_delete", "external_control", "scripts")
PAGES: tuple[str, ...] = ("dashboard", "hosts", "host", "library", "discover", "downloads", "launch", "wizard",
                          "instances", "instance", "metrics", "bench", "chat", "arena", "usage", "settings", "learn")
FIELDS: tuple[str, ...] = ("host", "gpus", "root", "image", "engine", "model", "preset", "concurrency", "name", "ttl",
                           "pin", "profile", "revision", "allow_patterns", "quant", "target_host", "beginner_mode")

# ---- limits (mirrors learn/README.md; test_limits_match_readme keeps the two in step) --------------------------------
EXPLAIN_LIMITS = {"summary": 160, "details": 900, "when_to_change": 400, "risks": 400, "example": 200}
TERM_LIMITS = {"short": 160, "long": 1500}
SINGLE_LINE_FIELDS = frozenset({"title", "short", "summary", "example"})
ACTION_FAMILIES = frozenset({"state", "link", "check", "alert", "verdict", "download"})   # must say what to do

# ---- key shapes -----------------------------------------------------------------------------------------------------
_SNAKE = r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*"
_NAME = r"[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*"
KEY_SHAPES = {
    "terms": re.compile(_SNAKE), "metrics": re.compile(_SNAKE), "params": re.compile(_SNAKE),
    "presets": re.compile(_NAME), "concepts": re.compile(rf"[a-z]+\.{_NAME}"),
}
SECTION_OF_FILE = {"glossary.json": "terms", "metrics.json": "metrics", "concepts.json": "concepts"}
STRICT_FILES = ("glossary.json", "metrics.json", "concepts.json")   # jargon links are enforced here, advised elsewhere

# ---- wording rules --------------------------------------------------------------------------------------------------
PLACEHOLDER = re.compile(r"\b(?:TODO|TBD|FIXME|XXX)\b|lorem ipsum|\?\?\?", re.I)
VAGUE = re.compile(r"\b(?:may|might|could|can) (?:cause|lead to|result in) (?:some |various |other )?"
                   r"(?:issues|problems)\b", re.I)   # README rule 4: name the failure instead
MARKETING = re.compile(r"\b(?:blazing(?:ly)?|revolutionary|seamless(?:ly)?|cutting[- ]edge|state[- ]of[- ]the[- ]art|"
                       r"best[- ]in[- ]class|world[- ]class|game[- ]chang\w+|supercharg\w+|turbocharg\w+|"
                       r"effortless(?:ly)?|unleash\w*)\b", re.I)
_PICTOGRAPH_RANGES = ((0x1F000, 0x1FAFF), (0x2600, 0x26FF), (0x2700, 0x27BF))       # emoji, symbols, dingbats
PICTOGRAPH = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _PICTOGRAPH_RANGES) + "]")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`'\"(])")
HYGIENE_RULES = (("placeholder text", PLACEHOLDER), ("vague 'may cause issues' wording: name the failure", VAGUE),
                 ("promotional wording", MARKETING), ("emoji or pictograph", PICTOGRAPH))

# Jargon a beginner cannot be expected to know: if the text uses it, the glossary key must be in `terms` / `see_also`
# (README rule 7). Everyday words (GPU, model, token, container) are left out on purpose.
JARGON: tuple[tuple[str, str], ...] = (
    (r"\bKV[ -]?cache", "kv_cache"), (r"\bVRAM\b", "vram"), (r"\bunified memory\b", "unified_memory"),
    (r"\bNVLink\b", "nvlink"), (r"\bPCIe\b", "pcie"), (r"\bpeer-to-peer\b|\bP2P\b", "peer_to_peer"),
    (r"\bNCCL\b", "nccl"), (r"\btensor[ -]parallel", "tensor_parallelism"),
    (r"\bpipeline[ -]parallel", "pipeline_parallelism"), (r"\bexpert[ -]parallel", "expert_parallelism"),
    (r"\bdata[ -]parallel", "data_parallelism"), (r"\bquantiz|\bquantis", "quantization"),
    (r"\bFP8\b", "fp8"), (r"\bNVFP4\b", "nvfp4"), (r"\bMXFP4\b", "mxfp4"), (r"\bBF16\b|\bbfloat16\b", "bf16"),
    (r"\bFP16\b", "fp16"), (r"\bINT8\b", "int8"), (r"\bINT4\b", "int4"), (r"\bAWQ\b", "awq"), (r"\bGPTQ\b", "gptq"),
    (r"\bGGUF\b", "gguf"), (r"\bsafetensors\b", "safetensors"), (r"\bMoE\b|\bmixture[ -]of[ -]experts\b", "moe"),
    (r"\bactive parameters\b", "active_parameters"), (r"(?<!chunked )\bprefill\b", "prefill"),
    (r"\bchunked prefill\b", "chunked_prefill"), (r"\bdecode\b", "decode"),
    (r"\bTTFT\b|\btime to first token\b", "ttft"), (r"\bITL\b|\binter-token latency\b", "itl"),
    (r"\bp(?:50|95|99)\b|\bpercentiles?\b", "percentile"), (r"\bcgroups?\b", "cgroup"),
    (r"\bpage cache\b", "page_cache"), (r"\bswap\b", "swap"), (r"\bshared memory\b", "shared_memory"),
    (r"\bCUDA graphs?\b", "cuda_graphs"), (r"\btorch\.compile\b", "torch_compile"), (r"\bYaRN\b", "yarn"),
    (r"\bRoPE\b", "rope_scaling"), (r"\bLoRA\b", "lora"), (r"\bprefix[ -]cach", "prefix_caching"),
    (r"\bpaged attention\b", "paged_attention"), (r"\bcontinuous batching\b", "continuous_batching"),
    (r"\bspeculative decoding\b", "speculative_decoding"), (r"\bdraft model\b", "draft_model"),
    (r"\bMTP\b|\bmulti-token prediction\b", "mtp"), (r"\bNUMA\b", "numa"), (r"\bmmap\b", "mmap"),
    (r"\bmlock\b", "mlock"), (r"\bcompute capability\b|\bsm_\d+", "compute_capability"),
    (r"\bBlackwell\b", "blackwell"), (r"\bGQA\b|\bgrouped-query\b", "gqa"),
    (r"\bMLA\b|\blatent attention\b", "mla"), (r"\bsliding[ -]window\b", "sliding_window_attention"),
    (r"\bFlash-?Attention\b", "flash_attention"), (r"\battention backend\b", "attention_backend"),
    (r"\btool[ -]call", "tool_calling"), (r"\breasoning model", "reasoning_model"),
    (r"\bchat template\b", "chat_template"), (r"\btrust_remote_code\b", "trust_remote_code"),
    (r"\bserved model name\b", "served_model_name"), (r"\bCPU offload", "cpu_offload"),
    (r"\bOOM\b|\bout of memory\b", "oom"), (r"\bnode agent\b|\becnode\b", "node_agent"), (r"\bNFS\b", "nfs"),
    (r"\bZFS\b", "zfs"), (r"\begress proxy\b", "egress_proxy"), (r"\bElo\b", "elo_rating"),
    (r"\bmonitor mode\b", "monitor_mode"), (r"\bmanage mode\b", "manage_mode"),
    (r"\brootless\b", "docker_rootless"),
)
JARGON_RE = tuple((re.compile(p, re.I), key) for p, key in JARGON)

# ---- publish safety -------------------------------------------------------------------------------------------------
# Written in halves so a plain-text search of the repository for home paths finds real leaks only, not this lint.
HOME_PREFIXES = ("/ho" "me/", "/Us" "ers/")
HOME_PATH = re.compile("|".join(re.escape(p) for p in HOME_PREFIXES) + r"|[A-Za-z]:\\Users\\", re.I)
EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w)")
DOCUMENTATION_NETS = tuple(ipaddress.ip_network(n) for n in
                           ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "10.0.0.0/8"))
SECRET_SHAPES = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "API key": re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "ssh public key": re.compile(r"\bssh-(?:ed25519|rsa|dss)\s+AAAA[0-9A-Za-z+/=]{20,}"),
}
_COMMON_NAMES = frozenset({"localhost", "ubuntu", "debian", "runner", "github", "docker", "builder", "worker",
                           "vagrant", "jenkins", "gitlab", "codespace", "vscode", "devcontainer", "default",
                           "administrator", "ec2-user"})


# ---- helpers --------------------------------------------------------------------------------------------------------
def _report(title: str, problems: list[str], limit: int = 250) -> str:
    shown = problems[:limit]
    more = f"\n  ... and {len(problems) - limit} more" if len(problems) > limit else ""
    return f"{title} ({len(problems)}):\n  - " + "\n  - ".join(shown) + more


def _expect_none(title: str, problems: list[str]) -> None:
    """Fail with just the list of offenders (no traceback noise) when there are any."""
    if problems:
        pytest.fail(_report(title, problems), pytrace=False)


def _wrap(keys: list[str]) -> str:
    return textwrap.fill(", ".join(keys), width=110, initial_indent="    ", subsequent_indent="    ")


def _strict_json(text: str) -> tuple[Any, list[str]]:
    """json.loads that also reports duplicate object keys (the stock parser silently keeps the last one)."""
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        return dict(pairs)

    return json.loads(text, object_pairs_hook=hook), duplicates


def _learn_files(root: Path = LEARN_DIR) -> list[tuple[Path, str]]:
    """Every JSON file under the learn directory with the section it must contain ('' = not an allowed file)."""
    out: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*.json")):
        rel = path.relative_to(root).as_posix()
        section = SECTION_OF_FILE.get(rel) or ("params" if rel.startswith("params/") else
                                               "presets" if rel.startswith("presets/") else "")
        out.append((path, section))
    return out


def _explain_items(b: LearnBundle) -> list[tuple[str, str, Explain]]:
    """(file label, key, Explain) for every explanation in the bundle except glossary terms."""
    items = [("metrics.json", k, v) for k, v in b.metrics.items()]
    items += [("concepts.json", k, v) for k, v in b.concepts.items()]
    for engine, params in b.params.items():
        items += [(f"params/{engine}.json", k, v) for k, v in params.items()]
    for engine, presets in b.presets.items():
        items += [(f"presets/{engine}.json", k, v) for k, v in presets.items()]
    return items


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def _str(node: ast.expr | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def emitted_codes(constructors: frozenset[str], helpers: frozenset[str] = frozenset(),
                  root: Path = SRC_ROOT) -> dict[str, list[str]]:
    """String codes the source emits, found with `ast` (so multi-line calls and odd spacing do not matter):
    `Constructor(..., code="x", ...)` for each name in `constructors`, and `helper("ok|warn|block", "x", ...)` for
    each name in `helpers` (the vLLM adapter builds its Compat objects through a local `add()`).
    Returns code -> ["path/in/package.py:line", ...]."""
    found: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=rel)):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            code: str | None = None
            if name in constructors:
                code = next((_str(kw.value) for kw in node.keywords if kw.arg == "code"), None)
            elif name in helpers and len(node.args) >= 2 and _str(node.args[0]) in ("ok", "warn", "block"):
                code = _str(node.args[1])
            if code:
                found.setdefault(code, []).append(f"{rel}:{node.lineno}")
    return found


def check_codes() -> dict[str, list[str]]:
    """Every preflight / compatibility code the backend emits today, with where."""
    return emitted_codes(frozenset({"PreflightCheck", "Compat"}), frozenset({"add"}))


def raised_codes() -> dict[str, list[str]]:
    """Codes of ProblemError and its subclasses raised with a literal `code=`."""
    return emitted_codes(frozenset({"ProblemError", "Unprocessable", "Conflict", "BadRequest", "NotFound", "Forbidden",
                                    "UpstreamError"}))


def alert_codes() -> dict[str, list[str]]:
    return emitted_codes(frozenset({"HostAlert"}))


def _origin(values: Iterable[object], origin: str) -> dict[str, str]:
    return {str(v): origin for v in values}


@lru_cache(maxsize=1)
def required_concepts() -> dict[str, dict[str, str]]:
    """family -> {value: where that value comes from}. Closed Literal types are read from the code itself."""
    kinds = get_args(Download.model_fields["kind"].annotation)
    emitted = {code: ", ".join(where[:2]) for code, where in check_codes().items()}
    checks = (dict.fromkeys(V2_CHECK_CODES, "contract v2 (ARCHITECTURE 10.4, 10.11)")
              | dict.fromkeys(DYNAMIC_CHECK_CODES, "raised dynamically by the launch build (services/lifecycle.py)")
              | emitted)                                                              # a found literal wins
    alerts = dict.fromkeys(V2_ALERT_CODES, "dashboard") | {c: ", ".join(w[:2]) for c, w in alert_codes().items()}
    return {
        "state": _origin(get_args(InstanceState), "domain/models.py InstanceState"),
        "link": _origin(get_args(LinkState), "domain/fleet.py LinkState"),
        "management": _origin(get_args(Management), "domain/refs.py Management"),
        "verdict": _origin(get_args(Verdict) + get_args(HostRamFit.model_fields["verdict"].annotation),
                           "domain/models.py Verdict / HostRamFit"),
        "confidence": _origin(get_args(Confidence), "domain/models.py Confidence"),
        "check": checks,
        "format": _origin(get_args(LibraryFormat), "domain/fleet.py LibraryFormat"),
        "root": _origin(ROOT_KINDS, "docs/FLEET-PROTOCOL.md 6.1"),
        "alert": alerts,
        "download": _origin(get_args(DownloadState), "domain/models.py DownloadState")
                    | _origin([f"kind_{k}" for k in kinds], "domain/models.py Download.kind"),
        "mode": _origin(MODES, "docs/FLEET-PROTOCOL.md 6.2"),
        "policy": _origin(POLICY_KEYS, "docs/FLEET-PROTOCOL.md 6"),
        "page": _origin(PAGES, "ARCHITECTURE 10.7"),
        "field": _origin(FIELDS, "ARCHITECTURE 10.7"),
    }


def ui_requested_keys() -> tuple[dict[str, list[str]], list[str]] | None:
    """The concept and metric keys the frontend says it asks for, read from frontend/js/learn-keys.js (None outside
    the repository). The UI looks up `<family>.<id>` and a list of metric keys; a missing one shows no help icon."""
    path = REPO_ROOT / "frontend" / "js" / "learn-keys.js"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    block = re.search(r"CONCEPT_FAMILIES\s*=\s*\{(.*?)\n\};", text, re.S)
    metrics = re.search(r"METRIC_KEYS\s*=\s*\[(.*?)\];", text, re.S)
    if not block or not metrics:
        return {}, []
    families = {name: re.findall(r"'([^']+)'", body)
                for name, body in re.findall(r"(\w+):\s*\[(.*?)\]", block.group(1), re.S)}
    return families, re.findall(r"'([a-z0-9_]+)'", metrics.group(1))


# ---- the checks (pure functions: the real tests call them on the real bundle, the self-tests on small fakes) ----------
def check_file_text(rel: str, section: str, text: str) -> list[str]:
    """Shape of one learn file: JSON, no duplicate keys, {"version": 1, "<section>": {...}}, sorted, known fields."""
    if not section:
        return [f"{rel}: not an allowed learn file (glossary.json, metrics.json, concepts.json, "
                "params/<engine>.json, presets/<engine>.json)"]
    try:
        doc, duplicates = _strict_json(text)
    except json.JSONDecodeError as e:
        return [f"{rel}: invalid JSON ({e})"]
    problems = [f"{rel}: duplicate key '{k}' (the second silently replaces the first)" for k in duplicates]
    if not isinstance(doc, dict) or set(doc) != {"version", section}:
        return [*problems, f'{rel}: top level must be exactly {{"version": 1, "{section}": {{...}}}}']
    if type(doc["version"]) is not int or doc["version"] != 1:
        problems.append(f"{rel}: version must be the number 1")
    body = doc[section]
    if not isinstance(body, dict):
        return [*problems, f"{rel}: '{section}' must be an object"]
    if list(body) != sorted(body):
        first = next(k for k, s in zip(body, sorted(body), strict=True) if k != s)
        problems.append(f"{rel}: keys are not sorted (first out of place: '{first}')")
    allowed = set(Term.model_fields) if section == "terms" else set(Explain.model_fields)
    for key, entry in body.items():
        if not KEY_SHAPES[section].fullmatch(key):
            problems.append(f"{rel}: malformed key '{key}'")
        if section == "concepts" and "." in key and key.split(".", 1)[0] not in FAMILIES:
            problems.append(f"{rel}: '{key}' has an unknown concept family (known: {', '.join(FAMILIES)})")
        if not isinstance(entry, dict):
            problems.append(f"{rel}: '{key}' must be an object")
            continue
        for field in sorted(set(entry) - allowed):
            problems.append(f"{rel}: '{key}' has unknown field '{field}' (allowed: {', '.join(sorted(allowed))})")
    return problems


def check_limits(b: LearnBundle) -> list[str]:
    problems: list[str] = []
    for key, t in b.glossary.items():
        if not t.title.strip():
            problems.append(f"glossary.json: '{key}' has an empty title")
        if not t.short.strip():
            problems.append(f"glossary.json: '{key}' has an empty short")
        for field, limit in TERM_LIMITS.items():
            if len(getattr(t, field)) > limit:
                problems.append(f"glossary.json: '{key}'.{field} is {len(getattr(t, field))} chars (limit {limit})")
    for file, key, e in _explain_items(b):
        if not e.summary.strip():
            problems.append(f"{file}: '{key}' has an empty summary")
        for field, limit in EXPLAIN_LIMITS.items():
            if len(getattr(e, field)) > limit:
                problems.append(f"{file}: '{key}'.{field} is {len(getattr(e, field))} chars (limit {limit})")
    return problems


def check_required_text(b: LearnBundle) -> list[str]:
    """Fields a reader of our own files must always find (params and presets only need a summary: advised below)."""
    problems: list[str] = []
    for key, t in b.glossary.items():
        if not t.long.strip():
            problems.append(f"glossary.json: '{key}' has no long text")
        elif len(t.long) <= len(t.short):
            problems.append(f"glossary.json: '{key}'.long is not longer than its short")
    for file, key, e in _explain_items(b):
        need = {"metrics.json": ["summary", "details", "when_to_change", "risks", "example"],
                "concepts.json": ["summary", "details"]}.get(file, [])
        if file == "concepts.json" and key.split(".", 1)[0] in ACTION_FAMILIES:
            need.append("when_to_change")
        problems += [f"{file}: '{key}' has no {f}" for f in need if not getattr(e, f).strip()]
    return problems


def _all_text(b: LearnBundle) -> Iterator[tuple[str, str, str, str]]:
    """(file, key, field, text) for every string a reader sees."""
    for key, t in b.glossary.items():
        for field in ("title", "short", "long"):
            yield "glossary.json", key, field, getattr(t, field)
    for file, key, e in _explain_items(b):
        for field in ("summary", "details", "when_to_change", "risks", "example"):
            yield file, key, field, getattr(e, field)


def check_hygiene(b: LearnBundle) -> list[str]:
    problems: list[str] = []
    for file, key, field, text in _all_text(b):
        where = f"{file}: '{key}'.{field}"
        if not text:
            continue
        if text != text.strip():
            problems.append(f"{where}: leading or trailing whitespace")
        if "  " in text or "\t" in text or any(ord(c) < 32 and c != "\n" for c in text):
            problems.append(f"{where}: double space, tab or control character")
        if field in SINGLE_LINE_FIELDS and "\n" in text:
            problems.append(f"{where}: must be one line")
        for label, rx in HYGIENE_RULES:
            if (m := rx.search(text)):
                problems.append(f"{where}: {label} ('{m.group(0)}')")
        if field in ("summary", "short"):
            if not text.endswith("."):
                problems.append(f"{where}: must be a sentence ending with a full stop")
            if SENTENCE_BREAK.search(text):
                problems.append(f"{where}: must be ONE sentence (README: 'One plain sentence')")
    return problems


def check_links(b: LearnBundle) -> list[str]:
    """Every `terms` / `see_also` key exists in the glossary; no duplicates; no self links; unique titles."""
    problems: list[str] = []
    for key, t in b.glossary.items():
        problems += [f"glossary.json: '{key}'.see_also -> unknown term '{k}'" for k in t.see_also if k not in b.glossary]
        if key in t.see_also:
            problems.append(f"glossary.json: '{key}'.see_also links to itself")
        if len(set(t.see_also)) != len(t.see_also):
            problems.append(f"glossary.json: '{key}'.see_also has duplicates")
    for file, key, e in _explain_items(b):
        problems += [f"{file}: '{key}'.terms -> unknown term '{k}'" for k in e.terms if k not in b.glossary]
        if len(set(e.terms)) != len(e.terms):
            problems.append(f"{file}: '{key}'.terms has duplicates")
    titles: dict[str, str] = {}
    for key, t in b.glossary.items():
        other = titles.setdefault(t.title.casefold(), key)
        if other != key:
            problems.append(f"glossary.json: '{key}' and '{other}' have the same title '{t.title}'")
    return problems


def _jargon(text: str) -> set[str]:
    return {key for rx, key in JARGON_RE if rx.search(text)}


def check_jargon(b: LearnBundle, strict: tuple[str, ...] = STRICT_FILES) -> tuple[list[str], list[str]]:
    """README rule 7: a glossary word used in the text is listed in `terms` (or `see_also`). Returns
    (problems in strict files, advice for the others)."""
    hard: list[str] = []
    soft: list[str] = []
    for key, t in b.glossary.items():
        missing = _jargon(f"{t.short} {t.long}") - set(t.see_also) - {key}
        hard += [f"glossary.json: '{key}' uses '{m}' but does not list it in see_also" for m in sorted(missing)
                 if "glossary.json" in strict]
    for file, key, e in _explain_items(b):
        text = " ".join((e.summary, e.details, e.when_to_change, e.risks, e.example))
        missing = _jargon(text) - set(e.terms)
        (hard if file in strict else soft).extend(
            f"{file}: '{key}' uses '{m}' but does not list it in terms" for m in sorted(missing))
    return hard, soft


def _private_markers() -> list[str]:
    """Names that identify whoever runs the tests (account, home directory, host). Read at run time so the public
    repository never has to contain them. Short and generic names are skipped to avoid false alarms."""
    names: set[str] = {Path.home().name, socket.gethostname().split(".")[0]}
    with contextlib.suppress(Exception):
        names.add(getpass.getuser())
    return sorted({n.lower() for n in names if len(n) >= 6 and n.lower() not in _COMMON_NAMES})


def scan_publish_safety(files: dict[str, str], markers: list[str] | None = None) -> list[str]:
    """file label -> text. Reports file:line and the kind of problem, never the private value itself."""
    problems: list[str] = []
    markers = _private_markers() if markers is None else markers
    marker_rx = [re.compile(rf"(?<![A-Za-z0-9]){re.escape(m)}(?![A-Za-z0-9])", re.I) for m in markers]
    for label, text in files.items():
        for n, line in enumerate(text.splitlines(), 1):
            where = f"{label}:{n}"
            if HOME_PATH.search(line):
                problems.append(f"{where}: absolute home-directory path")
            if EMAIL.search(line):
                problems.append(f"{where}: e-mail address")
            for m in IPV4.finditer(line):
                try:
                    ip = ipaddress.ip_address(m.group(0))
                except ValueError:
                    continue     # not an address (e.g. 999.1.1.1)
                if not any(ip in net for net in DOCUMENTATION_NETS):
                    problems.append(f"{where}: IP address outside the documentation ranges "
                                    "(use 192.0.2.x, 198.51.100.x, 203.0.113.x or 10.x.x.x)")
            problems += [f"{where}: looks like a {kind}" for kind, rx in SECRET_SHAPES.items() if rx.search(line)]
            if any(rx.search(line) for rx in marker_rx):
                problems.append(f"{where}: the name of the account or machine running the tests")
    return problems


# ---- fixtures -------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bundle() -> LearnBundle:
    return load_bundle()


ADAPTERS = {a.id: a for a in load_default_adapters().list()}


# ---- loading and shape ----------------------------------------------------------------------------------------------
def test_bundle_loads(bundle: LearnBundle) -> None:
    missing = [name for name in SECTION_OF_FILE if not (LEARN_DIR / name).is_file()]
    if missing:
        pytest.fail(f"missing learn file(s): {', '.join(missing)}", pytrace=False)
    assert bundle.version == 1
    assert bundle.glossary and bundle.metrics and bundle.concepts


def test_files_are_well_formed() -> None:
    problems: list[str] = []
    for path, section in _learn_files():
        rel = path.relative_to(LEARN_DIR).as_posix()
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            problems.append(f"{rel}: starts with a byte-order mark")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            problems.append(f"{rel}: not UTF-8 ({e})")
            continue
        problems += check_file_text(rel, section, text)
    _expect_none("learn files that do not follow learn/README.md", problems)


def test_limits_match_readme() -> None:
    text = (LEARN_DIR / "README.md").read_text(encoding="utf-8")
    explain = {m[1]: int(m[2]) for m in
               re.finditer(r"^\|\s*`(summary|details|when_to_change|risks|example)`\s*\|\s*(\d+)\s*chars", text, re.M)}
    term = {m[1]: int(m[2]) for m in re.finditer(r"`(short|long)`\s*\((\d+)\s*chars\)", text)}
    assert explain == EXPLAIN_LIMITS, f"README Explain limits {explain} differ from this test's {EXPLAIN_LIMITS}"
    assert term == TERM_LIMITS, f"README Term limits {term} differ from this test's {TERM_LIMITS}"


def test_limits(bundle: LearnBundle) -> None:
    problems = check_limits(bundle)
    _expect_none("text over the limits of learn/README.md or empty", problems)


def test_required_text(bundle: LearnBundle) -> None:
    problems = check_required_text(bundle)
    _expect_none("fields a reader of glossary / metrics / concepts must find", problems)


def test_hygiene(bundle: LearnBundle) -> None:
    problems = check_hygiene(bundle)
    _expect_none("wording and formatting problems", problems)


def test_links_resolve(bundle: LearnBundle) -> None:
    problems = check_links(bundle)
    _expect_none("broken or duplicate glossary links", problems)


def test_jargon_is_linked(bundle: LearnBundle) -> None:
    hard, soft = check_jargon(bundle)
    if soft:
        warnings.warn(_report("per-engine text uses glossary words without listing them in terms (advice)", soft),
                      stacklevel=1)
    _expect_none("glossary / metrics / concepts use jargon without linking it", hard)


def test_advice_engine_text_is_not_thin(bundle: LearnBundle) -> None:
    """Advice only: a parameter or preset popover with just a summary is thin (README shows all fields filled)."""
    thin = [f"{file}: '{key}' has no details" for file, key, e in _explain_items(bundle)
            if file.startswith(("params/", "presets/")) and not e.details.strip()]
    thin += [f"{file}: '{key}' has no when_to_change" for file, key, e in _explain_items(bundle)
             if file.startswith("params/") and e.details.strip() and not e.when_to_change.strip()]
    if thin:
        warnings.warn(_report("thin per-engine explanations (advice)", thin), stacklevel=1)


# ---- coverage: glossary, metrics, concepts --------------------------------------------------------------------------
def test_required_glossary_terms(bundle: LearnBundle) -> None:
    missing = [k for k in REQUIRED_TERMS if k not in bundle.glossary]
    if missing:
        pytest.fail(f"{len(missing)} required glossary term(s) missing (engine specialists link to them):\n"
                    + _wrap(missing), pytrace=False)
    assert len(set(REQUIRED_TERMS)) == len(REQUIRED_TERMS), "REQUIRED_TERMS has a duplicate"


def test_metric_coverage(bundle: LearnBundle) -> None:
    missing = [k for k in (*CANONICAL_METRICS, *HOST_METRICS) if k not in bundle.metrics]
    if missing:
        pytest.fail(f"{len(missing)} metric(s) without an explanation in metrics.json:\n" + _wrap(missing),
                    pytrace=False)


def test_canonical_metrics_match_architecture_doc() -> None:
    doc = REPO_ROOT / "docs" / "ARCHITECTURE.md"
    if not doc.is_file():
        pytest.skip("docs/ARCHITECTURE.md not found (running outside the repository)")
    text = doc.read_text(encoding="utf-8")
    m = re.search(r"^## 5\. Canonical metric keys[^\n]*\n(.*?)^## ", text, re.S | re.M)
    assert m, "ARCHITECTURE.md has no '## 5. Canonical metric keys' section any more: update this test"
    listed = set(re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", m.group(1)))
    assert listed == set(CANONICAL_METRICS), (
        f"ARCHITECTURE.md section 5 and this test disagree. Only in the doc: {sorted(listed - set(CANONICAL_METRICS))}; "
        f"only in the test: {sorted(set(CANONICAL_METRICS) - listed)}. Add the metric to metrics.json and to this test.")


def test_ui_requested_keys_are_explained(bundle: LearnBundle) -> None:
    """Contract with the frontend: every key listed in frontend/js/learn-keys.js has an explanation."""
    parsed = ui_requested_keys()
    if parsed is None:
        pytest.skip("frontend/js/learn-keys.js not found (running outside the repository)")
    families, metrics = parsed
    if sum(len(ids) for ids in families.values()) < 40 or len(metrics) < 15:
        pytest.fail("could not read the key lists from frontend/js/learn-keys.js any more: update ui_requested_keys()",
                    pytrace=False)
    missing = [f"concept {fam}.{i}" for fam, ids in families.items() for i in ids
               if f"{fam}.{i}" not in bundle.concepts]
    missing += [f"metric {k}" for k in metrics if k not in bundle.metrics and k not in bundle.concepts]
    _expect_none("keys the UI asks for (frontend/js/learn-keys.js) that have no explanation", missing)


@pytest.mark.parametrize("family", FAMILIES)
def test_concept_family_coverage(bundle: LearnBundle, family: str) -> None:
    required = required_concepts()[family]
    missing = {v: w for v, w in required.items() if f"{family}.{v}" not in bundle.concepts}
    lines = [f"{family}.{v}   (from {w})" for v, w in sorted(missing.items())]
    _expect_none(f"concepts.json has no explanation for {len(missing)} '{family}.*' key(s)", lines)


def test_no_stale_concepts(bundle: LearnBundle) -> None:
    """Closed families list exactly what exists now: a key for a removed value or code is a stale explanation."""
    required = required_concepts()
    stale = []
    for key in bundle.concepts:
        family, _, value = key.partition(".")
        if family in CLOSED_FAMILIES and value not in required[family]:
            stale.append(key)
    _expect_none("concepts.json explains values that no longer exist (remove or rename)", stale)


def test_root_kinds_match_protocol_doc() -> None:
    doc = REPO_ROOT / "docs" / "FLEET-PROTOCOL.md"
    if not doc.is_file():
        pytest.skip("docs/FLEET-PROTOCOL.md not found (running outside the repository)")
    m = re.search(r"^### 6\.1 Root kinds\n(.*?)^### ", doc.read_text(encoding="utf-8"), re.S | re.M)
    assert m, "FLEET-PROTOCOL.md has no '### 6.1 Root kinds' section any more: update this test"
    kinds = set(re.findall(r"^\|\s*`([a-z-]+)`\s*\|", m.group(1), re.M))
    assert kinds == set(ROOT_KINDS), f"root kinds in the protocol doc {sorted(kinds)} differ from ROOT_KINDS"


def test_code_scan_finds_the_known_emitters() -> None:
    """Guards the coverage tests above: if a refactor hides the codes from the `ast` scan, fail loudly instead of
    letting 'every emitted code is explained' pass on an empty list."""
    codes = check_codes()
    anchors = ("invalid_params", "fit", "gpu_shared",       # PreflightCheck(code=...) in the lifecycle service
               "nvfp4_moe_sm120",                            # vLLM adapter: add("warn", "<code>", ...)
               "gpus_insufficient")                          # SGLang adapter: Compat(level=..., code=...)
    lost = [c for c in anchors if c not in codes]
    _expect_none("the code scan no longer finds these emitters (update emitted_codes?)", lost)
    assert {"host_ram_low", "shmem_high", "swap_heavy"} <= set(alert_codes()), "alert scan lost services/hostmem.py"


def test_dynamic_check_codes_are_still_raised() -> None:
    """DYNAMIC_CHECK_CODES must stay true: each is still raised somewhere with a literal `code=`."""
    raised = raised_codes()
    gone = [c for c in DYNAMIC_CHECK_CODES if c not in raised]
    _expect_none("DYNAMIC_CHECK_CODES names codes that no ProblemError raises any more (remove them here and the "
                 "check.* concept)", gone)


# ---- coverage: engine parameters and presets ------------------------------------------------------------------------
def test_adapters_registered() -> None:
    """The registry skips a module that fails to import, which would silently drop that engine from coverage."""
    registered = {type(a).__module__ for a in ADAPTERS.values()}
    absent = [m for m in DEFAULT_MODULES if m not in registered]
    if absent:
        pytest.fail(f"adapter module(s) registered no adapter (import error? see the log): {absent}", pytrace=False)
    assert {"vllm", "sglang"} <= set(ADAPTERS), f"registered adapters: {sorted(ADAPTERS)}"


@pytest.mark.parametrize("engine", sorted(ADAPTERS))
def test_param_coverage(bundle: LearnBundle, engine: str) -> None:
    """Every ParamSpec.key of the adapter has an Explain in learn/params/<engine>.json (and no leftovers)."""
    wanted = [p.key for p in ADAPTERS[engine].param_catalog()]
    have = bundle.params.get(engine, {})
    missing = [k for k in wanted if k not in have]
    stale = sorted(set(have) - set(wanted))
    if not (LEARN_DIR / "params" / f"{engine}.json").is_file():
        head = f"learn/params/{engine}.json does not exist; {len(missing)} of {len(wanted)} parameters unexplained."
    else:
        head = f"learn/params/{engine}.json is missing {len(missing)} of {len(wanted)} parameter explanation(s)."
    msg = head + (f"\n  missing keys (ParamSpec.key):\n{_wrap(missing)}" if missing else "")
    msg += (f"\n  stale keys (not in the {engine} catalog any more):\n{_wrap(stale)}" if stale else "")
    if missing or stale:
        pytest.fail(msg, pytrace=False)


@pytest.mark.parametrize("engine", sorted(ADAPTERS))
def test_preset_coverage(bundle: LearnBundle, engine: str) -> None:
    wanted = sorted(ADAPTERS[engine].presets())
    have = bundle.presets.get(engine, {})
    missing = [k for k in wanted if k not in have]
    stale = sorted(set(have) - set(wanted))
    msg = f"learn/presets/{engine}.json is missing {len(missing)} of {len(wanted)} preset explanation(s): " \
          f"{', '.join(missing)}"
    msg += f"\n  stale preset names: {', '.join(stale)}" if stale else ""
    if missing or stale:
        pytest.fail(msg, pytrace=False)


# ---- publish safety -------------------------------------------------------------------------------------------------
def test_publish_safety() -> None:
    """Public repository: scan every learn file, README included, as raw text."""
    files = {p.relative_to(LEARN_DIR).as_posix(): p.read_text(encoding="utf-8")
             for p in sorted(LEARN_DIR.rglob("*")) if p.suffix in (".json", ".md")}
    problems = scan_publish_safety(files)
    _expect_none("content that must not be published", problems)


# ---- the gate tests itself ------------------------------------------------------------------------------------------
def _bundle(**kw: Any) -> LearnBundle:
    """A tiny valid bundle; keyword arguments replace its sections."""
    term = Term(title="KV cache", short="The notes an engine keeps.", long="The notes an engine keeps while it works.")
    base: dict[str, Any] = {
        "glossary": {"kv_cache": term},
        "metrics": {"load1": Explain(summary="The load.", details="d", when_to_change="w", risks="r", example="e")},
        "concepts": {"state.ready": Explain(summary="Ready.", details="d", when_to_change="w")},
    }
    return LearnBundle(**(base | kw))


def test_gate_flags_limit_and_text_problems() -> None:
    bad = Explain(summary="x" * 161, details="y" * 901, when_to_change=" padded ", risks="  ", example="z" * 201)
    problems = check_limits(_bundle(metrics={"load1": bad}))
    assert {"summary", "details", "example"} <= {p.split("'.")[1].split()[0] for p in problems if "chars" in p}
    two = Explain(summary="One thing. Another thing.", details="Do not wait; it may cause issues. TODO")
    hygiene = " | ".join(check_hygiene(_bundle(metrics={"load1": two})))
    assert "ONE sentence" in hygiene and "vague" in hygiene and "placeholder" in hygiene
    assert "full stop" in " | ".join(check_hygiene(_bundle(metrics={"load1": Explain(summary="No stop")})))
    assert "promotional" in " | ".join(check_hygiene(_bundle(metrics={"load1": Explain(summary="A seamless fix.")})))


def test_gate_flags_missing_required_text() -> None:
    thin = Explain(summary="Only a summary.")
    assert len(check_required_text(_bundle(metrics={"load1": thin}))) == 4          # details, when, risks, example
    assert check_required_text(_bundle(concepts={"state.ready": thin})) == [
        "concepts.json: 'state.ready' has no details", "concepts.json: 'state.ready' has no when_to_change"]
    assert check_required_text(_bundle(concepts={"page.chat": thin})) == ["concepts.json: 'page.chat' has no details"]


def test_gate_flags_dangling_links_and_unlinked_jargon() -> None:
    linked = Explain(summary="See the VRAM.", details="d", terms=["nonsense", "nonsense"])
    problems = check_links(_bundle(concepts={"state.ready": linked}))
    assert any("unknown term 'nonsense'" in p for p in problems) and any("duplicates" in p for p in problems)
    glossary = {"kv_cache": Term(title="KV cache", short="s.", long="Longer text.", see_also=["kv_cache", "gone"])}
    problems = check_links(_bundle(glossary=glossary))
    assert any("itself" in p for p in problems) and any("unknown term 'gone'" in p for p in problems)
    hard, soft = check_jargon(_bundle(concepts={"state.ready": linked}, metrics={}))
    assert hard == ["concepts.json: 'state.ready' uses 'vram' but does not list it in terms"] and not soft
    hard, soft = check_jargon(_bundle(concepts={}, metrics={}), strict=())
    assert not hard and not soft
    hard, soft = check_jargon(LearnBundle(params={"vllm": {"gpu_memory_utilization": Explain(summary="Uses VRAM.")}}))
    assert not hard and soft == ["params/vllm.json: 'gpu_memory_utilization' uses 'vram' but does not list it in terms"]


def test_gate_flags_malformed_files() -> None:
    ok = '{"version": 1, "metrics": {"a_b": {"summary": "S."}, "c": {"summary": "S."}}}'
    assert check_file_text("metrics.json", "metrics", ok) == []
    assert check_file_text("x.json", "", ok)
    assert any("invalid JSON" in p for p in check_file_text("metrics.json", "metrics", "{"))
    dup = '{"version": 1, "metrics": {"a": {"summary": "S."}, "a": {"summary": "T."}}}'
    assert any("duplicate key 'a'" in p for p in check_file_text("metrics.json", "metrics", dup))
    unsorted = '{"version": 1, "metrics": {"b": {"summary": "S."}, "a": {"summary": "S."}}}'
    assert any("not sorted" in p for p in check_file_text("metrics.json", "metrics", unsorted))
    typo = '{"version": 1, "metrics": {"a": {"summary": "S.", "when_to_chnage": "x"}}}'
    assert any("unknown field 'when_to_chnage'" in p for p in check_file_text("metrics.json", "metrics", typo))
    wrong = '{"version": 2, "terms": {}}'
    assert check_file_text("metrics.json", "metrics", wrong)
    assert any("malformed key" in p for p in check_file_text(
        "metrics.json", "metrics", '{"version": 1, "metrics": {"Bad Key": {"summary": "S."}}}'))
    assert any("unknown concept family" in p for p in check_file_text(
        "concepts.json", "concepts", '{"version": 1, "concepts": {"colour.red": {"summary": "S."}}}'))
    assert not check_file_text("presets/vllm.json", "presets",
                               '{"version": 1, "presets": {"max-throughput": {"summary": "S."}}}')


def test_gate_flags_unsafe_content() -> None:
    clean = {"a.json": 'Use 192.0.2.7 or 10.1.2.3, and ~/.cache/huggingface. Version 0.5.14. Token: [set, 37 chars].'}
    assert scan_publish_safety(clean, markers=[]) == []
    leaky = {"a.json": "\n".join([
        f'path "{HOME_PREFIXES[0]}someone/models"', "mail me at someone@example.com",
        "server at " + ".".join(["8"] * 4) + " and 127.0.0." + "1",          # a public and a loopback address
        "token hf_" + "A" * 34, "key sk-" + "b" * 30, "owner is Exampleperson here",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
    ])}
    found = scan_publish_safety(leaky, markers=["exampleperson"])
    assert len(found) == 8 and all(p.startswith("a.json:") for p in found), found
    assert not any("exampleperson" in p.lower() for p in found), "findings must not echo the private value"
    assert [p.split(":")[1] for p in found] == ["1", "2", "3", "3", "4", "5", "6", "7"]


def test_gate_scans_code_for_emitted_codes(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        "def f():\n"
        "    a = Compat(level='warn',\n"
        "               code='multi_line', message='x')\n"
        "    b = PreflightCheck(code='plain', level='ok', message='y')\n"
        "    add('block', 'via_helper', 'z')\n"
        "    add(level, code, msg)\n"
        "    other(code='ignored')\n"
        "    PreflightCheck(code=e.code, level='block', message='dynamic')\n", encoding="utf-8")
    found = emitted_codes(frozenset({"PreflightCheck", "Compat"}), frozenset({"add"}), root=tmp_path)
    assert found == {"multi_line": ["m.py:2"], "plain": ["m.py:4"], "via_helper": ["m.py:5"]}
