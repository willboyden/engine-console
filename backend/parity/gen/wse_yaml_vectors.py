"""Write the golden vectors for services/docker.rs (the "copy as command" snippets):

* backend/crates/ec-hub/tests/data/wse_yaml_vectors.json: PyYAML's `yaml.safe_dump(doc, sort_keys=False)` of documents
  shaped like a compose file, with scalars that exercise every quoting rule and line folding at 80 columns;
* backend/crates/ec-hub/tests/data/wse_snippet_vectors.json: `services.docker.command_snippets(...)` on run specs.

Deterministic (fixed seed). Run with the Python reference (gen/pyref.py):

    "$(node backend/parity/lib/python-release.mjs)/backend/.venv/bin/python" -I backend/parity/gen/wse_yaml_vectors.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml

from engine_console.services.docker import command_snippets

DATA = Path(__file__).resolve().parents[2] / "crates" / "ec-hub" / "tests" / "data"

SCALARS: list[Any] = [
    "plain", "", " ", "a b", " lead", "trail ", "yes", "No", "on", "OFF", "true", "null", "Null", "~", "123", "0o7", "0x1F",
    "1.5", "1e3", "1.0e+3", ".5", "-.inf", ".NaN", "12:30", "1:20:30.5", "2026-10-27", "2026-10-27 12:00:00", "<<", "=",
    "!", "&", "*", "- x", "-x", "--model", "---", "...", "? x", "?x", ":x", "a: b", "a:b", "a #b", "a#b", "#x", "[x]",
    "a,b", "{x}", "${VAR}", "'quoted'", '"dq"', "it's", "tab\there", "line\nbreak", "two\n\nbreaks", "trailing\n",
    "\nleading", "space \nbreak", "break\n space", "café", " nbsp", "emoji \U0001f600", "bell\x07", "nul\x00",
    "back\\slash", " sep", "x" * 130, "no-new-privileges:true", "%percent", "@at", "`tick", "|pipe", ">gt",
    "16g", "8000", "0", "-1", "+1", "1_000", "0b101", "Yes please", "TRUE", "FALSE", "y", "n",
    "/data/hf:/root/.cache/huggingface:ro", "ec-vllm-cache:/root/.cache/vllm", 4096, 0, -7, 1.5, 1e20, 2.0, True, False,
    None, 1e-07,
]


def long_texts(rng: random.Random) -> list[str]:
    words = ["alpha", "b", "gamma-delta", "x" * 23, "it's", "a:b", "#hash", "café", "tab\t", "q\"q", "  ", "z" * 70,
             "line\nnext", "--flag", "{json: 1}", "1", "end"]
    out = []
    for _ in range(60):
        n = rng.randint(2, 30)
        out.append(" ".join(rng.choice(words) for _ in range(n)))
    for width in (70, 75, 78, 79, 80, 81, 82, 85, 90, 100):
        out.append(("word " * 40)[:width].strip())
        out.append(("w " * 60)[:width])
        out.append("é" + ("ab " * 40)[:width])
        out.append("'" + ("cd " * 40)[:width])
    return out


def docs() -> list[Any]:
    rng = random.Random(20261027)
    out: list[Any] = []
    for s in SCALARS:
        out.append({"key": s})
        out.append({"services": {"svc": {"command": [s, "x"], "labels": {"k": s}}}})
        if isinstance(s, str):
            out.append({s or "empty-key-not-used": "v"})
    for t in long_texts(rng):
        out.append({"services": {"ec-model-abc123": {"command": [t], "labels": {"engine-console.name": t},
                                                     "deploy": {"r": {"devices": [{"driver": t, "ids": [t]}]}}}}})
    out.append({"x" * 122: 1, "y" * 123: 2, "z" * 130: [1, 2]})
    out.append({"empty_list": [], "empty_map": {}, "nested": [[], {}, [1, [2]], {"a": []}]})
    out.append([1, "two", {"three": 3}, [4]])
    out.append({"deep": {"a": {"b": {"c": {"d": {"e": "word " * 30}}}}}})
    return out


def specs() -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "name": "ec-qwen-abc123", "instance": "inst_abc", "image": "vllm/vllm-openai:v0.27.1",
        "argv": ["--model", "Qwen/Qwen3-32B", "--served-model-name", "Qwen/Qwen3-32B", "--port", "8000",
                 "--chat-template", "{% for m in messages %}{{ m.content }} {% endfor %}", "--max-model-len", "32768",
                 "--speculative-config", '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}'],
        "container_port": 8000, "gpu_uuids": ["GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"], "network_alias": "ec-qwen-abc123",
        "labels": {"engine-console.role": "engine", "engine-console.instance": "inst_abc",
                   "engine-console.name": "my long model name with spaces that goes past the eighty column fold",
                   "engine-console.port": "18000", "engine-console.gpu_ids": "0"},
        "env": {"HF_HOME": "/root/.cache/huggingface", "HF_HUB_OFFLINE": "1", "VLLM_LOGGING_LEVEL": "INFO"},
        "secret_env": {"VLLM_API_KEY": "[set]"},
        "mounts": [{"root": "hf", "rel": "hub", "dest": "/root/.cache/huggingface/hub", "ro": True},
                   {"root": "gone", "rel": "", "dest": "/x", "ro": True}],
        "volumes": [{"name": "ec-vllm-cache", "dest": "/root/.cache/vllm"}], "shm_size": "16g", "ipc_host": False}
    with_files = {**base, "files": {"/etc/b.yaml": "x", "/etc/a.json": "{}"}, "ipc_host": True, "shm_size": ""}
    bare = {"name": "", "argv": [], "mounts": [{"root": "hf", "dest": "/m"}]}
    return [base, with_files, bare]


def main() -> None:
    vectors = [{"doc": d, "yaml": yaml.safe_dump(d, sort_keys=False)} for d in docs()]
    (DATA / "wse_yaml_vectors.json").write_text(json.dumps(vectors, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    snippets = []
    for spec in specs():
        for rendered in (None, ["docker", "--context", "rootless", "run", "-d", "--name", spec.get("name") or "x", "img"]):
            for network in (None, "engines"):
                paths = {"hf": "/data/hf/"}
                out = command_snippets(spec, rendered, host_paths=paths, network=network)
                snippets.append({"spec": spec, "rendered": rendered, "host_paths": paths, "network": network, "out": out})
    (DATA / "wse_snippet_vectors.json").write_text(json.dumps(snippets, indent=1, ensure_ascii=True) + "\n",
                                                   encoding="utf-8")
    print(f"{len(vectors)} yaml vectors, {len(snippets)} snippet vectors")


if __name__ == "__main__":
    main()
