"""Goldens for the Rust profile YAML: Python's own ProfileService (PyYAML 6.0.3, pydantic, the real vLLM and
SGLang adapters) run on a fresh in-memory store, written to backend/crates/ec-hub/tests/data/wsc_profile_goldens.json.

  * export: a profiles row inserted as-is (any JSON params, names that need quoting or folding), then export_yaml:
    the exact text.
  * import: import_yaml(text) on a fresh store: the created profile (name, engine, repo_id, description, params) or
    the problem (status, code, detail, extra members), or {"status": 500, "exception": <class>} for an exception that
    is not a ProblemError (FastAPI's generic 500).

The Rust test (tests/test_wsc_profiles.rs) replays both lists through ProfileService. Hermetic: no network, no files
outside the output.

    "$(node backend/parity/lib/python-release.mjs)/backend/.venv/bin/python" -I backend/parity/gen/wsc_profile_goldens.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyref import DATA  # noqa: E402

from engine_console.adapters import load_default_adapters  # noqa: E402
from engine_console.domain.errors import ProblemError  # noqa: E402
from engine_console.services.profiles import ProfileService  # noqa: E402
from engine_console.services.store import Store  # noqa: E402

OUT = DATA / "wsc_profile_goldens.json"
ADAPTERS = ("engine_console.adapters.vllm", "engine_console.adapters.sglang")

LONG = "word " * 30
EXPORTS: list[dict[str, Any]] = [
    {"name": "long", "engine": "vllm", "repo_id": "Qwen/Qwen3-32B", "description": "",
     "params": {"max_model_len": 8192, "gpu_memory_utilization": 0.9, "enable_prefix_caching": True}},
    {"name": "no repo", "engine": "sglang", "repo_id": None, "description": "a profile", "params": {}},
    {"name": "yes", "engine": "vllm", "repo_id": None, "description": "on",
     "params": {"a": None, "b": "null", "c": "~", "d": "true", "e": "1.5", "f": "0777", "g": "1:30", "h": "2024-01-01",
                "i": "", "j": " lead", "k": "trail ", "l": "x: y", "m": "#c", "n": "a #b", "o": "-", "p": "- x",
                "q": "?", "r": "'q'", "s": '"dq"', "t": "it's", "u": "<<", "v": "=", "w": "!", "x": "&a", "y": "*b",
                "z": "%x"}},
    {"name": "numbers", "engine": "vllm", "repo_id": None, "description": "",
     "params": {"i": 0, "neg": -5, "big": 18446744073709551615, "f": 1.0, "g": 1e16, "h": 1.5e-7, "k": -0.0,
                "l": 123456789.123, "m": 1e-4, "n": 0.1, "t": True, "u": False}},
    {"name": "nested", "engine": "vllm", "repo_id": "org/m", "description": "",
     "params": {"speculative_config": {"method": "ngram", "num_speculative_tokens": 4, "deep": {"x": [1, [2, 3], {}]}},
                "served_model_name": ["a", "b"], "empty_list": [], "empty_dict": {}, "list_of_dicts": [{"a": 1, "b": [
                    {"c": None}]}, {}], "list_of_lists": [[1, 2], [], [[3]]]}},
    {"name": "unicode é中\U0001f600", "engine": "vllm", "repo_id": None,
     "description": "tab\there\nnew line\r\nand \u0085   ﻿ \x07 \x00 end",
     "params": {"kéy": "vé", "multi": "line one\nline two\n", "trailing nl": "x\n", "sp nl": "a \nb",
                "nl sp": "a\n b", "only nl": "\n", "quote'n\"both": "it's \"x\""}},
    {"name": "folding", "engine": "vllm", "repo_id": None, "description": LONG,
     "params": {"plain": LONG.strip(), "single": "'" + LONG, "double": "é " + LONG, "nospace": "x" * 200,
                "long key " + "k" * 130: 1, "k" * 122: "just under the simple key limit",
                "k" * 123: "at the simple key limit", "multi\nline key": "v", "": "empty key"}},
    {"name": "secret", "engine": "vllm", "repo_id": None, "description": "",
     "params": {"api_key": "[set]", "max_model_len": 4096}},
]

YAML_OK = """name: yaml1
engine: vllm
repo_id: Qwen/Qwen3-8B
description: imported
params:
  max_model_len: 8192
  gpu_memory_utilization: 0.85
  enable_prefix_caching: yes
  kv_cache_dtype: fp8
  served_model_name: [a, b]
"""

IMPORTS: list[str] = [
    YAML_OK,
    "name: minimal\nengine: vllm\n",
    "{name: flow, engine: sglang, params: {}}",
    "name: x\nengine: vllm\npreset: long-context\n",
    "name: x\nengine: nope\n",
    "name: x\nengine: vllm\nparams: {bad_param: 1}\n",
    "name: x\nengine: vllm\nparams: {max_model_len: 2024-01-01}\n",
    "name: x\nengine: vllm\nparams: {max_model_len: 0x1000}\n",
    "name: x\nengine: vllm\nparams: {max_model_len: 0o17}\n",
    "name: x\nengine: vllm\nparams: {max_model_len: 017}\n",
    "name: x\nengine: vllm\nparams: {max_model_len: 1_024}\n",
    "name: x\nengine: vllm\nparams: {max_model_len: 1:30}\n",
    "name: x\nengine: vllm\nparams: {gpu_memory_utilization: .5}\n",
    "name: x\nengine: vllm\nparams: {gpu_memory_utilization: 1e3}\n",
    "name: x\nengine: vllm\nparams: {gpu_memory_utilization: .nan}\n",
    "name: x\nengine: vllm\nparams: {enable_prefix_caching: on, enforce_eager: No}\n",
    "name: x\nengine: vllm\nparams: {api_key: s3cr3t-value}\n",
    "name: x\nengine: vllm\nparams: {api_key: '[set]'}\n",
    "name: x\nengine: vllm\nparams: {speculative_config: {method: ngram, when: 2024-01-01}}\n",
    "name: x\nengine: vllm\nparams: {speculative_config: {method: ngram, 1: one, 2.5: two, true: t, null: n}}\n",
    "name: x\nengine: vllm\nparams: {speculative_config: &a {self: *a}}\n",
    "name: x\nengine: vllm\nparams: {speculative_config: !!binary aGVsbG8=}\n",
    "name: x\nengine: vllm\nparams: {served_model_name: !!omap [{a: 1}]}\n",
    "name: x\nengine: vllm\nparams: {served_model_name: !!set {a, b}}\n",
    "name: x\nengine: vllm\nparams: {kv_cache_dtype: 2024-01-01}\n",
    "name: !!binary bmFtZQ==\nengine: vllm\n",
    "name: 123\nengine: vllm\n",
    "name: ''\nengine: vllm\n",
    "name: " + "n" * 121 + "\nengine: vllm\n",
    "engine: vllm\n",
    "name: x\nengine: vllm\nrepo_id: not a repo\n",
    "name: x\nengine: vllm\nrepo_id: null\n",
    "name: x\nengine: vllm\ndescription: null\n",
    "name: x\nengine: vllm\nparams: null\n",
    "name: x\nengine: vllm\nparams: [1, 2]\n",
    "name: x\nengine: vllm\nparams: {1: one}\n",
    "name: x\nengine: vllm\nparams: {!!binary a2V5: 1}\n",
    "name: x\nengine: vllm\nextra: ignored\nname2: also\n",
    "base: &b {engine: vllm}\nname: merged\n<<: *b\n",
    "name: x\nengine: vllm\nname: dup\n",
    "- just\n- a list",
    "just text",
    "",
    "# only a comment\n",
    "~",
    "a: b: c",
    "[1, 2",
    "key: [1, 2\nother: 3",
    "*undefined",
    "!!python/object:os.system x",
    "name: x\nengine: vllm\nparams: !!python/name:os.system {}\n",
    "name: \x07bell\n",
    "name: x\nengine: vllm\nwhen: 2024-13-45\n",
    "name: x\nengine: vllm\nwhen: !!timestamp 2024-02-30\n",
    "name: x\nengine: vllm\nbig: !!int 99999999999999999999999999\n",
    "name: x\nengine: vllm\n---\nname: y\n",
    "%YAML 1.1\n---\nname: x\nengine: vllm\n",
    "%YAML 2.0\n---\nname: x\nengine: vllm\n",
    "name: x\n\tengine: vllm\n",
    "name: \"bad \\q escape\"\nengine: vllm\n",
    "name: 'unterminated\nengine: vllm\n",
    "name: x\nengine: vllm\nparams: {max_model_len: 4096}\nparams: {max_model_len: 8192}\n",
    "? [a, b]\n: unhashable\n",
    "name: |\n  literal\n  block\nengine: vllm\n",
    "name: >\n  folded\n  block\nengine: vllm\n",
    "﻿name: bom\nengine: vllm\n",
    "name: x\nengine: vllm\nparams: {max_model_len: -.inf}\n",
    "name: x\nengine: vllm\nparams: {max_model_len: 4096.0}\n",
    "name: x\nengine: vllm\nparams: {max_model_len: true}\n",
    "name: [" * 2000 + "]" * 2000 + "\n",
]


def outcome(fn: Any) -> dict[str, Any]:
    try:
        p = fn()
    except ProblemError as e:
        out: dict[str, Any] = {"status": e.status, "code": e.code, "detail": e.detail}
        if e.extra:
            out["extra"] = e.extra
        return out
    except Exception as e:  # noqa: BLE001 - recorded as FastAPI's generic 500
        return {"status": 500, "exception": type(e).__name__}
    return {"status": 201, "profile": {"name": p.name, "engine": p.engine, "repo_id": p.repo_id,
                                       "description": p.description, "params": p.params}}


def main() -> int:
    sys.setrecursionlimit(1000)
    exports = []
    for i, prof in enumerate(EXPORTS):
        svc = ProfileService(Store(":memory:"), load_default_adapters(ADAPTERS))
        pid = f"prof_{i:012x}"
        svc._db.execute("INSERT INTO profiles VALUES(?,?,?,?,?,?,?,?)",  # noqa: SLF001 - a row as stored
                        (pid, prof["name"], prof["engine"], prof["repo_id"], json.dumps(prof["params"]),
                         prof["description"], 1.0, 1.0))
        exports.append({"row": prof, "yaml": svc.export_yaml(pid)})
    imports = []
    for text in IMPORTS:
        svc = ProfileService(Store(":memory:"), load_default_adapters(ADAPTERS))
        imports.append({"yaml": text, "result": outcome(lambda svc=svc, text=text: svc.import_yaml(text))})
    # a duplicate import: the second one is the conflict
    svc = ProfileService(Store(":memory:"), load_default_adapters(ADAPTERS))
    first = outcome(lambda: svc.import_yaml(YAML_OK))
    dup = {"yaml": YAML_OK, "first": first, "result": outcome(lambda: svc.import_yaml(YAML_OK))}
    # every export reads back as the same profile (renamed: a fresh store, so no conflict anyway)
    roundtrips = []
    for ex in exports:
        svc = ProfileService(Store(":memory:"), load_default_adapters(ADAPTERS))
        roundtrips.append({"yaml": ex["yaml"], "result": outcome(lambda svc=svc, y=ex["yaml"]: svc.import_yaml(y))})
    OUT.write_text(json.dumps({"exports": exports, "imports": imports, "duplicate": dup, "roundtrips": roundtrips},
                              indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(DATA.parents[4])}: {len(exports)} exports, {len(imports)} imports, {len(roundtrips)} roundtrips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
