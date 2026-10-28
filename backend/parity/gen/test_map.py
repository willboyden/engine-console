"""Write backend/TEST-MAP.md and its machine-readable sidecar backend/parity/test-map.json: one row per pytest node id of the
last Python release's hub (backend/) and node agent (node/), with the area of the code under test and the Rust test(s)
that cover it.

  python3 -I backend/parity/gen/test_map.py              collect, merge into test-map.json, rewrite TEST-MAP.md
  python3 -I backend/parity/gen/test_map.py --autofill   also map rows whose test name matches exactly one `fn <name>`
                                                      in the crates of that suite (backend: ec-hub; node: ec-proto,
                                                      ecnode)

The sidecar is MERGED, never rebuilt: a row's `rust` value (a list of "<crate>::<fn>" names, or "N/A: <reason>")
survives every rerun, and a pytest id that disappears is kept and marked stale instead of being dropped.
Standard library only; it runs `pytest --collect-only` with each venv of the Python reference (gen/pyref.py: the
tag python-final, prepared by backend/parity/lib/python-release.mjs).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyref import REPO, WORKSPACE, tree  # noqa: E402

SIDECAR = WORKSPACE / "parity" / "test-map.json"
MD = WORKSPACE / "TEST-MAP.md"
SUITES = ("backend", "node")
SUITE_CRATES = {"backend": ["ec-hub"], "node": ["ec-proto", "ecnode"]}

# Areas: what the Python test file exercises (backend/PORTING.md section 4). Matched on the path inside the suite.
CORE_BACKEND = {"tests/core/test_api_smoke.py", "tests/core/test_security.py", "tests/core/test_stream_caps.py",
                "tests/core/test_audit_fixes.py", "tests/core/test_env_prefix.py", "tests/core/test_desktop.py",
                "tests/core/test_helpers.py", "tests/core/test_units.py", "tests/core/test_wiring.py",
                "tests/core/test_nodeerr.py", "tests/core/test_learn_engines_api.py"}
FLEET_BACKEND = {"tests/core/test_hardware.py", "tests/core/test_network.py"}
LIBRARY_BACKEND = {"tests/core/test_fit.py", "tests/core/test_fit_host_ram.py", "tests/core/test_fit_offload.py",
                   "tests/core/test_nodehttp_chat_bench.py"}
LIFECYCLE_BACKEND = {"tests/core/test_lifecycle_fleet.py", "tests/core/test_externals.py",
                     "tests/core/test_metrics_fleet.py", "tests/core/test_metrics_prune.py",
                     "tests/core/test_model_mounts.py", "tests/core/test_runspec_ulimits.py",
                     "tests/core/test_scripts.py", "tests/core/test_library_lifecycle_wiring.py"}
CORE_NODE = {"tests/test_proto.py", "tests/test_server.py"}
# Area names, in the order of PORTING.md section 4.
CORE, ADAPTERS_A, ADAPTERS_B, FLEET, LIBRARY, NODE, LIFECYCLE = (
    "core", "adapters: vLLM, SGLang", "adapters: TensorRT-LLM, llama.cpp, Ollama", "fleet", "library and data",
    "node agent", "lifecycle")


def owner(suite: str, path: str) -> tuple[str, bool]:
    """(area, listed): `listed` is False when the file is not in the table above (core, with a warning)."""
    if suite == "node":
        return (CORE if path in CORE_NODE else NODE), True
    if path in CORE_BACKEND or path.startswith("tests/learn/"):
        return CORE, True
    if path.split("/")[1] in ("vllm", "sglang", "trtllm", "llamacpp", "ollama"):
        return "adapters", True
    if path.startswith("tests/fleet/") or path in FLEET_BACKEND:
        return FLEET, True
    if path.startswith("tests/data/") or path in LIBRARY_BACKEND:
        return LIBRARY, True
    if path in LIFECYCLE_BACKEND:
        return LIFECYCLE, True
    return CORE, False


def split_adapters(own: str, test_id: str) -> str:
    """Adapter rows whose id (path, name or parameters) names trtllm, llamacpp or ollama belong to those three engines;
    the rest, including tests that cover every engine, to vLLM and SGLang."""
    if own != "adapters":
        return own
    return ADAPTERS_B if re.search(r"trtllm|llamacpp|ollama", test_id) else ADAPTERS_A


def collect(suite: str) -> list[str]:
    cwd = tree() / suite
    python = cwd / ".venv" / "bin" / "python"
    res = subprocess.run([str(python), "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"], cwd=cwd,
                         capture_output=True, text=True, check=False)
    ids = [line.strip() for line in res.stdout.splitlines() if "::" in line and not line.startswith(" ")]
    if res.returncode not in (0, 5) or not ids:
        sys.stderr.write(f"test_map: pytest --collect-only failed in {cwd} (exit {res.returncode}):\n{res.stderr[-2000:]}\n")
        raise SystemExit(2)
    return ids


def rust_fns() -> dict[str, dict[str, int]]:
    """crate -> {fn name: count} for every `fn name` under backend/crates/<crate>/ (src and tests)."""
    out: dict[str, dict[str, int]] = {}
    pat = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)")
    for crate_dir in sorted((WORKSPACE / "crates").glob("*")):
        if not crate_dir.is_dir():
            continue
        names: dict[str, int] = {}
        for f in crate_dir.rglob("*.rs"):
            if "target" in f.parts:
                continue
            for m in pat.finditer(f.read_text(encoding="utf-8", errors="replace")):
                names[m.group(1)] = names.get(m.group(1), 0) + 1
        out[crate_dir.name] = names
    return out


def base_name(test_id: str) -> str:
    name = test_id.split("::")[-1]
    return name.split("[", 1)[0]


def main(argv: list[str]) -> int:
    autofill = "--autofill" in argv
    old: dict[str, dict] = json.loads(SIDECAR.read_text()) if SIDECAR.exists() else {}
    rows: dict[str, dict] = {}
    unlisted: set[str] = set()
    fns = rust_fns() if autofill else {}
    filled = 0
    for suite in SUITES:
        for tid in collect(suite):
            path = tid.split("::", 1)[0]
            own, listed = owner(suite, path)
            if not listed:
                unlisted.add(f"{suite}/{path}")
            key = f"{suite}/{tid}"
            prev = old.get(key, {})
            row = {"owner": own, "rust": prev.get("rust", [])}
            if prev.get("reassigned"):     # a per-test area decision (with its reason) beats the file table
                row.update(owner=prev["owner"], reassigned=True, note=prev.get("note", ""))
            row["owner"] = split_adapters(row["owner"], tid)
            if autofill and not row["rust"]:
                hits = [c for c in SUITE_CRATES[suite] if fns.get(c, {}).get(base_name(tid)) == 1]
                if len(hits) == 1:
                    row["rust"] = [f"{hits[0]}::{base_name(tid)}"]
                    filled += 1
            rows[key] = row
    stale = [k for k in old if k not in rows]
    for k in stale:
        rows[k] = {**old[k], "stale": True}
    SIDECAR.write_text(json.dumps(dict(sorted(rows.items())), indent=1) + "\n")
    write_md(rows)
    for u in sorted(unlisted):
        print(f"test_map: WARNING: {u} is not in the area table; listed as core")
    counts: dict[str, list[int]] = {}
    for r in rows.values():
        c = counts.setdefault(r["owner"], [0, 0, 0])
        rust = r["rust"]
        c[0 if isinstance(rust, list) and rust else 1 if isinstance(rust, str) and rust.startswith("N/A") else 2] += 1
    print(f"test_map: {len(rows)} rows ({len(stale)} stale), {filled} autofilled; wrote {MD.relative_to(REPO)} and "
          f"{SIDECAR.relative_to(REPO)}")
    for o, (m, na, un) in sorted(counts.items()):
        print(f"  {o}: {m} mapped, {na} N/A, {un} unmapped")
    return 0


def write_md(rows: dict[str, dict]) -> None:
    lines = [
        "# TEST-MAP: every Python test and the Rust test that covers it",
        "",
        "Generated by `python3 -I backend/parity/gen/test_map.py` from `pytest --collect-only` of `backend/` and `node/` of",
        "the last Python release (the tag `python-final`), the code the Rust port was verified against;",
        "the source of truth is `backend/parity/test-map.json` (edit the `rust` field there, then rerun the generator).",
        "`backend/parity/check-test-map.mjs` verifies every named Rust test exists and counts unmapped rows.",
        "A Rust test is written `<crate>::<fn name>`; `N/A: <reason>` only for Python-only mechanics. The area is the part",
        "of the code a test exercises (backend/PORTING.md section 4); a row whose area differs from its file's carries the",
        "reason in the note column.",
        "",
        "| area | pytest ids | mapped | N/A | unmapped |",
        "|---|---|---|---|---|",
    ]
    counts: dict[str, list[int]] = {}
    for r in rows.values():
        c = counts.setdefault(r["owner"], [0, 0, 0])
        rust = r["rust"]
        c[0 if isinstance(rust, list) and rust else 1 if isinstance(rust, str) and rust.startswith("N/A") else 2] += 1
    for o, (m, na, un) in sorted(counts.items()):
        lines.append(f"| {o} | {m + na + un} | {m} | {na} | {un} |")
    tot = [sum(c[i] for c in counts.values()) for i in range(3)]
    lines += [f"| **all** | {sum(tot)} | {tot[0]} | {tot[1]} | {tot[2]} |", "",
              "| suite | pytest id | area | rust test | note |", "|---|---|---|---|---|"]
    for key, r in rows.items():
        suite, tid = key.split("/", 1)
        rust = r["rust"]
        cell = ", ".join(f"`{x}`" for x in rust) if isinstance(rust, list) else str(rust)
        if r.get("stale"):
            cell = f"(stale: no longer collected) {cell}"
        shown = tid.replace("|", "\\|")
        note = str(r.get("note", "")).replace("|", "\\|")
        lines.append(f"| {suite} | `{shown}` | {r['owner']} | {cell} | {note} |")
    MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
