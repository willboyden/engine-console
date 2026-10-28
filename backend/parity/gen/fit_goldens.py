"""Goldens for the Rust fit estimator: random-but-seeded inputs run through Python's estimate_fit and
estimate_host_ram, written to backend/crates/ec-hub/tests/data/fit_goldens.json. The Rust test (tests/test_fit.rs,
test_fit_matches_python_goldens) runs the same inputs and compares the JSON values exactly, so "unknown" and every
number appear exactly where Python puts them. Inputs that make Python raise are recorded as {"error": "<type>"}.

    "$(node backend/parity/lib/python-release.mjs)/backend/.venv/bin/python" -I backend/parity/gen/fit_goldens.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyref import DATA, tree  # noqa: E402

from engine_console.adapters.base import Compat, Hardware, ModelInfo  # noqa: E402
from engine_console.domain.models import HostMemInfo, ResidentUse  # noqa: E402
from engine_console.services.fit import estimate_fit, estimate_host_ram  # noqa: E402

OUT = DATA / "fit_goldens.json"
N = 600


def maybe(r: random.Random, p: float, v: object) -> object:
    return v if r.random() < p else None


def model(r: random.Random) -> dict[str, object]:
    layers = r.choice([None, 0, 1, 2, 12, 27, 36, 48, 64, 80, 126])
    heads = r.choice([None, 0, 2, 4, 6, 16, 32, 64, 128])
    raw: dict[str, object] = {}
    if r.random() < 0.3 and layers:
        raw["layer_types"] = [r.choice(["sliding_attention", "full_attention"]) for _ in range(layers)]
    if r.random() < 0.2:
        raw["sliding_window_pattern"] = r.choice([0, 1, 2, 4, 6, True])
    if r.random() < 0.15:
        raw["use_sliding_window"] = r.choice([False, True, None])
    if r.random() < 0.2:
        raw["qk_rope_head_dim"] = r.choice([0, 32, 64, 64.7, "64", None, -512])
    if r.random() < 0.1:
        raw = {"text_config": raw}
    return {
        "repo_id": "x/y",
        "num_params": maybe(r, 0.6, r.choice([0, 1_500_000_000, 7_000_000_000, 32_762_000_000, 405_000_000_000])),
        "weight_bytes": maybe(r, 0.6, r.choice([0, 10 * 1024**3, 3_000_000_000, 65_530_000_000, 141_107_000_000,
                                                 800_000_000_000])),
        "quantization": maybe(r, 0.4, r.choice(["fp8", "nvfp4", "awq", "gptq", "mxfp4", "int4", "bnb4", "q4_k_m"])),
        "dtype": maybe(r, 0.5, r.choice(["bfloat16", "float16", "float32", "BF16", ""])),
        "hidden_size": maybe(r, 0.8, r.choice([0, 64, 2048, 2880, 4096, 5120, 8192])),
        "num_layers": layers,
        "num_attention_heads": heads,
        "num_kv_heads": maybe(r, 0.7, r.choice([0, 1, 2, 4, 8, 16])),
        "head_dim": maybe(r, 0.5, r.choice([0, 64, 128, 256])),
        "max_position_embeddings": maybe(r, 0.8, r.choice([0, 2048, 4096, 32768, 40960, 131072, 1_000_000])),
        "sliding_window": maybe(r, 0.3, r.choice([0, 128, 512, 4096])),
        "kv_lora_rank": maybe(r, 0.2, r.choice([0, 512])),
        "raw_config": raw,
    }


def mem(r: random.Random) -> dict[str, float]:
    m: dict[str, float] = {}
    for k, choices in (("mem_fraction", [0.5, 0.85, 0.9, 0.95, 1.0]), ("max_len", [0.0, 1.0, 512.0, 4096.0, 32768.0,
                                                                                131072.0]),
                       ("tp", [0.0, 1.0, 2.0, 4.0, 8.0]), ("kv_bytes_per_elem", [1.0, 2.0, 0.5, 0.0]),
                       ("cuda_graph_gib", [0.0, 0.75, 1.0]), ("overhead_gib", [0.0, 1.0]),
                       ("activations_gib", [0.0, 2.5]), ("max_batched_tokens", [2048.0, 8192.0, 16384.0]),
                       ("nonstatic_reserve_frac", [0.01, 0.15]), ("share_of_free", [0.0, 1.0]),
                       ("claims_share", [0.0, 1.0]), ("outside_share_gib", [0.0, 2.5]),
                       ("kv_cache_fixed_gib", [0.0, 10.0, 80.5]), ("weights_gpu_fraction", [0.0, 0.5, 1.0, 1.7, -3.0]),
                       ("kv_gpu_fraction", [0.0, 0.25, 0.5, 1.0])):
        if r.random() < 0.3:
            m[k] = r.choice(choices)
    return m


def hw(r: random.Random) -> dict[str, object]:
    n = r.choice([0, 1, 1, 2, 2, 4, 8])
    total = r.choice([24.0, 48.0, 80.0, 96.0, 128.0])
    frees = [r.choice([total, total * 0.5, 10.0, 70.0, 75.0, 90.0, 0.0]) for _ in range(n)]
    return {"gpu_ids": list(range(n)), "gpu_uuids": [f"GPU-{i}" for i in range(n)], "gpu_names": ["G"] * n,
            "gpu_total_gib": [total] * n, "gpu_free_gib": frees, "compute_capability": "12.0", "host_ram_gib": 246.0}


def resident(r: random.Random, n: int) -> list[dict[str, object]]:
    return [{"name": f"eng-{j}", "gpu_ids": sorted(r.sample(range(max(n, 1)), k=1)), "gib_per_gpu": r.choice(
        [1.0, 21.0, 30.0, 60.0])} for j in range(r.choice([0, 0, 1, 2]))]


def host_case(r: random.Random) -> dict[str, object]:
    bd = {k: r.choice([0.0, 1.25, 8.0, 76.0, float("nan")]) for k in r.sample(
        ["pool", "hicache", "cpu_weight_offload", "pinned", "engine_process"], k=r.choice([0, 1, 2, 3]))}
    host = None if r.random() < 0.2 else {"total_gib": r.choice([16.0, 100.0, 200.0]), "used_gib": 1.0,
                                           "available_gib": r.choice([0.0, 3.0, 5.0, 50.0, 100.0]), "free_gib": 1.0,
                                           "cached_gib": 1.0, "shmem_gib": r.choice([0.0, 40.0]),
                                           "swap_total_gib": 0.0, "swap_used_gib": 0.0, "updated_at": 0.0}
    return {"breakdown": [[k, (None if v != v else v)] for k, v in bd.items()], "host": host}


def main() -> None:
    r = random.Random(20261028)
    fit_cases = []
    for _ in range(N):
        m, me, h = model(r), mem(r), hw(r)
        conc = r.choice([1, 1, 3, 4, 16])
        res = resident(r, len(h["gpu_ids"]))  # type: ignore[arg-type]
        compat = [{"level": "block", "code": "c", "message": "m"}] if r.random() < 0.1 else []
        hw_all = hw(r) if r.random() < 0.3 else None
        case: dict[str, object] = {"model": m, "mem": me, "hw": h, "concurrency": conc, "resident": res,
                                   "compat": compat, "hw_all": hw_all}
        try:
            rep = estimate_fit(ModelInfo(**m), dict(me), Hardware(**h), concurrency=conc,  # type: ignore[arg-type]
                               resident=[ResidentUse(**x) for x in res], compat=[Compat(**c) for c in compat],
                               hw_all=Hardware(**hw_all) if hw_all else None)  # type: ignore[arg-type]
            case["report"] = rep.model_dump(mode="json")
        except Exception as e:  # noqa: BLE001 - recorded: Rust must fail too
            case["error"] = type(e).__name__
        fit_cases.append(case)
    host_cases = []
    for _ in range(200):
        c = host_case(r)
        bd = {k: (float("nan") if v is None else v) for k, v in c["breakdown"]}  # type: ignore[union-attr]
        rep = estimate_host_ram(bd, HostMemInfo(**c["host"]) if c["host"] else None)  # type: ignore[arg-type]
        c["result"] = rep.model_dump(mode="json")
        host_cases.append(c)
    text = json.dumps({"fit": fit_cases, "host_ram": host_cases}, allow_nan=False)
    OUT.write_text(text + "\n")
    errors = sum(1 for c in fit_cases if "error" in c)
    unknown = sum(1 for c in fit_cases if c.get("report", {}).get("verdict") == "unknown")  # type: ignore[union-attr]
    print(f"wrote {OUT}: {len(fit_cases)} fit cases ({errors} errors, {unknown} unknown), {len(host_cases)} host cases")


if __name__ == "__main__":
    assert math.isfinite(1.0)
    main()
