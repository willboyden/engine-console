"""Host side (backend virtualenv): write `cases.json` for validate_launch.py from the adapter's own output.

    cd backend && uv run python tests/trtllm/fixtures/capture/make_cases.py <scratch dir>/cases.json

Every catalog key, every scenario and every preset becomes one case. Write the file into a scratch directory next
to a copy of validate_launch.py (README.md shows the docker command); it is a generated artifact, not committed."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(out: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))      # the backend root, so `tests.trtllm` imports
    from tests.trtllm.samples import SAMPLES, SCENARIOS, sample_params

    from engine_console.adapters.base import Hardware, LaunchContext
    from engine_console.adapters.trtllm import TrtllmAdapter

    adapter = TrtllmAdapter()
    hw = Hardware(gpu_ids=[0, 1], gpu_uuids=["GPU-a", "GPU-b"], gpu_names=["x", "x"], gpu_total_gib=[95.6, 95.6],
                  gpu_free_gib=[90.0, 90.0], compute_capability="12.0", host_ram_gib=246.0)
    cases: list[dict[str, object]] = []

    def add(name: str, params: dict[str, object]) -> None:
        problems = adapter.validate(params)
        if problems:
            raise SystemExit(f"{name}: {problems}")
        spec = adapter.build_launch_ctx(LaunchContext(model="example-org/chat-model-36B", served_name="m",
                                                      hf_cache_container_path="/hf", params=params, hw=hw))
        cases.append({"name": name, "argv": spec.argv, "yaml": next(iter(spec.files.values()), "")})

    for key in SAMPLES:
        add(f"key:{key}", sample_params(key))
    for name, params in SCENARIOS.items():
        add(f"scenario:{name}", params)
    for name, params in adapter.presets().items():
        add(f"preset:{name}", params)
    out.write_text(json.dumps(cases, indent=1), encoding="utf-8")
    print(f"{len(cases)} cases written to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
