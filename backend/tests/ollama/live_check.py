"""Developer tool (not collected by pytest): repeat the live environment check and rewrite fixtures/live_env_0.30.11.json.

    cd backend && uv run python tests/ollama/live_check.py

For the adapter's launch environment and for each value form in live_cases.py, a bare `ollama serve` is started inside ONE
throwaway container (`--network none`, `--cap-drop ALL`, no GPU, no model, an empty read-only store that has `blobs/` and
`manifests/`) and the effective values from its "server config" log line are recorded. The container comes from the docker
context in your environment (`DOCKER_CONTEXT`). The image must already be present locally; this script never pulls.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from engine_console.adapters.base import Hardware, LaunchContext  # noqa: E402
from engine_console.adapters.ollama import OllamaAdapter  # noqa: E402
from ollama.live_cases import ADAPTER_PARAMS, FORMS  # noqa: E402

DEST = HERE / "fixtures" / "live_env_0.30.11.json"


def parse_config(line: str) -> dict[str, str]:
    inner = line.split('env="map[', 1)[1].rsplit(']"', 1)[0]
    return {m.group(1): m.group(2) for m in re.finditer(r"([A-Za-z_]+):(\[[^\]]*\]|[^\s\[\]]*)", inner)}


def main() -> int:
    hw = Hardware(gpu_ids=[], gpu_uuids=[], gpu_names=[], gpu_total_gib=[], gpu_free_gib=[], compute_capability="", host_ram_gib=1.0)
    spec = OllamaAdapter().build_launch_ctx(LaunchContext(model="", model_format="ollama", served_name="x", hf_cache_container_path="/hf",
                                                          params=ADAPTER_PARAMS, hw=hw))
    runs = [("adapter", dict(spec.env)), *FORMS]
    lines = ["set -u"]
    for i, (_, env) in enumerate(runs):
        assigns = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        host = "" if "OLLAMA_HOST" in env else f"OLLAMA_HOST=127.0.0.1:{12000 + i} "
        lines.append(f"env {host}OLLAMA_NO_CLOUD=1 OLLAMA_NOPRUNE=1 {assigns} /bin/ollama serve > /tmp/run{i}.log 2>&1 &")
        lines.append("pid=$!; sleep 2.5")
        lines.append(f'echo "@@RUN {i}"; grep "server config" /tmp/run{i}.log; kill $pid 2>/dev/null; wait $pid 2>/dev/null')
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "blobs").mkdir()
        (Path(tmp) / "manifests").mkdir()
        cmd = ["docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL", "-v", f"{tmp}:/root/.ollama/models:ro",
               "--entrypoint", "/bin/bash", spec.image, "-c", "\n".join(lines)]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False).stdout
    chunks = re.split(r"@@RUN (\d+)\n", out)
    effective = {int(chunks[i]): parse_config(chunks[i + 1].strip().splitlines()[0])
                 for i in range(1, len(chunks) - 1, 2) if chunks[i + 1].strip()}
    if len(effective) != len(runs):
        print(f"expected {len(runs)} results, got {len(effective)}: is the image present locally?", file=sys.stderr)
        return 2
    result = {"image": spec.image,
              "container": "bare server, no GPU, no network, empty read-only store (blobs/ and manifests/)",
              "runs": [{"name": name, "env": env, "effective": {k: effective[i][k] for k in sorted(env) if k in effective[i]}}
                       for i, (name, env) in enumerate(runs)]}
    DEST.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(runs)} runs recorded in {DEST.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
