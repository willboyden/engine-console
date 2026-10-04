"""Developer tool (not collected by pytest): repeat the live parser check and rewrite fixtures/parser_acceptance_b10156.json.

    cd backend && uv run python tests/llamacpp/live_check.py

It gives every launch argv the adapter builds (see acceptance_cases.py) to the pinned `llama-server` in ONE throwaway
container: `--network none`, `--cap-drop ALL`, no GPU, and a model path that does not exist, so no model is ever loaded and
no other container is touched. The container comes from the docker context in your environment (`DOCKER_CONTEXT`). The image
must already be present locally; this script never pulls.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from engine_console.adapters.llamacpp import LlamaCppAdapter  # noqa: E402
from engine_console.adapters.llamacpp_params import DEFAULT_IMAGE  # noqa: E402
from llamacpp.acceptance_cases import NEGATIVE_CASES, TEMPLATE_TEXT, cases  # noqa: E402

DEST = HERE / "fixtures" / "parser_acceptance_b10156.json"


def main() -> int:
    all_cases = cases(LlamaCppAdapter())
    runs = [(c["argv"], c["env"]) for c in all_cases] + [(argv, {}) for _, argv in NEGATIVE_CASES]
    lines = ["set -u", "mkdir -p /etc/engine-console",
             f"printf %s {shlex.quote(TEMPLATE_TEXT)} > /etc/engine-console/chat-template.jinja"]
    for i, (argv, env_map) in enumerate(runs):
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_map.items())
        args = " ".join(shlex.quote(a) for a in argv)
        lines.append(f'echo "@@CASE {i}"; env {env} timeout 30 /app/llama-server {args} 2>&1 | head -40')
    cmd = ["docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL", "--entrypoint", "/bin/sh", DEFAULT_IMAGE,
           "-c", "\n".join(lines)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1500, check=False).stdout
    chunks = re.split(r"@@CASE (\d+)\n", out)
    logs = {int(chunks[i]): chunks[i + 1] for i in range(1, len(chunks) - 1, 2)}
    if len(logs) != len(runs):
        print(f"expected {len(runs)} results, got {len(logs)}: is the image present locally?", file=sys.stderr)
        return 2

    result: dict[str, object] = {
        "image": DEFAULT_IMAGE, "container": "no network, no GPU, model path does not exist",
        "accepted_means": ("no line starting with \"error\" was printed (a rejected flag or value prints `error: invalid "
                           "argument` or `error while handling argument`) and the server printed \"loading model '<path>'\" "
                           "(the log line is silenced at verbosity 0, so that one case needs only the first condition; the "
                           "negative controls show an error line still appears at verbosity 0)")}
    recorded: list[dict[str, object]] = []
    for i, c in enumerate(all_cases):
        log = logs[i]
        silenced = "--verbosity" in c["argv"] and c["argv"][c["argv"].index("--verbosity") + 1] == "0"
        accepted = not re.search(r"^error", log, re.M) and ("loading model '" in log or silenced)
        entry: dict[str, object] = {"name": c["name"], "argv": c["argv"], "env": c["env"], "files": c["files"],
                                    "accepted": accepted}
        if silenced:
            entry["note"] = "log line silenced at verbosity 0; no error line"
        if not accepted:
            entry["first_lines"] = [ln[:200] for ln in log.splitlines()[:6]]
        recorded.append(entry)
    result["cases"] = recorded
    controls: list[dict[str, object]] = []
    for j, (name, argv) in enumerate(NEGATIVE_CASES):
        log = logs[len(all_cases) + j]
        errors = [ln[:200] for ln in log.splitlines() if ln.startswith("error")]
        controls.append({"name": name, "argv": argv, "rejected": bool(errors) and "loading model '" not in log,
                         "error_line": errors[0] if errors else None})
    result["negative_controls"] = controls
    DEST.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    refused = [e["name"] for e in recorded if not e["accepted"]]
    print(f"{len(recorded)} cases, {len(refused)} not accepted {refused or ''}; "
          f"{sum(1 for c in controls if c['rejected'])}/{len(controls)} negative controls rejected")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
