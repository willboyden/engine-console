"""A fake `nvidia-smi`: GPUs and compute apps from $FAKE_SMI_STATE (JSON); `--query-*` CSV like the real one. Each
call is appended to $FAKE_SMI_STATE.log as {"args": [...], "env_names": [...]} (CHILD_ENV, set by the wrapper)."""
from __future__ import annotations

import json
import os
import sys

CHILD_ENV = None        # the wrapper sets it before main()


def main():
    with open(os.environ["FAKE_SMI_STATE"]) as f:
        state = json.load(f)
    args = sys.argv[1:]
    with open(os.environ["FAKE_SMI_STATE"] + ".log", "a") as f:
        f.write(json.dumps({"args": args, "env_names": sorted(os.environ if CHILD_ENV is None else CHILD_ENV)}) + "\n")
    if not state["gpus"]:
        print("No devices were found")
        return 6
    if not args:
        print(f"| NVIDIA-SMI {state['driver']}   Driver Version: {state['driver']}   CUDA Version: {state['cuda']}  |")
        return 0
    query = next((a for a in args if a.startswith("--query-")), None)
    if query is None:
        return 2
    kind, _, fields = query.partition("=")
    rows = state["gpus"] if kind == "--query-gpu" else state.get("apps", [])
    for row in rows:
        vals = []
        for field in fields.split(","):
            if field not in row:
                print(f'Field "{field}" is not a valid field to query.', file=sys.stderr)
                return 2
            vals.append(str(row[field]))
        print(", ".join(vals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
