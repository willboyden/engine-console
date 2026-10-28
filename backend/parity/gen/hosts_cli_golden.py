#!/usr/bin/env python3
"""Golden transcripts of `engine-console hosts ...` from the Python reference, for the Rust port's byte-for-byte test.

    python3 -I backend/parity/gen/hosts_cli_golden.py      writes backend/crates/ec-hub/tests/data/hosts_cli_golden.json

The Python CLI is the last Python release's (gen/pyref.py).

Each case runs the Python CLI (`python -I -m engine_console.main hosts ...`) in a fresh temp dir with an environment
built from scratch: HOME, data dir, hosts.json and hub key inside the temp dir, PATH = the case's own fake bin dir,
then backend/parity/fakes/bin, then /usr/bin:/bin. A remote host is reached through a fake `ssh` written by the case
(first on PATH), never a real one, and its address is a `.invalid` name. Only `ssh-keygen` is real (a local key
in the temp dir). The Rust test `tests/test_hosts_cli_golden.rs` replays every case with the Rust binary and compares.

Normalisation, applied to both sides: the temp dir becomes `<T>`; the base64 of an `ssh-ed25519` key becomes
`<PUBKEY>`. A case may cut a stream after a marker (`cut`): everything after it is a documented deviation
(PORTING.md §14), so only the text up to the marker is compared.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyref import DATA, REPO, WORKSPACE, hub_python  # noqa: E402

PY = hub_python()
FAKES = WORKSPACE / "parity" / "fakes" / "bin"
OUT = DATA / "hosts_cli_golden.json"

FAKE_SSH = ("#!/bin/sh\n# fake ssh: the remote machine refuses the connection; nothing leaves this machine\n"
            "echo 'ssh: connect to host gpu-box.invalid port 22: Connection refused' >&2\nexit 255\n")

HOSTS_FULL = {"hosts": [
    {"id": "local", "label": "This box"},
    {"id": "gpu-box", "label": "GPU box", "ssh": "ops@gpu-box.invalid", "port": 2222},
    {"id": "aliased", "label": "Aliased", "ssh": "aliased-box", "ssh_config": True},
    {"id": "spare", "label": "Spare (off)", "ssh": "ops@spare.invalid", "enabled": False},
]}
HOSTS_REMOTE = {"hosts": [
    {"id": "local", "label": "This box", "enabled": False},
    {"id": "gpu-box", "label": "GPU box", "ssh": "ops@gpu-box.invalid"},
]}
HOSTS_ALL_OFF = {"hosts": [
    {"id": "local", "label": "This box", "enabled": False},
    {"id": "spare", "label": "Spare", "ssh": "ops@spare.invalid", "enabled": False},
]}


def hosts_file(doc: dict | str, mode: int = 0o600) -> dict:
    text = doc if isinstance(doc, str) else json.dumps(doc, indent=1) + "\n"
    return {"hosts.json": {"content": text, "mode": mode}}


KEYGEN = [["init-key", "--from", "10.0.0.5"]]
SSH_BIN = {"bin/ssh": {"content": FAKE_SSH, "mode": 0o755}}

# name, argv, files, env, setup (argv run first with the same implementation), cut {stream: marker}
CASES: list[dict] = [
    {"name": "no command", "argv": []},
    {"name": "help", "argv": ["--help"]},
    {"name": "help short", "argv": ["-h"]},
    {"name": "init-key help", "argv": ["init-key", "--help"]},
    {"name": "check help", "argv": ["check", "--help"]},
    {"name": "list help", "argv": ["list", "--help"]},
    {"name": "unknown command", "argv": ["bogus"]},
    {"name": "init-key without --from", "argv": ["init-key"]},
    {"name": "check with a bad timeout", "argv": ["check", "--timeout", "abc"]},
    {"name": "list with an extra argument", "argv": ["list", "extra"]},
    {"name": "list", "argv": ["list"], "files": hosts_file(HOSTS_FULL)},
    {"name": "list: hosts.json not JSON", "argv": ["list"], "files": hosts_file("{not json\n"),
     "cut": {"stderr": "not valid JSON: "}},
    {"name": "list: hosts.json group-writable", "argv": ["list"], "files": hosts_file(HOSTS_FULL, 0o664)},
    {"name": "list: unknown key", "argv": ["list"],
     "files": hosts_file({"hosts": [{"id": "local", "label": "L", "colour": "red"}]})},
    {"name": "list: duplicate id", "argv": ["list"],
     "files": hosts_file({"hosts": [{"id": "a", "label": "A", "ssh": "u@a.invalid"},
                                    {"id": "a", "label": "B", "ssh": "u@b.invalid"}]})},
    {"name": "init-key: new key", "argv": ["init-key", "--from", "10.0.0.5"]},
    {"name": "init-key: existing key", "argv": ["init-key", "--from", "10.0.0.5"], "setup": KEYGEN},
    {"name": "init-key: from anywhere", "argv": ["init-key", "--from", "*"]},
    {"name": "init-key: two addresses and a key file", "argv": ["init-key", "--from", "10.0.0.5,10.0.0.6",
                                                                "--key-file", "<T>/other/hub-key"]},
    {"name": "init-key: a bad address", "argv": ["init-key", "--from", "10.0.0.5 evil"]},
    {"name": "init-key: configured remote command", "argv": ["init-key", "--from", "10.0.0.5"],
     "env": {"REMOTE_NODE_COMMAND": "/opt/ecnode/ecnode --stdio"}},
    {"name": "check: every host disabled", "argv": ["check", "--timeout", "5"], "files": hosts_file(HOSTS_ALL_OFF),
     "setup": KEYGEN},
    {"name": "check: a remote host refuses ssh", "argv": ["check", "--timeout", "10"],
     "files": {**hosts_file(HOSTS_REMOTE), **SSH_BIN}, "setup": KEYGEN},
    {"name": "check: no hub key", "argv": ["check", "--timeout", "5"],
     "files": {**hosts_file(HOSTS_REMOTE), **SSH_BIN}},
]

PUBKEY = re.compile(r"(ssh-ed25519 )[A-Za-z0-9+/]+={0,3}")


def normalise(text: str, tmp: str) -> str:
    return PUBKEY.sub(r"\1<PUBKEY>", text.replace(tmp, "<T>"))


def case_env(tmp: str, extra: dict[str, str]) -> dict[str, str]:
    env = {
        "HOME": f"{tmp}/home", "LANG": "C.UTF-8", "PATH": f"{tmp}/bin:{FAKES}:/usr/bin:/bin",
        "XDG_CONFIG_HOME": f"{tmp}/xdg-config", "XDG_DATA_HOME": f"{tmp}/xdg-data",
        "XDG_STATE_HOME": f"{tmp}/xdg-state", "XDG_CACHE_HOME": f"{tmp}/xdg-cache", "XDG_RUNTIME_DIR": f"{tmp}/xdg-run",
        "ENGINE_CONSOLE_ENV_FILE": f"{tmp}/no-env-file", "ENGINE_CONSOLE_DATA_DIR": f"{tmp}/data",
        "HOSTS_FILE": f"{tmp}/hosts.json", "HUB_KEY_FILE": f"{tmp}/ssh/hub-key",
        "HF_ENDPOINT": "https://127.0.0.1:9", "OTLP_ENABLED": "false", "OTLP_ENDPOINT": "127.0.0.1:9",
    }
    env.update(extra)
    return env


def run(argv: list[str], tmp: str, env: dict[str, str]) -> dict:
    p = subprocess.run([str(PY), "-I", "-m", "engine_console.main", "hosts",
                        *[a.replace("<T>", tmp) for a in argv]],
                       cwd=f"{tmp}/home", env=env, capture_output=True, text=True, timeout=60, check=False)
    return {"code": p.returncode, "stdout": normalise(p.stdout, tmp), "stderr": normalise(p.stderr, tmp)}


def main() -> int:
    if not PY.exists():
        print(f"{PY} missing: prepare the Python reference with `node backend/parity/lib/python-release.mjs`", file=sys.stderr)
        return 2
    out = []
    for c in CASES:
        with tempfile.TemporaryDirectory(prefix="ec-hosts-golden-") as tmp:
            for d in ("home", "bin", "data"):
                os.makedirs(f"{tmp}/{d}", mode=0o700, exist_ok=True)
            for rel, f in (c.get("files") or {}).items():
                path = Path(tmp) / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f["content"])
                path.chmod(f["mode"])
            env = case_env(tmp, c.get("env") or {})
            for s in c.get("setup") or []:
                r = run(s, tmp, env)
                if r["code"] != 0:
                    print(f"setup of {c['name']!r} failed: {r}", file=sys.stderr)
                    return 1
            res = run(c["argv"], tmp, env)
            for stream, marker in (c.get("cut") or {}).items():
                i = res[stream].find(marker)
                if i < 0:
                    print(f"{c['name']!r}: marker {marker!r} not in {stream}", file=sys.stderr)
                    return 1
                res[stream] = res[stream][: i + len(marker)]
            out.append({**{k: v for k, v in c.items()}, "expect": res})
            print(f"{c['name']}: exit {res['code']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"generator": "backend/parity/gen/hosts_cli_golden.py", "cases": out}, indent=1,
                              ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} cases to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
