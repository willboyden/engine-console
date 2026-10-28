"""Where the golden generators find the Python reference: the last Python release of the hub (backend/) and node agent
(node/), as backend/parity/lib/python-release.mjs prepares it (`git archive` of the `python-final` tag, or of the
commit it names, into backend/target/python-release/<commit>/, each package with its own `uv sync --frozen` venv).

The generators that import `engine_console` or `ecnode` run under that release's interpreter, from the repository root:

    PYREL=$(node backend/parity/lib/python-release.mjs)       # prepare once; prints the release root
    "$PYREL/backend/.venv/bin/python" -I backend/parity/gen/<generator>.py
    "$PYREL/node/.venv/bin/python" -I backend/parity/proto/gen_vectors.py

`EC_PYTHON_TREE=<dir>` names a prepared tree explicitly; otherwise the tree is the one whose venv runs this
interpreter, and failing that the default release (prepared on demand with `node`). Standard library only.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]  # backend/: the Cargo workspace
REPO = WORKSPACE.parent
DATA = WORKSPACE / "crates" / "ec-hub" / "tests" / "data"


def _is_tree(p: Path) -> bool:
    return (p / "backend" / "pyproject.toml").is_file() and (p / "node" / "pyproject.toml").is_file()


def tree() -> Path:
    """The root of the Python reference tree (holds backend/ and node/, each with its .venv)."""
    env = os.environ.get("EC_PYTHON_TREE")
    if env:
        p = Path(env).resolve()
        if not _is_tree(p):
            raise SystemExit(f"EC_PYTHON_TREE={env} holds no backend/ and node/ Python packages")
        return p
    # <root>/<backend|node>/.venv is this interpreter's prefix when run as the docstring says
    parents = Path(sys.prefix).resolve().parents
    if len(parents) > 1 and _is_tree(parents[1]):
        return parents[1]
    out = subprocess.run(["node", str(WORKSPACE / "parity" / "lib" / "python-release.mjs")], check=True,
                         capture_output=True, text=True).stdout.strip()
    return Path(out)


def hub_python() -> Path:
    return tree() / "backend" / ".venv" / "bin" / "python"


def node_python() -> Path:
    return tree() / "node" / ".venv" / "bin" / "python"
