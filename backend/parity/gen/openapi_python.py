"""Write backend/crates/ec-hub/assets/openapi.json: the Python app's OpenAPI document, which the Rust hub serves as is
at GET /api/v1/openapi.json (PORTING.md D5). It was generated while Python was the reference:

    "$(node backend/parity/lib/python-release.mjs)/backend/.venv/bin/python" -I backend/parity/gen/openapi_python.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

from engine_console.config import Settings
from engine_console.main import create_app

OUT = Path(__file__).resolve().parents[2] / "crates" / "ec-hub" / "assets" / "openapi.json"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Settings(data_dir=Path(tmp) / "data", fleet_enabled=False, background_tasks=False, otlp_enabled=False,
                       hosts_file=Path(tmp) / "hosts.json", hf_cache_dir=Path(tmp) / "hf")
        doc = create_app(cfg).openapi()
    # the document describes the Rust hub's release: its info.version is the workspace version (backend/Cargo.toml),
    # as tests/test_version.rs requires; everything else is the Python app's document as generated
    cargo = (Path(__file__).resolve().parents[2] / "Cargo.toml").read_text(encoding="utf-8")
    doc["info"]["version"] = re.search(r'^version = "([^"]+)"', cargo, re.M).group(1)  # type: ignore[union-attr]
    OUT.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    print(f"wrote {OUT} ({len(doc.get('paths', {}))} paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
