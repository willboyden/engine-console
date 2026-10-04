"""Host side, standard library only: build a one-file Hugging Face cache laid out the way the console writes it
(`hub/models--<org>--<name>/{refs/main, snapshots/<sha>/config.json -> ../../blobs/<sha256>, blobs/<sha256>}`).

    python make_hf_cache.py <scratch dir>/hfcache

Used with download_probe.py to see the engine resolve a model offline from a read-only mount (README.md)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def main(out: Path) -> None:
    repo = out / "hub" / "models--example-org--tiny"
    (repo / "refs").mkdir(parents=True)
    (repo / "blobs").mkdir()
    (repo / "snapshots" / COMMIT).mkdir(parents=True)
    (repo / "refs" / "main").write_text(COMMIT, encoding="utf-8")
    config = json.dumps({"architectures": ["LlamaForCausalLM"], "max_position_embeddings": 2048}).encode()
    blob = hashlib.sha256(config).hexdigest()
    (repo / "blobs" / blob).write_bytes(config)
    os.symlink(f"../../blobs/{blob}", repo / "snapshots" / COMMIT / "config.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
