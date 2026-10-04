"""Readers for the captured Ollama output (fixtures/): the `ollama serve --help` variable list and the "server config"
line the server logs at start-up (the effective value of every variable)."""
from __future__ import annotations

import re
from pathlib import Path

FIX = Path(__file__).parent / "fixtures"


def read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def help_env(text: str | None = None) -> dict[str, str]:
    """name -> description of every variable under "Environment Variables:" in the help."""
    body = (text if text is not None else read("ollama-serve-help.txt")).split("Environment Variables:", 1)[1]
    out: dict[str, str] = {}
    for line in body.splitlines():
        m = re.match(r"\s+([A-Z][A-Z0-9_]+)\s{2,}(.+)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def server_config(text: str | None = None) -> dict[str, str]:
    """name -> effective value, from `msg="server config" env="map[NAME:value ...]"` (lists keep their brackets)."""
    line = next(ln for ln in (text if text is not None else read("serve_bare_0.30.11.log")).splitlines() if 'msg="server config"' in ln)
    inner = line.split('env="map[', 1)[1].rsplit(']"', 1)[0]
    return {m.group(1): m.group(2) for m in re.finditer(r"([A-Za-z_]+):(\[[^\]]*\]|[^\s\[\]]*)", inner)}
