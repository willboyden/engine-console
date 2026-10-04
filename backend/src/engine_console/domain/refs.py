"""Small contract-v2 shapes shared by domain.models (Instance, requests) and domain.fleet.

LOCKED CONTRACT v2 (docs/ARCHITECTURE.md §10). Kept in their own module so models.py and fleet.py can both
import them without a cycle; fleet.py re-exports them.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from engine_console.adapters.base import ModelFormat

Management = Literal["managed", "script", "external"]   # who may start/stop an engine (§10.4)


class ModelRef(BaseModel):
    """What to load, on the host chosen for the instance. Exactly the fields of its format are used."""
    format: ModelFormat = "hf"
    repo_id: str | None = None           # hf: org/name, found in one of the host's HF cache roots
    root_id: str | None = None           # gguf / plain: a model root on that host ...
    rel: str | None = None               # ... and the path inside it (a .gguf file or first shard, or a model dir)
    name: str | None = None              # ollama: name:tag inside the Ollama server's store


class ModelLocation(BaseModel):
    host_id: str
    root_id: str
    root_label: str | None = None
    path: str                            # absolute path on that host
    drive: str | None = None             # MountInfo.label of the filesystem it lives on
    fstype: str | None = None
    remote: bool = False
    shared_from: str | None = None


class InstanceVram(BaseModel):
    gpu_index: int
    gpu_uuid: str
    used_gib: float | None = None        # this engine's processes on that GPU (None: driver cannot attribute)


class InstanceControls(BaseModel):
    """What the UI may offer for one instance; `reason` says why something is not offered."""
    start: bool = False
    stop: bool = False
    restart: bool = False
    remove: bool = False
    logs: bool = False
    configure: bool = False              # open the launch form prefilled ("clone as managed" for externals)
    reason: str | None = None
