"""Ports (Protocols) for everything with side effects, so tests can supply trivial fakes."""
from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from engine_console.domain.models import GpuStat


class GpuProbe(Protocol):
    name: str

    def query(self) -> list[GpuStat]: ...


@dataclass
class CmdResult:
    rc: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    """Runs an argv list (never a shell). `env_extra` is merged over os.environ for the child only."""

    async def run(self, argv: Sequence[str], *, timeout: float, env_extra: Mapping[str, str] | None = None,
                  merge_stderr: bool = False) -> CmdResult: ...

    def stream(self, argv: Sequence[str]) -> AsyncIterator[str]: ...


@dataclass
class RepoFile:
    path: str
    size: int
    sha256: str | None = None  # LFS oid; None for plain git blobs (size-checked only)


@dataclass
class HubModel:
    """Raw facts from the hub about one repo revision (already parsed, no HTTP objects)."""
    repo_id: str
    sha: str | None
    gated: bool
    license: str | None
    pipeline_tag: str | None
    library_name: str | None
    tags: list[str]
    files: list[RepoFile]
    safetensors_params: dict[str, int] = field(default_factory=dict)  # dtype -> param count
    safetensors_total: int | None = None
    config: dict[str, Any] | None = None
    downloads: int | None = None
    likes: int | None = None


class ByteStream(Protocol):
    total: int | None
    offset: int  # byte offset the stream actually starts at (0 if the server ignored Range)

    def chunks(self) -> AsyncIterator[bytes]: ...
    async def aclose(self) -> None: ...


class HubClient(Protocol):
    async def search(self, *, q: str, task: str | None, library: str | None, quant: str | None,
                     sort: str, limit: int, token_ok: bool = True) -> list[HubModel]: ...

    async def model(self, repo_id: str, revision: str | None) -> HubModel: ...

    async def card(self, repo_id: str, revision: str | None) -> str: ...

    async def open_file(self, repo_id: str, revision: str, path: str, start: int) -> ByteStream: ...
