"""The slice of H2b's LibraryService that the runtime services need, as a Protocol so tests can supply a fake and the
composition root can run before the library exists. A copy is one model on one host's root (`LibraryCopy`)."""
from __future__ import annotations

from typing import Protocol

from engine_console.adapters.base import ModelInfo
from engine_console.domain.fleet import LibraryCopy, RootInfo
from engine_console.domain.refs import ModelRef


class LibraryPort(Protocol):
    def find(self, host_id: str, ref: ModelRef) -> LibraryCopy | None: ...
    def locate(self, host_id: str, ref: ModelRef) -> LibraryCopy | None: ...
    async def model_info(self, host_id: str, ref: ModelRef) -> ModelInfo | None: ...
    def hf_root(self, host_id: str) -> RootInfo | None: ...
    def ollama_root(self, host_id: str) -> RootInfo | None: ...
    def copies(self, host_id: str) -> list[LibraryCopy]: ...


class NullLibrary:
    """No library: every model is missing. Used only while the library service is not wired in."""

    def find(self, host_id: str, ref: ModelRef) -> LibraryCopy | None:
        return None

    def locate(self, host_id: str, ref: ModelRef) -> LibraryCopy | None:
        return None

    async def model_info(self, host_id: str, ref: ModelRef) -> ModelInfo | None:
        return None

    def hf_root(self, host_id: str) -> RootInfo | None:
        return None

    def ollama_root(self, host_id: str) -> RootInfo | None:
        return None

    def copies(self, host_id: str) -> list[LibraryCopy]:
        return []
