"""Pruner interface.

A pruner inspects a blob of agent output and decides whether it can usefully
shrink it. The orchestrator tries pruners in order and stops at the first one
whose compress returns a non-None summary.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class PrunerMetadata:
    name: str
    description: str
    version: str = "0.1.0"


class BasePruner(ABC):
    metadata: PrunerMetadata

    @abstractmethod
    def matches(self, text: str) -> bool:
        """Cheap gate run before compress() to skip irrelevant output."""
        ...

    @abstractmethod
    def compress(self, text: str) -> str | None:
        """Return a summary, or None to pass the text through verbatim."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.metadata.name!r})"

    def __str__(self) -> str:
        return f"{self.metadata.name} v{self.metadata.version} — {self.metadata.description}"
