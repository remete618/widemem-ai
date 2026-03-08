from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from widemem.core.types import VectorStoreConfig


class BaseVectorStore(ABC):
    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config

    @abstractmethod
    def insert(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def search(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Returns list of (id, score, metadata) tuples."""
        ...

    @abstractmethod
    def update(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def delete(self, id: str) -> None:
        ...

    @abstractmethod
    def get(self, id: str) -> tuple[list[float], dict[str, Any]] | None:
        ...

    def list_all(
        self,
        filters: dict[str, Any] | None = None,
        max_results: int = 1000,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Returns list of (id, metadata) for all matching entries. Override for efficiency."""
        raise NotImplementedError
