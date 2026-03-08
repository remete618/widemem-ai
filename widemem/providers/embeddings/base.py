from __future__ import annotations

from abc import ABC, abstractmethod

from widemem.core.types import EmbeddingConfig


class BaseEmbedder(ABC):
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    def dimensions(self) -> int:
        return self.config.dimensions
