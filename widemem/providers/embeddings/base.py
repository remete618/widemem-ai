from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict

from widemem.core.types import EmbeddingConfig


class BaseEmbedder(ABC):
    def __init__(self, config: EmbeddingConfig, cache_size: int = 1024) -> None:
        self.config = config
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_size = cache_size

    def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached
        result = self._embed(text)
        self._cache[text] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        if uncached_texts:
            new_embeddings = self._embed_batch(uncached_texts)
            for i, embedding in zip(uncached_indices, new_embeddings):
                results[i] = embedding
                self._cache[texts[i]] = embedding
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return results  # type: ignore[return-value]

    @abstractmethod
    def _embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    def dimensions(self) -> int:
        return self.config.dimensions
