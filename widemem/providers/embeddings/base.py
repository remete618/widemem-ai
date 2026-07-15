from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict

from widemem.core.exceptions import ProviderError
from widemem.core.types import EmbeddingConfig

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    def __init__(
        self,
        config: EmbeddingConfig,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_size: int = 1024,
    ) -> None:
        self.config = config
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_size = cache_size
        self._cache_lock = threading.RLock()

    def embed(self, text: str) -> list[float]:
        with self._cache_lock:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                return cached
        result = self._retry(self._embed, text)
        self._validate_vector(result)
        with self._cache_lock:
            self._cache[text] = result
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        with self._cache_lock:
            for i, text in enumerate(texts):
                cached = self._cache.get(text)
                if cached is not None:
                    self._cache.move_to_end(text)
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        if uncached_texts:
            new_embeddings = self._retry(self._embed_batch, uncached_texts)
            if len(new_embeddings) != len(uncached_texts):
                raise ProviderError(
                    f"Embedding provider returned {len(new_embeddings)} vectors "
                    f"for {len(uncached_texts)} inputs"
                )
            with self._cache_lock:
                for i, embedding in zip(uncached_indices, new_embeddings):
                    self._validate_vector(embedding)
                    results[i] = embedding
                    self._cache[texts[i]] = embedding
                    if len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
        return [result for result in results if result is not None]

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self.config.dimensions:
            raise ProviderError(
                f"Embedding dimension mismatch: expected {self.config.dimensions}, "
                f"got {len(vector)}"
            )

    def _retry(self, fn, arg):
        last_error = None
        for attempt in range(self._max_retries):
            try:
                return fn(arg)
            except ProviderError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning("Embedding attempt %d failed (%s), retrying in %.1fs", attempt + 1, e, delay)
                    time.sleep(delay)
        raise ProviderError(
            f"Embedding failed after {self._max_retries} retries: {last_error}"
        ) from last_error

    @abstractmethod
    def _embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    def dimensions(self) -> int:
        return self.config.dimensions
