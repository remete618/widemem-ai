from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from widemem.core.exceptions import ProviderError
from widemem.core.types import EmbeddingConfig

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    def __init__(
        self,
        config: EmbeddingConfig,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.config = config
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def embed(self, text: str) -> list[float]:
        return self._retry(self._embed, text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._retry(self._embed_batch, texts)

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
