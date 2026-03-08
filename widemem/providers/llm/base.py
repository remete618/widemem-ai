from __future__ import annotations

import time
from abc import ABC

from widemem.core.exceptions import ProviderError
from widemem.core.types import LLMConfig


class BaseLLM(ABC):
    def __init__(self, config: LLMConfig, max_retries: int = 3, retry_delay: float = 1.0) -> None:
        self.config = config
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _generate(self, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError

    def _generate_json(self, prompt: str, system: str | None = None) -> dict:
        raise NotImplementedError

    def generate(self, prompt: str, system: str | None = None) -> str:
        return self._retry(self._generate, prompt, system)

    def generate_json(self, prompt: str, system: str | None = None) -> dict:
        return self._retry(self._generate_json, prompt, system)

    def _retry(self, fn, prompt: str, system: str | None):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return fn(prompt, system)
            except ProviderError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
        raise ProviderError(
            f"Failed after {self.max_retries} retries: {last_error}"
        ) from last_error
