from __future__ import annotations

from widemem.core.exceptions import ProviderError
from widemem.core.types import EmbeddingConfig
from widemem.providers.embeddings.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError('Install openai: pip install "widemem-ai[openai]"')
        self.client = OpenAI(
            api_key=config.api_key.get_secret_value() if config.api_key else None,
            base_url=(config.base_url or "").strip() or None,
        )

    def _embed(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            if self._supports_dimensions_param():
                response = self.client.embeddings.create(
                    model=self.config.model,
                    input=texts,
                    dimensions=self.config.dimensions,
                )
            else:
                response = self.client.embeddings.create(
                    model=self.config.model,
                    input=texts,
                )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise ProviderError(f"Embedding failed: {e}") from e

    def _supports_dimensions_param(self) -> bool:
        return self.config.model.startswith("text-embedding-3")
