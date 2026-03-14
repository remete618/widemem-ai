from __future__ import annotations

from openai import OpenAI

from widemem.core.exceptions import ProviderError
from widemem.core.types import EmbeddingConfig
from widemem.providers.embeddings.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self.client = OpenAI(api_key=config.api_key.get_secret_value() if config.api_key else None)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(
                model=self.config.model,
                input=texts,
                dimensions=self.config.dimensions,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise ProviderError(f"Embedding failed: {e}") from e
