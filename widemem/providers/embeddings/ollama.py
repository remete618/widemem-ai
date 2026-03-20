from __future__ import annotations

from widemem.core.exceptions import ProviderError
from widemem.core.types import EmbeddingConfig
from widemem.providers.embeddings.base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        try:
            from ollama import Client
        except ImportError:
            raise ProviderError("Install ollama: pip install widemem[ollama]")
        self.client = Client(host=config.base_url or "http://localhost:11434")
        self._model = config.model or "nomic-embed-text"
        self._dimensions = config.dimensions

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            results = []
            for text in texts:
                response = self.client.embed(model=self._model, input=text)
                embeddings = response.get("embeddings", [])
                if not embeddings:
                    raise ProviderError(f"Ollama returned no embeddings for model {self._model}")
                results.append(embeddings[0])
            return results
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Ollama embedding failed: {e}") from e

    @property
    def dimensions(self) -> int:
        return self._dimensions
