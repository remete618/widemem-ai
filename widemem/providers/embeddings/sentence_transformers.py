from __future__ import annotations

from typing import List

from widemem.core.exceptions import ProviderError
from widemem.core.types import EmbeddingConfig
from widemem.providers.embeddings.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ProviderError(
                "Install sentence-transformers: pip install widemem[sentence-transformers]"
            )
        self._model = SentenceTransformer(
            config.model or "all-MiniLM-L6-v2",
        )
        actual_dim = self._model.get_sentence_embedding_dimension()
        if config.dimensions and config.dimensions != actual_dim:
            self.config = EmbeddingConfig(
                provider=config.provider,
                model=config.model,
                api_key=config.api_key,
                dimensions=actual_dim,
            )

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    @property
    def dimensions(self) -> int:
        return self.config.dimensions
