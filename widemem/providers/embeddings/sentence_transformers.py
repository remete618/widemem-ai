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
                "Install sentence-transformers: pip install \"widemem-ai[sentence-transformers]\""
            )
        model_name = config.model or "all-MiniLM-L6-v2"
        self._model = SentenceTransformer(model_name)
        actual_dim = self._model.get_sentence_embedding_dimension()
        if model_name != config.model or (config.dimensions and config.dimensions != actual_dim):
            self.config = config.model_copy(update={
                "model": model_name,
                "dimensions": actual_dim if config.dimensions else config.dimensions,
            })

    def _embed(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    @property
    def dimensions(self) -> int:
        return self.config.dimensions
