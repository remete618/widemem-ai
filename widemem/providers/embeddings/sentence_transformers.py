from __future__ import annotations

import logging
import sys
from typing import List

from widemem.core.exceptions import ProviderError
from widemem.core.types import EmbeddingConfig
from widemem.providers.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


def _pin_torch_if_faiss_loaded(model) -> None:
    # faiss and torch each bundle libomp on macOS. With faiss imported first,
    # torch's CPU thread pool segfaults; one torch thread avoids it. Only the
    # CPU path is affected (MPS does not use the OpenMP pool).
    if sys.platform != "darwin" or "faiss" not in sys.modules:
        return
    if getattr(getattr(model, "device", None), "type", None) != "cpu":
        return
    import torch

    torch.set_num_threads(1)
    logger.info(
        "faiss was loaded before torch on macOS and the embedder runs on CPU; "
        "pinned torch to one thread to avoid a libomp crash (slower batch encoding)."
    )


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
        _pin_torch_if_faiss_loaded(self._model)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        actual_dim = get_dim()
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
