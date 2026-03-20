from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from widemem.core.exceptions import ProviderError
from widemem.core.types import EmbeddingConfig

ollama = pytest.importorskip("ollama", reason="ollama package not installed")


class TestOllamaEmbedder:
    def _make_config(self, **kwargs):
        defaults = {
            "provider": "ollama",
            "model": "nomic-embed-text",
            "dimensions": 768,
        }
        defaults.update(kwargs)
        return EmbeddingConfig(**defaults)

    @patch("ollama.Client")
    def test_embed_single(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_client_cls.return_value = mock_client

        from widemem.providers.embeddings.ollama import OllamaEmbedder

        embedder = OllamaEmbedder(self._make_config(dimensions=3))
        result = embedder.embed("hello world")

        assert result == [0.1, 0.2, 0.3]
        mock_client.embed.assert_called_once_with(model="nomic-embed-text", input="hello world")

    @patch("ollama.Client")
    def test_embed_batch(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.embed.side_effect = [
            {"embeddings": [[0.1, 0.2, 0.3]]},
            {"embeddings": [[0.4, 0.5, 0.6]]},
        ]
        mock_client_cls.return_value = mock_client

        from widemem.providers.embeddings.ollama import OllamaEmbedder

        embedder = OllamaEmbedder(self._make_config(dimensions=3))
        result = embedder.embed_batch(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @patch("ollama.Client")
    def test_embed_batch_empty(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()

        from widemem.providers.embeddings.ollama import OllamaEmbedder

        embedder = OllamaEmbedder(self._make_config(dimensions=3))
        result = embedder.embed_batch([])
        assert result == []

    @patch("ollama.Client")
    def test_embed_empty_response_raises(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.embed.return_value = {"embeddings": []}
        mock_client_cls.return_value = mock_client

        from widemem.providers.embeddings.ollama import OllamaEmbedder

        embedder = OllamaEmbedder(self._make_config(dimensions=3))
        with pytest.raises(ProviderError, match="no embeddings"):
            embedder.embed("test")

    @patch("ollama.Client")
    def test_custom_host(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()

        from widemem.providers.embeddings.ollama import OllamaEmbedder

        OllamaEmbedder(self._make_config(base_url="http://myhost:11434", dimensions=3))
        mock_client_cls.assert_called_once_with(host="http://myhost:11434")

    @patch("ollama.Client")
    def test_default_host(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()

        from widemem.providers.embeddings.ollama import OllamaEmbedder

        OllamaEmbedder(self._make_config(dimensions=3))
        mock_client_cls.assert_called_once_with(host="http://localhost:11434")

    @patch("ollama.Client")
    def test_default_model(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.embed.return_value = {"embeddings": [[0.1]]}
        mock_client_cls.return_value = mock_client

        from widemem.providers.embeddings.ollama import OllamaEmbedder

        config = self._make_config(dimensions=1)
        config.model = ""
        embedder = OllamaEmbedder(config)
        embedder.embed("test")
        mock_client.embed.assert_called_once_with(model="nomic-embed-text", input="test")

    @patch("ollama.Client")
    def test_dimensions_property(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()

        from widemem.providers.embeddings.ollama import OllamaEmbedder

        embedder = OllamaEmbedder(self._make_config(dimensions=768))
        assert embedder.dimensions == 768

    @patch("ollama.Client")
    def test_connection_error_wraps(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.embed.side_effect = ConnectionError("connection refused")
        mock_client_cls.return_value = mock_client

        from widemem.providers.embeddings.ollama import OllamaEmbedder

        embedder = OllamaEmbedder(self._make_config(dimensions=3))
        with pytest.raises(ProviderError, match="Ollama embedding failed"):
            embedder.embed("test")


class TestOllamaEmbedderFactory:
    def test_memory_creates_ollama_embedder(self):
        from widemem.core.memory import WideMemory
        from widemem.core.types import MemoryConfig

        config = MemoryConfig(
            embedding=EmbeddingConfig(provider="ollama", model="nomic-embed-text", dimensions=768),
        )
        with patch("ollama.Client"):
            mem = WideMemory.__new__(WideMemory)
            mem.config = config
            embedder = mem._create_embedder()
            assert type(embedder).__name__ == "OllamaEmbedder"
