from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from widemem.core.types import EmbeddingConfig
from widemem.providers.embeddings.openai import OpenAIEmbedder


def _response(vector):
    return SimpleNamespace(data=[SimpleNamespace(embedding=vector)])


def test_openai_embedder_omits_dimensions_for_legacy_models():
    client = MagicMock()
    client.embeddings.create.return_value = _response([0.1, 0.2, 0.3])
    with patch("openai.OpenAI", return_value=client):
        embedder = OpenAIEmbedder(EmbeddingConfig(model="text-embedding-ada-002", dimensions=3))
        assert embedder.embed("hello") == [0.1, 0.2, 0.3]

    kwargs = client.embeddings.create.call_args.kwargs
    assert kwargs["model"] == "text-embedding-ada-002"
    assert kwargs["input"] == ["hello"]
    assert "dimensions" not in kwargs


def test_openai_embedder_passes_dimensions_for_v3_models():
    client = MagicMock()
    client.embeddings.create.return_value = _response([0.1, 0.2, 0.3])
    with patch("openai.OpenAI", return_value=client):
        embedder = OpenAIEmbedder(EmbeddingConfig(model="text-embedding-3-small", dimensions=3))
        assert embedder.embed("hello") == [0.1, 0.2, 0.3]

    kwargs = client.embeddings.create.call_args.kwargs
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["input"] == ["hello"]
    assert kwargs["dimensions"] == 3


def test_openai_embedder_without_base_url_reaches_the_openai_api(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    embedder = OpenAIEmbedder(EmbeddingConfig(provider="openai", api_key="sk-test"))
    assert str(embedder.client.base_url) == "https://api.openai.com/v1/"


def test_openai_embedder_passes_an_explicit_base_url_to_the_sdk(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env-gateway.internal/v1")
    embedder = OpenAIEmbedder(
        EmbeddingConfig(provider="openai", api_key="sk-test", base_url="http://gateway.internal:8080/v1")
    )
    assert str(embedder.client.base_url) == "http://gateway.internal:8080/v1/"


def test_openai_embedder_without_base_url_honours_openai_base_url_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env-gateway.internal/v1")
    embedder = OpenAIEmbedder(EmbeddingConfig(provider="openai", api_key="sk-test"))
    assert str(embedder.client.base_url) == "http://env-gateway.internal/v1/"


def test_openai_embedder_treats_empty_base_url_as_unset(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env-gateway.internal/v1")
    embedder = OpenAIEmbedder(EmbeddingConfig(provider="openai", api_key="sk-test", base_url=""))
    assert str(embedder.client.base_url) == "http://env-gateway.internal/v1/"



def test_openai_embedder_treats_whitespace_base_url_as_unset(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env-gateway.internal/v1")
    embedder = OpenAIEmbedder(EmbeddingConfig(provider="openai", api_key="sk-test", base_url="   "))
    assert str(embedder.client.base_url) == "http://env-gateway.internal/v1/"

