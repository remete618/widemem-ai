"""Local-first defaults: MemoryConfig() runs on Ollama plus sentence-transformers,
a cloud provider is used only when configured, and each provider gets its own
default model unless one is set explicitly."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from widemem.core.memory import WideMemory
from widemem.core.types import EmbeddingConfig, LLMConfig, MemoryConfig


def _memory(config: MemoryConfig) -> WideMemory:
    m = WideMemory.__new__(WideMemory)
    m.config = config
    return m


def test_default_config_is_local():
    cfg = MemoryConfig()
    assert (cfg.llm.provider, cfg.llm.model) == ("ollama", "llama3.1:8b")
    assert (cfg.embedding.provider, cfg.embedding.model, cfg.embedding.dimensions) == (
        "sentence-transformers", "all-MiniLM-L6-v2", 384,
    )


@patch("ollama.Client")
def test_an_openai_key_in_the_env_does_not_switch_the_default_to_the_cloud(_client, monkeypatch):
    from widemem.providers.llm.ollama import OllamaLLM

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert isinstance(_memory(MemoryConfig())._create_llm(), OllamaLLM)


@pytest.mark.parametrize(
    "provider, model",
    [("ollama", "llama3.1:8b"), ("openai", "gpt-4o-mini"), ("anthropic", "claude-haiku-4-5-20251001")],
)
def test_each_llm_provider_gets_its_own_default_model(provider, model):
    llm = WideMemory._llm_config(LLMConfig(provider=provider))
    assert (llm.provider, llm.model) == (provider, model)


def test_explicit_llm_model_is_kept():
    assert WideMemory._llm_config(LLMConfig(provider="openai", model="gpt-4.1")).model == "gpt-4.1"


@pytest.mark.parametrize(
    "provider, model, dims",
    [
        ("sentence-transformers", "all-MiniLM-L6-v2", 384),
        ("ollama", "nomic-embed-text", 768),
        ("openai", "text-embedding-3-small", 1536),
    ],
)
def test_each_embedding_provider_gets_its_own_default_model(provider, model, dims):
    emb = WideMemory._embedding_config(EmbeddingConfig(provider=provider))
    assert (emb.model, emb.dimensions) == (model, dims)


def test_explicit_embedding_model_and_dimensions_are_kept():
    emb = WideMemory._embedding_config(
        EmbeddingConfig(provider="openai", model="text-embedding-3-large", dimensions=256)
    )
    assert (emb.model, emb.dimensions) == ("text-embedding-3-large", 256)


def test_explicit_openai_is_honoured(monkeypatch):
    from widemem.providers.llm.openai import OpenAILLM

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = MemoryConfig(llm=LLMConfig(provider="openai", api_key="sk-test"))
    assert isinstance(_memory(cfg)._create_llm(), OpenAILLM)
