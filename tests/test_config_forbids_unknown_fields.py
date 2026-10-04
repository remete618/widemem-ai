"""Config models reject unknown fields, so a typo like `embeddings=` fails
loudly instead of being ignored and silently running on defaults."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from widemem.core.types import (
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
    ScoringConfig,
    TopicConfig,
    VectorStoreConfig,
    YMYLConfig,
)

CONFIGS = [EmbeddingConfig, LLMConfig, MemoryConfig, ScoringConfig, TopicConfig, VectorStoreConfig, YMYLConfig]


@pytest.mark.parametrize("cls", CONFIGS, ids=lambda c: c.__name__)
def test_unknown_field_is_rejected(cls):
    with pytest.raises(ValidationError) as err:
        cls(definitely_not_a_field=1)
    assert "definitely_not_a_field" in str(err.value)


@pytest.mark.parametrize("cls", CONFIGS, ids=lambda c: c.__name__)
def test_known_defaults_still_construct(cls):
    cls()


def test_the_common_typo_names_the_field():
    with pytest.raises(ValidationError) as err:
        MemoryConfig(embeddings=EmbeddingConfig(provider="openai"))
    assert "embeddings" in str(err.value)


def test_nested_dict_config_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        MemoryConfig(llm={"provider": "ollama", "modle": "llama3.1:8b"})


def test_model_copy_and_dump_round_trip_still_work():
    # model_copy(update=...) is not validated by pydantic, so the forbid rule does
    # not apply there; widemem only ever updates known fields.
    cfg = MemoryConfig(llm=LLMConfig(provider="openai"))
    again = MemoryConfig.model_validate(cfg.model_dump())
    assert again.llm.provider == "openai"
    assert cfg.model_copy(update={"ttl_days": 3}).ttl_days == 3
