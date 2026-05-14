"""End-to-end wiring tests for MemoryConfig.parse_temporal_hints.

Verifies the feature flag composes correctly with search():
- Default off: no behavior change.
- Flag on + temporal query: hints auto-parsed and applied as time filters.
- Flag on + non-temporal query: no filtering applied.
- Explicit time_after / time_before: always wins over parsed hints.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest

from widemem.core.memory import WideMemory
from widemem.core.types import (
    EmbeddingConfig,
    Fact,
    LLMConfig,
    MemoryConfig,
    VectorStoreConfig,
)
from widemem.extraction.base import BaseExtractor
from widemem.providers.embeddings.base import BaseEmbedder
from widemem.providers.llm.base import BaseLLM
from widemem.storage.vector.faiss_store import FAISSVectorStore


class MockLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig())

    def _generate(self, prompt: str, system: str | None = None) -> str:
        return json.dumps({"facts": []})

    def _generate_json(self, prompt: str, system: str | None = None) -> dict:
        return {"facts": []}


class MockEmbedder(BaseEmbedder):
    def __init__(self, dimensions: int = 64) -> None:
        super().__init__(
            EmbeddingConfig(dimensions=dimensions), max_retries=1, retry_delay=0
        )
        self._vectors: dict[str, list[float]] = {}

    def _embed(self, text: str) -> list[float]:
        if text not in self._vectors:
            rng = np.random.RandomState(hash(text) % 2**31)
            vec = rng.randn(self.config.dimensions).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            self._vectors[text] = vec.tolist()
        return self._vectors[text]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


class MockExtractor(BaseExtractor):
    def extract(self, text: str):
        return [Fact(content=text, importance=7.0)]


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def store_factory(tmp_dir):
    """Returns a callable that builds a WideMemory with the requested flag."""

    def _make(parse_temporal_hints: bool):
        config = MemoryConfig(
            history_db_path=f"{tmp_dir}/history_{parse_temporal_hints}.db",
            parse_temporal_hints=parse_temporal_hints,
        )
        vector_store = FAISSVectorStore(
            VectorStoreConfig(path=f"{tmp_dir}/vectors_{parse_temporal_hints}"),
            dimensions=64,
        )
        mem = WideMemory(
            config=config,
            llm=MockLLM(),
            embedder=MockEmbedder(),
            vector_store=vector_store,
        )
        return mem

    return _make


def test_default_config_flag_is_off():
    """Backwards compatibility: existing users see no behavior change."""
    config = MemoryConfig()
    assert config.parse_temporal_hints is False


def test_flag_off_skips_parser(store_factory):
    """When flag is off, the parser is not invoked even on temporal queries."""
    mem = store_factory(parse_temporal_hints=False)
    with patch("widemem.core.memory.parse_temporal_hints") as mock_parse:
        mem.search("What happened last July?", user_id="alice")
        mock_parse.assert_not_called()


def test_flag_on_invokes_parser_for_temporal(store_factory):
    """When flag is on and query looks temporal, parser runs."""
    mem = store_factory(parse_temporal_hints=True)
    with patch(
        "widemem.core.memory.parse_temporal_hints",
        return_value=(
            datetime(2023, 7, 1, tzinfo=timezone.utc),
            datetime(2023, 7, 31, tzinfo=timezone.utc),
        ),
    ) as mock_parse:
        mem.search("What happened in July 2023?", user_id="alice")
        mock_parse.assert_called_once()


def test_flag_on_skips_parser_for_non_temporal(store_factory):
    """When flag is on but query isn't temporal, parser is skipped (cheap heuristic)."""
    mem = store_factory(parse_temporal_hints=True)
    with patch("widemem.core.memory.parse_temporal_hints") as mock_parse:
        mem.search("What is Alice's job?", user_id="alice")
        mock_parse.assert_not_called()


def test_explicit_time_args_win_over_parsed(store_factory):
    """Explicit time_after / time_before always override parsed hints."""
    mem = store_factory(parse_temporal_hints=True)
    explicit_after = datetime(2020, 1, 1, tzinfo=timezone.utc)
    explicit_before = datetime(2020, 12, 31, tzinfo=timezone.utc)
    with patch(
        "widemem.core.memory.parse_temporal_hints",
        return_value=(
            datetime(2023, 7, 1, tzinfo=timezone.utc),
            datetime(2023, 7, 31, tzinfo=timezone.utc),
        ),
    ) as mock_parse:
        mem.search(
            "What happened in July 2023?",
            user_id="alice",
            time_after=explicit_after,
            time_before=explicit_before,
        )
        mock_parse.assert_not_called()


def test_flag_on_no_temporal_match_returns_no_filter(store_factory):
    """Query looks temporal but parser returns None -> no filtering, no crash."""
    mem = store_factory(parse_temporal_hints=True)
    with patch(
        "widemem.core.memory.parse_temporal_hints",
        return_value=(None, None),
    ) as mock_parse:
        result = mem.search("When did Alice move?", user_id="alice")
        mock_parse.assert_called_once()
        assert result is not None
