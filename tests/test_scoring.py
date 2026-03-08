"""Tests for importance scoring, decay functions, and temporal retrieval."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from widemem.core.types import (
    DecayFunction,
    Memory,
    MemorySearchResult,
    ScoringConfig,
)
from widemem.retrieval.temporal import score_and_rank
from widemem.scoring.decay import compute_recency_score
from widemem.scoring.importance import normalize_importance


class TestDecayFunctions:
    def test_no_decay(self):
        now = datetime(2026, 3, 8)
        old = datetime(2025, 1, 1)
        score = compute_recency_score(old, now, DecayFunction.NONE)
        assert score == 1.0

    def test_exponential_decay_recent(self):
        now = datetime(2026, 3, 8)
        yesterday = now - timedelta(days=1)
        score = compute_recency_score(yesterday, now, DecayFunction.EXPONENTIAL, decay_rate=0.01)
        assert score == pytest.approx(math.exp(-0.01), rel=1e-6)

    def test_exponential_decay_old(self):
        now = datetime(2026, 3, 8)
        old = now - timedelta(days=100)
        score = compute_recency_score(old, now, DecayFunction.EXPONENTIAL, decay_rate=0.01)
        assert score == pytest.approx(math.exp(-1.0), rel=1e-6)

    def test_exponential_decay_very_old(self):
        now = datetime(2026, 3, 8)
        old = now - timedelta(days=365)
        score = compute_recency_score(old, now, DecayFunction.EXPONENTIAL, decay_rate=0.01)
        assert score < 0.05

    def test_linear_decay(self):
        now = datetime(2026, 3, 8)
        ten_days_ago = now - timedelta(days=10)
        score = compute_recency_score(ten_days_ago, now, DecayFunction.LINEAR, decay_rate=0.01)
        assert score == pytest.approx(0.9, rel=1e-6)

    def test_linear_decay_floors_at_zero(self):
        now = datetime(2026, 3, 8)
        old = now - timedelta(days=200)
        score = compute_recency_score(old, now, DecayFunction.LINEAR, decay_rate=0.01)
        assert score == 0.0

    def test_step_decay_within_week(self):
        now = datetime(2026, 3, 8)
        recent = now - timedelta(days=3)
        assert compute_recency_score(recent, now, DecayFunction.STEP) == 1.0

    def test_step_decay_within_month(self):
        now = datetime(2026, 3, 8)
        two_weeks = now - timedelta(days=14)
        assert compute_recency_score(two_weeks, now, DecayFunction.STEP) == 0.7

    def test_step_decay_within_quarter(self):
        now = datetime(2026, 3, 8)
        two_months = now - timedelta(days=60)
        assert compute_recency_score(two_months, now, DecayFunction.STEP) == 0.4

    def test_step_decay_very_old(self):
        now = datetime(2026, 3, 8)
        old = now - timedelta(days=180)
        assert compute_recency_score(old, now, DecayFunction.STEP) == 0.1

    def test_same_time_returns_one(self):
        now = datetime(2026, 3, 8)
        for fn in [DecayFunction.EXPONENTIAL, DecayFunction.LINEAR, DecayFunction.STEP]:
            assert compute_recency_score(now, now, fn) == 1.0


class TestImportanceScoring:
    def test_normalize_mid(self):
        assert normalize_importance(5.0) == 0.5

    def test_normalize_max(self):
        assert normalize_importance(10.0) == 1.0

    def test_normalize_min(self):
        assert normalize_importance(0.0) == 0.0

    def test_normalize_clamps_high(self):
        assert normalize_importance(15.0) == 1.0

    def test_normalize_clamps_low(self):
        assert normalize_importance(-3.0) == 0.0


class TestScoreAndRank:
    def _make_result(self, content, similarity, importance, days_ago, now):
        return MemorySearchResult(
            memory=Memory(
                content=content,
                importance=importance,
                created_at=now - timedelta(days=days_ago),
            ),
            similarity_score=similarity,
        )

    def test_recent_high_importance_wins(self):
        now = datetime(2026, 3, 8)
        config = ScoringConfig()

        results = [
            self._make_result("old low", similarity=0.9, importance=3.0, days_ago=60, now=now),
            self._make_result("new high", similarity=0.8, importance=9.0, days_ago=1, now=now),
        ]

        ranked = score_and_rank(results, config, now=now)
        assert ranked[0].memory.content == "new high"

    def test_high_similarity_can_still_win(self):
        now = datetime(2026, 3, 8)
        config = ScoringConfig(similarity_weight=0.8, importance_weight=0.1, recency_weight=0.1)

        results = [
            self._make_result("exact match", similarity=0.99, importance=3.0, days_ago=30, now=now),
            self._make_result("vague match", similarity=0.3, importance=10.0, days_ago=0, now=now),
        ]

        ranked = score_and_rank(results, config, now=now)
        assert ranked[0].memory.content == "exact match"

    def test_time_filter_after(self):
        now = datetime(2026, 3, 8)
        config = ScoringConfig()
        cutoff = now - timedelta(days=7)

        results = [
            self._make_result("old", similarity=0.9, importance=8.0, days_ago=30, now=now),
            self._make_result("recent", similarity=0.8, importance=5.0, days_ago=3, now=now),
        ]

        ranked = score_and_rank(results, config, now=now, time_after=cutoff)
        assert len(ranked) == 1
        assert ranked[0].memory.content == "recent"

    def test_time_filter_before(self):
        now = datetime(2026, 3, 8)
        config = ScoringConfig()
        cutoff = now - timedelta(days=7)

        results = [
            self._make_result("old", similarity=0.9, importance=8.0, days_ago=30, now=now),
            self._make_result("recent", similarity=0.8, importance=5.0, days_ago=3, now=now),
        ]

        ranked = score_and_rank(results, config, now=now, time_before=cutoff)
        assert len(ranked) == 1
        assert ranked[0].memory.content == "old"

    def test_scores_are_populated(self):
        now = datetime(2026, 3, 8)
        config = ScoringConfig()

        results = [
            self._make_result("test", similarity=0.8, importance=7.0, days_ago=5, now=now),
        ]

        ranked = score_and_rank(results, config, now=now)
        r = ranked[0]
        assert r.similarity_score == 0.8
        assert r.importance_score == 0.7
        assert r.temporal_score > 0
        assert r.temporal_score <= 1.0
        assert r.final_score > 0

    def test_no_decay_config(self):
        now = datetime(2026, 3, 8)
        config = ScoringConfig(decay_function=DecayFunction.NONE)

        results = [
            self._make_result("old", similarity=0.5, importance=5.0, days_ago=365, now=now),
            self._make_result("new", similarity=0.5, importance=5.0, days_ago=1, now=now),
        ]

        ranked = score_and_rank(results, config, now=now)
        # With no decay, both should have the same final score
        assert ranked[0].final_score == pytest.approx(ranked[1].final_score, rel=1e-6)

    def test_empty_results(self):
        config = ScoringConfig()
        ranked = score_and_rank([], config)
        assert ranked == []


class TestTemporalIntegration:
    """Test that scoring integrates correctly with WideMemory.search()."""

    def test_search_returns_scored_results(self, tmp_dir):

        import numpy as np

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

        class MockLLM(BaseLLM):
            def __init__(self):
                super().__init__(LLMConfig())
            def generate(self, prompt, system=None):
                return "{}"
            def generate_json(self, prompt, system=None):
                return {"facts": []}

        class MockEmbedder(BaseEmbedder):
            def __init__(self):
                super().__init__(EmbeddingConfig(dimensions=64))
            def embed(self, text):
                rng = np.random.RandomState(hash(text) % 2**31)
                vec = rng.randn(64).astype(np.float32)
                return (vec / np.linalg.norm(vec)).tolist()
            def embed_batch(self, texts):
                return [self.embed(t) for t in texts]

        class DirectExtractor(BaseExtractor):
            def extract(self, text):
                return [Fact(content=text, importance=7.0)]

        from widemem.storage.vector.faiss_store import FAISSVectorStore
        config = MemoryConfig(
            history_db_path=f"{tmp_dir}/history.db",
            vector_store=VectorStoreConfig(path=f"{tmp_dir}/vectors"),
            embedding=EmbeddingConfig(dimensions=64),
        )
        embedder = MockEmbedder()
        vs = FAISSVectorStore(config.vector_store, dimensions=64)
        mem = WideMemory(config=config, llm=MockLLM(), embedder=embedder, vector_store=vs)
        mem.pipeline.extractor = DirectExtractor()

        mem.add("Lives in Berlin", user_id="alice")
        results = mem.search("Berlin", user_id="alice")

        assert len(results) >= 1
        r = results[0]
        assert r.final_score > 0
        assert r.temporal_score > 0
        assert r.importance_score > 0


@pytest.fixture
def tmp_dir():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        yield d
