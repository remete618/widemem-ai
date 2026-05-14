"""Unit + wiring tests for the hybrid (vector + BM25) retrieval path.

Covers:
- blend_hybrid_scores mutates similarity_score in place using a min-max
  normalized blend.
- Boundary inputs (empty results, empty query, zero weight, full weight,
  equal vector scores, equal BM25 scores) are handled gracefully.
- bm25_weight=0 produces vector-only behavior; bm25_weight=1 produces
  pure-keyword behavior.
- MemoryConfig.enable_hybrid_search wires through to the blend function
  inside search(); default off means the function is never called.
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
    Memory,
    MemoryConfig,
    MemorySearchResult,
    MemoryTier,
    VectorStoreConfig,
)
from widemem.extraction.base import BaseExtractor
from widemem.providers.embeddings.base import BaseEmbedder
from widemem.providers.llm.base import BaseLLM
from widemem.retrieval.hybrid import _min_max_normalize, blend_hybrid_scores
from widemem.storage.vector.faiss_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# _min_max_normalize
# ---------------------------------------------------------------------------
def test_normalize_basic():
    assert _min_max_normalize([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]


def test_normalize_negatives():
    assert _min_max_normalize([-1.0, 0.0, 1.0]) == [0.0, 0.5, 1.0]


def test_normalize_all_equal_returns_half():
    assert _min_max_normalize([5.0, 5.0, 5.0]) == [0.5, 0.5, 0.5]


def test_normalize_single_element():
    assert _min_max_normalize([42.0]) == [0.5]


def test_normalize_empty():
    assert _min_max_normalize([]) == []


# ---------------------------------------------------------------------------
# blend_hybrid_scores  (direct unit tests on synthetic MemorySearchResults)
# ---------------------------------------------------------------------------
def _make_result(id_: str, content: str, similarity: float) -> MemorySearchResult:
    now = datetime.now(timezone.utc)
    return MemorySearchResult(
        memory=Memory(
            id=id_,
            content=content,
            user_id="alice",
            tier=MemoryTier.FACT,
            importance=7.0,
            created_at=now,
            updated_at=now,
        ),
        similarity_score=similarity,
    )


@pytest.fixture
def candidate_pool():
    return [
        _make_result("m1", "Caroline moved from Sweden", similarity=0.30),
        _make_result("m2", "Penicillin allergy critical", similarity=0.40),
        _make_result("m3", "Her cat is named Mochi", similarity=0.50),
        _make_result("m4", "Annual income $80,000", similarity=0.25),
    ]


def test_blend_mutates_similarity_score(candidate_pool):
    before = [r.similarity_score for r in candidate_pool]
    blend_hybrid_scores(candidate_pool, "penicillin", bm25_weight=0.5)
    after = [r.similarity_score for r in candidate_pool]
    assert before != after
    # All blended scores in [0, 1] after min-max normalization.
    assert all(0.0 <= s <= 1.0 for s in after)


def test_blend_bm25_zero_weight_recovers_vector_normalization(candidate_pool):
    """With bm25_weight=0, the blended score equals the normalized vector score."""
    raw_vec = [r.similarity_score for r in candidate_pool]
    expected = _min_max_normalize(raw_vec)
    blend_hybrid_scores(candidate_pool, "penicillin", bm25_weight=0.0)
    blended = [r.similarity_score for r in candidate_pool]
    for got, want in zip(blended, expected):
        assert abs(got - want) < 1e-9


def test_blend_bm25_full_weight_uses_only_keyword_signal(candidate_pool):
    """With bm25_weight=1, vector similarity is ignored; pure BM25-driven ranking."""
    blend_hybrid_scores(candidate_pool, "penicillin", bm25_weight=1.0)
    # m2 contains "penicillin" exactly; it should now have the highest score.
    scored = {r.memory.id: r.similarity_score for r in candidate_pool}
    assert scored["m2"] == max(scored.values())


def test_blend_with_keyword_promotes_exact_match(candidate_pool):
    """With balanced weight, an exact keyword match should outrank pure
    vector top by enough to flip the order."""
    # Pre-blend: m3 (Mochi) has highest similarity (0.50).
    pre = max(candidate_pool, key=lambda r: r.similarity_score).memory.id
    assert pre == "m3"
    blend_hybrid_scores(candidate_pool, "penicillin allergy", bm25_weight=0.5)
    post = max(candidate_pool, key=lambda r: r.similarity_score).memory.id
    # m2 has the keyword match for "penicillin allergy"; with bm25_weight=0.5
    # it should top the list after the blend.
    assert post == "m2"


def test_blend_empty_pool_no_error():
    blend_hybrid_scores([], "anything", bm25_weight=0.5)  # should not raise


def test_blend_empty_query_is_noop(candidate_pool):
    before = [r.similarity_score for r in candidate_pool]
    blend_hybrid_scores(candidate_pool, "", bm25_weight=0.5)
    after = [r.similarity_score for r in candidate_pool]
    assert before == after


def test_blend_whitespace_query_is_noop(candidate_pool):
    before = [r.similarity_score for r in candidate_pool]
    blend_hybrid_scores(candidate_pool, "   \n\t  ", bm25_weight=0.5)
    after = [r.similarity_score for r in candidate_pool]
    assert before == after


def test_blend_rejects_out_of_range_weight(candidate_pool):
    with pytest.raises(ValueError):
        blend_hybrid_scores(candidate_pool, "anything", bm25_weight=1.5)
    with pytest.raises(ValueError):
        blend_hybrid_scores(candidate_pool, "anything", bm25_weight=-0.1)


def test_blend_all_equal_vector_scores_still_uses_bm25():
    """When vector similarity is uninformative (all equal), BM25 alone drives
    the ranking."""
    pool = [
        _make_result("m1", "Caroline moved to Boston", similarity=0.5),
        _make_result("m2", "She is allergic to penicillin", similarity=0.5),
        _make_result("m3", "Her cat is named Mochi", similarity=0.5),
    ]
    blend_hybrid_scores(pool, "penicillin", bm25_weight=0.5)
    # m2 has the keyword match; should top the list.
    top = max(pool, key=lambda r: r.similarity_score).memory.id
    assert top == "m2"


# ---------------------------------------------------------------------------
# Wiring tests through WideMemory.search()
# ---------------------------------------------------------------------------
class _MockLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig())

    def _generate(self, prompt: str, system: str | None = None) -> str:
        return json.dumps({"facts": []})

    def _generate_json(self, prompt: str, system: str | None = None) -> dict:
        return {"facts": []}


class _MockEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        super().__init__(EmbeddingConfig(dimensions=64), max_retries=1, retry_delay=0)
        self._cache: dict[str, list[float]] = {}

    def _embed(self, text: str) -> list[float]:
        if text not in self._cache:
            rng = np.random.RandomState(abs(hash(text)) % 2**31)
            v = rng.randn(self.config.dimensions).astype(np.float32)
            v = v / np.linalg.norm(v)
            self._cache[text] = v.tolist()
        return self._cache[text]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


class _MockExtractor(BaseExtractor):
    def extract(self, text: str):
        return [Fact(content=text, importance=7.0)]


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def store_factory(tmp_dir):
    def _make(enable_hybrid_search: bool):
        config = MemoryConfig(
            history_db_path=f"{tmp_dir}/history_{enable_hybrid_search}.db",
            enable_hybrid_search=enable_hybrid_search,
        )
        vector_store = FAISSVectorStore(
            VectorStoreConfig(path=f"{tmp_dir}/vectors_{enable_hybrid_search}"),
            dimensions=64,
        )
        return WideMemory(
            config=config,
            llm=_MockLLM(),
            embedder=_MockEmbedder(),
            vector_store=vector_store,
        )

    return _make


def test_default_config_flag_is_off():
    """Backwards compatibility: existing users see no behavior change."""
    config = MemoryConfig()
    assert config.enable_hybrid_search is False
    assert config.hybrid_bm25_weight == 0.5


def test_flag_off_skips_blend(store_factory):
    """When flag is off, blend_hybrid_scores must not be called."""
    mem = store_factory(enable_hybrid_search=False)
    # Patch at the source module since the import inside search() is lazy.
    with patch("widemem.retrieval.hybrid.blend_hybrid_scores") as mock_blend:
        mem.search("anything", user_id="alice")
        mock_blend.assert_not_called()


def test_flag_on_no_candidates_skips_blend(store_factory):
    """Flag on but empty candidate pool: blend should not be invoked."""
    mem = store_factory(enable_hybrid_search=True)
    with patch("widemem.retrieval.hybrid.blend_hybrid_scores") as mock_blend:
        # Fresh store -> no memories -> vector search returns []
        mem.search("penicillin", user_id="alice")
        mock_blend.assert_not_called()
