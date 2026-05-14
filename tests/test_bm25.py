"""Tests for the BM25 retriever and Reciprocal Rank Fusion helper."""

from __future__ import annotations

import pytest

from widemem.retrieval.bm25 import (
    BM25Retriever,
    reciprocal_rank_fusion,
    tokenize,
)


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------
def test_tokenize_lowercases():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_drops_punctuation():
    assert tokenize("Hello, world!") == ["hello", "world"]


def test_tokenize_keeps_digits():
    assert "401k" in tokenize("Her 401k balance")


def test_tokenize_handles_apostrophes():
    # "Caroline's" should produce one token "caroline's", not two
    tokens = tokenize("Caroline's necklace")
    assert "caroline's" in tokens


def test_tokenize_removes_stopwords_by_default():
    tokens = tokenize("the cat is on the mat")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "on" not in tokens
    assert "cat" in tokens and "mat" in tokens


def test_tokenize_can_keep_stopwords():
    tokens = tokenize("the cat", remove_stopwords=False)
    assert tokens == ["the", "cat"]


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_tokenize_only_stopwords_returns_empty():
    assert tokenize("the and of") == []


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------
@pytest.fixture
def docs():
    return [
        ("m1", "Caroline moved from Sweden to San Francisco."),
        ("m2", "She is severely allergic to penicillin."),
        ("m3", "Her 401k contribution rate is 12%."),
        ("m4", "She likes pottery and painting."),
        ("m5", "Her cat is named Mochi."),
    ]


def test_index_and_search_basic(docs):
    r = BM25Retriever()
    r.index(docs)
    assert len(r) == 5
    assert bool(r) is True

    results = r.search("penicillin allergy", top_k=3)
    assert results, "expected at least one result for an exact-keyword query"
    # The penicillin memory should rank first.
    assert results[0][0] == "m2"


def test_search_returns_ranked_by_score(docs):
    r = BM25Retriever()
    r.index(docs)
    results = r.search("Caroline Sweden", top_k=5)
    # m1 has both keywords; should top the list.
    assert results[0][0] == "m1"
    # Scores should be monotonically non-increasing.
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_search_top_k_caps_results(docs):
    r = BM25Retriever()
    r.index(docs)
    results = r.search("she", top_k=2)
    assert len(results) <= 2


def test_search_min_score_filters(docs):
    r = BM25Retriever()
    r.index(docs)
    # With a very high min_score, expect few or zero results.
    results = r.search("Caroline", top_k=10, min_score=100.0)
    assert results == []


def test_empty_index_returns_empty(docs):
    r = BM25Retriever()
    # No index() call yet.
    assert len(r) == 0
    assert bool(r) is False
    assert r.search("anything", top_k=3) == []


def test_index_with_empty_docs():
    r = BM25Retriever()
    r.index([])
    assert len(r) == 0
    assert r.search("anything", top_k=3) == []


def test_index_with_all_empty_content():
    """All documents tokenize to nothing (stopwords only)."""
    docs = [("m1", "the and of"), ("m2", "it is")]
    r = BM25Retriever()
    r.index(docs)
    # Tokenized corpus is two empty lists. BM25 should not crash; search
    # returns []. Behavior contract: graceful degradation.
    results = r.search("the cat", top_k=3)
    assert results == []


def test_empty_query(docs):
    r = BM25Retriever()
    r.index(docs)
    assert r.search("", top_k=3) == []
    assert r.search("   ", top_k=3) == []


def test_query_with_only_stopwords(docs):
    """A query that tokenizes to nothing should return []."""
    r = BM25Retriever()
    r.index(docs)
    assert r.search("the and of", top_k=3) == []


def test_no_matching_tokens(docs):
    r = BM25Retriever()
    r.index(docs)
    # Words that appear in no document should rank everything at ~0.
    results = r.search("zebra unicorn dragon", top_k=5)
    # Either zero results (with min_score) or all-zero scores filtered.
    for _, score in results:
        assert score > 0


def test_reindex_replaces_corpus(docs):
    r = BM25Retriever()
    r.index(docs)
    # BM25 IDF needs a few docs in the corpus to assign useful weight to
    # rare terms; 4 docs is enough for "astrophysics" to score positive.
    new_docs = [
        ("new1", "astrophysics neutron star observation"),
        ("new2", "cooking recipe pasta tomatoes"),
        ("new3", "house plant watering schedule"),
        ("new4", "annual bicycle maintenance log"),
    ]
    r.index(new_docs)
    assert len(r) == 4
    results = r.search("astrophysics", top_k=5)
    assert results[0][0] == "new1"


def test_case_insensitive(docs):
    r = BM25Retriever()
    r.index(docs)
    lower = r.search("penicillin", top_k=3)
    upper = r.search("PENICILLIN", top_k=3)
    assert lower == upper


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------
def test_rrf_combines_two_runs():
    """Items that rank well in both runs should beat items in only one."""
    vector_run = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    bm25_run = [("a", 10.0), ("c", 8.0), ("d", 5.0)]
    fused = reciprocal_rank_fusion([vector_run, bm25_run])
    fused_ids = [doc_id for doc_id, _ in fused]
    # 'a' is top of both runs -> highest fused score
    assert fused_ids[0] == "a"
    # 'b' and 'd' each appear in only one run, 'c' in both -> 'c' beats them
    assert fused_ids.index("c") < fused_ids.index("b")
    assert fused_ids.index("c") < fused_ids.index("d")


def test_rrf_single_run():
    """RRF on a single run preserves the input order."""
    run = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
    fused = reciprocal_rank_fusion([run])
    assert [doc_id for doc_id, _ in fused] == ["a", "b", "c"]


def test_rrf_empty_runs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_top_k():
    run = [("a", 1), ("b", 1), ("c", 1), ("d", 1)]
    fused = reciprocal_rank_fusion([run], top_k=2)
    assert len(fused) == 2


def test_rrf_scores_are_positive():
    run = [("a", 0.9), ("b", 0.1)]
    fused = reciprocal_rank_fusion([run])
    for _, score in fused:
        assert score > 0


def test_rrf_does_not_use_input_scores():
    """RRF is rank-based: identical input ordering should produce identical
    fused output regardless of the raw score magnitudes."""
    run_small_scores = [("a", 0.001), ("b", 0.0001), ("c", 0.00001)]
    run_large_scores = [("a", 5000.0), ("b", 4000.0), ("c", 3000.0)]
    fused_small = reciprocal_rank_fusion([run_small_scores])
    fused_large = reciprocal_rank_fusion([run_large_scores])
    assert [doc_id for doc_id, _ in fused_small] == [
        doc_id for doc_id, _ in fused_large
    ]
    # And the fused scores should also be identical.
    assert dict(fused_small) == dict(fused_large)


# ---------------------------------------------------------------------------
# Behavioral integration assertion (catches the "BM25 strictly helps" property)
# ---------------------------------------------------------------------------
def test_bm25_finds_exact_keywords_vector_search_might_miss(docs):
    """The property that makes hybrid search worth shipping: keyword matches
    that semantic similarity would not naturally rank first. This test asserts
    BM25 surfaces them.
    """
    r = BM25Retriever()
    r.index(docs)
    # 401k is a specific identifier; it should rank above general 'rate' matches.
    results = r.search("401k", top_k=3)
    assert results[0][0] == "m3"

    # Mochi is a proper noun; it should surface despite low context.
    results = r.search("Mochi", top_k=3)
    assert results[0][0] == "m5"
