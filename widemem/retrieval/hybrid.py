"""Hybrid retrieval: blend BM25 keyword scores into the vector similarity signal.

The integration is deliberately small. Existing search() behavior is untouched
when MemoryConfig.enable_hybrid_search is False (default). When enabled, the
vector-search candidate pool is reranked once via BM25; the combined score
replaces each candidate's similarity_score before the existing scoring
pipeline (importance + recency + YMYL boosts) runs.

Why this shape:
- Adds no new fields to MemorySearchResult or Memory.
- Does not change the candidate set: every memory the vector store returned
  still passes through. BM25 only reshapes the relative ranking.
- Downstream (score_and_rank, hierarchy routing, confidence assessment)
  operates on the new similarity_score without any other change.

Why not RRF directly:
- score_and_rank operates on numeric scores, not ranks. Replacing similarity_score
  with a blended numeric is one mutation; switching the whole pipeline to a
  rank-based path is many.

Tradeoff accepted:
- BM25 can only rerank within what the vector store fetched. A pure-keyword
  match the vector search ranked at position N + 1 (just below the cut)
  cannot be rescued. For widemem's per-user corpus sizes (1K to 10K memories)
  and the typical fetch_k of 50 to 250 in balanced / deep mode, this captures
  the common case. If we ever scale past 100K memories per user, the
  candidate generation step itself will need to widen.
"""

from __future__ import annotations

from typing import List, Sequence

from widemem.core.types import MemorySearchResult
from widemem.retrieval.bm25 import BM25Retriever


def _min_max_normalize(values: Sequence[float]) -> List[float]:
    """Map a sequence of floats to [0, 1] via min-max. Returns 0.5 for every
    item if all values are equal (no signal in this batch)."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0:
        return [0.5] * len(values)
    return [(v - lo) / span for v in values]


def blend_hybrid_scores(
    results: List[MemorySearchResult],
    query: str,
    bm25_weight: float = 0.5,
) -> None:
    """Replace each result's similarity_score with a blend of (normalized
    vector similarity, normalized BM25 score) computed within the batch.

    Mutates ``results`` in place. No-op for empty input or empty query.
    Safe to call even when rank-bm25 is not installed: a clean ImportError
    propagates from BM25Retriever's constructor. Callers should gate this
    function behind ``MemoryConfig.enable_hybrid_search``.

    bm25_weight is the proportion of the blended score taken from the BM25
    side. 0.0 disables the BM25 contribution (vector-only behavior),
    1.0 disables the vector contribution (pure keyword retrieval). 0.5
    matches the common balanced hybrid baseline in published RAG work.
    """
    if not results or not query or not query.strip():
        return
    if not (0.0 <= bm25_weight <= 1.0):
        raise ValueError(
            f"bm25_weight must be in [0, 1], got {bm25_weight}"
        )

    vec_weight = 1.0 - bm25_weight

    # Build BM25 over the candidate pool.
    bm25 = BM25Retriever()
    bm25.index([(r.memory.id, r.memory.content) for r in results])

    # Fetch BM25 scores for every candidate id, defaulting to 0 for ids
    # that BM25 dropped (e.g. all-stopword content).
    bm25_ranked = bm25.search(query, top_k=len(results), min_score=0.0)
    bm25_score_by_id = {doc_id: score for doc_id, score in bm25_ranked}

    raw_bm25_per_result = [bm25_score_by_id.get(r.memory.id, 0.0) for r in results]
    raw_vec_per_result = [r.similarity_score for r in results]

    norm_bm25 = _min_max_normalize(raw_bm25_per_result)
    norm_vec = _min_max_normalize(raw_vec_per_result)

    for r, nv, nb in zip(results, norm_vec, norm_bm25):
        r.similarity_score = vec_weight * nv + bm25_weight * nb
