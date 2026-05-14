"""BM25 keyword retriever used alongside vector retrieval for hybrid search.

Pure-text relevance scoring with Okapi BM25. Catches exact keyword matches
that semantic embedding similarity misses, especially for names, dates,
numeric identifiers, and other specific tokens.

Module is self-contained: it does not import any vector-store or scoring
internals. Integration into the search pipeline lives in a follow-up PR
behind a feature flag; this PR only ships the retriever and its tests.

Usage:
    from widemem.retrieval.bm25 import BM25Retriever

    docs = [(id_, content) for id_, content in user_memories]
    retriever = BM25Retriever()
    retriever.index(docs)
    top = retriever.search("penicillin allergy", top_k=5)
    # top is a list of (memory_id, bm25_score) sorted high to low

Optional dependency: install with `pip install widemem-ai[bm25]`.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

# Cheap, dependency-free tokenizer. BM25 quality is much more sensitive
# to the document corpus than to tokenizer sophistication; this matches
# what most BM25 retrieval baselines use.
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

# Conservative stopword list. Aggressive removal hurts retrieval on short
# memory facts where every token carries signal. This list catches only
# the highest-frequency function words.
_STOPWORDS = frozenset(
    {
        "a", "an", "the",
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did",
        "and", "or", "but", "of", "in", "on", "at", "to", "for", "from",
        "with", "by", "as", "that", "this", "it",
    }
)


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Lowercase + alnum tokens. Conservative stopword removal."""
    tokens = [t.lower() for t in _TOKEN_PATTERN.findall(text or "")]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


class BM25Retriever:
    """Wraps rank_bm25's BM25Okapi with a (id, content) document interface
    and a top-k search.

    Build the index once with ``index(docs)``. Subsequent ``search(query, k)``
    calls are O(corpus_size) per query because BM25 scores against every
    document; this is fine for widemem's typical per-user corpus sizes
    (1K to 10K facts) and unacceptable above that. If we ever need to
    scale past 100K facts per user, switch to a posting-list-backed BM25
    or split the corpus by tier.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """k1 and b are BM25's standard hyperparameters; the rank-bm25
        defaults match Robertson's original Okapi paper."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "BM25Retriever requires the rank-bm25 package. "
                "Install with: pip install widemem-ai[bm25]"
            ) from e
        self._bm25_cls = BM25Okapi
        self._k1 = k1
        self._b = b
        self._ids: List[str] = []
        self._bm25 = None

    def index(self, docs: Sequence[Tuple[str, str]]) -> None:
        """Build the index from (memory_id, content) pairs.

        Empty corpora are valid: ``search()`` on an empty index returns []
        rather than crashing.
        """
        self._ids = [doc_id for doc_id, _ in docs]
        tokenized = [tokenize(content) for _, content in docs]
        # rank-bm25 crashes on a fully empty corpus. Skip building.
        if not tokenized or not any(tokenized):
            self._bm25 = None
            return
        # rank-bm25's BM25Okapi takes k1, b as named kwargs but their
        # constructor signature has shifted across versions; pass via the
        # documented `k1` / `b` keys only if the version accepts them.
        try:
            self._bm25 = self._bm25_cls(tokenized, k1=self._k1, b=self._b)
        except TypeError:
            self._bm25 = self._bm25_cls(tokenized)

    def search(
        self, query: str, top_k: int = 10, min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Score every indexed document against the query, return the
        top-k as (id, score) sorted score-desc.

        Returns [] for empty queries, empty indices, or no positive scores.
        ``min_score`` filters out near-zero matches that BM25 still emits
        for documents with one stopword-adjacent overlap.
        """
        if self._bm25 is None or not self._ids or not query.strip():
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Pair with ids and rank
        ranked = sorted(
            ((self._ids[i], float(scores[i])) for i in range(len(self._ids))),
            key=lambda x: x[1],
            reverse=True,
        )
        # Drop trailing zeros / negatives
        ranked = [(i, s) for i, s in ranked if s > min_score]
        return ranked[:top_k]

    def __len__(self) -> int:
        return len(self._ids)

    def __bool__(self) -> bool:
        return self._bm25 is not None and len(self._ids) > 0


def reciprocal_rank_fusion(
    runs: Sequence[Sequence[Tuple[str, float]]],
    k: int = 60,
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Combine multiple ranked lists into one via Reciprocal Rank Fusion.

    Each run is a ``[(id, score), ...]`` list ordered by relevance. RRF
    sums ``1 / (k + rank)`` across runs, where ``rank`` is the 1-indexed
    position of the id in that run.

    Used downstream to merge BM25 results with vector-search results when
    hybrid mode is enabled. Lives here rather than in a separate utility
    module because BM25 has no other consumer yet.

    The standard RRF constant is ``k = 60`` (Cormack et al. 2009). It
    de-emphasizes deep-rank documents enough that one method's top
    candidates dominate even when both methods produce 25 candidates.
    """
    scores: dict[str, float] = {}
    for run in runs:
        for rank, (doc_id, _doc_score) in enumerate(run, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        fused = fused[:top_k]
    return fused
