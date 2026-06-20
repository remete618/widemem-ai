"""explain=True trust verdict: answerable / requires_review (option c) logic."""
from __future__ import annotations

from widemem.core.types import (
    Memory,
    MemorySearchResult,
    RetrievalConfidence,
)
from widemem.retrieval.explain import build_explanation


def _res(sim, ymyl=None):
    return MemorySearchResult(
        memory=Memory(content="x", ymyl_category=ymyl),
        similarity_score=sim,
        importance_score=0.7,
        temporal_score=1.0,
        final_score=sim,
    )


def test_none_confidence_not_answerable():
    e = build_explanation([], RetrievalConfidence.NONE)
    assert e.answerable is False
    assert e.requires_review is True
    assert "No relevant memory" in e.reason
    assert e.memories == []


def test_high_confidence_non_ymyl_answerable():
    e = build_explanation([_res(0.62)], RetrievalConfidence.HIGH)
    assert e.answerable is True
    assert e.requires_review is False
    assert e.confidence == 0.62
    assert "safe to answer" in e.reason


def test_low_confidence_requires_review():
    e = build_explanation([_res(0.15)], RetrievalConfidence.LOW)
    assert e.requires_review is True
    assert e.answerable is False


def test_ymyl_with_non_high_confidence_requires_review():
    # option c: YMYL present AND not HIGH -> review
    e = build_explanation([_res(0.30, ymyl="medical")], RetrievalConfidence.MODERATE)
    assert e.requires_review is True
    assert e.answerable is False
    assert "medical" in e.reason


def test_safety_critical_ymyl_always_requires_review_even_at_high():
    # c-strict: medical/safety/pharmaceutical always review, any confidence
    e = build_explanation([_res(0.70, ymyl="medical")], RetrievalConfidence.HIGH)
    assert e.requires_review is True
    assert e.answerable is False
    assert "medical" in e.reason


def test_non_safety_ymyl_high_confidence_no_review():
    # plain (c) still applies to financial/legal/tax/insurance
    e = build_explanation([_res(0.70, ymyl="financial")], RetrievalConfidence.HIGH)
    assert e.requires_review is False
    assert e.answerable is True


def test_score_provenance_surfaced():
    e = build_explanation([_res(0.5)], RetrievalConfidence.HIGH)
    m = e.memories[0]
    assert m.similarity == 0.5 and m.importance == 0.7 and m.recency == 1.0


# --- recalibration: text-embedding-3-small baselines are high, so an
# unrelated memory at ~0.49 cosine must NOT read as "high / safe to answer" ---

def test_assess_confidence_recalibrated_separates_relevant_from_noise():
    from widemem.retrieval.uncertainty import assess_confidence
    assert assess_confidence([_res(0.82)]) == RetrievalConfidence.HIGH   # real match
    assert assess_confidence([_res(0.63)]) == RetrievalConfidence.HIGH   # real match
    assert assess_confidence([_res(0.49)]) == RetrievalConfidence.LOW    # unrelated noise
    assert assess_confidence([_res(0.20)]) == RetrievalConfidence.NONE   # nothing


def test_irrelevant_top_match_is_not_answerable_end_to_end():
    # Reproduces the "Where does Alice work?" false positive: top match 0.49 to
    # an unrelated memory previously returned answerable=true "safe to answer".
    from widemem.retrieval.uncertainty import assess_confidence
    results = [_res(0.49), _res(0.45)]
    e = build_explanation(results, assess_confidence(results))
    assert e.answerable is False
    assert e.requires_review is True
