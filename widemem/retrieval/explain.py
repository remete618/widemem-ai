from __future__ import annotations

from typing import List

from widemem.core.types import (
    ExplainedMemory,
    MemorySearchResult,
    RetrievalConfidence,
    RetrievalExplanation,
)
from widemem.retrieval.abstention import detect_abstention

# Map the confidence enum to a representative numeric floor, used only when no
# result similarity is available (empty pool). When results exist, the top
# memory's raw similarity is the reported confidence.
_LEVEL_FLOOR = {
    RetrievalConfidence.HIGH: 0.8,
    RetrievalConfidence.MODERATE: 0.5,
    RetrievalConfidence.LOW: 0.2,
    RetrievalConfidence.NONE: 0.0,
}

# Categories where a wrong answer can harm health or safety. These always
# recommend review, even at high confidence (an allergy or medication deserves
# a second look every time). Other YMYL categories (financial/legal/tax/
# insurance) follow plain option (c): review only when confidence is not high.
_SAFETY_CRITICAL = {"medical", "safety", "pharmaceutical"}


def build_explanation(
    results: List[MemorySearchResult],
    confidence: RetrievalConfidence,
    answer: str | None = None,
) -> RetrievalExplanation:
    """Turn ranked results + a confidence verdict into a trust explanation.

    requires_review policy (option c): flag review when retrieval confidence is
    low/none OR a high-stakes (YMYL) memory is present without HIGH confidence.
    answerable: there is relevant memory, the generated answer (if provided) is
    not a refusal, and the result is safe to answer without review.

    On `confidence`: this number is RETRIEVAL STRENGTH (top cosine similarity),
    NOT a probability the answer is correct. An empirical LoCoMo sweep found
    top-similarity correlates only ~0.25 with answer correctness, so do not read
    a high confidence as "the answer is right". The reliable signals here are
    the extremes (no memory -> not answerable) and, when `answer` is supplied,
    refusal detection at the generation layer (the real answerability signal).

    Note: this does NOT do domain contradiction reasoning (e.g. penicillin/
    amoxicillin cross-allergy); that is the separate contradiction-ledger work.
    """
    is_high = confidence == RetrievalConfidence.HIGH
    has_relevant = confidence != RetrievalConfidence.NONE
    ymyl_cats = sorted({r.memory.ymyl_category for r in results if r.memory.ymyl_category})
    has_ymyl = bool(ymyl_cats)

    abstained = detect_abstention(answer) if answer is not None else False

    has_safety_critical = any(c in _SAFETY_CRITICAL for c in ymyl_cats)
    requires_review = (
        confidence in (RetrievalConfidence.LOW, RetrievalConfidence.NONE)
        or (has_ymyl and not is_high)
        or has_safety_critical
    )
    # answer-layer refusal is authoritative: if the model abstained, it is not
    # answerable regardless of retrieval confidence.
    answerable = has_relevant and not requires_review and not abstained

    confidence_num = (
        round(max(0.0, min(1.0, results[0].similarity_score)), 2)
        if results else _LEVEL_FLOOR[confidence]
    )

    reason = _reason(confidence, ymyl_cats, requires_review, answerable, abstained)

    memories = [
        ExplainedMemory(
            content=r.memory.content,
            final_score=round(r.final_score, 3),
            similarity=round(r.similarity_score, 3),
            importance=round(r.importance_score, 3),
            recency=round(r.temporal_score, 3),
            ymyl_category=r.memory.ymyl_category,
        )
        for r in results
    ]

    return RetrievalExplanation(
        answerable=answerable,
        confidence=confidence_num,
        confidence_level=confidence.value,
        requires_review=requires_review,
        reason=reason,
        memories=memories,
    )


def _reason(confidence, ymyl_cats, requires_review, answerable, abstained=False) -> str:
    if confidence == RetrievalConfidence.NONE:
        return "No relevant memory found; not answerable from stored memory."
    if abstained:
        return "The generated answer is a refusal; treat as not answerable from memory."
    if requires_review and ymyl_cats:
        cats = ", ".join(ymyl_cats)
        return (
            f"High-stakes ({cats}) memory present at {confidence.value} confidence. "
            "Recommend review or an explicit confirmation before using in an answer."
        )
    if requires_review:
        return (
            f"Low retrieval confidence ({confidence.value}); related memory exists but "
            "is not a reliable basis for a direct answer."
        )
    if answerable and confidence == RetrievalConfidence.HIGH:
        return "High-confidence match from stored memory; safe to answer."
    return "Moderate-confidence match from stored memory; answerable with normal caution."
