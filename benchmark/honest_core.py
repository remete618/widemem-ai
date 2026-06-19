"""Pure, testable core for the honest LoCoMo evaluation.

Three credibility features that vendor benchmarks (e.g. mem0's published
LoCoMo runs) commonly omit, each implemented here as a deterministic,
network-free function so it can be unit-tested without an API key:

1. The judge model MUST differ from the answerer model. A system grading its
   own answers inflates scores; ``self_grading_check`` refuses that setup.
2. The adversarial category (5) is INCLUDED, not dropped. Adversarial LoCoMo
   questions are unanswerable; the correct behavior is to abstain, scored by
   ``is_correct_abstention`` / ``score_question``.
3. A full-context baseline is produced by ``build_full_context_prompt``. A
   memory layer that cannot beat stuffing the whole transcript into the prompt
   is not adding value, so a credible benchmark reports that ceiling.

See benchmark/HONEST_LOCOMO.md for the methodology and why it differs from the
vendor benchmarks.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "open-domain",
    4: "multi-hop",
    5: "adversarial",
}

ADVERSARIAL_CATEGORY = 5

# Phrases that count as a correct abstention on adversarial (unanswerable)
# questions. Kept lowercase; matched as substrings.
_ABSTENTION_MARKERS = (
    "not mentioned",
    "no information",
    "cannot be determined",
    "cannot determine",
    "can't be determined",
    "not enough information",
    "insufficient information",
    "don't know",
    "do not know",
    "unanswerable",
    "no relevant",
    "not available",
    "not provided",
    "not stated",
    "unknown",
)

FULL_CONTEXT_PROMPT = """You are answering a question using the FULL conversation transcript below,
with no retrieval or memory layer. This is the baseline ceiling.

Conversation:
{conversation}

Question: {question}

Answer in fewer than six words. If the conversation does not contain the answer, reply exactly "Not mentioned".
Answer:"""


def self_grading_check(answerer_model: str, judge_model: str) -> tuple[bool, str]:
    """Return ``(ok, message)``. ``ok`` is False when the judge equals the
    answerer (self-grading) or when no judge is set."""
    a = (answerer_model or "").strip().lower()
    j = (judge_model or "").strip().lower()
    if not j:
        return False, "No judge model set; refusing to self-grade. Pass a distinct --judge-model."
    if a == j:
        return False, (
            f"Judge model ({judge_model}) equals the answerer model. That is "
            "self-grading and not a neutral result; use a different judge model."
        )
    return True, f"Judge ({judge_model}) differs from answerer ({answerer_model})."


def is_adversarial(category: int) -> bool:
    return category == ADVERSARIAL_CATEGORY


def is_correct_abstention(predicted: str | None) -> bool:
    """For adversarial (unanswerable) questions, the correct behavior is to
    abstain. True when the answer signals the information is not present."""
    if not predicted:
        return False
    p = predicted.strip().lower()
    return any(marker in p for marker in _ABSTENTION_MARKERS)


def score_question(category: int, predicted: str | None, judge_correct: bool) -> float:
    """Resolve correctness to 1.0/0.0. Adversarial questions are correct only
    when the answer abstains; all other categories use the judge verdict."""
    if is_adversarial(category):
        return 1.0 if is_correct_abstention(predicted) else 0.0
    return 1.0 if judge_correct else 0.0


def stratified_sample(
    questions: list[dict[str, Any]],
    per_category: dict[int, int],
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministic stratified sample across categories, including adversarial.

    Each category is sorted by ``(sample_id, question)`` for a stable base
    order, then shuffled with a per-category seed and truncated to its count.
    """
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        cat = q.get("category")
        if cat in per_category:
            buckets[cat].append(q)

    selected: list[dict[str, Any]] = []
    for cat, count in per_category.items():
        pool = sorted(
            buckets.get(cat, []),
            key=lambda q: (str(q.get("sample_id", "")), str(q.get("question", ""))),
        )
        rng = random.Random(seed + cat)
        rng.shuffle(pool)
        selected.extend(pool[:count])
    return selected


def build_full_context_prompt(conversation_text: str, question: str) -> str:
    """Full-context baseline prompt: answer from the entire conversation, no
    retrieval. The ceiling a memory layer must beat to justify itself."""
    return FULL_CONTEXT_PROMPT.format(conversation=conversation_text, question=question)


def aggregate(
    predictions: list[dict[str, Any]],
    category_names: dict[int, str] = CATEGORY_NAMES,
) -> dict[str, Any]:
    """Per-category and overall J score (0-100). Each prediction is a dict with
    ``category`` (int) and ``score`` (0.0/1.0)."""
    by_cat: dict[int, list[float]] = defaultdict(list)
    for p in predictions:
        by_cat[p["category"]].append(p["score"])

    cat_j = {
        category_names.get(cat, str(cat)): (sum(v) / len(v) * 100 if v else 0.0)
        for cat, v in by_cat.items()
    }
    overall = (
        sum(p["score"] for p in predictions) / len(predictions) * 100
        if predictions
        else 0.0
    )
    return {
        "overall_j": round(overall, 2),
        "by_category": {k: round(v, 2) for k, v in cat_j.items()},
        "n": len(predictions),
    }
