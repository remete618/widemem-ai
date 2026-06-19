"""Tests for the honest-LoCoMo evaluation core (benchmark/honest_core.py).

These cover the three credibility features that vendor benchmarks commonly
omit, with no network or API key:
1. judge model must differ from the answerer (no self-grading)
2. the adversarial category (5) is included and scored as abstention
3. a full-context baseline prompt is produced
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark"))

import honest_core as hc  # noqa: E402


def test_self_grading_check_flags_equal_models():
    ok, msg = hc.self_grading_check("gpt-4o-mini", "gpt-4o-mini")
    assert ok is False
    assert "self-grading" in msg.lower()


def test_self_grading_check_allows_distinct_models():
    ok, msg = hc.self_grading_check("gpt-4o-mini", "claude-sonnet-4-6")
    assert ok is True


def test_self_grading_check_requires_a_judge():
    ok, _ = hc.self_grading_check("gpt-4o-mini", "")
    assert ok is False


def test_adversarial_category_is_included():
    assert hc.ADVERSARIAL_CATEGORY == 5
    assert hc.CATEGORY_NAMES[5] == "adversarial"


def test_correct_abstention_detection():
    assert hc.is_correct_abstention("This is not mentioned in the conversation.")
    assert hc.is_correct_abstention("Cannot be determined from the memories")
    assert not hc.is_correct_abstention("She lives in Boston")
    assert not hc.is_correct_abstention("")


def test_score_question_adversarial_rewards_abstention():
    # adversarial: abstaining is correct, answering confidently is wrong
    assert hc.score_question(5, "Not mentioned", judge_correct=False) == 1.0
    assert hc.score_question(5, "She lives in Boston", judge_correct=True) == 0.0


def test_score_question_normal_uses_judge():
    assert hc.score_question(1, "Boston", judge_correct=True) == 1.0
    assert hc.score_question(1, "Boston", judge_correct=False) == 0.0


def test_stratified_sample_includes_adversarial_and_respects_counts():
    questions = []
    for cat in (1, 2, 3, 4, 5):
        for i in range(20):
            questions.append({"category": cat, "sample_id": f"s{cat}", "question": f"q{cat}-{i}"})
    per_cat = {1: 3, 2: 3, 3: 3, 4: 3, 5: 3}
    sample = hc.stratified_sample(questions, per_cat, seed=42)
    from collections import Counter
    counts = Counter(q["category"] for q in sample)
    assert counts == {1: 3, 2: 3, 3: 3, 4: 3, 5: 3}
    # adversarial questions are actually present
    assert any(q["category"] == 5 for q in sample)


def test_stratified_sample_is_deterministic():
    questions = [{"category": 1, "sample_id": "s", "question": f"q{i}"} for i in range(50)]
    a = hc.stratified_sample(questions, {1: 5}, seed=7)
    b = hc.stratified_sample(questions, {1: 5}, seed=7)
    assert [q["question"] for q in a] == [q["question"] for q in b]


def test_full_context_prompt_includes_conversation_and_question():
    p = hc.build_full_context_prompt("ALICE: I moved to Boston in May.", "Where does Alice live?")
    assert "Boston" in p
    assert "Where does Alice live?" in p


def test_aggregate_reports_per_category_and_overall():
    preds = [
        {"category": 1, "score": 1.0},
        {"category": 1, "score": 0.0},
        {"category": 5, "score": 1.0},
    ]
    out = hc.aggregate(preds)
    assert out["n"] == 3
    assert out["by_category"]["single-hop"] == 50.0
    assert out["by_category"]["adversarial"] == 100.0
    assert out["overall_j"] == round(2 / 3 * 100, 2)
