"""Tests for the pure, network-free parts of benchmark/honest_locomo.py.

Covers the model guard, question building (including the adversarial category
the regression gate drops), transcript flattening, prompt selection, and judge
parsing. No API key or network.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark"))

import honest_locomo as hl  # noqa: E402


def _data():
    return [{
        "sample_id": "s1",
        "conversation": {
            "speaker_a": "Alice", "speaker_b": "Bob",
            "session_1": [{"speaker": "Alice", "text": "I moved to Boston"}],
            "session_2": [{"speaker": "Bob", "text": "Nice"}],
            "session_1_date_time": "2026-05-01",  # not a session_N turn list
        },
        "qa": [
            {"question": "where", "answer": "Boston", "category": 1},
            {"question": "adv", "answer": "Not mentioned", "category": 5},
            {"question": "skip", "answer": "y", "category": 9},
        ],
    }]


def test_parse_args_default_judge_differs_from_answerer():
    args = hl.parse_args([])
    ok, _ = __import__("honest_core").self_grading_check(args.answerer_model, args.judge_model)
    assert ok is True


def test_validate_models_refuses_self_grading():
    with pytest.raises(SystemExit):
        hl.validate_models("gpt-4o-mini", "gpt-4o-mini")


def test_validate_models_allows_distinct():
    assert hl.validate_models("gpt-4o-mini", "gpt-4o") is True


def test_build_questions_includes_adversarial_and_skips_unknown():
    qs = hl.build_questions(_data(), include_adversarial=True)
    cats = sorted(q["category"] for q in qs)
    assert cats == [1, 5]  # category 9 skipped, adversarial kept
    assert qs[0]["_conv_idx"] == 0 and qs[0]["_sample_id"] == "s1"


def test_build_questions_can_exclude_adversarial():
    qs = hl.build_questions(_data(), include_adversarial=False)
    assert [q["category"] for q in qs] == [1]


def test_conversation_to_text_flattens_sessions_only():
    txt = hl.conversation_to_text(_data()[0])
    assert "Alice: I moved to Boston" in txt
    assert "Bob: Nice" in txt
    assert "2026-05-01" not in txt  # date_time key is not a turn list


def test_pick_answer_prompt_temporal_vs_default():
    import mini_locomo as ml
    assert hl.pick_answer_prompt(2) == ml.TEMPORAL_ANSWER_PROMPT
    assert hl.pick_answer_prompt(1) == ml.ANSWER_PROMPT


def test_parse_judge_label():
    assert hl.parse_judge_label('{"label": "CORRECT"}') is True
    assert hl.parse_judge_label('{"label": "WRONG"}') is False
    assert hl.parse_judge_label("The answer is CORRECT") is True
    assert hl.parse_judge_label("This is WRONG") is False
    assert hl.parse_judge_label(None) is None


def test_per_category_counts_respects_adversarial_and_limit():
    full = hl.per_category_counts(include_adversarial=True, limit=0)
    assert 5 in full
    no_adv = hl.per_category_counts(include_adversarial=False, limit=0)
    assert 5 not in no_adv
    capped = hl.per_category_counts(include_adversarial=True, limit=2)
    assert all(n <= 2 for n in capped.values())
