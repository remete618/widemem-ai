"""Refusal detection: the real answer-layer answerability signal."""
from __future__ import annotations

import pytest

from widemem.retrieval.abstention import detect_abstention


@pytest.mark.parametrize("text", [
    "I don't have that information.",
    "I do not recall any details about that.",
    "There is no record of that in memory.",
    "I'm not sure about this.",
    "I cannot determine the answer from the memories.",
    "That wasn't mentioned.",
    "Not enough information to answer.",
    "I couldn't find anything about that.",
    "N/A",
    "unknown",
    "",
    "   ",
    None,
])
def test_abstentions_detected(text):
    assert detect_abstention(text) is True


@pytest.mark.parametrize("text", [
    "Sweden",
    "She works at Google as a software engineer.",
    "Mental health awareness.",
    "Caroline moved from Sweden in 2019.",
    "7 May 2023",
    "Her necklace symbolizes love, faith, and strength.",
])
def test_real_answers_not_flagged(text):
    assert detect_abstention(text) is False
