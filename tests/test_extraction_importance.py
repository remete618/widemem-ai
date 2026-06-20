"""Importance-parse robustness: a single bad importance value must not drop
the whole turn's facts (regression for the float(None) ingest crash)."""
from __future__ import annotations

from widemem.extraction.llm_extractor import LLMExtractor


class FakeLLM:
    def __init__(self, payload):
        self._payload = payload
        self.config = type("C", (), {"model": "fake"})()

    def generate_json(self, prompt, system=None):
        return self._payload


def _extract(facts_payload):
    return LLMExtractor(FakeLLM({"facts": facts_payload})).extract("anything")


def test_null_importance_does_not_drop_turn():
    facts = _extract([
        {"content": "Caroline moved from Sweden", "importance": None},
        {"content": "Caroline likes pottery", "importance": 6},
    ])
    assert [f.content for f in facts] == ["Caroline moved from Sweden", "Caroline likes pottery"]
    assert facts[0].importance == 5.0  # safe default for null


def test_missing_importance_defaults():
    facts = _extract([{"content": "fact with no importance key"}])
    assert len(facts) == 1 and facts[0].importance == 5.0


def test_string_importance_coerced_or_defaulted():
    facts = _extract([
        {"content": "numeric string", "importance": "8"},
        {"content": "garbage string", "importance": "high"},
    ])
    assert facts[0].importance == 8.0
    assert facts[1].importance == 5.0


def test_out_of_range_importance_clamped():
    facts = _extract([
        {"content": "too high", "importance": 15},
        {"content": "negative", "importance": -3},
    ])
    assert facts[0].importance == 10.0
    assert facts[1].importance == 0.0


def test_one_bad_fact_does_not_lose_the_good_ones():
    facts = _extract([
        {"content": "good 1", "importance": 7},
        {"content": "bad", "importance": None},
        {"content": "good 2", "importance": 9},
    ])
    assert [f.content for f in facts] == ["good 1", "bad", "good 2"]
