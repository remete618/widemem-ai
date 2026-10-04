"""YMYL importance floors as YMYL.md documents them: the configured floor for a
strong regex match or an LLM tag in a configured category, nothing for a single
weak keyword."""
from __future__ import annotations

import pytest

from widemem.core.types import YMYLConfig
from widemem.extraction.llm_extractor import LLMExtractor


class FakeLLM:
    def __init__(self, payload):
        self._payload = payload
        self.config = type("C", (), {"model": "fake"})()

    def generate_json(self, prompt, system=None):
        return self._payload


def _extract_one(content, importance=5, ymyl_category=None, **ymyl):
    item = {"content": content, "importance": importance, "ymyl_category": ymyl_category}
    extractor = LLMExtractor(FakeLLM({"facts": [item]}), ymyl_config=YMYLConfig(enabled=True, **ymyl))
    (fact,) = extractor.extract("anything")
    return fact


def test_single_weak_keyword_gets_no_floor():
    fact = _extract_one("went to the doctor")
    assert (fact.importance, fact.ymyl_category) == (5.0, None)


def test_llm_tag_gets_the_configured_floor():
    fact = _extract_one("went to the doctor", ymyl_category="health")
    assert (fact.importance, fact.ymyl_category) == (8.0, "health")


def test_llm_tag_outside_configured_categories_is_ignored():
    fact = _extract_one("went to the doctor", ymyl_category="astrology")
    assert (fact.importance, fact.ymyl_category) == (5.0, None)


@pytest.mark.parametrize("floor", [8.0, 7.0])
def test_strong_match_gets_the_configured_floor(floor):
    fact = _extract_one("my bank account balance is low", min_importance=floor)
    assert fact.importance == floor and fact.ymyl_category == "financial"


def test_floor_never_lowers_importance():
    assert _extract_one("my bank account balance is low", importance=9.5).importance == 9.5


def test_disabled_ymyl_applies_nothing():
    item = {"content": "my bank account balance is low", "importance": 5, "ymyl_category": "financial"}
    (fact,) = LLMExtractor(FakeLLM({"facts": [item]})).extract("anything")
    assert (fact.importance, fact.ymyl_category) == (5.0, None)
