"""Unit tests for the zero-dependency entity extractor."""

from __future__ import annotations

import pytest

from widemem.extraction.entities import extract_entities


@pytest.mark.parametrize("text,expected_subset", [
    ("Caroline moved from Sweden to Berlin", {"caroline", "sweden", "berlin"}),
    ("Melanie went camping at Lake Tahoe", {"melanie", "lake tahoe"}),
    ('She said "peanut allergy" is serious', {"peanut allergy"}),
    ("He works at NASA and IBM", {"nasa", "ibm"}),
    ("Anne-Marie O'Brien called", {"anne-marie o'brien"}),
])
def test_extracts_expected_entities(text, expected_subset):
    got = set(extract_entities(text))
    assert expected_subset <= got, f"{expected_subset} not all in {got}"


@pytest.mark.parametrize("text", ["", None, "   ", "the cat sat on the mat", "12345 67 89"])
def test_no_entities_returns_empty(text):
    assert extract_entities(text) == []


def test_stopword_sentence_leaders_excluded():
    # "The" / "When" lead sentences but are not entities
    out = extract_entities("When did the meeting happen? The room was cold.")
    assert "when" not in out and "the" not in out


def test_deterministic_and_deduped():
    text = "Caroline likes Berlin. Caroline visited Berlin again."
    a = extract_entities(text)
    b = extract_entities(text)
    assert a == b
    assert a.count("caroline") == 1
    assert a.count("berlin") == 1


def test_capped_at_24():
    names = " ".join(f"Person{chr(65 + (i % 26))}name" for i in range(60))
    assert len(extract_entities(names)) <= 24


def test_never_raises_on_arbitrary_input():
    for junk in ["((((", "\x00\x01", "🙂🙂🙂", "A" * 5000, 12345]:
        # extract_entities is typed for str|None; pass odd values defensively
        try:
            extract_entities(junk if isinstance(junk, str) else str(junk))
        except Exception as e:  # pragma: no cover
            pytest.fail(f"raised on {junk!r}: {e}")
