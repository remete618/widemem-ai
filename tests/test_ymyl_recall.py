"""YMYL recall: genuinely high-stakes medical facts (allergies, prescriptions)
must classify as STRONG so the extractor force-tags them, instead of falling
through to the LLM (which missed 'allergic to penicillin')."""
from __future__ import annotations

from widemem.core.types import YMYLConfig
from widemem.scoring.ymyl import classify_ymyl_detailed, is_ymyl_strong

CFG = YMYLConfig(enabled=True)


def test_allergy_to_drug_is_strong_medical():
    r = classify_ymyl_detailed("Alice is allergic to penicillin.", CFG)
    assert r.is_strong is True
    assert r.category == "medical"


def test_allergy_noun_is_strong():
    assert is_ymyl_strong("I have a severe peanut allergy.", CFG)


def test_anaphylaxis_is_strong():
    assert is_ymyl_strong("He had an anaphylaxis episode last year.", CFG)


def test_doctor_prescribed_medication_stays_health_not_medical():
    # Guard the collision: the pre-existing health-tier two-weak-hit case must
    # not be captured by the new allergy patterns.
    r = classify_ymyl_detailed("doctor prescribed medication", CFG)
    assert r.category == "health"
    assert r.is_strong


def test_non_medical_stays_clean():
    r = classify_ymyl_detailed("Alice's favorite color is teal.", CFG)
    assert r.is_ymyl is False


def test_disabled_config_is_noop():
    assert classify_ymyl_detailed("allergic to penicillin", YMYLConfig(enabled=False)).is_ymyl is False
