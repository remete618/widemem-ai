import pytest

from widemem.core.types import Memory, MemorySearchResult, RetrievalConfidence, UncertaintyMode
from widemem.retrieval.uncertainty import assess_confidence, build_frustration_response

MINILM = "all-MiniLM-L6-v2"
WITH_FACT = "I told you I moved to Boston!"
NO_FACT = "I told you this already!"


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch):
    for key in ("WIDEMEM_CONFIDENCE_HIGH", "WIDEMEM_CONFIDENCE_MODERATE", "WIDEMEM_CONFIDENCE_LOW"):
        monkeypatch.delenv(key, raising=False)


def _res(sim):
    return MemorySearchResult(memory=Memory(content="x"), similarity_score=sim, final_score=sim)


@pytest.mark.parametrize("mode", list(UncertaintyMode))
def test_high_confidence_reassures(mode):
    r = build_frustration_response(WITH_FACT, RetrievalConfidence.HIGH, mode)
    assert r["action"] == "reassure" and r["pin_fact"] is None


@pytest.mark.parametrize("mode", list(UncertaintyMode))
def test_moderate_with_extractable_fact_recovers_and_pins(mode):
    r = build_frustration_response(WITH_FACT, RetrievalConfidence.MODERATE, mode)
    assert r["action"] == "recover_and_pin"
    assert r["pin_fact"] == "I moved to Boston"


@pytest.mark.parametrize("mode", list(UncertaintyMode))
def test_moderate_without_extractable_fact_apologizes_and_asks(mode):
    r = build_frustration_response(NO_FACT, RetrievalConfidence.MODERATE, mode)
    assert r["action"] == "apologize_and_ask" and r["pin_fact"] is None


@pytest.mark.parametrize("level", [RetrievalConfidence.MODERATE, RetrievalConfidence.LOW,
                                   RetrievalConfidence.NONE])
def test_below_high_never_reassures(level):
    assert build_frustration_response(WITH_FACT, level, UncertaintyMode.HELPFUL)["action"] != "reassure"


def test_not_frustrated_returns_none():
    assert build_frustration_response("where do I live", RetrievalConfidence.MODERATE,
                                      UncertaintyMode.HELPFUL) is None


@pytest.mark.parametrize("sim,action", [
    # Measured with all-MiniLM-L6-v2: "I told you I moved to Boston!" against a
    # store holding only "live in San Francisco" scores 0.422 (MODERATE under
    # MiniLM thresholds); once "moved to Boston" is stored it scores 0.799.
    (0.422, "recover_and_pin"),
    (0.799, "reassure"),
])
def test_minilm_regression_never_stored_fact_is_not_reassured(sim, action):
    confidence = assess_confidence([_res(sim)], MINILM)
    assert build_frustration_response(WITH_FACT, confidence, UncertaintyMode.HELPFUL)["action"] == action
