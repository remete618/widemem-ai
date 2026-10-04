import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from widemem.core.types import Memory, MemorySearchResult, RetrievalConfidence
from widemem.retrieval.uncertainty import assess_confidence, get_confidence_thresholds

FIXTURE = Path(__file__).parent / "fixtures" / "confidence_minilm.json"
OPENAI = {"high": 0.60, "moderate": 0.50, "low": 0.30}
MINILM = {"high": 0.60, "moderate": 0.30, "low": 0.20}
ENV_KEYS = ("WIDEMEM_CONFIDENCE_HIGH", "WIDEMEM_CONFIDENCE_MODERATE", "WIDEMEM_CONFIDENCE_LOW")


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _res(sim: float) -> MemorySearchResult:
    return MemorySearchResult(
        memory=Memory(content="x"),
        similarity_score=sim,
        raw_similarity_score=sim,
        final_score=sim,
    )


@pytest.mark.parametrize("model", [None, "text-embedding-3-small", "text-embedding-3-large",
                                   "nomic-embed-text", "some-unknown-model", "", 123])
def test_unknown_or_missing_model_keeps_openai_calibration(model):
    assert get_confidence_thresholds(model) == OPENAI


@pytest.mark.parametrize("model", ["all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2",
                                   "ALL-MINILM-L6-V2", " all-MiniLM-L6-v2 "])
def test_minilm_selects_its_own_thresholds(model):
    assert get_confidence_thresholds(model) == MINILM


def test_similar_but_different_model_name_does_not_match():
    assert get_confidence_thresholds("all-MiniLM-L12-v2") == OPENAI
    assert get_confidence_thresholds("paraphrase-all-MiniLM-L6-v2") == OPENAI


def test_env_override_wins_over_per_model_default(monkeypatch):
    monkeypatch.setenv("WIDEMEM_CONFIDENCE_MODERATE", "0.45")
    t = get_confidence_thresholds("all-MiniLM-L6-v2")
    assert t == {"high": 0.60, "moderate": 0.45, "low": 0.20}
    assert assess_confidence([_res(0.40)], "all-MiniLM-L6-v2") == RetrievalConfidence.LOW


def test_returned_thresholds_are_copies():
    get_confidence_thresholds("all-MiniLM-L6-v2")["moderate"] = 0.99
    assert get_confidence_thresholds("all-MiniLM-L6-v2")["moderate"] == 0.30


@pytest.mark.parametrize("sim,model,level", [
    (0.31, "all-MiniLM-L6-v2", RetrievalConfidence.MODERATE),
    (0.31, None, RetrievalConfidence.LOW),
    (0.30, "all-MiniLM-L6-v2", RetrievalConfidence.MODERATE),
    (0.2999, "all-MiniLM-L6-v2", RetrievalConfidence.LOW),
    (0.20, "all-MiniLM-L6-v2", RetrievalConfidence.LOW),
    (0.1999, "all-MiniLM-L6-v2", RetrievalConfidence.NONE),
    (0.60, "all-MiniLM-L6-v2", RetrievalConfidence.HIGH),
    (0.08, "all-MiniLM-L6-v2", RetrievalConfidence.NONE),
])
def test_boundaries(sim, model, level):
    assert assess_confidence([_res(sim)], model) == level


def test_memory_search_passes_the_embedder_model():
    from widemem.core.memory import WideMemory

    mem = WideMemory.__new__(WideMemory)
    mem.embedder = SimpleNamespace(config=SimpleNamespace(model="all-MiniLM-L6-v2"))
    mem._search_ranked = lambda **_: [_res(0.31)]
    assert mem.search("where does alice live").confidence == RetrievalConfidence.MODERATE

    mem.embedder.config.model = "text-embedding-3-small"
    assert mem.search("where does alice live").confidence == RetrievalConfidence.LOW


def test_memory_search_tolerates_embedder_without_config():
    from widemem.core.memory import WideMemory

    mem = WideMemory.__new__(WideMemory)
    mem.embedder = object()
    mem._search_ranked = lambda **_: [_res(0.55)]
    assert mem.search("q").confidence == RetrievalConfidence.MODERATE


# --- regression over the labeled fixture (facts stored by a real add() run) ---

def _load():
    return json.loads(FIXTURE.read_text())


def _precision_recall(cases, model):
    """'Answerable' = confidence MODERATE or HIGH, the line explain=True uses."""
    predicted = [
        assess_confidence([_res(c["top_similarity"])], model)
        in (RetrievalConfidence.HIGH, RetrievalConfidence.MODERATE)
        for c in cases
    ]
    actual = [bool(c["answer_facts"]) for c in cases]
    tp = sum(p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    return tp / (tp + fp), tp / sum(actual)


def _split(name, kinds=("answerable", "hard_negative", "unrelated")):
    return [c for c in _load()["cases"] if c["split"] == name and c["kind"] in kinds]


def test_fixture_shape():
    data = _load()
    assert data["meta"]["embedding_model"] == "all-MiniLM-L6-v2"
    kinds = {c["kind"] for c in data["cases"]}
    assert kinds == {"answerable", "hard_negative", "unrelated"}
    for c in data["cases"]:
        facts = data["users"][c["user"]]["facts"]
        assert all(0 <= i < len(facts) for i in c["answer_facts"])
        assert bool(c["answer_facts"]) == (c["kind"] == "answerable")
    assert {c["split"] for c in data["cases"]} == {"train", "holdout"}


@pytest.mark.parametrize("split", ["train", "holdout"])
def test_minilm_recall_floor(split):
    _, recall = _precision_recall(_split(split), "all-MiniLM-L6-v2")
    assert recall == 1.0


@pytest.mark.parametrize("split", ["train", "holdout"])
def test_minilm_beats_old_thresholds_on_recall(split):
    _, old = _precision_recall(_split(split), None)
    _, new = _precision_recall(_split(split), "all-MiniLM-L6-v2")
    assert old < 0.95 <= new


@pytest.mark.parametrize("split,floor", [("train", 0.90), ("holdout", 1.0)])
def test_minilm_keeps_unrelated_queries_out(split, floor):
    precision, _ = _precision_recall(_split(split, ("answerable", "unrelated")), "all-MiniLM-L6-v2")
    assert precision >= floor


@pytest.mark.parametrize("split", ["train", "holdout"])
def test_unrelated_queries_never_reach_high(split):
    for c in _split(split, ("unrelated",)):
        assert assess_confidence([_res(c["top_similarity"])], "all-MiniLM-L6-v2") != RetrievalConfidence.HIGH


def test_fixture_scores_match_the_live_model():
    st = pytest.importorskip("sentence_transformers")
    try:
        model = st.SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:  # no network and no cached weights
        pytest.skip(f"all-MiniLM-L6-v2 unavailable: {exc}")
    data = _load()
    for c in data["cases"]:
        facts = data["users"][c["user"]]["facts"]
        f = model.encode(facts, normalize_embeddings=True)
        q = model.encode([c["query"]], normalize_embeddings=True)[0]
        assert abs(float((f @ q).max()) - c["top_similarity"]) < 0.01, c["query"]


@pytest.mark.parametrize("split", ["train", "holdout"])
def test_minilm_precision_floor_with_hard_negatives(split):
    # Same-person questions with no stored answer score like real answers under
    # MiniLM (the name in the query dominates), so precision sits near the base
    # rate. This floor records that limit; thresholds cannot fix it.
    precision, _ = _precision_recall(_split(split), "all-MiniLM-L6-v2")
    assert precision >= 0.70
