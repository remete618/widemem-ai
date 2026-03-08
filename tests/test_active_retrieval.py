"""Tests for active retrieval: contradiction detection and clarification callbacks."""

from __future__ import annotations

import json
import tempfile
from typing import List

import numpy as np
import pytest

from widemem.core.memory import WideMemory
from widemem.core.pipeline import AddResult
from widemem.core.types import (
    EmbeddingConfig,
    Fact,
    LLMConfig,
    Memory,
    MemoryConfig,
    MemorySearchResult,
    VectorStoreConfig,
)
from widemem.extraction.base import BaseExtractor
from widemem.providers.embeddings.base import BaseEmbedder
from widemem.providers.llm.base import BaseLLM
from widemem.retrieval.active import ActiveRetrieval, Clarification
from widemem.storage.vector.faiss_store import FAISSVectorStore


class MockLLM(BaseLLM):
    def __init__(self):
        super().__init__(LLMConfig())
        self.responses: list = []

    def set_responses(self, *responses):
        self.responses = list(responses)

    def generate(self, prompt, system=None):
        return json.dumps(self.generate_json(prompt, system))

    def generate_json(self, prompt, system=None):
        if self.responses:
            return self.responses.pop(0)
        return {}


class MockEmbedder(BaseEmbedder):
    def __init__(self, dimensions=64):
        super().__init__(EmbeddingConfig(dimensions=dimensions))

    def embed(self, text):
        rng = np.random.RandomState(hash(text) % 2**31)
        vec = rng.randn(self.config.dimensions).astype(np.float32)
        return (vec / np.linalg.norm(vec)).tolist()

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class DirectExtractor(BaseExtractor):
    def __init__(self):
        self.facts: List[Fact] = []

    def set_facts(self, *facts):
        self.facts = list(facts)

    def extract(self, text):
        if self.facts:
            return list(self.facts)
        return [Fact(content=text, importance=5.0)]


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mock_llm():
    return MockLLM()


# --- ActiveRetrieval Unit Tests ---

class TestActiveRetrieval:
    def test_no_conflicts_when_no_existing(self, mock_llm):
        ar = ActiveRetrieval(mock_llm)
        facts = [Fact(content="Lives in Berlin")]
        result = ar.detect_conflicts(facts, [])
        assert result == []

    def test_no_conflicts_when_no_facts(self, mock_llm):
        ar = ActiveRetrieval(mock_llm)
        result = ar.detect_conflicts([], [
            MemorySearchResult(memory=Memory(content="old"), similarity_score=0.9)
        ])
        assert result == []

    def test_no_conflicts_below_similarity_threshold(self, mock_llm):
        ar = ActiveRetrieval(mock_llm, similarity_threshold=0.8)
        facts = [Fact(content="Lives in Berlin")]
        existing = [
            MemorySearchResult(memory=Memory(content="Likes coffee"), similarity_score=0.3)
        ]
        result = ar.detect_conflicts(facts, existing)
        assert result == []

    def test_detects_contradiction(self, mock_llm):
        ar = ActiveRetrieval(mock_llm, similarity_threshold=0.5)
        facts = [Fact(content="Lives in Berlin")]
        existing = [
            MemorySearchResult(
                memory=Memory(id="mem-1", content="Lives in Paris"),
                similarity_score=0.85,
            )
        ]

        mock_llm.set_responses({
            "has_conflict": True,
            "conflicts": [{
                "new_fact": "Lives in Berlin",
                "existing_memory_id": 1,
                "existing_content": "Lives in Paris",
                "type": "contradiction",
                "question": "Did you move from Paris to Berlin?",
            }]
        })

        result = ar.detect_conflicts(facts, existing)
        assert len(result) == 1
        assert result[0].conflict_type == "contradiction"
        assert result[0].question == "Did you move from Paris to Berlin?"
        assert result[0].existing_memory_id == "mem-1"

    def test_detects_ambiguity(self, mock_llm):
        ar = ActiveRetrieval(mock_llm, similarity_threshold=0.5)
        facts = [Fact(content="In Berlin this week")]
        existing = [
            MemorySearchResult(
                memory=Memory(id="mem-1", content="Lives in Paris"),
                similarity_score=0.7,
            )
        ]

        mock_llm.set_responses({
            "has_conflict": True,
            "conflicts": [{
                "new_fact": "In Berlin this week",
                "existing_memory_id": 1,
                "existing_content": "Lives in Paris",
                "type": "ambiguity",
                "question": "Are you visiting Berlin or did you move there?",
            }]
        })

        result = ar.detect_conflicts(facts, existing)
        assert len(result) == 1
        assert result[0].conflict_type == "ambiguity"

    def test_no_conflict_detected_by_llm(self, mock_llm):
        ar = ActiveRetrieval(mock_llm, similarity_threshold=0.5)
        facts = [Fact(content="Enjoys hiking")]
        existing = [
            MemorySearchResult(
                memory=Memory(content="Lives in Paris"),
                similarity_score=0.6,
            )
        ]

        mock_llm.set_responses({"has_conflict": False, "conflicts": []})

        result = ar.detect_conflicts(facts, existing)
        assert result == []

    def test_llm_error_gracefully_handled(self, mock_llm):
        ActiveRetrieval(mock_llm, similarity_threshold=0.5)

        # Make LLM raise
        class BrokenLLM(BaseLLM):
            def __init__(self):
                super().__init__(LLMConfig())
            def generate(self, prompt, system=None):
                raise RuntimeError("LLM down")
            def generate_json(self, prompt, system=None):
                raise RuntimeError("LLM down")

        ar_broken = ActiveRetrieval(BrokenLLM(), similarity_threshold=0.5)
        facts = [Fact(content="test")]
        existing = [MemorySearchResult(memory=Memory(content="old"), similarity_score=0.9)]

        result = ar_broken.detect_conflicts(facts, existing)
        assert result == []


# --- Clarification Model Tests ---

class TestClarificationModel:
    def test_clarification_fields(self):
        c = Clarification(
            new_fact="Lives in Berlin",
            existing_content="Lives in Paris",
            existing_memory_id="mem-1",
            conflict_type="contradiction",
            question="Did you move?",
        )
        assert c.new_fact == "Lives in Berlin"
        assert c.conflict_type == "contradiction"
        assert c.question == "Did you move?"


# --- AddResult Tests ---

class TestAddResult:
    def test_no_clarifications(self):
        r = AddResult(memories=[Memory(content="test")])
        assert not r.has_clarifications
        assert r.clarifications == []

    def test_with_clarifications(self):
        c = Clarification(
            new_fact="a", existing_content="b",
            conflict_type="contradiction", question="which?",
        )
        r = AddResult(memories=[], clarifications=[c])
        assert r.has_clarifications
        assert len(r.clarifications) == 1


# --- Integration Tests ---

class TestActiveRetrievalIntegration:
    def _make_memory(self, tmp_dir, mock_llm, enable_active=True):
        emb = MockEmbedder(dimensions=64)
        vs = FAISSVectorStore(VectorStoreConfig(path=f"{tmp_dir}/vectors"), dimensions=64)
        config = MemoryConfig(
            history_db_path=f"{tmp_dir}/history.db",
            embedding=EmbeddingConfig(dimensions=64),
            enable_active_retrieval=enable_active,
            active_retrieval_threshold=-1.0,  # Accept all similarities in tests
        )
        mem = WideMemory(config=config, llm=mock_llm, embedder=emb, vector_store=vs)
        ext = DirectExtractor()
        mem.pipeline.extractor = ext
        return mem, ext

    def test_add_with_no_conflict(self, tmp_dir, mock_llm):
        mem, ext = self._make_memory(tmp_dir, mock_llm)

        ext.set_facts(Fact(content="Lives in Berlin", importance=7.0))
        result = mem.add("Lives in Berlin", user_id="alice")
        assert len(result.memories) == 1
        assert not result.has_clarifications

    def test_add_with_conflict_and_callback(self, tmp_dir, mock_llm):
        mem, ext = self._make_memory(tmp_dir, mock_llm)

        # First add
        ext.set_facts(Fact(content="Lives in Paris", importance=7.0))
        mem.add("Lives in Paris", user_id="alice")

        # Second add triggers conflict
        ext.set_facts(Fact(content="Lives in Berlin", importance=7.0))

        mock_llm.set_responses(
            # conflict detection
            {
                "has_conflict": True,
                "conflicts": [{
                    "new_fact": "Lives in Berlin",
                    "existing_memory_id": 1,
                    "existing_content": "Lives in Paris",
                    "type": "contradiction",
                    "question": "Did you move from Paris to Berlin?",
                }]
            },
            # conflict resolution (batch resolver)
            {"actions": [{"fact_index": 0, "action": "add", "target_id": None, "importance": 7}]},
        )

        callback_called = []

        def on_clarification(clarifications):
            callback_called.extend(clarifications)
            return ["Yes, I moved"]

        result = mem.add("Lives in Berlin", user_id="alice", on_clarification=on_clarification)
        assert result.has_clarifications
        assert len(callback_called) == 1
        assert callback_called[0].question == "Did you move from Paris to Berlin?"
        # Memories were still added because callback returned responses (not None)
        assert len(result.memories) >= 1

    def test_callback_returning_none_aborts_add(self, tmp_dir, mock_llm):
        mem, ext = self._make_memory(tmp_dir, mock_llm)

        # First add
        ext.set_facts(Fact(content="Lives in Paris", importance=7.0))
        mem.add("Lives in Paris", user_id="alice")

        # Second add with conflict
        ext.set_facts(Fact(content="Lives in Berlin", importance=7.0))

        mock_llm.set_responses({
            "has_conflict": True,
            "conflicts": [{
                "new_fact": "Lives in Berlin",
                "existing_memory_id": 1,
                "existing_content": "Lives in Paris",
                "type": "contradiction",
                "question": "Did you move?",
            }]
        })

        def abort_callback(clarifications):
            return None  # abort

        result = mem.add("Lives in Berlin", user_id="alice", on_clarification=abort_callback)
        assert result.has_clarifications
        assert len(result.memories) == 0  # nothing was added

    def test_active_retrieval_disabled(self, tmp_dir, mock_llm):
        mem, ext = self._make_memory(tmp_dir, mock_llm, enable_active=False)

        ext.set_facts(Fact(content="Lives in Paris", importance=7.0))
        mem.add("Lives in Paris", user_id="alice")

        ext.set_facts(Fact(content="Lives in Berlin", importance=7.0))
        result = mem.add("Lives in Berlin", user_id="alice")
        # No conflict detection when disabled
        assert not result.has_clarifications
        assert len(result.memories) >= 1

    def test_add_without_callback_still_returns_clarifications(self, tmp_dir, mock_llm):
        mem, ext = self._make_memory(tmp_dir, mock_llm)

        ext.set_facts(Fact(content="Lives in Paris", importance=7.0))
        mem.add("Lives in Paris", user_id="alice")

        ext.set_facts(Fact(content="Lives in Berlin", importance=7.0))

        mock_llm.set_responses(
            {
                "has_conflict": True,
                "conflicts": [{
                    "new_fact": "Lives in Berlin",
                    "existing_memory_id": 1,
                    "existing_content": "Lives in Paris",
                    "type": "contradiction",
                    "question": "Did you move?",
                }]
            },
            {"actions": [{"fact_index": 0, "action": "add", "target_id": None, "importance": 7}]},
        )

        # No callback — should still proceed and report clarifications
        result = mem.add("Lives in Berlin", user_id="alice")
        assert result.has_clarifications
        assert len(result.memories) >= 1
