"""Tests for hierarchical memory: grouping, summarization, themes, and query routing."""

from __future__ import annotations

import tempfile
from typing import List

import numpy as np
import pytest

from widemem.core.types import (
    EmbeddingConfig,
    Fact,
    LLMConfig,
    Memory,
    MemoryConfig,
    MemorySearchResult,
    MemoryTier,
    VectorStoreConfig,
)
from widemem.extraction.base import BaseExtractor
from widemem.hierarchy.manager import HierarchyManager
from widemem.hierarchy.query_router import classify_query, route_results
from widemem.hierarchy.summarizer import MemorySummarizer
from widemem.providers.embeddings.base import BaseEmbedder
from widemem.providers.llm.base import BaseLLM
from widemem.storage.history import HistoryStore
from widemem.storage.vector.faiss_store import FAISSVectorStore


class MockLLM(BaseLLM):
    def __init__(self):
        super().__init__(LLMConfig())
        self.responses: list = []

    def set_responses(self, *responses):
        self.responses = list(responses)

    def generate(self, prompt, system=None):
        import json
        return json.dumps(self.generate_json(prompt, system))

    def generate_json(self, prompt, system=None):
        if self.responses:
            return self.responses.pop(0)
        return {}


class MockEmbedder(BaseEmbedder):
    def __init__(self, dimensions=64):
        super().__init__(EmbeddingConfig(dimensions=dimensions))

    def _embed(self, text):
        rng = np.random.RandomState(hash(text) % 2**31)
        vec = rng.randn(self.config.dimensions).astype(np.float32)
        return (vec / np.linalg.norm(vec)).tolist()

    def _embed_batch(self, texts):
        return [self._embed(t) for t in texts]


class DirectExtractor(BaseExtractor):
    def __init__(self):
        self.facts: List[Fact] = []

    def set_facts(self, *facts):
        self.facts = list(facts)

    def extract(self, text):
        if self.facts:
            return self.facts
        return [Fact(content=text, importance=5.0)]


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def embedder():
    return MockEmbedder(dimensions=64)


@pytest.fixture
def vector_store(tmp_dir):
    return FAISSVectorStore(VectorStoreConfig(path=f"{tmp_dir}/vectors"), dimensions=64)


@pytest.fixture
def history(tmp_dir):
    return HistoryStore(f"{tmp_dir}/history.db")


# --- Query Router Tests ---

class TestQueryRouter:
    def test_broad_query_routes_to_theme(self):
        assert classify_query("tell me about alice") == MemoryTier.THEME
        assert classify_query("who is alice") == MemoryTier.THEME
        assert classify_query("give me an overview of the project") == MemoryTier.THEME

    def test_specific_query_routes_to_fact(self):
        assert classify_query("where does alice live") == MemoryTier.FACT
        assert classify_query("when did she move") == MemoryTier.FACT
        assert classify_query("what is her job") == MemoryTier.FACT

    def test_short_query_routes_to_fact(self):
        assert classify_query("alice email") == MemoryTier.FACT
        assert classify_query("job title") == MemoryTier.FACT

    def test_medium_query_routes_to_summary(self):
        assert classify_query("alice work experience and skills in engineering") == MemoryTier.SUMMARY

    def test_route_results_preferred_tier(self):
        results = [
            MemorySearchResult(
                memory=Memory(content="fact1", tier=MemoryTier.FACT),
                final_score=0.9,
            ),
            MemorySearchResult(
                memory=Memory(content="summary1", tier=MemoryTier.SUMMARY),
                final_score=0.8,
            ),
            MemorySearchResult(
                memory=Memory(content="fact2", tier=MemoryTier.FACT),
                final_score=0.7,
            ),
            MemorySearchResult(
                memory=Memory(content="fact3", tier=MemoryTier.FACT),
                final_score=0.6,
            ),
        ]

        routed = route_results(results, MemoryTier.FACT)
        assert all(r.memory.tier == MemoryTier.FACT for r in routed)

    def test_route_results_fallback_when_insufficient(self):
        results = [
            MemorySearchResult(
                memory=Memory(content="theme1", tier=MemoryTier.THEME),
                final_score=0.9,
            ),
            MemorySearchResult(
                memory=Memory(content="fact1", tier=MemoryTier.FACT),
                final_score=0.8,
            ),
        ]

        routed = route_results(results, MemoryTier.THEME, min_results=3)
        assert len(routed) == 2  # only 2 total available
        assert routed[0].memory.tier == MemoryTier.THEME


# --- Summarizer Tests ---

class TestSummarizer:
    def test_group_facts(self, mock_llm):
        summarizer = MemorySummarizer(mock_llm)
        facts = [
            Memory(content="Lives in Berlin"),
            Memory(content="Works at Google"),
            Memory(content="Enjoys hiking"),
            Memory(content="Speaks German"),
        ]

        mock_llm.set_responses({
            "groups": [
                {"label": "Location & Language", "fact_indices": [0, 3]},
                {"label": "Work & Hobbies", "fact_indices": [1, 2]},
            ]
        })

        groups = summarizer.group_facts(facts)
        assert len(groups) == 2
        assert groups[0][0] == "Location & Language"
        assert len(groups[0][1]) == 2
        assert groups[1][0] == "Work & Hobbies"

    def test_group_facts_skips_small_groups(self, mock_llm):
        summarizer = MemorySummarizer(mock_llm)
        facts = [Memory(content="fact1"), Memory(content="fact2"), Memory(content="fact3")]

        mock_llm.set_responses({
            "groups": [
                {"label": "big group", "fact_indices": [0, 1]},
                {"label": "solo", "fact_indices": [2]},  # only 1 fact
            ]
        })

        groups = summarizer.group_facts(facts)
        assert len(groups) == 1

    def test_summarize_group(self, mock_llm):
        summarizer = MemorySummarizer(mock_llm)
        facts = [Memory(content="Lives in Berlin"), Memory(content="Speaks German")]

        mock_llm.set_responses({"summary": "Lives in Berlin and speaks German.", "importance": 8.0})

        text, importance = summarizer.summarize_group(facts)
        assert text == "Lives in Berlin and speaks German."
        assert importance == 8.0

    def test_synthesize_theme(self, mock_llm):
        summarizer = MemorySummarizer(mock_llm)
        summaries = [
            Memory(content="Lives in Berlin and speaks German."),
            Memory(content="Works at Google as a senior engineer."),
        ]

        mock_llm.set_responses({
            "theme": "A German-speaking senior engineer at Google based in Berlin.",
            "importance": 9.0,
        })

        text, importance = summarizer.synthesize_theme(summaries)
        assert "Berlin" in text
        assert importance == 9.0

    def test_group_facts_too_few(self, mock_llm):
        summarizer = MemorySummarizer(mock_llm)
        groups = summarizer.group_facts([Memory(content="only one")])
        assert groups == []


# --- Hierarchy Manager Tests ---

class TestHierarchyManager:
    def _insert_facts(self, vector_store, embedder, history, facts, user_id="alice"):
        from widemem.utils.hashing import content_hash
        memories = []
        for text in facts:
            mem = Memory(
                content=text,
                user_id=user_id,
                tier=MemoryTier.FACT,
                importance=7.0,
                content_hash=content_hash(text),
            )
            embedding = embedder.embed(text)
            vector_store.insert(
                id=mem.id,
                vector=embedding,
                metadata={
                    "content": mem.content,
                    "user_id": mem.user_id,
                    "tier": mem.tier.value,
                    "importance": mem.importance,
                    "content_hash": mem.content_hash,
                    "created_at": mem.created_at.isoformat(),
                    "updated_at": mem.updated_at.isoformat(),
                },
            )
            memories.append(mem)
        return memories

    def test_below_threshold_no_summarize(self, mock_llm, embedder, vector_store, history):
        manager = HierarchyManager(
            summarizer=MemorySummarizer(mock_llm),
            embedder=embedder,
            vector_store=vector_store,
            history=history,
            summarize_threshold=10,
        )

        self._insert_facts(vector_store, embedder, history, [
            "Likes coffee", "Lives in Berlin",
        ])

        result = manager.maybe_summarize(user_id="alice")
        assert result == []

    def test_force_summarize(self, mock_llm, embedder, vector_store, history):
        manager = HierarchyManager(
            summarizer=MemorySummarizer(mock_llm),
            embedder=embedder,
            vector_store=vector_store,
            history=history,
            summarize_threshold=100,
            theme_threshold=100,
        )

        self._insert_facts(vector_store, embedder, history, [
            "Lives in Berlin", "Speaks German",
            "Works at Google", "Senior engineer",
        ])

        mock_llm.set_responses(
            # group_facts
            {"groups": [{"label": "Location", "fact_indices": [0, 1]}, {"label": "Work", "fact_indices": [2, 3]}]},
            # summarize_group for Location
            {"summary": "Based in Berlin, speaks German.", "importance": 8.0},
            # summarize_group for Work
            {"summary": "Senior engineer at Google.", "importance": 8.0},
        )

        result = manager.maybe_summarize(user_id="alice", force=True)
        assert len(result) == 2
        assert all(m.tier == MemoryTier.SUMMARY for m in result)

    def test_theme_generated_when_enough_summaries(self, mock_llm, embedder, vector_store, history):
        manager = HierarchyManager(
            summarizer=MemorySummarizer(mock_llm),
            embedder=embedder,
            vector_store=vector_store,
            history=history,
            summarize_threshold=1,
            theme_threshold=3,
        )

        self._insert_facts(vector_store, embedder, history, [
            "Lives in Berlin", "Speaks German",
            "Works at Google", "Senior engineer",
            "Enjoys hiking", "Runs marathons",
        ])

        mock_llm.set_responses(
            # group_facts
            {"groups": [
                {"label": "Location", "fact_indices": [0, 1]},
                {"label": "Work", "fact_indices": [2, 3]},
                {"label": "Sports", "fact_indices": [4, 5]},
            ]},
            # 3 summarize_group calls
            {"summary": "Based in Berlin, German speaker.", "importance": 8.0},
            {"summary": "Senior engineer at Google.", "importance": 8.0},
            {"summary": "Active outdoors person, hiker and runner.", "importance": 6.0},
            # synthesize_theme
            {"theme": "A German-speaking senior Google engineer in Berlin who enjoys outdoor sports.", "importance": 9.0},
        )

        result = manager.maybe_summarize(user_id="alice", force=True)
        summaries = [m for m in result if m.tier == MemoryTier.SUMMARY]
        themes = [m for m in result if m.tier == MemoryTier.THEME]
        assert len(summaries) == 3
        assert len(themes) == 1
        assert "Berlin" in themes[0].content


# --- Integration Test ---

class TestHierarchyIntegration:
    def test_search_with_hierarchy_routing(self, tmp_dir):
        from widemem.core.memory import WideMemory

        mock_llm = MockLLM()
        emb = MockEmbedder(dimensions=64)
        vs = FAISSVectorStore(VectorStoreConfig(path=f"{tmp_dir}/vectors"), dimensions=64)

        config = MemoryConfig(
            history_db_path=f"{tmp_dir}/history.db",
            embedding=EmbeddingConfig(dimensions=64),
            enable_hierarchy=True,
        )

        mem = WideMemory(config=config, llm=mock_llm, embedder=emb, vector_store=vs)
        extractor = DirectExtractor()
        mem.pipeline.extractor = extractor

        # Add facts
        for fact in ["Lives in Berlin", "Works at Google", "Speaks German", "Enjoys hiking"]:
            extractor.set_facts(Fact(content=fact, importance=7.0))
            mem.add(fact, user_id="alice")

        # Summarize
        mock_llm.set_responses(
            {"groups": [{"label": "Profile", "fact_indices": [0, 1, 2, 3]}]},
            {"summary": "German-speaking Googler in Berlin who hikes.", "importance": 8.0},
        )
        summaries = mem.summarize(user_id="alice", force=True)
        assert len(summaries) >= 1

        # Search with specific query - should prefer facts
        results = mem.search("where does alice live", user_id="alice")
        assert len(results) >= 1

        # Search with broad query - should prefer themes/summaries if available
        results = mem.search("tell me about alice", user_id="alice")
        assert len(results) >= 1

    def test_search_tier_filter(self, tmp_dir):
        from widemem.core.memory import WideMemory

        mock_llm = MockLLM()
        emb = MockEmbedder(dimensions=64)
        vs = FAISSVectorStore(VectorStoreConfig(path=f"{tmp_dir}/vectors"), dimensions=64)

        config = MemoryConfig(
            history_db_path=f"{tmp_dir}/history.db",
            embedding=EmbeddingConfig(dimensions=64),
        )

        mem = WideMemory(config=config, llm=mock_llm, embedder=emb, vector_store=vs)
        extractor = DirectExtractor()
        mem.pipeline.extractor = extractor

        extractor.set_facts(Fact(content="Lives in Berlin", importance=7.0))
        mem.add("Lives in Berlin", user_id="alice")

        # Search for facts only
        results = mem.search("Berlin", user_id="alice", tier=MemoryTier.FACT)
        assert len(results) >= 1
        assert all(r.memory.tier == MemoryTier.FACT for r in results)

        # Search for summaries only - should be empty
        results = mem.search("Berlin", user_id="alice", tier=MemoryTier.SUMMARY)
        assert len(results) == 0
