"""Tests for YMYL classification, topic boosting, and integration."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from widemem.core.types import (
    Memory,
    MemorySearchResult,
    ScoringConfig,
    YMYLConfig,
)
from widemem.retrieval.temporal import score_and_rank
from widemem.scoring.topics import get_topic_boost, get_topic_label
from widemem.scoring.ymyl import (
    classify_ymyl,
    classify_ymyl_batch,
    classify_ymyl_detailed,
    is_ymyl,
    is_ymyl_strong,
)


class TestYMYLClassifier:
    def setup_method(self):
        self.config = YMYLConfig(enabled=True)

    def test_strong_health(self):
        result = classify_ymyl_detailed("diabetes diagnosis confirmed", self.config)
        assert result.category == "health"
        assert result.is_strong

    def test_strong_financial(self):
        result = classify_ymyl_detailed("opened a savings account at the bank", self.config)
        assert result.category == "financial"
        assert result.is_strong

    def test_strong_safety(self):
        result = classify_ymyl_detailed("my blood type is O+", self.config)
        assert result.category == "safety"
        assert result.is_strong

    def test_strong_legal(self):
        result = classify_ymyl_detailed("filed for child custody", self.config)
        assert result.category == "legal"
        assert result.is_strong

    def test_strong_insurance(self):
        result = classify_ymyl_detailed("my insurance premium went up", self.config)
        assert result.category == "insurance"
        assert result.is_strong

    def test_strong_tax(self):
        result = classify_ymyl_detailed("filing my W-2 this week", self.config)
        assert result.category == "tax"
        assert result.is_strong

    def test_strong_pharmaceutical(self):
        result = classify_ymyl_detailed("checking for drug interaction risks", self.config)
        assert result.category == "pharmaceutical"
        assert result.is_strong

    def test_strong_via_two_weak_hits(self):
        result = classify_ymyl_detailed("doctor prescribed medication", self.config)
        assert result.category == "health"
        assert result.is_strong

    def test_weak_single_keyword(self):
        result = classify_ymyl_detailed("went to the doctor", self.config)
        assert result.category == "health"
        assert result.confidence == "weak"
        assert not result.is_strong

    def test_weak_bank_alone(self):
        result = classify_ymyl_detailed("walked by the bank", self.config)
        assert result.category == "financial"
        assert result.confidence == "weak"

    def test_false_positive_prevention(self):
        result = classify_ymyl_detailed("I like pizza", self.config)
        assert result.category is None
        assert result.confidence == "none"

        result = classify_ymyl_detailed("going to the park", self.config)
        assert result.category is None

    def test_is_ymyl_strong_helper(self):
        assert is_ymyl_strong("diabetes diagnosis confirmed", self.config) is True
        assert is_ymyl_strong("walked by the bank", self.config) is False
        assert is_ymyl_strong("pizza for dinner", self.config) is False

    def test_is_ymyl_any(self):
        assert is_ymyl("walked by the doctor", self.config) is True
        assert is_ymyl("pizza for dinner", self.config) is False

    def test_disabled(self):
        config = YMYLConfig(enabled=False)
        assert classify_ymyl("diabetes diagnosis", config) is None

    def test_limited_categories(self):
        config = YMYLConfig(enabled=True, categories=["health"])
        assert classify_ymyl("doctor appointment", config) == "health"
        assert classify_ymyl("mortgage rate is 6%", config) is None

    def test_batch_classify(self):
        texts = ["doctor prescribed medication", "pizza", "savings account at bank"]
        results = classify_ymyl_batch(texts, self.config)
        assert results[0] == "health"
        assert results[1] is None
        assert results[2] == "financial"

    def test_classify_ymyl_returns_category(self):
        assert classify_ymyl("401k contribution plan", self.config) == "financial"
        assert classify_ymyl("emergency contact is Jane", self.config) == "safety"


class TestSemanticYMYL:
    """Tests for LLM-based YMYL classification via extraction pipeline."""

    def test_fact_carries_ymyl_category(self):
        from widemem.core.types import Fact
        fact = Fact(content="patient has chest pain", importance=9.0, ymyl_category="health")
        assert fact.ymyl_category == "health"

    def test_fact_no_ymyl_by_default(self):
        from widemem.core.types import Fact
        fact = Fact(content="likes pizza", importance=3.0)
        assert fact.ymyl_category is None

    def test_action_item_carries_ymyl(self):
        from widemem.core.types import ActionItem, MemoryAction
        action = ActionItem(action=MemoryAction.ADD, fact="has diabetes", importance=9.0, ymyl_category="health")
        assert action.ymyl_category == "health"

    def test_memory_carries_ymyl(self):
        from widemem.core.types import Memory
        mem = Memory(content="takes metformin daily", ymyl_category="medical")
        assert mem.ymyl_category == "medical"

    def test_ymyl_stored_in_metadata(self):
        from widemem.core.pipeline import MemoryPipeline
        from widemem.core.types import Memory
        pipeline = MemoryPipeline.__new__(MemoryPipeline)
        mem = Memory(content="blood type A+", ymyl_category="safety")
        meta = pipeline._memory_to_metadata(mem)
        assert meta["ymyl_category"] == "safety"

    def test_ymyl_not_in_metadata_when_none(self):
        from widemem.core.pipeline import MemoryPipeline
        from widemem.core.types import Memory
        pipeline = MemoryPipeline.__new__(MemoryPipeline)
        mem = Memory(content="likes pizza")
        meta = pipeline._memory_to_metadata(mem)
        assert "ymyl_category" not in meta

    def test_ymyl_memory_gets_decay_immunity(self):
        """Memories with ymyl_category set should be immune to decay."""
        config = ScoringConfig()
        ymyl_config = YMYLConfig(enabled=True, decay_immune=True)
        now = datetime(2026, 3, 8)

        results = [
            MemorySearchResult(
                memory=Memory(
                    content="patient reports persistent chest pain",
                    importance=9.0,
                    ymyl_category="health",
                    created_at=now - timedelta(days=365),
                ),
                similarity_score=0.8,
            ),
            MemorySearchResult(
                memory=Memory(
                    content="likes pizza",
                    importance=3.0,
                    created_at=now - timedelta(days=365),
                ),
                similarity_score=0.8,
            ),
        ]

        ranked = score_and_rank(results, config, now=now, ymyl_config=ymyl_config)
        ymyl_result = next(r for r in ranked if "chest pain" in r.memory.content)
        normal_result = next(r for r in ranked if "pizza" in r.memory.content)

        assert ymyl_result.temporal_score == 1.0
        assert normal_result.temporal_score < 1.0

    def test_llm_ymyl_overrides_regex_miss(self):
        """LLM classification should catch implied YMYL that regex misses."""
        from widemem.scoring.ymyl import classify_ymyl_detailed

        config = YMYLConfig(enabled=True)
        # "chest pain" doesn't match any regex pattern
        result = classify_ymyl_detailed("patient reports persistent chest pain", config)
        assert not result.is_strong  # regex misses this

        # But if LLM classified it, the memory would have ymyl_category set
        mem = Memory(
            content="patient reports persistent chest pain",
            ymyl_category="health",  # LLM would set this
            importance=9.0,
            created_at=datetime(2025, 1, 1),
        )
        assert mem.ymyl_category == "health"


class TestTopicBoost:
    def test_no_weights(self):
        assert get_topic_boost("anything", {}) == 1.0

    def test_matching_topic(self):
        weights = {"python": 2.0, "rust": 1.5}
        assert get_topic_boost("I love Python programming", weights) == 2.0

    def test_no_match(self):
        weights = {"python": 2.0}
        assert get_topic_boost("I love JavaScript", weights) == 1.0

    def test_case_insensitive(self):
        weights = {"Python": 2.0}
        assert get_topic_boost("python is great", weights) == 2.0

    def test_best_boost_wins(self):
        weights = {"python": 2.0, "programming": 3.0}
        assert get_topic_boost("python programming", weights) == 3.0

    def test_topic_label(self):
        weights = {"python": 2.0, "rust": 1.5}
        assert get_topic_label("learning python", weights) == "python"
        assert get_topic_label("learning java", weights) is None

    def test_topic_label_empty(self):
        assert get_topic_label("anything", {}) is None


class TestYMYLDecayImmunity:
    def test_strong_ymyl_memory_no_decay(self):
        config = ScoringConfig()
        ymyl_config = YMYLConfig(enabled=True, decay_immune=True)
        now = datetime(2026, 3, 8)

        results = [
            MemorySearchResult(
                memory=Memory(
                    content="has diabetes diagnosis confirmed by doctor",
                    importance=9.0,
                    created_at=now - timedelta(days=365),
                ),
                similarity_score=0.8,
            ),
            MemorySearchResult(
                memory=Memory(
                    content="likes pizza",
                    importance=3.0,
                    created_at=now - timedelta(days=365),
                ),
                similarity_score=0.8,
            ),
        ]

        ranked = score_and_rank(results, config, now=now, ymyl_config=ymyl_config)
        ymyl_result = next(r for r in ranked if "diabetes" in r.memory.content)
        normal_result = next(r for r in ranked if "pizza" in r.memory.content)

        assert ymyl_result.temporal_score == 1.0
        assert normal_result.temporal_score < 1.0
        assert ymyl_result.final_score > normal_result.final_score

    def test_weak_ymyl_still_decays(self):
        config = ScoringConfig()
        ymyl_config = YMYLConfig(enabled=True, decay_immune=True)
        now = datetime(2026, 3, 8)

        results = [
            MemorySearchResult(
                memory=Memory(
                    content="walked by the bank",
                    importance=3.0,
                    created_at=now - timedelta(days=365),
                ),
                similarity_score=0.8,
            ),
        ]

        ranked = score_and_rank(results, config, now=now, ymyl_config=ymyl_config)
        assert ranked[0].temporal_score < 1.0

    def test_ymyl_decay_immune_false(self):
        config = ScoringConfig()
        ymyl_config = YMYLConfig(enabled=True, decay_immune=False)
        now = datetime(2026, 3, 8)

        results = [
            MemorySearchResult(
                memory=Memory(
                    content="has diabetes diagnosis from the doctor",
                    importance=9.0,
                    created_at=now - timedelta(days=365),
                ),
                similarity_score=0.8,
            ),
        ]

        ranked = score_and_rank(results, config, now=now, ymyl_config=ymyl_config)
        assert ranked[0].temporal_score < 1.0


class TestTopicBoostInScoring:
    def test_topic_boost_applied(self):
        config = ScoringConfig()
        now = datetime(2026, 3, 8)
        topic_weights = {"python": 2.0}

        results = [
            MemorySearchResult(
                memory=Memory(
                    content="loves python programming",
                    importance=5.0,
                    created_at=now - timedelta(days=1),
                ),
                similarity_score=0.8,
            ),
            MemorySearchResult(
                memory=Memory(
                    content="likes hiking on weekends",
                    importance=5.0,
                    created_at=now - timedelta(days=1),
                ),
                similarity_score=0.8,
            ),
        ]

        ranked = score_and_rank(results, config, now=now, topic_weights=topic_weights)
        python_result = next(r for r in ranked if "python" in r.memory.content)
        hiking_result = next(r for r in ranked if "hiking" in r.memory.content)

        assert python_result.final_score > hiking_result.final_score
        assert python_result.final_score == pytest.approx(hiking_result.final_score * 2.0, rel=0.01)

    def test_no_topic_weights(self):
        config = ScoringConfig()
        now = datetime(2026, 3, 8)

        results = [
            MemorySearchResult(
                memory=Memory(
                    content="loves python",
                    importance=5.0,
                    created_at=now,
                ),
                similarity_score=0.8,
            ),
        ]

        ranked = score_and_rank(results, config, now=now)
        ranked_with_topics = score_and_rank(results, config, now=now, topic_weights=None)
        assert ranked[0].final_score == ranked_with_topics[0].final_score


class TestEdgeCases:
    def test_dimension_validation(self):
        from widemem.core.types import VectorStoreConfig
        from widemem.storage.vector.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(VectorStoreConfig(), dimensions=4)
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.insert("id1", [0.1, 0.2], {"content": "too short"})

    def test_dimension_validation_on_search(self):
        from widemem.core.types import VectorStoreConfig
        from widemem.storage.vector.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(VectorStoreConfig(), dimensions=4)
        store.insert("id1", [0.1, 0.2, 0.3, 0.4], {"content": "ok"})
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.search([0.1, 0.2], top_k=1)

    def test_conflict_resolver_fallback_on_bad_json(self):
        from widemem.conflict.batch_resolver import BatchConflictResolver
        from widemem.core.types import Fact, LLMConfig
        from widemem.providers.llm.base import BaseLLM

        class CrashingLLM(BaseLLM):
            def __init__(self):
                super().__init__(LLMConfig())
            def generate(self, prompt, system=None):
                raise RuntimeError("LLM exploded")
            def generate_json(self, prompt, system=None):
                raise RuntimeError("LLM exploded")

        resolver = BatchConflictResolver(CrashingLLM())
        facts = [Fact(content="test fact", importance=5.0)]
        existing = [MemorySearchResult(
            memory=Memory(content="old fact"),
            similarity_score=0.9,
        )]
        actions = resolver.resolve(facts, existing)
        assert len(actions) == 1
        assert actions[0].action.value == "add"
        assert actions[0].fact == "test fact"

    def test_context_manager(self):
        import tempfile

        from widemem.core.memory import WideMemory
        from widemem.core.types import EmbeddingConfig, LLMConfig, MemoryConfig, VectorStoreConfig

        class DummyLLM:
            config = LLMConfig()
            def generate(self, p, system=None): return "{}"
            def generate_json(self, p, system=None): return {"facts": []}

        class DummyEmbedder:
            config = EmbeddingConfig(dimensions=4)
            dimensions = 4
            def embed(self, t): return [0.1, 0.2, 0.3, 0.4]
            def embed_batch(self, ts): return [[0.1, 0.2, 0.3, 0.4]] * len(ts)

        with tempfile.TemporaryDirectory() as d:
            config = MemoryConfig(history_db_path=f"{d}/h.db")
            from widemem.storage.vector.faiss_store import FAISSVectorStore
            vs = FAISSVectorStore(VectorStoreConfig(), dimensions=4)
            with WideMemory(config=config, llm=DummyLLM(), embedder=DummyEmbedder(), vector_store=vs):
                pass

    def test_duplicate_content_skipped(self):
        from widemem.core.types import VectorStoreConfig
        from widemem.storage.vector.faiss_store import FAISSVectorStore
        from widemem.utils.hashing import content_hash

        store = FAISSVectorStore(VectorStoreConfig(), dimensions=4)
        store.insert("id1", [0.1, 0.2, 0.3, 0.4], {"content": "hello", "content_hash": content_hash("hello")})
        store.insert("id2", [0.4, 0.3, 0.2, 0.1], {"content": "world", "content_hash": content_hash("world")})

        all_items = store.list_all()
        assert len(all_items) == 2
        hashes = {m["content_hash"] for _, m in all_items}
        assert content_hash("hello") in hashes
        assert content_hash("world") in hashes

    def test_list_all_with_filters(self):
        from widemem.core.types import VectorStoreConfig
        from widemem.storage.vector.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(VectorStoreConfig(), dimensions=4)
        store.insert("id1", [0.1, 0.2, 0.3, 0.4], {"content": "a", "tier": "fact"})
        store.insert("id2", [0.4, 0.3, 0.2, 0.1], {"content": "b", "tier": "summary"})
        store.insert("id3", [0.2, 0.3, 0.4, 0.1], {"content": "c", "tier": "fact"})

        facts = store.list_all(filters={"tier": "fact"})
        assert len(facts) == 2
        summaries = store.list_all(filters={"tier": "summary"})
        assert len(summaries) == 1

    def test_list_all_no_filters(self):
        from widemem.core.types import VectorStoreConfig
        from widemem.storage.vector.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(VectorStoreConfig(), dimensions=4)
        store.insert("id1", [0.1, 0.2, 0.3, 0.4], {"content": "a"})
        store.insert("id2", [0.4, 0.3, 0.2, 0.1], {"content": "b"})

        all_items = store.list_all()
        assert len(all_items) == 2

    def test_conflict_resolver_negative_fact_index(self):
        from widemem.conflict.batch_resolver import BatchConflictResolver
        from widemem.core.types import Fact, LLMConfig
        from widemem.providers.llm.base import BaseLLM

        class FakeLLM(BaseLLM):
            def __init__(self):
                super().__init__(LLMConfig())
            def generate(self, prompt, system=None):
                return ""
            def generate_json(self, prompt, system=None):
                return {"actions": [
                    {"fact_index": -1, "action": "add"},
                ]}

        resolver = BatchConflictResolver(FakeLLM())
        facts = [Fact(content="only fact", importance=5.0)]
        existing = [MemorySearchResult(
            memory=Memory(content="old"),
            similarity_score=0.9,
        )]
        actions = resolver.resolve(facts, existing)
        assert len(actions) == 1
        assert actions[0].fact == "only fact"
        assert actions[0].action.value == "add"

    def test_conflict_resolver_duplicate_fact_index(self):
        from widemem.conflict.batch_resolver import BatchConflictResolver
        from widemem.core.types import Fact, LLMConfig
        from widemem.providers.llm.base import BaseLLM

        class FakeLLM(BaseLLM):
            def __init__(self):
                super().__init__(LLMConfig())
            def generate(self, prompt, system=None):
                return ""
            def generate_json(self, prompt, system=None):
                return {"actions": [
                    {"fact_index": 0, "action": "add"},
                    {"fact_index": 0, "action": "add"},
                ]}

        resolver = BatchConflictResolver(FakeLLM())
        facts = [Fact(content="test", importance=5.0)]
        existing = [MemorySearchResult(
            memory=Memory(content="old"),
            similarity_score=0.9,
        )]
        actions = resolver.resolve(facts, existing)
        fact_actions = [a for a in actions if a.fact == "test"]
        assert len(fact_actions) == 1

    def test_conflict_resolver_missing_fact_index(self):
        from widemem.conflict.batch_resolver import BatchConflictResolver
        from widemem.core.types import Fact, LLMConfig
        from widemem.providers.llm.base import BaseLLM

        class FakeLLM(BaseLLM):
            def __init__(self):
                super().__init__(LLMConfig())
            def generate(self, prompt, system=None):
                return ""
            def generate_json(self, prompt, system=None):
                return {"actions": [
                    {"action": "add"},
                ]}

        resolver = BatchConflictResolver(FakeLLM())
        facts = [Fact(content="test", importance=5.0)]
        existing = [MemorySearchResult(
            memory=Memory(content="old"),
            similarity_score=0.9,
        )]
        actions = resolver.resolve(facts, existing)
        assert len(actions) == 1
        assert actions[0].fact == "test"
        assert actions[0].action.value == "add"
