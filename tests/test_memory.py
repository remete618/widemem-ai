"""Integration tests for WideMemory using mock LLM and embedder."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone

import numpy as np
import pytest

from widemem.conflict.batch_resolver import BatchConflictResolver
from widemem.core._time import as_utc
from widemem.core.memory import WideMemory
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
from widemem.storage.vector.faiss_store import FAISSVectorStore


class MockLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig())
        self.responses: list[dict] = []

    def set_responses(self, *responses: dict) -> None:
        self.responses = list(responses)

    def _generate(self, prompt: str, system: str | None = None) -> str:
        return json.dumps(self._generate_json(prompt, system))

    def _generate_json(self, prompt: str, system: str | None = None) -> dict:
        if self.responses:
            return self.responses.pop(0)
        return {"facts": []}


class MockEmbedder(BaseEmbedder):
    def __init__(self, dimensions: int = 64) -> None:
        super().__init__(EmbeddingConfig(dimensions=dimensions), max_retries=1, retry_delay=0)
        self._vectors: dict[str, list[float]] = {}

    def _embed(self, text: str) -> list[float]:
        if text not in self._vectors:
            rng = np.random.RandomState(hash(text) % 2**31)
            vec = rng.randn(self.config.dimensions).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            self._vectors[text] = vec.tolist()
        return self._vectors[text]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


class MockExtractor(BaseExtractor):
    def __init__(self) -> None:
        self.facts_to_return: list[Fact] = []

    def extract(self, text: str) -> list[Fact]:
        if self.facts_to_return:
            return self.facts_to_return
        return [Fact(content=text, importance=7.0)]


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_embedder():
    return MockEmbedder(dimensions=64)


@pytest.fixture
def vector_store(tmp_dir):
    config = VectorStoreConfig(path=f"{tmp_dir}/vectors")
    return FAISSVectorStore(config, dimensions=64)


@pytest.fixture
def memory(tmp_dir, mock_llm, mock_embedder, vector_store):
    config = MemoryConfig(history_db_path=f"{tmp_dir}/history.db")
    mem = WideMemory(config=config, llm=mock_llm, embedder=mock_embedder, vector_store=vector_store)
    return mem


class TestFAISSVectorStore:
    def test_insert_and_search(self, vector_store, mock_embedder):
        vec = mock_embedder.embed("hello world")
        vector_store.insert("id1", vec, {"content": "hello world", "user_id": "alice"})

        results = vector_store.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "id1"
        assert results[0][2]["content"] == "hello world"

    def test_search_with_filters(self, vector_store, mock_embedder):
        vec1 = mock_embedder.embed("fact for alice")
        vec2 = mock_embedder.embed("fact for bob")
        vector_store.insert("id1", vec1, {"content": "fact for alice", "user_id": "alice"})
        vector_store.insert("id2", vec2, {"content": "fact for bob", "user_id": "bob"})

        results = vector_store.search(vec1, top_k=10, filters={"user_id": "alice"})
        assert all(r[2]["user_id"] == "alice" for r in results)

    def test_delete(self, vector_store, mock_embedder):
        vec = mock_embedder.embed("to delete")
        vector_store.insert("id1", vec, {"content": "to delete"})
        vector_store.delete("id1")
        result = vector_store.get("id1")
        assert result is None

    def test_update(self, vector_store, mock_embedder):
        vec1 = mock_embedder.embed("original")
        vector_store.insert("id1", vec1, {"content": "original"})

        vec2 = mock_embedder.embed("updated")
        vector_store.update("id1", vec2, {"content": "updated"})

        result = vector_store.get("id1")
        assert result is not None
        assert result[1]["content"] == "updated"

    def test_empty_search(self, vector_store, mock_embedder):
        vec = mock_embedder.embed("anything")
        results = vector_store.search(vec)
        assert results == []


class TestFAISSPersistence:
    def test_save_and_reload(self, tmp_dir):
        config = VectorStoreConfig(path=f"{tmp_dir}/persist_test")
        store = FAISSVectorStore(config, dimensions=4)

        store.insert("id1", [0.1, 0.2, 0.3, 0.4], {"content": "hello"})
        store.insert("id2", [0.4, 0.3, 0.2, 0.1], {"content": "world"})

        store2 = FAISSVectorStore(config, dimensions=4)
        assert store2._index.ntotal == 2

        results = store2.search([0.1, 0.2, 0.3, 0.4], top_k=2)
        assert len(results) == 2
        contents = {r[2]["content"] for r in results}
        assert contents == {"hello", "world"}

    def test_get_after_reload(self, tmp_dir):
        config = VectorStoreConfig(path=f"{tmp_dir}/persist_get")
        store = FAISSVectorStore(config, dimensions=4)
        store.insert("id1", [0.1, 0.2, 0.3, 0.4], {"content": "hello", "user_id": "alice"})

        store2 = FAISSVectorStore(config, dimensions=4)
        result = store2.get("id1")
        assert result is not None
        assert result[1]["content"] == "hello"
        assert result[1]["user_id"] == "alice"

    def test_insert_after_reload(self, tmp_dir):
        config = VectorStoreConfig(path=f"{tmp_dir}/persist_insert")
        store = FAISSVectorStore(config, dimensions=4)
        store.insert("id1", [0.1, 0.2, 0.3, 0.4], {"content": "first"})

        store2 = FAISSVectorStore(config, dimensions=4)
        store2.insert("id2", [0.4, 0.3, 0.2, 0.1], {"content": "second"})
        assert store2._index.ntotal == 2

        store3 = FAISSVectorStore(config, dimensions=4)
        assert store3._index.ntotal == 2

    def test_delete_after_reload(self, tmp_dir):
        config = VectorStoreConfig(path=f"{tmp_dir}/persist_delete")
        store = FAISSVectorStore(config, dimensions=4)
        store.insert("id1", [0.1, 0.2, 0.3, 0.4], {"content": "hello"})
        store.insert("id2", [0.4, 0.3, 0.2, 0.1], {"content": "world"})

        store2 = FAISSVectorStore(config, dimensions=4)
        store2.delete("id1")
        assert store2.get("id1") is None
        assert store2._index.ntotal == 1

        store3 = FAISSVectorStore(config, dimensions=4)
        assert store3.get("id1") is None
        assert store3._index.ntotal == 1

    def test_update_after_reload(self, tmp_dir):
        config = VectorStoreConfig(path=f"{tmp_dir}/persist_update")
        store = FAISSVectorStore(config, dimensions=4)
        store.insert("id1", [0.1, 0.2, 0.3, 0.4], {"content": "v1"})

        store2 = FAISSVectorStore(config, dimensions=4)
        store2.update("id1", [0.5, 0.5, 0.0, 0.0], {"content": "v2"})

        store3 = FAISSVectorStore(config, dimensions=4)
        result = store3.get("id1")
        assert result is not None
        assert result[1]["content"] == "v2"
        assert store3._index.ntotal == 1

    def test_filters_after_reload(self, tmp_dir):
        config = VectorStoreConfig(path=f"{tmp_dir}/persist_filters")
        store = FAISSVectorStore(config, dimensions=4)
        store.insert("id1", [0.1, 0.2, 0.3, 0.4], {"content": "a", "user_id": "alice"})
        store.insert("id2", [0.4, 0.3, 0.2, 0.1], {"content": "b", "user_id": "bob"})

        store2 = FAISSVectorStore(config, dimensions=4)
        results = store2.search([0.1, 0.2, 0.3, 0.4], top_k=10, filters={"user_id": "alice"})
        assert len(results) == 1
        assert results[0][2]["user_id"] == "alice"

    def test_no_path_means_no_persistence(self, tmp_dir):
        config = VectorStoreConfig(path=None)
        store = FAISSVectorStore(config, dimensions=4)
        store.insert("id1", [0.1, 0.2, 0.3, 0.4], {"content": "ephemeral"})
        assert store._storage_path is None


class TestFAISSThreadSafety:
    def test_concurrent_inserts(self, tmp_dir):
        import threading

        config = VectorStoreConfig(path=f"{tmp_dir}/thread_test")
        store = FAISSVectorStore(config, dimensions=4)
        errors = []

        def insert_batch(start: int) -> None:
            try:
                for i in range(start, start + 20):
                    store.insert(f"id-{i}", [0.1 * (i % 10), 0.2, 0.3, 0.4], {"content": f"item {i}"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=insert_batch, args=(i * 20,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent inserts raised: {errors}"
        assert store._index.ntotal == 100

    def test_concurrent_reads_during_write(self, tmp_dir):
        import threading

        config = VectorStoreConfig(path=f"{tmp_dir}/rw_test")
        store = FAISSVectorStore(config, dimensions=4)
        for i in range(10):
            store.insert(f"id-{i}", [0.1 * i, 0.2, 0.3, 0.4], {"content": f"item {i}"})

        errors = []
        stop = threading.Event()

        def reader() -> None:
            try:
                while not stop.is_set():
                    store.search([0.1, 0.2, 0.3, 0.4], top_k=5)
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(10, 30):
                    store.insert(f"id-{i}", [0.1, 0.2, 0.3, 0.4], {"content": f"item {i}"})
            except Exception as e:
                errors.append(e)

        readers = [threading.Thread(target=reader) for _ in range(3)]
        for t in readers:
            t.start()
        writer_t = threading.Thread(target=writer)
        writer_t.start()
        writer_t.join()
        stop.set()
        for t in readers:
            t.join()

        assert not errors, f"Concurrent read/write raised: {errors}"
        assert store._index.ntotal == 30


class TestBatchConflictResolver:
    def test_all_add_when_no_existing(self, mock_llm):
        resolver = BatchConflictResolver(mock_llm)
        facts = [Fact(content="Lives in Berlin", importance=8.0)]
        actions = resolver.resolve(facts, [])

        assert len(actions) == 1
        assert actions[0].action.value == "add"
        assert actions[0].fact == "Lives in Berlin"

    def test_update_existing(self, mock_llm):
        resolver = BatchConflictResolver(mock_llm)
        facts = [Fact(content="Lives in Berlin", importance=8.0)]

        from widemem.core.types import Memory
        existing = [MemorySearchResult(
            memory=Memory(id="mem-1", content="Lives in Paris"),
            similarity_score=0.85,
        )]

        mock_llm.set_responses({
            "actions": [{"fact_index": 0, "action": "update", "target_id": 1, "importance": 8}]
        })

        actions = resolver.resolve(facts, existing)
        assert len(actions) == 1
        assert actions[0].action.value == "update"
        assert actions[0].target_id == "mem-1"

    def test_llm_failure_falls_back_to_add_with_dedup(self):
        from widemem.core.types import Memory

        class FailingLLM(BaseLLM):
            def __init__(self):
                super().__init__(LLMConfig())

            def _generate(self, prompt, system=None):
                raise ConnectionError("network down")

            def _generate_json(self, prompt, system=None):
                raise ConnectionError("network down")

        resolver = BatchConflictResolver(FailingLLM())
        facts = [
            Fact(content="Lives in Berlin", importance=8.0),
            Fact(content="Works at Google", importance=7.0),
        ]
        existing = [MemorySearchResult(
            memory=Memory(id="mem-1", content="Lives in Berlin"),
            similarity_score=0.9,
        )]

        actions = resolver.resolve(facts, existing)
        # "Lives in Berlin" should be deduped (exact match), only "Works at Google" added
        assert len(actions) == 1
        assert actions[0].fact == "Works at Google"
        assert actions[0].action.value == "add"

    def test_llm_failure_does_not_catch_keyboard_interrupt(self):
        from widemem.core.types import Memory

        class InterruptLLM(BaseLLM):
            def __init__(self):
                super().__init__(LLMConfig())

            def _generate(self, prompt, system=None):
                raise KeyboardInterrupt

            def _generate_json(self, prompt, system=None):
                raise KeyboardInterrupt

        resolver = BatchConflictResolver(InterruptLLM())
        facts = [Fact(content="test", importance=5.0)]
        existing = [MemorySearchResult(
            memory=Memory(id="mem-1", content="old"),
            similarity_score=0.5,
        )]

        with pytest.raises(KeyboardInterrupt):
            resolver.resolve(facts, existing)


class TestWideMemory:
    def test_add_and_search(self, memory, mock_embedder):
        extractor = MockExtractor()
        extractor.facts_to_return = [Fact(content="Lives in Berlin", importance=8.0)]
        memory.pipeline.extractor = extractor

        result = memory.add("I live in Berlin", user_id="alice")
        assert len(result.memories) == 1
        assert result.memories[0].content == "Lives in Berlin"
        assert result.memories[0].user_id == "alice"

        search_results = memory.search("where does alice live", user_id="alice")
        assert len(search_results) >= 1
        assert search_results[0].memory.content == "Lives in Berlin"

    def test_add_multiple_facts(self, memory):
        extractor = MockExtractor()
        extractor.facts_to_return = [
            Fact(content="Lives in Berlin", importance=8.0),
            Fact(content="Works at Google", importance=7.0),
        ]
        memory.pipeline.extractor = extractor

        result = memory.add("I live in Berlin and work at Google", user_id="alice")
        assert len(result.memories) == 2

    def test_get_memory(self, memory):
        extractor = MockExtractor()
        extractor.facts_to_return = [Fact(content="Test fact", importance=5.0)]
        memory.pipeline.extractor = extractor

        result = memory.add("Test fact", user_id="alice")
        assert len(result.memories) == 1

        retrieved = memory.get(result.memories[0].id)
        assert retrieved is not None
        assert retrieved.content == "Test fact"

    def test_get_preserves_all_metadata(self, memory):
        extractor = MockExtractor()
        extractor.facts_to_return = [
            Fact(content="Has a peanut allergy", importance=9.0, ymyl_category="health")
        ]
        memory.pipeline.extractor = extractor

        result = memory.add("I have a peanut allergy", user_id="alice", agent_id="nurse-bot")
        assert len(result.memories) == 1
        original = result.memories[0]

        retrieved = memory.get(original.id)
        assert retrieved is not None
        assert retrieved.id == original.id
        assert retrieved.content == "Has a peanut allergy"
        assert retrieved.user_id == "alice"
        assert retrieved.agent_id == "nurse-bot"
        assert retrieved.ymyl_category == "health"
        assert retrieved.importance == 9.0
        assert retrieved.tier == original.tier
        assert retrieved.content_hash == original.content_hash
        assert retrieved.created_at == original.created_at
        assert retrieved.updated_at == original.updated_at

    def test_delete_memory(self, memory):
        extractor = MockExtractor()
        extractor.facts_to_return = [Fact(content="To be deleted", importance=5.0)]
        memory.pipeline.extractor = extractor

        result = memory.add("To be deleted", user_id="alice")
        memory.delete(result.memories[0].id)

        retrieved = memory.get(result.memories[0].id)
        assert retrieved is None


class TestInputValidation:
    def test_empty_text_returns_empty(self, memory):
        result = memory.add("", user_id="alice")
        assert len(result.memories) == 0

    def test_whitespace_text_returns_empty(self, memory):
        result = memory.add("   ", user_id="alice")
        assert len(result.memories) == 0

    def test_text_too_long_raises(self, memory):
        with pytest.raises(ValueError, match="Text too long"):
            memory.add("x" * 50_001, user_id="alice")

    def test_text_at_limit_accepted(self, memory):
        extractor = MockExtractor()
        memory.pipeline.extractor = extractor
        result = memory.add("x" * 50_000, user_id="alice")
        assert len(result.memories) >= 1

    def test_pin_empty_returns_empty(self, memory):
        result = memory.pin("", user_id="alice")
        assert len(result.memories) == 0

    def test_pin_too_long_raises(self, memory):
        with pytest.raises(ValueError, match="Text too long"):
            memory.pin("x" * 50_001, user_id="alice")


class TestBatchAdd:
    def test_add_batch_multiple_texts(self, memory):
        extractor = MockExtractor()
        memory.pipeline.extractor = extractor

        results = memory.add_batch(
            ["I live in Berlin", "I work at Google"],
            user_id="alice",
        )
        assert len(results) == 2
        assert all(len(r.memories) >= 1 for r in results)

    def test_add_batch_empty(self, memory):
        results = memory.add_batch([], user_id="alice")
        assert results == []


class TestMemoryCount:
    def test_count_all(self, memory):
        extractor = MockExtractor()
        memory.pipeline.extractor = extractor

        memory.add("fact one", user_id="alice")
        memory.add("fact two", user_id="alice")
        assert memory.count(user_id="alice") == 2

    def test_count_empty(self, memory):
        assert memory.count(user_id="alice") == 0

    def test_count_filtered(self, memory):
        extractor = MockExtractor()
        memory.pipeline.extractor = extractor

        memory.add("fact one", user_id="alice")
        memory.add("fact two", user_id="bob")
        assert memory.count(user_id="alice") == 1
        assert memory.count(user_id="bob") == 1


class TestExportImport:
    def test_export_json(self, memory):
        extractor = MockExtractor()
        memory.pipeline.extractor = extractor

        memory.add("Lives in Berlin", user_id="alice")
        data = memory.export_json(user_id="alice")
        parsed = json.loads(data)
        assert parsed["count"] == 1
        assert parsed["memories"][0]["content"] == "Lives in Berlin"

    def test_import_json(self, tmp_dir):
        config = MemoryConfig(history_db_path=f"{tmp_dir}/h2.db")
        embedder = MockEmbedder(dimensions=64)
        vs = FAISSVectorStore(VectorStoreConfig(), dimensions=64)
        mem = WideMemory(config=config, llm=MockLLM(), embedder=embedder, vector_store=vs)

        data = json.dumps({"memories": [
            {"id": "test-id-1", "content": "Lives in Berlin", "user_id": "alice", "importance": 8.0},
            {"id": "test-id-2", "content": "Works at Google", "user_id": "alice", "importance": 7.0},
        ]})
        imported = mem.import_json(data)
        assert imported == 2
        assert mem.count(user_id="alice") == 2

    def test_import_skips_existing(self, tmp_dir):
        config = MemoryConfig(history_db_path=f"{tmp_dir}/h3.db")
        embedder = MockEmbedder(dimensions=64)
        vs = FAISSVectorStore(VectorStoreConfig(), dimensions=64)
        mem = WideMemory(config=config, llm=MockLLM(), embedder=embedder, vector_store=vs)

        data = json.dumps({"memories": [
            {"id": "test-id-1", "content": "Lives in Berlin", "importance": 8.0},
        ]})
        assert mem.import_json(data) == 1
        assert mem.import_json(data) == 0

    def test_roundtrip(self, memory):
        extractor = MockExtractor()
        memory.pipeline.extractor = extractor
        memory.add("Lives in Berlin", user_id="alice")
        memory.add("Works at Google", user_id="alice")

        exported = memory.export_json(user_id="alice")
        parsed = json.loads(exported)
        assert parsed["count"] == 2


class TestTTL:
    def test_ttl_filters_old_memories(self, tmp_dir):
        config = MemoryConfig(history_db_path=f"{tmp_dir}/ttl.db", ttl_days=7)
        embedder = MockEmbedder(dimensions=64)
        vs = FAISSVectorStore(VectorStoreConfig(), dimensions=64)
        mem = WideMemory(config=config, llm=MockLLM(), embedder=embedder, vector_store=vs)

        from datetime import timedelta
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        new_time = datetime.now(timezone.utc).isoformat()

        vec1 = embedder.embed("old fact")
        vs.insert("id1", vec1, {
            "content": "old fact", "created_at": old_time, "updated_at": old_time,
            "user_id": "alice", "tier": "fact", "importance": 5.0,
        })
        vec2 = embedder.embed("new fact")
        vs.insert("id2", vec2, {
            "content": "new fact", "created_at": new_time, "updated_at": new_time,
            "user_id": "alice", "tier": "fact", "importance": 5.0,
        })

        results = mem.search("fact", user_id="alice")
        contents = [r.memory.content for r in results]
        assert "new fact" in contents
        assert "old fact" not in contents

    def test_no_ttl_returns_all(self, tmp_dir):
        config = MemoryConfig(history_db_path=f"{tmp_dir}/nottl.db", ttl_days=None)
        embedder = MockEmbedder(dimensions=64)
        vs = FAISSVectorStore(VectorStoreConfig(), dimensions=64)
        mem = WideMemory(config=config, llm=MockLLM(), embedder=embedder, vector_store=vs)

        from datetime import timedelta
        old_time = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()

        vec = embedder.embed("ancient fact")
        vs.insert("id1", vec, {
            "content": "ancient fact", "created_at": old_time, "updated_at": old_time,
            "user_id": "alice", "tier": "fact", "importance": 5.0,
        })

        results = mem.search("ancient", user_id="alice")
        assert len(results) == 1

    def test_search_handles_naive_legacy_timestamps(self, tmp_dir):
        """Search over legacy naive ISO timestamps must not TypeError against the aware now."""
        config = MemoryConfig(history_db_path=f"{tmp_dir}/legacy.db", ttl_days=7)
        embedder = MockEmbedder(dimensions=64)
        vs = FAISSVectorStore(VectorStoreConfig(), dimensions=64)
        mem = WideMemory(config=config, llm=MockLLM(), embedder=embedder, vector_store=vs)

        from datetime import timedelta
        naive_recent = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None).isoformat()
        assert "+" not in naive_recent and "Z" not in naive_recent

        vec = embedder.embed("legacy fact")
        vs.insert("legacy_id", vec, {
            "content": "legacy fact", "created_at": naive_recent, "updated_at": naive_recent,
            "user_id": "alice", "tier": "fact", "importance": 5.0,
        })

        results = mem.search("legacy", user_id="alice")
        assert len(results) == 1
        assert results[0].memory.content == "legacy fact"
        assert results[0].memory.created_at.tzinfo is not None

    def test_search_survives_corrupt_stored_timestamp(self, tmp_dir):
        """A garbage created_at/updated_at in stored metadata (e.g. from a bad
        import_json) must not crash search; it falls back to an aware now."""
        config = MemoryConfig(history_db_path=f"{tmp_dir}/corrupt.db")
        embedder = MockEmbedder(dimensions=64)
        vs = FAISSVectorStore(VectorStoreConfig(), dimensions=64)
        mem = WideMemory(config=config, llm=MockLLM(), embedder=embedder, vector_store=vs)

        vec = embedder.embed("corrupt fact")
        vs.insert("corrupt_id", vec, {
            "content": "corrupt fact", "created_at": "not-a-timestamp",
            "updated_at": "", "user_id": "alice", "tier": "fact", "importance": 5.0,
        })

        results = mem.search("corrupt", user_id="alice")
        assert len(results) == 1
        assert results[0].memory.content == "corrupt fact"
        assert results[0].memory.created_at.tzinfo is not None
        assert results[0].memory.updated_at.tzinfo is not None


class TestEventTime:
    def test_memory_event_time_defaults_none(self):
        assert Memory(content="x").event_time is None

    def test_add_timestamp_sets_event_time(self, memory):
        memory.pipeline.extractor = MockExtractor()
        memory.pipeline.extractor.facts_to_return = [Fact(content="Caroline joined the LGBTQ group", importance=8.0)]
        dt = datetime(2023, 5, 7, tzinfo=timezone.utc)

        memory.add("Caroline joined the LGBTQ support group", user_id="caro", timestamp=dt)
        results = memory.search("LGBTQ group", user_id="caro")
        assert len(results) == 1
        assert results[0].memory.event_time == as_utc(dt)

    def test_add_without_timestamp_leaves_event_time_none(self, memory):
        memory.pipeline.extractor = MockExtractor()
        memory.pipeline.extractor.facts_to_return = [Fact(content="Caroline likes hiking", importance=6.0)]

        memory.add("Caroline likes hiking", user_id="caro")
        results = memory.search("hiking", user_id="caro")
        assert len(results) == 1
        assert results[0].memory.event_time is None

    def test_naive_timestamp_normalized_to_utc(self, memory):
        memory.pipeline.extractor = MockExtractor()
        memory.pipeline.extractor.facts_to_return = [Fact(content="Melanie went camping", importance=5.0)]

        memory.add("Melanie went camping", user_id="mel", timestamp=datetime(2023, 6, 27))
        results = memory.search("camping", user_id="mel")
        assert results[0].memory.event_time is not None
        assert results[0].memory.event_time.tzinfo is not None

    def test_event_time_roundtrip_export_import(self, tmp_dir):
        dt = datetime(2023, 7, 3, tzinfo=timezone.utc)
        src = WideMemory(
            config=MemoryConfig(history_db_path=f"{tmp_dir}/src.db"),
            llm=MockLLM(), embedder=MockEmbedder(64),
            vector_store=FAISSVectorStore(VectorStoreConfig(), dimensions=64),
        )
        src.pipeline.extractor = MockExtractor()
        src.pipeline.extractor.facts_to_return = [Fact(content="Caroline went to pride", importance=7.0)]
        src.add("Caroline went to a pride parade", user_id="caro", timestamp=dt)
        dumped = src.export_json()

        dst = WideMemory(
            config=MemoryConfig(history_db_path=f"{tmp_dir}/dst.db"),
            llm=MockLLM(), embedder=MockEmbedder(64),
            vector_store=FAISSVectorStore(VectorStoreConfig(), dimensions=64),
        )
        assert dst.import_json(dumped) == 1
        results = dst.search("pride parade", user_id="caro")
        assert len(results) == 1
        assert results[0].memory.event_time == as_utc(dt)

    def test_metadata_serializes_event_time_only_when_set(self, memory):
        dt = datetime(2023, 8, 13, tzinfo=timezone.utc)
        with_ts = memory.pipeline._memory_to_metadata(Memory(content="a", event_time=dt))
        without_ts = memory.pipeline._memory_to_metadata(Memory(content="b"))
        assert with_ts["event_time"] == dt.isoformat()
        assert "event_time" not in without_ts

    def test_leading_date_in_text_populates_event_time(self, memory):
        memory.pipeline.extractor = MockExtractor()
        memory.pipeline.extractor.facts_to_return = [Fact(content="Caroline went to pride", importance=7.0)]

        memory.add("[2023-07-03] Caroline went to a pride parade", user_id="caro")
        results = memory.search("pride parade", user_id="caro")
        assert len(results) == 1
        et = results[0].memory.event_time
        assert et is not None and (et.year, et.month, et.day) == (2023, 7, 3)

    def test_explicit_timestamp_beats_leading_date_in_text(self, memory):
        memory.pipeline.extractor = MockExtractor()
        memory.pipeline.extractor.facts_to_return = [Fact(content="Caroline event", importance=7.0)]
        explicit = datetime(2023, 5, 7, tzinfo=timezone.utc)

        memory.add("[2020-01-01] some old-looking prefix", user_id="caro", timestamp=explicit)
        results = memory.search("Caroline event", user_id="caro")
        assert results[0].memory.event_time == as_utc(explicit)


class TestRetryBackoff:
    def test_retry_on_transient_error(self):
        call_count = 0

        class FlakeyLLM(BaseLLM):
            def __init__(self):
                super().__init__(LLMConfig(), max_retries=3, retry_delay=0.01)

            def _generate(self, prompt, system=None):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("network flake")
                return "ok"

            def _generate_json(self, prompt, system=None):
                return {}

        llm = FlakeyLLM()
        result = llm.generate("test")
        assert result == "ok"
        assert call_count == 3

    def test_provider_error_not_retried(self):
        from widemem.core.exceptions import ProviderError

        class BadLLM(BaseLLM):
            def __init__(self):
                super().__init__(LLMConfig(), max_retries=3, retry_delay=0.01)

            def _generate(self, prompt, system=None):
                raise ProviderError("bad request")

            def _generate_json(self, prompt, system=None):
                return {}

        llm = BadLLM()
        with pytest.raises(ProviderError, match="bad request"):
            llm.generate("test")


class TestScoreBreakdown:
    def test_search_results_have_score_components(self, memory):
        extractor = MockExtractor()
        extractor.facts_to_return = [Fact(content="Lives in Berlin", importance=8.0)]
        memory.pipeline.extractor = extractor
        memory.add("I live in Berlin", user_id="alice")

        results = memory.search("Lives in Berlin", user_id="alice")
        assert len(results) >= 1
        r = results[0]
        assert r.similarity_score > 0
        assert r.temporal_score > 0
        assert r.importance_score > 0
        assert r.final_score > 0


class TestIDMapping:
    def test_uuid_to_int_mapping(self):
        from widemem.utils.id_mapping import IDMapper

        mapper = IDMapper()
        idx1 = mapper.add("uuid-1")
        idx2 = mapper.add("uuid-2")

        assert idx1 == 1
        assert idx2 == 2
        assert mapper.to_uuid(1) == "uuid-1"
        assert mapper.to_uuid(2) == "uuid-2"
        assert mapper.to_int("uuid-1") == 1

    def test_duplicate_add(self):
        from widemem.utils.id_mapping import IDMapper

        mapper = IDMapper()
        idx1 = mapper.add("uuid-1")
        idx2 = mapper.add("uuid-1")
        assert idx1 == idx2

    @pytest.mark.parametrize(
        "uuids",
        [
            ["uuid-1"],
            ["uuid-1", "uuid-2", "uuid-3"],
            ["a", "b", "a", "c"],
        ],
    )
    def test_round_trip(self, uuids):
        from widemem.utils.id_mapping import IDMapper

        mapper = IDMapper()
        for u in uuids:
            idx = mapper.add(u)
            assert mapper.to_uuid(idx) == u

    @pytest.mark.parametrize(
        "uuids",
        [
            ["uuid-1"],
            ["uuid-1", "uuid-2", "uuid-3"],
        ],
    )
    def test_reverse_round_trip(self, uuids):
        from widemem.utils.id_mapping import IDMapper

        mapper = IDMapper()
        for u in uuids:
            idx = mapper.add(u)
            assert mapper.to_int(u) == idx

    def test_unknown_lookups_return_none(self):
        from widemem.utils.id_mapping import IDMapper

        mapper = IDMapper()
        assert mapper.to_uuid(1) is None
        assert mapper.to_int("missing") is None

        mapper.add("uuid-1")
        assert mapper.to_uuid(99) is None
        assert mapper.to_int("other") is None

    def test_monotonic_allocation_with_duplicates(self):
        from widemem.utils.id_mapping import IDMapper

        mapper = IDMapper()
        a1 = mapper.add("a")
        b1 = mapper.add("b")
        a2 = mapper.add("a")
        c1 = mapper.add("c")
        assert (a1, b1, a2, c1) == (1, 2, 1, 3)

    def test_duplicate_does_not_advance_counter(self):
        from widemem.utils.id_mapping import IDMapper

        mapper = IDMapper()
        mapper.add("a")
        mapper.add("a")
        mapper.add("a")
        assert mapper.add("b") == 2


class TestEmbeddingCache:
    def test_cache_avoids_recomputation(self):
        call_count = 0

        class CountingEmbedder(BaseEmbedder):
            def __init__(self):
                super().__init__(EmbeddingConfig(dimensions=4), max_retries=1, retry_delay=0)

            def _embed(self, text):
                nonlocal call_count
                call_count += 1
                return [0.1, 0.2, 0.3, 0.4]

            def _embed_batch(self, texts):
                return [self._embed(t) for t in texts]

        embedder = CountingEmbedder()
        embedder.embed("hello")
        embedder.embed("hello")
        embedder.embed("hello")
        assert call_count == 1

    def test_cache_evicts_oldest(self):
        class TinyEmbedder(BaseEmbedder):
            def __init__(self):
                super().__init__(EmbeddingConfig(dimensions=4), max_retries=1, retry_delay=0, cache_size=2)

            def _embed(self, text):
                return [hash(text) % 100 / 100.0, 0.2, 0.3, 0.4]

            def _embed_batch(self, texts):
                return [self._embed(t) for t in texts]

        embedder = TinyEmbedder()
        embedder.embed("a")
        embedder.embed("b")
        embedder.embed("c")  # evicts "a"
        assert len(embedder._cache) == 2
        assert "a" not in embedder._cache

    def test_batch_uses_cache(self):
        call_count = 0

        class CountingEmbedder(BaseEmbedder):
            def __init__(self):
                super().__init__(EmbeddingConfig(dimensions=4), max_retries=1, retry_delay=0)

            def _embed(self, text):
                nonlocal call_count
                call_count += 1
                return [0.1, 0.2, 0.3, 0.4]

            def _embed_batch(self, texts):
                nonlocal call_count
                call_count += len(texts)
                return [[0.1, 0.2, 0.3, 0.4]] * len(texts)

        embedder = CountingEmbedder()
        embedder.embed("a")  # 1 call
        embedder.embed("b")  # 1 call
        call_count = 0
        embedder.embed_batch(["a", "b", "c"])  # only "c" hits provider
        assert call_count == 1


class TestEmbeddingRetry:
    def test_retry_on_transient_error(self):
        call_count = 0

        class FlakeyEmbedder(BaseEmbedder):
            def __init__(self):
                super().__init__(EmbeddingConfig(dimensions=4), max_retries=3, retry_delay=0.01)

            def _embed(self, text):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("network flake")
                return [0.1, 0.2, 0.3, 0.4]

            def _embed_batch(self, texts):
                return [self._embed(t) for t in texts]

        embedder = FlakeyEmbedder()
        result = embedder.embed("test")
        assert result == [0.1, 0.2, 0.3, 0.4]
        assert call_count == 3

    def test_provider_error_not_retried(self):
        from widemem.core.exceptions import ProviderError

        class BadEmbedder(BaseEmbedder):
            def __init__(self):
                super().__init__(EmbeddingConfig(dimensions=4), max_retries=3, retry_delay=0.01)

            def _embed(self, text):
                raise ProviderError("invalid model")

            def _embed_batch(self, texts):
                raise ProviderError("invalid model")

        embedder = BadEmbedder()
        with pytest.raises(ProviderError, match="invalid model"):
            embedder.embed("test")


class TestContentHash:
    def test_same_content_same_hash(self):
        from widemem.utils.hashing import content_hash

        assert content_hash("hello") == content_hash("hello")

    def test_different_content_different_hash(self):
        from widemem.utils.hashing import content_hash

        assert content_hash("hello") != content_hash("world")
