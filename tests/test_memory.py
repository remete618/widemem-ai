"""Integration tests for WideMemory using mock LLM and embedder."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime

import numpy as np
import pytest

from widemem.conflict.batch_resolver import BatchConflictResolver
from widemem.core.memory import WideMemory
from widemem.core.types import (
    EmbeddingConfig,
    Fact,
    LLMConfig,
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
        super().__init__(EmbeddingConfig(dimensions=dimensions))
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        if text not in self._cache:
            rng = np.random.RandomState(hash(text) % 2**31)
            vec = rng.randn(self.config.dimensions).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            self._cache[text] = vec.tolist()
        return self._cache[text]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


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

    def test_delete_memory(self, memory):
        extractor = MockExtractor()
        extractor.facts_to_return = [Fact(content="To be deleted", importance=5.0)]
        memory.pipeline.extractor = extractor

        result = memory.add("To be deleted", user_id="alice")
        memory.delete(result.memories[0].id)

        retrieved = memory.get(result.memories[0].id)
        assert retrieved is None


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
        old_time = (datetime.utcnow() - timedelta(days=30)).isoformat()
        new_time = datetime.utcnow().isoformat()

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
        old_time = (datetime.utcnow() - timedelta(days=365)).isoformat()

        vec = embedder.embed("ancient fact")
        vs.insert("id1", vec, {
            "content": "ancient fact", "created_at": old_time, "updated_at": old_time,
            "user_id": "alice", "tier": "fact", "importance": 5.0,
        })

        results = mem.search("ancient", user_id="alice")
        assert len(results) == 1


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


class TestContentHash:
    def test_same_content_same_hash(self):
        from widemem.utils.hashing import content_hash

        assert content_hash("hello") == content_hash("hello")

    def test_different_content_different_hash(self):
        from widemem.utils.hashing import content_hash

        assert content_hash("hello") != content_hash("world")
