"""Focused tests for BaseVectorStore.count(), the qdrant live-payload copy
fix, the pgvector finiteness guard, and import_json batched embedding.

Backends with optional extras (qdrant, pgvector) are exercised through fakes or
importorskip so the suite stays green without a live server. FAISS count() is
tested directly against the real store.
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest

from widemem.core.memory import WideMemory
from widemem.core.types import (
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
    VectorStoreConfig,
)
from widemem.providers.embeddings.base import BaseEmbedder
from widemem.providers.llm.base import BaseLLM
from widemem.storage.vector.faiss_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------
class _CountingEmbedder(BaseEmbedder):
    """Embedder that records batch vs single embed calls."""

    def __init__(self, dimensions: int = 8) -> None:
        super().__init__(EmbeddingConfig(dimensions=dimensions), max_retries=1, retry_delay=0)
        self.single_calls = 0
        self.batch_calls = 0

    def _embed(self, text: str) -> list[float]:
        self.single_calls += 1
        return self._vec(text)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % 2**31)
        v = rng.randn(self.config.dimensions).astype(np.float32)
        v = v / np.linalg.norm(v)
        return v.tolist()


class _NullLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig())

    def _generate(self, prompt: str, system: str | None = None) -> str:
        return json.dumps({"facts": []})

    def _generate_json(self, prompt: str, system: str | None = None) -> dict:
        return {"facts": []}


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# FAISS count()
# ---------------------------------------------------------------------------
class TestFaissCount:
    def _store(self, tmp_dir):
        return FAISSVectorStore(VectorStoreConfig(path=f"{tmp_dir}/vec"), dimensions=4)

    def test_count_empty(self, tmp_dir):
        assert self._store(tmp_dir).count() == 0

    def test_count_after_inserts(self, tmp_dir):
        store = self._store(tmp_dir)
        store.insert("a", [1.0, 0.0, 0.0, 0.0], {"content": "a", "user_id": "alice"})
        store.insert("b", [0.0, 1.0, 0.0, 0.0], {"content": "b", "user_id": "alice"})
        store.insert("c", [0.0, 0.0, 1.0, 0.0], {"content": "c", "user_id": "bob"})
        assert store.count() == 3

    def test_count_with_filters(self, tmp_dir):
        store = self._store(tmp_dir)
        store.insert("a", [1.0, 0.0, 0.0, 0.0], {"content": "a", "user_id": "alice"})
        store.insert("b", [0.0, 1.0, 0.0, 0.0], {"content": "b", "user_id": "alice"})
        store.insert("c", [0.0, 0.0, 1.0, 0.0], {"content": "c", "user_id": "bob"})
        assert store.count(filters={"user_id": "alice"}) == 2
        assert store.count(filters={"user_id": "bob"}) == 1
        assert store.count(filters={"user_id": "nobody"}) == 0

    def test_count_tracks_delete(self, tmp_dir):
        store = self._store(tmp_dir)
        store.insert("a", [1.0, 0.0, 0.0, 0.0], {"content": "a"})
        store.insert("b", [0.0, 1.0, 0.0, 0.0], {"content": "b"})
        store.delete("a")
        assert store.count() == 1


# ---------------------------------------------------------------------------
# Base default count() fallback
# ---------------------------------------------------------------------------
def test_base_count_default_falls_back_to_list_all():
    from widemem.storage.vector.base import BaseVectorStore

    class _ListOnlyStore(BaseVectorStore):
        def __init__(self):
            self._rows = [("a", {"user_id": "x"}), ("b", {"user_id": "y"})]

        def insert(self, id, vector, metadata):  # pragma: no cover - unused
            ...

        def search(self, vector, top_k=10, filters=None):  # pragma: no cover
            return []

        def update(self, id, vector, metadata):  # pragma: no cover
            ...

        def delete(self, id):  # pragma: no cover
            ...

        def get(self, id):  # pragma: no cover
            return None

        def list_all(self, filters=None, max_results=1000):
            if filters:
                return [r for r in self._rows if all(r[1].get(k) == v for k, v in filters.items())]
            return list(self._rows)

    store = _ListOnlyStore()
    assert store.count() == 2
    assert store.count(filters={"user_id": "x"}) == 1


# ---------------------------------------------------------------------------
# WideMemory.count() delegates to store.count()
# ---------------------------------------------------------------------------
def test_memory_count_uses_store_count(tmp_dir):
    store = FAISSVectorStore(VectorStoreConfig(path=f"{tmp_dir}/vec"), dimensions=8)
    embedder = _CountingEmbedder(dimensions=8)
    config = MemoryConfig(history_db_path=f"{tmp_dir}/history.db")
    mem = WideMemory(config=config, llm=_NullLLM(), embedder=embedder, vector_store=store)

    data = json.dumps(
        {
            "memories": [
                {"id": "m1", "content": "one", "user_id": "alice"},
                {"id": "m2", "content": "two", "user_id": "alice"},
                {"id": "m3", "content": "three", "user_id": "bob"},
            ]
        }
    )
    mem.import_json(data)

    assert mem.count() == 3
    assert mem.count(user_id="alice") == 2
    assert mem.count(user_id="bob") == 1


# ---------------------------------------------------------------------------
# import_json uses batched embedding and preserves ordering
# ---------------------------------------------------------------------------
def test_import_json_batches_embeddings_and_preserves_order(tmp_dir):
    store = FAISSVectorStore(VectorStoreConfig(path=f"{tmp_dir}/vec"), dimensions=8)
    embedder = _CountingEmbedder(dimensions=8)
    config = MemoryConfig(history_db_path=f"{tmp_dir}/history.db")
    mem = WideMemory(config=config, llm=_NullLLM(), embedder=embedder, vector_store=store)

    items = [{"id": f"m{i}", "content": f"fact number {i}"} for i in range(5)]
    imported = mem.import_json(json.dumps({"memories": items}))

    assert imported == 5
    # A single batched embed call, no per-item serial embed.
    assert embedder.batch_calls == 1
    assert embedder.single_calls == 0
    # Ordering preserved: stored ids match input order.
    stored_ids = [id_ for id_, _ in store.list_all(max_results=100)]
    assert stored_ids == [f"m{i}" for i in range(5)]


def test_import_json_skips_duplicates_and_blanks(tmp_dir):
    store = FAISSVectorStore(VectorStoreConfig(path=f"{tmp_dir}/vec"), dimensions=8)
    embedder = _CountingEmbedder(dimensions=8)
    config = MemoryConfig(history_db_path=f"{tmp_dir}/history.db")
    mem = WideMemory(config=config, llm=_NullLLM(), embedder=embedder, vector_store=store)

    mem.import_json(json.dumps({"memories": [{"id": "m1", "content": "already here"}]}))

    data = json.dumps(
        {
            "memories": [
                {"id": "m1", "content": "duplicate of existing"},  # skipped: exists
                {"id": "m2", "content": ""},  # skipped: blank
                {"id": "m3", "content": "brand new"},  # imported
                {"id": "m3", "content": "same-batch dup"},  # skipped: seen this batch
            ]
        }
    )
    imported = mem.import_json(data)
    assert imported == 1
    assert mem.count() == 2


# ---------------------------------------------------------------------------
# Qdrant: get/search/list_all must not mutate the client's live payload
# ---------------------------------------------------------------------------
class _FakePoint:
    def __init__(self, id, payload, score=0.9, vector=None):
        self.id = id
        self.payload = payload
        self.score = score
        self.vector = vector


class _FakeQueryResult:
    def __init__(self, points):
        self.points = points


class _FakeQdrantClient:
    """Returns the SAME live payload dict on every read, mimicking an
    in-process (embedded) client whose payloads are not defensively copied."""

    def __init__(self, payload):
        self.payload = payload
        self.point_id = "00000000-0000-0000-0000-000000000001"

    def retrieve(self, collection_name, ids, with_vectors=False, with_payload=True):
        return [_FakePoint(self.point_id, self.payload, vector=[0.1, 0.2, 0.3])]

    def query_points(self, collection_name, query, limit, query_filter, with_payload):
        return _FakeQueryResult([_FakePoint(self.point_id, self.payload)])

    def scroll(self, collection_name, scroll_filter, limit, with_payload, with_vectors):
        return ([_FakePoint(self.point_id, self.payload)], None)


def _make_qdrant_store_with_fake(live_payload):
    pytest.importorskip("qdrant_client")
    from widemem.storage.vector.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore.__new__(QdrantVectorStore)  # bypass real client init
    store.collection_name = "widemem"
    store.dimensions = 3
    store.client = _FakeQdrantClient(live_payload)
    return store


def test_qdrant_get_does_not_corrupt_live_payload():
    live = {"_widemem_id": "alice", "content": "x"}
    store = _make_qdrant_store_with_fake(live)
    result = store.get("alice")
    assert result is not None
    _, meta = result
    assert "_widemem_id" not in meta  # stripped from the returned copy
    assert live.get("_widemem_id") == "alice"  # live object untouched
    assert meta["content"] == "x"


def test_qdrant_search_does_not_corrupt_live_payload():
    live = {"_widemem_id": "alice", "content": "x"}
    store = _make_qdrant_store_with_fake(live)
    # Two reads in a row: the first must not strip the id from the shared object,
    # or the second would fall back to the raw point id.
    first = store.search([0.1, 0.2, 0.3], top_k=1)
    second = store.search([0.1, 0.2, 0.3], top_k=1)
    assert first[0][0] == "alice"
    assert second[0][0] == "alice"
    assert "_widemem_id" not in first[0][2]
    assert live.get("_widemem_id") == "alice"


def test_qdrant_list_all_does_not_corrupt_live_payload():
    live = {"_widemem_id": "alice", "content": "x"}
    store = _make_qdrant_store_with_fake(live)
    first = store.list_all()
    second = store.list_all()
    assert first[0][0] == "alice"
    assert second[0][0] == "alice"
    assert live.get("_widemem_id") == "alice"


# ---------------------------------------------------------------------------
# Pgvector: reject non-finite vector components before building the literal
# ---------------------------------------------------------------------------
def _make_pgvector_store_no_db(dimensions=3):
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    from widemem.storage.vector.pgvector_store import PgVectorStore

    store = PgVectorStore.__new__(PgVectorStore)  # bypass DB connection
    store.dimensions = dimensions
    return store


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_pgvector_rejects_non_finite(bad):
    store = _make_pgvector_store_no_db(dimensions=3)
    with pytest.raises(ValueError, match="non-finite"):
        store._validate_vector([0.1, bad, 0.3])


def test_pgvector_accepts_finite(tmp_dir):
    store = _make_pgvector_store_no_db(dimensions=3)
    # Should not raise.
    store._validate_vector([0.1, -0.2, 0.3])


def test_pgvector_dimension_mismatch_still_raises():
    store = _make_pgvector_store_no_db(dimensions=3)
    with pytest.raises(ValueError, match="dimension mismatch"):
        store._validate_vector([0.1, 0.2])
