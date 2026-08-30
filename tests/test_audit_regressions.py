"""Regression tests for the adversarial audit findings.

Each test here failed before its fix landed. They are grouped by the
defect they pin, not by module, so a reader can trace a test back to
the audit finding it closes.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone

import numpy as np
import pytest

from widemem.core.memory import WideMemory
from widemem.core.pipeline import MemoryPipeline
from widemem.core.types import (
    ActionItem,
    EmbeddingConfig,
    Fact,
    LLMConfig,
    MemoryAction,
    MemoryConfig,
    VectorStoreConfig,
)
from widemem.extraction.base import BaseExtractor
from widemem.providers.embeddings.base import BaseEmbedder
from widemem.providers.llm.base import BaseLLM
from widemem.storage.history import HistoryStore
from widemem.storage.vector.faiss_store import FAISSVectorStore


class StubLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig())

    def _generate(self, prompt: str, system: str | None = None) -> str:
        return "{}"

    def _generate_json(self, prompt: str, system: str | None = None) -> dict:
        return {}


class StubEmbedder(BaseEmbedder):
    def __init__(self, dimensions: int = 32) -> None:
        super().__init__(EmbeddingConfig(dimensions=dimensions), max_retries=1, retry_delay=0)

    def _embed(self, text: str) -> list[float]:
        rng = np.random.RandomState(abs(hash(text)) % 2**31)
        vec = rng.randn(self.config.dimensions).astype(np.float32)
        return (vec / np.linalg.norm(vec)).tolist()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


class StubExtractor(BaseExtractor):
    def __init__(self, facts: list[Fact]) -> None:
        self._facts = facts

    def extract(self, text: str) -> list[Fact]:
        return list(self._facts)


class StubResolver:
    """Stands in for BatchConflictResolver, returning a scripted action list."""

    def __init__(self, actions: list[ActionItem]) -> None:
        self._actions = actions

    def resolve(self, facts, existing, linked_memories=None) -> list[ActionItem]:
        return list(self._actions)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def store(tmp_dir):
    return FAISSVectorStore(VectorStoreConfig(path=f"{tmp_dir}/vectors"), dimensions=32)


@pytest.fixture
def history(tmp_dir):
    return HistoryStore(f"{tmp_dir}/history.db")


@pytest.fixture
def embedder():
    return StubEmbedder(dimensions=32)


def _pipeline(extractor, resolver, embedder, store, history):
    return MemoryPipeline(
        extractor=extractor,
        resolver=resolver,
        embedder=embedder,
        vector_store=store,
        history=history,
    )


def _seed(store, embedder, id, content, **meta):
    base = {"content": content, "content_hash": f"hash-{id}"}
    base.update(meta)
    store.insert(id=id, vector=embedder.embed(content), metadata=base)
    return id


class TestCrossTenantWrite:
    """pipeline.py:133 - an unscoped add() must not mutate a scoped memory."""

    def test_unscoped_update_cannot_rewrite_another_users_memory(
        self, store, embedder, history
    ):
        _seed(store, embedder, "alice-1", "alice takes warfarin", user_id="alice")

        pipeline = _pipeline(
            StubExtractor([Fact(content="bob takes aspirin", importance=7.0)]),
            StubResolver([
                ActionItem(
                    action=MemoryAction.UPDATE,
                    fact="bob takes aspirin",
                    target_id="alice-1",
                    importance=7.0,
                )
            ]),
            embedder,
            store,
            history,
        )

        pipeline.process("bob takes aspirin")

        survivor = store.get("alice-1")
        assert survivor is not None, "alice's memory was deleted by an unscoped write"
        assert survivor[1]["user_id"] == "alice", (
            "alice's memory was re-scoped to the caller; it is now invisible to her"
        )
        assert survivor[1]["content"] == "alice takes warfarin", (
            "alice's memory content was overwritten by another caller"
        )

    def test_unscoped_delete_cannot_remove_another_users_memory(
        self, store, embedder, history
    ):
        _seed(store, embedder, "alice-1", "alice takes warfarin", user_id="alice")

        pipeline = _pipeline(
            StubExtractor([Fact(content="bob takes aspirin", importance=7.0)]),
            StubResolver([
                ActionItem(
                    action=MemoryAction.DELETE,
                    fact="bob takes aspirin",
                    target_id="alice-1",
                    importance=7.0,
                )
            ]),
            embedder,
            store,
            history,
        )

        pipeline.process("bob takes aspirin")

        assert store.get("alice-1") is not None, (
            "alice's memory was deleted by an unscoped caller"
        )

    def test_scoped_caller_cannot_touch_a_different_users_memory(
        self, store, embedder, history
    ):
        _seed(store, embedder, "alice-1", "alice takes warfarin", user_id="alice")

        pipeline = _pipeline(
            StubExtractor([Fact(content="bob takes aspirin", importance=7.0)]),
            StubResolver([
                ActionItem(
                    action=MemoryAction.DELETE,
                    fact="bob takes aspirin",
                    target_id="alice-1",
                    importance=7.0,
                )
            ]),
            embedder,
            store,
            history,
        )

        pipeline.process("bob takes aspirin", user_id="bob")

        assert store.get("alice-1") is not None, (
            "bob deleted alice's memory"
        )

    def test_owner_can_still_update_their_own_memory(self, store, embedder, history):
        _seed(store, embedder, "alice-1", "alice takes warfarin", user_id="alice")

        pipeline = _pipeline(
            StubExtractor([Fact(content="alice takes apixaban", importance=8.0)]),
            StubResolver([
                ActionItem(
                    action=MemoryAction.UPDATE,
                    fact="alice takes apixaban",
                    target_id="alice-1",
                    importance=8.0,
                )
            ]),
            embedder,
            store,
            history,
        )

        pipeline.process("alice takes apixaban", user_id="alice")

        updated = store.get("alice-1")
        assert updated is not None
        assert updated[1]["content"] == "alice takes apixaban"
        assert updated[1]["user_id"] == "alice"

    def test_single_tenant_update_still_works(self, store, embedder, history):
        """No user_id anywhere is the documented single-tenant mode; it must keep working."""
        _seed(store, embedder, "solo-1", "the cat is called mimi")

        pipeline = _pipeline(
            StubExtractor([Fact(content="the cat is called cleo", importance=6.0)]),
            StubResolver([
                ActionItem(
                    action=MemoryAction.UPDATE,
                    fact="the cat is called cleo",
                    target_id="solo-1",
                    importance=6.0,
                )
            ]),
            embedder,
            store,
            history,
        )

        pipeline.process("the cat is called cleo")

        updated = store.get("solo-1")
        assert updated is not None
        assert updated[1]["content"] == "the cat is called cleo"


class TestUpdatePreservesMetadata:
    """pipeline.py:222 - UPDATE rebuilt metadata from scratch, wiping fields."""

    def test_update_preserves_event_time_ymyl_and_run_id(self, store, embedder, history):
        _seed(
            store,
            embedder,
            "alice-1",
            "alice takes warfarin",
            user_id="alice",
            run_id="run-42",
            tier="fact",
            ymyl_category="health",
            event_time="2024-03-01T00:00:00+00:00",
        )

        pipeline = _pipeline(
            StubExtractor([Fact(content="alice takes warfarin 5mg", importance=8.0)]),
            StubResolver([
                ActionItem(
                    action=MemoryAction.UPDATE,
                    fact="alice takes warfarin 5mg",
                    target_id="alice-1",
                    importance=8.0,
                )
            ]),
            embedder,
            store,
            history,
        )

        pipeline.process("alice takes warfarin 5mg", user_id="alice")

        meta = store.get("alice-1")[1]
        assert meta["content"] == "alice takes warfarin 5mg"
        assert meta["ymyl_category"] == "health", "decay immunity was dropped by an update"
        assert meta["event_time"] == "2024-03-01T00:00:00+00:00", "event_time was wiped"
        assert meta["run_id"] == "run-42", "run_id was wiped"


class TestFilteredSearchRecall:
    """faiss_store.py:89 - fixed top_k*3 over-fetch starved minority tenants."""

    def test_minority_tenant_gets_all_matches(self, store, embedder):
        for i in range(200):
            _seed(store, embedder, f"bob-{i}", f"bob fact number {i}", user_id="bob")
        for i in range(5):
            _seed(store, embedder, f"alice-{i}", f"alice fact number {i}", user_id="alice")

        results = store.search(
            vector=embedder.embed("alice fact number 0"),
            top_k=5,
            filters={"user_id": "alice"},
        )

        assert len(results) == 5, (
            f"alice has 5 memories and count() agrees, but search returned {len(results)}"
        )
        assert all(r[2]["user_id"] == "alice" for r in results)

    def test_filter_result_count_matches_count_api(self, store, embedder):
        for i in range(120):
            _seed(store, embedder, f"bob-{i}", f"bob fact number {i}", user_id="bob")
        for i in range(3):
            _seed(store, embedder, f"alice-{i}", f"alice fact number {i}", user_id="alice")

        assert store.count(filters={"user_id": "alice"}) == 3
        results = store.search(
            vector=embedder.embed("alice fact number 1"),
            top_k=10,
            filters={"user_id": "alice"},
        )
        assert len(results) == 3


class TestYMYLDecayImmunity:
    """memory.py:438 - ttl_days hard-filtered YMYL rows before scoring."""

    def test_ttl_does_not_evict_ymyl_memories(self, tmp_dir, embedder):
        store = FAISSVectorStore(VectorStoreConfig(path=f"{tmp_dir}/v"), dimensions=32)
        config = MemoryConfig(history_db_path=f"{tmp_dir}/h.db", ttl_days=30)
        mem = WideMemory(
            config=config,
            llm=StubLLM(),
            embedder=embedder,
            vector_store=store,
        )

        old = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        store.insert(
            id="ymyl-1",
            vector=embedder.embed("alice is allergic to penicillin"),
            metadata={
                "content": "alice is allergic to penicillin",
                "user_id": "alice",
                "importance": 9.5,
                "tier": "fact",
                "ymyl_category": "health",
                "created_at": old,
                "updated_at": old,
                "content_hash": "h1",
            },
        )
        store.insert(
            id="trivial-1",
            vector=embedder.embed("alice had pizza on tuesday"),
            metadata={
                "content": "alice had pizza on tuesday",
                "user_id": "alice",
                "importance": 2.0,
                "tier": "fact",
                "created_at": old,
                "updated_at": old,
                "content_hash": "h2",
            },
        )

        found = mem.search("penicillin allergy", user_id="alice", top_k=10)
        ids = [r.memory.id for r in found]

        assert "ymyl-1" in ids, (
            "a YMYL fact older than ttl_days was evicted; decay immunity is the headline promise"
        )
        assert "trivial-1" not in ids, "a non-YMYL fact past its TTL should still be evicted"


class TestEmbedderStoreDimensionAgreement:
    """memory.py:837 - store was sized from config, not from the actual embedder."""

    def test_store_dimensions_follow_the_embedder(self, tmp_dir, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        config = MemoryConfig(
            history_db_path=f"{tmp_dir}/h.db",
            vector_store=VectorStoreConfig(path=f"{tmp_dir}/v"),
        )
        assert config.embedding.dimensions != 768, "fixture assumes a differing config default"

        mem = WideMemory(
            config=config,
            llm=StubLLM(),
            embedder=StubEmbedder(dimensions=768),
        )

        assert mem.vector_store.dimensions == 768, (
            "vector store was sized from config.embedding.dimensions instead of the "
            "embedder actually in use; every add and search would raise"
        )


class TestAgentScope:
    """The scope guard must cover agent_id, not just user_id.

    Found by mutation testing: replacing the stored agent_id with the
    caller's left every test green.
    """

    def test_sibling_agent_cannot_delete_another_agents_memory(
        self, store, embedder, history
    ):
        _seed(
            store, embedder, "a1-1", "alice takes warfarin",
            user_id="alice", agent_id="agent-1",
        )

        pipeline = _pipeline(
            StubExtractor([Fact(content="alice takes aspirin", importance=7.0)]),
            StubResolver([
                ActionItem(
                    action=MemoryAction.DELETE,
                    fact="alice takes aspirin",
                    target_id="a1-1",
                    importance=7.0,
                )
            ]),
            embedder,
            store,
            history,
        )

        pipeline.process("alice takes aspirin", user_id="alice", agent_id="agent-2")

        assert store.get("a1-1") is not None, "agent-2 deleted agent-1's memory"

    def test_update_does_not_restamp_the_owning_agent(self, store, embedder, history):
        _seed(
            store, embedder, "a1-1", "alice takes warfarin",
            user_id="alice", agent_id="agent-1",
        )

        pipeline = _pipeline(
            StubExtractor([Fact(content="alice takes apixaban", importance=8.0)]),
            StubResolver([
                ActionItem(
                    action=MemoryAction.UPDATE,
                    fact="alice takes apixaban",
                    target_id="a1-1",
                    importance=8.0,
                )
            ]),
            embedder,
            store,
            history,
        )

        pipeline.process("alice takes apixaban", user_id="alice", agent_id="agent-1")

        meta = store.get("a1-1")[1]
        assert meta["content"] == "alice takes apixaban"
        assert meta["agent_id"] == "agent-1", "the update re-stamped the owning agent"

    def test_unscoped_agent_still_reaches_the_users_own_memories(
        self, store, embedder, history
    ):
        """agent_id is only enforced when supplied, mirroring _find_existing."""
        _seed(
            store, embedder, "a1-1", "alice takes warfarin",
            user_id="alice", agent_id="agent-1",
        )

        pipeline = _pipeline(
            StubExtractor([Fact(content="alice takes apixaban", importance=8.0)]),
            StubResolver([
                ActionItem(
                    action=MemoryAction.UPDATE,
                    fact="alice takes apixaban",
                    target_id="a1-1",
                    importance=8.0,
                )
            ]),
            embedder,
            store,
            history,
        )

        pipeline.process("alice takes apixaban", user_id="alice")

        meta = store.get("a1-1")[1]
        assert meta["content"] == "alice takes apixaban"
        assert meta["agent_id"] == "agent-1"


class TestCandidateScoping:
    """_find_existing must not hand cross-scope rows to the conflict resolver.

    The store filter cannot carry a None user_id (pgvector renders it as
    `= NULL`, qdrant cannot express it), so the pipeline enforces scope
    itself and this must hold on every backend.
    """

    def test_unscoped_call_sees_only_unscoped_rows(self, store, embedder, history):
        _seed(store, embedder, "alice-1", "shared topic alpha", user_id="alice")
        _seed(store, embedder, "solo-1", "shared topic alpha two")

        captured = {}

        class CapturingResolver:
            def resolve(self, facts, existing, linked_memories=None):
                captured["ids"] = [e.memory.id for e in existing]
                return []

        pipeline = _pipeline(
            StubExtractor([Fact(content="shared topic alpha", importance=5.0)]),
            CapturingResolver(),
            embedder,
            store,
            history,
        )
        pipeline.process("shared topic alpha")

        assert "alice-1" not in captured["ids"], (
            "another user's memory was handed to the conflict resolver, which "
            "both leaks it into the prompt and makes it a mutation target"
        )

    def test_scoped_call_sees_only_its_own_rows(self, store, embedder, history):
        _seed(store, embedder, "alice-1", "shared topic alpha", user_id="alice")
        _seed(store, embedder, "bob-1", "shared topic alpha two", user_id="bob")

        captured = {}

        class CapturingResolver:
            def resolve(self, facts, existing, linked_memories=None):
                captured["ids"] = [e.memory.id for e in existing]
                return []

        pipeline = _pipeline(
            StubExtractor([Fact(content="shared topic alpha", importance=5.0)]),
            CapturingResolver(),
            embedder,
            store,
            history,
        )
        pipeline.process("shared topic alpha", user_id="bob")

        assert "alice-1" not in captured["ids"]

    def test_scope_holds_when_the_backend_ignores_filters(self, store, embedder, history):
        """The pipeline must not delegate its trust boundary to the store.

        A backend that silently drops or mishandles a filter (pgvector
        renders a None as `= NULL`) must not be able to widen scope.
        """
        _seed(store, embedder, "a1-1", "shared topic alpha", user_id="alice", agent_id="agent-1")
        _seed(store, embedder, "a2-1", "shared topic alpha two", user_id="alice", agent_id="agent-2")

        class FilterIgnoringStore:
            def __init__(self, inner):
                self._inner = inner

            def search(self, vector, top_k=10, filters=None):
                return self._inner.search(vector, top_k=top_k, filters=None)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        captured = {}

        class CapturingResolver:
            def resolve(self, facts, existing, linked_memories=None):
                captured["ids"] = [e.memory.id for e in existing]
                return []

        pipeline = _pipeline(
            StubExtractor([Fact(content="shared topic alpha", importance=5.0)]),
            CapturingResolver(),
            embedder,
            FilterIgnoringStore(store),
            history,
        )
        pipeline.process("shared topic alpha", user_id="alice", agent_id="agent-1")

        assert "a2-1" not in captured["ids"], (
            "agent-2's memory reached the resolver because the pipeline trusted "
            "the store to enforce the agent filter"
        )
