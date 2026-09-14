"""The LangChain retriever adapter, exercised through LangChain's own entry points.

Every test goes through `invoke` / `ainvoke` rather than calling
`_get_relevant_documents` directly. The private method is not what a chain
calls, and testing it directly would pass even if the pydantic field wiring
or the Runnable plumbing were broken.
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest

pytest.importorskip("langchain_core", reason="langchain extra not installed")

from widemem.core.memory import WideMemory  # noqa: E402
from widemem.core.types import (  # noqa: E402
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
    RetrievalConfidence,
    RetrievalMode,
    SearchResult,
    VectorStoreConfig,
)
from widemem.integrations.langchain import WidememRetriever  # noqa: E402
from widemem.providers.embeddings.base import BaseEmbedder  # noqa: E402
from widemem.providers.llm.base import BaseLLM  # noqa: E402


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


@pytest.fixture
def memory():
    with tempfile.TemporaryDirectory() as d:
        config = MemoryConfig(
            embedding=EmbeddingConfig(dimensions=32),
            vector_store=VectorStoreConfig(path=f"{d}/vectors"),
            history_db_path=f"{d}/history.db",
            enable_hierarchy=False,
        )
        m = WideMemory(config=config, llm=StubLLM(), embedder=StubEmbedder(32))
        yield m
        m.close()


def _seed(memory, rows):
    assert memory.import_json(json.dumps({"memories": rows})) == len(rows)


# ---------------------------------------------------------------------------
# The chain-facing path
# ---------------------------------------------------------------------------
def test_an_empty_store_returns_no_documents(memory):
    retriever = WidememRetriever(memory=memory)

    assert retriever.invoke("anything") == []


def test_documents_carry_the_memory_and_its_scores(memory):
    _seed(memory, [{"id": "m1", "content": "allergic to penicillin", "user_id": "alice",
                    "importance": 9.0, "ymyl_category": "medical"}])

    docs = WidememRetriever(memory=memory, user_id="alice").invoke("allergies")

    assert len(docs) == 1
    doc = docs[0]
    assert doc.page_content == "allergic to penicillin"
    assert doc.id == "m1"
    assert doc.metadata["memory_id"] == "m1"
    assert doc.metadata["user_id"] == "alice"
    assert doc.metadata["ymyl_category"] == "medical"
    assert doc.metadata["importance"] == 9.0
    assert doc.metadata["created_at"], "a document with no timestamp cannot be reasoned about"
    assert isinstance(doc.metadata["similarity_score"], float)
    assert isinstance(doc.metadata["final_score"], float)


def test_top_k_caps_the_documents_returned(memory):
    _seed(memory, [{"id": f"m{i}", "content": f"fact number {i}"} for i in range(10)])

    assert len(WidememRetriever(memory=memory, top_k=3).invoke("fact")) <= 3
    assert len(WidememRetriever(memory=memory, top_k=7).invoke("fact")) <= 7


def test_the_user_filter_does_not_cross_tenants(memory):
    _seed(memory, [
        {"id": "a1", "content": "alice works at Acme", "user_id": "alice"},
        {"id": "b1", "content": "bob works at Globex", "user_id": "bob"},
    ])

    docs = WidememRetriever(memory=memory, user_id="alice", top_k=10).invoke("work")

    assert {d.metadata["user_id"] for d in docs} == {"alice"}


def test_without_a_user_filter_every_tenant_is_in_scope(memory):
    _seed(memory, [
        {"id": "a1", "content": "alice works at Acme", "user_id": "alice"},
        {"id": "b1", "content": "bob works at Globex", "user_id": "bob"},
    ])

    docs = WidememRetriever(memory=memory, top_k=10).invoke("work")

    assert {d.metadata["user_id"] for d in docs} == {"alice", "bob"}


def test_unicode_content_survives_the_round_trip(memory):
    _seed(memory, [{"id": "u1", "content": "ist in Wien geboren, spricht Deutsch, 好的"}])

    docs = WidememRetriever(memory=memory).invoke("wien")

    assert docs[0].page_content == "ist in Wien geboren, spricht Deutsch, 好的"


# ---------------------------------------------------------------------------
# Confidence gating
# ---------------------------------------------------------------------------
def _fixed_confidence(memory, confidence, monkeypatch):
    """Pin the confidence of whatever search returns, keeping the real results."""
    real = memory.search

    def search(**kwargs):
        result = real(**kwargs)
        return SearchResult(results=list(result), confidence=confidence)

    monkeypatch.setattr(memory, "search", search)


@pytest.mark.parametrize(
    "reported,threshold,expect_documents",
    [
        (RetrievalConfidence.HIGH, RetrievalConfidence.HIGH, True),
        (RetrievalConfidence.MODERATE, RetrievalConfidence.HIGH, False),
        (RetrievalConfidence.MODERATE, RetrievalConfidence.MODERATE, True),
        (RetrievalConfidence.LOW, RetrievalConfidence.MODERATE, False),
        (RetrievalConfidence.NONE, RetrievalConfidence.LOW, False),
        (RetrievalConfidence.LOW, None, True),
    ],
)
def test_confidence_threshold_is_all_or_nothing(memory, monkeypatch, reported, threshold, expect_documents):
    _seed(memory, [{"id": "m1", "content": "lives in Vienna"}])
    _fixed_confidence(memory, reported, monkeypatch)

    docs = WidememRetriever(memory=memory, min_confidence=threshold).invoke("vienna")

    assert bool(docs) is expect_documents


def test_the_reported_confidence_reaches_the_document(memory, monkeypatch):
    _seed(memory, [{"id": "m1", "content": "lives in Vienna"}])
    _fixed_confidence(memory, RetrievalConfidence.MODERATE, monkeypatch)

    docs = WidememRetriever(memory=memory).invoke("vienna")

    assert docs[0].metadata["retrieval_confidence"] == "moderate"


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------
async def test_the_async_path_returns_what_the_sync_path_returns(memory):
    _seed(memory, [{"id": "m1", "content": "lives in Vienna", "user_id": "alice"}])
    retriever = WidememRetriever(memory=memory, user_id="alice")

    sync_docs = retriever.invoke("vienna")
    async_docs = await retriever.ainvoke("vienna")

    assert [d.page_content for d in async_docs] == [d.page_content for d in sync_docs]
    # `final_score` carries a recency term that moves between the two calls,
    # so it is compared approximately and the rest exactly.
    volatile = "final_score"
    for got, want in zip(async_docs, sync_docs):
        assert got.metadata[volatile] == pytest.approx(want.metadata[volatile], rel=1e-6)
        assert {k: v for k, v in got.metadata.items() if k != volatile} == {
            k: v for k, v in want.metadata.items() if k != volatile
        }


async def test_the_async_path_does_not_block_the_event_loop(memory):
    """`search` is synchronous and slow; awaiting it inline stalls the chain.

    Progress is measured across the retriever call specifically, not across
    the whole test: LangChain's own `ainvoke` awaits before reaching the
    handler, so a ticker started beforehand advances either way and proves
    nothing. A blocking implementation lets the ticker gain nothing while
    the search runs; a threaded one lets it keep counting.
    """
    import asyncio
    import time

    _seed(memory, [{"id": "m1", "content": "lives in Vienna"}])
    real = memory.search
    ticks = 0
    running = True
    search_seconds = 0.3
    tick_seconds = 0.01

    def slow_search(**kwargs):
        time.sleep(search_seconds)
        return real(**kwargs)

    memory.search = slow_search

    async def ticker():
        nonlocal ticks
        while running:
            await asyncio.sleep(tick_seconds)
            ticks += 1

    task = asyncio.create_task(ticker())
    await asyncio.sleep(tick_seconds * 2)  # let the ticker reach its loop

    before = ticks
    await WidememRetriever(memory=memory).ainvoke("vienna")
    during = ticks - before

    running = False
    await task

    # ~30 ticks fit inside the search window; anything above a handful proves
    # the loop kept running. A blocking implementation scores 0.
    assert during >= 5, f"the event loop advanced {during} ticks while the retriever ran"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, -1])
def test_a_top_k_below_one_is_refused(memory, bad):
    with pytest.raises(ValueError):
        WidememRetriever(memory=memory, top_k=bad)


def test_the_retrieval_mode_reaches_search(memory, monkeypatch):
    seen = {}
    real = memory.search

    def search(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(memory, "search", search)
    _seed(memory, [{"id": "m1", "content": "lives in Vienna"}])

    WidememRetriever(memory=memory, retrieval_mode=RetrievalMode.DEEP, top_k=4).invoke("vienna")

    assert seen["mode"] is RetrievalMode.DEEP
    assert seen["top_k"] == 4


def test_it_is_a_real_langchain_retriever(memory):
    """A chain accepts it because of what it inherits, not what it looks like."""
    from langchain_core.retrievers import BaseRetriever

    assert isinstance(WidememRetriever(memory=memory), BaseRetriever)


def test_every_confidence_level_can_be_compared():
    """An unranked level would raise KeyError inside a chain, not return a result.

    The rank table is hand-written because the enum has no ordering of its
    own, so adding a member to `RetrievalConfidence` must not leave it out.
    """
    from widemem.integrations.langchain.retriever import _CONFIDENCE_RANK

    unranked = [level.value for level in RetrievalConfidence if level not in _CONFIDENCE_RANK]
    assert not unranked, f"RetrievalConfidence members with no rank: {unranked}"
