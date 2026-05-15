from __future__ import annotations

import asyncio
import tempfile
import uuid
from datetime import datetime, timezone

import numpy as np

from widemem.core.memory import WideMemory
from widemem.core.types import EmbeddingConfig, LLMConfig, MemoryConfig, MemoryTier, VectorStoreConfig
from widemem.providers.embeddings.base import BaseEmbedder
from widemem.providers.llm.base import BaseLLM
from widemem.storage.vector.faiss_store import FAISSVectorStore


class MockLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig())

    def _generate(self, prompt: str, system: str | None = None) -> str:
        return "{}"

    def _generate_json(self, prompt: str, system: str | None = None) -> dict:
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


def _build_memory() -> WideMemory:
    tmp_dir = tempfile.TemporaryDirectory()
    config = MemoryConfig(history_db_path=f"{tmp_dir.name}/history.db")
    vector_store = FAISSVectorStore(VectorStoreConfig(path=f"{tmp_dir.name}/vectors"), dimensions=64)
    mem = WideMemory(
        config=config,
        llm=MockLLM(),
        embedder=MockEmbedder(dimensions=64),
        vector_store=vector_store,
    )
    mem._tmp_dir = tmp_dir  # keep temp dir alive for test lifetime
    return mem


def _seed(mem: WideMemory, user_id: str = "alice") -> None:
    now = datetime.now(timezone.utc).isoformat()
    for content in (
        "Alice lives in Berlin",
        "Alice loves climbing",
        "Alice works as an engineer",
    ):
        vector = mem.embedder.embed(content)
        mem.vector_store.insert(
            id=str(uuid.uuid4()),
            vector=vector,
            metadata={
                "content": content,
                "user_id": user_id,
                "agent_id": None,
                "run_id": None,
                "tier": MemoryTier.FACT.value,
                "importance": 7.0,
                "content_hash": "",
                "created_at": now,
                "updated_at": now,
            },
        )


def test_search_stream_empty_store() -> None:
    mem = _build_memory()

    async def _collect() -> list:
        results = []
        async for item in mem.search_stream("anything", user_id="alice"):
            results.append(item)
        return results

    streamed = asyncio.run(_collect())
    assert streamed == []


def test_search_stream_early_termination() -> None:
    mem = _build_memory()
    _seed(mem)

    async def _first():
        async for item in mem.search_stream("where does alice live", user_id="alice"):
            return item
        return None

    first = asyncio.run(_first())
    baseline = mem.search("where does alice live", user_id="alice")
    assert first is not None
    assert len(baseline) > 0
    assert first.memory.id == baseline[0].memory.id


def test_search_stream_basic_concurrency() -> None:
    mem = _build_memory()
    _seed(mem)

    async def _collect_ids() -> list[str]:
        ids = []
        async for item in mem.search_stream("alice", user_id="alice"):
            ids.append(item.memory.id)
        return ids

    async def _run() -> tuple[list[str], list[str]]:
        return await asyncio.gather(_collect_ids(), _collect_ids())

    ids_a, ids_b = asyncio.run(_run())
    assert ids_a == ids_b
    assert len(ids_a) > 0


def test_search_stream_order_matches_search() -> None:
    mem = _build_memory()
    _seed(mem)

    async def _collect_ids() -> list[str]:
        ids = []
        async for item in mem.search_stream("tell me about alice", user_id="alice"):
            ids.append(item.memory.id)
        return ids

    streamed_ids = asyncio.run(_collect_ids())
    search_ids = [r.memory.id for r in mem.search("tell me about alice", user_id="alice")]
    assert streamed_ids == search_ids
