"""Integration tests for the entity index: flag-off behavior parity,
flag-on storage/reconstruction/round-trip, and the no-LLM no-re-embed
backfill migration."""

from __future__ import annotations

import json
import tempfile

import numpy as np

from widemem.core.memory import WideMemory
from widemem.core.types import (
    EmbeddingConfig,
    Fact,
    LLMConfig,
    MemoryConfig,
    VectorStoreConfig,
)
from widemem.extraction.base import BaseExtractor
from widemem.providers.embeddings.base import BaseEmbedder
from widemem.providers.llm.base import BaseLLM
from widemem.storage.vector.faiss_store import FAISSVectorStore


class MockLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig())
        self.calls = 0

    def _generate(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        return "{}"

    def _generate_json(self, prompt: str, system: str | None = None) -> dict:
        self.calls += 1
        return {"facts": []}


class CountingEmbedder(BaseEmbedder):
    def __init__(self, dimensions: int = 64) -> None:
        super().__init__(EmbeddingConfig(dimensions=dimensions), max_retries=1, retry_delay=0)
        self.calls = 0

    def _embed(self, text: str) -> list[float]:
        self.calls += 1
        rng = np.random.RandomState(hash(text) % 2**31)
        v = rng.randn(self.config.dimensions).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


class MockExtractor(BaseExtractor):
    def __init__(self, facts: list[Fact]) -> None:
        self._facts = facts

    def extract(self, text: str) -> list[Fact]:
        return self._facts


def _mem(enable_entities: bool):
    d = tempfile.mkdtemp()
    cfg = MemoryConfig(
        history_db_path=f"{d}/h.db", enable_entity_index=enable_entities
    )
    m = WideMemory(
        config=cfg,
        llm=MockLLM(),
        embedder=CountingEmbedder(64),
        vector_store=FAISSVectorStore(VectorStoreConfig(path=f"{d}/v"), dimensions=64),
    )
    return m


def test_flag_off_is_behavior_neutral():
    m = _mem(enable_entities=False)
    m.pipeline.extractor = MockExtractor([Fact(content="Caroline moved from Sweden", importance=7.0)])
    m.add("Caroline moved from Sweden", user_id="u")
    results = m.search("Caroline", user_id="u")
    assert len(results) == 1
    assert results[0].memory.entities == []
    raw = json.loads(m.export_json())
    assert "entities" not in raw["memories"][0]


def test_flag_on_stores_and_reconstructs():
    m = _mem(enable_entities=True)
    m.pipeline.extractor = MockExtractor([Fact(content="Caroline moved from Sweden to Berlin", importance=7.0)])
    res = m.add("Caroline moved from Sweden to Berlin", user_id="u")
    mid = res.memories[0].id

    r = m.search("Caroline", user_id="u")[0]
    assert {"caroline", "sweden", "berlin"} <= set(r.memory.entities)
    assert {"caroline", "sweden", "berlin"} <= set(m.get(mid).entities)

    dumped = m.export_json()
    m2 = _mem(enable_entities=True)
    assert m2.import_json(dumped) == 1
    assert {"caroline", "sweden", "berlin"} <= set(m2.search("Caroline", user_id="u")[0].memory.entities)


def test_backfill_no_llm_no_reembed_idempotent():
    m = _mem(enable_entities=False)  # ingested WITHOUT entities
    m.pipeline.extractor = MockExtractor([Fact(content="Melanie visited Lake Tahoe", importance=6.0)])
    m.add("Melanie visited Lake Tahoe", user_id="u")

    # nothing stored entities yet
    assert m.search("Melanie", user_id="u")[0].memory.entities == []

    llm_calls = m.llm.calls
    embed_calls = m.embedder.calls

    n = m.backfill_entities()
    assert n == 1
    assert m.llm.calls == llm_calls, "backfill must not call the LLM"
    assert m.embedder.calls == embed_calls, "backfill must reuse stored vectors, not re-embed"

    ents = set(m.search("Melanie", user_id="u")[0].memory.entities)
    assert {"melanie", "lake tahoe"} <= ents

    assert m.backfill_entities() == 0, "backfill must be idempotent"
