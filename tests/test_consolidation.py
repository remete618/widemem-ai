from __future__ import annotations

import hashlib
import json
import tempfile

import numpy as np

from widemem.conflict.batch_resolver import BatchConflictResolver
from widemem.core.memory import WideMemory
from widemem.core.types import (
    EmbeddingConfig,
    Fact,
    LLMConfig,
    Memory,
    MemoryAction,
    MemoryConfig,
    MemorySearchResult,
    VectorStoreConfig,
)
from widemem.extraction.base import BaseExtractor
from widemem.providers.embeddings.base import BaseEmbedder
from widemem.providers.llm.base import BaseLLM


class CountingLLM(BaseLLM):
    def __init__(self, responses: list[dict]) -> None:
        super().__init__(LLMConfig())
        self.responses = list(responses)
        self.calls = 0
        self.prompts: list[str] = []

    def _generate(self, prompt: str, system: str | None = None) -> str:
        return json.dumps(self._generate_json(prompt, system))

    def _generate_json(self, prompt: str, system: str | None = None) -> dict:
        self.calls += 1
        self.prompts.append(prompt)
        if not self.responses:
            return {"actions": []}
        return self.responses.pop(0)


class MockEmbedder(BaseEmbedder):
    def __init__(self, dimensions: int = 8) -> None:
        super().__init__(EmbeddingConfig(dimensions=dimensions), max_retries=1, retry_delay=0)
        self._vectors: dict[str, list[float]] = {}

    def _embed(self, text: str) -> list[float]:
        if text not in self._vectors:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = []
            for i in range(self.config.dimensions):
                byte = digest[i % len(digest)]
                values.append((byte / 255.0) * 2 - 1)
            vec = np.array(values, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm:
                vec = vec / norm
            self._vectors[text] = vec.tolist()
        return self._vectors[text]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


class MockExtractor(BaseExtractor):
    def __init__(self) -> None:
        self.facts_to_return: list[Fact] = []

    def extract(self, text: str) -> list[Fact]:
        return list(self.facts_to_return)


def _make_existing(memory_id: str, content: str, importance: float = 5.0) -> MemorySearchResult:
    return MemorySearchResult(
        memory=Memory(id=memory_id, content=content, importance=importance),
        similarity_score=0.9,
    )


def _make_memory(
    tmp_dir: str,
    llm: BaseLLM,
    embedder: BaseEmbedder,
) -> WideMemory:
    config = MemoryConfig(
        embedding=EmbeddingConfig(dimensions=embedder.config.dimensions),
        history_db_path=f"{tmp_dir}/history.db",
        vector_store=VectorStoreConfig(provider="faiss", path=f"{tmp_dir}/vectors"),
        enable_fact_consolidation=True,
    )
    return WideMemory(config=config, llm=llm, embedder=embedder)


def test_resolver_uses_linked_candidates_and_prompt_contract() -> None:
    llm = CountingLLM(
        responses=[
            {
                "actions": [
                    {"fact_index": 0, "action": "add", "target_id": None, "importance": 7},
                    {"fact_index": 1, "action": "update", "target_id": 999, "importance": 8},
                    {"fact_index": 2, "action": "delete", "target_id": 3, "importance": 5},
                    {"fact_index": 3, "action": "none", "target_id": None, "importance": 5},
                ]
            }
        ]
    )
    resolver = BatchConflictResolver(llm)

    existing = [
        _make_existing("mem-a", "Lives in Berlin"),
        _make_existing("mem-b", "Works at Google"),
        _make_existing("mem-c", "Likes coffee"),
    ]
    facts = [
        Fact(content="New fact", importance=7.0),
        Fact(content="Works at Google in Zurich", importance=8.0),
        Fact(content="Likes coffee", importance=5.0),
        Fact(content="Already captured", importance=5.0),
    ]
    linked = [
        [existing[0]],
        [existing[1], existing[2]],
        [existing[2]],
        [existing[0]],
    ]

    actions = resolver.resolve(facts, existing, linked)

    assert llm.calls == 1
    assert "linked_memory_ids" in llm.prompts[0]
    assert "cascade tiebreaker" in llm.prompts[0]
    assert [a.action for a in actions] == [
        MemoryAction.ADD,
        MemoryAction.UPDATE,
        MemoryAction.DELETE,
        MemoryAction.NONE,
    ]
    assert actions[1].target_id == "mem-b"
    assert actions[2].target_id == "mem-c"
    assert actions[3].target_id is None


def test_resolver_invalid_update_degrades_to_none_when_unchanged() -> None:
    llm = CountingLLM(
        responses=[
            {
                "actions": [
                    {"fact_index": 0, "action": "update", "target_id": 999, "importance": 7},
                ]
            }
        ]
    )
    resolver = BatchConflictResolver(llm)
    existing = [_make_existing("mem-a", "Lives in Berlin")]
    facts = [Fact(content="Lives in Berlin", importance=7.0)]
    linked = [[existing[0]]]

    actions = resolver.resolve(facts, existing, linked)

    assert llm.calls == 1
    assert len(actions) == 1
    assert actions[0].action == MemoryAction.NONE
    assert actions[0].target_id is None


def test_resolver_invalid_delete_degrades_to_none() -> None:
    llm = CountingLLM(
        responses=[
            {
                "actions": [
                    {"fact_index": 0, "action": "delete", "target_id": 999, "importance": 7},
                ]
            }
        ]
    )
    resolver = BatchConflictResolver(llm)
    existing = [_make_existing("mem-a", "Lives in Berlin")]
    facts = [Fact(content="Lives in Berlin", importance=7.0)]
    linked = [[existing[0]]]

    actions = resolver.resolve(facts, existing, linked)

    assert llm.calls == 1
    assert len(actions) == 1
    assert actions[0].action == MemoryAction.NONE
    assert actions[0].target_id is None


def test_resolver_valid_update_noops_when_content_matches() -> None:
    llm = CountingLLM(
        responses=[
            {
                "actions": [
                    {"fact_index": 0, "action": "update", "target_id": 0, "importance": 7},
                ]
            }
        ]
    )
    resolver = BatchConflictResolver(llm)
    existing = [_make_existing("mem-a", "Lives in Berlin")]
    facts = [Fact(content="Lives in Berlin", importance=7.0)]
    linked = [[existing[0]]]

    actions = resolver.resolve(facts, existing, linked)

    assert llm.calls == 1
    assert len(actions) == 1
    assert actions[0].action == MemoryAction.NONE
    assert actions[0].target_id is None


def test_pipeline_is_idempotent_for_same_input() -> None:
    llm = CountingLLM(
        responses=[
            {
                "actions": [
                    {"fact_index": 0, "action": "update", "target_id": 999, "importance": 7},
                ]
            }
        ]
    )
    embedder = MockEmbedder(dimensions=8)
    extractor = MockExtractor()
    extractor.facts_to_return = [Fact(content="Lives in Berlin", importance=7.0)]

    with tempfile.TemporaryDirectory() as tmp_dir:
        memory = _make_memory(tmp_dir, llm=llm, embedder=embedder)
        memory.pipeline.extractor = extractor

        first = memory.add("Lives in Berlin", user_id="alice")
        second = memory.add("Lives in Berlin", user_id="alice")

        assert len(first.memories) == 1
        assert len(second.memories) == 0
        assert memory.count(user_id="alice") == 1
        assert llm.calls == 1


def test_config_defaults_consolidation_off() -> None:
    assert MemoryConfig().enable_fact_consolidation is False
