from __future__ import annotations

import asyncio
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pytest

from widemem.core._time import as_utc
from widemem.core.memory import WideMemory
from widemem.core.types import (
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
    MemoryTier,
    ScoringConfig,
    VectorStoreConfig,
    YMYLConfig,
)
from widemem.providers.embeddings.base import BaseEmbedder
from widemem.providers.llm.base import BaseLLM
from widemem.retrieval import temporal as temporal_mod
from widemem.scoring.decay import compute_recency_score
from widemem.scoring.importance import normalize_importance
from widemem.scoring.topics import get_topic_boost
from widemem.scoring.ymyl import classify_ymyl_detailed
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


class SlowEmbedder(MockEmbedder):
    def __init__(self, dimensions: int = 64, delay_s: float = 0.20) -> None:
        super().__init__(dimensions=dimensions)
        self.delay_s = delay_s

    def _embed(self, text: str) -> list[float]:
        time.sleep(self.delay_s)
        return super()._embed(text)


def _build_memory(
    *,
    embedder_factory: Callable[[], BaseEmbedder] | None = None,
    config: MemoryConfig | None = None,
) -> WideMemory:
    tmp_dir = tempfile.TemporaryDirectory()
    cfg = config or MemoryConfig(history_db_path=f"{tmp_dir.name}/history.db")
    vector_store = FAISSVectorStore(VectorStoreConfig(path=f"{tmp_dir.name}/vectors"), dimensions=64)
    mem = WideMemory(
        config=cfg,
        llm=MockLLM(),
        embedder=embedder_factory() if embedder_factory else MockEmbedder(dimensions=64),
        vector_store=vector_store,
    )
    mem._tmp_dir = tmp_dir  # keep temp dir alive for test lifetime
    return mem


def _seed(mem: WideMemory, contents: list[str], user_id: str = "alice") -> None:
    now = datetime.now(timezone.utc).isoformat()
    for i, content in enumerate(contents):
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
                "importance": float((i % 10) + 1),
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
    _seed(mem, ["Alice lives in Berlin", "Alice loves climbing", "Alice works as an engineer"])

    async def _first():
        async for item in mem.search_stream("where does alice live", user_id="alice"):
            return item
        return None

    first = asyncio.run(_first())
    assert first is not None


def test_search_stream_basic_concurrency() -> None:
    mem = _build_memory()
    _seed(mem, ["Alice lives in Berlin", "Alice loves climbing", "Alice works as an engineer"])

    async def _collect_ids() -> list[str]:
        ids = []
        async for item in mem.search_stream("alice", user_id="alice"):
            ids.append(item.memory.id)
        return ids

    async def _run() -> tuple[list[str], list[str]]:
        return await asyncio.gather(_collect_ids(), _collect_ids())

    ids_a, ids_b = asyncio.run(_run())
    assert len(ids_a) > 0
    assert ids_a == ids_b


def test_search_stream_returns_same_candidates_as_search() -> None:
    mem = _build_memory()
    _seed(
        mem,
        [
            "Alice lives in Berlin",
            "Alice loves climbing",
            "Alice works as an engineer",
            "Alice moved from Paris",
            "Alice likes coffee",
        ],
    )

    async def _collect_ids() -> list[str]:
        ids = []
        async for item in mem.search_stream("tell me about alice", user_id="alice", top_k=5):
            ids.append(item.memory.id)
        return ids

    streamed_ids = asyncio.run(_collect_ids())
    search_ids = [r.memory.id for r in mem.search("tell me about alice", user_id="alice", top_k=5)]
    assert set(streamed_ids) == set(search_ids)


def test_search_scoring_refactor_keeps_search_identical() -> None:
    mem = _build_memory(
        config=MemoryConfig(
            history_db_path=":memory:",
            enable_hierarchy=False,
            parse_temporal_hints=False,
            scoring=ScoringConfig(),
        )
    )
    _seed(
        mem,
        [
            "Alice profile one",
            "Alice profile two",
            "Alice profile three",
            "Alice profile four",
            "Alice profile five",
            "Alice profile six",
        ],
    )

    current = mem.search("alice profile summary", user_id="alice", top_k=6)

    def legacy_score_and_rank(*args, **kwargs):
        # Snapshot of pre-refactor scoring loop, intentionally inline.
        results = kwargs["results"]
        config = kwargs["config"]
        now = kwargs.get("now")
        time_after = kwargs.get("time_after")
        time_before = kwargs.get("time_before")
        topic_weights = kwargs.get("topic_weights")
        ymyl_config = kwargs.get("ymyl_config")
        similarity_first = kwargs.get("similarity_first", False)
        similarity_boost = kwargs.get("similarity_boost", 0.15)
        temporal_boost_window = kwargs.get("temporal_boost_window")
        temporal_boost_strength = kwargs.get("temporal_boost_strength", 0.10)

        now = as_utc(now) if now is not None else datetime.now(timezone.utc)
        time_after = as_utc(time_after) if time_after is not None else None
        time_before = as_utc(time_before) if time_before is not None else None
        ymyl_config = ymyl_config or YMYLConfig()

        boost_after = None
        boost_before = None
        if temporal_boost_window is not None:
            raw_after, raw_before = temporal_boost_window
            boost_after = as_utc(raw_after) if raw_after is not None else None
            boost_before = as_utc(raw_before) if raw_before is not None else None

        scored = []
        for result in results:
            created_at = as_utc(result.memory.created_at)
            if time_after and created_at < time_after:
                continue
            if time_before and created_at > time_before:
                continue

            content = result.memory.content
            mem_is_ymyl_strong = False
            if ymyl_config.enabled:
                if result.memory.ymyl_category is not None:
                    mem_is_ymyl_strong = True
                else:
                    ymyl_result = classify_ymyl_detailed(content, ymyl_config)
                    mem_is_ymyl_strong = ymyl_result is not None and ymyl_result.is_strong

            if mem_is_ymyl_strong and ymyl_config.decay_immune:
                recency = 1.0
            else:
                recency = compute_recency_score(
                    created_at=created_at,
                    now=now,
                    decay_function=config.decay_function,
                    decay_rate=config.decay_rate,
                )

            importance = normalize_importance(result.memory.importance)
            final = (
                config.similarity_weight * result.similarity_score
                + config.importance_weight * importance
                + config.recency_weight * recency
            )

            if temporal_boost_window is not None and (boost_after or boost_before):
                in_window = True
                if boost_after and created_at < boost_after:
                    in_window = False
                if boost_before and created_at > boost_before:
                    in_window = False
                if in_window:
                    final += temporal_boost_strength

            if topic_weights:
                final *= get_topic_boost(content, topic_weights)

            result.temporal_score = recency
            result.importance_score = importance
            result.final_score = final
            scored.append(result)

        scored.sort(key=lambda r: r.final_score, reverse=True)
        if similarity_first and len(scored) > 5:
            by_sim = sorted(scored, key=lambda r: r.similarity_score, reverse=True)
            top_sim_ids = {id(r) for r in by_sim[:5]}
            top_final = scored[0].final_score if scored else 1.0
            for r in scored:
                if id(r) in top_sim_ids:
                    r.final_score += top_final * similarity_boost
            scored.sort(key=lambda r: r.final_score, reverse=True)
        return scored

    import widemem.core.memory as memory_mod
    monkey = pytest.MonkeyPatch()
    monkey.setattr(memory_mod, "score_and_rank", legacy_score_and_rank)
    try:
        legacy = mem.search("alice profile summary", user_id="alice", top_k=6)
    finally:
        monkey.undo()

    # FAISS computes the inner-product similarity with non-bit-reproducible
    # parallel reductions, so final_score carries ~1e-8 run-to-run noise.
    # Compare with a tolerance well above that floor but far below the real
    # score gaps (~1e-3 here), so equivalence is still meaningfully checked.
    assert [r.memory.id for r in current] == [r.memory.id for r in legacy]
    assert [r.final_score for r in current] == pytest.approx(
        [r.final_score for r in legacy], abs=1e-6
    )


def test_search_stream_does_not_block_event_loop() -> None:
    mem = _build_memory(embedder_factory=lambda: SlowEmbedder(dimensions=64, delay_s=0.20))
    _seed(mem, ["Alice lives in Berlin", "Alice loves climbing", "Alice works as an engineer"])

    async def _consume() -> None:
        async for _ in mem.search_stream("alice", user_id="alice", top_k=1):
            break

    async def _ticker() -> int:
        ticks = 0
        end = asyncio.get_running_loop().time() + 0.35
        while asyncio.get_running_loop().time() < end:
            ticks += 1
            await asyncio.sleep(0.01)
        return ticks

    async def _run() -> tuple[int, None]:
        return await asyncio.gather(_ticker(), _consume())

    ticks, _ = asyncio.run(_run())
    assert ticks >= 10


def test_search_stream_first_result_before_full_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    mem = _build_memory()
    contents = [f"Alice memory {i}" for i in range(64)]
    _seed(mem, contents)

    original_score_candidate = temporal_mod.score_candidate

    def slow_score_candidate(*args, **kwargs):
        time.sleep(0.01)
        return original_score_candidate(*args, **kwargs)

    import widemem.core.memory as memory_mod
    monkeypatch.setattr(temporal_mod, "score_candidate", slow_score_candidate)
    monkeypatch.setattr(memory_mod, "score_candidate", slow_score_candidate)

    full_start = time.perf_counter()
    _ = mem.search("alice memory", user_id="alice", top_k=50)
    full_elapsed = time.perf_counter() - full_start

    async def _first_latency() -> float:
        start = time.perf_counter()
        async for _ in mem.search_stream("alice memory", user_id="alice", top_k=50):
            return time.perf_counter() - start
        return time.perf_counter() - start

    first_elapsed = asyncio.run(_first_latency())

    assert first_elapsed < full_elapsed
