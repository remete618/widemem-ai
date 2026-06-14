"""Unit + integration tests for entity-aware additive re-rank."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from widemem.core.memory import WideMemory
from widemem.core.types import (
    EmbeddingConfig,
    LLMConfig,
    Memory,
    MemoryConfig,
    MemorySearchResult,
    MemoryTier,
    VectorStoreConfig,
)
from widemem.providers.embeddings.base import BaseEmbedder
from widemem.providers.llm.base import BaseLLM
from widemem.retrieval.entity_boost import apply_entity_boost
from widemem.storage.vector.faiss_store import FAISSVectorStore


def _r(score: float, entities: list[str], rid: str) -> MemorySearchResult:
    return MemorySearchResult(
        memory=Memory(id=rid, content=rid, entities=entities),
        final_score=score,
    )


def test_noop_when_weight_zero():
    rs = [_r(0.1, ["caroline"], "a"), _r(0.9, [], "b")]
    out = apply_entity_boost(rs, ["caroline"], weight=0.0, attenuation=0.001)
    assert [x.memory.id for x in out] == ["a", "b"]
    assert out[0].final_score == 0.1 and out[1].final_score == 0.9


def test_noop_when_no_query_entities():
    rs = [_r(0.1, ["caroline"], "a")]
    apply_entity_boost(rs, [], weight=0.5, attenuation=0.001)
    assert rs[0].final_score == 0.1


def test_noop_when_no_overlap():
    rs = [_r(0.5, ["berlin"], "a"), _r(0.4, ["sweden"], "b")]
    apply_entity_boost(rs, ["tokyo"], weight=0.5, attenuation=0.001)
    assert rs[0].final_score == 0.5 and rs[1].final_score == 0.4


def test_overlap_boosts_and_resorts():
    rs = [_r(0.50, [], "plain"), _r(0.45, ["caroline"], "entity")]
    out = apply_entity_boost(rs, ["caroline"], weight=0.5, attenuation=0.001)
    # entity match got +0.5*(1/1)/1 = +0.5 -> 0.95, now first
    assert out[0].memory.id == "entity"
    assert abs(out[0].final_score - 0.95) < 1e-9
    assert out[1].final_score == 0.50  # untouched


def test_boost_is_bounded_by_weight():
    rs = [_r(0.0, ["a", "b"], "x")]
    apply_entity_boost(rs, ["a", "b"], weight=0.5, attenuation=0.0)
    assert rs[0].final_score <= 0.5 + 1e-9


def test_attenuation_damps_common_entities():
    common = [_r(0.0, ["x"], f"c{i}") for i in range(10)]
    apply_entity_boost(common, ["x"], weight=1.0, attenuation=0.05)
    rare = [_r(0.0, ["y"], "r0")]
    apply_entity_boost(rare, ["y"], weight=1.0, attenuation=0.05)
    assert common[0].final_score < rare[0].final_score


# ---- integration: behavior-neutral at weight 0, token-neutral when on ----

class _LLM(BaseLLM):
    def __init__(self):
        super().__init__(LLMConfig())

    def _generate(self, p, system=None):
        return "{}"

    def _generate_json(self, p, system=None):
        return {"facts": []}


class _Emb(BaseEmbedder):
    def __init__(self):
        super().__init__(EmbeddingConfig(dimensions=64), max_retries=1, retry_delay=0)

    def _embed(self, text):
        rng = np.random.RandomState(hash(text) % 2**31)
        v = rng.randn(64).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    def _embed_batch(self, texts):
        return [self._embed(t) for t in texts]


def _seed(store, emb):
    for rid, content, ents in [
        ("m1", "Caroline moved to Berlin", ["caroline", "berlin"]),
        ("m2", "the weather was nice", []),
        ("m3", "Melanie went camping", ["melanie"]),
    ]:
        store.insert(rid, emb.embed(content), {
            "content": content, "user_id": "u", "tier": MemoryTier.FACT.value,
            "importance": 5.0, "entities": ents,
        })


def _mk(weight: float):
    d = tempfile.mkdtemp()
    vs = FAISSVectorStore(VectorStoreConfig(path=f"{d}/v"), dimensions=64)
    emb = _Emb()
    _seed(vs, emb)
    cfg = MemoryConfig(
        history_db_path=f"{d}/h.db",
        enable_entity_index=True,
        entity_boost_weight=weight,
    )
    return WideMemory(config=cfg, llm=_LLM(), embedder=emb, vector_store=vs)


def test_weight_zero_matches_baseline():
    base = _mk(0.0).search("Caroline", user_id="u")
    again = _mk(0.0).search("Caroline", user_id="u")
    assert [r.memory.id for r in base] == [r.memory.id for r in again]
    # FAISS similarity uses non-bit-reproducible parallel reductions (~1e-8
    # run-to-run noise), so compare scores with a tolerance, not exact rounding.
    assert [r.final_score for r in base] == pytest.approx(
        [r.final_score for r in again], abs=1e-6
    )


def test_weight_on_lifts_entity_match_token_neutral():
    off = _mk(0.0).search("Caroline", user_id="u")
    on = _mk(0.5).search("Caroline", user_id="u")
    assert len(on) == len(off)  # token-neutral: same number of memories
    off_s = {r.memory.id: r.final_score for r in off}
    on_s = {r.memory.id: r.final_score for r in on}
    # entity-matching memory boosted by ~weight (well above recency jitter)
    assert on_s["m1"] > off_s["m1"] + 0.4
    # non-matching memory gets no entity boost (tolerate ~1e-10 recency
    # jitter from datetime.now() between the two independent searches)
    assert on_s["m2"] == pytest.approx(off_s["m2"], abs=1e-6)
