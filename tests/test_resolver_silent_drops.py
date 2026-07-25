"""Regression tests for silent fact drops in the batch conflict resolver.

Drop paths reproduced here (found via benchmark/results/extraction_grid):
1. Resolver labels a NEW fact "none" with no target -> fact must be stored,
   not silently discarded.
2. Resolver labels a complementary fact UPDATE -> both facts must survive
   unless the new fact genuinely supersedes or contradicts the old one.
3. Genuine contradiction UPDATE -> overwrite is allowed but must leave an
   audit history entry carrying the old content.
"""

from __future__ import annotations

import tempfile

from tests.test_consolidation import CountingLLM, MockEmbedder, MockExtractor, _make_existing
from widemem.conflict.batch_resolver import BatchConflictResolver, _supersedes
from widemem.core.memory import WideMemory
from widemem.core.types import (
    EmbeddingConfig,
    Fact,
    MemoryAction,
    MemoryConfig,
    VectorStoreConfig,
)


def _make_flat_memory(tmp_dir: str, llm, embedder) -> WideMemory:
    config = MemoryConfig(
        embedding=EmbeddingConfig(dimensions=embedder.config.dimensions),
        history_db_path=f"{tmp_dir}/history.db",
        vector_store=VectorStoreConfig(provider="faiss", path=f"{tmp_dir}/vectors"),
        enable_fact_consolidation=False,
    )
    return WideMemory(config=config, llm=llm, embedder=embedder)


# ---------------------------------------------------------------------------
# Resolver-level: NONE semantics
# ---------------------------------------------------------------------------

def test_flat_none_without_target_becomes_add() -> None:
    llm = CountingLLM(responses=[
        {"actions": [{"fact_index": 0, "action": "none", "target_id": None, "importance": 5}]}
    ])
    resolver = BatchConflictResolver(llm)
    existing = [_make_existing("mem-a", "Likes tea")]
    facts = [Fact(content="Owns a red bicycle", importance=5.0)]

    actions = resolver.resolve(facts, existing)

    assert len(actions) == 1
    assert actions[0].action == MemoryAction.ADD
    assert actions[0].fact == "Owns a red bicycle"


def test_flat_none_with_target_is_justified_skip() -> None:
    llm = CountingLLM(responses=[
        {"actions": [{"fact_index": 0, "action": "none", "target_id": 1, "importance": 5}]}
    ])
    resolver = BatchConflictResolver(llm)
    existing = [_make_existing("mem-a", "Likes tea")]
    facts = [Fact(content="Enjoys drinking tea", importance=5.0)]

    actions = resolver.resolve(facts, existing)

    assert len(actions) == 1
    assert actions[0].action == MemoryAction.NONE
    assert actions[0].target_id == "mem-a"


def test_linked_none_without_target_and_no_duplicate_becomes_add() -> None:
    llm = CountingLLM(responses=[
        {"actions": [{"fact_index": 0, "action": "none", "target_id": None, "importance": 5}]}
    ])
    resolver = BatchConflictResolver(llm)
    existing = [_make_existing("mem-a", "Likes tea")]
    facts = [Fact(content="Owns a red bicycle", importance=5.0)]

    actions = resolver.resolve(facts, existing, [[existing[0]]])

    assert len(actions) == 1
    assert actions[0].action == MemoryAction.ADD


def test_linked_none_with_exact_duplicate_keeps_justified_skip() -> None:
    llm = CountingLLM(responses=[
        {"actions": [{"fact_index": 0, "action": "none", "target_id": None, "importance": 5}]}
    ])
    resolver = BatchConflictResolver(llm)
    existing = [_make_existing("mem-a", "Likes tea")]
    facts = [Fact(content="Likes tea", importance=5.0)]

    actions = resolver.resolve(facts, existing, [[existing[0]]])

    assert len(actions) == 1
    assert actions[0].action == MemoryAction.NONE
    assert actions[0].target_id == "mem-a"


# ---------------------------------------------------------------------------
# Resolver-level: UPDATE semantics
# ---------------------------------------------------------------------------

def test_complementary_update_falls_back_to_add() -> None:
    llm = CountingLLM(responses=[
        {"actions": [{
            "fact_index": 0, "action": "update", "target_id": 1,
            "update_kind": "refinement", "importance": 6,
        }]}
    ])
    resolver = BatchConflictResolver(llm)
    existing = [_make_existing("mem-a", "Joanna baked a chocolate cake in May 2022")]
    facts = [Fact(content="Joanna used strawberry filling in the cake", importance=6.0)]

    actions = resolver.resolve(facts, existing)

    assert len(actions) == 1
    assert actions[0].action == MemoryAction.ADD
    assert actions[0].target_id is None


def test_superseding_refinement_update_goes_through() -> None:
    llm = CountingLLM(responses=[
        {"actions": [{
            "fact_index": 0, "action": "update", "target_id": 1,
            "update_kind": "refinement", "importance": 7,
        }]}
    ])
    resolver = BatchConflictResolver(llm)
    existing = [_make_existing("mem-a", "Works at Google")]
    facts = [Fact(content="Works as a senior engineer at Google", importance=7.0)]

    actions = resolver.resolve(facts, existing)

    assert len(actions) == 1
    assert actions[0].action == MemoryAction.UPDATE
    assert actions[0].target_id == "mem-a"


def test_contradiction_update_goes_through() -> None:
    llm = CountingLLM(responses=[
        {"actions": [{
            "fact_index": 0, "action": "update", "target_id": 1,
            "update_kind": "contradiction", "contradicts": "Lives in Boston",
            "importance": 8,
        }]}
    ])
    resolver = BatchConflictResolver(llm)
    existing = [_make_existing("mem-a", "Lives in Boston")]
    facts = [Fact(content="Moved to San Francisco", importance=8.0)]

    actions = resolver.resolve(facts, existing)

    assert len(actions) == 1
    assert actions[0].action == MemoryAction.UPDATE
    assert actions[0].target_id == "mem-a"


def test_contradiction_without_verbatim_quote_falls_back_to_add() -> None:
    llm = CountingLLM(responses=[
        {"actions": [{
            "fact_index": 0, "action": "update", "target_id": 1,
            "update_kind": "contradiction", "importance": 8,
        }]}
    ])
    resolver = BatchConflictResolver(llm)
    existing = [_make_existing("mem-a", "Caroline made a stained glass window for a local church")]
    facts = [Fact(content="Caroline is a transgender woman", importance=8.0)]

    actions = resolver.resolve(facts, existing)

    assert len(actions) == 1
    assert actions[0].action == MemoryAction.ADD


def test_contradiction_with_fragment_quote_keeps_multi_detail_memory() -> None:
    llm = CountingLLM(responses=[
        {"actions": [{
            "fact_index": 0, "action": "update", "target_id": 1,
            "update_kind": "contradiction",
            "contradicts": "two pets named Luna and Oliver",
            "importance": 6,
        }]}
    ])
    resolver = BatchConflictResolver(llm)
    existing = [_make_existing(
        "mem-a",
        "Melanie has two pets named Luna and Oliver who are sweet and playful "
        "and liven up the house",
    )]
    facts = [Fact(content="Melanie has a cat named Oliver", importance=6.0)]

    actions = resolver.resolve(facts, existing)

    assert len(actions) == 1
    assert actions[0].action == MemoryAction.ADD


def test_contradiction_with_full_quote_on_composite_memory_keeps_both() -> None:
    # gpt-4o-mini copies the whole memory verbatim into "contradicts" while
    # mislabeling complementary facts as contradictions (observed in both
    # validation runs). A composite memory bundles several details and can be
    # contradicted at most in part, so it must never be replaced in place,
    # even with a perfect quote.
    old = ("Caroline made a stained glass window to remind herself and others "
           "about discovering true potential and living their best life")
    llm = CountingLLM(responses=[
        {"actions": [{
            "fact_index": 0, "action": "update", "target_id": 1,
            "update_kind": "contradiction", "contradicts": old,
            "importance": 8,
        }]}
    ])
    resolver = BatchConflictResolver(llm)
    existing = [_make_existing("mem-a", old)]
    facts = [Fact(content="Caroline is a transgender woman", importance=8.0)]

    actions = resolver.resolve(facts, existing)

    assert len(actions) == 1
    assert actions[0].action == MemoryAction.ADD


def test_update_without_update_kind_requires_supersession() -> None:
    llm = CountingLLM(responses=[
        {"actions": [{"fact_index": 0, "action": "update", "target_id": 1, "importance": 6}]}
    ])
    resolver = BatchConflictResolver(llm)
    existing = [_make_existing("mem-a", "Nate plays games on the Gamecube")]
    facts = [Fact(content="Nate plays games on the PC and Playstation", importance=6.0)]

    actions = resolver.resolve(facts, existing)

    assert len(actions) == 1
    assert actions[0].action == MemoryAction.ADD


def test_supersedes_heuristic() -> None:
    assert _supersedes("Works as a senior engineer at Google", "Works at Google")
    assert not _supersedes(
        "Joanna used strawberry filling in the cake",
        "Joanna baked a chocolate cake in May 2022",
    )


# ---------------------------------------------------------------------------
# Pipeline-level: end-to-end drop paths with a mocked resolver LLM
# ---------------------------------------------------------------------------

def test_pipeline_stores_fact_when_resolver_says_bare_none() -> None:
    llm = CountingLLM(responses=[
        {"actions": [{"fact_index": 0, "action": "none", "target_id": None, "importance": 5}]}
    ])
    embedder = MockEmbedder(dimensions=8)
    extractor = MockExtractor()

    with tempfile.TemporaryDirectory() as tmp_dir:
        memory = _make_flat_memory(tmp_dir, llm=llm, embedder=embedder)
        memory.pipeline.extractor = extractor

        extractor.facts_to_return = [Fact(content="User likes green tea", importance=5.0)]
        first = memory.add("t1", user_id="alice")
        extractor.facts_to_return = [Fact(content="User owns a red bicycle", importance=5.0)]
        second = memory.add("t2", user_id="alice")

        assert len(first.memories) == 1
        assert len(second.memories) == 1
        assert memory.count(user_id="alice") == 2
        assert memory.pipeline.stats["added"] == 2
        memory.close()


def test_pipeline_justified_skip_is_logged_and_counted() -> None:
    llm = CountingLLM(responses=[
        {"actions": [{"fact_index": 0, "action": "none", "target_id": 1, "importance": 5}]}
    ])
    embedder = MockEmbedder(dimensions=8)
    extractor = MockExtractor()

    with tempfile.TemporaryDirectory() as tmp_dir:
        memory = _make_flat_memory(tmp_dir, llm=llm, embedder=embedder)
        memory.pipeline.extractor = extractor

        extractor.facts_to_return = [Fact(content="User likes green tea", importance=5.0)]
        first = memory.add("t1", user_id="alice")
        extractor.facts_to_return = [Fact(content="User enjoys green tea", importance=5.0)]
        second = memory.add("t2", user_id="alice")

        assert len(second.memories) == 0
        assert memory.count(user_id="alice") == 1
        assert memory.pipeline.stats["skipped"] == 1

        target_id = first.memories[0].id
        entries = memory.pipeline.history.get_history(target_id)
        skips = [e for e in entries if e.action == MemoryAction.NONE]
        assert len(skips) == 1
        assert skips[0].new_content == "User enjoys green tea"
        assert skips[0].old_content == "User likes green tea"
        memory.close()


def test_pipeline_complementary_update_keeps_both_facts() -> None:
    llm = CountingLLM(responses=[
        {"actions": [{
            "fact_index": 0, "action": "update", "target_id": 1,
            "update_kind": "refinement", "importance": 6,
        }]}
    ])
    embedder = MockEmbedder(dimensions=8)
    extractor = MockExtractor()

    with tempfile.TemporaryDirectory() as tmp_dir:
        memory = _make_flat_memory(tmp_dir, llm=llm, embedder=embedder)
        memory.pipeline.extractor = extractor

        extractor.facts_to_return = [
            Fact(content="Joanna baked a chocolate cake in May 2022", importance=6.0)
        ]
        first = memory.add("t1", user_id="alice")
        extractor.facts_to_return = [
            Fact(content="Joanna used strawberry filling in the cake", importance=6.0)
        ]
        second = memory.add("t2", user_id="alice")

        assert len(second.memories) == 1
        assert memory.count(user_id="alice") == 2
        old = memory.pipeline.vector_store.get(first.memories[0].id)
        assert old is not None
        assert old[1]["content"] == "Joanna baked a chocolate cake in May 2022"
        memory.close()


def test_pipeline_contradiction_update_overwrites_with_history_entry() -> None:
    llm = CountingLLM(responses=[
        {"actions": [{
            "fact_index": 0, "action": "update", "target_id": 1,
            "update_kind": "contradiction", "contradicts": "User lives in Boston",
            "importance": 8,
        }]}
    ])
    embedder = MockEmbedder(dimensions=8)
    extractor = MockExtractor()

    with tempfile.TemporaryDirectory() as tmp_dir:
        memory = _make_flat_memory(tmp_dir, llm=llm, embedder=embedder)
        memory.pipeline.extractor = extractor

        extractor.facts_to_return = [Fact(content="User lives in Boston", importance=8.0)]
        first = memory.add("t1", user_id="alice")
        extractor.facts_to_return = [Fact(content="User moved to San Francisco", importance=8.0)]
        second = memory.add("t2", user_id="alice")

        assert len(second.memories) == 1
        assert memory.count(user_id="alice") == 1
        target_id = first.memories[0].id
        stored = memory.pipeline.vector_store.get(target_id)
        assert stored is not None
        assert stored[1]["content"] == "User moved to San Francisco"

        entries = memory.pipeline.history.get_history(target_id)
        updates = [e for e in entries if e.action == MemoryAction.UPDATE]
        assert len(updates) == 1
        assert updates[0].old_content == "User lives in Boston"
        assert updates[0].new_content == "User moved to San Francisco"
        assert memory.pipeline.stats["updated"] == 1
        memory.close()
