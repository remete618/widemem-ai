"""Every state change to a stored memory must produce a history entry.

Four public methods wrote to the vector store without one: `delete()`,
`pin()`, `import_json()` and `backfill_entities()`. The behavioural tests
below pin each of those paths. `test_no_unlogged_mutation_site` is the part
that keeps them pinned: it reads `core/memory.py` and fails when a method
mutates the store without also writing to the history store, so a new write
path cannot land unlogged.
"""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from widemem.core.memory import WideMemory
from widemem.core.types import (
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

ROOT = Path(__file__).resolve().parent.parent
MUTATORS = ("insert", "update", "delete")
LOG_CALLS = ("log", "log_many")


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


@pytest.fixture
def mem():
    with tempfile.TemporaryDirectory() as d:
        config = MemoryConfig(
            embedding=EmbeddingConfig(dimensions=32),
            vector_store=VectorStoreConfig(path=f"{d}/vectors"),
            history_db_path=f"{d}/history.db",
            enable_hierarchy=False,
        )
        m = WideMemory(config=config, llm=StubLLM(), embedder=StubEmbedder(32))
        m.pipeline.extractor = StubExtractor(
            [Fact(content="allergic to penicillin", importance=8.0, ymyl_category="medical")]
        )
        yield m
        m.close()


def _actions(mem, memory_id):
    return [e.action for e in mem.get_history(memory_id)]


# ---------------------------------------------------------------------------
# Behaviour: each write path leaves a trace
# ---------------------------------------------------------------------------
def test_delete_is_logged_with_the_content_it_removed(mem):
    memory_id = mem.pin(text="allergic to penicillin", user_id="alice").memories[0].id

    mem.delete(memory_id)

    entries = mem.get_history(memory_id)
    deletes = [e for e in entries if e.action == MemoryAction.DELETE]
    assert len(deletes) == 1, f"delete() left no history entry; got {_actions(mem, memory_id)}"
    assert deletes[0].old_content == "allergic to penicillin", (
        "a delete entry that does not carry the removed content cannot "
        "reconstruct the record it destroyed"
    )


def test_delete_of_an_absent_memory_logs_nothing(mem):
    mem.delete("no-such-memory")
    assert mem.get_history("no-such-memory") == []


def test_pin_logs_the_importance_write(mem):
    memory_id = mem.pin(text="allergic to penicillin", user_id="alice").memories[0].id

    actions = _actions(mem, memory_id)
    assert MemoryAction.ADD in actions
    assert MemoryAction.UPDATE in actions, (
        f"pin() raised importance without a history entry; got {actions}"
    )


def test_import_logs_one_entry_per_imported_memory(mem):
    payload = json.dumps(
        {
            "memories": [
                {"id": "imported-1", "content": "takes warfarin daily", "user_id": "alice"},
                {"id": "imported-2", "content": "lives in Vienna", "user_id": "alice"},
            ]
        }
    )

    assert mem.import_json(payload) == 2

    for memory_id, content in (("imported-1", "takes warfarin daily"), ("imported-2", "lives in Vienna")):
        entries = mem.get_history(memory_id)
        assert [e.action for e in entries] == [MemoryAction.ADD], (
            f"import_json inserted {memory_id} with no history entry"
        )
        assert entries[0].new_content == content


def test_backfill_entities_logs_the_rows_it_rewrites(mem):
    # Content chosen so extract_entities returns a non-empty list; without
    # that the backfill skips the row and the assertion below is vacuous.
    payload = json.dumps(
        {"memories": [{"id": "backfill-1", "content": "seen at Vienna General Hospital"}]}
    )
    assert mem.import_json(payload) == 1
    before = len(mem.get_history("backfill-1"))

    assert mem.backfill_entities() == 1, "fixture did not exercise the backfill path"

    entries = mem.get_history("backfill-1")
    assert len(entries) == before + 1, (
        "backfill_entities rewrote a stored row without a history entry"
    )
    assert entries[-1].action == MemoryAction.UPDATE


# ---------------------------------------------------------------------------
# Structure: a new write path cannot land unlogged
# ---------------------------------------------------------------------------
def _mutating_methods_without_a_log_call() -> list[str]:
    """Return methods in core/memory.py that write to the vector store but
    never touch the history store.

    Delegation counts: a method that hands the write to `self.pipeline` is
    logged inside the pipeline, so only direct `self.vector_store.<mutator>`
    calls are examined here.
    """
    tree = ast.parse((ROOT / "widemem" / "core" / "memory.py").read_text(encoding="utf-8"))
    offenders = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        mutates = False
        logs = False
        for call in ast.walk(node):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            target = call.func
            owner = target.value
            if (
                target.attr in MUTATORS
                and isinstance(owner, ast.Attribute)
                and owner.attr == "vector_store"
            ):
                mutates = True
            if (
                target.attr in LOG_CALLS
                and isinstance(owner, ast.Attribute)
                and owner.attr == "_history_store"
            ):
                logs = True
        if mutates and not logs:
            offenders.append(node.name)

    return offenders


def test_no_unlogged_mutation_site():
    offenders = _mutating_methods_without_a_log_call()
    assert not offenders, (
        "these methods write to the vector store without a history entry: "
        f"{sorted(offenders)}. Every state change to a stored memory is "
        "claimed to be logged (README 'History & Audit Trail'); either log "
        "the write or change the claim."
    )
