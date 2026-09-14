"""`purge_expired` is the deletion counterpart to the `ttl_days` search filter.

`ttl_days` hides old memories from `search()` and leaves them on disk. These
tests pin the difference, the YMYL carve-out, and the audit entries every
removal is required to leave behind.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from widemem.core.memory import WideMemory
from widemem.core.types import (
    EmbeddingConfig,
    LLMConfig,
    MemoryAction,
    MemoryConfig,
    VectorStoreConfig,
)
from widemem.providers.embeddings.base import BaseEmbedder
from widemem.providers.llm.base import BaseLLM


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


def _ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


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
        yield m
        m.close()


def _seed(mem, rows):
    assert mem.import_json(json.dumps({"memories": rows})) == len(rows)


def test_purge_removes_only_rows_past_the_cutoff(mem):
    _seed(mem, [
        {"id": "old", "content": "used to work at Acme", "created_at": _ago(400)},
        {"id": "fresh", "content": "now works at Globex", "created_at": _ago(3)},
    ])

    assert mem.purge_expired(older_than_days=365) == 1
    assert mem.get("old") is None
    assert mem.get("fresh") is not None


def test_purge_leaves_an_audit_entry_carrying_the_removed_content(mem):
    _seed(mem, [{"id": "old", "content": "used to work at Acme", "created_at": _ago(400)}])

    mem.purge_expired(older_than_days=365)

    deletes = [e for e in mem.get_history("old") if e.action == MemoryAction.DELETE]
    assert len(deletes) == 1, "a retention sweep must be reconstructible from the log"
    assert deletes[0].old_content == "used to work at Acme"


def test_ymyl_rows_survive_by_default_and_go_when_asked(mem):
    _seed(mem, [
        {"id": "allergy", "content": "allergic to penicillin", "created_at": _ago(400),
         "ymyl_category": "medical"},
    ])

    assert mem.purge_expired(older_than_days=365) == 0
    assert mem.get("allergy") is not None, "decay immunity must not be silently overridden"

    assert mem.purge_expired(older_than_days=365, include_ymyl=True) == 1
    assert mem.get("allergy") is None


def test_dry_run_counts_without_removing(mem):
    _seed(mem, [{"id": "old", "content": "used to work at Acme", "created_at": _ago(400)}])

    assert mem.purge_expired(older_than_days=365, dry_run=True) == 1
    assert mem.get("old") is not None
    assert mem.get_history("old") == [
        e for e in mem.get_history("old") if e.action == MemoryAction.ADD
    ]


def test_purge_is_scoped_to_one_user_when_asked(mem):
    _seed(mem, [
        {"id": "alice-old", "content": "alice used to work at Acme",
         "user_id": "alice", "created_at": _ago(400)},
        {"id": "bob-old", "content": "bob used to work at Acme",
         "user_id": "bob", "created_at": _ago(400)},
    ])

    assert mem.purge_expired(older_than_days=365, user_id="alice") == 1
    assert mem.get("alice-old") is None
    assert mem.get("bob-old") is not None, "a scoped purge must not cross tenants"


def test_zero_days_purges_everything_non_ymyl(mem):
    _seed(mem, [
        {"id": "just-now", "content": "typed this second"},
        {"id": "allergy", "content": "allergic to penicillin", "ymyl_category": "medical"},
    ])

    assert mem.purge_expired(older_than_days=0) == 1
    assert mem.get("just-now") is None
    assert mem.get("allergy") is not None


def test_negative_days_is_refused(mem):
    with pytest.raises(ValueError):
        mem.purge_expired(older_than_days=-1)


def test_purge_on_an_empty_store_is_a_no_op(mem):
    assert mem.purge_expired(older_than_days=1) == 0


def test_unparseable_created_at_is_kept_not_purged(mem):
    _seed(mem, [{"id": "broken", "content": "timestamp is garbage", "created_at": "not-a-date"}])

    assert mem.purge_expired(older_than_days=365) == 0
    assert mem.get("broken") is not None, (
        "an unreadable timestamp must fail closed: deleting a record whose age "
        "cannot be established is unrecoverable"
    )


# ---------------------------------------------------------------------------
# The sweep must not be truncated by its own request size
# ---------------------------------------------------------------------------
def _record_sizing(mem):
    """Wrap the store so the test can see what purge asked it for."""
    seen = {"counts": [], "requests": []}
    real_list, real_count = mem.vector_store.list_all, mem.vector_store.count

    def count(filters=None):
        n = real_count(filters=filters)
        seen["counts"].append((filters, n))
        return n

    def list_all(filters=None, max_results=1000):
        seen["requests"].append((filters, max_results))
        return real_list(filters=filters, max_results=max_results)

    mem.vector_store.count = count
    mem.vector_store.list_all = list_all
    return seen


def test_the_request_size_is_derived_from_the_store_not_a_constant(mem):
    """A fixed cap silently shortened the sweep on any larger store.

    Every backend honours `max_results`, so requesting a constant meant the
    tail was never examined while the returned count still read as a complete
    purge. Asserting merely that the request is "large" would not catch it:
    the old 1,000,000 is larger than any store a test can seed. What is pinned
    here is that the number comes from the store's own count.
    """
    _seed(mem, [{"id": f"m{i}", "content": f"fact {i}", "created_at": _ago(400)} for i in range(8)])

    seen = _record_sizing(mem)
    removed = mem.purge_expired(older_than_days=365)

    assert removed == 8
    assert seen["counts"], "purge_expired never asked the store how many rows are in scope"
    _, counted = seen["counts"][0]
    _, requested = seen["requests"][0]
    assert requested == counted + 1, (
        f"asked for {requested} against a store reporting {counted}; the request must be "
        "derived from the store so that our own cap cannot be what truncates the view"
    )


def test_a_scoped_purge_sizes_its_request_to_the_scope(mem):
    """The count must carry the same filter, or a scoped sweep is sized off
    the whole store and pays for rows it will never look at."""
    _seed(mem, [
        {"id": "a1", "content": "alice one", "user_id": "alice", "created_at": _ago(400)},
        {"id": "a2", "content": "alice two", "user_id": "alice", "created_at": _ago(400)},
        {"id": "b1", "content": "bob one", "user_id": "bob", "created_at": _ago(400)},
    ])

    seen = _record_sizing(mem)

    assert mem.purge_expired(older_than_days=365, user_id="alice") == 2
    assert seen["counts"][0][0] == {"user_id": "alice"}
    assert seen["requests"][0][0] == {"user_id": "alice"}
    assert seen["requests"][0][1] == seen["counts"][0][1] + 1 == 3
    assert mem.get("b1") is not None


def test_dry_run_is_sized_the_same_way(mem):
    _seed(mem, [{"id": f"m{i}", "content": f"fact {i}", "created_at": _ago(400)} for i in range(5)])

    seen = _record_sizing(mem)

    assert mem.purge_expired(older_than_days=365, dry_run=True) == 5
    _, counted = seen["counts"][0]
    _, requested = seen["requests"][0]
    assert requested == counted + 1
