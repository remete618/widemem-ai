"""Regression tests for WM-1 (atomic FAISS save) and WM-3 (thread-safe HistoryStore).

WM-1: FAISSVectorStore._save() must be crash-atomic. A failure partway through
persisting the index + state must leave the previously-saved, consistent state
on disk, never a half-written index/state mismatch.

WM-3: HistoryStore must be usable from threads other than the one that created
it (FastAPI runs sync handlers in a threadpool; search_stream uses
asyncio.to_thread). A connection opened with the sqlite default
check_same_thread=True raises when touched from another thread.
"""
from __future__ import annotations

import json as real_json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from widemem.core.types import MemoryAction, VectorStoreConfig
from widemem.storage.history import HistoryStore
from widemem.storage.vector import faiss_store as faiss_module
from widemem.storage.vector.faiss_store import FAISSVectorStore


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# WM-1: atomic save
# ---------------------------------------------------------------------------


class _FlakyJson:
    """Stand-in for the json module that can fail dump() on demand.

    load() always delegates to the real implementation so reload works.
    """

    def __init__(self) -> None:
        self.fail = False

    def dump(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("simulated crash during state write")
        return real_json.dump(*args, **kwargs)

    def load(self, *args, **kwargs):
        return real_json.load(*args, **kwargs)


def test_save_is_atomic_on_crash(tmp_dir, monkeypatch):
    """A crash during the second write must not corrupt the saved store."""
    config = VectorStoreConfig(path=f"{tmp_dir}/atomic")
    store = FAISSVectorStore(config, dimensions=4)

    flaky = _FlakyJson()
    monkeypatch.setattr(faiss_module, "json", flaky)

    # Clean first write: one consistent memory on disk.
    store.insert("id1", [0.1, 0.2, 0.3, 0.4], {"content": "first"})

    # Now simulate a crash between writing the index and writing the state.
    flaky.fail = True
    with pytest.raises(RuntimeError):
        store.insert("id2", [0.4, 0.3, 0.2, 0.1], {"content": "second"})
    flaky.fail = False

    # Reload from disk. The store must be the last *consistent* state (id1 only),
    # with index and metadata in agreement.
    store2 = FAISSVectorStore(config, dimensions=4)
    assert store2._index.ntotal == len(store2._metadata), (
        "index and metadata disagree after a crashed save"
    )
    assert store2._index.ntotal == 1
    assert store2.get("id1") is not None
    assert store2.get("id2") is None


def test_normal_save_still_round_trips(tmp_dir):
    """The atomic path must not regress ordinary persistence."""
    config = VectorStoreConfig(path=f"{tmp_dir}/roundtrip")
    store = FAISSVectorStore(config, dimensions=4)
    store.insert("id1", [0.1, 0.2, 0.3, 0.4], {"content": "hello"})
    store.insert("id2", [0.4, 0.3, 0.2, 0.1], {"content": "world"})

    store2 = FAISSVectorStore(config, dimensions=4)
    assert store2._index.ntotal == 2
    assert store2.get("id1")[1]["content"] == "hello"
    assert store2.get("id2")[1]["content"] == "world"


# ---------------------------------------------------------------------------
# WM-3: thread-safe HistoryStore
# ---------------------------------------------------------------------------


def test_log_from_a_different_thread(tmp_dir):
    """A store created on the main thread must be writable from a worker thread."""
    store = HistoryStore(db_path=f"{tmp_dir}/history.db")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            entry = pool.submit(
                store.log, "mem-1", MemoryAction.ADD, None, "hello"
            ).result()
        assert entry.memory_id == "mem-1"
        assert len(store.get_history("mem-1")) == 1
    finally:
        store.close()


def test_concurrent_logs_do_not_lose_writes(tmp_dir):
    """Concurrent writers from many threads must all persist, with no errors."""
    store = HistoryStore(db_path=f"{tmp_dir}/history_concurrent.db")
    n_threads = 8
    per_thread = 25
    errors: list[Exception] = []
    barrier = threading.Barrier(n_threads)

    def worker(thread_id: int) -> None:
        barrier.wait()  # maximize contention
        try:
            for i in range(per_thread):
                store.log(f"mem-{thread_id}", MemoryAction.ADD, None, f"v{i}")
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    try:
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(pool.map(worker, range(n_threads)))

        assert not errors, f"concurrent logging raised: {errors}"
        total = sum(len(store.get_history(f"mem-{t}")) for t in range(n_threads))
        assert total == n_threads * per_thread
    finally:
        store.close()


# ---------------------------------------------------------------------------
# WM-2: batched saves (avoid O(n) full rewrite per write during bulk ingestion)
# ---------------------------------------------------------------------------


def _one_hot(i: int, dims: int = 4) -> list[float]:
    return [1.0 if (i % dims) == j else 0.0 for j in range(dims)]


def _count_saves(store: FAISSVectorStore) -> dict[str, int]:
    """Replace store._save with a counting wrapper; return the counter dict."""
    calls = {"n": 0}
    original = store._save

    def counting() -> None:
        calls["n"] += 1
        original()

    store._save = counting  # type: ignore[method-assign]
    return calls


def test_batch_writes_persists_with_a_single_save(tmp_dir):
    """N inserts inside batch_writes() must flush once, not N times."""
    config = VectorStoreConfig(path=f"{tmp_dir}/batch")
    store = FAISSVectorStore(config, dimensions=4)
    saves = _count_saves(store)

    n = 20
    with store.batch_writes():
        for i in range(n):
            store.insert(f"id{i}", _one_hot(i), {"content": f"c{i}"})
        # In memory immediately; not yet flushed to disk.
        assert store._index.ntotal == n

    assert saves["n"] == 1, f"expected one save for the batch, got {saves['n']}"

    # Durable after the batch: a fresh store on the same path sees everything.
    store2 = FAISSVectorStore(config, dimensions=4)
    assert store2._index.ntotal == n


def test_single_insert_still_saves_immediately(tmp_dir):
    """Outside a batch, every write must persist immediately (durability)."""
    config = VectorStoreConfig(path=f"{tmp_dir}/single")
    store = FAISSVectorStore(config, dimensions=4)
    saves = _count_saves(store)

    store.insert("a", _one_hot(0), {"content": "x"})
    assert saves["n"] == 1


def test_batch_writes_flushes_even_on_error(tmp_dir):
    """A failure mid-batch must still persist the writes that succeeded."""
    config = VectorStoreConfig(path=f"{tmp_dir}/batch_err")
    store = FAISSVectorStore(config, dimensions=4)

    with pytest.raises(ValueError):
        with store.batch_writes():
            store.insert("ok", _one_hot(0), {"content": "kept"})
            raise ValueError("boom")

    store2 = FAISSVectorStore(config, dimensions=4)
    assert store2.get("ok") is not None
