"""A FAISS index saved with one embedding size must not load under another:
the default embedder changed from 1536 to 384 dimensions in 1.7.0."""
from __future__ import annotations

import pytest

pytest.importorskip("faiss")

from widemem.core.exceptions import StorageError  # noqa: E402
from widemem.core.types import VectorStoreConfig  # noqa: E402
from widemem.storage.vector.faiss_store import FAISSVectorStore  # noqa: E402


def _saved_store(path, dims):
    store = FAISSVectorStore(VectorStoreConfig(provider="faiss", path=str(path)), dimensions=dims)
    store.insert("m1", [1.0] + [0.0] * (dims - 1), {"user_id": "u", "content": "x"})
    return store


def test_reopening_with_another_embedding_size_names_the_mismatch(tmp_path):
    _saved_store(tmp_path, 1536)
    with pytest.raises(StorageError, match=r"1536.*384"):
        FAISSVectorStore(VectorStoreConfig(provider="faiss", path=str(tmp_path)), dimensions=384)


def test_reopening_with_the_same_size_still_loads(tmp_path):
    _saved_store(tmp_path, 8)
    reopened = FAISSVectorStore(VectorStoreConfig(provider="faiss", path=str(tmp_path)), dimensions=8)
    assert reopened.count() == 1


def test_macos_pins_faiss_to_one_openmp_thread(monkeypatch, tmp_path):
    """faiss and torch each ship libomp on macOS; a multithreaded faiss search
    after torch loads segfaults the process (the default local stack)."""
    import faiss

    import widemem.storage.vector.faiss_store as store_mod

    calls = []
    monkeypatch.setattr(store_mod.sys, "platform", "darwin")
    monkeypatch.setitem(store_mod.sys.modules, "torch", object())
    monkeypatch.setattr(faiss, "omp_set_num_threads", calls.append)
    FAISSVectorStore(VectorStoreConfig(provider="faiss"), dimensions=4)
    assert calls == [1]


def test_macos_without_torch_keeps_faiss_threading(monkeypatch):
    import faiss

    import widemem.storage.vector.faiss_store as store_mod

    calls = []
    monkeypatch.setattr(store_mod.sys, "platform", "darwin")
    monkeypatch.delitem(store_mod.sys.modules, "torch", raising=False)
    monkeypatch.setattr(faiss, "omp_set_num_threads", calls.append)
    FAISSVectorStore(VectorStoreConfig(provider="faiss"), dimensions=4)
    assert calls == []


def test_other_platforms_keep_faiss_threading(monkeypatch):
    import faiss

    import widemem.storage.vector.faiss_store as store_mod

    calls = []
    monkeypatch.setattr(store_mod.sys, "platform", "linux")
    monkeypatch.setattr(faiss, "omp_set_num_threads", calls.append)
    FAISSVectorStore(VectorStoreConfig(provider="faiss"), dimensions=4)
    assert calls == []


def test_store_built_before_torch_pins_at_first_search(monkeypatch):
    import faiss

    import widemem.storage.vector.faiss_store as store_mod

    calls = []
    monkeypatch.setattr(store_mod.sys, "platform", "darwin")
    monkeypatch.delitem(store_mod.sys.modules, "torch", raising=False)
    monkeypatch.setattr(faiss, "omp_set_num_threads", calls.append)
    store = FAISSVectorStore(VectorStoreConfig(provider="faiss"), dimensions=4)
    assert calls == []
    monkeypatch.setitem(store_mod.sys.modules, "torch", object())
    store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert calls == [1]
