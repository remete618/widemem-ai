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
