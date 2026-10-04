"""An existing Qdrant collection built with one embedding size must not open
under another: the default embedder changed from 1536 to 384 dimensions."""
from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")

from widemem.core.exceptions import StorageError  # noqa: E402
from widemem.core.types import VectorStoreConfig  # noqa: E402
from widemem.storage.vector.qdrant_store import QdrantVectorStore  # noqa: E402


def _open(path, dims):
    return QdrantVectorStore(VectorStoreConfig(provider="qdrant", path=str(path)), dimensions=dims)


def test_reopening_with_another_embedding_size_names_the_mismatch(tmp_path):
    _open(tmp_path, 1536).client.close()
    with pytest.raises(StorageError, match=r"1536.*384"):
        _open(tmp_path, 384)


def test_reopening_with_the_same_size_still_opens(tmp_path):
    _open(tmp_path, 8).client.close()
    assert _open(tmp_path, 8).dimensions == 8


def test_tilde_in_path_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _open("~/qdrant-data", 8).client.close()
    assert (tmp_path / "qdrant-data").is_dir()
    assert not (tmp_path / "~").exists()
