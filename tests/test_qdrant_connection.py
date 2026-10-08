"""VectorStoreConfig picks the Qdrant client: path runs embedded, url
connects to a remote server, neither means localhost:6333."""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("qdrant_client")

from widemem.core.exceptions import StorageError  # noqa: E402
from widemem.core.types import VectorStoreConfig  # noqa: E402
from widemem.storage.vector.qdrant_store import QdrantVectorStore  # noqa: E402


class _Calls(list):
    pass


@pytest.fixture
def client_calls(monkeypatch):
    from unittest.mock import MagicMock

    import qdrant_client

    calls = _Calls()
    calls_existing: dict = {}

    def fake_client(*args, **kwargs):
        calls.append((args, kwargs))
        client = MagicMock()
        collections = []
        for name, size in calls_existing.items():
            collection = MagicMock()
            collection.name = name
            collections.append(collection)
            client.get_collection.return_value.config.params.vectors.size = size
        client.get_collections.return_value.collections = collections
        return client

    monkeypatch.setattr(qdrant_client, "QdrantClient", fake_client)
    calls.existing = calls_existing
    return calls


def _open_config(dimensions=8, **config):
    return QdrantVectorStore(VectorStoreConfig(provider="qdrant", **config), dimensions=dimensions)


def test_url_connects_to_a_remote_server(client_calls):
    _open_config(url="https://qdrant.example.com:6333")
    assert client_calls == [((), {"url": "https://qdrant.example.com:6333"})]


def test_path_wins_over_url(client_calls, tmp_path):
    _open_config(path=str(tmp_path), url="https://qdrant.example.com:6333")
    assert client_calls == [((), {"path": str(tmp_path)})]


@pytest.mark.parametrize("url", [None, ""])
def test_no_path_and_no_url_falls_back_to_localhost(client_calls, url):
    _open_config(url=url)
    assert client_calls == [((), {"host": "localhost", "port": 6333})]


SECRET = "qdrant-secret-7f3a"


def test_url_with_api_key_passes_the_key_to_the_client(client_calls):
    _open_config(url="https://qdrant.example.com:6333", api_key=SECRET)
    assert client_calls == [((), {"url": "https://qdrant.example.com:6333", "api_key": SECRET})]


def test_api_key_is_not_passed_to_an_embedded_client(client_calls, tmp_path):
    _open_config(path=str(tmp_path), url="https://qdrant.example.com:6333", api_key=SECRET)
    assert client_calls == [((), {"path": str(tmp_path)})]


def test_api_key_never_appears_in_plain_text(client_calls, caplog):
    caplog.set_level(logging.DEBUG)
    config = VectorStoreConfig(provider="qdrant", url="https://qdrant.example.com:6333", api_key=SECRET)
    store = QdrantVectorStore(config, dimensions=8)
    for text in (repr(config), str(config), config.model_dump_json(), repr(store.config), caplog.text):
        assert SECRET not in text
    assert config.api_key.get_secret_value() == SECRET


def test_remote_collection_with_another_size_names_the_mismatch(client_calls):
    client_calls.existing["widemem"] = 384
    with pytest.raises(StorageError, match=r"384-dimensional.*produces 1536"):
        _open_config(dimensions=1536, url="https://qdrant.example.com:6333")


def test_remote_collection_with_the_same_size_opens(client_calls):
    client_calls.existing["widemem"] = 1536
    store = _open_config(dimensions=1536, url="https://qdrant.example.com:6333")
    store.client.create_collection.assert_not_called()
