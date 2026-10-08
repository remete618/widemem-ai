"""Integration test for PgVectorStore.get() against a real Postgres.

Skipped unless PGVECTOR_TEST_URL points at a database with the pgvector
extension installed, for example:

    PGVECTOR_TEST_URL=postgresql://localhost/widemem_test pytest tests/test_pgvector_get_integration.py

The test creates and drops its own uniquely named table.
"""

from __future__ import annotations

import os
import uuid

import pytest

PGVECTOR_TEST_URL = os.environ.get("PGVECTOR_TEST_URL")

pytestmark = pytest.mark.skipif(
    not PGVECTOR_TEST_URL, reason="PGVECTOR_TEST_URL not set"
)


@pytest.fixture
def pg_store():
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    from widemem.core.types import VectorStoreConfig
    from widemem.storage.vector.pgvector_store import PgVectorStore

    table = f"widemem_get_it_{uuid.uuid4().hex[:12]}"
    store = PgVectorStore(
        VectorStoreConfig(provider="pgvector", url=PGVECTOR_TEST_URL, table_name=table),
        dimensions=4,
    )
    try:
        yield store
    finally:
        with store._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {store.table_name}")
        store.close()


def test_get_returns_float_list_from_real_postgres(pg_store):
    pg_store.insert("m1", [0.5, 0.25, 0.0, -1.0], {"content": "hello", "user_id": "alice"})
    got = pg_store.get("m1")
    assert got is not None
    vec, meta = got
    assert isinstance(vec, list)
    assert vec == [0.5, 0.25, 0.0, -1.0]
    assert all(type(x) is float for x in vec)
    assert meta["content"] == "hello"
    assert meta["user_id"] == "alice"


def test_update_after_get_round_trips(pg_store):
    pg_store.insert("m1", [1.0, 0.0, 0.0, 0.0], {"content": "a"})
    vec, meta = pg_store.get("m1")
    meta["content"] = "b"
    pg_store.update("m1", vec, meta)
    vec2, meta2 = pg_store.get("m1")
    assert vec2 == [1.0, 0.0, 0.0, 0.0]
    assert meta2["content"] == "b"


def test_get_missing_returns_none(pg_store):
    assert pg_store.get("nope") is None
