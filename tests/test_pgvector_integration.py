"""PgVectorStore against a real Postgres with pgvector.

Skipped unless PGVECTOR_TEST_URL points at a database where the current
user may create tables and the vector extension is installed, e.g.
PGVECTOR_TEST_URL=postgresql://localhost/widemem_test pytest tests/test_pgvector_integration.py
Every test drops the tables it creates.
"""
from __future__ import annotations

import os
import uuid

import pytest

URL = os.environ.get("PGVECTOR_TEST_URL")
pytestmark = pytest.mark.skipif(not URL, reason="PGVECTOR_TEST_URL is not set")

if URL:
    psycopg = pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")

from widemem.core.exceptions import StorageError  # noqa: E402
from widemem.core.types import VectorStoreConfig  # noqa: E402


@pytest.fixture
def sql():
    conn = psycopg.connect(URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    created: list[str] = []

    def run(statement: str, table: str | None = None) -> None:
        if table:
            created.append(table)
        conn.execute(statement)

    yield run
    for table in created:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.close()


def _name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _open(table: str, dimensions: int):
    from widemem.storage.vector.pgvector_store import PgVectorStore

    return PgVectorStore(
        VectorStoreConfig(provider="pgvector", url=URL, table_name=table), dimensions=dimensions
    )


def test_table_built_for_1536_refuses_a_384_embedder(sql):
    table = _name("wm_dims")
    sql("SELECT 1", table)
    _open(table, 1536).close()
    with pytest.raises(StorageError, match=r"1536-dimensional.*produces 384"):
        _open(table, 384)
    store = _open(table, 1536)
    store.insert("a", [0.1] * 1536, {"content": "x", "user_id": "u"})
    assert [id_ for id_, _ in store.list_all()] == ["a"]
    store.close()


def test_bare_vector_column_opens_under_any_size(sql):
    table = _name("wm_bare")
    sql(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, embedding vector NOT NULL)", table)
    _open(table, 384).close()
    _open(table, 1536).close()


def test_table_without_embedding_column_is_refused(sql):
    table = _name("wm_noemb")
    sql(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, content TEXT)", table)
    with pytest.raises(StorageError, match="exists but has no embedding column"):
        _open(table, 384)


def test_mixed_case_table_name_is_found_and_guarded(sql):
    table = "WmMixed_" + uuid.uuid4().hex[:8]
    sql("SELECT 1", table.lower())
    _open(table, 8).close()
    with pytest.raises(StorageError, match=r"8-dimensional.*produces 4"):
        _open(table, 4)
