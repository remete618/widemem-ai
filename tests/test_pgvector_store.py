"""Unit tests for PgVectorStore using a mocked psycopg connection.

Why mocks instead of a real Postgres in CI:
- CI is lean: pytest only, no docker-based services
- The interesting logic is SQL generation + parameter binding + result
  parsing, not the network round-trip
- Integration tests against a real Postgres can live in a separate file
  gated on the PGVECTOR_TEST_URL env var so developers run them
  locally with `pytest tests/test_pgvector_integration.py` when they have
  a DB handy

Mocks here verify:
- DDL fires once on init (CREATE TABLE IF NOT EXISTS)
- Parameter binding is correct (no SQL injection)
- SELECT result rows round-trip back into expected (id, similarity, metadata)
  tuples
- Filters split correctly between indexed columns and JSONB containment
- The graceful errors fire (missing extension, missing URL, missing extra)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from widemem.core.exceptions import StorageError
from widemem.core.types import VectorStoreConfig


# Module-level mocked psycopg so importing PgVectorStore doesn't try to
# actually install / use real psycopg until the test wants to.
@pytest.fixture
def mock_psycopg(monkeypatch):
    fake_psycopg = MagicMock()
    fake_pgvector = MagicMock()
    fake_register = MagicMock()
    fake_pgvector.register_vector = fake_register

    import sys
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "pgvector", MagicMock())
    monkeypatch.setitem(sys.modules, "pgvector.psycopg", fake_pgvector)
    return fake_psycopg, fake_register


@pytest.fixture
def fake_conn(mock_psycopg):
    fake_psycopg, _ = mock_psycopg
    conn = MagicMock()
    conn.closed = False
    cur = MagicMock()
    # Make the context manager return the cursor.
    conn.cursor.return_value.__enter__.return_value = cur
    conn.cursor.return_value.__exit__.return_value = False
    # By default, vector extension exists (verify_extension passes).
    cur.fetchone.return_value = (1,)
    fake_psycopg.connect.return_value = conn
    return conn, cur


@pytest.fixture
def store(fake_conn):
    # Lazy import so the monkeypatched psycopg is what gets pulled in.
    from widemem.storage.vector.pgvector_store import PgVectorStore

    config = VectorStoreConfig(
        provider="pgvector",
        url="postgresql://test:test@localhost/widemem_test",
        table_name="test_memories",
    )
    return PgVectorStore(config, dimensions=4)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_missing_url_raises_storage_error(mock_psycopg):
    from widemem.storage.vector.pgvector_store import PgVectorStore

    config = VectorStoreConfig(provider="pgvector")
    with pytest.raises(StorageError, match="connection URL"):
        PgVectorStore(config, dimensions=4)


def test_missing_extension_raises_storage_error(mock_psycopg):
    fake_psycopg, _ = mock_psycopg
    conn = MagicMock()
    conn.closed = False
    cur = MagicMock()
    cur.fetchone.return_value = None  # Extension not installed
    conn.cursor.return_value.__enter__.return_value = cur
    fake_psycopg.connect.return_value = conn

    from widemem.storage.vector.pgvector_store import PgVectorStore

    config = VectorStoreConfig(
        provider="pgvector", url="postgresql://test/test"
    )
    with pytest.raises(StorageError, match="pgvector extension is not installed"):
        PgVectorStore(config, dimensions=4)


def test_init_creates_table_and_indexes(store, fake_conn):
    _, cur = fake_conn
    executed = [call.args[0] for call in cur.execute.call_args_list]
    create_table_calls = [s for s in executed if "CREATE TABLE IF NOT EXISTS" in s]
    assert len(create_table_calls) == 1
    assert "test_memories" in create_table_calls[0]
    assert "vector(4)" in create_table_calls[0]

    index_calls = [s for s in executed if "CREATE INDEX IF NOT EXISTS" in s]
    assert len(index_calls) >= 4  # at least user_id, agent_id, tier, created_at


def test_register_vector_called_on_init(mock_psycopg, fake_conn):
    _, register = mock_psycopg
    from widemem.storage.vector.pgvector_store import PgVectorStore

    config = VectorStoreConfig(
        provider="pgvector", url="postgresql://test/test", table_name="t"
    )
    PgVectorStore(config, dimensions=4)
    assert register.called


# ---------------------------------------------------------------------------
# Identifier sanitization (SQL injection guard)
# ---------------------------------------------------------------------------
def test_table_name_sanitized():
    from widemem.storage.vector.pgvector_store import PgVectorStore

    assert PgVectorStore._sanitize_identifier("good_name") == "good_name"
    assert PgVectorStore._sanitize_identifier("with spaces") == "with_spaces"
    # SQL injection attempt
    assert ";" not in PgVectorStore._sanitize_identifier("evil;DROP TABLE")
    # Starts with digit -> prefixed
    sanitized = PgVectorStore._sanitize_identifier("123name")
    assert sanitized.startswith("t_")
    # Length cap
    assert len(PgVectorStore._sanitize_identifier("x" * 100)) <= 63


# ---------------------------------------------------------------------------
# insert / update / get / delete
# ---------------------------------------------------------------------------
def test_insert_executes_upsert_with_correct_params(store, fake_conn):
    _, cur = fake_conn
    cur.execute.reset_mock()

    metadata = {
        "content": "Patient is allergic to penicillin.",
        "user_id": "alice",
        "tier": "fact",
        "ymyl_category": "health",
        "importance": 9.0,
    }
    vector = [0.1, 0.2, 0.3, 0.4]
    store.insert("m1", vector, metadata)

    assert cur.execute.called
    sql, params = cur.execute.call_args.args
    assert "INSERT INTO test_memories" in sql
    assert "ON CONFLICT (id) DO UPDATE" in sql
    assert params[0] == "m1"
    assert params[1] == vector
    assert params[2] == "Patient is allergic to penicillin."
    assert params[3] == "alice"
    assert params[6] == "fact"
    assert params[7] == "health"
    assert params[8] == 9.0


def test_insert_rejects_wrong_dimension(store):
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        store.insert("m1", [0.1, 0.2], {"content": "too short"})


def test_update_uses_same_upsert(store, fake_conn):
    _, cur = fake_conn
    cur.execute.reset_mock()
    store.update("m1", [0.1, 0.2, 0.3, 0.4], {"content": "updated"})
    sql = cur.execute.call_args.args[0]
    assert "INSERT INTO test_memories" in sql
    assert "ON CONFLICT" in sql


def test_delete_uses_parameterized_query(store, fake_conn):
    _, cur = fake_conn
    cur.execute.reset_mock()
    store.delete("m1")
    sql, params = cur.execute.call_args.args
    assert sql.strip().startswith("DELETE FROM test_memories")
    assert params == ("m1",)


def test_get_returns_none_when_missing(store, fake_conn):
    _, cur = fake_conn
    cur.fetchone.return_value = None
    result = store.get("does-not-exist")
    assert result is None


def test_get_round_trips_metadata(store, fake_conn):
    _, cur = fake_conn
    now = datetime(2026, 5, 14, tzinfo=timezone.utc)
    cur.fetchone.return_value = (
        [0.1, 0.2, 0.3, 0.4],                # embedding
        "Patient allergy",                    # content
        "alice",                              # user_id
        None,                                 # agent_id
        None,                                 # run_id
        "fact",                               # tier
        "health",                             # ymyl_category
        9.0,                                  # importance
        '{"extra": "field"}',                 # metadata JSONB
        now,                                  # created_at
        now,                                  # updated_at
    )
    vec, meta = store.get("m1")
    assert vec == [0.1, 0.2, 0.3, 0.4]
    assert meta["content"] == "Patient allergy"
    assert meta["user_id"] == "alice"
    assert meta["ymyl_category"] == "health"
    assert meta["extra"] == "field"  # arbitrary JSONB key preserved
    assert meta["created_at"] == now.isoformat()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
def test_search_orders_by_cosine_distance(store, fake_conn):
    _, cur = fake_conn
    cur.fetchall.return_value = []
    store.search([0.1, 0.2, 0.3, 0.4], top_k=5)
    sql = cur.execute.call_args.args[0]
    # The query vector MUST be cast to ::vector. Without it, real Postgres
    # rejects the bound list param as double precision[] (verified live on
    # Supabase; the bare-operator form passed mocks but failed in production).
    assert "1 - (embedding <=> %s::vector)" in sql
    assert "ORDER BY embedding <=> %s::vector" in sql
    assert "LIMIT %s" in sql


def test_search_passes_vector_as_castable_literal(store, fake_conn):
    # Param must be a pgvector string literal, not a raw list (which psycopg
    # adapts to double precision[] and the <=> operator can't match).
    _, cur = fake_conn
    cur.fetchall.return_value = []
    store.search([0.1, 0.2, 0.3, 0.4], top_k=5)
    params = cur.execute.call_args.args[1]
    assert params[0] == "[0.1,0.2,0.3,0.4]"


def test_search_returns_similarity_as_one_minus_distance(store, fake_conn):
    _, cur = fake_conn
    now = datetime(2026, 5, 14, tzinfo=timezone.utc)
    cur.fetchall.return_value = [
        (
            "m1", 0.87, "Caroline moved from Sweden",
            "alice", None, None, "fact", None, 7.0,
            '{}', now, now,
        ),
    ]
    results = store.search([0.1, 0.2, 0.3, 0.4], top_k=5)
    assert len(results) == 1
    id_, similarity, meta = results[0]
    assert id_ == "m1"
    assert similarity == 0.87
    assert meta["content"] == "Caroline moved from Sweden"


def test_search_indexed_filter_uses_column(store, fake_conn):
    _, cur = fake_conn
    cur.fetchall.return_value = []
    store.search([0.1, 0.2, 0.3, 0.4], top_k=5, filters={"user_id": "alice"})
    sql, params = cur.execute.call_args.args
    assert "user_id = %s" in sql
    assert "alice" in params


def test_search_unknown_filter_uses_jsonb_containment(store, fake_conn):
    _, cur = fake_conn
    cur.fetchall.return_value = []
    store.search([0.1, 0.2, 0.3, 0.4], top_k=5, filters={"custom_key": "value"})
    sql, params = cur.execute.call_args.args
    assert "metadata @> %s" in sql
    assert json.dumps({"custom_key": "value"}) in params


def test_search_validates_dimension(store):
    with pytest.raises(ValueError):
        store.search([0.1, 0.2], top_k=5)


# ---------------------------------------------------------------------------
# list_all
# ---------------------------------------------------------------------------
def test_list_all_default_max_results(store, fake_conn):
    _, cur = fake_conn
    cur.fetchall.return_value = []
    store.list_all()
    sql, params = cur.execute.call_args.args
    assert sql.strip().startswith("SELECT id")
    assert "ORDER BY created_at DESC NULLS LAST" in sql
    assert params[-1] == 1000  # default max_results


def test_list_all_applies_filters(store, fake_conn):
    _, cur = fake_conn
    cur.fetchall.return_value = []
    store.list_all(filters={"user_id": "alice"}, max_results=50)
    sql, params = cur.execute.call_args.args
    assert "user_id = %s" in sql
    assert "alice" in params
    assert params[-1] == 50


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------
def test_close_closes_connection(store, fake_conn):
    conn, _ = fake_conn
    store.close()
    conn.close.assert_called_once()


def test_close_safe_when_already_closed(store, fake_conn):
    conn, _ = fake_conn
    conn.closed = True
    store.close()  # should not raise
    conn.close.assert_not_called()
