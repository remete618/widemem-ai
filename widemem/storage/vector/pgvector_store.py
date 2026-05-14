"""Postgres + pgvector backend for widemem.

Unlocks hosted deployments: AWS RDS, Google Cloud SQL, Azure Postgres,
Neon, Supabase, or any Postgres-as-a-service with the pgvector extension
installed.

Same `BaseVectorStore` interface as FAISS and Qdrant. Drop-in. The widemem
data shape (id, content, importance, ymyl_category, timestamps, etc.) is
stored as both real columns (for indexing and SQL filtering) and as JSONB
(for the round-trip metadata contract the rest of widemem assumes).

Why a single-table schema:
- Compliance teams already audit Postgres tables. One table per widemem
  instance is the easiest mental model.
- Common filters (user_id, agent_id, tier, time range) hit indexed columns
  for free.
- The full metadata dict round-trips through a JSONB column, so callers
  that store arbitrary metadata still work.

Why not a connection pool:
- widemem stays a library. Each WideMemory instance owns its connection.
- Users wanting pooling wrap construction with `psycopg_pool.ConnectionPool`
  at the application layer.

Optional dependency: install with `pip install widemem-ai[pgvector]`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from widemem.core.exceptions import StorageError
from widemem.core.types import VectorStoreConfig
from widemem.storage.vector.base import BaseVectorStore

# Columns lifted out of the metadata JSONB for native SQL indexing.
_INDEXED_FIELDS = (
    "user_id",
    "agent_id",
    "run_id",
    "tier",
    "ymyl_category",
)


class PgVectorStore(BaseVectorStore):
    """Postgres + pgvector vector store.

    Stores one row per memory. The `embedding` column uses pgvector's
    native vector type. Common filters (user_id, agent_id, tier) hit real
    indexed columns. The full metadata dict round-trips through a `metadata`
    JSONB column so callers can store arbitrary fields.
    """

    def __init__(self, config: VectorStoreConfig, dimensions: int = 1536) -> None:
        super().__init__(config)
        if not config.url:
            raise StorageError(
                "PgVectorStore requires a connection URL. "
                "Set MemoryConfig(vector_store=VectorStoreConfig("
                "provider='pgvector', url='postgresql://...')) "
                "or use the PGVECTOR_URL env var."
            )

        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as e:
            raise StorageError(
                "PgVectorStore requires the [pgvector] extra. "
                "Install with: pip install widemem-ai[pgvector]"
            ) from e

        self._psycopg = psycopg
        self._register_vector = register_vector

        self.dimensions = dimensions
        self.table_name = self._sanitize_identifier(config.table_name)
        self._conn = psycopg.connect(config.url, autocommit=True)
        self._verify_extension()
        register_vector(self._conn)
        self._create_table_if_needed()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_identifier(name: str) -> str:
        """Allow only ASCII alphanum + underscore in identifiers.

        Postgres identifier rules are more permissive but we restrict to
        avoid SQL injection through a user-supplied table_name. Anything
        outside [A-Za-z0-9_] becomes underscore. Cannot start with a digit.
        """
        clean = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        if not clean or clean[0].isdigit():
            clean = "t_" + clean
        return clean[:63]  # Postgres NAMEDATALEN

    def _verify_extension(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if cur.fetchone() is None:
                raise StorageError(
                    "The pgvector extension is not installed on the target "
                    "database. Run `CREATE EXTENSION vector;` as a superuser "
                    "before using PgVectorStore. On Supabase, this extension "
                    "is enabled via the dashboard under Database > Extensions."
                )

    def _create_table_if_needed(self) -> None:
        ddl = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                embedding vector({self.dimensions}) NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                user_id TEXT,
                agent_id TEXT,
                run_id TEXT,
                tier TEXT,
                ymyl_category TEXT,
                importance REAL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ
            )
        """
        index_ddls = [
            f"CREATE INDEX IF NOT EXISTS {self.table_name}_user_id_idx "
            f"ON {self.table_name} (user_id)",
            f"CREATE INDEX IF NOT EXISTS {self.table_name}_agent_id_idx "
            f"ON {self.table_name} (agent_id)",
            f"CREATE INDEX IF NOT EXISTS {self.table_name}_tier_idx "
            f"ON {self.table_name} (tier)",
            f"CREATE INDEX IF NOT EXISTS {self.table_name}_created_at_idx "
            f"ON {self.table_name} (created_at)",
            # HNSW index on the embedding column for fast ANN search.
            # Cosine distance to match widemem's similarity convention.
            f"CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx "
            f"ON {self.table_name} USING hnsw (embedding vector_cosine_ops)",
        ]
        with self._conn.cursor() as cur:
            cur.execute(ddl)
            for ddl_stmt in index_ddls:
                try:
                    cur.execute(ddl_stmt)
                except Exception:
                    # HNSW index may fail on older pgvector versions; fall
                    # back to IVFFlat or no index. The store still works,
                    # just slower on large corpora.
                    pass

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_vector(self, vector: List[float]) -> None:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimensions}, "
                f"got {len(vector)}"
            )

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------
    def insert(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        self._validate_vector(vector)
        cols, vals = self._split_metadata(metadata)
        sql = f"""
            INSERT INTO {self.table_name}
                (id, embedding, content, user_id, agent_id, run_id, tier,
                 ymyl_category, importance, metadata, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                content = EXCLUDED.content,
                user_id = EXCLUDED.user_id,
                agent_id = EXCLUDED.agent_id,
                run_id = EXCLUDED.run_id,
                tier = EXCLUDED.tier,
                ymyl_category = EXCLUDED.ymyl_category,
                importance = EXCLUDED.importance,
                metadata = EXCLUDED.metadata,
                updated_at = EXCLUDED.updated_at
        """
        with self._conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    id,
                    vector,
                    cols["content"],
                    cols["user_id"],
                    cols["agent_id"],
                    cols["run_id"],
                    cols["tier"],
                    cols["ymyl_category"],
                    cols["importance"],
                    json.dumps(vals),
                    cols["created_at"],
                    cols["updated_at"],
                ),
            )

    def search(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        self._validate_vector(vector)
        where_clauses: List[str] = []
        params: List[Any] = [vector]
        if filters:
            for key, value in filters.items():
                if key in _INDEXED_FIELDS:
                    where_clauses.append(f"{key} = %s")
                    params.append(value)
                else:
                    # Fall back to JSONB containment for arbitrary keys
                    where_clauses.append("metadata @> %s")
                    params.append(json.dumps({key: value}))
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # Cosine distance via pgvector's <=> operator. similarity = 1 - distance.
        sql = f"""
            SELECT id, 1 - (embedding <=> %s) AS similarity, content,
                   user_id, agent_id, run_id, tier, ymyl_category,
                   importance, metadata, created_at, updated_at
            FROM {self.table_name}
            {where}
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        # Add the second vector binding for ORDER BY, plus the top_k.
        params.append(vector)
        params.append(top_k)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [self._row_to_search_result(row) for row in rows]

    def update(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        # insert() uses ON CONFLICT DO UPDATE; reuse it.
        self.insert(id, vector, metadata)

    def delete(self, id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table_name} WHERE id = %s", (id,))

    def get(self, id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        sql = f"""
            SELECT embedding, content, user_id, agent_id, run_id, tier,
                   ymyl_category, importance, metadata, created_at, updated_at
            FROM {self.table_name}
            WHERE id = %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (id,))
            row = cur.fetchone()
        if row is None:
            return None
        vector = self._vector_to_list(row[0])
        metadata = self._row_to_metadata(row, start_idx=1)
        return vector, metadata

    def list_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 1000,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        where_clauses: List[str] = []
        params: List[Any] = []
        if filters:
            for key, value in filters.items():
                if key in _INDEXED_FIELDS:
                    where_clauses.append(f"{key} = %s")
                    params.append(value)
                else:
                    where_clauses.append("metadata @> %s")
                    params.append(json.dumps({key: value}))
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"""
            SELECT id, content, user_id, agent_id, run_id, tier,
                   ymyl_category, importance, metadata, created_at, updated_at
            FROM {self.table_name}
            {where}
            ORDER BY created_at DESC NULLS LAST
            LIMIT %s
        """
        params.append(max_results)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [(row[0], self._row_to_metadata(row, start_idx=1)) for row in rows]

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _split_metadata(metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split metadata into (column_values, json_remainder).

        Indexed fields go into real columns. Everything else stays in the
        JSONB blob for round-tripping unknown keys. ``content`` is always
        a column. Timestamps that aren't datetime instances pass through
        as-is; psycopg can handle ISO strings via the timestamptz cast.
        """
        cols = {
            "content": metadata.get("content", ""),
            "user_id": metadata.get("user_id"),
            "agent_id": metadata.get("agent_id"),
            "run_id": metadata.get("run_id"),
            "tier": metadata.get("tier"),
            "ymyl_category": metadata.get("ymyl_category"),
            "importance": metadata.get("importance"),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
        }
        # Keep the full dict in JSONB so callers see exactly what they wrote.
        return cols, dict(metadata)

    def _row_to_metadata(self, row: tuple, start_idx: int = 0) -> Dict[str, Any]:
        """Reconstruct a metadata dict from a SELECT row.

        Layout:
          start_idx + 0: content
          start_idx + 1: user_id
          start_idx + 2: agent_id
          start_idx + 3: run_id
          start_idx + 4: tier
          start_idx + 5: ymyl_category
          start_idx + 6: importance
          start_idx + 7: metadata (jsonb)
          start_idx + 8: created_at
          start_idx + 9: updated_at
        """
        jsonb = row[start_idx + 7] or {}
        if isinstance(jsonb, str):
            try:
                jsonb = json.loads(jsonb)
            except json.JSONDecodeError:
                jsonb = {}
        # Always-present fields override JSONB so users see canonical values.
        meta = dict(jsonb)
        meta["content"] = row[start_idx + 0]
        meta["user_id"] = row[start_idx + 1]
        meta["agent_id"] = row[start_idx + 2]
        meta["run_id"] = row[start_idx + 3]
        meta["tier"] = row[start_idx + 4]
        meta["ymyl_category"] = row[start_idx + 5]
        meta["importance"] = row[start_idx + 6]
        if row[start_idx + 8] is not None:
            meta["created_at"] = row[start_idx + 8].isoformat() if hasattr(
                row[start_idx + 8], "isoformat"
            ) else row[start_idx + 8]
        if row[start_idx + 9] is not None:
            meta["updated_at"] = row[start_idx + 9].isoformat() if hasattr(
                row[start_idx + 9], "isoformat"
            ) else row[start_idx + 9]
        return meta

    def _row_to_search_result(
        self, row: tuple
    ) -> Tuple[str, float, Dict[str, Any]]:
        """SELECT row -> (id, similarity, metadata)."""
        id_, similarity = row[0], float(row[1])
        metadata = self._row_to_metadata(row, start_idx=2)
        return id_, similarity, metadata

    @staticmethod
    def _vector_to_list(vec: Any) -> List[float]:
        """Convert a pgvector return value to a Python list of floats."""
        if isinstance(vec, list):
            return [float(x) for x in vec]
        if hasattr(vec, "tolist"):
            return [float(x) for x in vec.tolist()]
        return list(vec)
