from __future__ import annotations

import os
import re
import sqlite3
from typing import Dict, Iterable, List, Set, Tuple

from widemem.graph.extract import Triple

_NOISE = {  # too-generic anchors that would over-connect the graph
    "i", "me", "you", "it", "they", "we", "he", "she", "them", "thing", "things",
}

# Pure dates / numbers / years. These are valid edge endpoints (a memory that
# links caroline -> 2019 should still surface), but they must NOT be traversed
# THROUGH: a shared year would otherwise bridge two unrelated people. So they
# are leaves in BFS, never frontier nodes.
_LEAF_RE = re.compile(r"^[\d\s,./:-]+$|^(19|20)\d{2}$")


def _is_leaf(entity: str) -> bool:
    return bool(_LEAF_RE.match(entity.strip()))


class GraphStore:
    """SQLite-backed typed-triple store with bounded BFS traversal.

    One row per (subject, relation, object) edge, tagged with the source
    memory_id and user_id so traversal can return both neighbor entities and
    the memories that asserted each edge. Undirected for traversal (an edge
    connects subject<->object regardless of direction)."""

    def __init__(self, path: str) -> None:
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init()

    def _init(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   TEXT,
                subject   TEXT NOT NULL,
                relation  TEXT NOT NULL,
                object    TEXT NOT NULL,
                memory_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_subject ON edges(user_id, subject);
            CREATE INDEX IF NOT EXISTS idx_object  ON edges(user_id, object);
            CREATE INDEX IF NOT EXISTS idx_mem     ON edges(memory_id);
            """
        )
        self._conn.commit()

    def add_triples(self, triples: Iterable[Triple], memory_id: str, user_id: str | None) -> int:
        rows = [
            (user_id, s, r, o, memory_id)
            for (s, r, o) in triples
            if s not in _NOISE and o not in _NOISE
        ]
        if not rows:
            return 0
        self._conn.executemany(
            "INSERT INTO edges(user_id, subject, relation, object, memory_id) VALUES (?,?,?,?,?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def delete_memory(self, memory_id: str) -> None:
        self._conn.execute("DELETE FROM edges WHERE memory_id = ?", (memory_id,))
        self._conn.commit()

    def neighbors(self, entity: str, user_id: str | None) -> List[Tuple[str, str, str]]:
        """Edges touching `entity` (as subject or object), scoped to user_id."""
        cur = self._conn.execute(
            "SELECT subject, relation, object, memory_id FROM edges "
            "WHERE user_id IS ? AND (subject = ? OR object = ?)",
            (user_id, entity, entity),
        )
        return cur.fetchall()

    def expand(
        self,
        seeds: Iterable[str],
        user_id: str | None,
        hops: int = 2,
        max_nodes: int = 40,
    ) -> Tuple[Set[str], Dict[str, int]]:
        """BFS up to `hops` from seed entities. Returns (memory_ids touched,
        {entity: hop_distance}). memory_ids are the edges traversed, i.e. the
        memories that relationally connect the seeds to their neighborhood."""
        seeds = [s for s in (seeds or []) if s and s not in _NOISE]
        dist: Dict[str, int] = {s: 0 for s in seeds}
        frontier: Set[str] = set(seeds)
        mem_ids: Set[str] = set()
        for hop in range(1, hops + 1):
            nxt: Set[str] = set()
            for ent in frontier:
                for s, _r, o, mid in self.neighbors(ent, user_id):
                    if mid:
                        mem_ids.add(mid)
                    for other in (s, o):
                        if other != ent and other not in dist:
                            dist[other] = hop
                            # Dates/numbers are reachable endpoints but not
                            # bridges: record the distance, never expand through.
                            if not _is_leaf(other):
                                nxt.add(other)
                            if len(dist) >= max_nodes:
                                return mem_ids, dist
            frontier = nxt
            if not frontier:
                break
        return mem_ids, dist

    def edge_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
