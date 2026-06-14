from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from widemem.core._time import as_utc
from widemem.core.types import HistoryEntry, MemoryAction


class HistoryStore:
    def __init__(self, db_path: str = "~/.widemem/history.db") -> None:
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        # The server runs sync handlers in a threadpool and search_stream uses
        # asyncio.to_thread, so a single connection is touched from threads
        # other than the one that created it. check_same_thread=False allows
        # that; the lock serializes access so writes don't interleave.
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    old_content TEXT,
                    new_content TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_memory_id ON history(memory_id)
            """)
            self.conn.commit()

    def log(
        self,
        memory_id: str,
        action: MemoryAction,
        old_content: str | None = None,
        new_content: str | None = None,
    ) -> HistoryEntry:
        entry = HistoryEntry(
            id=str(uuid.uuid4()),
            memory_id=memory_id,
            action=action,
            old_content=old_content,
            new_content=new_content,
            timestamp=datetime.now(timezone.utc),
        )
        with self._lock:
            self.conn.execute(
                "INSERT INTO history (id, memory_id, action, old_content, new_content, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (entry.id, entry.memory_id, entry.action.value, entry.old_content, entry.new_content, entry.timestamp.isoformat()),
            )
            self.conn.commit()
        return entry

    def get_history(self, memory_id: str) -> list[HistoryEntry]:
        with self._lock:
            cursor = self.conn.execute(
                "SELECT id, memory_id, action, old_content, new_content, timestamp FROM history WHERE memory_id = ? ORDER BY timestamp",
                (memory_id,),
            )
            rows = cursor.fetchall()
        return [
            HistoryEntry(
                id=row[0],
                memory_id=row[1],
                action=MemoryAction(row[2]),
                old_content=row[3],
                new_content=row[4],
                timestamp=as_utc(datetime.fromisoformat(row[5])),
            )
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self.conn.close()
