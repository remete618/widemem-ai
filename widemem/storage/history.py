from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from widemem.core.types import HistoryEntry, MemoryAction


class HistoryStore:
    def __init__(self, db_path: str = "~/.widemem/history.db") -> None:
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self._init_db()

    def _init_db(self) -> None:
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
            timestamp=datetime.utcnow(),
        )
        self.conn.execute(
            "INSERT INTO history (id, memory_id, action, old_content, new_content, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (entry.id, entry.memory_id, entry.action.value, entry.old_content, entry.new_content, entry.timestamp.isoformat()),
        )
        self.conn.commit()
        return entry

    def get_history(self, memory_id: str) -> list[HistoryEntry]:
        cursor = self.conn.execute(
            "SELECT id, memory_id, action, old_content, new_content, timestamp FROM history WHERE memory_id = ? ORDER BY timestamp",
            (memory_id,),
        )
        return [
            HistoryEntry(
                id=row[0],
                memory_id=row[1],
                action=MemoryAction(row[2]),
                old_content=row[3],
                new_content=row[4],
                timestamp=datetime.fromisoformat(row[5]),
            )
            for row in cursor.fetchall()
        ]

    def close(self) -> None:
        self.conn.close()
