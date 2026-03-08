from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from widemem.core.types import Fact


class ExtractionCollector:
    """Logs LLM extraction pairs (input text → extracted facts) as training data."""

    def __init__(self, db_path: str = "~/.widemem/extractions.db") -> None:
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self._init_db()

    def _init_db(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS extractions (
                id TEXT PRIMARY KEY,
                input_text TEXT NOT NULL,
                facts_json TEXT NOT NULL,
                model TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def log(
        self,
        input_text: str,
        facts: List[Fact],
        model: Optional[str] = None,
    ) -> str:
        entry_id = str(uuid.uuid4())
        facts_json = json.dumps([{"content": f.content, "importance": f.importance} for f in facts])
        self.conn.execute(
            "INSERT INTO extractions (id, input_text, facts_json, model, timestamp) VALUES (?, ?, ?, ?, ?)",
            (entry_id, input_text, facts_json, model, datetime.utcnow().isoformat()),
        )
        self.conn.commit()
        return entry_id

    def export(self, output_path: str, limit: Optional[int] = None) -> int:
        query = "SELECT input_text, facts_json FROM extractions ORDER BY timestamp"
        if limit:
            query += f" LIMIT {limit}"

        cursor = self.conn.execute(query)
        rows = cursor.fetchall()

        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for input_text, facts_json in rows:
                f.write(json.dumps({"input": input_text, "output": json.loads(facts_json)}) + "\n")

        return len(rows)

    def count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM extractions")
        return cursor.fetchone()[0]

    def close(self) -> None:
        self.conn.close()
