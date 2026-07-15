from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from widemem.core.types import Fact

_ENV_FLAG = "WIDEMEM_COLLECT_EXTRACTIONS"
_TRUTHY = {"1", "true", "yes", "on"}


def _env_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "").strip().lower() in _TRUTHY


class ExtractionCollector:
    """Logs LLM extraction pairs (input text → extracted facts) as training data.

    Collection is OPT-IN. Because it persists raw, pre-sanitization input text
    (a PII risk), it is disabled unless explicitly enabled via the ``enabled``
    argument or the ``WIDEMEM_COLLECT_EXTRACTIONS=1`` environment variable.
    When disabled, no database is opened and all operations are no-ops.
    """

    def __init__(
        self,
        db_path: str = "~/.widemem/extractions.db",
        enabled: Optional[bool] = None,
    ) -> None:
        self.enabled = _env_enabled() if enabled is None else enabled
        self.conn: Optional[sqlite3.Connection] = None
        if not self.enabled:
            return
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self._init_db()

    def _init_db(self) -> None:
        assert self.conn is not None
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
    ) -> Optional[str]:
        if not self.enabled or self.conn is None:
            return None
        entry_id = str(uuid.uuid4())
        facts_json = json.dumps([{"content": f.content, "importance": f.importance} for f in facts])
        self.conn.execute(
            "INSERT INTO extractions (id, input_text, facts_json, model, timestamp) VALUES (?, ?, ?, ?, ?)",
            (entry_id, input_text, facts_json, model, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()
        return entry_id

    def export(self, output_path: str, limit: Optional[int] = None) -> int:
        if not self.enabled or self.conn is None:
            return 0
        query = "SELECT input_text, facts_json FROM extractions ORDER BY timestamp"
        params: tuple = ()
        if limit:
            query += " LIMIT ?"
            params = (int(limit),)

        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()

        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for input_text, facts_json in rows:
                f.write(json.dumps({"input": input_text, "output": json.loads(facts_json)}) + "\n")

        return len(rows)

    def count(self) -> int:
        if not self.enabled or self.conn is None:
            return 0
        cursor = self.conn.execute("SELECT COUNT(*) FROM extractions")
        return cursor.fetchone()[0]

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
