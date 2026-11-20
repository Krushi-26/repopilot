"""SQLite response cache: identical (model, messages, params) never hit the API twice.

This keeps free-tier quota for new work and makes eval reruns nearly free.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY, provider TEXT, model TEXT, response TEXT, created REAL)"
        )

    @staticmethod
    def key(model: str, messages: list[dict], params: dict) -> str:
        blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        row = self.conn.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, provider: str, model: str, response: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?)",
            (key, provider, model, json.dumps(response), time.time()),
        )
        self.conn.commit()
