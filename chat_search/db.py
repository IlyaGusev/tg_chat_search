from pathlib import Path
from typing import Optional

import aiosqlite


class QueryLogger:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    async def init_db(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    results_count INTEGER,
                    error TEXT
                )
                """
            )
            await db.commit()

    async def log_query(
        self,
        query: str,
        top_k: int,
        results_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO queries (query, top_k, results_count, error)
                VALUES (?, ?, ?, ?)
                """,
                (query, top_k, results_count, error),
            )
            await db.commit()
