from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TickRecord:
    symbol: str
    time_msc: int
    bid: float
    ask: float
    last: float
    volume: int
    flags: int


class TickStore:
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.sqlite_path)
        self._init_schema()

    def close(self) -> None:
        self.connection.close()

    def insert_tick(self, tick: TickRecord) -> bool:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO ticks
                (symbol, time_msc, bid, ask, last, volume, flags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tick.symbol,
                tick.time_msc,
                tick.bid,
                tick.ask,
                tick.last,
                tick.volume,
                tick.flags,
            ),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def count_ticks(self, symbol: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM ticks WHERE symbol = ?",
            (symbol,),
        ).fetchone()
        return int(row[0])

    def latest_ticks(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT symbol, time_msc, bid, ask, last, volume, flags, inserted_at
            FROM ticks
            WHERE symbol = ?
            ORDER BY time_msc DESC
            LIMIT ?
            """,
            (symbol, limit),
        ).fetchall()
        columns = ["symbol", "time_msc", "bid", "ask", "last", "volume", "flags", "inserted_at"]
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def _init_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ticks (
                symbol TEXT NOT NULL,
                time_msc INTEGER NOT NULL,
                bid REAL NOT NULL,
                ask REAL NOT NULL,
                last REAL NOT NULL,
                volume INTEGER NOT NULL,
                flags INTEGER NOT NULL,
                inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, time_msc)
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time ON ticks(symbol, time_msc)"
        )
        self.connection.commit()
