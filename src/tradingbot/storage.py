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
    volume_real: float = 0.0


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
            INSERT OR IGNORE INTO ticks_v2
                (symbol, time_msc, bid, ask, last, volume, flags, volume_real)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tick.symbol,
                tick.time_msc,
                tick.bid,
                tick.ask,
                tick.last,
                tick.volume,
                tick.flags,
                tick.volume_real,
            ),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def count_ticks(self, symbol: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM ticks_v2 WHERE symbol = ?",
            (symbol,),
        ).fetchone()
        return int(row[0])

    def latest_ticks(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT symbol, time_msc, bid, ask, last, volume, flags, volume_real, inserted_at
            FROM ticks_v2
            WHERE symbol = ?
            ORDER BY time_msc DESC
            LIMIT ?
            """,
            (symbol, limit),
        ).fetchall()
        columns = ["symbol", "time_msc", "bid", "ask", "last", "volume", "flags", "volume_real", "inserted_at"]
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
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS ticks_v2 (
                symbol TEXT NOT NULL, time_msc INTEGER NOT NULL,
                bid REAL NOT NULL, ask REAL NOT NULL, last REAL NOT NULL,
                volume INTEGER NOT NULL, flags INTEGER NOT NULL,
                volume_real REAL NOT NULL DEFAULT 0,
                inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, time_msc, bid, ask, last, volume, flags, volume_real)
            )"""
        )
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version < 2:
            self.connection.execute(
                """INSERT OR IGNORE INTO ticks_v2
                   (symbol, time_msc, bid, ask, last, volume, flags, inserted_at)
                   SELECT symbol, time_msc, bid, ask, last, volume, flags, inserted_at FROM ticks"""
            )
            self.connection.execute("PRAGMA user_version = 2")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_ticks_v2_symbol_time ON ticks_v2(symbol, time_msc)")
        self.connection.commit()
