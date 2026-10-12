from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MT5Config:
    terminal_path: str | None = None
    login: int | None = None
    server: str | None = None
    password_env: str = "MT5_PASSWORD"
    timeout_ms: int = 60_000
    portable: bool = False


@dataclass(frozen=True)
class AppConfig:
    symbol: str
    data_dir: Path
    poll_seconds: float
    max_tick_age_seconds: int
    mt5: MT5Config

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "market_ticks.sqlite"


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = json.load(file)

    data_dir = Path(raw.get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = config_path.parent / data_dir

    mt5_raw = raw.get("mt5", {})
    mt5 = MT5Config(
        terminal_path=mt5_raw.get("terminal_path"),
        login=mt5_raw.get("login"),
        server=mt5_raw.get("server"),
        password_env=mt5_raw.get("password_env", "MT5_PASSWORD"),
        timeout_ms=int(mt5_raw.get("timeout_ms", 60_000)),
        portable=bool(mt5_raw.get("portable", False)),
    )

    return AppConfig(
        symbol=str(raw.get("symbol", "XAUUSD")),
        data_dir=data_dir,
        poll_seconds=float(raw.get("poll_seconds", 1.0)),
        max_tick_age_seconds=int(raw.get("max_tick_age_seconds", 10)),
        mt5=mt5,
    )
