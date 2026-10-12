from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .history import resource_guard
from .mt5_client import MT5Client, MT5ConnectionError


BAR_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
]


def normalize_m1_rates(rates, symbol: str) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rates)
    if frame.empty:
        return pd.DataFrame(columns=["symbol", *BAR_COLUMNS, "timestamp_utc"])
    for name in BAR_COLUMNS:
        if name not in frame:
            frame[name] = 0
    frame = frame[BAR_COLUMNS].copy()
    frame.insert(0, "symbol", symbol)
    frame["time"] = frame["time"].astype("int64")
    frame["timestamp_utc"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame = frame.drop_duplicates(subset=["symbol", "time"])
    return frame.sort_values("time", kind="stable").reset_index(drop=True)


def m1_bar_path(data_dir: Path, symbol: str) -> Path:
    return data_dir / "bars" / f"{symbol}_M1.parquet"


def m1_metadata_path(data_dir: Path, symbol: str) -> Path:
    return data_dir / "bars" / f"{symbol}_M1.json"


def load_m1_bars(data_dir: Path, symbol: str) -> pd.DataFrame:
    path = m1_bar_path(data_dir, symbol)
    if not path.exists():
        raise RuntimeError(f"No M1 bars loaded for {symbol}. Run load-m1 first.")
    return pd.read_parquet(path)


def collect_all_m1(
    client: MT5Client,
    symbol: str,
    data_dir: Path,
    chunk_bars: int = 10_000,
    max_bars: int = 2_000_000,
    min_free_gb: float = 3.0,
    min_ram_mb: int = 700,
) -> dict[str, object]:
    if chunk_bars < 100 or chunk_bars > 100_000:
        raise ValueError("chunk_bars must be between 100 and 100000.")
    if max_bars < chunk_bars:
        raise ValueError("max_bars must be at least chunk_bars.")

    root = data_dir / "bars"
    resource_guard(root, min_free_gb, min_ram_mb)
    started_at = datetime.now(timezone.utc)
    pieces: list[pd.DataFrame] = []
    start_pos = 0
    returned = 0
    client.ensure_symbol(symbol)
    warning = None

    while True:
        resource_guard(root, min_free_gb, min_ram_mb)
        try:
            batch = client.history_m1_from_pos(symbol, start_pos, min(chunk_bars, max_bars - returned))
        except MT5ConnectionError as error:
            if not pieces:
                raise
            warning = (
                f"MT5 rejected the next request at position {start_pos:,}: {error}. "
                "The terminal's Max bars in chart or broker history limit was reached."
            )
            break
        frame = normalize_m1_rates(batch, symbol)
        if frame.empty:
            break
        pieces.append(frame)
        rows = len(frame)
        returned += rows
        start_pos += rows
        if returned >= max_bars:
            raise RuntimeError(
                f"MT5 exposed at least {returned:,} M1 bars, reaching the safety limit of {max_bars:,}. "
                "Increase --max-bars deliberately if you need more history."
            )
        if rows < chunk_bars:
            break

    if not pieces:
        raise RuntimeError(f"MT5 returned no M1 bars for {symbol}.")

    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.drop_duplicates(subset=["symbol", "time"])
    frame = frame.sort_values("time", kind="stable").reset_index(drop=True)
    output = m1_bar_path(data_dir, symbol)
    metadata_path = m1_metadata_path(data_dir, symbol)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False, compression="zstd")
    os.replace(temp, output)

    finished_at = datetime.now(timezone.utc)
    metadata = {
        "symbol": symbol,
        "timeframe": "M1",
        "source": "MetaTrader 5 copy_rates_from_pos",
        "loaded_at_utc": finished_at.isoformat(),
        "requested_from": "earliest broker/terminal-available bar",
        "requested_to": finished_at.isoformat(),
        "rows": int(len(frame)),
        "oldest_utc": frame["timestamp_utc"].iloc[0].isoformat(),
        "newest_utc": frame["timestamp_utc"].iloc[-1].isoformat(),
        "chunk_bars": chunk_bars,
        "max_bars": max_bars,
        "duration_seconds": round((finished_at - started_at).total_seconds(), 2),
        "terminal_history_limit_applies": True,
        "note": "MT5 history is limited by broker retention and the terminal Max bars in chart setting.",
    }
    if warning:
        metadata["warning"] = warning
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
