from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from .mt5_client import MT5Client


TICK_COLUMNS = ["time", "time_msc", "bid", "ask", "last", "volume", "flags", "volume_real"]


def resource_guard(path: Path, min_free_gb: float = 3.0, min_ram_mb: int = 700) -> None:
    path.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(path)
    ram = psutil.virtual_memory()
    if disk.free < min_free_gb * 1024**3:
        raise RuntimeError(f"Only {disk.free / 1024**3:.1f} GB disk free; need {min_free_gb:.1f} GB.")
    if ram.available < min_ram_mb * 1024**2:
        raise RuntimeError(f"Only {ram.available / 1024**2:.0f} MB RAM available; need {min_ram_mb} MB.")


def normalize_ticks(ticks, symbol: str) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(ticks)
    if frame.empty:
        return pd.DataFrame(columns=["symbol", *TICK_COLUMNS, "timestamp_utc"])
    for name in TICK_COLUMNS:
        if name not in frame:
            frame[name] = 0
    frame = frame[TICK_COLUMNS].copy()
    frame.insert(0, "symbol", symbol)
    frame["timestamp_utc"] = pd.to_datetime(frame.time_msc, unit="ms", utc=True)
    # Identical full records from overlapping MT5 requests are repeats. Distinct
    # quotes at the same millisecond remain separate observations.
    frame = frame.drop_duplicates(subset=["symbol", *TICK_COLUMNS])
    return frame.sort_values("time_msc", kind="stable").reset_index(drop=True)


def quality_report(frame: pd.DataFrame, max_gap_seconds: float = 30, max_spread: float = 1.0) -> dict:
    if frame.empty:
        return {"ticks": 0, "empty": True, "warning": "No broker ticks returned in this interval."}
    gap = frame.time_msc.diff().fillna(0) / 1000
    spread = frame.ask - frame.bid
    return {
        "ticks": int(len(frame)),
        "first_utc": frame.timestamp_utc.iloc[0].isoformat(),
        "last_utc": frame.timestamp_utc.iloc[-1].isoformat(),
        "gaps_over_threshold": int((gap > max_gap_seconds).sum()),
        "largest_gap_seconds": float(gap.max()),
        "invalid_quotes": int(((frame.bid <= 0) | (frame.ask <= 0) | (spread < 0)).sum()),
        "spreads_over_threshold": int((spread > max_spread).sum()),
        "max_spread": float(spread.max()),
    }


def collect_history(
    client: MT5Client, symbol: str, start: datetime, end: datetime, data_dir: Path,
    chunk_minutes: int = 60, max_ticks_per_chunk: int = 250_000,
    min_free_gb: float = 3.0, min_ram_mb: int = 700,
    max_gap_seconds: float = 30, max_spread: float = 1.0,
) -> list[dict]:
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("Start and end must be timezone-aware and end must follow start.")
    if not 1 <= chunk_minutes <= 1440 or max_ticks_per_chunk < 1000:
        raise ValueError("Invalid chunk size or tick limit.")
    start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
    root = data_dir / "raw" / symbol
    resource_guard(root, min_free_gb, min_ram_mb)
    info = client.symbol_summary(symbol)
    (root / "contract.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    reports = []
    cursor = start
    while cursor < end:
        stop = min(cursor + timedelta(minutes=chunk_minutes), end)
        key = f"{cursor:%Y%m%dT%H%M%S%fZ}_{stop:%Y%m%dT%H%M%S%fZ}"
        part = root / f"{key}.parquet"
        report_path = root / f"{key}.json"
        if part.exists() and report_path.exists():
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            cursor = stop
            continue
        resource_guard(root, min_free_gb, min_ram_mb)
        # MT5 range endpoints are inclusive. Keep [cursor, stop) so adjacent
        # chunks do not duplicate boundary ticks.
        ticks = client.history_ticks(symbol, cursor, stop)
        if len(ticks) > max_ticks_per_chunk:
            raise RuntimeError(
                f"MT5 returned {len(ticks)} ticks in one window (limit {max_ticks_per_chunk}); "
                "retry with smaller --chunk-minutes."
            )
        frame = normalize_ticks(ticks, symbol)
        frame = frame[(frame.timestamp_utc >= cursor) & (frame.timestamp_utc < stop)]
        report = {"symbol": symbol, "window_start": cursor.isoformat(), "window_end": stop.isoformat(),
                  **quality_report(frame, max_gap_seconds, max_spread)}
        if not frame.empty:
            temp = part.with_suffix(".tmp.parquet")
            frame.to_parquet(temp, index=False, compression="zstd")
            os.replace(temp, part)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports.append(report)
        cursor = stop
    return reports


def available_parts(data_dir: Path, symbol: str) -> list[Path]:
    return sorted((data_dir / "raw" / symbol).glob("*.parquet"))


def load_ticks(data_dir: Path, symbol: str, start: datetime, end: datetime, max_rows: int = 400_000) -> pd.DataFrame:
    if end <= start:
        raise ValueError("End must follow start.")
    pieces = []
    count = 0
    for path in available_parts(data_dir, symbol):
        name = path.stem.split("_")
        if len(name) != 2:
            continue
        part_start = datetime.strptime(name[0], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
        part_end = datetime.strptime(name[1], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
        if part_end <= start or part_start >= end:
            continue
        import pyarrow.parquet as pq
        rows = pq.ParquetFile(path).metadata.num_rows
        count += rows
        if count > max_rows:
            raise RuntimeError(f"Requested window exceeds {max_rows} raw ticks. Use a shorter interval or raise --max-rows cautiously.")
        piece = pd.read_parquet(path)
        pieces.append(piece)
    if not pieces:
        raise RuntimeError("No collected broker tick data overlaps the requested interval.")
    frame = pd.concat(pieces, ignore_index=True)
    frame["timestamp_utc"] = pd.to_datetime(frame.time_msc, unit="ms", utc=True)
    frame = frame[(frame.timestamp_utc >= start) & (frame.timestamp_utc < end)]
    frame = frame.drop_duplicates(subset=["symbol", *TICK_COLUMNS])
    return frame.sort_values("time_msc", kind="stable").reset_index(drop=True)
