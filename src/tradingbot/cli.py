from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .mt5_client import MT5Client, MT5ConnectionError
from .storage import TickStore


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    try:
        if args.command == "check":
            return check_connection(config)
        if args.command == "discover":
            return discover_symbols(config, args.query)
        if args.command == "symbol-info":
            return symbol_info(config)
        if args.command == "collect-live":
            return collect_live(config, args.duration_seconds)
        if args.command == "export-ticks":
            return export_ticks(config, args.start, args.end)
    except MT5ConnectionError as error:
        print(f"ERROR: {error}")
        return 1

    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument("--config", default="config.json", help="Path to JSON config file")

    parser = argparse.ArgumentParser(
        description="Phase 1 MT5 market-data collector",
        parents=[config_parent],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Check MT5 terminal connection", parents=[config_parent])

    discover = subparsers.add_parser("discover", help="Search broker symbols", parents=[config_parent])
    discover.add_argument("--query", default="XAU", help="Symbol search text")

    subparsers.add_parser(
        "symbol-info",
        help="Show broker contract settings for the configured symbol",
        parents=[config_parent],
    )

    collect = subparsers.add_parser(
        "collect-live",
        help="Record live ticks into SQLite",
        parents=[config_parent],
    )
    collect.add_argument(
        "--duration-seconds",
        type=int,
        default=300,
        help="How long to collect ticks",
    )

    export = subparsers.add_parser(
        "export-ticks",
        help="Export historical ticks to CSV",
        parents=[config_parent],
    )
    export.add_argument("--from", dest="start", required=True, help="UTC start time, ISO format")
    export.add_argument("--to", dest="end", required=True, help="UTC end time, ISO format")
    return parser


def check_connection(config) -> int:
    with MT5Client(config.mt5) as mt5:
        summary = mt5.terminal_summary()
        mt5.ensure_symbol(config.symbol)
        print(json.dumps(summary, indent=2, default=str))
        print(f"Symbol ready: {config.symbol}")
    return 0


def discover_symbols(config, query: str) -> int:
    with MT5Client(config.mt5) as mt5:
        results = mt5.discover_symbols(query)
        if not results:
            print(f"No symbols found for query {query!r}")
            return 0
        for result in results:
            print(
                f"{result['name']} | {result['description']} | "
                f"contract={result['trade_contract_size']} | "
                f"min={result['volume_min']} | step={result['volume_step']} | "
                f"digits={result['digits']} | spread={result['spread']}"
            )
    return 0


def symbol_info(config) -> int:
    with MT5Client(config.mt5) as mt5:
        print(json.dumps(mt5.symbol_summary(config.symbol), indent=2, default=str))
    return 0


def collect_live(config, duration_seconds: int) -> int:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + duration_seconds
    inserted = 0

    store = TickStore(config.sqlite_path)
    try:
        with MT5Client(config.mt5) as mt5:
            mt5.ensure_symbol(config.symbol)
            while time.monotonic() < deadline:
                tick = mt5.latest_tick(config.symbol)
                age_seconds = max(0.0, time.time() - (tick.time_msc / 1000.0))
                if age_seconds <= config.max_tick_age_seconds and store.insert_tick(tick):
                    inserted += 1
                    print(
                        f"{datetime.fromtimestamp(tick.time_msc / 1000.0, timezone.utc).isoformat()} "
                        f"{tick.symbol} bid={tick.bid} ask={tick.ask} stored={inserted}"
                    )
                time.sleep(config.poll_seconds)

        total = store.count_ticks(config.symbol)
        print(f"Done. Inserted {inserted} new ticks. Total stored for {config.symbol}: {total}")
        print(f"SQLite database: {config.sqlite_path}")
        return 0
    finally:
        store.close()


def export_ticks(config, start_raw: str, end_raw: str) -> int:
    start = parse_iso_datetime(start_raw)
    end = parse_iso_datetime(end_raw)
    stamp = f"{start:%Y%m%d_%H%M%S}_{end:%Y%m%d_%H%M%S}"
    output_path = config.data_dir / "history" / f"{config.symbol}_{stamp}_ticks.csv"

    with MT5Client(config.mt5) as mt5:
        rows = mt5.export_ticks(config.symbol, start, end, output_path)

    print(f"Exported {rows} ticks to {output_path}")
    return 0


def parse_iso_datetime(raw: str) -> datetime:
    normalized = raw.replace("Z", "+00:00")
    value = datetime.fromisoformat(normalized)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())
