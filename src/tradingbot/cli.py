from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import load_config
from .mt5_client import MT5Client, MT5ConnectionError
from .storage import TickStore
from .history import collect_history, load_ticks, resource_guard
from .research import TradeAssumptions, create_dataset, save_dataset
from .model import train_models, walk_forward_summary
from .simulation import RiskLimits, backtest, performance, research_patterns
import pandas as pd


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
        if args.command == "collect-history":
            with MT5Client(config.mt5) as mt5:
                reports = collect_history(mt5, config.symbol, parse_iso_datetime(args.start),
                    parse_iso_datetime(args.end), config.data_dir, args.chunk_minutes,
                    args.max_ticks_per_chunk)
            warnings = [x for x in reports if x.get("empty") or x.get("invalid_quotes") or x.get("gaps_over_threshold")]
            print(json.dumps({"windows": len(reports), "ticks": sum(x["ticks"] for x in reports),
                              "warning_windows": len(warnings), "first_warnings": warnings[:5]}, indent=2))
            return 0
        if args.command == "make-dataset":
            resource_guard(config.data_dir)
            ticks = load_ticks(config.data_dir, config.symbol, parse_iso_datetime(args.start),
                               parse_iso_datetime(args.end), args.max_rows)
            contract_path = config.data_dir / "raw" / config.symbol / "contract.json"
            if not contract_path.exists():
                raise RuntimeError("Broker contract snapshot is missing. Collect history first.")
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            assumptions = TradeAssumptions(contract_size=float(contract["trade_contract_size"]),
                lots=args.lots, commission_per_lot_round_turn=args.commission_per_lot,
                slippage_price=args.slippage_price, profit_usd=args.profit_usd,
                loss_usd=args.loss_usd, max_spread_price=args.max_spread_price)
            assumptions.validate()
            dataset = create_dataset(ticks, assumptions, args.sample_seconds, max_samples=args.max_samples)
            if dataset.empty:
                raise RuntimeError("No valid labeled samples. Collect more history or inspect quality warnings.")
            stamp = f"{parse_iso_datetime(args.start):%Y%m%dT%H%M%SZ}_{parse_iso_datetime(args.end):%Y%m%dT%H%M%SZ}"
            path = config.data_dir / "datasets" / f"{config.symbol}_{stamp}.parquet"
            save_dataset(dataset, path, assumptions)
            print(f"Saved {len(dataset)} labeled rows to {path}; outcomes: {dataset.outcome.value_counts().to_dict() if len(dataset) else {}}")
            return 0
        if args.command == "train":
            resource_guard(config.data_dir)
            report = train_models([Path(p) for p in args.dataset], config.data_dir / "models", args.max_rows, args.threads)
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "evaluate":
            dataset = load_datasets(args.dataset, args.max_rows)
            report = {f"{direction}_{horizon}": walk_forward_summary(dataset, horizon, direction)
                      for direction in ("BUY", "SELL") for horizon in (60, 120, 180)}
            path = config.data_dir / "reports" / "walk_forward.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "research":
            result = research_patterns(load_datasets(args.dataset, args.max_rows))
            path = config.data_dir / "reports" / "patterns.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(path, index=False)
            print(result.to_string(index=False))
            return 0
        if args.command == "backtest":
            if not args.report_name.replace("_", "").isalnum():
                raise ValueError("Report name may contain only letters, digits, and underscores.")
            dataset = load_datasets(args.dataset, args.max_dataset_rows)
            definitions = [json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8")) for path in args.dataset]
            if any(item["assumptions"] != definitions[0]["assumptions"] for item in definitions):
                raise RuntimeError("Backtest datasets have different execution assumptions.")
            assumptions = TradeAssumptions(**definitions[0]["assumptions"])
            limits = RiskLimits(balance_usd=args.balance_usd, basket_risk_fraction=args.risk_fraction,
                max_lots=args.max_lots, max_positions=args.positions, max_spread_price=args.max_spread_price,
                max_duration_s=args.max_duration_s, max_daily_drawdown_fraction=args.max_daily_drawdown_fraction,
                max_total_drawdown_fraction=args.max_total_drawdown_fraction,
                max_slippage_price=args.max_slippage_price, cooldown_s=args.cooldown_s,
                emergency_stop=args.emergency_stop)
            signals = dataset[dataset.horizon_s == 180]
            signals = signals[(signals.direction.eq("BUY") & (signals.return_20 > 0) & (signals.acceleration > 0)) |
                              (signals.direction.eq("SELL") & (signals.return_20 < 0) & (signals.acceleration < 0))]
            if args.session is not None:
                signals = signals[signals.session.eq(args.session)]
            start = parse_iso_datetime(args.start)
            end = parse_iso_datetime(args.end)
            if end <= start:
                raise ValueError("End must follow start.")
            slices = []
            cursor = start
            equity = limits.balance_usd
            while cursor < end:
                stop = min(cursor + timedelta(days=1), end)
                selected = signals[(signals.time_msc >= int(cursor.timestamp() * 1000)) &
                                   (signals.time_msc < int(stop.timestamp() * 1000))]
                if not selected.empty:
                    ticks = load_ticks(config.data_dir, config.symbol, cursor, stop, args.max_rows)
                    frame, _ = backtest(ticks, selected, assumptions, limits, args.positions, equity)
                    if not frame.empty:
                        slices.append(frame)
                        equity += float(frame.loc[frame["mode"].eq("burst"), "pnl_usd"].sum())
                if equity <= limits.balance_usd * (1 - limits.max_total_drawdown_fraction):
                    break
                cursor = stop
            result = pd.concat(slices, ignore_index=True) if slices else pd.DataFrame()
            summary = {mode: performance(group, limits.balance_usd) for mode, group in result.groupby("mode")} if not result.empty else {}
            regimes = {f"{mode}_session{session}_{regime}": performance(group, limits.balance_usd)
                       for (mode, session, regime), group in result.groupby(["mode", "session", "volatility_regime"])} if not result.empty else {}
            report = {"assumptions": definitions[0]["assumptions"], "limits": vars(limits),
                      "period": [start.isoformat(), end.isoformat()], "session_filter": args.session,
                      "summary": summary, "regimes": regimes,
                      "comparison": "Single entry uses the lots filled by the paired burst (ex post).",
                      "warning": "Historical quotes do not guarantee executable fills or future returns."}
            path = config.data_dir / "reports"
            path.mkdir(parents=True, exist_ok=True)
            result.to_parquet(path / f"{args.report_name}.parquet", index=False)
            (path / f"{args.report_name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            return 0
    except (MT5ConnectionError, RuntimeError, ValueError, OSError) as error:
        print(f"ERROR: {error}")
        return 1

    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument("--config", default="config.json", help="Path to JSON config file")

    parser = argparse.ArgumentParser(
        description="Local MT5 market-data and AI research tools (no order execution)",
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
    history = subparsers.add_parser("collect-history", parents=[config_parent])
    history.add_argument("--from", dest="start", required=True)
    history.add_argument("--to", dest="end", required=True)
    history.add_argument("--chunk-minutes", type=int, default=60)
    history.add_argument("--max-ticks-per-chunk", type=int, default=250_000)
    dataset = subparsers.add_parser("make-dataset", parents=[config_parent])
    dataset.add_argument("--from", dest="start", required=True)
    dataset.add_argument("--to", dest="end", required=True)
    dataset.add_argument("--sample-seconds", type=int, default=10)
    dataset.add_argument("--max-rows", type=int, default=400_000)
    dataset.add_argument("--max-samples", type=int, default=20_000)
    dataset.add_argument("--lots", type=float, default=0.1)
    dataset.add_argument("--commission-per-lot", type=float, default=7.0, help="USD round trip")
    dataset.add_argument("--slippage-price", type=float, default=0.02)
    dataset.add_argument("--profit-usd", type=float, default=10.0)
    dataset.add_argument("--loss-usd", type=float, default=10.0)
    dataset.add_argument("--max-spread-price", type=float, default=0.5)
    train = subparsers.add_parser("train", parents=[config_parent])
    train.add_argument("--dataset", required=True, nargs="+")
    train.add_argument("--max-rows", type=int, default=50_000)
    train.add_argument("--threads", type=int, default=2)
    for name in ("evaluate", "research"):
        item = subparsers.add_parser(name, parents=[config_parent])
        item.add_argument("--dataset", required=True, nargs="+")
        item.add_argument("--max-rows", type=int, default=100_000)
    back = subparsers.add_parser("backtest", parents=[config_parent])
    back.add_argument("--dataset", required=True, nargs="+")
    back.add_argument("--from", dest="start", required=True)
    back.add_argument("--to", dest="end", required=True)
    back.add_argument("--max-rows", type=int, default=400_000)
    back.add_argument("--max-dataset-rows", type=int, default=100_000)
    back.add_argument("--balance-usd", type=float, default=5000.0)
    back.add_argument("--risk-fraction", type=float, default=0.01)
    back.add_argument("--max-lots", type=float, default=0.3)
    back.add_argument("--positions", type=int, default=3)
    back.add_argument("--max-spread-price", type=float, default=0.5)
    back.add_argument("--max-duration-s", type=int, default=180)
    back.add_argument("--max-daily-drawdown-fraction", type=float, default=0.03)
    back.add_argument("--max-total-drawdown-fraction", type=float, default=0.10)
    back.add_argument("--max-slippage-price", type=float, default=0.05)
    back.add_argument("--cooldown-s", type=int, default=180)
    back.add_argument("--emergency-stop", action="store_true")
    back.add_argument("--report-name", default="backtest")
    back.add_argument("--session", type=int, choices=range(4), help="Evaluate one UTC session independently")
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


def load_datasets(paths: list[str], max_rows: int) -> pd.DataFrame:
    import pyarrow.parquet as pq
    if sum(pq.ParquetFile(path).metadata.num_rows for path in paths) > max_rows:
        raise RuntimeError(f"Combined datasets exceed {max_rows} rows.")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True).drop_duplicates(
        subset=["time_msc", "direction", "horizon_s"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
