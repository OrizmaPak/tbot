from __future__ import annotations

import sqlite3
import json
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import psutil

from tradingbot.config import load_config
from tradingbot.mt5_client import MT5Client, MT5ConnectionError
from tradingbot.storage import TickStore
from tradingbot.history import available_parts, collect_history
from tradingbot.history import resource_guard
from tradingbot.model import load_model, predict, recommend, train_models
from tradingbot.bars import collect_all_m1, m1_metadata_path


DEFAULT_CONFIG = "config.json"


def main() -> None:
    st.set_page_config(page_title="AI Burst Trading System", layout="wide")
    st.title("AI Burst Trading System")
    st.caption("Local market data and AI research. No live orders are placed.")

    config_path = sidebar()
    try:
        config = load_config(config_path)
    except Exception as error:
        st.error(f"Could not load config: {error}")
        return

    overview_tab, live_tab, history_tab, data_tab, ai_tab, research_tab, backtest_tab, health_tab, config_tab = st.tabs(
        ["Overview", "Live Collector", "Historical Export", "Stored Data", "AI Training", "AI Research", "Backtesting", "System Health", "Config"]
    )

    with overview_tab:
        render_overview(config)

    with live_tab:
        render_live_collector(config)

    with history_tab:
        render_historical_export(config)

    with data_tab:
        render_stored_data(config)

    with ai_tab:
        render_ai_training(config)

    with research_tab:
        render_ai_research(config)

    with backtest_tab:
        render_backtesting(config)

    with health_tab:
        render_health(config)

    with config_tab:
        render_config(config_path)


def sidebar() -> str:
    st.sidebar.header("Control Panel")
    config_path = st.sidebar.text_input("Config file", value=DEFAULT_CONFIG)
    st.sidebar.divider()
    st.sidebar.write("Launch checklist")
    st.sidebar.checkbox("MT5 terminal is open", value=True)
    st.sidebar.checkbox("Broker account is logged in", value=True)
    st.sidebar.checkbox("Gold symbol is selected", value=True)
    st.sidebar.caption("This interface reads data only. Trading controls will come in a later phase.")
    return config_path


def render_overview(config) -> None:
    st.subheader("Connection")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Symbol", config.symbol)
    col_b.metric("Database", config.sqlite_path.name)
    col_c.metric("Poll seconds", config.poll_seconds)

    if st.button("Check MT5 Connection", type="primary"):
        try:
            with MT5Client(config.mt5) as mt5:
                summary = mt5.terminal_summary()
                mt5.ensure_symbol(config.symbol)
                st.success("MT5 connected and symbol is ready.")
                st.json(summary)
        except MT5ConnectionError as error:
            st.error(str(error))

    st.divider()
    st.subheader("Symbol Contract Settings")
    if st.button("Load Symbol Info"):
        try:
            with MT5Client(config.mt5) as mt5:
                info = mt5.symbol_summary(config.symbol)
                show_symbol_metrics(info)
                st.dataframe(pd.DataFrame([info]), width="stretch")
        except MT5ConnectionError as error:
            st.error(str(error))

    st.divider()
    st.subheader("Find Broker Symbols")
    query = st.text_input("Search text", value="XAU")
    if st.button("Search Symbols"):
        try:
            with MT5Client(config.mt5) as mt5:
                results = mt5.discover_symbols(query)
            if results:
                st.dataframe(pd.DataFrame(results), width="stretch")
            else:
                st.info("No matching symbols found.")
        except MT5ConnectionError as error:
            st.error(str(error))


def render_live_collector(config) -> None:
    st.subheader("Live Tick Collector")
    col_a, col_b, col_c = st.columns(3)
    duration_seconds = col_a.number_input("Duration seconds", min_value=5, max_value=3600, value=60, step=5)
    poll_seconds = col_b.number_input(
        "Poll seconds",
        min_value=0.2,
        max_value=10.0,
        value=float(config.poll_seconds),
        step=0.2,
    )
    max_tick_age = col_c.number_input(
        "Max tick age seconds",
        min_value=1,
        max_value=120,
        value=int(config.max_tick_age_seconds),
        step=1,
    )

    st.info("Collection runs inside the browser session. Keep this page open until it finishes.")

    if st.button("Start Live Collection", type="primary"):
        progress = st.progress(0)
        status = st.empty()
        latest = st.empty()
        inserted = 0
        deadline = time.monotonic() + int(duration_seconds)

        store = TickStore(config.sqlite_path)
        try:
            with MT5Client(config.mt5) as mt5:
                mt5.ensure_symbol(config.symbol)
                while time.monotonic() < deadline:
                    tick = mt5.latest_tick(config.symbol)
                    age_seconds = max(0.0, time.time() - (tick.time_msc / 1000.0))
                    if age_seconds <= max_tick_age and store.insert_tick(tick):
                        inserted += 1
                        latest.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "symbol": tick.symbol,
                                        "time_utc": datetime.fromtimestamp(
                                            tick.time_msc / 1000.0,
                                            timezone.utc,
                                        ).isoformat(),
                                        "bid": tick.bid,
                                        "ask": tick.ask,
                                        "spread": round(tick.ask - tick.bid, 5),
                                        "stored": inserted,
                                    }
                                ]
                            ),
                            width="stretch",
                        )

                    elapsed = int(duration_seconds) - max(0, int(deadline - time.monotonic()))
                    progress.progress(min(1.0, elapsed / int(duration_seconds)))
                    status.write(f"Stored {inserted} new ticks.")
                    time.sleep(float(poll_seconds))

            total = store.count_ticks(config.symbol)
            st.success(f"Done. Stored {inserted} new ticks. Total stored for {config.symbol}: {total}.")
        except MT5ConnectionError as error:
            st.error(str(error))
        finally:
            store.close()


def render_historical_export(config) -> None:
    st.subheader("Historical Tick Export")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    default_start = now - timedelta(hours=1)

    col_a, col_b = st.columns(2)
    start_date = col_a.date_input("Start date", value=default_start.date())
    start_time = col_a.time_input("Start time UTC", value=default_start.time())
    end_date = col_b.date_input("End date", value=now.date())
    end_time = col_b.time_input("End time UTC", value=now.time())

    chunk_minutes = st.number_input("Parquet chunk minutes", min_value=1, max_value=1440, value=60)
    if st.button("Collect Resumable Parquet History"):
        start = datetime.combine(start_date, start_time, timezone.utc)
        end = datetime.combine(end_date, end_time, timezone.utc)
        try:
            with st.spinner("Collecting broker history in UTC windows..."):
                with MT5Client(config.mt5) as mt5:
                    reports = collect_history(mt5, config.symbol, start, end, config.data_dir, int(chunk_minutes))
            st.success(f"Processed {len(reports)} windows and {sum(item['ticks'] for item in reports):,} ticks.")
            st.dataframe(pd.DataFrame(reports), width="stretch")
        except (MT5ConnectionError, RuntimeError, ValueError) as error:
            st.error(str(error))

    if st.button("Export Ticks", type="primary"):
        start = datetime.combine(start_date, start_time, timezone.utc)
        end = datetime.combine(end_date, end_time, timezone.utc)
        if end <= start:
            st.error("End time must be after start time.")
            return

        output_path = config.data_dir / "history" / (
            f"{config.symbol}_{start:%Y%m%d_%H%M%S}_{end:%Y%m%d_%H%M%S}_ticks.csv"
        )
        try:
            with MT5Client(config.mt5) as mt5:
                rows = mt5.export_ticks(config.symbol, start, end, output_path)
            st.success(f"Exported {rows} ticks.")
            st.code(str(output_path), language="text")
        except MT5ConnectionError as error:
            st.error(str(error))


def render_stored_data(config) -> None:
    st.subheader("Stored Tick Data")
    col_a, col_b = st.columns([1, 3])
    limit = col_a.number_input("Rows to preview", min_value=10, max_value=1000, value=100, step=10)

    if col_a.button("Refresh Data", type="primary"):
        st.session_state["refresh_data"] = time.time()

    if not config.sqlite_path.exists():
        st.info("No SQLite database found yet. Run live collection first.")
        return

    store = TickStore(config.sqlite_path)
    try:
        total = store.count_ticks(config.symbol)
        col_b.metric(f"Total stored ticks for {config.symbol}", total)
        rows = store.latest_ticks(config.symbol, int(limit))
    finally:
        store.close()

    if rows:
        df = pd.DataFrame(rows)
        # Databases created before Phase 2 do not have a volume_real column.
        # Keep the dashboard usable while their existing rows are migrated.
        if "volume_real" not in df.columns:
            df["volume_real"] = 0.0
        df["time_utc"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
        df["spread"] = df["ask"] - df["bid"]
        st.dataframe(
            df[["time_utc", "symbol", "bid", "ask", "spread", "volume", "volume_real", "flags", "inserted_at"]],
            width="stretch",
        )
    else:
        st.info("The database exists, but no rows are stored for this symbol yet.")

    st.divider()
    st.subheader("History Exports")
    history_dir = config.data_dir / "history"
    files = sorted(history_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if files:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "file": file.name,
                        "size_kb": round(file.stat().st_size / 1024, 2),
                        "path": str(file),
                    }
                    for file in files
                ]
            ),
            width="stretch",
        )
    else:
        st.info("No historical CSV exports found yet.")


def render_config(config_path: str) -> None:
    st.subheader("Configuration")
    path = Path(config_path)
    if not path.exists():
        st.error(f"Config file not found: {path}")
        return
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("password", "api_key", "token"):
        if key in config.get("mt5", {}):
            config["mt5"][key] = "[hidden]"
    st.code(json.dumps(config, indent=2), language="json")


def render_ai_training(config) -> None:
    st.subheader("AI Training")
    st.write("M1 candle source data")
    m1_meta = m1_metadata_path(config.data_dir, config.symbol)
    if st.button("Load all available M1 candles from MT5", type="primary"):
        try:
            with st.spinner("Loading broker-available M1 candles..."):
                with MT5Client(config.mt5) as mt5:
                    report = collect_all_m1(mt5, config.symbol, config.data_dir)
            st.success(f"Loaded {report['rows']:,} M1 candles.")
            st.json(report)
        except (MT5ConnectionError, RuntimeError, ValueError, OSError) as error:
            st.error(str(error))
    if m1_meta.exists():
        metadata = json.loads(m1_meta.read_text(encoding="utf-8"))
        cols = st.columns(4)
        cols[0].metric("M1 candles", f"{metadata['rows']:,}")
        cols[1].metric("Oldest", metadata["oldest_utc"][:10])
        cols[2].metric("Newest", metadata["newest_utc"][:10])
        cols[3].metric("File", f"{m1_meta.with_suffix('.parquet').stat().st_size / 1024**2:.1f} MB")
        st.caption("Loaded source data is ready for the next feature-building step. No training starts from this button.")
    else:
        st.info("No M1 source data loaded yet.")
    st.divider()
    files = sorted((config.data_dir / "datasets").glob("*.parquet"))
    if not files:
        st.info("No feature dataset yet. Collect history, then run make-dataset from PowerShell.")
    else:
        st.dataframe(pd.DataFrame([{"dataset": p.name, "size_mb": round(p.stat().st_size / 1024**2, 2)} for p in files]))
        selected = st.multiselect("Training datasets", files, default=files[-1:], format_func=lambda path: path.name)
        max_rows = st.number_input("Maximum combined rows", min_value=1000, max_value=200000, value=50000, step=1000)
        if st.button("Train LightGBM", type="primary"):
            try:
                resource_guard(config.data_dir)
                with st.spinner("Training, calibrating, and testing chronological splits..."):
                    report = train_models(selected, config.data_dir / "models", int(max_rows))
                st.success(f"Training complete: {report['version']}")
                st.rerun()
            except (RuntimeError, ValueError, OSError) as error:
                st.error(str(error))
    reports = sorted((config.data_dir / "models").glob("*/report.json"), reverse=True)
    if reports:
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        st.metric("Model version", report["version"])
        st.write("Training datasets:", report.get("datasets", [report.get("dataset")]))
        st.dataframe(pd.DataFrame([{"model": name, **{k: v for k, v in metrics.items() if k in ("rows", "brier", "auc", "calibration_reliable", "status", "reason")}} for name, metrics in report["models"].items()]))
        with st.expander("Feature definitions"):
            st.json(report["feature_definitions"])
    else:
        st.info("No trained model yet. Training is available from PowerShell.")


def render_ai_research(config) -> None:
    st.subheader("AI Research")
    report = config.data_dir / "reports" / "patterns.parquet"
    if report.exists():
        st.dataframe(pd.read_parquet(report), width="stretch")
    else:
        st.info("Run the research command after generating a labeled dataset.")
    walk = config.data_dir / "reports" / "walk_forward.json"
    if walk.exists():
        with st.expander("Walk-forward evaluation"):
            st.json(json.loads(walk.read_text(encoding="utf-8")))
    model_reports = sorted((config.data_dir / "models").glob("*/report.json"), reverse=True)
    datasets = sorted((config.data_dir / "datasets").glob("*.parquet"), reverse=True)
    if model_reports and datasets:
        report = json.loads(model_reports[0].read_text(encoding="utf-8"))
        import pyarrow.parquet as pq
        if pq.ParquetFile(datasets[0]).metadata.num_rows <= 50_000:
            data = pd.read_parquet(datasets[0])
            latest = data.sort_values("time_msc").drop_duplicates("time_msc").tail(1)
            if not latest.empty:
                st.write("Latest recorded historical decision:", latest.timestamp_utc.iloc[0])
                predictions = []
                artifacts = {}
                for path in sorted(model_reports[0].parent.glob("*.joblib")):
                    artifact = load_model(path)
                    artifacts[path.stem] = artifact
                    result = predict(artifact, latest).iloc[0]
                    predictions.append({"model": path.stem, "probability": result.profit_probability,
                                        "expected_pnl_usd": result.expected_pnl_usd, "action": result.action})
                if predictions:
                    st.dataframe(pd.DataFrame(predictions), width="stretch")
                    st.caption("These are predictions for a past recorded tick, not current trade instructions.")
                combined = []
                for horizon in (60, 120, 180):
                    if f"BUY_{horizon}" in artifacts and f"SELL_{horizon}" in artifacts:
                        combined.append(recommend(artifacts[f"BUY_{horizon}"], artifacts[f"SELL_{horizon}"], latest).iloc[0].to_dict())
                if combined:
                    st.dataframe(pd.DataFrame(combined), width="stretch")
        with st.expander("Latest feature importance"):
            for name, metrics in report["models"].items():
                if "feature_importance" in metrics:
                    st.write(name)
                    st.bar_chart(pd.Series(metrics["feature_importance"]).sort_values(ascending=False).head(12))
    st.caption("Pattern families are hypotheses. Sample counts and unseen-period evaluation matter more than in-sample win rates.")


def render_backtesting(config) -> None:
    st.subheader("Backtesting")
    reports = sorted((config.data_dir / "reports").glob("backtest*.json"))
    if not reports:
        st.info("No backtest report yet. Run backtest from PowerShell.")
        return
    report = st.selectbox("Report", reports, format_func=lambda path: path.stem)
    trades = report.with_suffix(".parquet")
    if not report.exists() or not trades.exists():
        st.info("No backtest report yet. Run backtest from PowerShell.")
        return
    summary = json.loads(report.read_text(encoding="utf-8"))
    st.dataframe(pd.DataFrame([{"mode": name, **item} for name, item in summary["summary"].items()]), width="stretch")
    frame = pd.read_parquet(trades)
    if not frame.empty:
        frame["time_utc"] = pd.to_datetime(frame.time_msc, unit="ms", utc=True)
        for mode, group in frame.groupby("mode"):
            st.write(mode)
            st.line_chart(group.set_index("time_utc").pnl_usd.cumsum() + summary["limits"]["balance_usd"])
        st.dataframe(frame.tail(100), width="stretch")


def render_health(config) -> None:
    st.subheader("System Health")
    disk = shutil.disk_usage(config.data_dir)
    memory = psutil.virtual_memory()
    cols = st.columns(3)
    cols[0].metric("Free RAM", f"{memory.available / 1024**3:.1f} GB")
    cols[1].metric("Free disk", f"{disk.free / 1024**3:.1f} GB")
    cols[2].metric("CPU load", f"{psutil.cpu_percent(interval=0.1):.0f}%")
    parts = available_parts(config.data_dir, config.symbol)
    st.metric("Parquet history parts", len(parts))
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in (config.data_dir / "raw" / config.symbol).glob("*.json") if p.name != "contract.json"]
    if reports:
        st.write("Available broker history:", min(p["window_start"] for p in reports), "to", max(p["window_end"] for p in reports))
        st.metric("Collected ticks", sum(p["ticks"] for p in reports))
        warnings = [p for p in reports if p.get("empty") or p.get("invalid_quotes") or p.get("gaps_over_threshold")]
        if warnings:
            st.warning(f"{len(warnings)} windows have empty data, gaps, or invalid quotes.")
            st.dataframe(pd.DataFrame(warnings).tail(100), width="stretch")
    model_files = list((config.data_dir / "models").glob("*/*.joblib"))
    st.write(f"Saved models: {len(model_files)}")
    for path in (Path("streamlit.err.log"),):
        if path.exists():
            with st.expander("Recent error log"):
                st.code(path.read_text(encoding="utf-8", errors="replace")[-4000:])


def show_symbol_metrics(info: dict[str, object]) -> None:
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Contract size", info.get("trade_contract_size"))
    col_b.metric("Tick size", info.get("trade_tick_size"))
    col_c.metric("Tick value", info.get("trade_tick_value"))
    col_d.metric("Spread points", info.get("spread"))

    col_e, col_f, col_g, col_h = st.columns(4)
    col_e.metric("Min lot", info.get("volume_min"))
    col_f.metric("Max lot", info.get("volume_max"))
    col_g.metric("Lot step", info.get("volume_step"))
    col_h.metric("Digits", info.get("digits"))


if __name__ == "__main__":
    main()
