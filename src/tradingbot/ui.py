from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from tradingbot.config import load_config
from tradingbot.mt5_client import MT5Client, MT5ConnectionError
from tradingbot.storage import TickStore


DEFAULT_CONFIG = "config.json"


def main() -> None:
    st.set_page_config(page_title="AI Burst Trading System", layout="wide")
    st.title("AI Burst Trading System")
    st.caption("Phase 1: MT5 connection, XAUUSD market data, and tick recording. No trades are placed.")

    config_path = sidebar()
    try:
        config = load_config(config_path)
    except Exception as error:
        st.error(f"Could not load config: {error}")
        return

    overview_tab, live_tab, history_tab, data_tab, config_tab = st.tabs(
        ["Overview", "Live Collector", "Historical Export", "Stored Data", "Config"]
    )

    with overview_tab:
        render_overview(config)

    with live_tab:
        render_live_collector(config)

    with history_tab:
        render_historical_export(config)

    with data_tab:
        render_stored_data(config)

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
                st.dataframe(pd.DataFrame([info]), use_container_width=True)
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
                st.dataframe(pd.DataFrame(results), use_container_width=True)
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
                            use_container_width=True,
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
        df["time_utc"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
        df["spread"] = df["ask"] - df["bid"]
        st.dataframe(
            df[["time_utc", "symbol", "bid", "ask", "spread", "volume", "flags", "inserted_at"]],
            use_container_width=True,
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
            use_container_width=True,
        )
    else:
        st.info("No historical CSV exports found yet.")


def render_config(config_path: str) -> None:
    st.subheader("Configuration")
    path = Path(config_path)
    if not path.exists():
        st.error(f"Config file not found: {path}")
        return
    st.code(path.read_text(encoding="utf-8"), language="json")


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
