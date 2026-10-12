"""Synthetic fixtures test mechanics only; they are not profitability evidence."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from tradingbot.history import collect_history, load_ticks, normalize_ticks, quality_report
from tradingbot.model import chronological_split
from tradingbot.model import FEATURES, load_model, predict, recommend, train_models
from tradingbot.research import TradeAssumptions, build_features, create_m1_dataset, feature_definitions, label_opportunity, pnl_usd
from tradingbot.simulation import RiskLimits, backtest, performance, simulate_basket
from tradingbot.storage import TickRecord, TickStore
from tradingbot.bars import normalize_m1_rates
import json
import sqlite3
from dataclasses import asdict


def quotes(count=400, step_ms=1000, start=1_700_000_000_000):
    index = np.arange(count)
    bid = 2000 + index * 0.01
    return pd.DataFrame({"time_msc": start + index * step_ms, "bid": bid, "ask": bid + 0.2,
                         "last": np.zeros(count), "volume": np.ones(count, dtype=int),
                         "flags": np.ones(count, dtype=int), "volume_real": np.ones(count)})


def test_full_quote_duplicate_and_utc(tmp_path):
    store = TickStore(tmp_path / "ticks.sqlite")
    first = TickRecord("XAUUSD", 1000, 2000, 2000.2, 0, 1, 2)
    second = TickRecord("XAUUSD", 1000, 2000.1, 2000.3, 0, 1, 2)
    assert store.insert_tick(first)
    assert not store.insert_tick(first)
    assert store.insert_tick(second)
    assert store.count_ticks("XAUUSD") == 2
    store.close()
    frame = normalize_ticks(quotes(2).to_records(index=False), "XAUUSD")
    assert frame.timestamp_utc.dt.tz is not None
    assert frame.timestamp_utc.iloc[0] == pd.Timestamp(1_700_000_000_000, unit="ms", tz="UTC")


def test_m1_normalization_is_utc_sorted_and_deduplicated():
    rates = np.array([(120, 2, 3, 1, 2.5, 10, 4, 10), (60, 1, 2, 0, 1.5, 8, 3, 8),
                      (60, 1, 2, 0, 1.5, 8, 3, 8)], dtype=[
        ("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
        ("close", "f8"), ("tick_volume", "i8"), ("spread", "i8"), ("real_volume", "i8")])
    frame = normalize_m1_rates(rates, "XAUUSD")
    assert len(frame) == 2
    assert frame.time.tolist() == [60, 120]
    assert str(frame.timestamp_utc.dt.tz) == "UTC"
    assert frame.timestamp_utc.iloc[0] == pd.Timestamp(60, unit="s", tz="UTC")


def test_m1_dataset_uses_candle_history_and_shared_model_contract():
    count = 400
    times = 1_700_000_000 + np.arange(count) * 60
    close = 2000 + np.arange(count) * 0.01
    bars = pd.DataFrame({"time": times, "open": close, "high": close + 0.2,
                         "low": close - 0.2, "close": close, "tick_volume": 1,
                         "spread": 1, "real_volume": 1})
    assumptions = TradeAssumptions(profit_usd=1, loss_usd=1, slippage_price=0,
                                   commission_per_lot_round_turn=0)
    dataset = create_m1_dataset(bars, assumptions, sample_seconds=300, max_samples=100)
    assert not dataset.empty
    assert set(dataset.direction) == {"BUY", "SELL"}
    assert set(dataset.horizon_s) == {60, 120, 180}
    assert set(feature_definitions()).issubset(dataset.columns)


def test_history_collection_resume_and_distinct_same_millisecond(tmp_path):
    class FakeBroker:
        calls = 0

        def symbol_summary(self, symbol):
            return {"name": symbol, "trade_contract_size": 100.0}

        def history_ticks(self, symbol, start, end):
            self.calls += 1
            stamp = int(start.timestamp() * 1000)
            return pd.DataFrame({"time": [stamp // 1000] * 3, "time_msc": [stamp] * 3,
                                 "bid": [2000, 2000.1, 2000], "ask": [2000.2, 2000.3, 2000.2],
                                 "last": [0.0] * 3, "volume": [1] * 3, "flags": [2] * 3,
                                 "volume_real": [1.0] * 3}).to_records(index=False)

    broker = FakeBroker()
    start = datetime(2026, 10, 12, tzinfo=timezone.utc)
    end = pd.Timestamp(start) + pd.Timedelta(minutes=2)
    args = (broker, "XAUUSD", start, end, tmp_path)
    reports = collect_history(*args, chunk_minutes=1, min_free_gb=0, min_ram_mb=0)
    assert [item["ticks"] for item in reports] == [2, 2]
    assert broker.calls == 2
    collect_history(*args, chunk_minutes=1, min_free_gb=0, min_ram_mb=0)
    assert broker.calls == 2
    loaded = load_ticks(tmp_path, "XAUUSD", start, end)
    assert len(loaded) == 4


def test_phase1_sqlite_migration(tmp_path):
    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE ticks (symbol TEXT, time_msc INTEGER, bid REAL, ask REAL,
                            last REAL, volume INTEGER, flags INTEGER, inserted_at TEXT)""")
        connection.execute("INSERT INTO ticks VALUES ('XAUUSD', 1000, 2000, 2000.2, 0, 1, 2, '2026-01-01')")
    store = TickStore(path)
    assert store.count_ticks("XAUUSD") == 1
    store.close()
    store = TickStore(path)
    assert store.count_ticks("XAUUSD") == 1
    store.close()


def test_features_use_only_history():
    base = quotes(100)
    before = build_features(base).iloc[80].copy()
    changed = base.copy()
    changed.loc[81:, ["bid", "ask"]] += 500
    after = build_features(changed).iloc[80]
    for name in ("return_20", "velocity_5", "volatility_60", "breakout", "spread"):
        assert before[name] == after[name]


def test_labels_buy_sell_spread_and_gaps():
    data = quotes(250)
    settings = TradeAssumptions(profit_usd=1, loss_usd=3, slippage_price=0, commission_per_lot_round_turn=0)
    assert label_opportunity(data, 0, "BUY", 180, settings)["outcome"] == "profit"
    assert label_opportunity(data, 0, "SELL", 180, TradeAssumptions(profit_usd=1, loss_usd=1, slippage_price=0, commission_per_lot_round_turn=0))["outcome"] == "loss"
    assert pnl_usd("BUY", 2000.2, 2000.2, settings) == 0
    assert pnl_usd("SELL", 2000, 2000.2, settings) < 0
    gap = data.copy()
    gap.loc[1:, "time_msc"] += 20_000
    assert label_opportunity(gap, 0, "BUY", 180, settings)["outcome"] == "insufficient"
    assert label_opportunity(data, 240, "BUY", 180, settings)["outcome"] == "insufficient"
    ambiguous = pd.DataFrame({"time_msc": [0, 1000, 1000, 2000, 3000],
                              "bid": [100, 101, 99, 100, 100],
                              "ask": [100, 101, 99, 100, 100]})
    conservative = TradeAssumptions(contract_size=1, lots=1, profit_usd=1, loss_usd=1,
                                     slippage_price=0, commission_per_lot_round_turn=0)
    assert label_opportunity(ambiguous, 0, "BUY", 2, conservative)["outcome"] == "loss"


def test_quality_and_chronological_purge():
    data = quotes(400)
    data.loc[5:, "time_msc"] += 60_000
    report = quality_report(normalize_ticks(data.to_records(index=False), "XAUUSD"), max_gap_seconds=30)
    assert report["gaps_over_threshold"] == 1
    frame = pd.DataFrame({"time_msc": np.arange(500) * 10_000,
                          "timestamp_utc": pd.to_datetime(np.arange(500) * 10_000, unit="ms", utc=True)})
    train, valid, test = chronological_split(frame, 180)
    assert valid.time_msc.min() - train.time_msc.max() >= 180_000
    assert test.time_msc.min() - valid.time_msc.max() >= 180_000


def test_basket_risk_accounting():
    data = quotes(250)
    settings = TradeAssumptions(profit_usd=2, loss_usd=10, slippage_price=0)
    limits = RiskLimits(max_spread_price=0.5)
    burst = simulate_basket(data, 0, "BUY", settings, limits, positions=3)
    single = simulate_basket(data, 0, "BUY", settings, limits, positions=3, single_equivalent=True)
    assert burst["weighted_entry"] >= data.ask.iloc[0]
    assert burst["lots"] <= limits.max_lots + 1e-9
    assert single["lots"] == pytest.approx(0.3)
    assert burst["commission_usd"] == pytest.approx(7 * burst["lots"])
    with pytest.raises(ValueError):
        simulate_basket(data, 0, "BUY", TradeAssumptions(loss_usd=51), limits)
    with pytest.raises(RuntimeError):
        simulate_basket(data, 0, "BUY", settings, RiskLimits(emergency_stop=True))
    with pytest.raises(ValueError):
        simulate_basket(data, 0, "BUY", TradeAssumptions(slippage_price=0.1), limits)
    signals = pd.DataFrame({"time_msc": [int(data.time_msc.iloc[0])], "direction": ["BUY"], "session": [1]})
    trades, report = backtest(data, signals, settings, limits)
    assert set(trades["mode"]) == {"burst", "single_equal_exposure"}
    assert all(item["baskets"] == 1 for item in report["summary"].values())
    stopped, _ = backtest(data, signals, settings, limits, starting_equity=4499)
    assert stopped.empty


def test_model_serialization_and_no_trade_gate(tmp_path):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(500, len(FEATURES))), columns=FEATURES)
    frame["spread"] = 0.2
    frame["time_msc"] = 1_700_000_000_000 + np.arange(500) * 30_000
    frame["timestamp_utc"] = pd.to_datetime(frame.time_msc, unit="ms", utc=True)
    frame["direction"] = "BUY"
    frame["horizon_s"] = 60
    frame["outcome"] = np.where(rng.random(500) > 0.5, "profit", "loss")
    frame["pnl_usd"] = np.where(frame.outcome.eq("profit"), 10.0, -10.0)
    path = tmp_path / "synthetic.parquet"
    frame.to_parquet(path)
    path.with_suffix(".json").write_text(json.dumps({"features": feature_definitions(),
                                                       "assumptions": asdict(TradeAssumptions())}))
    report = train_models(path, tmp_path / "models")
    artifact = load_model(tmp_path / "models" / report["version"] / "BUY_60.joblib")
    result = predict(artifact, frame.iloc[:3])
    assert len(result) == 3
    assert result.action.eq("NO TRADE").all()
    opposite = {**artifact, "direction": "SELL"}
    assert recommend(artifact, opposite, frame.iloc[:3]).recommended_action.eq("NO TRADE").all()
