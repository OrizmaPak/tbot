from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_WINDOWS = (5, 20, 60)


@dataclass(frozen=True)
class TradeAssumptions:
    contract_size: float = 100.0
    lots: float = 0.1
    commission_per_lot_round_turn: float = 7.0
    slippage_price: float = 0.02
    profit_usd: float = 10.0
    loss_usd: float = 10.0
    max_gap_seconds: float = 15.0
    max_spread_price: float = 0.5

    def validate(self) -> None:
        values = asdict(self)
        if any(value < 0 for value in values.values()) or self.contract_size <= 0 or self.lots <= 0:
            raise ValueError("Trade assumptions must be nonnegative, with positive contract size and lots.")
        if self.profit_usd <= 0 or self.loss_usd <= 0:
            raise ValueError("Profit and loss thresholds must be positive.")


def feature_definitions() -> dict[str, str]:
    definitions = {
        "spread": "ask - bid, price units, at decision tick",
        "spread_change": "current spread - prior spread, price units",
        "bid_change": "bid - prior bid, price units",
        "ask_change": "ask - prior ask, price units",
        "tick_interval_s": "seconds since preceding tick, clipped to 300",
        "hour_utc": "UTC hour at decision tick",
        "weekday_utc": "UTC weekday, Monday=0",
        "session": "UTC session category: 0 Asia, 1 Europe, 2 US, 3 late",
    }
    for window in FEATURE_WINDOWS:
        definitions.update({
            f"return_{window}": f"mid / mid {window} ticks ago - 1",
            f"velocity_{window}": f"mid change over {window} ticks / elapsed seconds, price/sec",
            f"range_{window}": f"high - low of last {window} mid ticks, price units",
            f"volatility_{window}": f"standard deviation of last {window} mid returns",
            f"high_distance_{window}": f"current mid - trailing {window}-tick high, price units",
            f"low_distance_{window}": f"current mid - trailing {window}-tick low, price units",
            f"activity_{window}": f"{window} ticks / elapsed seconds",
            f"trend_{window}": f"net mid change / sum absolute changes over last {window} ticks",
        })
    definitions.update({
        "acceleration": "velocity_5 - velocity_20, price/sec",
        "vol_expansion": "volatility_5 / volatility_60",
        "momentum_persistence": "fraction of last 20 mid changes with same sign as 20-tick return",
        "pullback": "distance from last 20-tick high for rising 60-tick trend, else distance from low",
        "breakout": "mid - prior 20-tick high if above, or mid - prior 20-tick low if below, else 0",
        "abnormal_move": "absolute 5-tick return / trailing 60-tick return volatility",
    })
    return definitions


def build_features(ticks: pd.DataFrame) -> pd.DataFrame:
    if ticks.empty:
        raise ValueError("No ticks supplied.")
    frame = ticks.sort_values("time_msc", kind="stable").reset_index(drop=True).copy()
    frame["timestamp_utc"] = pd.to_datetime(frame.time_msc, unit="ms", utc=True)
    mid = (frame.bid + frame.ask) / 2
    delta = mid.diff()
    seconds = frame.time_msc / 1000
    out = frame[["time_msc", "timestamp_utc", "bid", "ask"]].copy()
    out["spread"] = frame.ask - frame.bid
    out["spread_change"] = out.spread.diff()
    out["bid_change"] = frame.bid.diff()
    out["ask_change"] = frame.ask.diff()
    out["tick_interval_s"] = seconds.diff().clip(0, 300)
    out["hour_utc"] = out.timestamp_utc.dt.hour
    out["weekday_utc"] = out.timestamp_utc.dt.dayofweek
    out["session"] = np.select([out.hour_utc < 7, out.hour_utc < 13, out.hour_utc < 21], [0, 1, 2], 3)
    ret = mid.pct_change(fill_method=None)
    for window in FEATURE_WINDOWS:
        elapsed = (seconds - seconds.shift(window)).replace(0, np.nan)
        trailing = mid.rolling(window, min_periods=window)
        out[f"return_{window}"] = mid / mid.shift(window) - 1
        out[f"velocity_{window}"] = (mid - mid.shift(window)) / elapsed
        out[f"range_{window}"] = trailing.max() - trailing.min()
        out[f"volatility_{window}"] = ret.rolling(window, min_periods=window).std()
        out[f"high_distance_{window}"] = mid - trailing.max()
        out[f"low_distance_{window}"] = mid - trailing.min()
        out[f"activity_{window}"] = window / elapsed
        out[f"trend_{window}"] = (mid - mid.shift(window)) / delta.abs().rolling(window).sum().replace(0, np.nan)
    out["acceleration"] = out.velocity_5 - out.velocity_20
    out["vol_expansion"] = out.volatility_5 / out.volatility_60.replace(0, np.nan)
    sign = np.sign(out.return_20)
    out["momentum_persistence"] = (np.sign(delta) == sign).astype(float).rolling(20).mean()
    out["pullback"] = np.where(out.return_60 >= 0, out.high_distance_20, out.low_distance_20)
    prior_high = mid.shift(1).rolling(20).max()
    prior_low = mid.shift(1).rolling(20).min()
    out["breakout"] = np.where(mid > prior_high, mid - prior_high, np.where(mid < prior_low, mid - prior_low, 0))
    out["abnormal_move"] = out.return_5.abs() / out.volatility_60.replace(0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def pnl_usd(direction: str, entry: float, exit_price: float, assumptions: TradeAssumptions) -> float:
    multiplier = 1 if direction == "BUY" else -1
    return (exit_price - entry) * multiplier * assumptions.contract_size * assumptions.lots - assumptions.commission_per_lot_round_turn * assumptions.lots


def label_opportunity(ticks: pd.DataFrame, index: int, direction: str, horizon_s: int,
                      assumptions: TradeAssumptions) -> dict:
    assumptions.validate()
    if direction not in ("BUY", "SELL") or horizon_s <= 0:
        raise ValueError("Invalid direction or horizon.")
    row = ticks.iloc[index]
    entry = float(row.ask if direction == "BUY" else row.bid)
    entry += assumptions.slippage_price * (1 if direction == "BUY" else -1)
    deadline = int(row.time_msc) + horizon_s * 1000
    end_index = int(np.searchsorted(ticks.time_msc.to_numpy(), deadline, side="right"))
    future = ticks.iloc[index + 1:end_index]
    result = {"outcome": "insufficient", "pnl_usd": np.nan, "holding_s": np.nan,
              "entry_price": entry, "exit_price": np.nan}
    if future.empty or int(ticks.time_msc.iloc[-1]) < deadline:
        return result
    previous_time = int(row.time_msc)
    pending_time = None
    pending = []
    def first_hit(candidates):
        if not candidates:
            return None
        losses = [value for value in candidates if value[1] <= -assumptions.loss_usd]
        profits = [value for value in candidates if value[1] >= assumptions.profit_usd]
        return ("loss", losses[0]) if losses else ("profit", profits[0]) if profits else None
    for item in future.itertuples():
        if pending_time is not None and item.time_msc != pending_time:
            hit = first_hit(pending)
            if hit:
                outcome, (exit_price, profit) = hit
                result.update(outcome=outcome, pnl_usd=profit,
                              holding_s=(pending_time - row.time_msc) / 1000, exit_price=exit_price)
                return result
            pending = []
        if (int(item.time_msc) - previous_time) / 1000 > assumptions.max_gap_seconds or item.ask < item.bid or item.bid <= 0:
            return result
        previous_time = int(item.time_msc)
        pending_time = int(item.time_msc)
        exit_price = float(item.bid if direction == "BUY" else item.ask)
        exit_price -= assumptions.slippage_price * (1 if direction == "BUY" else -1)
        profit = pnl_usd(direction, entry, exit_price, assumptions)
        pending.append((exit_price, profit))
    hit = first_hit(pending)
    if hit:
        outcome, (exit_price, profit) = hit
        result.update(outcome=outcome, pnl_usd=profit,
                      holding_s=(pending_time - row.time_msc) / 1000, exit_price=exit_price)
        return result
    last = future.iloc[-1]
    if (deadline - int(last.time_msc)) / 1000 > assumptions.max_gap_seconds:
        return result
    exit_price = float(last.bid if direction == "BUY" else last.ask)
    exit_price -= assumptions.slippage_price * (1 if direction == "BUY" else -1)
    result.update(outcome="timeout", pnl_usd=pnl_usd(direction, entry, exit_price, assumptions),
                  holding_s=(last.time_msc - row.time_msc) / 1000, exit_price=exit_price)
    return result


def create_dataset(ticks: pd.DataFrame, assumptions: TradeAssumptions, sample_seconds: int = 10,
                   horizons: tuple[int, ...] = (60, 120, 180), max_samples: int = 20_000) -> pd.DataFrame:
    if sample_seconds < 1 or max_samples < 1:
        raise ValueError("Invalid sampling limits.")
    features = build_features(ticks)
    times = ticks.time_msc.to_numpy()
    bad_quote = (ticks.bid <= 0) | (ticks.ask <= 0) | (ticks.ask < ticks.bid)
    bad_gap = ticks.time_msc.diff().fillna(0) > assumptions.max_gap_seconds * 1000
    bad_history = (bad_quote | bad_gap).rolling(61, min_periods=61).max().fillna(1).astype(bool)
    selected = []
    last = -10**18
    for index, stamp in enumerate(times):
        if stamp - last >= sample_seconds * 1000 and index >= 61:
            selected.append(index)
            last = stamp
    if len(selected) > max_samples:
        raise RuntimeError(f"{len(selected)} samples exceeds limit {max_samples}; shorten interval or increase sample spacing.")
    columns = list(feature_definitions())
    records = []
    for index in selected:
        row = features.iloc[index]
        if bad_history.iloc[index] or not np.isfinite(row[columns].to_numpy(dtype=float)).all() or row.spread > assumptions.max_spread_price:
            continue
        for horizon in horizons:
            for direction in ("BUY", "SELL"):
                label = label_opportunity(ticks, index, direction, horizon, assumptions)
                records.append({"time_msc": int(row.time_msc), "timestamp_utc": row.timestamp_utc,
                                "direction": direction, "horizon_s": horizon, **{name: float(row[name]) for name in columns},
                                **label})
    return pd.DataFrame.from_records(records)


def save_dataset(dataset: pd.DataFrame, path: Path, assumptions: TradeAssumptions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.parquet")
    dataset.to_parquet(temp, index=False, compression="zstd")
    os.replace(temp, path)
    path.with_suffix(".json").write_text(json.dumps({"features": feature_definitions(),
        "assumptions": asdict(assumptions), "rows": len(dataset)}, indent=2), encoding="utf-8")
