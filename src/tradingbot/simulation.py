from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .research import TradeAssumptions


@dataclass(frozen=True)
class RiskLimits:
    balance_usd: float = 5000.0
    basket_risk_fraction: float = 0.01
    max_daily_drawdown_fraction: float = 0.03
    max_total_drawdown_fraction: float = 0.10
    max_lots: float = 0.3
    max_positions: int = 3
    max_spread_price: float = 0.5
    max_slippage_price: float = 0.05
    max_duration_s: int = 180
    cooldown_s: int = 180
    emergency_stop: bool = False

    def check(self, assumptions: TradeAssumptions, entries: int) -> None:
        assumptions.validate()
        if self.emergency_stop:
            raise RuntimeError("Emergency stop is active.")
        if self.balance_usd <= 0 or self.max_lots <= 0 or self.max_positions < 1 or self.max_duration_s < 1 or self.cooldown_s < 0 or self.max_spread_price < 0 or self.max_slippage_price < 0:
            raise ValueError("Risk limits contain invalid numeric values.")
        if not 0 < self.basket_risk_fraction <= 0.01 or not 0 < self.max_daily_drawdown_fraction <= 0.1 or not 0 < self.max_total_drawdown_fraction <= 0.9:
            raise ValueError("Risk fractions outside allowed range.")
        if assumptions.loss_usd > self.balance_usd * self.basket_risk_fraction:
            raise ValueError("Basket loss boundary exceeds risk budget.")
        if assumptions.slippage_price > self.max_slippage_price:
            raise ValueError("Slippage exceeds risk limit.")
        if entries > self.max_positions or assumptions.lots * entries > self.max_lots + 1e-9:
            raise ValueError("Burst exposure exceeds position or lot limit.")


def _basket_pnl(direction: str, entries: list[tuple[float, float]], exit_price: float,
                assumptions: TradeAssumptions) -> float:
    sign = 1 if direction == "BUY" else -1
    gross = sum((exit_price - price) * sign * assumptions.contract_size * lots for price, lots in entries)
    costs = assumptions.commission_per_lot_round_turn * sum(lots for _, lots in entries)
    return gross - costs


def simulate_basket(ticks: pd.DataFrame, index: int, direction: str, assumptions: TradeAssumptions,
                    limits: RiskLimits, positions: int = 3, entry_spacing_s: int = 5,
                    single_equivalent: bool = False, single_lots: float | None = None) -> dict:
    limits.check(assumptions, positions)
    if direction not in ("BUY", "SELL") or entry_spacing_s < 1:
        raise ValueError("Invalid direction or entry spacing.")
    first = ticks.iloc[index]
    if first.ask - first.bid > limits.max_spread_price:
        return {"outcome": "rejected_spread"}
    sign = 1 if direction == "BUY" else -1
    def entry_price(row) -> float:
        return float(row.ask if direction == "BUY" else row.bid) + sign * assumptions.slippage_price
    if single_lots is not None and (single_lots <= 0 or single_lots > limits.max_lots + 1e-9):
        raise ValueError("Comparison lots outside exposure limit.")
    entries = [(entry_price(first), (single_lots or assumptions.lots * positions) if single_equivalent else assumptions.lots)]
    entry_spreads = [(float(first.ask) - float(first.bid), entries[0][1])]
    next_entry_ms = int(first.time_msc) + entry_spacing_s * 1000
    deadline = int(first.time_msc) + limits.max_duration_s * 1000
    previous_ms = int(first.time_msc)
    previous_mid = (first.bid + first.ask) / 2
    last_exit = None
    last_pnl = None
    for row in ticks.iloc[index + 1:].itertuples():
        if row.time_msc > deadline:
            break
        if (row.time_msc - previous_ms) / 1000 > assumptions.max_gap_seconds:
            return {"outcome": "insufficient"}
        previous_ms = int(row.time_msc)
        if row.ask < row.bid or row.bid <= 0:
            return {"outcome": "insufficient"}
        spread = row.ask - row.bid
        if not single_equivalent and len(entries) < positions and row.time_msc >= next_entry_ms:
            # Add only while price has moved favorably and the spread is still acceptable.
            mid_now = (row.bid + row.ask) / 2
            mid_first = (first.bid + first.ask) / 2
            if spread <= limits.max_spread_price and sign * (mid_now - mid_first) > 0 and sign * (mid_now - previous_mid) > 0:
                entries.append((entry_price(row), assumptions.lots))
                entry_spreads.append((spread, assumptions.lots))
            else:
                next_entry_ms = deadline + 1
            next_entry_ms += entry_spacing_s * 1000
        exit_price = float(row.bid if direction == "BUY" else row.ask) - sign * assumptions.slippage_price
        pnl = _basket_pnl(direction, entries, exit_price, assumptions)
        previous_mid = (row.bid + row.ask) / 2
        last_exit, last_pnl = exit_price, pnl
        if pnl >= assumptions.profit_usd or pnl <= -assumptions.loss_usd:
            break
    if last_pnl is None:
        return {"outcome": "insufficient"}
    outcome = "profit" if last_pnl >= assumptions.profit_usd else "loss" if last_pnl <= -assumptions.loss_usd else "timeout"
    lots = sum(lot for _, lot in entries)
    return {"outcome": outcome, "pnl_usd": float(last_pnl),
            "holding_s": (previous_ms - int(first.time_msc)) / 1000,
            "positions": len(entries), "lots": lots,
            "weighted_entry": sum(price * lot for price, lot in entries) / lots,
            "exit_price": last_exit,
            "commission_usd": assumptions.commission_per_lot_round_turn * lots,
            "spread_entry_usd": sum(spread * assumptions.contract_size * lot for spread, lot in entry_spreads),
            "slippage_usd": 2 * assumptions.slippage_price * assumptions.contract_size * lots}


def performance(rows: pd.DataFrame, initial_balance: float = 5000.0) -> dict:
    if rows.empty:
        return {"trades": 0, "net_pnl_usd": 0.0, "warning": "No eligible opportunities."}
    pnl = rows.pnl_usd.astype(float)
    equity = initial_balance + pnl.cumsum()
    peak = np.maximum.accumulate(np.r_[initial_balance, equity.to_numpy()])[1:]
    losses = pnl[pnl < 0]
    wins = pnl[pnl > 0]
    streak = max_streak = 0
    for value in pnl:
        streak = streak + 1 if value < 0 else 0
        max_streak = max(max_streak, streak)
    return {"baskets": int(len(rows)), "trades": int(rows.positions.sum()),
            "win_rate": float((pnl > 0).mean()),
            "average_profit_usd": float(wins.mean()) if len(wins) else 0.0,
            "average_loss_usd": float(losses.mean()) if len(losses) else 0.0,
            "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) else None,
            "net_expectancy_usd": float(pnl.mean()), "net_pnl_usd": float(pnl.sum()),
            "max_drawdown_usd": float(np.max(peak - equity.to_numpy())),
            "consecutive_losses": max_streak,
            "average_holding_s": float(rows.holding_s.mean()),
            "commission_usd": float(rows.commission_usd.sum()),
            "estimated_transaction_costs_usd": float((rows.commission_usd + rows.spread_entry_usd + rows.slippage_usd).sum())}


def research_patterns(dataset: pd.DataFrame) -> pd.DataFrame:
    data = dataset[dataset.outcome.isin(["profit", "loss", "timeout"])].copy()
    if data.empty:
        raise RuntimeError("No labeled outcomes available.")
    cutoff = int(data.time_msc.quantile(0.8))
    data = data[data.time_msc >= cutoff].copy()
    direction = np.where(data.direction.eq("BUY"), 1, -1)
    masks = {
        "Momentum Burst": (direction * data.return_20 > 0) & (direction * data.acceleration > 0),
        "Breakout Burst": (direction * data.breakout > 0) & (data.activity_20 > data.activity_60),
        "Pullback-Reversal Burst": (direction * data.pullback < 0) & (direction * data.return_5 > 0),
        "Volatility Expansion": (data.vol_expansion > 1.3) & (direction * data.return_5 > 0),
    }
    results = []
    data["volatility_regime"] = np.where(data.vol_expansion > 1.3, "expanding", "ordinary")
    for family, mask in masks.items():
        for (session, regime), group in data[mask].groupby(["session", "volatility_regime"]):
            count = len(group)
            results.append({"family": family, "session": int(session), "volatility_regime": regime,
                            "period": "final 20% of supplied history", "samples": count,
                            "win_rate": float(group.outcome.eq("profit").mean()),
                            "mean_pnl_usd": float(group.pnl_usd.mean()),
                            "conclusion": "insufficient evidence" if count < 100 else "research only"})
    return pd.DataFrame(results)


def backtest(ticks: pd.DataFrame, signals: pd.DataFrame, assumptions: TradeAssumptions,
             limits: RiskLimits, positions: int = 3, starting_equity: float | None = None) -> tuple[pd.DataFrame, dict]:
    limits.check(assumptions, positions)
    starting_equity = limits.balance_usd if starting_equity is None else starting_equity
    results = []
    next_allowed = -1
    day_pnl = {}
    basket_pnl = 0.0
    times = ticks.time_msc.to_numpy()
    for signal in signals.sort_values("time_msc").itertuples():
        if signal.time_msc < next_allowed or signal.direction not in ("BUY", "SELL"):
            continue
        day = pd.Timestamp(signal.time_msc, unit="ms", tz="UTC").date().isoformat()
        equity = starting_equity + basket_pnl
        if equity <= limits.balance_usd * (1 - limits.max_total_drawdown_fraction):
            break
        if assumptions.loss_usd > equity * limits.basket_risk_fraction:
            break
        if day_pnl.get(day, 0) <= -starting_equity * limits.max_daily_drawdown_fraction:
            continue
        index = int(np.searchsorted(times, signal.time_msc, side="left"))
        if index >= len(ticks) or ticks.time_msc.iloc[index] != signal.time_msc:
            continue
        burst = simulate_basket(ticks, index, signal.direction, assumptions, limits, positions)
        if burst.get("outcome") in ("insufficient", "rejected_spread"):
            continue
        single_result = simulate_basket(ticks, index, signal.direction, assumptions, limits, positions,
                                        single_equivalent=True, single_lots=burst["lots"])
        for mode, result in (("burst", burst), ("single_equal_exposure", single_result)):
            if result.get("outcome") in ("insufficient", "rejected_spread"):
                continue
            results.append({"time_msc": signal.time_msc, "direction": signal.direction,
                            "mode": mode, "session": getattr(signal, "session", None),
                            "volatility_regime": "expanding" if getattr(signal, "vol_expansion", 0) > 1.3 else "ordinary",
                            **result})
            if mode == "burst":
                day_pnl[day] = day_pnl.get(day, 0) + result["pnl_usd"]
                basket_pnl += result["pnl_usd"]
        next_allowed = signal.time_msc + (limits.max_duration_s + limits.cooldown_s) * 1000
    frame = pd.DataFrame(results)
    summary = {mode: performance(group, limits.balance_usd) for mode, group in frame.groupby("mode")} if not frame.empty else {}
    regimes = {f"{mode}_session{session}_{regime}": performance(group, limits.balance_usd)
               for (mode, session, regime), group in frame.groupby(["mode", "session", "volatility_regime"])} if not frame.empty else {}
    return frame, {"assumptions": asdict(assumptions), "limits": asdict(limits), "summary": summary,
                   "regimes": regimes,
                   "comparison": "The single entry uses the same lots the burst actually filled, measured after simulation.",
                   "warning": "Historical quotes do not prove executable liquidity or future profitability."}
