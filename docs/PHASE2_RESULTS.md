# Phase 2 Research Results

Run date: 2026-10-12. Broker: FP Markets demo, symbol: XAUUSD. No orders were sent.

## Data and method

- Requested September 14 through October 11, 2026 (UTC), in 672 one-hour MT5 windows.
- Received 2,936,582 real broker ticks, stored as compressed Parquet. Some windows are empty around market closures; five windows reported a gap over 30 seconds. No invalid bid/ask quotes were reported in the four-week collection.
- Built 20 nonoverlapping weekday datasets with 81,942 labeled rows. Each decision was sampled about every 120 seconds and labeled for BUY and SELL at 60, 120, and 180-second horizons.
- P&L assumed a 100-unit XAUUSD contract, 0.10 lot per entry, $7 round-trip commission per lot, $0.02 price-unit slippage each side, and $10 basket target/loss boundary. These cost assumptions need account-level verification.
- Training used chronological train, calibration, and test periods, with a purge of at least the prediction horizon. The final test spans five days. Two earlier walk-forward windows were also evaluated.

## Findings

- All six LightGBM direction/horizon models were trained and saved. Several beat a constant-rate Brier baseline, but mean realized outcomes on the unseen period were roughly **-$3.3 to -$3.5** per proposed trade.
- Estimated-positive model selections were too few and did not show positive held-out returns. The system's combined recommendation is therefore **NO TRADE** for every horizon.
- Momentum, breakout, pullback reversal, and volatility expansion were compared on the final 20% of supplied history. Most session/regime groups had negative mean P&L. One pullback subgroup was about +$0.07 per trade from 91 observations; that is insufficient evidence of an edge.
- With the default 10% cumulative drawdown stop on a $5,000 reference balance, the preliminary momentum burst simulation stopped after 79 baskets / 109 entries: **-$507.50** for bursts versus **-$411.30** for one-entry baskets with matched filled lots. The stop overshot $500 because historical prices moved past the loss boundary between quotes.
- Independent session runs also lost money. Europe: burst **-$505.60**, matched single **-$420.00**. US: burst **-$506.30**, matched single **-$423.20**. Late UTC: burst **-$509.90**, matched single **-$471.90**. Each starts with a fresh reference balance; do not add these results together.

## Interpretation

This engineering phase establishes a working local data-to-model-to-simulation pipeline. It does **not** establish a profitable strategy. Historical bid/ask quotes cannot prove fills, latency, available liquidity, or stop execution. The same four-week sample was used for exploratory feature and strategy development, so independent later data and forward paper testing are required before reconsidering live execution. The code contains no live order path.

Detailed machine-readable metrics and trades are available locally under `data/models/` and `data/reports/`; those directories are intentionally excluded from Git.
