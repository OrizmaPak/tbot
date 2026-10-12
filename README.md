# AI Burst Trading System

Local XAUUSD data collection, LightGBM research, and historical burst simulation for MetaTrader 5. This application does **not** place live orders.

The latest real-data findings are in [Phase 2 Research Results](docs/PHASE2_RESULTS.md). No profitable edge was established, and the model currently returns `NO TRADE`.

## Install on Windows 11

Open PowerShell in this repository. Install Python 3.12 and MetaTrader 5, sign in to the broker terminal, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e . pytest
Copy-Item config.example.json config.json
```

Edit `config.json` for the broker symbol and terminal path. Keep credentials in MT5 or the environment variable named by `mt5.password_env`. `config.json` and `data/` are ignored by Git. An existing virtual environment only needs the `pip install -e . pytest` line.

```powershell
.\.venv\Scripts\python.exe -m tradingbot.cli check --config config.json
.\.venv\Scripts\python.exe -m tradingbot.cli discover --config config.json --query XAU
.\.venv\Scripts\python.exe -m tradingbot.cli symbol-info --config config.json
```

## Collect real broker ticks

All command timestamps are UTC. MT5 must be open and connected. The resumable collector writes Zstandard-compressed Parquet windows under `data/raw/XAUUSD/`, JSON quality reports, and a broker contract snapshot. It preserves bid, ask, millisecond timestamp, flags, and volume. Repeating a command skips completed windows. Empty windows are recorded; prices are never fabricated.

```powershell
.\.venv\Scripts\python.exe -m tradingbot.cli collect-history --config config.json --from 2026-09-14T00:00:00Z --to 2026-10-12T00:00:00Z --chunk-minutes 60
```

That example requests four weeks. Broker retention varies, and nontrading hours can be empty. The collector checks RAM and disk before each window and caps one MT5 response at 250,000 ticks. If a window exceeds the cap, rerun with `--chunk-minutes 10` or lower. Collection requires at least 3 GB free disk and 700 MB free RAM. On this 16 GB laptop, use smaller date ranges when other applications are open.

Phase 1 commands remain available:

```powershell
.\.venv\Scripts\python.exe -m tradingbot.cli collect-live --config config.json --duration-seconds 300
.\.venv\Scripts\python.exe -m tradingbot.cli export-ticks --config config.json --from 2026-10-12T13:30:00Z --to 2026-10-12T14:30:00Z
```

## Build research datasets

Generate a bounded interval that has already been collected. The output Parquet contains past-only features and BUY/SELL labels for 60, 120, and 180 seconds. Its adjacent JSON file records feature definitions and execution assumptions.

```powershell
.\.venv\Scripts\python.exe -m tradingbot.cli make-dataset --config config.json --from 2026-10-12T13:30:00Z --to 2026-10-12T16:30:00Z --sample-seconds 10 --max-rows 400000 --lots 0.1 --commission-per-lot 7 --slippage-price 0.02 --profit-usd 10 --loss-usd 10 --max-spread-price 0.5
```

The saved broker contract size converts price movement to USD P&L. Commission and slippage above are **estimates** to verify against your account. Labels are `profit`, `loss`, `timeout`, or `insufficient`; missing or invalid quotes are not silently repaired. Rolling windows are in **ticks**, not seconds. A decision is sampled every 10 seconds by default. Limits of 400,000 raw ticks and 20,000 decision samples bound memory use. For longer history, create several smaller datasets using identical assumptions.

## Train and evaluate

Use the dataset path printed by `make-dataset`. Multiple paths may follow `--dataset`; the combined default limit is 50,000 rows. Models, calibration maps, and metrics go under `data/models/`.

```powershell
.\.venv\Scripts\python.exe -m tradingbot.cli train --config config.json --dataset data\datasets\XAUUSD_20261012T133000Z_20261012T163000Z.parquet --threads 2
.\.venv\Scripts\python.exe -m tradingbot.cli evaluate --config config.json --dataset data\datasets\XAUUSD_20261012T133000Z_20261012T163000Z.parquet
.\.venv\Scripts\python.exe -m tradingbot.cli research --config config.json --dataset data\datasets\XAUUSD_20261012T133000Z_20261012T163000Z.parquet
```

Training uses chronological train, calibration, and unseen test intervals with a purge of at least the prediction horizon. Evaluation also runs two forward windows. A model only offers a BUY/SELL research recommendation when test calibration beats a constant baseline with at least 1,000 examples across five days, and estimated net return is positive; otherwise it returns `NO TRADE`. This is still not live authorization. Pattern research compares momentum, breakout, pullback reversal, and volatility expansion by session and volatility regime. They are hypotheses, not proven strategies.

To retrain after more collection, run `make-dataset` for nonoverlapping intervals with the **same** target and cost settings, then pass all Parquet paths to `train --dataset`. New model versions do not replace old ones.

For the four-week research set, create one dataset per weekday with `--sample-seconds 120`, then use the saved files together:

```powershell
$start = [datetime]'2026-09-14'
for ($i = 0; $i -lt 28; $i++) {
    $day = $start.AddDays($i)
    if ($day.DayOfWeek -in @('Saturday', 'Sunday')) { continue }
    $from = $day.ToString('yyyy-MM-ddT00:00:00Z')
    $to = $day.AddDays(1).ToString('yyyy-MM-ddT00:00:00Z')
    .\.venv\Scripts\python.exe -m tradingbot.cli make-dataset --config config.json --from $from --to $to --sample-seconds 120
}
$datasets = @(Get-ChildItem data\datasets\*T000000Z_*T000000Z.parquet | Sort-Object Name | ForEach-Object FullName)
.\.venv\Scripts\python.exe -m tradingbot.cli train --config config.json --dataset $datasets --max-rows 100000 --threads 2
.\.venv\Scripts\python.exe -m tradingbot.cli evaluate --config config.json --dataset $datasets --max-rows 100000
.\.venv\Scripts\python.exe -m tradingbot.cli research --config config.json --dataset $datasets --max-rows 100000
```

## Backtest and dashboard

The preliminary backtest uses a momentum hypothesis based only on decision-time features. It pairs a staggered burst with one entry at the **same lots the burst actually filled**. That single entry is an ex post research benchmark. Both use bid/ask, commission, slippage, basket profit/loss boundaries, exposure limits, cooldown, and daily drawdown limits.

```powershell
.\.venv\Scripts\python.exe -m tradingbot.cli backtest --config config.json --dataset data\datasets\XAUUSD_20261012T133000Z_20261012T163000Z.parquet --from 2026-10-12T13:30:00Z --to 2026-10-12T16:30:00Z --balance-usd 5000 --risk-fraction 0.01 --positions 3 --max-lots 0.3
.\launch_interface.ps1
```

To simulate the full collected month using day-sized raw-data batches:

```powershell
.\.venv\Scripts\python.exe -m tradingbot.cli backtest --config config.json --dataset $datasets --from 2026-09-14T00:00:00Z --to 2026-10-12T00:00:00Z --max-dataset-rows 100000
```

The default `backtest` report enforces the 10% cumulative stop. Use `--session 1` (or 0, 2, 3) and `--report-name backtest_session_1` to compare a UTC session independently; each run starts with the same reference balance. These are separate research scenarios, not results to add together.

Open [http://localhost:8501](http://localhost:8501). The dashboard shows collection health, model metrics, pattern research, backtest equity, and system resources. Run longer research commands in PowerShell so they do not stall the browser. Tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Limits and risk

The simulation defaults to a 1% basket risk budget on a $5,000 reference account ($50), with a 3% daily and 10% cumulative drawdown stop. The basket budget tightens as simulated equity falls. These are ceilings, not proof that 0.10 lots or three entries are safe live. A historical quote does not guarantee fill price, liquidity, or stop execution; jumps can exceed a loss boundary. Simulation omits latency, partial fills, margin calls, swap, and broker rejections. There is no live order code, and a model cannot change risk settings. Longer independent unseen periods and forward paper testing are needed before any profitability claim.
