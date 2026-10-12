# AI Burst Trading System - Phase 1

Phase 1 connects a local Python app to MetaTrader 5, reads XAUUSD/gold market data, and records ticks without placing trades.

## What this phase does

- Checks that Python can connect to your local MetaTrader 5 terminal.
- Finds broker-specific gold symbols such as `XAUUSD`, `XAUUSDm`, or `GOLD`.
- Records live bid/ask ticks into SQLite.
- Exports historical ticks from MT5 into CSV.
- Does not place orders.

## Setup

1. Install Python 3.11 or newer for Windows.
2. Open MetaTrader 5 and log in to your broker account.
3. From this folder, create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

4. Copy the config:

```powershell
Copy-Item config.example.json config.json
```

5. Edit `config.json` if your broker uses a different gold symbol.

## Commands

Launch the interface:

```powershell
.\.venv\Scripts\streamlit.exe run src\tradingbot\ui.py
```

Or use the launcher:

```powershell
.\launch_interface.ps1
```

Check the MT5 connection:

```powershell
python -m tradingbot.cli check --config config.json
```

Search for available gold-like symbols:

```powershell
python -m tradingbot.cli discover --config config.json --query XAU
```

Inspect the configured symbol's broker contract settings:

```powershell
python -m tradingbot.cli symbol-info --config config.json
```

Collect live ticks for five minutes:

```powershell
python -m tradingbot.cli collect-live --config config.json --duration-seconds 300
```

Export historical ticks:

```powershell
python -m tradingbot.cli export-ticks --config config.json --from "2026-10-01T00:00:00" --to "2026-10-02T00:00:00"
```

## Output

Live ticks are stored in:

```text
data/market_ticks.sqlite
```

Historical exports are stored in:

```text
data/history/
```

## Safety

This phase is market-data only. There is no order-placement code in this project yet.
