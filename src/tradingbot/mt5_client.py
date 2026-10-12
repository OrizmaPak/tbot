from __future__ import annotations

import csv
import os
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Iterable

from .config import MT5Config
from .storage import TickRecord


class MT5ConnectionError(RuntimeError):
    pass


class MT5Client(AbstractContextManager["MT5Client"]):
    def __init__(self, config: MT5Config) -> None:
        self.config = config
        self.mt5 = self._import_mt5()

    def __enter__(self) -> "MT5Client":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.mt5.shutdown()

    def connect(self) -> None:
        kwargs: dict[str, object] = {
            "timeout": self.config.timeout_ms,
            "portable": self.config.portable,
        }

        password = os.environ.get(self.config.password_env)
        if self.config.login is not None:
            kwargs["login"] = int(self.config.login)
        if password:
            kwargs["password"] = password
        if self.config.server:
            kwargs["server"] = self.config.server

        if self.config.terminal_path:
            initialized = self.mt5.initialize(self.config.terminal_path, **kwargs)
        else:
            initialized = self.mt5.initialize(**kwargs)

        if not initialized:
            raise MT5ConnectionError(self._format_last_error("MT5 initialize failed"))

    def terminal_summary(self) -> dict[str, object]:
        terminal = self.mt5.terminal_info()
        account = self.mt5.account_info()
        if terminal is None:
            raise MT5ConnectionError(self._format_last_error("Could not read terminal info"))

        return {
            "terminal_name": getattr(terminal, "name", None),
            "terminal_company": getattr(terminal, "company", None),
            "connected": getattr(terminal, "connected", None),
            "trade_allowed": getattr(terminal, "trade_allowed", None),
            "account_login": getattr(account, "login", None) if account else None,
            "account_server": getattr(account, "server", None) if account else None,
            "account_currency": getattr(account, "currency", None) if account else None,
            "account_balance": getattr(account, "balance", None) if account else None,
        }

    def discover_symbols(self, query: str) -> list[dict[str, object]]:
        symbols = self.mt5.symbols_get(f"*{query}*")
        if symbols is None:
            raise MT5ConnectionError(self._format_last_error("Symbol search failed"))

        results: list[dict[str, object]] = []
        for symbol in symbols:
            results.append(
                {
                    "name": symbol.name,
                    "description": symbol.description,
                    "path": symbol.path,
                    "currency_profit": symbol.currency_profit,
                    "trade_contract_size": symbol.trade_contract_size,
                    "volume_min": symbol.volume_min,
                    "volume_step": symbol.volume_step,
                    "digits": symbol.digits,
                    "spread": symbol.spread,
                    "trade_mode": symbol.trade_mode,
                }
            )
        return results

    def symbol_summary(self, symbol: str) -> dict[str, object]:
        self.ensure_symbol(symbol)
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise MT5ConnectionError(self._format_last_error(f"Could not read symbol info for {symbol!r}"))

        fields = [
            "name",
            "description",
            "path",
            "currency_base",
            "currency_profit",
            "digits",
            "point",
            "trade_tick_size",
            "trade_tick_value",
            "trade_contract_size",
            "volume_min",
            "volume_max",
            "volume_step",
            "spread",
            "spread_float",
            "trade_stops_level",
            "trade_freeze_level",
            "trade_mode",
            "filling_mode",
            "order_mode",
        ]
        return {field: getattr(info, field, None) for field in fields}

    def ensure_symbol(self, symbol: str) -> None:
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise MT5ConnectionError(
                f"Symbol {symbol!r} was not found. Run discover to find your broker's gold symbol."
            )
        if not info.visible and not self.mt5.symbol_select(symbol, True):
            raise MT5ConnectionError(self._format_last_error(f"Could not select symbol {symbol!r}"))

    def latest_tick(self, symbol: str) -> TickRecord:
        self.ensure_symbol(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            raise MT5ConnectionError(self._format_last_error(f"No tick available for {symbol!r}"))

        return TickRecord(
            symbol=symbol,
            time_msc=int(tick.time_msc),
            bid=float(tick.bid),
            ask=float(tick.ask),
            last=float(tick.last),
            volume=int(tick.volume),
            flags=int(tick.flags),
        )

    def export_ticks(self, symbol: str, start: datetime, end: datetime, output_path: Path) -> int:
        self.ensure_symbol(symbol)
        utc_start = self._as_utc(start)
        utc_end = self._as_utc(end)
        ticks = self.mt5.copy_ticks_range(symbol, utc_start, utc_end, self.mt5.COPY_TICKS_ALL)
        if ticks is None:
            raise MT5ConnectionError(self._format_last_error("Historical tick export failed"))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["time", "time_msc", "bid", "ask", "last", "volume", "flags", "volume_real"]
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for tick in ticks:
                writer.writerow({field: tick[field] for field in fieldnames})

        return len(ticks)

    @staticmethod
    def _import_mt5() -> ModuleType:
        try:
            import MetaTrader5 as mt5
        except ImportError as error:
            raise MT5ConnectionError(
                "MetaTrader5 package is not installed. Run: pip install -r requirements.txt"
            ) from error
        return mt5

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _format_last_error(self, message: str) -> str:
        code, detail = self.mt5.last_error()
        return f"{message}: [{code}] {detail}"
