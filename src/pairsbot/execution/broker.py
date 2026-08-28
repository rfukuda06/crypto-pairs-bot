# src/pairsbot/execution/broker.py
from __future__ import annotations

from typing import Protocol

from pairsbot.core.types import Fill, Order, Position


class Broker(Protocol):
    def submit(self, order: Order, price: float) -> Fill: ...
    def mark(self, prices: dict[str, float]) -> None: ...
    def positions(self) -> dict[str, Position]: ...
    def equity(self) -> float: ...


class PaperBroker:
    """Simulated fills with fees + slippage. Used by BOTH backtest and live —
    the only difference between modes is who feeds it prices."""

    def __init__(self, starting_cash: float, fee_pct: float, slippage_pct: float):
        self.cash = starting_cash
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self._pos: dict[str, Position] = {}
        self._marks: dict[str, float] = {}

    def mark(self, prices: dict[str, float]) -> None:
        self._marks.update(prices)

    def submit(self, order: Order, price: float) -> Fill:
        buy = order.notional > 0
        fill_price = price * (1 + self.slippage_pct) if buy else price * (1 - self.slippage_pct)
        qty = order.notional / fill_price          # signed
        fee = abs(order.notional) * self.fee_pct
        self.cash -= order.notional + fee          # buy: cash down; short: cash up
        pos = self._pos.setdefault(order.symbol, Position(order.symbol))
        new_qty = pos.qty + qty
        if pos.qty == 0 or (pos.qty > 0) == (qty > 0):
            # opening or adding: update weighted avg price
            total = abs(pos.qty) + abs(qty)
            pos.avg_price = (abs(pos.qty) * pos.avg_price + abs(qty) * fill_price) / total if total else 0.0
        pos.qty = new_qty
        if abs(pos.qty) < 1e-12:
            pos.qty = 0.0
            pos.avg_price = 0.0
        self._marks[order.symbol] = price
        return Fill(order.symbol, order.notional, fill_price, qty, fee)

    def positions(self) -> dict[str, Position]:
        return {s: p for s, p in self._pos.items() if p.qty != 0}

    def equity(self) -> float:
        val = self.cash
        for sym, pos in self._pos.items():
            if pos.qty != 0:
                val += pos.qty * self._marks.get(sym, pos.avg_price)
        return val


class LiveBroker:
    """Real-execution seam. Intentionally unimplemented in v1 — proves the
    architecture supports live trading without enabling it."""

    def submit(self, order: Order, price: float) -> Fill:
        raise NotImplementedError("real execution is disabled in v1 (paper only)")

    def mark(self, prices: dict[str, float]) -> None:
        raise NotImplementedError

    def positions(self) -> dict[str, Position]:
        raise NotImplementedError

    def equity(self) -> float:
        raise NotImplementedError
