# src/pairsbot/risk/manager.py
from __future__ import annotations

from pairsbot.core.types import Order, Position, Signal, SpreadSide


class RiskManager:
    def __init__(self, gross_exposure_pct: float, max_drawdown_pct: float):
        self.gross_exposure_pct = gross_exposure_pct
        self.max_drawdown_pct = max_drawdown_pct
        self._peak = 0.0

    def update_peak(self, equity: float) -> None:
        self._peak = max(self._peak, equity)

    def allow_entry(self, equity: float) -> bool:
        if self._peak <= 0:
            return True
        drawdown = 1.0 - equity / self._peak
        return drawdown < self.max_drawdown_pct

    def entry_orders(self, signal: Signal, equity: float, a: str, b: str) -> list[Order]:
        leg = self.gross_exposure_pct * equity / 2.0
        # long spread = long A, short B
        sign_a = 1.0 if signal.spread_side == SpreadSide.LONG else -1.0
        return [
            Order(a, sign_a * leg, reason=signal.reason),
            Order(b, -sign_a * leg, reason=signal.reason),
        ]

    def flatten_orders(self, positions: dict[str, Position],
                       prices: dict[str, float]) -> list[Order]:
        orders = []
        for sym, pos in positions.items():
            if pos.qty != 0:
                orders.append(Order(sym, -pos.qty * prices[sym], reason="exit"))
        return orders
