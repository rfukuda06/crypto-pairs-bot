# src/pairsbot/strategy/base.py
from __future__ import annotations

from typing import Protocol

from pairsbot.core.types import Signal, StrategyContext


class Strategy(Protocol):
    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        ...
