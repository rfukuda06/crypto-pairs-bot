# src/pairsbot/strategy/pairs.py
from __future__ import annotations

import numpy as np

from pairsbot.core.types import Signal, SpreadSide, StrategyContext


def current_zscore(ctx: StrategyContext) -> float:
    """Rolling z-score of the log spread at the LAST (current) bar only.
    Uses the trailing z_window bars up to and including the current bar."""
    a = np.log(ctx.closes[ctx.a].to_numpy())
    b = np.log(ctx.closes[ctx.b].to_numpy())
    spread = a - ctx.beta * b
    window = spread[-ctx.z_window:]
    mu = window.mean()
    sigma = window.std()
    if sigma == 0:
        return 0.0
    return float((spread[-1] - mu) / sigma)


class PairsStrategy:
    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        if len(ctx.closes) < ctx.z_window:
            return []
        z = current_zscore(ctx)

        if ctx.in_position:
            if ctx.bars_in_position >= ctx.max_holding_bars:
                return [Signal("exit", reason="time-stop")]
            if abs(z) >= ctx.stop_z:
                return [Signal("exit", reason=f"stop-z {z:.2f}")]
            if abs(z) <= ctx.exit_z:
                return [Signal("exit", reason=f"mean-revert {z:.2f}")]
            return []

        # flat -> look for entry
        if z >= ctx.entry_z:
            return [Signal("enter", SpreadSide.SHORT, reason=f"z {z:.2f}")]
        if z <= -ctx.entry_z:
            return [Signal("enter", SpreadSide.LONG, reason=f"z {z:.2f}")]
        return []
