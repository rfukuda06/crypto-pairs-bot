# src/pairsbot/backtest/engine.py
from __future__ import annotations

import pandas as pd

from pairsbot.core.types import (BacktestResult, Order, PairSelection,
                                  SpreadSide, StrategyContext)
from pairsbot.data.historical import align_closes


class Backtester:
    def __init__(self, strategy, risk, starting_equity: float, fee_pct: float,
                 slippage_pct: float, strategy_cfg: dict):
        self.strategy = strategy
        self.risk = risk
        self.starting_equity = starting_equity
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.cfg = strategy_cfg

    def run(self, data: dict[str, pd.DataFrame], sel: PairSelection) -> BacktestResult:
        # Local import avoids a hard module cycle and keeps broker swappable.
        from pairsbot.execution.broker import PaperBroker

        a, b = sel.a, sel.b
        closes = align_closes(data)[[a, b]]
        opens = pd.DataFrame({a: data[a]["open"], b: data[b]["open"]}).loc[closes.index]

        broker = PaperBroker(self.starting_equity, self.fee_pct, self.slippage_pct)
        self.risk.update_peak(self.starting_equity)

        in_position = False
        side: SpreadSide | None = None
        bars_in = 0
        pending: list[Order] = []
        equity_points: list[tuple[pd.Timestamp, float]] = []
        trades: list[dict] = []

        index = closes.index
        for i, ts in enumerate(index):
            # 1. fill orders decided on the previous bar, at THIS bar's open
            for o in pending:
                fill = broker.submit(o, float(opens.loc[ts, o.symbol]))
                trades.append({"ts": ts.isoformat(), "symbol": o.symbol,
                               "notional": o.notional, "price": fill.price,
                               "qty": fill.qty, "fee": fill.fee, "reason": o.reason})
            pending = []

            # 2. mark to this bar's close
            broker.mark({a: float(closes.loc[ts, a]), b: float(closes.loc[ts, b])})
            equity = broker.equity()
            equity_points.append((ts, equity))
            self.risk.update_peak(equity)

            if in_position:
                bars_in += 1

            # 3. build context from data up to and including this bar
            ctx = StrategyContext(
                a=a, b=b, beta=sel.beta, closes=closes.iloc[: i + 1],
                in_position=in_position, position_side=side, bars_in_position=bars_in,
                **self.cfg)

            # 4. get signals, translate to orders for NEXT bar's open
            signals = self.strategy.on_bar(ctx)
            for sig in signals:
                if sig.kind == "exit" and in_position:
                    prices = {a: float(closes.loc[ts, a]), b: float(closes.loc[ts, b])}
                    pending = self.risk.flatten_orders(broker.positions(), prices)
                    in_position, side, bars_in = False, None, 0
                elif sig.kind == "enter" and not in_position:
                    if self.risk.allow_entry(equity):
                        pending = self.risk.entry_orders(sig, equity, a, b)
                        in_position, side, bars_in = True, sig.spread_side, 0

        eq = pd.Series({ts: v for ts, v in equity_points})
        return BacktestResult(equity=eq, trades=trades, selection=sel)
