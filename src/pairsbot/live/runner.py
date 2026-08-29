# src/pairsbot/live/runner.py
from __future__ import annotations

import pandas as pd

from pairsbot.core.types import Position, SpreadSide, StrategyContext
from pairsbot.monitor import build_live_snapshot, format_live_line


class LiveRunner:
    def __init__(self, feed, broker, strategy, risk, store, selection, symbols,
                 strategy_cfg: dict, sleep, poll_seconds: int = 3600,
                 starting_equity: float | None = None, initial_closes=None,
                 run_id: int | None = None):
        self.feed = feed
        self.broker = broker
        self.strategy = strategy
        self.risk = risk
        self.store = store
        self.sel = selection
        self.symbols = symbols
        self.cfg = strategy_cfg
        self.sleep = sleep
        self.poll_seconds = poll_seconds
        self._starting_equity = broker.equity() if starting_equity is None else starting_equity
        self.run_id = run_id if run_id is not None else store.start_run(
            mode="live", pair=f"{selection.a}/{selection.b}")
        if initial_closes is not None and len(initial_closes):
            self._closes = initial_closes[[selection.a, selection.b]].astype(float).copy()
        else:
            self._closes = pd.DataFrame(columns=[selection.a, selection.b], dtype=float)
        self._pending: list = []
        self.in_position = False
        self.side = None
        self.bars_in = 0
        if run_id is not None:
            self._restore_state()

    @staticmethod
    def restore_broker(broker, store, run_id: int) -> None:
        """Rehydrate broker positions AND cash from the DB so a restart resumes
        cleanly with a continuous equity curve. Cash is reconstructed so that
        equity marked at cost basis equals the last recorded equity; the first
        live mark then moves it by the genuine unrealised PnL since entry."""
        positions = store.load_positions(run_id)
        for sym, (qty, avg) in positions.items():
            broker._pos[sym] = Position(sym, qty=qty, avg_price=avg)
        eq = store.load_equity(run_id)
        if eq:
            last_equity = eq[-1][1]
            pos_value = sum(qty * avg for qty, avg in positions.values())
            broker.cash = last_equity - pos_value

    def _restore_state(self) -> None:
        """Rehydrate in_position/side/bars_in from the restored broker + DB so a
        restart mid-position keeps managing that position instead of thinking it
        is flat. bars_in is approximated as equity bars recorded after the last
        trade (the open entry)."""
        a = self.sel.a
        held = self.broker.positions()
        if a not in held or held[a].qty == 0:
            return
        self.in_position = True
        self.side = SpreadSide.LONG if held[a].qty > 0 else SpreadSide.SHORT
        trades = self.store.load_trades(self.run_id)
        if trades:
            entry_ts = trades[-1]["ts"]              # while in_position, last trade is the entry
            eq = self.store.load_equity(self.run_id)
            self.bars_in = sum(1 for ts, _ in eq if ts > entry_ts)

    def _step(self) -> None:
        a, b = self.sel.a, self.sel.b
        bar = self.feed.latest_closed_bar(self.symbols)
        ts = bar[a]["ts"]
        if ts in self._closes.index:
            return                       # no new bar closed since last poll

        # 1. fill orders decided last bar at this bar's open
        for o in self._pending:
            fill = self.broker.submit(o, float(bar[o.symbol]["open"]))
            self.store.record_trade(self.run_id, ts.isoformat(), o.symbol,
                                    o.notional, fill.price, fill.qty, fill.fee, o.reason)
            self.store.upsert_position(self.run_id, o.symbol,
                                       self.broker._pos[o.symbol].qty,
                                       self.broker._pos[o.symbol].avg_price)
        self._pending = []

        # 2. mark + record equity
        self.broker.mark({a: float(bar[a]["close"]), b: float(bar[b]["close"])})
        equity = self.broker.equity()
        self.store.record_equity(self.run_id, ts.isoformat(), equity)
        self.risk.update_peak(equity)

        # 2b. read-only risk/PnL monitor line for this bar
        snap = build_live_snapshot(
            positions=self.broker.positions(),
            marks={a: float(bar[a]["close"]), b: float(bar[b]["close"])},
            equity=equity, peak=self.risk._peak,
            starting_equity=self._starting_equity,
            max_dd_pct=self.risk.max_drawdown_pct, ts=ts.isoformat())
        print(format_live_line(snap))

        # 3. append close and build context
        self._closes.loc[ts] = [float(bar[a]["close"]), float(bar[b]["close"])]
        if self.in_position:
            self.bars_in += 1
        ctx = StrategyContext(a=a, b=b, beta=self.sel.beta, closes=self._closes,
                              in_position=self.in_position, position_side=self.side,
                              bars_in_position=self.bars_in, **self.cfg)

        # 4. signals -> pending orders for next bar
        for sig in self.strategy.on_bar(ctx):
            if sig.kind == "exit" and self.in_position:
                prices = {a: float(bar[a]["close"]), b: float(bar[b]["close"])}
                self._pending = self.risk.flatten_orders(self.broker.positions(), prices)
                self.in_position, self.side, self.bars_in = False, None, 0
            elif sig.kind == "enter" and not self.in_position:
                if self.risk.allow_entry(equity):
                    self._pending = self.risk.entry_orders(sig, equity, a, b)
                    self.in_position, self.side, self.bars_in = True, sig.spread_side, 0

    def run(self, max_iterations: int | None = None) -> None:
        n = 0
        while max_iterations is None or n < max_iterations:
            self._step()
            n += 1
            if max_iterations is None or n < max_iterations:
                self.sleep(self.poll_seconds)
