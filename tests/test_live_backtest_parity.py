# tests/test_live_backtest_parity.py
import numpy as np
import pandas as pd
import pytest

from pairsbot.backtest.engine import Backtester
from pairsbot.core.types import PairSelection
from pairsbot.execution.broker import PaperBroker
from pairsbot.risk.manager import RiskManager
from pairsbot.storage import Store
from pairsbot.strategy.pairs import PairsStrategy
from pairsbot.live.runner import LiveRunner


class ReplayFeed:
    def __init__(self, a, b, idx):
        self.a, self.b, self.idx, self.i = a, b, idx, 0

    def latest_closed_bar(self, symbols):
        ts = self.idx[self.i]
        bar = {"A": {"ts": ts, "open": self.a[self.i], "close": self.a[self.i]},
               "B": {"ts": ts, "open": self.b[self.i], "close": self.b[self.i]}}
        self.i += 1
        return bar


def _cointegrated(n=400):
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    common = np.cumsum(rng.normal(0, 0.5, n)) + 300
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.92 * spread[t - 1] + rng.normal(0, 0.6)
    a = (common + spread + 80).tolist()
    b = common.tolist()
    return a, b, idx


def test_live_matches_backtest_on_identical_prices(tmp_path):
    a, b, idx = _cointegrated()
    sel = PairSelection(a="A", b="B", beta=1.0, pvalue=0.001)
    cfg = dict(z_window=100, entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)

    # BACKTEST: open == close so fills and marks use the same price the feed serves.
    data = {s: pd.DataFrame({"open": series, "high": series, "low": series,
                             "close": series, "volume": 1.0}, index=idx)
            for s, series in (("A", a), ("B", b))}
    bt = Backtester(PairsStrategy(),
                    RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2),
                    10000, 0.001, 0.0005, cfg)
    bt_res = bt.run(data, sel)

    # LIVE: same bars, NO backfill (mirror the backtest's cold accumulation).
    store = Store(str(tmp_path / "live.db"))
    runner = LiveRunner(feed=ReplayFeed(a, b, idx), broker=PaperBroker(10000, 0.001, 0.0005),
                        strategy=PairsStrategy(),
                        risk=RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2),
                        store=store, selection=sel, symbols=["A", "B"],
                        strategy_cfg=cfg, sleep=lambda s: None)
    runner.run(max_iterations=len(idx))
    live_eq = store.load_equity(runner.run_id)
    live_trades = store.load_trades(runner.run_id)

    # Same number of bars and fills.
    assert len(live_eq) == len(bt_res.equity)
    assert len(live_trades) == len(bt_res.trades)
    # Identical equity curve (tight tolerance).
    for (lts, leq), (bts, beq) in zip(live_eq, bt_res.equity.items()):
        assert lts == bts.isoformat()
        assert leq == pytest.approx(beq, rel=1e-9, abs=1e-6)
    # Identical fills (symbol, price, qty) in order.
    for lt, bt_t in zip(live_trades, bt_res.trades):
        assert lt["symbol"] == bt_t["symbol"]
        assert lt["price"] == pytest.approx(bt_t["price"], rel=1e-9)
        assert lt["qty"] == pytest.approx(bt_t["qty"], rel=1e-9)
