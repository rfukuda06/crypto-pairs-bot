# tests/test_live_runner.py
import pandas as pd
from pairsbot.core.types import PairSelection
from pairsbot.strategy.pairs import PairsStrategy
from pairsbot.risk.manager import RiskManager
from pairsbot.execution.broker import PaperBroker
from pairsbot.storage import Store
from pairsbot.live.runner import LiveRunner


class FakeFeed:
    """Replays a prebuilt sequence of closed bars, one call at a time."""
    def __init__(self, closes_a, closes_b, index):
        self.a, self.b, self.index, self.i = closes_a, closes_b, index, 0

    def latest_closed_bar(self, symbols):
        ts = self.index[self.i]
        bar = {"A": {"ts": ts, "open": self.a[self.i], "close": self.a[self.i]},
               "B": {"ts": ts, "open": self.b[self.i], "close": self.b[self.i]}}
        self.i += 1
        return bar


def _series(n=300):
    import numpy as np
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    common = np.cumsum(rng.normal(0, 0.5, n)) + 200
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.95 * spread[t - 1] + rng.normal(0, 0.4)
    return (common + spread + 50).tolist(), common.tolist(), idx


def test_live_runner_processes_bars_and_persists(tmp_path):
    a, b, idx = _series()
    feed = FakeFeed(a, b, idx)
    store = Store(str(tmp_path / "live.db"))
    sel = PairSelection(a="A", b="B", beta=1.0, pvalue=0.001)
    cfg = dict(z_window=100, entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)
    runner = LiveRunner(feed=feed, broker=PaperBroker(10000, 0.001, 0.0005),
                        strategy=PairsStrategy(),
                        risk=RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2),
                        store=store, selection=sel, symbols=["A", "B"],
                        strategy_cfg=cfg, sleep=lambda s: None)
    runner.run(max_iterations=250)   # bounded for the test
    eq = store.load_equity(runner.run_id)
    assert len(eq) == 250
    assert len(store.load_trades(runner.run_id)) > 0


def test_backfill_lets_runner_trade_on_first_bars(tmp_path):
    a, b, idx = _series(n=300)
    # seed with the first 200 bars; feed replays the remaining 100
    seed = pd.DataFrame({"A": a[:200], "B": b[:200]}, index=idx[:200])
    feed = FakeFeed(a[200:], b[200:], idx[200:])
    store = Store(str(tmp_path / "live.db"))
    sel = PairSelection(a="A", b="B", beta=1.0, pvalue=0.001)
    cfg = dict(z_window=100, entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)
    runner = LiveRunner(feed=feed, broker=PaperBroker(10000, 0.001, 0.0005),
                        strategy=PairsStrategy(),
                        risk=RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2),
                        store=store, selection=sel, symbols=["A", "B"],
                        strategy_cfg=cfg, sleep=lambda s: None,
                        initial_closes=seed)
    runner.run(max_iterations=5)          # only 5 live bars, but window is pre-filled
    assert len(runner._closes) >= 100      # window already deep enough to compute z


def test_duplicate_bar_is_skipped(tmp_path):
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")

    class StuckFeed:
        """Always returns the SAME (first) bar."""
        def latest_closed_bar(self, symbols):
            return {"A": {"ts": idx[0], "open": 100.0, "close": 100.0},
                    "B": {"ts": idx[0], "open": 50.0, "close": 50.0}}

    class CountingStrategy:
        """Records how many times a bar is actually processed."""
        def __init__(self):
            self.calls = 0
        def on_bar(self, ctx):
            self.calls += 1
            return []

    store = Store(str(tmp_path / "live.db"))
    sel = PairSelection(a="A", b="B", beta=1.0, pvalue=0.001)
    cfg = dict(z_window=5, entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)
    strat = CountingStrategy()
    runner = LiveRunner(feed=StuckFeed(), broker=PaperBroker(10000, 0.001, 0.0005),
                        strategy=strat,
                        risk=RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2),
                        store=store, selection=sel, symbols=["A", "B"],
                        strategy_cfg=cfg, sleep=lambda s: None)
    runner.run(max_iterations=5)
    assert strat.calls == 1                          # duplicate bars skipped, not re-processed
    assert len(runner._closes) == 1                  # only the first bar recorded


def test_live_runner_recovers_positions_on_restart(tmp_path):
    db = str(tmp_path / "live.db")
    store = Store(db)
    run_id = store.start_run(mode="live", pair="A/B")
    store.upsert_position(run_id, "A", qty=2.0, avg_price=250.0)
    store.upsert_position(run_id, "B", qty=-2.0, avg_price=200.0)
    broker = PaperBroker(10000, 0.001, 0.0005)
    LiveRunner.restore_broker(broker, store, run_id)
    pos = broker.positions()
    assert pos["A"].qty == 2.0 and pos["B"].qty == -2.0
