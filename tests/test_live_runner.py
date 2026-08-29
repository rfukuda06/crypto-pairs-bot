# tests/test_live_runner.py
import pandas as pd
import pytest
from pairsbot.core.types import PairSelection, Position, SpreadSide
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


def test_no_double_entry_when_broker_already_holds(tmp_path):
    # Six bars; a spread spike drives an entry signal on the later bars.
    idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    a = [100.0, 100.0, 100.0, 100.0, 140.0, 140.0]
    b = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    feed = FakeFeed(a, b, idx)
    store = Store(str(tmp_path / "live.db"))
    sel = PairSelection(a="A", b="B", beta=1.0, pvalue=0.001)
    cfg = dict(z_window=3, entry_z=0.5, exit_z=0.1, stop_z=9.0, max_holding_bars=999)
    broker = PaperBroker(10000, 0.001, 0.0005)
    # a pre-existing position the fresh runner is unaware of (in_position stays False)
    broker._pos["A"] = Position("A", qty=1.0, avg_price=100.0)
    broker._pos["B"] = Position("B", qty=-1.0, avg_price=100.0)
    runner = LiveRunner(feed=feed, broker=broker, strategy=PairsStrategy(),
                        risk=RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2),
                        store=store, selection=sel, symbols=["A", "B"],
                        strategy_cfg=cfg, sleep=lambda s: None)
    runner.run(max_iterations=6)
    # guard must prevent stacking a second position on the existing one
    assert broker.positions()["A"].qty == pytest.approx(1.0)


def test_run_loop_survives_a_feed_error(tmp_path):
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")

    class OneErrorFeed:
        def __init__(self):
            self.calls = 0
        def latest_closed_bar(self, symbols):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            i = self.calls - 2
            return {"A": {"ts": idx[i], "open": 100.0 + i, "close": 100.0 + i},
                    "B": {"ts": idx[i], "open": 50.0, "close": 50.0}}

    store = Store(str(tmp_path / "live.db"))
    sel = PairSelection(a="A", b="B", beta=1.0, pvalue=0.001)
    cfg = dict(z_window=2, entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)
    runner = LiveRunner(feed=OneErrorFeed(), broker=PaperBroker(10000, 0.001, 0.0005),
                        strategy=PairsStrategy(),
                        risk=RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2),
                        store=store, selection=sel, symbols=["A", "B"],
                        strategy_cfg=cfg, sleep=lambda s: None)
    runner.run(max_iterations=3)                   # 1st poll errors, loop must continue
    assert len(store.load_equity(runner.run_id)) >= 1


def test_poll_seconds_longer_than_timeframe_is_rejected(tmp_path):
    class TfFeed:
        timeframe = "1h"
        def latest_closed_bar(self, symbols):  # pragma: no cover - not reached
            return {}
    store = Store(str(tmp_path / "live.db"))
    sel = PairSelection(a="A", b="B", beta=1.0, pvalue=0.001)
    cfg = dict(z_window=5, entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)
    with pytest.raises(ValueError):
        LiveRunner(feed=TfFeed(), broker=PaperBroker(10000, 0.001, 0.0005),
                   strategy=PairsStrategy(),
                   risk=RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2),
                   store=store, selection=sel, symbols=["A", "B"],
                   strategy_cfg=cfg, sleep=lambda s: None, poll_seconds=7200)


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


def test_restart_ignores_dust_positions(tmp_path):
    store = Store(str(tmp_path / "live.db"))
    sel = PairSelection(a="A", b="B", beta=1.0, pvalue=0.001)
    run_id = store.start_run(mode="live", pair="A/B")
    # dust left by a prior flatten: tiny notional relative to equity
    store.upsert_position(run_id, "A", qty=0.001, avg_price=250.0)   # ~$0.25
    store.upsert_position(run_id, "B", qty=-0.002, avg_price=125.0)  # ~$0.25
    store.record_equity(run_id, "t0", 10000.0)
    broker = PaperBroker(10000, 0.001, 0.0005)
    LiveRunner.restore_broker(broker, store, run_id)
    seed = pd.DataFrame({"A": [250.0] * 10, "B": [125.0] * 10},
                        index=pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC"))
    cfg = dict(z_window=5, entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)
    runner = LiveRunner(feed=object(), broker=broker, strategy=PairsStrategy(),
                        risk=RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2),
                        store=store, selection=sel, symbols=["A", "B"],
                        strategy_cfg=cfg, sleep=lambda s: None,
                        initial_closes=seed, run_id=run_id)
    assert runner.in_position is False                 # dust is not a real position


def test_restart_keeps_equity_continuous(tmp_path):
    store = Store(str(tmp_path / "live.db"))
    run_id = store.start_run(mode="live", pair="A/B")
    store.upsert_position(run_id, "A", qty=2.0, avg_price=250.0)
    store.upsert_position(run_id, "B", qty=-4.0, avg_price=125.0)
    store.record_equity(run_id, "t0", 9500.0)          # prior equity reflects fees/PnL
    broker = PaperBroker(10000, 0.001, 0.0005)
    LiveRunner.restore_broker(broker, store, run_id)
    # equity right after restore (marked at cost basis) must match the last recorded
    # equity — not jump back toward the starting balance.
    assert broker.equity() == pytest.approx(9500.0)


def test_restart_restores_position_and_in_position_flag(tmp_path):
    db = str(tmp_path / "live.db")
    store = Store(db)
    sel = PairSelection(a="A", b="B", beta=1.0, pvalue=0.001)
    # simulate a prior live run that entered a LONG-spread position (long A, short B)
    run_id = store.start_run(mode="live", pair="A/B")
    store.upsert_position(run_id, "A", qty=2.0, avg_price=250.0)
    store.upsert_position(run_id, "B", qty=-4.0, avg_price=125.0)
    store.record_equity(run_id, "2024-01-01T00:00:00", 10000.0)
    store.record_trade(run_id, "2024-01-01T00:00:00", "A", 500.0, 250.0, 2.0, 0.5, "z -2.1")

    # restart: restore broker, then build a resuming runner
    broker = PaperBroker(10000, 0.001, 0.0005)
    LiveRunner.restore_broker(broker, store, run_id)
    seed = pd.DataFrame({"A": [250.0] * 10, "B": [125.0] * 10},
                        index=pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC"))
    cfg = dict(z_window=5, entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)
    runner = LiveRunner(feed=object(), broker=broker, strategy=PairsStrategy(),
                        risk=RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2),
                        store=store, selection=sel, symbols=["A", "B"],
                        strategy_cfg=cfg, sleep=lambda s: None,
                        initial_closes=seed, run_id=run_id)

    assert runner.run_id == run_id                     # resumed, not a new run
    assert runner.in_position is True
    assert runner.side == SpreadSide.LONG              # qty A > 0 => long spread
    assert broker.positions()["A"].qty == 2.0
