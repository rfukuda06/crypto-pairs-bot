# tests/test_live_monitor.py
import pandas as pd
from pairsbot.core.types import PairSelection
from pairsbot.strategy.pairs import PairsStrategy
from pairsbot.risk.manager import RiskManager
from pairsbot.execution.broker import PaperBroker
from pairsbot.storage import Store
from pairsbot.live.runner import LiveRunner


class _Feed:
    """Minimal fake feed replaying a few flat-ish bars keyed by 'A'/'B'."""
    def __init__(self, idx, a, b):
        self.idx, self.a, self.b, self.i = idx, a, b, 0

    def latest_closed_bar(self, symbols):
        ts = self.idx[self.i]
        bar = {"A": {"ts": ts, "open": self.a[self.i], "close": self.a[self.i]},
               "B": {"ts": ts, "open": self.b[self.i], "close": self.b[self.i]}}
        self.i += 1
        return bar


def _runner(tmp_path):
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    feed = _Feed(idx, [100.0, 101.0, 102.0], [100.0, 100.0, 100.0])
    store = Store(str(tmp_path / "live.db"))
    sel = PairSelection(a="A", b="B", beta=1.0, pvalue=0.001)
    cfg = dict(z_window=100, entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)
    return LiveRunner(feed=feed, broker=PaperBroker(10000, 0.001, 0.0005),
                      strategy=PairsStrategy(),
                      risk=RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2),
                      store=store, selection=sel, symbols=["A", "B"],
                      strategy_cfg=cfg, sleep=lambda s: None)


def test_step_prints_live_monitor_line(tmp_path, capsys):
    runner = _runner(tmp_path)
    runner.run(max_iterations=1)     # first bar: no position yet
    out = capsys.readouterr().out
    assert "00:00Z" in out
    assert "eq $10,000" in out
    assert "pos: flat" in out


def test_starting_equity_defaults_to_broker_equity(tmp_path):
    runner = _runner(tmp_path)
    assert runner._starting_equity == 10000.0
