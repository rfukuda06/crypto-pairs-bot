# tests/test_backtest_golden.py
import numpy as np
import pandas as pd
from pairsbot.core.types import PairSelection
from pairsbot.strategy.pairs import PairsStrategy
from pairsbot.risk.manager import RiskManager
from pairsbot.backtest.engine import Backtester


def _ohlcv(close_series):
    idx = close_series.index
    return pd.DataFrame({"open": close_series.values, "high": close_series.values,
                         "low": close_series.values, "close": close_series.values,
                         "volume": np.ones(len(idx))}, index=idx)


def _golden_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    common = np.cumsum(rng.normal(0, 0.5, n)) + 200
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.95 * spread[t - 1] + rng.normal(0, 0.4)
    a = pd.Series(common + spread + 50, index=idx)
    b = pd.Series(common, index=idx)
    return {"A": _ohlcv(a), "B": _ohlcv(b)}


def test_backtest_runs_trades_and_is_deterministic():
    data = _golden_data()
    sel = PairSelection(a="A", b="B", beta=1.0, pvalue=0.001)
    cfg = dict(z_window=100, entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)
    bt = Backtester(strategy=PairsStrategy(),
                    risk=RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2),
                    starting_equity=10000, fee_pct=0.001, slippage_pct=0.0005,
                    strategy_cfg=cfg)
    r1 = bt.run(data, sel)
    r2 = bt.run(data, sel)
    assert len(r1.trades) > 0                     # it actually traded
    assert r1.equity.iloc[-1] == r2.equity.iloc[-1]  # deterministic
    assert len(r1.equity) == len(next(iter(data.values())))  # one equity point per bar
