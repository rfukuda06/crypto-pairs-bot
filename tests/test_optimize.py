# tests/test_optimize.py
import numpy as np
import pandas as pd
from pairsbot.core.types import PairSelection
from pairsbot.research.optimize import optimize_params


def _ohlcv(close):
    idx = close.index
    return pd.DataFrame({"open": close.values, "high": close.values, "low": close.values,
                         "close": close.values, "volume": np.ones(len(idx))}, index=idx)


def _mean_reverting(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    common = np.cumsum(rng.normal(0, 0.5, n)) + 200
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.4)
    a = pd.Series(common + spread + 50, index=idx)
    b = pd.Series(common, index=idx)
    return {"A": _ohlcv(a), "B": _ohlcv(b)}


def _run(**over):
    data = _mean_reverting()
    sel = PairSelection("A", "B", 1.0, 0.001)
    kw = dict(z_window=100, gross_exposure_pct=0.5, max_drawdown_pct=0.2,
              starting_equity=10000, fee_pct=0.001, slippage_pct=0.0005, min_trades=1)
    kw.update(over)
    return optimize_params(data, sel, **kw)


def test_optimize_returns_valid_best_config():
    best, ranked = _run()
    assert set(best) == {"z_window", "entry_z", "exit_z", "stop_z", "max_holding_bars"}
    # every evaluated config must respect the ordering constraint
    for r in ranked:
        c = r["cfg"]
        assert c["exit_z"] < c["entry_z"] < c["stop_z"]


def test_optimize_best_is_top_ranked_by_objective():
    best, ranked = _run(objective="sharpe")
    objs = [r["metrics"]["sharpe"] for r in ranked]
    assert objs == sorted(objs, reverse=True)   # ranked is sorted desc by objective
    assert best == ranked[0]["cfg"]


def test_optimize_honors_a_custom_single_point_grid():
    grid = {"entry_z": [2.0], "exit_z": [0.5], "stop_z": [3.5], "max_holding_bars": [168]}
    best, ranked = _run(grid=grid)
    assert len(ranked) == 1
    assert best == {"z_window": 100, "entry_z": 2.0, "exit_z": 0.5,
                    "stop_z": 3.5, "max_holding_bars": 168}
