# tests/test_screen.py
import numpy as np
import pandas as pd
from pairsbot.research.screen import screen, hedge_ratio


def _cointegrated_pair(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 1, n)) + 100      # shared random walk
    spread = np.zeros(n)
    for t in range(1, n):                              # mean-reverting spread
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.5)
    a = common + spread
    b = common
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_hedge_ratio_recovers_unit_beta():
    closes = _cointegrated_pair()
    beta = hedge_ratio(closes["A"], closes["B"])
    assert 0.7 < beta < 1.3           # A ≈ 1*B + stationary spread


def test_screen_selects_cointegrated_pair_below_threshold():
    closes = _cointegrated_pair()
    sel = screen(closes, p_threshold=0.05)
    assert sel is not None
    assert {sel.a, sel.b} == {"A", "B"}
    assert sel.pvalue < 0.05


def test_screen_returns_none_when_nothing_cointegrated():
    rng = np.random.default_rng(1)
    n = 800
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    closes = pd.DataFrame({
        "A": np.cumsum(rng.normal(0, 1, n)) + 100,     # independent walks
        "B": np.cumsum(rng.normal(0, 1, n)) + 100,
    }, index=idx)
    assert screen(closes, p_threshold=0.01) is None
