# tests/test_screen.py
import numpy as np
import pandas as pd
import pytest
from pairsbot.research.screen import (screen, hedge_ratio, num_candidate_pairs,
                                      sidak_pvalue)


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


def test_num_candidate_pairs_counts_all_symbol_combinations():
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    cols = list("ABCDEFGHIJK")                       # 11 symbols, like the universe
    closes = pd.DataFrame({c: range(5) for c in cols}, index=idx)
    assert num_candidate_pairs(closes) == 55         # C(11, 2) tests were run


def test_sidak_pvalue_corrects_for_search_size():
    # p=0.02166 is the MINIMUM over 55 pairs; after multiple-testing correction
    # it is no longer significant (~0.70), which is the honest disclosure.
    assert sidak_pvalue(0.02166, 55) == pytest.approx(0.70, abs=0.02)
    assert sidak_pvalue(0.01, 1) == pytest.approx(0.01)   # a single test is a no-op


def test_screen_returns_none_when_nothing_cointegrated():
    rng = np.random.default_rng(1)
    n = 800
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    closes = pd.DataFrame({
        "A": np.cumsum(rng.normal(0, 1, n)) + 100,     # independent walks
        "B": np.cumsum(rng.normal(0, 1, n)) + 100,
    }, index=idx)
    assert screen(closes, p_threshold=0.01) is None
