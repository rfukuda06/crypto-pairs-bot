# tests/test_pairs_strategy.py
import numpy as np
import pandas as pd
from pairsbot.core.types import StrategyContext, SpreadSide
from pairsbot.strategy.pairs import PairsStrategy, current_zscore


def _ctx(closes, in_position=False, side=None, bars_in=0, **over):
    base = dict(a="A", b="B", beta=1.0, closes=closes, in_position=in_position,
                position_side=side, bars_in_position=bars_in, z_window=100,
                entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)
    base.update(over)
    return StrategyContext(**base)


def _closes_with_final_z(target_z, n=200, window=100):
    # Build a spread series whose last-bar z-score ≈ target_z.
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    b = np.full(n, 100.0)
    spread = np.zeros(n)
    spread[-1] = 0.0  # placeholder; set A so log-spread hits target
    # Use flat B=100 so log(A)-log(B) = log(A)-log(100). Make spread mean 0, std s.
    rng = np.random.default_rng(0)
    s = rng.normal(0, 0.01, n)
    s[-1] = 0.0
    log_spread = s.copy()
    win = s[-window:]
    log_spread[-1] = win.mean() + target_z * win.std()
    a = np.exp(log_spread) * 100.0
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_enters_short_spread_when_z_high_and_flat():
    closes = _closes_with_final_z(2.5)
    sigs = PairsStrategy().on_bar(_ctx(closes))
    assert len(sigs) == 1
    assert sigs[0].kind == "enter"
    assert sigs[0].spread_side == SpreadSide.SHORT   # z high -> short the spread


def test_enters_long_spread_when_z_low_and_flat():
    closes = _closes_with_final_z(-2.5)
    sigs = PairsStrategy().on_bar(_ctx(closes))
    assert sigs[0].spread_side == SpreadSide.LONG


def test_no_entry_when_z_between_thresholds():
    closes = _closes_with_final_z(1.0)
    assert PairsStrategy().on_bar(_ctx(closes)) == []


def test_exits_on_mean_reversion():
    closes = _closes_with_final_z(0.2)
    sigs = PairsStrategy().on_bar(_ctx(closes, in_position=True, side=SpreadSide.SHORT))
    assert len(sigs) == 1 and sigs[0].kind == "exit"


def test_exits_on_stop_z():
    closes = _closes_with_final_z(4.0)
    sigs = PairsStrategy().on_bar(_ctx(closes, in_position=True, side=SpreadSide.SHORT))
    assert sigs[0].kind == "exit" and "stop" in sigs[0].reason


def test_exits_on_time_stop_even_if_z_still_extreme():
    closes = _closes_with_final_z(2.5)
    sigs = PairsStrategy().on_bar(
        _ctx(closes, in_position=True, side=SpreadSide.SHORT, bars_in=168))
    assert sigs[0].kind == "exit" and "time" in sigs[0].reason
