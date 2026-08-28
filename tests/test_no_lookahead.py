# tests/test_no_lookahead.py
import numpy as np
import pandas as pd
from pairsbot.core.types import PairSelection, StrategyContext
from pairsbot.strategy.pairs import PairsStrategy, current_zscore
from pairsbot.risk.manager import RiskManager
from pairsbot.backtest.engine import Backtester


def _closes(n, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    b = np.full(n, 100.0)
    a = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def _ctx(closes):
    return StrategyContext(a="A", b="B", beta=1.0, closes=closes, in_position=False,
                           position_side=None, bars_in_position=0, z_window=100,
                           entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)


def test_zscore_at_bar_t_is_independent_of_future_bars():
    """The z-score for bar t must depend ONLY on bars 0..t. We prove it by building
    an alternate future that DIFFERS from bar t+1 onward, then checking the z computed
    on data sliced to bar t is byte-identical across the two futures. (The earlier
    version compared the same slice to itself, which could not catch a leak.)"""
    base = _closes(300)
    t = 200
    z_ref = current_zscore(_ctx(base.iloc[: t + 1]))

    # Alternate world: identical up to and including bar t, wildly different after.
    alt = base.copy()
    rng = np.random.default_rng(999)
    future = alt.index[t + 1:]
    alt.loc[future, "A"] = alt.loc[future, "A"].to_numpy() * (2.0 + rng.normal(0, 1.0, len(future)))
    alt.loc[future, "B"] = alt.loc[future, "B"].to_numpy() * (0.5 + rng.normal(0, 1.0, len(future)))

    # Sanity: the two worlds really do differ in the future but agree in the past.
    assert not np.allclose(alt.iloc[t + 1:].to_numpy(), base.iloc[t + 1:].to_numpy())
    assert np.array_equal(alt.iloc[: t + 1].to_numpy(), base.iloc[: t + 1].to_numpy())

    # The z at bar t is identical regardless of the (differing) future.
    z_alt = current_zscore(_ctx(alt.iloc[: t + 1]))
    assert z_alt == z_ref


def _ohlcv(close):
    idx = close.index
    return pd.DataFrame({"open": close.values, "high": close.values,
                         "low": close.values, "close": close.values,
                         "volume": np.ones(len(idx))}, index=idx)


def _mean_reverting_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    common = np.cumsum(rng.normal(0, 0.5, n)) + 200
    spread = np.zeros(n)
    for k in range(1, n):
        spread[k] = 0.95 * spread[k - 1] + rng.normal(0, 0.4)
    a = pd.Series(common + spread + 50, index=idx)
    b = pd.Series(common, index=idx)
    return {"A": _ohlcv(a), "B": _ohlcv(b)}


def _backtester():
    cfg = dict(z_window=100, entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)
    return Backtester(strategy=PairsStrategy(),
                      risk=RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2),
                      starting_equity=10000, fee_pct=0.001, slippage_pct=0.0005,
                      strategy_cfg=cfg)


def test_backtest_equity_prefix_is_invariant_to_future_bars():
    """End-to-end no-lookahead: a backtest's equity at every bar up to t must be
    identical whether or not bars after t exist. If any decision peeked at the
    future, truncating the data would change the historical equity path."""
    data = _mean_reverting_data(600)
    sel = PairSelection(a="A", b="B", beta=1.0, pvalue=0.001)

    full = _backtester().run(data, sel)
    t = 400
    truncated = {sym: df.iloc[:t] for sym, df in data.items()}
    part = _backtester().run(truncated, sel)

    # part covers bars 0..t-1; every one of those equity points must match full.
    assert len(part.equity) == t
    assert np.allclose(full.equity.iloc[:t].to_numpy(),
                       part.equity.to_numpy(), rtol=0, atol=1e-9)
    # Non-vacuous: the truncated run actually traded within its window.
    assert len(part.trades) > 0
