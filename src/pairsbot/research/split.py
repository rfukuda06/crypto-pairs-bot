# src/pairsbot/research/split.py
from __future__ import annotations

import pandas as pd

# Bars per calendar day for each supported timeframe.
_BARS_PER_DAY = {"1h": 24, "1d": 1}


def train_bars(train_window_days: int, timeframe: str) -> int:
    """Number of bars in the in-sample training window for a timeframe."""
    try:
        per_day = _BARS_PER_DAY[timeframe]
    except KeyError as e:
        raise ValueError(f"unsupported timeframe: {timeframe!r}") from e
    return train_window_days * per_day


def split_closes(closes: pd.DataFrame, train_window_days: int,
                 timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split aligned closes into (in_sample, out_of_sample) by the training window.

    The pair and hedge ratio are chosen on the in-sample slice only; the backtest
    then trades the out-of-sample slice, so selection never peeks at the data it is
    evaluated on. If there are fewer bars than the window, out_of_sample is empty.
    """
    n = train_bars(train_window_days, timeframe)
    return closes.iloc[:n], closes.iloc[n:]
