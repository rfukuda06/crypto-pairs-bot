# src/pairsbot/research/screen.py
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from pairsbot.core.types import PairSelection


def hedge_ratio(a: pd.Series, b: pd.Series) -> float:
    """OLS beta of log(a) on log(b): log(a) = alpha + beta*log(b) + eps."""
    la, lb = np.log(a.to_numpy()), np.log(b.to_numpy())
    X = sm.add_constant(lb)
    model = sm.OLS(la, X).fit()
    return float(model.params[1])


def screen(closes: pd.DataFrame, p_threshold: float) -> PairSelection | None:
    """Engle-Granger cointegration test over all symbol pairs (on log prices).
    Return the lowest-p-value pair below threshold, or None."""
    best: PairSelection | None = None
    for a, b in itertools.combinations(closes.columns, 2):
        la, lb = np.log(closes[a]), np.log(closes[b])
        _, pvalue, _ = coint(la, lb)
        if pvalue < p_threshold and (best is None or pvalue < best.pvalue):
            best = PairSelection(a=a, b=b, beta=hedge_ratio(closes[a], closes[b]),
                                 pvalue=float(pvalue))
    return best
