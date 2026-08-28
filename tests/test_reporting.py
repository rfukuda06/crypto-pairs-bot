# tests/test_reporting.py
import numpy as np
import pandas as pd
from pairsbot.reporting.report import compute_metrics


def test_metrics_on_known_series():
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    equity = pd.Series([10000, 10100, 10050, 10200, 10150], index=idx, dtype=float)
    m = compute_metrics(equity, trades=[])
    assert m["total_return"] == (10150 / 10000 - 1)
    assert m["max_drawdown"] < 0            # there was a dip
    assert "sharpe" in m and "num_trades" in m


def test_metrics_num_trades_counts_fills():
    idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    equity = pd.Series([10000, 10100], index=idx, dtype=float)
    trades = [{"reason": "enter"}, {"reason": "enter"}, {"reason": "exit"}, {"reason": "exit"}]
    m = compute_metrics(equity, trades=trades)
    assert m["num_trades"] == 4
