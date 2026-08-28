# tests/test_split.py
import pandas as pd
import pytest
from pairsbot.research.split import train_bars, split_closes


def _closes(n):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"A": range(n), "B": range(n)}, index=idx)


def test_train_bars_scales_with_timeframe():
    assert train_bars(180, "1h") == 180 * 24
    assert train_bars(180, "1d") == 180


def test_train_bars_rejects_unknown_timeframe():
    with pytest.raises(ValueError):
        train_bars(180, "5m")


def test_split_partitions_in_and_out_of_sample():
    closes = _closes(1000)
    ins, oos = split_closes(closes, train_window_days=10, timeframe="1h")  # 240 bars
    assert len(ins) == 240
    assert len(oos) == 760
    # exact partition: concatenation reproduces the original with no overlap
    assert list(ins.index) + list(oos.index) == list(closes.index)
    assert ins.index.max() < oos.index.min()


def test_split_out_of_sample_empty_when_data_shorter_than_window():
    closes = _closes(100)
    ins, oos = split_closes(closes, train_window_days=10, timeframe="1h")  # 240 > 100
    assert len(ins) == 100
    assert len(oos) == 0
