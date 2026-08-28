# tests/test_historical.py
import os

import pandas as pd
from pairsbot.data.historical import align_closes, HistoricalLoader


def _frame(closes, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": [1.0] * len(closes)}, index=idx)


def test_align_closes_inner_joins_and_drops_missing():
    a = _frame([10, 11, 12, 13])
    b = _frame([20, 21, 22])          # one bar shorter
    out = align_closes({"ETH": a, "SOL": b})
    assert list(out.columns) == ["ETH", "SOL"]
    assert len(out) == 3              # dropped the unmatched 4th bar
    assert out.iloc[0].tolist() == [10.0, 20.0]


def test_align_closes_drops_rows_with_any_nan():
    a = _frame([10, 11, 12])
    b = _frame([20, None, 22])
    out = align_closes({"ETH": a, "SOL": b})
    assert len(out) == 2              # middle row dropped


def test_align_closes_drops_dead_symbol_instead_of_wiping_everything():
    # A symbol that returned zero bars must not empty the whole aligned frame
    # (real case: an exchange lists a market but serves no history for it).
    a = _frame([10, 11, 12])
    dead = _frame([])                 # no rows at all
    out = align_closes({"ETH": a, "ATOM": dead})
    assert list(out.columns) == ["ETH"]   # dead symbol dropped
    assert len(out) == 3                   # ETH's rows survive


class _FakeExchange:
    """Minimal ccxt-like stub: returns preloaded bars at/after `since`."""

    def __init__(self, by_market):
        self.by_market = by_market

    def fetch_ohlcv(self, market, timeframe, since, limit):
        rows = self.by_market.get(market, [])
        return [r for r in rows if r[0] >= since][:limit]


def test_loader_skips_symbols_that_return_no_data(tmp_path):
    base = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    step = 3_600_000
    eth_rows = [[base + i * step, 10, 10, 10, 10, 1.0] for i in range(5)]
    ex = _FakeExchange({"ETH/USD": eth_rows, "ATOM/USD": []})  # ATOM has nothing
    loader = HistoricalLoader(str(tmp_path), quote="USD", timeframe="1h", exchange=ex)
    out = loader.load(["ETH", "ATOM"], "2024-01-01")
    assert set(out) == {"ETH"}            # dead symbol skipped, not included
    assert len(out["ETH"]) == 5
    assert not os.path.exists(loader._cache_path("ATOM"))  # no poisoned empty cache
