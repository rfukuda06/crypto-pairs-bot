# src/pairsbot/data/live.py
from __future__ import annotations

import pandas as pd


class LiveFeed:
    """Fetches the most recently CLOSED hourly bar for each symbol."""

    def __init__(self, quote: str = "USD", timeframe: str = "1h", exchange=None,
                 exchange_name: str = "bitstamp"):
        self.quote = quote
        self.timeframe = timeframe
        if exchange is None:
            import ccxt
            exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})
        self.exchange = exchange

    def latest_closed_bar(self, symbols: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for sym in symbols:
            ohlcv = self.exchange.fetch_ohlcv(
                f"{sym}/{self.quote}", timeframe=self.timeframe, limit=2)
            ts, o, h, l, c, v = ohlcv[-2]   # -2 = last fully closed bar
            out[sym] = {"ts": pd.to_datetime(ts, unit="ms", utc=True),
                        "open": o, "close": c}
        return out
