# src/pairsbot/data/live.py
from __future__ import annotations

import logging
import time

import pandas as pd

log = logging.getLogger(__name__)


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

    def _fetch(self, market: str) -> list[list]:
        """Single network boundary with bounded retry (mirrors HistoricalLoader)."""
        for attempt in range(5):
            try:
                return self.exchange.fetch_ohlcv(market, timeframe=self.timeframe, limit=2)
            except Exception as e:  # ccxt network/rate-limit errors
                wait = 2 ** attempt
                log.warning("live fetch_ohlcv %s failed (%s); retry in %ss", market, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"live fetch_ohlcv failed for {market} after 5 attempts")

    def latest_closed_bar(self, symbols: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for sym in symbols:
            market = f"{sym}/{self.quote}"
            ohlcv = self._fetch(market)
            if len(ohlcv) < 2:
                raise RuntimeError(f"{market}: expected >=2 bars, got {len(ohlcv)}")
            ts, o, h, l, c, v = ohlcv[-2]   # -2 = last fully closed bar
            out[sym] = {"ts": pd.to_datetime(ts, unit="ms", utc=True),
                        "open": o, "close": c}
        return out
