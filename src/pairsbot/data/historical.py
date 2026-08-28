# src/pairsbot/data/historical.py
from __future__ import annotations

import logging
import os
import time

import pandas as pd

log = logging.getLogger(__name__)

_TF_MS = {"1h": 3_600_000, "1d": 86_400_000}


def align_closes(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-join close prices across symbols; drop any timestamp missing a
    value for any symbol. Symbols with no data at all are dropped entirely first,
    so one dead symbol cannot empty the whole frame. Pure function — no I/O."""
    closes = pd.DataFrame({sym: df["close"] for sym, df in frames.items()})
    dead = [c for c in closes.columns if closes[c].isna().all()]
    if dead:
        log.warning("align_closes dropping symbols with no data: %s", dead)
        closes = closes.drop(columns=dead)
    before = len(closes)
    closes = closes.dropna(how="any")
    dropped = before - len(closes)
    if dropped:
        log.warning("align_closes dropped %d incomplete timestamps", dropped)
    return closes


class HistoricalLoader:
    def __init__(self, cache_dir: str, quote: str = "USD", timeframe: str = "1h",
                 exchange=None, exchange_name: str = "bitstamp"):
        self.cache_dir = cache_dir
        self.quote = quote
        self.timeframe = timeframe
        os.makedirs(cache_dir, exist_ok=True)
        if exchange is None:
            import ccxt
            exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})
        self.exchange = exchange

    def _cache_path(self, symbol: str) -> str:
        return os.path.join(self.cache_dir, f"{symbol}_{self.quote}_{self.timeframe}.parquet")

    def _fetch_ohlcv(self, symbol: str, since_ms: int) -> list[list]:
        """Single network boundary. Wrapped with bounded retry."""
        market = f"{symbol}/{self.quote}"
        for attempt in range(5):
            try:
                return self.exchange.fetch_ohlcv(
                    market, timeframe=self.timeframe, since=since_ms, limit=1000)
            except Exception as e:  # ccxt network/rate-limit errors
                wait = 2 ** attempt
                log.warning("fetch_ohlcv %s failed (%s); retry in %ss", market, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"fetch_ohlcv failed for {market} after 5 attempts")

    def _download(self, symbol: str, start: str) -> pd.DataFrame:
        since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        step = _TF_MS[self.timeframe]
        rows: list[list] = []
        while True:
            batch = self._fetch_ohlcv(symbol, since)
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1][0] + step
            if len(batch) < 1000:
                break
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("ts")
        df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df[["open", "high", "low", "close", "volume"]]

    def load(self, symbols: list[str], start: str) -> dict[str, pd.DataFrame]:
        """Load OHLCV per symbol, using parquet cache when present. Symbols that
        return no data are skipped (with a warning) rather than cached empty, so a
        market the exchange lists but does not serve can't poison the pipeline."""
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            path = self._cache_path(sym)
            if os.path.exists(path):
                df = pd.read_parquet(path)
            else:
                df = self._download(sym, start)
                if len(df):
                    df.to_parquet(path)
            if not len(df):
                log.warning("no data returned for %s; skipping", sym)
                continue
            out[sym] = df
        return out
