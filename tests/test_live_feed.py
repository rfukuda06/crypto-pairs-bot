# tests/test_live_feed.py
import pytest

from pairsbot.data.live import LiveFeed


class _FlakyExchange:
    def __init__(self, fail_times):
        self.fail_times, self.calls = fail_times, 0

    def fetch_ohlcv(self, market, timeframe, limit):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("network boom")
        ts = 1_700_000_000_000
        return [[ts, 1, 1, 1, 1, 1], [ts + 3_600_000, 2, 2, 2, 2, 2]]


def test_feed_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("pairsbot.data.live.time.sleep", lambda *_: None)
    feed = LiveFeed("USD", "1h", exchange=_FlakyExchange(fail_times=2))
    bar = feed.latest_closed_bar(["A"])
    assert bar["A"]["close"] == 1                 # -2 element = last CLOSED bar


def test_feed_raises_after_exhausting_all_retries(monkeypatch):
    # The 'clean error on failure' half of network resilience: after 5 failed
    # attempts the feed must give up with a RuntimeError, not hang or retry forever.
    monkeypatch.setattr("pairsbot.data.live.time.sleep", lambda *_: None)
    always_failing = _FlakyExchange(fail_times=99)      # never succeeds
    feed = LiveFeed("USD", "1h", exchange=always_failing)
    with pytest.raises(RuntimeError, match="after 5 attempts"):
        feed.latest_closed_bar(["A"])
    assert always_failing.calls == 5                     # retried exactly 5 times, then gave up


def test_feed_raises_clean_error_when_too_few_bars(monkeypatch):
    monkeypatch.setattr("pairsbot.data.live.time.sleep", lambda *_: None)

    class _OneBar:
        def fetch_ohlcv(self, market, timeframe, limit):
            return [[1_700_000_000_000, 1, 1, 1, 1, 1]]

    feed = LiveFeed("USD", "1h", exchange=_OneBar())
    with pytest.raises(RuntimeError):
        feed.latest_closed_bar(["A"])
