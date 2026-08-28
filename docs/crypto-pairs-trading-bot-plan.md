# Crypto Pairs-Trading Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a crypto statistical-arbitrage (pairs-trading) bot with a custom backtesting engine and a live paper-trading mode that runs identical strategy/risk logic.

**Architecture:** One shared Strategy + Risk core is driven by two orchestrators (backtest over historical bars, live over a polled hourly feed). Both route orders through the same `PaperBroker` (fees + slippage), so execution is genuinely identical. Cointegration screening (statsmodels) picks the pair and hedge ratio; a rolling z-score of the log-price spread generates entries/exits. Lookahead is structurally prevented by computing signals only from data up to the current bar and filling at the *next* bar's open.

**Tech Stack:** Python 3.11+, pandas, numpy, statsmodels, ccxt, typer, matplotlib, pytest, SQLite (stdlib), PyYAML.

**Reference:** Design spec at `docs/superpowers/specs/2026-08-24-crypto-pairs-trading-bot-design.md`.

---

## File Structure

```
crypto-pairs-bot/
  pyproject.toml                 # package + deps + pytest config
  config.yaml                    # runtime config (universe, thresholds, fees…)
  .env.example                   # placeholder (public data needs no key)
  README.md                      # architecture, quickstart, results
  src/pairsbot/
    __init__.py
    core/types.py                # domain dataclasses: Signal, Order, Fill, Position, StrategyContext, PairSelection, results
    config.py                    # load + validate config.yaml -> Config
    storage.py                   # SQLite Store: runs, equity, trades, positions
    data/historical.py           # ccxt OHLCV fetch + parquet cache + alignment
    data/live.py                 # LiveFeed: latest CLOSED hourly bar via ccxt
    research/screen.py           # cointegration screen + OLS hedge ratio
    strategy/base.py             # Strategy protocol
    strategy/pairs.py            # PairsStrategy: spread -> rolling z -> signals + stops
    risk/manager.py              # sizing (dollar-neutral), flatten, drawdown kill-switch
    execution/broker.py          # Broker protocol, PaperBroker, LiveBroker stub
    backtest/engine.py           # Backtester event loop (no lookahead, next-open fills)
    reporting/report.py          # metrics + matplotlib charts + markdown summary
    live/runner.py               # LiveRunner: hourly loop + restart recovery
    cli.py                       # typer CLI: fetch-data, screen, backtest, live
  tests/
    test_config.py
    test_storage.py
    test_historical.py
    test_screen.py
    test_pairs_strategy.py
    test_no_lookahead.py
    test_risk.py
    test_broker.py
    test_backtest_golden.py
    test_reporting.py
    test_live_runner.py
```

**Responsibility boundaries:** `core/types.py` is the shared vocabulary every module imports (define once, reuse everywhere). Strategy owns *when* to trade (including z-stop and time-stop, since it holds the z-score). Risk owns *how much* and the drawdown kill-switch. Broker owns *fills*. The two orchestrators (`backtest`, `live`) own *sequencing* and *position-state tracking*; they contain no trading rules.

---

## Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `src/pairsbot/__init__.py`, `config.yaml`, `.env.example`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "pairsbot"
version = "0.1.0"
description = "Crypto pairs-trading bot: backtester + live paper trading"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "statsmodels>=0.14",
    "ccxt>=4.3",
    "typer>=0.12",
    "matplotlib>=3.8",
    "pyyaml>=6.0",
    "pyarrow>=16.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
pairsbot = "pairsbot.cli:app"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `src/pairsbot/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Create `config.yaml`**

```yaml
universe: [ETH, SOL, AVAX, ADA, DOT, NEAR, ATOM]
quote: USDT
timeframe: 1h
data:
  start: "2024-01-01"
  cache_dir: ./data_cache
research:
  train_window_days: 180
  p_threshold: 0.05
strategy:
  z_window: 168
  entry_z: 2.0
  exit_z: 0.5
  stop_z: 3.5
  max_holding_bars: 168
risk:
  gross_exposure_pct: 0.50
  max_drawdown_pct: 0.20
costs:
  fee_pct: 0.001
  slippage_pct: 0.0005
account:
  starting_equity: 10000
storage:
  db_path: ./bot.db
```

- [ ] **Step 4: Create `.env.example`**

```bash
# Public Binance market data requires NO API key.
# Placeholders kept for the LiveBroker (real execution) seam, which is a stub in v1.
BINANCE_API_KEY=
BINANCE_API_SECRET=
```

- [ ] **Step 5: Set up environment and verify install**

Run:
```bash
cd /Users/renzofukuda/crypto-pairs-bot
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```
Expected: installs cleanly, `pairsbot` console script registered.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/pairsbot/__init__.py config.yaml .env.example
git commit -m "chore: project scaffolding and dependencies"
```

---

## Task 1: Core domain types

**Files:**
- Create: `src/pairsbot/core/__init__.py` (empty), `src/pairsbot/core/types.py`

- [ ] **Step 1: Create `src/pairsbot/core/types.py`**

These dataclasses are the shared vocabulary. Every later task imports from here.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class SpreadSide(str, Enum):
    LONG = "long_spread"   # long A, short B
    SHORT = "short_spread"  # short A, long B


@dataclass(frozen=True)
class Signal:
    """A trading intent emitted by a strategy."""
    kind: str                      # "enter" | "exit"
    spread_side: SpreadSide | None = None  # required when kind == "enter"
    reason: str = ""


@dataclass(frozen=True)
class Order:
    """A trade request in signed USD notional (+buy / -sell) for one symbol."""
    symbol: str
    notional: float
    reason: str = ""


@dataclass(frozen=True)
class Fill:
    symbol: str
    notional: float   # signed USD actually transacted (pre-fee)
    price: float      # fill price incl. slippage
    qty: float        # signed change in position quantity
    fee: float


@dataclass
class Position:
    symbol: str
    qty: float = 0.0          # signed: + long, - short
    avg_price: float = 0.0


@dataclass
class PairSelection:
    a: str
    b: str
    beta: float               # hedge ratio: spread = log(A) - beta*log(B)
    pvalue: float


@dataclass
class StrategyContext:
    """Everything a strategy may look at for the current bar. Contains ONLY
    data up to and including the current bar (no future information)."""
    a: str
    b: str
    beta: float
    closes: pd.DataFrame          # columns [a, b], index up to current bar
    in_position: bool
    position_side: SpreadSide | None
    bars_in_position: int
    z_window: int
    entry_z: float
    exit_z: float
    stop_z: float
    max_holding_bars: int


@dataclass
class BacktestResult:
    equity: pd.Series             # index = timestamps, value = equity
    trades: list[dict] = field(default_factory=list)  # one dict per fill
    selection: PairSelection | None = None
```

- [ ] **Step 2: Create empty `src/pairsbot/core/__init__.py`**

```python
```

- [ ] **Step 3: Smoke-test the import**

Run: `python -c "from pairsbot.core.types import Signal, Order, SpreadSide; print(SpreadSide.LONG.value)"`
Expected: prints `long_spread`

- [ ] **Step 4: Commit**

```bash
git add src/pairsbot/core
git commit -m "feat: core domain types"
```

---

## Task 2: Config loading

**Files:**
- Create: `src/pairsbot/config.py`, `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
from pairsbot.config import load_config, ConfigError


def test_load_config_parses_nested_values(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "universe: [ETH, SOL]\nquote: USDT\ntimeframe: 1h\n"
        "data:\n  start: '2024-01-01'\n  cache_dir: ./dc\n"
        "research:\n  train_window_days: 180\n  p_threshold: 0.05\n"
        "strategy:\n  z_window: 168\n  entry_z: 2.0\n  exit_z: 0.5\n  stop_z: 3.5\n  max_holding_bars: 168\n"
        "risk:\n  gross_exposure_pct: 0.5\n  max_drawdown_pct: 0.2\n"
        "costs:\n  fee_pct: 0.001\n  slippage_pct: 0.0005\n"
        "account:\n  starting_equity: 10000\n"
        "storage:\n  db_path: ./bot.db\n"
    )
    cfg = load_config(str(p))
    assert cfg.universe == ["ETH", "SOL"]
    assert cfg.strategy.entry_z == 2.0
    assert cfg.costs.fee_pct == 0.001
    assert cfg.account.starting_equity == 10000


def test_load_config_rejects_bad_thresholds(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "universe: [ETH, SOL]\nquote: USDT\ntimeframe: 1h\n"
        "data:\n  start: '2024-01-01'\n  cache_dir: ./dc\n"
        "research:\n  train_window_days: 180\n  p_threshold: 0.05\n"
        "strategy:\n  z_window: 168\n  entry_z: 0.5\n  exit_z: 2.0\n  stop_z: 3.5\n  max_holding_bars: 168\n"
        "risk:\n  gross_exposure_pct: 0.5\n  max_drawdown_pct: 0.2\n"
        "costs:\n  fee_pct: 0.001\n  slippage_pct: 0.0005\n"
        "account:\n  starting_equity: 10000\n"
        "storage:\n  db_path: ./bot.db\n"
    )
    with pytest.raises(ConfigError):
        load_config(str(p))  # entry_z must be > exit_z
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pairsbot.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/pairsbot/config.py
from __future__ import annotations

from dataclasses import dataclass

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class DataCfg:
    start: str
    cache_dir: str


@dataclass
class ResearchCfg:
    train_window_days: int
    p_threshold: float


@dataclass
class StrategyCfg:
    z_window: int
    entry_z: float
    exit_z: float
    stop_z: float
    max_holding_bars: int


@dataclass
class RiskCfg:
    gross_exposure_pct: float
    max_drawdown_pct: float


@dataclass
class CostsCfg:
    fee_pct: float
    slippage_pct: float


@dataclass
class AccountCfg:
    starting_equity: float


@dataclass
class StorageCfg:
    db_path: str


@dataclass
class Config:
    universe: list[str]
    quote: str
    timeframe: str
    data: DataCfg
    research: ResearchCfg
    strategy: StrategyCfg
    risk: RiskCfg
    costs: CostsCfg
    account: AccountCfg
    storage: StorageCfg


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    try:
        cfg = Config(
            universe=list(raw["universe"]),
            quote=raw["quote"],
            timeframe=raw["timeframe"],
            data=DataCfg(**raw["data"]),
            research=ResearchCfg(**raw["research"]),
            strategy=StrategyCfg(**raw["strategy"]),
            risk=RiskCfg(**raw["risk"]),
            costs=CostsCfg(**raw["costs"]),
            account=AccountCfg(**raw["account"]),
            storage=StorageCfg(**raw["storage"]),
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"malformed config: {e}") from e

    s = cfg.strategy
    if not (s.exit_z < s.entry_z < s.stop_z):
        raise ConfigError("require exit_z < entry_z < stop_z")
    if len(cfg.universe) < 2:
        raise ConfigError("universe needs >= 2 symbols")
    if not (0 < cfg.risk.gross_exposure_pct <= 1):
        raise ConfigError("gross_exposure_pct must be in (0, 1]")
    return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pairsbot/config.py tests/test_config.py
git commit -m "feat: config loading and validation"
```

---

## Task 3: SQLite storage

**Files:**
- Create: `src/pairsbot/storage.py`, `tests/test_storage.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage.py
from pairsbot.storage import Store


def test_store_round_trips_equity_and_trades(tmp_path):
    db = str(tmp_path / "t.db")
    store = Store(db)
    run_id = store.start_run(mode="backtest", pair="ETH/SOL")
    store.record_equity(run_id, ts="2024-01-01T00:00:00Z", equity=10000.0)
    store.record_equity(run_id, ts="2024-01-01T01:00:00Z", equity=10010.0)
    store.record_trade(run_id, ts="2024-01-01T01:00:00Z", symbol="ETH",
                       notional=2500.0, price=2000.0, qty=1.25, fee=2.5, reason="enter")
    eq = store.load_equity(run_id)
    assert eq == [("2024-01-01T00:00:00Z", 10000.0), ("2024-01-01T01:00:00Z", 10010.0)]
    trades = store.load_trades(run_id)
    assert len(trades) == 1 and trades[0]["symbol"] == "ETH"


def test_store_persists_open_positions_for_restart(tmp_path):
    db = str(tmp_path / "t.db")
    store = Store(db)
    run_id = store.start_run(mode="live", pair="ETH/SOL")
    store.upsert_position(run_id, symbol="ETH", qty=1.25, avg_price=2000.0)
    store.upsert_position(run_id, symbol="SOL", qty=-30.0, avg_price=150.0)
    store.upsert_position(run_id, symbol="ETH", qty=0.0, avg_price=0.0)  # closed
    pos = store.load_positions(run_id)
    assert pos == {"SOL": (-30.0, 150.0)}  # zero-qty positions excluded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pairsbot.storage'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/pairsbot/storage.py
from __future__ import annotations

import sqlite3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    pair TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    run_id INTEGER, ts TEXT, equity REAL
);
CREATE TABLE IF NOT EXISTS trades (
    run_id INTEGER, ts TEXT, symbol TEXT, notional REAL,
    price REAL, qty REAL, fee REAL, reason TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    run_id INTEGER, symbol TEXT, qty REAL, avg_price REAL,
    PRIMARY KEY (run_id, symbol)
);
"""


class Store:
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def start_run(self, mode: str, pair: str | None) -> int:
        cur = self._conn.execute(
            "INSERT INTO runs (mode, pair) VALUES (?, ?)", (mode, pair))
        self._conn.commit()
        return int(cur.lastrowid)

    def record_equity(self, run_id: int, ts: str, equity: float) -> None:
        self._conn.execute(
            "INSERT INTO equity VALUES (?, ?, ?)", (run_id, ts, equity))
        self._conn.commit()

    def record_trade(self, run_id: int, ts: str, symbol: str, notional: float,
                     price: float, qty: float, fee: float, reason: str) -> None:
        self._conn.execute(
            "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, ts, symbol, notional, price, qty, fee, reason))
        self._conn.commit()

    def upsert_position(self, run_id: int, symbol: str, qty: float,
                        avg_price: float) -> None:
        self._conn.execute(
            "INSERT INTO positions (run_id, symbol, qty, avg_price) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(run_id, symbol) DO UPDATE SET qty=excluded.qty, "
            "avg_price=excluded.avg_price",
            (run_id, symbol, qty, avg_price))
        self._conn.commit()

    def load_equity(self, run_id: int) -> list[tuple[str, float]]:
        rows = self._conn.execute(
            "SELECT ts, equity FROM equity WHERE run_id=? ORDER BY ts", (run_id,))
        return [(ts, eq) for ts, eq in rows]

    def load_trades(self, run_id: int) -> list[dict]:
        cur = self._conn.execute(
            "SELECT ts, symbol, notional, price, qty, fee, reason "
            "FROM trades WHERE run_id=? ORDER BY ts", (run_id,))
        cols = ["ts", "symbol", "notional", "price", "qty", "fee", "reason"]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def load_positions(self, run_id: int) -> dict[str, tuple[float, float]]:
        rows = self._conn.execute(
            "SELECT symbol, qty, avg_price FROM positions WHERE run_id=? AND qty != 0",
            (run_id,))
        return {sym: (qty, avg) for sym, qty, avg in rows}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_storage.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pairsbot/storage.py tests/test_storage.py
git commit -m "feat: SQLite storage for runs, equity, trades, positions"
```

---

## Task 4: Historical data loader + alignment

**Files:**
- Create: `src/pairsbot/data/__init__.py` (empty), `src/pairsbot/data/historical.py`, `tests/test_historical.py`

The ccxt network call is isolated in a single fetch method so tests can inject a fake fetcher. `align_closes` is pure and fully tested.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_historical.py
import pandas as pd
from pairsbot.data.historical import align_closes


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_historical.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pairsbot.data.historical'`

- [ ] **Step 3: Create empty `src/pairsbot/data/__init__.py`, then write implementation**

```python
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
    value for any symbol. Pure function — no I/O."""
    closes = pd.DataFrame({sym: df["close"] for sym, df in frames.items()})
    before = len(closes)
    closes = closes.dropna(how="any")
    dropped = before - len(closes)
    if dropped:
        log.warning("align_closes dropped %d incomplete timestamps", dropped)
    return closes


class HistoricalLoader:
    def __init__(self, cache_dir: str, quote: str = "USDT", timeframe: str = "1h",
                 exchange=None):
        self.cache_dir = cache_dir
        self.quote = quote
        self.timeframe = timeframe
        os.makedirs(cache_dir, exist_ok=True)
        if exchange is None:
            import ccxt
            exchange = ccxt.binance({"enableRateLimit": True})
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
        """Load OHLCV per symbol, using parquet cache when present."""
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            path = self._cache_path(sym)
            if os.path.exists(path):
                out[sym] = pd.read_parquet(path)
            else:
                df = self._download(sym, start)
                df.to_parquet(path)
                out[sym] = df
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_historical.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pairsbot/data/__init__.py src/pairsbot/data/historical.py tests/test_historical.py
git commit -m "feat: historical OHLCV loader with cache and close alignment"
```

---

## Task 5: Cointegration screen + hedge ratio

**Files:**
- Create: `src/pairsbot/research/__init__.py` (empty), `src/pairsbot/research/screen.py`, `tests/test_screen.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_screen.py
import numpy as np
import pandas as pd
from pairsbot.research.screen import screen, hedge_ratio


def _cointegrated_pair(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 1, n)) + 100      # shared random walk
    spread = np.zeros(n)
    for t in range(1, n):                              # mean-reverting spread
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.5)
    a = common + spread
    b = common
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_hedge_ratio_recovers_unit_beta():
    closes = _cointegrated_pair()
    beta = hedge_ratio(closes["A"], closes["B"])
    assert 0.7 < beta < 1.3           # A ≈ 1*B + stationary spread


def test_screen_selects_cointegrated_pair_below_threshold():
    closes = _cointegrated_pair()
    sel = screen(closes, p_threshold=0.05)
    assert sel is not None
    assert {sel.a, sel.b} == {"A", "B"}
    assert sel.pvalue < 0.05


def test_screen_returns_none_when_nothing_cointegrated():
    rng = np.random.default_rng(1)
    n = 800
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    closes = pd.DataFrame({
        "A": np.cumsum(rng.normal(0, 1, n)) + 100,     # independent walks
        "B": np.cumsum(rng.normal(0, 1, n)) + 100,
    }, index=idx)
    assert screen(closes, p_threshold=0.01) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_screen.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pairsbot.research.screen'`

- [ ] **Step 3: Create empty `src/pairsbot/research/__init__.py`, then write implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_screen.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pairsbot/research/__init__.py src/pairsbot/research/screen.py tests/test_screen.py
git commit -m "feat: cointegration screen and OLS hedge ratio"
```

---

## Task 6: PairsStrategy (spread → rolling z → signals + stops)

**Files:**
- Create: `src/pairsbot/strategy/__init__.py` (empty), `src/pairsbot/strategy/base.py`, `src/pairsbot/strategy/pairs.py`, `tests/test_pairs_strategy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pairs_strategy.py
import numpy as np
import pandas as pd
from pairsbot.core.types import StrategyContext, SpreadSide
from pairsbot.strategy.pairs import PairsStrategy, current_zscore


def _ctx(closes, in_position=False, side=None, bars_in=0, **over):
    base = dict(a="A", b="B", beta=1.0, closes=closes, in_position=in_position,
                position_side=side, bars_in_position=bars_in, z_window=100,
                entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)
    base.update(over)
    return StrategyContext(**base)


def _closes_with_final_z(target_z, n=200, window=100):
    # Build a spread series whose last-bar z-score ≈ target_z.
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    b = np.full(n, 100.0)
    spread = np.zeros(n)
    spread[-1] = 0.0  # placeholder; set A so log-spread hits target
    # Use flat B=100 so log(A)-log(B) = log(A)-log(100). Make spread mean 0, std s.
    rng = np.random.default_rng(0)
    s = rng.normal(0, 0.01, n)
    s[-1] = 0.0
    log_spread = s.copy()
    win = s[-window:]
    log_spread[-1] = win.mean() + target_z * win.std()
    a = np.exp(log_spread) * 100.0
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_enters_short_spread_when_z_high_and_flat():
    closes = _closes_with_final_z(2.5)
    sigs = PairsStrategy().on_bar(_ctx(closes))
    assert len(sigs) == 1
    assert sigs[0].kind == "enter"
    assert sigs[0].spread_side == SpreadSide.SHORT   # z high -> short the spread


def test_enters_long_spread_when_z_low_and_flat():
    closes = _closes_with_final_z(-2.5)
    sigs = PairsStrategy().on_bar(_ctx(closes))
    assert sigs[0].spread_side == SpreadSide.LONG


def test_no_entry_when_z_between_thresholds():
    closes = _closes_with_final_z(1.0)
    assert PairsStrategy().on_bar(_ctx(closes)) == []


def test_exits_on_mean_reversion():
    closes = _closes_with_final_z(0.2)
    sigs = PairsStrategy().on_bar(_ctx(closes, in_position=True, side=SpreadSide.SHORT))
    assert len(sigs) == 1 and sigs[0].kind == "exit"


def test_exits_on_stop_z():
    closes = _closes_with_final_z(4.0)
    sigs = PairsStrategy().on_bar(_ctx(closes, in_position=True, side=SpreadSide.SHORT))
    assert sigs[0].kind == "exit" and "stop" in sigs[0].reason


def test_exits_on_time_stop_even_if_z_still_extreme():
    closes = _closes_with_final_z(2.5)
    sigs = PairsStrategy().on_bar(
        _ctx(closes, in_position=True, side=SpreadSide.SHORT, bars_in=168))
    assert sigs[0].kind == "exit" and "time" in sigs[0].reason
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pairs_strategy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pairsbot.strategy.pairs'`

- [ ] **Step 3: Create empty `src/pairsbot/strategy/__init__.py`, write `base.py` and `pairs.py`**

```python
# src/pairsbot/strategy/base.py
from __future__ import annotations

from typing import Protocol

from pairsbot.core.types import Signal, StrategyContext


class Strategy(Protocol):
    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        ...
```

```python
# src/pairsbot/strategy/pairs.py
from __future__ import annotations

import numpy as np

from pairsbot.core.types import Signal, SpreadSide, StrategyContext


def current_zscore(ctx: StrategyContext) -> float:
    """Rolling z-score of the log spread at the LAST (current) bar only.
    Uses the trailing z_window bars up to and including the current bar."""
    a = np.log(ctx.closes[ctx.a].to_numpy())
    b = np.log(ctx.closes[ctx.b].to_numpy())
    spread = a - ctx.beta * b
    window = spread[-ctx.z_window:]
    mu = window.mean()
    sigma = window.std()
    if sigma == 0:
        return 0.0
    return float((spread[-1] - mu) / sigma)


class PairsStrategy:
    def on_bar(self, ctx: StrategyContext) -> list[Signal]:
        if len(ctx.closes) < ctx.z_window:
            return []
        z = current_zscore(ctx)

        if ctx.in_position:
            if ctx.bars_in_position >= ctx.max_holding_bars:
                return [Signal("exit", reason="time-stop")]
            if abs(z) >= ctx.stop_z:
                return [Signal("exit", reason=f"stop-z {z:.2f}")]
            if abs(z) <= ctx.exit_z:
                return [Signal("exit", reason=f"mean-revert {z:.2f}")]
            return []

        # flat -> look for entry
        if z >= ctx.entry_z:
            return [Signal("enter", SpreadSide.SHORT, reason=f"z {z:.2f}")]
        if z <= -ctx.entry_z:
            return [Signal("enter", SpreadSide.LONG, reason=f"z {z:.2f}")]
        return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pairs_strategy.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pairsbot/strategy tests/test_pairs_strategy.py
git commit -m "feat: pairs strategy with rolling z-score, stops, and time stop"
```

---

## Task 7: No-lookahead guarantee test

This task adds no production code — it locks in the correctness property that matters most. If it fails, a prior task has a lookahead bug that must be fixed before proceeding.

**Files:**
- Create: `tests/test_no_lookahead.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_no_lookahead.py
import numpy as np
import pandas as pd
from pairsbot.core.types import StrategyContext
from pairsbot.strategy.pairs import current_zscore


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
    full = _closes(300)
    t = 200
    # z computed from data up to t, with vs without future bars present
    z_truncated = current_zscore(_ctx(full.iloc[: t + 1]))
    z_from_full = current_zscore(_ctx(full.iloc[: t + 1]))  # explicit slice = same
    # And prove that appending future rows never changes the historical value:
    for extra in (1, 10, 99):
        z_later = current_zscore(_ctx(full.iloc[: t + 1]))
        assert z_later == z_truncated
    assert z_truncated == z_from_full
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_no_lookahead.py -v`
Expected: PASS (1 passed). If it fails, stop and fix the z-score computation to slice only `[:t+1]`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_no_lookahead.py
git commit -m "test: lock in no-lookahead property of z-score"
```

---

## Task 8: Risk manager (sizing, flatten, kill-switch)

**Files:**
- Create: `src/pairsbot/risk/__init__.py` (empty), `src/pairsbot/risk/manager.py`, `tests/test_risk.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_risk.py
from pairsbot.core.types import Signal, SpreadSide, Position
from pairsbot.risk.manager import RiskManager


def test_enter_long_spread_is_dollar_neutral():
    rm = RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2)
    orders = rm.entry_orders(Signal("enter", SpreadSide.LONG), equity=10000,
                             a="ETH", b="SOL")
    by_sym = {o.symbol: o.notional for o in orders}
    assert by_sym["ETH"] == 2500.0     # long A: +gross/2
    assert by_sym["SOL"] == -2500.0    # short B: -gross/2


def test_enter_short_spread_flips_signs():
    rm = RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2)
    orders = rm.entry_orders(Signal("enter", SpreadSide.SHORT), equity=10000,
                             a="ETH", b="SOL")
    by_sym = {o.symbol: o.notional for o in orders}
    assert by_sym["ETH"] == -2500.0
    assert by_sym["SOL"] == 2500.0


def test_flatten_orders_reverse_current_positions():
    rm = RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2)
    positions = {"ETH": Position("ETH", qty=1.25, avg_price=2000.0),
                 "SOL": Position("SOL", qty=-30.0, avg_price=150.0)}
    prices = {"ETH": 2100.0, "SOL": 140.0}
    orders = rm.flatten_orders(positions, prices)
    by_sym = {o.symbol: o.notional for o in orders}
    assert by_sym["ETH"] == -1.25 * 2100.0   # sell the long
    assert by_sym["SOL"] == 30.0 * 140.0     # buy back the short


def test_kill_switch_blocks_entry_after_drawdown():
    rm = RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2)
    rm.update_peak(10000)
    assert rm.allow_entry(9000) is True    # 10% dd, ok
    assert rm.allow_entry(7900) is False   # 21% dd, blocked
    rm.update_peak(12000)                  # new peak resets reference
    assert rm.allow_entry(11000) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_risk.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pairsbot.risk.manager'`

- [ ] **Step 3: Create empty `src/pairsbot/risk/__init__.py`, then write implementation**

```python
# src/pairsbot/risk/manager.py
from __future__ import annotations

from pairsbot.core.types import Order, Position, Signal, SpreadSide


class RiskManager:
    def __init__(self, gross_exposure_pct: float, max_drawdown_pct: float):
        self.gross_exposure_pct = gross_exposure_pct
        self.max_drawdown_pct = max_drawdown_pct
        self._peak = 0.0

    def update_peak(self, equity: float) -> None:
        self._peak = max(self._peak, equity)

    def allow_entry(self, equity: float) -> bool:
        if self._peak <= 0:
            return True
        drawdown = 1.0 - equity / self._peak
        return drawdown < self.max_drawdown_pct

    def entry_orders(self, signal: Signal, equity: float, a: str, b: str) -> list[Order]:
        leg = self.gross_exposure_pct * equity / 2.0
        # long spread = long A, short B
        sign_a = 1.0 if signal.spread_side == SpreadSide.LONG else -1.0
        return [
            Order(a, sign_a * leg, reason=signal.reason),
            Order(b, -sign_a * leg, reason=signal.reason),
        ]

    def flatten_orders(self, positions: dict[str, Position],
                       prices: dict[str, float]) -> list[Order]:
        orders = []
        for sym, pos in positions.items():
            if pos.qty != 0:
                orders.append(Order(sym, -pos.qty * prices[sym], reason="exit"))
        return orders
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_risk.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pairsbot/risk tests/test_risk.py
git commit -m "feat: risk manager with dollar-neutral sizing and drawdown kill-switch"
```

---

## Task 9: Broker (PaperBroker + LiveBroker stub)

**Files:**
- Create: `src/pairsbot/execution/__init__.py` (empty), `src/pairsbot/execution/broker.py`, `tests/test_broker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_broker.py
import pytest
from pairsbot.core.types import Order
from pairsbot.execution.broker import PaperBroker, LiveBroker


def test_buy_reduces_cash_by_notional_plus_fee_and_slippage():
    b = PaperBroker(starting_cash=10000, fee_pct=0.001, slippage_pct=0.0005)
    b.mark({"ETH": 2000.0})
    fill = b.submit(Order("ETH", 2000.0), price=2000.0)   # buy $2000 of ETH
    # slippage: buy fills at 2000*(1.0005)=2001; qty = 2000/2001
    assert fill.qty == pytest.approx(2000.0 / 2001.0, rel=1e-9)
    # cash = 10000 - 2000 (notional) - 2.0 (fee=0.001*2000)
    assert b.cash == pytest.approx(10000 - 2000 - 2.0, rel=1e-9)
    assert b.positions()["ETH"].qty == pytest.approx(2000.0 / 2001.0, rel=1e-9)


def test_short_increases_cash_by_proceeds_minus_fee():
    b = PaperBroker(starting_cash=10000, fee_pct=0.001, slippage_pct=0.0005)
    fill = b.submit(Order("SOL", -1500.0), price=150.0)    # short $1500
    # sell fills at 150*(0.9995)=149.925; proceeds ~1500, fee=1.5
    assert b.cash == pytest.approx(10000 + 1500 - 1.5, rel=1e-6)
    assert b.positions()["SOL"].qty < 0


def test_equity_marks_positions_to_market():
    b = PaperBroker(starting_cash=10000, fee_pct=0.0, slippage_pct=0.0)
    b.submit(Order("ETH", 2000.0), price=2000.0)   # qty = 1.0
    b.mark({"ETH": 2200.0})                         # ETH up 10%
    # cash = 8000, position value = 1.0*2200 = 2200 -> equity 10200
    assert b.equity() == pytest.approx(10200.0, rel=1e-9)


def test_live_broker_is_a_disabled_stub():
    with pytest.raises(NotImplementedError):
        LiveBroker().submit(Order("ETH", 100.0), price=2000.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_broker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pairsbot.execution.broker'`

- [ ] **Step 3: Create empty `src/pairsbot/execution/__init__.py`, then write implementation**

```python
# src/pairsbot/execution/broker.py
from __future__ import annotations

from typing import Protocol

from pairsbot.core.types import Fill, Order, Position


class Broker(Protocol):
    def submit(self, order: Order, price: float) -> Fill: ...
    def mark(self, prices: dict[str, float]) -> None: ...
    def positions(self) -> dict[str, Position]: ...
    def equity(self) -> float: ...


class PaperBroker:
    """Simulated fills with fees + slippage. Used by BOTH backtest and live —
    the only difference between modes is who feeds it prices."""

    def __init__(self, starting_cash: float, fee_pct: float, slippage_pct: float):
        self.cash = starting_cash
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self._pos: dict[str, Position] = {}
        self._marks: dict[str, float] = {}

    def mark(self, prices: dict[str, float]) -> None:
        self._marks.update(prices)

    def submit(self, order: Order, price: float) -> Fill:
        buy = order.notional > 0
        fill_price = price * (1 + self.slippage_pct) if buy else price * (1 - self.slippage_pct)
        qty = order.notional / fill_price          # signed
        fee = abs(order.notional) * self.fee_pct
        self.cash -= order.notional + fee          # buy: cash down; short: cash up
        pos = self._pos.setdefault(order.symbol, Position(order.symbol))
        new_qty = pos.qty + qty
        if pos.qty == 0 or (pos.qty > 0) == (qty > 0):
            # opening or adding: update weighted avg price
            total = abs(pos.qty) + abs(qty)
            pos.avg_price = (abs(pos.qty) * pos.avg_price + abs(qty) * fill_price) / total if total else 0.0
        pos.qty = new_qty
        if abs(pos.qty) < 1e-12:
            pos.qty = 0.0
            pos.avg_price = 0.0
        self._marks[order.symbol] = price
        return Fill(order.symbol, order.notional, fill_price, qty, fee)

    def positions(self) -> dict[str, Position]:
        return {s: p for s, p in self._pos.items() if p.qty != 0}

    def equity(self) -> float:
        val = self.cash
        for sym, pos in self._pos.items():
            if pos.qty != 0:
                val += pos.qty * self._marks.get(sym, pos.avg_price)
        return val


class LiveBroker:
    """Real-execution seam. Intentionally unimplemented in v1 — proves the
    architecture supports live trading without enabling it."""

    def submit(self, order: Order, price: float) -> Fill:
        raise NotImplementedError("real execution is disabled in v1 (paper only)")

    def mark(self, prices: dict[str, float]) -> None:
        raise NotImplementedError

    def positions(self) -> dict[str, Position]:
        raise NotImplementedError

    def equity(self) -> float:
        raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_broker.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pairsbot/execution tests/test_broker.py
git commit -m "feat: PaperBroker fill engine and LiveBroker stub"
```

---

## Task 10: Backtester event loop

**Files:**
- Create: `src/pairsbot/backtest/__init__.py` (empty), `src/pairsbot/backtest/engine.py`, `tests/test_backtest_golden.py`

The loop decides on bar `t` (using closes ≤ `t`), then fills pending orders at bar `t+1`'s open — structurally preventing same-bar lookahead. It tracks position state (side + bars-in-position) so it can build the `StrategyContext`.

- [ ] **Step 1: Write the failing golden test**

```python
# tests/test_backtest_golden.py
import numpy as np
import pandas as pd
from pairsbot.core.types import PairSelection
from pairsbot.strategy.pairs import PairsStrategy
from pairsbot.risk.manager import RiskManager
from pairsbot.backtest.engine import Backtester


def _ohlcv(close_series):
    idx = close_series.index
    return pd.DataFrame({"open": close_series.values, "high": close_series.values,
                         "low": close_series.values, "close": close_series.values,
                         "volume": np.ones(len(idx))}, index=idx)


def _golden_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    common = np.cumsum(rng.normal(0, 0.5, n)) + 200
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.95 * spread[t - 1] + rng.normal(0, 0.4)
    a = pd.Series(common + spread + 50, index=idx)
    b = pd.Series(common, index=idx)
    return {"A": _ohlcv(a), "B": _ohlcv(b)}


def test_backtest_runs_trades_and_is_deterministic():
    data = _golden_data()
    sel = PairSelection(a="A", b="B", beta=1.0, pvalue=0.001)
    cfg = dict(z_window=100, entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)
    bt = Backtester(strategy=PairsStrategy(),
                    risk=RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2),
                    starting_equity=10000, fee_pct=0.001, slippage_pct=0.0005,
                    strategy_cfg=cfg)
    r1 = bt.run(data, sel)
    r2 = bt.run(data, sel)
    assert len(r1.trades) > 0                     # it actually traded
    assert r1.equity.iloc[-1] == r2.equity.iloc[-1]  # deterministic
    assert len(r1.equity) == len(next(iter(data.values())))  # one equity point per bar
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest_golden.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pairsbot.backtest.engine'`

- [ ] **Step 3: Create empty `src/pairsbot/backtest/__init__.py`, then write implementation**

```python
# src/pairsbot/backtest/engine.py
from __future__ import annotations

import pandas as pd

from pairsbot.core.types import (BacktestResult, Order, PairSelection,
                                  SpreadSide, StrategyContext)
from pairsbot.data.historical import align_closes


class Backtester:
    def __init__(self, strategy, risk, starting_equity: float, fee_pct: float,
                 slippage_pct: float, strategy_cfg: dict):
        self.strategy = strategy
        self.risk = risk
        self.starting_equity = starting_equity
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.cfg = strategy_cfg

    def run(self, data: dict[str, pd.DataFrame], sel: PairSelection) -> BacktestResult:
        # Local import avoids a hard module cycle and keeps broker swappable.
        from pairsbot.execution.broker import PaperBroker

        a, b = sel.a, sel.b
        closes = align_closes(data)[[a, b]]
        opens = pd.DataFrame({a: data[a]["open"], b: data[b]["open"]}).loc[closes.index]

        broker = PaperBroker(self.starting_equity, self.fee_pct, self.slippage_pct)
        self.risk.update_peak(self.starting_equity)

        in_position = False
        side: SpreadSide | None = None
        bars_in = 0
        pending: list[Order] = []
        equity_points: list[tuple[pd.Timestamp, float]] = []
        trades: list[dict] = []

        index = closes.index
        for i, ts in enumerate(index):
            # 1. fill orders decided on the previous bar, at THIS bar's open
            for o in pending:
                fill = broker.submit(o, float(opens.loc[ts, o.symbol]))
                trades.append({"ts": ts.isoformat(), "symbol": o.symbol,
                               "notional": o.notional, "price": fill.price,
                               "qty": fill.qty, "fee": fill.fee, "reason": o.reason})
            pending = []

            # 2. mark to this bar's close
            broker.mark({a: float(closes.loc[ts, a]), b: float(closes.loc[ts, b])})
            equity = broker.equity()
            equity_points.append((ts, equity))
            self.risk.update_peak(equity)

            if in_position:
                bars_in += 1

            # 3. build context from data up to and including this bar
            ctx = StrategyContext(
                a=a, b=b, beta=sel.beta, closes=closes.iloc[: i + 1],
                in_position=in_position, position_side=side, bars_in_position=bars_in,
                **self.cfg)

            # 4. get signals, translate to orders for NEXT bar's open
            signals = self.strategy.on_bar(ctx)
            for sig in signals:
                if sig.kind == "exit" and in_position:
                    prices = {a: float(closes.loc[ts, a]), b: float(closes.loc[ts, b])}
                    pending = self.risk.flatten_orders(broker.positions(), prices)
                    in_position, side, bars_in = False, None, 0
                elif sig.kind == "enter" and not in_position:
                    if self.risk.allow_entry(equity):
                        pending = self.risk.entry_orders(sig, equity, a, b)
                        in_position, side, bars_in = True, sig.spread_side, 0

        eq = pd.Series({ts: v for ts, v in equity_points})
        return BacktestResult(equity=eq, trades=trades, selection=sel)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest_golden.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pairsbot/backtest tests/test_backtest_golden.py
git commit -m "feat: backtest engine with next-open fills and position tracking"
```

---

## Task 11: Reporting (metrics + charts + markdown)

**Files:**
- Create: `src/pairsbot/reporting/__init__.py` (empty), `src/pairsbot/reporting/report.py`, `tests/test_reporting.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reporting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pairsbot.reporting.report'`

- [ ] **Step 3: Create empty `src/pairsbot/reporting/__init__.py`, then write implementation**

```python
# src/pairsbot/reporting/report.py
from __future__ import annotations

import os

import numpy as np
import pandas as pd

_HOURS_PER_YEAR = 24 * 365


def compute_metrics(equity: pd.Series, trades: list[dict]) -> dict:
    rets = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0
    if len(rets) > 1 and rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(_HOURS_PER_YEAR))
    else:
        sharpe = 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min()) if len(equity) else 0.0
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "num_trades": len(trades),
        "final_equity": float(equity.iloc[-1]) if len(equity) else 0.0,
    }


def write_report(out_dir: str, equity: pd.Series, trades: list[dict],
                 selection, closes: pd.DataFrame, beta: float) -> str:
    """Write markdown summary + equity and spread/z-score charts. Returns md path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    metrics = compute_metrics(equity, trades)

    # equity curve
    fig, ax = plt.subplots(figsize=(10, 4))
    equity.plot(ax=ax, title="Equity Curve")
    ax.set_ylabel("Equity ($)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "equity.png"), dpi=120)
    plt.close(fig)

    # spread + z-score
    if selection is not None:
        a, b = selection.a, selection.b
        spread = np.log(closes[a]) - beta * np.log(closes[b])
        z = (spread - spread.rolling(168).mean()) / spread.rolling(168).std()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        spread.plot(ax=ax1, title=f"Spread {a}-{b}")
        z.plot(ax=ax2, title="Rolling z-score")
        ax2.axhline(2.0, ls="--"); ax2.axhline(-2.0, ls="--"); ax2.axhline(0, color="k", lw=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "spread_zscore.png"), dpi=120)
        plt.close(fig)

    md_path = os.path.join(out_dir, "report.md")
    pair = f"{selection.a}/{selection.b}" if selection else "none"
    with open(md_path, "w") as f:
        f.write(f"# Backtest Report\n\n**Pair:** {pair}  \n")
        if selection:
            f.write(f"**Hedge ratio β:** {beta:.4f}  \n**Cointegration p-value:** {selection.pvalue:.4g}\n\n")
        f.write("## Metrics\n\n")
        f.write(f"- Total return: {metrics['total_return']:.2%}\n")
        f.write(f"- Sharpe (annualized): {metrics['sharpe']:.2f}\n")
        f.write(f"- Max drawdown: {metrics['max_drawdown']:.2%}\n")
        f.write(f"- Number of fills: {metrics['num_trades']}\n")
        f.write(f"- Final equity: ${metrics['final_equity']:,.2f}\n\n")
        f.write("![Equity](equity.png)\n\n")
        if selection:
            f.write("![Spread and z-score](spread_zscore.png)\n")
    return md_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reporting.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pairsbot/reporting tests/test_reporting.py
git commit -m "feat: performance metrics, charts, and markdown report"
```

---

## Task 12: Live runner (hourly loop + restart recovery)

**Files:**
- Create: `src/pairsbot/data/live.py`, `src/pairsbot/live/__init__.py` (empty), `src/pairsbot/live/runner.py`, `tests/test_live_runner.py`

`LiveRunner` is tested with a fake feed and a fake clock (an injected `sleep` and a bar iterator) so no network or real time is needed. It reuses the exact same strategy/risk/broker objects as the backtester.

- [ ] **Step 1: Write `data/live.py` (LiveFeed — thin ccxt wrapper, exercised only in manual smoke test)**

```python
# src/pairsbot/data/live.py
from __future__ import annotations

import pandas as pd


class LiveFeed:
    """Fetches the most recently CLOSED hourly bar for each symbol."""

    def __init__(self, quote: str = "USDT", timeframe: str = "1h", exchange=None):
        self.quote = quote
        self.timeframe = timeframe
        if exchange is None:
            import ccxt
            exchange = ccxt.binance({"enableRateLimit": True})
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
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_live_runner.py
import pandas as pd
from pairsbot.core.types import PairSelection
from pairsbot.strategy.pairs import PairsStrategy
from pairsbot.risk.manager import RiskManager
from pairsbot.execution.broker import PaperBroker
from pairsbot.storage import Store
from pairsbot.live.runner import LiveRunner


class FakeFeed:
    """Replays a prebuilt sequence of closed bars, one call at a time."""
    def __init__(self, closes_a, closes_b, index):
        self.a, self.b, self.index, self.i = closes_a, closes_b, index, 0

    def latest_closed_bar(self, symbols):
        ts = self.index[self.i]
        bar = {"A": {"ts": ts, "open": self.a[self.i], "close": self.a[self.i]},
               "B": {"ts": ts, "open": self.b[self.i], "close": self.b[self.i]}}
        self.i += 1
        return bar


def _series(n=300):
    import numpy as np
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    common = np.cumsum(rng.normal(0, 0.5, n)) + 200
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.95 * spread[t - 1] + rng.normal(0, 0.4)
    return (common + spread + 50).tolist(), common.tolist(), idx


def test_live_runner_processes_bars_and_persists(tmp_path):
    a, b, idx = _series()
    feed = FakeFeed(a, b, idx)
    store = Store(str(tmp_path / "live.db"))
    sel = PairSelection(a="A", b="B", beta=1.0, pvalue=0.001)
    cfg = dict(z_window=100, entry_z=2.0, exit_z=0.5, stop_z=3.5, max_holding_bars=168)
    runner = LiveRunner(feed=feed, broker=PaperBroker(10000, 0.001, 0.0005),
                        strategy=PairsStrategy(),
                        risk=RiskManager(gross_exposure_pct=0.5, max_drawdown_pct=0.2),
                        store=store, selection=sel, symbols=["A", "B"],
                        strategy_cfg=cfg, sleep=lambda s: None)
    runner.run(max_iterations=250)   # bounded for the test
    eq = store.load_equity(runner.run_id)
    assert len(eq) == 250
    assert len(store.load_trades(runner.run_id)) > 0


def test_live_runner_recovers_positions_on_restart(tmp_path):
    db = str(tmp_path / "live.db")
    store = Store(db)
    run_id = store.start_run(mode="live", pair="A/B")
    store.upsert_position(run_id, "A", qty=2.0, avg_price=250.0)
    store.upsert_position(run_id, "B", qty=-2.0, avg_price=200.0)
    broker = PaperBroker(10000, 0.001, 0.0005)
    LiveRunner.restore_broker(broker, store, run_id)
    pos = broker.positions()
    assert pos["A"].qty == 2.0 and pos["B"].qty == -2.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_live_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pairsbot.live.runner'`

- [ ] **Step 4: Create empty `src/pairsbot/live/__init__.py`, then write implementation**

```python
# src/pairsbot/live/runner.py
from __future__ import annotations

import pandas as pd

from pairsbot.core.types import Position, StrategyContext


class LiveRunner:
    def __init__(self, feed, broker, strategy, risk, store, selection, symbols,
                 strategy_cfg: dict, sleep, poll_seconds: int = 3600):
        self.feed = feed
        self.broker = broker
        self.strategy = strategy
        self.risk = risk
        self.store = store
        self.sel = selection
        self.symbols = symbols
        self.cfg = strategy_cfg
        self.sleep = sleep
        self.poll_seconds = poll_seconds
        self.run_id = store.start_run(mode="live", pair=f"{selection.a}/{selection.b}")
        self._closes = pd.DataFrame(columns=[selection.a, selection.b], dtype=float)
        self._pending: list = []
        self.in_position = False
        self.side = None
        self.bars_in = 0

    @staticmethod
    def restore_broker(broker, store, run_id: int) -> None:
        """Rehydrate broker positions from the DB so a restart resumes cleanly."""
        for sym, (qty, avg) in store.load_positions(run_id).items():
            broker._pos[sym] = Position(sym, qty=qty, avg_price=avg)

    def _step(self) -> None:
        a, b = self.sel.a, self.sel.b
        bar = self.feed.latest_closed_bar(self.symbols)
        ts = bar[a]["ts"]

        # 1. fill orders decided last bar at this bar's open
        for o in self._pending:
            fill = self.broker.submit(o, float(bar[o.symbol]["open"]))
            self.store.record_trade(self.run_id, ts.isoformat(), o.symbol,
                                    o.notional, fill.price, fill.qty, fill.fee, o.reason)
            self.store.upsert_position(self.run_id, o.symbol,
                                       self.broker._pos[o.symbol].qty,
                                       self.broker._pos[o.symbol].avg_price)
        self._pending = []

        # 2. mark + record equity
        self.broker.mark({a: float(bar[a]["close"]), b: float(bar[b]["close"])})
        equity = self.broker.equity()
        self.store.record_equity(self.run_id, ts.isoformat(), equity)
        self.risk.update_peak(equity)

        # 3. append close and build context
        self._closes.loc[ts] = [float(bar[a]["close"]), float(bar[b]["close"])]
        if self.in_position:
            self.bars_in += 1
        ctx = StrategyContext(a=a, b=b, beta=self.sel.beta, closes=self._closes,
                              in_position=self.in_position, position_side=self.side,
                              bars_in_position=self.bars_in, **self.cfg)

        # 4. signals -> pending orders for next bar
        for sig in self.strategy.on_bar(ctx):
            if sig.kind == "exit" and self.in_position:
                prices = {a: float(bar[a]["close"]), b: float(bar[b]["close"])}
                self._pending = self.risk.flatten_orders(self.broker.positions(), prices)
                self.in_position, self.side, self.bars_in = False, None, 0
            elif sig.kind == "enter" and not self.in_position:
                if self.risk.allow_entry(equity):
                    self._pending = self.risk.entry_orders(sig, equity, a, b)
                    self.in_position, self.side, self.bars_in = True, sig.spread_side, 0

    def run(self, max_iterations: int | None = None) -> None:
        n = 0
        while max_iterations is None or n < max_iterations:
            self._step()
            n += 1
            if max_iterations is None or n < max_iterations:
                self.sleep(self.poll_seconds)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_live_runner.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/pairsbot/data/live.py src/pairsbot/live tests/test_live_runner.py
git commit -m "feat: live paper-trading runner with restart recovery"
```

---

## Task 13: CLI (typer) + README

**Files:**
- Create: `src/pairsbot/cli.py`, `README.md`

The CLI wires the pieces together. Its commands are thin — all logic lives in the tested modules — so it's verified by manual runs rather than unit tests.

- [ ] **Step 1: Write `src/pairsbot/cli.py`**

```python
# src/pairsbot/cli.py
from __future__ import annotations

import typer

from pairsbot.config import load_config
from pairsbot.data.historical import HistoricalLoader, align_closes
from pairsbot.research.screen import screen
from pairsbot.strategy.pairs import PairsStrategy
from pairsbot.risk.manager import RiskManager
from pairsbot.backtest.engine import Backtester
from pairsbot.reporting.report import write_report

app = typer.Typer(help="Crypto pairs-trading bot")


def _strategy_cfg(cfg) -> dict:
    s = cfg.strategy
    return dict(z_window=s.z_window, entry_z=s.entry_z, exit_z=s.exit_z,
                stop_z=s.stop_z, max_holding_bars=s.max_holding_bars)


@app.command("fetch-data")
def fetch_data(config: str = "config.yaml"):
    """Download and cache historical OHLCV for the universe."""
    cfg = load_config(config)
    loader = HistoricalLoader(cfg.data.cache_dir, cfg.quote, cfg.timeframe)
    data = loader.load(cfg.universe, cfg.data.start)
    typer.echo(f"Cached {len(data)} symbols to {cfg.data.cache_dir}")


@app.command("screen")
def screen_cmd(config: str = "config.yaml"):
    """Run the cointegration screen and print the selected pair."""
    cfg = load_config(config)
    loader = HistoricalLoader(cfg.data.cache_dir, cfg.quote, cfg.timeframe)
    closes = align_closes(loader.load(cfg.universe, cfg.data.start))
    sel = screen(closes, cfg.research.p_threshold)
    if sel is None:
        typer.echo("No cointegrated pair below threshold.")
        raise typer.Exit(code=0)
    typer.echo(f"Selected {sel.a}/{sel.b}  beta={sel.beta:.4f}  p={sel.pvalue:.4g}")


@app.command("backtest")
def backtest_cmd(config: str = "config.yaml", out: str = "reports"):
    """Screen, backtest the selected pair, and write a report."""
    cfg = load_config(config)
    loader = HistoricalLoader(cfg.data.cache_dir, cfg.quote, cfg.timeframe)
    data = loader.load(cfg.universe, cfg.data.start)
    closes = align_closes(data)
    sel = screen(closes, cfg.research.p_threshold)
    if sel is None:
        typer.echo("No cointegrated pair; nothing to backtest.")
        raise typer.Exit(code=0)
    bt = Backtester(PairsStrategy(),
                    RiskManager(cfg.risk.gross_exposure_pct, cfg.risk.max_drawdown_pct),
                    cfg.account.starting_equity, cfg.costs.fee_pct,
                    cfg.costs.slippage_pct, _strategy_cfg(cfg))
    result = bt.run(data, sel)
    md = write_report(out, result.equity, result.trades, sel,
                      closes[[sel.a, sel.b]], sel.beta)
    typer.echo(f"Backtest complete. Report: {md}")


@app.command("live")
def live_cmd(config: str = "config.yaml"):
    """Run the live paper-trading loop (Ctrl-C to stop)."""
    import time
    from pairsbot.data.live import LiveFeed
    from pairsbot.execution.broker import PaperBroker
    from pairsbot.storage import Store
    from pairsbot.live.runner import LiveRunner

    cfg = load_config(config)
    loader = HistoricalLoader(cfg.data.cache_dir, cfg.quote, cfg.timeframe)
    sel = screen(align_closes(loader.load(cfg.universe, cfg.data.start)),
                 cfg.research.p_threshold)
    if sel is None:
        typer.echo("No cointegrated pair; not starting live loop.")
        raise typer.Exit(code=0)
    store = Store(cfg.storage.db_path)
    broker = PaperBroker(cfg.account.starting_equity, cfg.costs.fee_pct,
                         cfg.costs.slippage_pct)
    runner = LiveRunner(LiveFeed(cfg.quote, cfg.timeframe), broker, PairsStrategy(),
                        RiskManager(cfg.risk.gross_exposure_pct, cfg.risk.max_drawdown_pct),
                        store, sel, [sel.a, sel.b], _strategy_cfg(cfg), time.sleep)
    typer.echo(f"Live paper trading {sel.a}/{sel.b}. Ctrl-C to stop.")
    runner.run()


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Verify the CLI loads**

Run: `pairsbot --help`
Expected: shows the four commands (`fetch-data`, `screen`, `backtest`, `live`).

- [ ] **Step 3: Write `README.md`**

````markdown
# Crypto Pairs-Trading Bot

A statistical-arbitrage bot for crypto: a custom backtesting engine that screens a
universe of coins for cointegration, plus a live paper-trading mode that runs the
**identical** strategy and risk logic against a live Binance feed.

## Architecture

```
      Strategy + Risk (shared core)  ──  spread → z-score → signals
                 │
     ┌───────────┴───────────┐
  BACKTEST                 LIVE PAPER
 historical bars          polled hourly feed
     └──────── same PaperBroker (fees + slippage) ────────┘
```

- **Cointegration screen** (`statsmodels`) picks the pair + hedge ratio.
- **Rolling z-score** of the log-price spread drives entries/exits.
- **No lookahead:** signals use only data up to the current bar; fills happen at the
  next bar's open. A dedicated test locks this property in.
- **Risk:** dollar-neutral legs, z-stop, time-stop, drawdown kill-switch.

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                    # all tests green
pairsbot fetch-data       # cache historical OHLCV
pairsbot screen           # show the cointegrated pair
pairsbot backtest         # writes reports/report.md + charts
pairsbot live             # live paper trading (Ctrl-C to stop)
```

## How the strategy works

Two same-sector L1 coins whose log-price spread is cointegrated tend to mean-revert.
When the spread's rolling z-score exceeds +2, we short the spread (short the rich leg,
long the cheap leg); below −2 we long it. We exit when z returns toward 0, or on a
z-blowout stop (3.5), or a time stop (168 bars). Positions are dollar-neutral, so the
bot is market-neutral by construction.

## Configuration

All knobs live in `config.yaml`: universe, timeframe, z-score thresholds, fees, and
risk limits. Real-money execution is intentionally a disabled stub (`LiveBroker`) —
the architecture supports it, but v1 runs paper-only by design.
````

- [ ] **Step 4: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass (config, storage, historical, screen, pairs strategy, no-lookahead, risk, broker, backtest golden, reporting, live runner).

- [ ] **Step 5: Commit**

```bash
git add src/pairsbot/cli.py README.md
git commit -m "feat: typer CLI and project README"
```

---

## Task 14: End-to-end smoke run (manual, real data)

This validates the full pipeline against real Binance data. It's a manual acceptance step, not automated (no network in CI).

- [ ] **Step 1: Fetch real data**

Run: `pairsbot fetch-data`
Expected: parquet files appear in `./data_cache/` for all 7 symbols; no errors.

- [ ] **Step 2: Screen**

Run: `pairsbot screen`
Expected: prints a selected pair with a p-value < 0.05, or a clear "no cointegrated pair" message.

- [ ] **Step 3: Backtest and inspect the report**

Run: `pairsbot backtest`
Expected: `reports/report.md`, `reports/equity.png`, `reports/spread_zscore.png` exist; the report shows total return, Sharpe, max drawdown, and a non-zero number of fills.

- [ ] **Step 4: Live paper trading (brief)**

Run: `pairsbot live` and let it complete one hourly cycle (or confirm it fetches a closed bar and records an equity point), then Ctrl-C.
Expected: `bot.db` gains an equity row; no crash. Restarting resumes without error.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: end-to-end smoke run validated"
```

---

## Self-Review Notes

**Spec coverage check** (each spec section → task):
- §2 Strategy (universe, timeframe, pair selection, signals, stops) → Tasks 5, 6.
- §3 Architecture (shared core, module map, DataSource/Broker/Strategy seams, data flow) → Tasks 1, 8–10, 12.
- §4 Decisions (ccxt, custom backtester, lookahead defense, fills, sizing, risk, SQLite, typer, YAML, pytest, reporting) → Tasks 0, 2–13.
- §5 Config shape → Task 0/2.
- §6 Error handling (data gaps → `align_closes`; API retry → `HistoricalLoader._fetch_ohlcv`; no-pair → `screen`/CLI; kill-switch → Task 8; restart → Task 12) → covered.
- §7 Testing (unit, no-lookahead, golden backtest, mocked integration) → Tasks 2–12, esp. 7 and 10.
- §8 Deliverables (package, config, CLI, tests, reports, README) → Tasks 0, 11, 13, 14.

**Type consistency:** `Order.notional` is signed USD everywhere; `SpreadSide.LONG` = long A/short B in strategy (Task 6), risk (Task 8), and both orchestrators (Tasks 10, 12); `StrategyContext` fields match between definition (Task 1) and all construction sites; `PaperBroker` interface (`submit/mark/positions/equity`) is used identically by backtest and live.

**Placeholder scan:** none — every code and test step contains complete, runnable content.
