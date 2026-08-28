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
