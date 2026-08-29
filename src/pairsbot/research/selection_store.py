# src/pairsbot/research/selection_store.py
from __future__ import annotations

import json
import os

from pairsbot.core.types import PairSelection


def save_selection(path: str, sel: PairSelection) -> None:
    """Persist the frozen pair so backtest and live trade the identical pair."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"a": sel.a, "b": sel.b, "beta": sel.beta, "pvalue": sel.pvalue}, f)


def load_selection(path: str) -> PairSelection | None:
    """Load the frozen pair, or None if absent/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        return PairSelection(a=d["a"], b=d["b"], beta=float(d["beta"]),
                             pvalue=float(d["pvalue"]))
    except (ValueError, KeyError, OSError):
        return None
