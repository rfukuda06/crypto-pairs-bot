# tests/test_cli_selection.py
import numpy as np
import pandas as pd

from pairsbot.cli import _frozen_selection
from pairsbot.config import load_config
from pairsbot.research.selection_store import load_selection


class _FakeLoader:
    """Returns two cointegrated OHLCV frames for symbols A,B."""
    def __init__(self, a, b, idx):
        self._a, self._b, self._idx = a, b, idx

    def load(self, symbols, start):
        out = {}
        for s, series in (("A", self._a), ("B", self._b)):
            df = pd.DataFrame({"open": series, "high": series, "low": series,
                               "close": series, "volume": 1.0}, index=self._idx)
            out[s] = df
        return out


def _cfg(tmp_path):
    cfg = load_config("config.yaml")
    cfg.universe = ["A", "B"]
    cfg.research.train_window_days = 5           # 5*24 = 120 in-sample bars
    cfg.research.selection_path = str(tmp_path / "sel.json")
    cfg.strategy.z_window = 20
    return cfg


def test_frozen_selection_screens_in_sample_and_persists(tmp_path):
    idx = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    common = np.cumsum(rng.normal(0, 0.5, 300)) + 200
    spread = np.zeros(300)
    for t in range(1, 300):
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.3)
    a = (common + spread + 50).tolist()
    b = common.tolist()
    cfg = _cfg(tmp_path)
    loader = _FakeLoader(a, b, idx)

    sel = _frozen_selection(cfg, loader)
    assert {sel.a, sel.b} == {"A", "B"}
    # persisted, and a second call loads the same object without re-screening
    assert load_selection(cfg.research.selection_path) is not None
    sel2 = _frozen_selection(cfg, loader)
    assert (sel2.a, sel2.b, sel2.beta) == (sel.a, sel.b, sel.beta)
