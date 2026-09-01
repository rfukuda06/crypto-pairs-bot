# tests/test_cli_selection.py
import numpy as np
import pandas as pd

import pairsbot.cli as cli_mod
from pairsbot.cli import _frozen_selection
from pairsbot.config import load_config
from pairsbot.data.historical import align_closes
from pairsbot.research.selection_store import load_selection
from pairsbot.research.split import split_closes


def _cointegrated_ab(n=300, seed=0):
    """Two cointegrated series (A = common + mean-reverting spread; B = common)."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 0.5, n)) + 200
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.3)
    return (common + spread + 50).tolist(), common.tolist()


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
    a, b = _cointegrated_ab(300, seed=0)
    cfg = _cfg(tmp_path)
    loader = _FakeLoader(a, b, idx)

    sel = _frozen_selection(cfg, loader)
    assert {sel.a, sel.b} == {"A", "B"}
    # persisted, and a second call loads the same object without re-screening
    assert load_selection(cfg.research.selection_path) is not None
    sel2 = _frozen_selection(cfg, loader)
    assert (sel2.a, sel2.b, sel2.beta) == (sel.a, sel.b, sel.beta)


def test_frozen_selection_screens_only_in_sample_bars(tmp_path, monkeypatch):
    # Enforces the "pair + beta chosen on the TRAINING window only" invariant that
    # test_split.py's partition test does not reach. A spy records how many bars
    # screen() actually receives; if _frozen_selection ever screened the full
    # series (a lookahead leak), that count would jump to the full length and this
    # test would fail.
    idx = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
    a, b = _cointegrated_ab(300, seed=0)
    cfg = _cfg(tmp_path)                          # universe A,B; 5 days -> 120 in-sample bars
    loader = _FakeLoader(a, b, idx)

    real_screen, seen = cli_mod.screen, {}
    def spy(closes, p_threshold):
        seen["rows"] = len(closes)
        return real_screen(closes, p_threshold)
    monkeypatch.setattr(cli_mod, "screen", spy)

    sel = _frozen_selection(cfg, loader)
    assert sel is not None

    closes_all = align_closes(loader.load(cfg.universe, cfg.data.start))
    in_sample, out_sample = split_closes(closes_all, cfg.research.train_window_days, cfg.timeframe)
    assert len(out_sample) > 0                    # there IS held-out data that could leak
    assert seen["rows"] == len(in_sample)         # screen saw exactly the training window
    assert seen["rows"] < len(closes_all)         # and never the full series
