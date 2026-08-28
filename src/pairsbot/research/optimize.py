# src/pairsbot/research/optimize.py
"""In-sample strategy-parameter search.

Grid-searches the z-score thresholds and holding time by backtesting each
candidate ON THE GIVEN (in-sample) DATA ONLY, then ranks by an in-sample metric.
The caller evaluates the winning parameters out-of-sample. Keeping the search
strictly in-sample is what makes the out-of-sample result an honest estimate —
tuning against the test window would be lookahead.
"""
from __future__ import annotations

from itertools import product

from pairsbot.backtest.engine import Backtester
from pairsbot.reporting.report import compute_metrics
from pairsbot.risk.manager import RiskManager
from pairsbot.strategy.pairs import PairsStrategy

DEFAULT_GRID = {
    "entry_z": [1.5, 2.0, 2.5, 3.0],
    "exit_z": [0.0, 0.25, 0.5, 1.0],
    "stop_z": [3.5, 4.0, 4.5],
    "max_holding_bars": [72, 168, 336],
}


def optimize_params(data, sel, *, z_window, gross_exposure_pct, max_drawdown_pct,
                    starting_equity, fee_pct, slippage_pct,
                    grid=None, min_trades=5, objective="sharpe"):
    """Return (best_cfg, ranked) for the strategy params, judged in-sample.

    ``ranked`` is a list of ``{"cfg": <strategy cfg dict>, "metrics": <dict>}``
    sorted by ``objective`` descending. Candidates with fewer than ``min_trades``
    fills are considered only if nothing clears the bar (so the search never
    rewards a lucky one-trade config while trades exist elsewhere).
    """
    grid = grid or DEFAULT_GRID
    results = []
    for entry_z, exit_z, stop_z, max_hold in product(
            grid["entry_z"], grid["exit_z"], grid["stop_z"], grid["max_holding_bars"]):
        if not (exit_z < entry_z < stop_z):
            continue
        cfg = dict(z_window=z_window, entry_z=entry_z, exit_z=exit_z,
                   stop_z=stop_z, max_holding_bars=max_hold)
        bt = Backtester(PairsStrategy(),
                        RiskManager(gross_exposure_pct, max_drawdown_pct),
                        starting_equity, fee_pct, slippage_pct, cfg)
        res = bt.run(data, sel)
        results.append({"cfg": cfg, "metrics": compute_metrics(res.equity, res.trades)})

    tradeable = [r for r in results if r["metrics"]["num_trades"] >= min_trades]
    pool = tradeable or results
    ranked = sorted(pool, key=lambda r: r["metrics"][objective], reverse=True)
    return ranked[0]["cfg"], ranked
