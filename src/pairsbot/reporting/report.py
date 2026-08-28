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
