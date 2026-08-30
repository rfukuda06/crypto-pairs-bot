#!/usr/bin/env python3
"""Render docs/img/replay.gif — an animated replay of the REAL out-of-sample
backtest (the same run behind docs/img/equity.png). The z-score line unspools
hour by hour; a marker fires the instant the bot opens or closes a trade; the
equity curve builds underneath and bleeds from $10,000 down to $8,223.62.

Nothing is faked — every point comes from `Backtester.run` on the frozen pair
(LTC/XLM), identical to what `pairsbot backtest` produces.

Usage:  PYTHONPATH=src python scripts/make_replay_gif.py
Offline: reads cached parquet in data_cache/ + frozen selection.json (no network).
"""
from __future__ import annotations

import os
import sys

# Allow running as a plain script (src/ layout) without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from PIL import Image

from pairsbot.config import load_config
from pairsbot.data.historical import HistoricalLoader, align_closes
from pairsbot.research.split import split_closes
from pairsbot.strategy.pairs import PairsStrategy
from pairsbot.risk.manager import RiskManager
from pairsbot.backtest.engine import Backtester
from pairsbot.cli import _strategy_cfg, _frozen_selection

# ---- palette (GitHub dark) ---------------------------------------------------
BG      = "#0d1117"
FG      = "#e6edf3"
MUTED   = "#8b949e"
DIM     = "#6e7681"
GRID    = "#21262d"
BLUE    = "#58a6ff"
RED     = "#f85149"
GREEN   = "#3fb950"
ORANGE  = "#f0883e"

FRAMES     = 130      # time-lapse frames (all ~10k bars compressed into these)
FRAME_MS   = 95       # ms per frame while sweeping (higher = slower, calmer)
KILL_PAUSE_MS = 900   # linger on the moment the kill-switch benches the bot
HOLD_MS    = 1700     # hold the final frame before the loop restarts
GIF_COLORS = 32       # palette size per frame (keeps the file small)

LEGEND_HANDLES = None  # built once in main(); reused every frame

plt.rcParams.update({
    "font.family": "monospace",
    "text.color": FG, "axes.labelcolor": MUTED,
    "xtick.color": DIM, "ytick.color": DIM,
})


# ---- 1. reproduce the default out-of-sample backtest (== `pairsbot backtest`)-
def build_run():
    cfg = load_config("config.yaml")
    loader = HistoricalLoader(cfg.data.cache_dir, cfg.quote, cfg.timeframe,
                              exchange_name=cfg.exchange)
    data = loader.load(cfg.universe, cfg.data.start)
    closes_all = align_closes(data)
    _, out_sample = split_closes(closes_all, cfg.research.train_window_days, cfg.timeframe)
    sel = _frozen_selection(cfg, loader)                 # frozen in-sample pick (LTC/XLM)

    oos_start = out_sample.index[0]
    oos_data = {sym: df.loc[df.index >= oos_start] for sym, df in data.items()}
    bt = Backtester(PairsStrategy(),
                    RiskManager(cfg.risk.gross_exposure_pct, cfg.risk.max_drawdown_pct),
                    cfg.account.starting_equity, cfg.costs.fee_pct,
                    cfg.costs.slippage_pct, _strategy_cfg(cfg))
    result = bt.run(oos_data, sel)

    # The exact closes the backtest traded on (so markers land on the z-line).
    closes = align_closes(oos_data)[[sel.a, sel.b]]
    idx = closes.index
    zw = cfg.strategy.z_window
    spread = np.log(closes[sel.a]) - sel.beta * np.log(closes[sel.b])
    # ddof=0 to match strategy.current_zscore (numpy population std).
    z = ((spread - spread.rolling(zw).mean()) / spread.rolling(zw).std(ddof=0)).to_numpy()
    eq = result.equity.reindex(idx).to_numpy()
    return cfg, sel, idx, z, eq, result.trades


# ---- 2. turn the real fill log into entry/close events + a position timeline -
def build_events(trades, a_sym, idx):
    pos_of_ts = {ts: i for i, ts in enumerate(idx)}
    events = []                                          # (bar, kind) kind: short|long|close
    for t in trades:                                     # one A-leg fill == one event
        if t["symbol"] != a_sym:
            continue
        bi = pos_of_ts.get(pd.Timestamp(t["ts"]))
        if bi is None:
            continue
        if t["reason"] == "exit":
            kind = "close"
        else:                                            # entry: A-leg sign gives the side
            kind = "long" if t["notional"] > 0 else "short"
        events.append((bi, kind))
    events.sort()

    pos_state = ["FLAT"] * len(idx)
    cur, ptr = "FLAT", 0
    for i in range(len(idx)):
        while ptr < len(events) and events[ptr][0] == i:
            k = events[ptr][1]
            cur = "FLAT" if k == "close" else ("LONG" if k == "long" else "SHORT")
            ptr += 1
        pos_state[i] = cur
    return events, pos_state


# ---- 3. render the animation -------------------------------------------------
def style_ax(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(length=0, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6)


def main():
    global LEGEND_HANDLES
    LEGEND_HANDLES = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor=ORANGE,
               markeredgecolor=BG, markersize=7, label="open SHORT"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor=BLUE,
               markeredgecolor=BG, markersize=7, label="open LONG"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DIM,
               markeredgecolor=BG, markersize=5, label="close"),
    ]
    cfg, sel, idx, z, eq, trades = build_run()
    events, pos_state = build_events(trades, sel.a, idx)
    n = len(idx)
    start_eq = float(cfg.account.starting_equity)
    pair = f"{sel.a}/{sel.b}"

    final_eq = float(eq[-1])
    running_max = np.maximum.accumulate(eq)
    dd_series = eq / running_max - 1.0
    max_dd = float(dd_series.min())
    peak_bar = int(np.argmax(eq))
    peak_val = float(eq[peak_bar])
    fills = sum(1 for t in trades)
    print(f"run: {pair}  final ${final_eq:,.2f}  maxDD {max_dd:.2%}  fills {fills}  bars {n}")

    x = np.arange(n)
    ez, entries = z, events
    short_bars = [b for b, k in entries if k == "short"]
    long_bars  = [b for b, k in entries if k == "long"]
    close_bars = [b for b, k in entries if k == "close"]

    zlo, zhi = -4.3, 4.3
    eqlo, eqhi = eq.min() * 0.995, max(eq.max(), start_eq) * 1.01

    # All the trading happens early; once the −20% kill-switch trips, entries are
    # blocked and the book goes flat for the rest of the sample. Spend most frames
    # on the lively part, then fast-forward the flat tail to the kill-switch beat.
    last_trade_bar = max((b for b, _ in events), default=n // 4)
    active_end = min(n, last_trade_bar + int(0.03 * n))
    n_active = int(FRAMES * 0.75)
    cuts = np.unique(np.clip(np.concatenate([
        np.linspace(2, active_end, n_active, endpoint=False),
        np.linspace(active_end, n, FRAMES - n_active),
    ]).astype(int), 2, n))

    fig = plt.figure(figsize=(7.4, 4.4), dpi=100, facecolor=BG)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2], hspace=0.28,
                           left=0.085, right=0.965, top=0.86, bottom=0.07)
    ax_z = fig.add_subplot(gs[0])
    ax_e = fig.add_subplot(gs[1])

    # HUD text artists — created ONCE and mutated per frame (fig.text is additive,
    # so re-adding every frame would stack 110 copies into a smear).
    fig.suptitle(f"{pair}  ·  out-of-sample replay", x=0.085, y=0.96,
                 ha="left", fontsize=13, color=FG, weight="bold")
    hour_txt = fig.text(0.965, 0.96, "", ha="right", va="center",
                        fontsize=9, color=MUTED)
    fig.text(0.085, 0.905, "pos", fontsize=9, color=MUTED)
    pos_val = fig.text(0.128, 0.905, "", fontsize=9, weight="bold")
    dd_txt = fig.text(0.30, 0.905, "", fontsize=9)

    frames, durs = [], []
    kill_paused = False

    for c in cuts:
        ax_z.clear(); ax_e.clear()
        style_ax(ax_z); style_ax(ax_e)

        # --- z-score panel ---
        ax_z.set_xlim(0, n); ax_z.set_ylim(zlo, zhi)
        ax_z.set_ylabel("z-score", fontsize=8)
        ax_z.set_xticks([])
        for lvl, col, ls in ((2.0, RED, "--"), (-2.0, RED, "--"),
                             (3.5, DIM, ":"), (-3.5, DIM, ":"),
                             (0.5, GREEN, "--"), (-0.5, GREEN, "--")):
            ax_z.axhline(lvl, color=col, lw=0.8, ls=ls, alpha=0.55)
        ax_z.axhline(0, color=GRID, lw=0.8)
        zc = np.clip(ez[:c], zlo, zhi)
        ax_z.plot(x[:c], zc, color="#adbac7", lw=1.1)

        def _draw(bars, marker, color, size, label):
            b = [i for i in bars if i < c]
            if b:
                ax_z.scatter(b, np.clip(ez[b], zlo, zhi), marker=marker, s=size,
                             c=color, edgecolors=BG, linewidths=0.5, zorder=5, label=label)

        _draw(short_bars, "^", ORANGE, 55, "open SHORT")
        _draw(long_bars,  "v", BLUE,   55, "open LONG")
        _draw(close_bars, "o", DIM,    26, "close")
        ax_z.legend(handles=LEGEND_HANDLES, loc="lower left", fontsize=7,
                    framealpha=0.55, facecolor=BG, edgecolor="none",
                    labelcolor=MUTED, handletextpad=0.3, ncol=3,
                    columnspacing=1.0, borderaxespad=0.2)

        # --- equity panel ---
        ax_e.set_xlim(0, n); ax_e.set_ylim(eqlo, eqhi)
        ax_e.set_ylabel("equity ($)", fontsize=8)
        ax_e.set_xticks([])
        ax_e.axhline(start_eq, color=DIM, lw=0.8, ls="--", alpha=0.7)
        ec = eq[:c]
        ax_e.plot(x[:c], ec, color=BLUE, lw=1.3)
        ax_e.fill_between(x[:c], ec, start_eq, where=(ec < start_eq),
                          color=RED, alpha=0.14, interpolate=True)
        ax_e.scatter([c - 1], [ec[-1]], s=22, c=BLUE, edgecolors=BG,
                     linewidths=0.6, zorder=5)
        ax_e.annotate(f"${ec[-1]:,.0f}", (c - 1, ec[-1]),
                      textcoords="offset points", xytext=(-7, 8), ha="right",
                      fontsize=8, color=FG)

        # --- peak marker: makes the drawdown a visible drop, not just a number --
        if (c - 1) >= peak_bar:
            ax_e.scatter([peak_bar], [peak_val], s=22, facecolors="none",
                         edgecolors=GREEN, linewidths=1.1, zorder=6)
            ax_e.annotate(f"peak ${peak_val:,.0f}", (peak_bar, peak_val),
                          textcoords="offset points", xytext=(5, 5), ha="left",
                          fontsize=7.5, color=MUTED)

        # --- kill-switch beat: mark where the drawdown limit blocked entries ---
        if (c - 1) >= last_trade_bar:
            for ax in (ax_z, ax_e):
                ax.axvline(last_trade_bar, color=DIM, ls=":", lw=0.8, alpha=0.6)
            ax_e.text(last_trade_bar + n * 0.015, eqhi * 0.985,
                      "kill-switch: -20% drawdown -> entries blocked",
                      fontsize=7.5, color=MUTED, va="top")

        # --- HUD (mutate the artists created before the loop) ---
        pos = pos_state[c - 1]
        pos_col = {"SHORT": ORANGE, "LONG": BLUE, "FLAT": MUTED}[pos]
        dd_now = float(ec[-1] / ec.max() - 1.0)
        hour_txt.set_text(f"hour {c:,}/{n:,}")
        pos_val.set_text(pos); pos_val.set_color(pos_col)
        dd_txt.set_text(f"drawdown {dd_now:+.2%}")
        dd_txt.set_color(RED if dd_now < -0.001 else MUTED)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(Image.fromarray(buf).convert(
            "P", palette=Image.ADAPTIVE, colors=GIF_COLORS))
        if (c - 1) >= last_trade_bar and not kill_paused:
            durs.append(KILL_PAUSE_MS)          # beat on the kill-switch reveal
            kill_paused = True
        else:
            durs.append(FRAME_MS)

    durs[-1] = HOLD_MS
    plt.close(fig)

    out_path = os.path.join("docs", "img", "replay.gif")
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=durs, loop=0, optimize=True, disposal=2)
    kb = os.path.getsize(out_path) / 1024
    print(f"wrote {out_path}: {len(frames)} frames, {kb:.0f} KB, "
          f"~{sum(durs) / 1000:.1f}s loop")


if __name__ == "__main__":
    main()
