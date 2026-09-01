# src/pairsbot/monitor.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from pairsbot.reporting.report import compute_metrics


@dataclass
class LegView:
    symbol: str
    qty: float
    avg_price: float
    mark: float | None
    market_value: float | None
    upnl: float | None
    cost_value: float          # qty * avg_price (signed notional at entry)


@dataclass
class Snapshot:
    ts: str | None             # ISO-8601 of the bar; None if no equity recorded
    legs: list[LegView]
    equity: float | None
    starting_equity: float
    pnl_abs: float | None
    pnl_pct: float | None
    peak: float | None
    drawdown_pct: float | None
    max_dd_pct: float
    room_pct: float | None
    kill_equity: float | None
    gross_abs: float
    net_abs: float
    gross_pct: float | None
    net_pct: float | None
    exposure_basis: str        # 'mark' (live) | 'entry' (status)
    run: tuple[int, str, str | None] | None   # (id, mode, pair); None for live
    metrics: dict | None       # compute_metrics() over the run; None for live


def _account_metrics(equity, starting_equity, peak, max_dd_pct):
    """Return (pnl_abs, pnl_pct, drawdown_pct, room_pct, kill_equity).

    All None when equity is None. Drawdown mirrors RiskManager.allow_entry
    exactly, including the peak <= 0 guard.
    """
    if equity is None:
        return None, None, None, None, None
    pnl_abs = equity - starting_equity
    pnl_pct = pnl_abs / starting_equity if starting_equity > 0 else None
    if peak is None or peak <= 0:
        drawdown_pct = 0.0
        kill_equity = 0.0
    else:
        drawdown_pct = 1.0 - equity / peak
        kill_equity = peak * (1.0 - max_dd_pct)
    room_pct = max_dd_pct - drawdown_pct
    return pnl_abs, pnl_pct, drawdown_pct, room_pct, kill_equity


def _exposure(values, equity):
    """Return (gross_abs, net_abs, gross_pct, net_pct). Percents are None when
    equity is None or 0 (no honest denominator)."""
    gross_abs = sum(abs(v) for v in values)
    net_abs = sum(values)
    if equity is None or equity == 0:
        return gross_abs, net_abs, None, None
    return gross_abs, net_abs, gross_abs / equity, net_abs / equity


def build_live_snapshot(*, positions, marks, equity, peak, starting_equity,
                        max_dd_pct, ts):
    """Live snapshot with true mark-to-market per-leg fields.

    positions: dict[str, Position] (e.g. broker.positions()).
    marks:     dict[str, float] current prices; falls back to avg_price like
               broker.equity() when a symbol has no mark.
    """
    legs = []
    for sym, pos in positions.items():
        if pos.qty == 0:
            continue
        mark = marks.get(sym, pos.avg_price)
        legs.append(LegView(symbol=sym, qty=pos.qty, avg_price=pos.avg_price,
                            mark=mark, market_value=pos.qty * mark,
                            upnl=pos.qty * (mark - pos.avg_price),
                            cost_value=pos.qty * pos.avg_price))
    gross_abs, net_abs, gross_pct, net_pct = _exposure(
        [leg.market_value for leg in legs], equity)
    pnl_abs, pnl_pct, dd, room, kill = _account_metrics(
        equity, starting_equity, peak, max_dd_pct)
    return Snapshot(ts=ts, legs=legs, equity=equity, starting_equity=starting_equity,
                    pnl_abs=pnl_abs, pnl_pct=pnl_pct, peak=peak, drawdown_pct=dd,
                    max_dd_pct=max_dd_pct, room_pct=room, kill_equity=kill,
                    gross_abs=gross_abs, net_abs=net_abs, gross_pct=gross_pct,
                    net_pct=net_pct, exposure_basis="mark", run=None, metrics=None)


def build_status_snapshot(*, run, positions, equity_rows, trades,
                          starting_equity, max_dd_pct):
    """DB-only snapshot: no live marks, so per-leg market_value/upnl are None
    and exposure is cost-basis ('entry'). equity/peak/metrics come from the
    recorded equity series (peak is NOT persisted, so it is recomputed).

    positions:   dict[str, tuple[float, float]] (Store.load_positions()).
    equity_rows: list[tuple[str, float]] (Store.load_equity()).
    trades:      list[dict] (Store.load_trades()).
    """
    legs = []
    for sym, (qty, avg) in positions.items():
        if qty == 0:
            continue
        legs.append(LegView(symbol=sym, qty=qty, avg_price=avg, mark=None,
                            market_value=None, upnl=None, cost_value=qty * avg))
    if equity_rows:
        ts = equity_rows[-1][0]
        equity = equity_rows[-1][1]
        equities = [eq for _, eq in equity_rows]
        peak = max(equities)
        metrics = compute_metrics(pd.Series(equities), trades)
    else:
        ts = equity = peak = metrics = None
    gross_abs, net_abs, gross_pct, net_pct = _exposure(
        [leg.cost_value for leg in legs], equity)
    pnl_abs, pnl_pct, dd, room, kill = _account_metrics(
        equity, starting_equity, peak, max_dd_pct)
    return Snapshot(ts=ts, legs=legs, equity=equity, starting_equity=starting_equity,
                    pnl_abs=pnl_abs, pnl_pct=pnl_pct, peak=peak, drawdown_pct=dd,
                    max_dd_pct=max_dd_pct, room_pct=room, kill_equity=kill,
                    gross_abs=gross_abs, net_abs=net_abs, gross_pct=gross_pct,
                    net_pct=net_pct, exposure_basis="entry", run=run, metrics=metrics)


def _money(x):
    return f"${x:,.0f}"


def _smoney(x):
    """Signed money with an explicit +/-; exact zero renders as '$0'."""
    if round(x) == 0:
        return "$0"
    sign = "+" if x > 0 else "-"
    return f"{sign}${abs(x):,.0f}"


def _pct2(x):
    return f"{x * 100:.2f}%"


def _pct1(x):
    return f"{x * 100:.1f}%"


def _spct2(x):
    return f"{x * 100:+.2f}%"


def _pctg(x):
    return f"{x * 100:g}%"


def _base(symbol):
    return symbol.split("/")[0]


def format_status_block(snap: Snapshot) -> str:
    rid, mode, pair = snap.run
    lines = []
    if snap.ts is not None:
        dt = datetime.fromisoformat(snap.ts)
        lines.append(f"── status · run #{rid} ({mode}, {pair}) "
                     f"· as of {dt:%Y-%m-%d %H:%M} UTC ──")
    else:
        lines.append(f"── status · run #{rid} ({mode}, {pair}) "
                     f"· no equity recorded yet ──")

    lines.append("Positions (cost basis @ entry, not marked)")
    if snap.legs:
        for leg in snap.legs:
            lines.append(f"  {leg.symbol:<9} {leg.qty:+.4f} @ {leg.avg_price:,.0f}"
                         f"   notional  {_smoney(leg.cost_value)}")
    else:
        lines.append("  (none)")

    gp = _pct1(snap.gross_pct) if snap.gross_pct is not None else "n/a"
    npct = _pct1(snap.net_pct) if snap.net_pct is not None else "n/a"
    lines.append(f"Exposure @ entry   gross {_money(snap.gross_abs)} ({gp})"
                 f"   net {_smoney(snap.net_abs)} ({npct})")

    if snap.equity is not None:
        pct = f" ({_spct2(snap.pnl_pct)})" if snap.pnl_pct is not None else ""
        lines.append(f"Account            equity {_money(snap.equity)}   "
                     f"PnL {_smoney(snap.pnl_abs)}{pct} vs "
                     f"{_money(snap.starting_equity)} start")
        lines.append(f"Risk               peak {_money(snap.peak)}   "
                     f"drawdown {_pct2(snap.drawdown_pct)}   "
                     f"room {_pct2(snap.room_pct)} to {_pctg(snap.max_dd_pct)} kill "
                     f"(kill equity {_money(snap.kill_equity)})")
    else:
        lines.append("Account            no equity recorded yet")

    if snap.metrics is not None:
        m = snap.metrics
        lines.append(f"Run summary        return {_spct2(m['total_return'])}   "
                     f"Sharpe {m['sharpe']:.2f}   "
                     f"max drawdown {_pct2(abs(m['max_drawdown']))}   "
                     f"trades {m['num_trades']}")
    return "\n".join(lines)


def format_live_line(snap: Snapshot) -> str:
    dt = datetime.fromisoformat(snap.ts)
    when = f"{dt:%H:%M}Z"
    pos = " / ".join(f"{_base(leg.symbol)} {leg.qty:+.3f}"
                     for leg in snap.legs) if snap.legs else "flat"
    pnl_pct = f" ({_spct2(snap.pnl_pct)})" if snap.pnl_pct is not None else ""
    gross_pct = _pct1(snap.gross_pct) if snap.gross_pct is not None else "n/a"
    return (f"{when}  eq {_money(snap.equity)}  "
            f"PnL {_smoney(snap.pnl_abs)}{pnl_pct}  "
            f"dd {_pct2(snap.drawdown_pct)} "
            f"(room {_pct2(snap.room_pct)} to {_pctg(snap.max_dd_pct)} kill)  "
            f"net {_smoney(snap.net_abs)}  "
            f"gross {_money(snap.gross_abs)} ({gross_pct})  "
            f"pos: {pos}")
