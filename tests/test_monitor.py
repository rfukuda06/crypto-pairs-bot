# tests/test_monitor.py
from pairsbot.core.types import Position
from pairsbot.monitor import (
    build_live_snapshot,
    build_status_snapshot,
    format_live_line,
    format_status_block,
)


def test_build_live_snapshot_marks_and_signs():
    pos = {"BTC/USD": Position("BTC/USD", 0.05, 60000.0),
           "ETH/USD": Position("ETH/USD", -1.02, 3000.0)}
    marks = {"BTC/USD": 61240.0, "ETH/USD": 2980.0}
    snap = build_live_snapshot(positions=pos, marks=marks, equity=10342.0,
                               peak=10510.0, starting_equity=10000.0,
                               max_dd_pct=0.25, ts="2024-01-01T14:00:00+00:00")
    btc, eth = snap.legs
    assert btc.market_value == 0.05 * 61240.0
    assert eth.upnl == -1.02 * (2980.0 - 3000.0)   # short profits as price falls
    assert eth.upnl > 0
    assert snap.gross_abs == abs(0.05 * 61240.0) + abs(-1.02 * 2980.0)
    assert snap.net_abs == 0.05 * 61240.0 + (-1.02 * 2980.0)
    assert snap.exposure_basis == "mark"
    assert snap.pnl_abs == 342.0
    assert abs(snap.drawdown_pct - (1 - 10342.0 / 10510.0)) < 1e-12
    assert abs(snap.room_pct - (0.25 - (1 - 10342.0 / 10510.0))) < 1e-12
    assert snap.kill_equity == 10510.0 * 0.75


def test_build_live_snapshot_peak_guard_and_mark_fallback():
    pos = {"X/USD": Position("X/USD", 2.0, 50.0)}
    snap = build_live_snapshot(positions=pos, marks={}, equity=100.0, peak=0.0,
                               starting_equity=100.0, max_dd_pct=0.2,
                               ts="2024-01-01T00:00:00+00:00")
    assert snap.legs[0].mark == 50.0     # fell back to avg_price when no mark
    assert snap.drawdown_pct == 0.0      # peak <= 0 guard (matches allow_entry)
    assert snap.room_pct == 0.2
    assert snap.kill_equity == 0.0


def test_build_status_snapshot_db_only_fields():
    snap = build_status_snapshot(
        run=(7, "live", "A/B"),
        positions={"A": (0.05, 60000.0), "B": (-1.0, 3000.0)},
        equity_rows=[("2024-01-01T00:00:00+00:00", 10000.0),
                     ("2024-01-01T01:00:00+00:00", 10120.0)],
        trades=[{"symbol": "A"}, {"symbol": "B"}],
        starting_equity=10000.0, max_dd_pct=0.2)
    a, b = snap.legs
    assert a.mark is None and a.market_value is None and a.upnl is None
    assert a.cost_value == 0.05 * 60000.0
    assert snap.exposure_basis == "entry"
    assert snap.equity == 10120.0
    assert snap.peak == 10120.0          # recomputed from the equity series
    assert snap.drawdown_pct == 0.0      # last == peak
    assert snap.metrics is not None and snap.metrics["num_trades"] == 2


def test_build_status_snapshot_no_equity_rows():
    snap = build_status_snapshot(run=(1, "live", "A/B"), positions={},
                                 equity_rows=[], trades=[],
                                 starting_equity=10000.0, max_dd_pct=0.2)
    assert snap.equity is None and snap.peak is None and snap.metrics is None
    assert snap.pnl_abs is None and snap.drawdown_pct is None
    assert snap.gross_abs == 0.0 and snap.gross_pct is None


def test_format_live_line_golden():
    pos = {"BTC/USD": Position("BTC/USD", 0.05, 60000.0),
           "ETH/USD": Position("ETH/USD", -1.02, 3000.0)}
    marks = {"BTC/USD": 61240.0, "ETH/USD": 2980.0}
    snap = build_live_snapshot(positions=pos, marks=marks, equity=10342.0,
                               peak=10510.0, starting_equity=10000.0,
                               max_dd_pct=0.25, ts="2024-01-01T14:00:00+00:00")
    assert format_live_line(snap) == (
        "14:00Z  eq $10,342  PnL +$342 (+3.42%)  "
        "dd 1.60% (room 23.40% to 25% kill)  net +$22  "
        "gross $6,102 (59.0%)  pos: BTC +0.050 / ETH -1.020")


def test_format_live_line_flat_and_zero_start():
    snap = build_live_snapshot(positions={}, marks={}, equity=10000.0, peak=10000.0,
                               starting_equity=0.0, max_dd_pct=0.2,
                               ts="2024-01-01T09:00:00+00:00")
    line = format_live_line(snap)
    assert "pos: flat" in line
    assert "net $0" in line and "gross $0 (0.0%)" in line
    assert "PnL +$10,000  dd 0.00%" in line   # no percent when starting_equity <= 0


def test_format_status_block_tokens():
    snap = build_status_snapshot(
        run=(7, "live", "A/B"),
        positions={"A": (0.05, 60000.0), "B": (-1.0, 3000.0)},
        equity_rows=[("2024-01-01T00:00:00+00:00", 10000.0),
                     ("2024-01-01T15:00:00+00:00", 10120.0)],
        trades=[{"symbol": "A"}],
        starting_equity=10000.0, max_dd_pct=0.2)
    block = format_status_block(snap)
    assert "run #7 (live, A/B)" in block
    assert "as of 2024-01-01 15:00 UTC" in block
    assert "cost basis @ entry" in block            # honesty label
    assert "Exposure @ entry" in block
    assert "equity $10,120" in block
    assert "drawdown 0.00%" in block                # current dd (last == peak)
    assert "Run summary" in block and "max drawdown" in block   # both dd's labeled


def test_format_status_block_no_equity():
    snap = build_status_snapshot(run=(3, "live", "A/B"), positions={},
                                 equity_rows=[], trades=[],
                                 starting_equity=10000.0, max_dd_pct=0.2)
    block = format_status_block(snap)
    assert "no equity recorded yet" in block
    assert "(none)" in block
    assert "gross $0 (n/a)" in block                # abs shown, pct n/a
