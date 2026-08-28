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
