# tests/test_broker.py
import pytest
from pairsbot.core.types import Order
from pairsbot.execution.broker import PaperBroker, LiveBroker


def test_buy_reduces_cash_by_notional_plus_fee_and_slippage():
    b = PaperBroker(starting_cash=10000, fee_pct=0.001, slippage_pct=0.0005)
    b.mark({"ETH": 2000.0})
    fill = b.submit(Order("ETH", 2000.0), price=2000.0)   # buy $2000 of ETH
    # slippage: buy fills at 2000*(1.0005)=2001; qty = 2000/2001
    assert fill.qty == pytest.approx(2000.0 / 2001.0, rel=1e-9)
    # cash = 10000 - 2000 (notional) - 2.0 (fee=0.001*2000)
    assert b.cash == pytest.approx(10000 - 2000 - 2.0, rel=1e-9)
    assert b.positions()["ETH"].qty == pytest.approx(2000.0 / 2001.0, rel=1e-9)


def test_short_increases_cash_by_proceeds_minus_fee():
    b = PaperBroker(starting_cash=10000, fee_pct=0.001, slippage_pct=0.0005)
    fill = b.submit(Order("SOL", -1500.0), price=150.0)    # short $1500
    # sell fills at 150*(0.9995)=149.925; proceeds ~1500, fee=1.5
    assert b.cash == pytest.approx(10000 + 1500 - 1.5, rel=1e-6)
    assert b.positions()["SOL"].qty < 0


def test_equity_marks_positions_to_market():
    b = PaperBroker(starting_cash=10000, fee_pct=0.0, slippage_pct=0.0)
    b.submit(Order("ETH", 2000.0), price=2000.0)   # qty = 1.0
    b.mark({"ETH": 2200.0})                         # ETH up 10%
    # cash = 8000, position value = 1.0*2200 = 2200 -> equity 10200
    assert b.equity() == pytest.approx(10200.0, rel=1e-9)


def test_live_broker_is_a_disabled_stub():
    with pytest.raises(NotImplementedError):
        LiveBroker().submit(Order("ETH", 100.0), price=2000.0)
