# tests/test_storage.py
from pairsbot.storage import Store


def test_store_round_trips_equity_and_trades(tmp_path):
    db = str(tmp_path / "t.db")
    store = Store(db)
    run_id = store.start_run(mode="backtest", pair="ETH/SOL")
    store.record_equity(run_id, ts="2024-01-01T00:00:00Z", equity=10000.0)
    store.record_equity(run_id, ts="2024-01-01T01:00:00Z", equity=10010.0)
    store.record_trade(run_id, ts="2024-01-01T01:00:00Z", symbol="ETH",
                       notional=2500.0, price=2000.0, qty=1.25, fee=2.5, reason="enter")
    eq = store.load_equity(run_id)
    assert eq == [("2024-01-01T00:00:00Z", 10000.0), ("2024-01-01T01:00:00Z", 10010.0)]
    trades = store.load_trades(run_id)
    assert len(trades) == 1 and trades[0]["symbol"] == "ETH"


def test_store_persists_open_positions_for_restart(tmp_path):
    db = str(tmp_path / "t.db")
    store = Store(db)
    run_id = store.start_run(mode="live", pair="ETH/SOL")
    store.upsert_position(run_id, symbol="ETH", qty=1.25, avg_price=2000.0)
    store.upsert_position(run_id, symbol="SOL", qty=-30.0, avg_price=150.0)
    store.upsert_position(run_id, symbol="ETH", qty=0.0, avg_price=0.0)  # closed
    pos = store.load_positions(run_id)
    assert pos == {"SOL": (-30.0, 150.0)}  # zero-qty positions excluded


def test_get_run_returns_latest_and_specific(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    assert store.get_run() is None                      # empty DB
    r1 = store.start_run(mode="backtest", pair="A/B")
    r2 = store.start_run(mode="live", pair="C/D")
    assert store.get_run() == (r2, "live", "C/D")       # latest by default
    assert store.get_run(r1) == (r1, "backtest", "A/B") # specific id
    assert store.get_run(999) is None                   # unknown id
