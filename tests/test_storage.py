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


def test_equity_is_deduped_by_run_and_ts(tmp_path):
    store = Store(str(tmp_path / "s.db"))
    rid = store.start_run("live", "A/B")
    store.record_equity(rid, "2024-01-01T00:00:00", 10000.0)
    store.record_equity(rid, "2024-01-01T00:00:00", 10500.0)  # same ts -> replace
    rows = store.load_equity(rid)
    assert rows == [("2024-01-01T00:00:00", 10500.0)]


def test_trades_dedupe_per_symbol_but_keep_both_legs(tmp_path):
    store = Store(str(tmp_path / "s.db"))
    rid = store.start_run("live", "A/B")
    store.record_trade(rid, "t0", "A", 100.0, 10.0, 10.0, 0.1, "z 2.0")
    store.record_trade(rid, "t0", "B", -100.0, 5.0, -20.0, 0.1, "z 2.0")
    store.record_trade(rid, "t0", "A", 999.0, 10.0, 10.0, 0.1, "z 2.0")  # replace leg A
    trades = store.load_trades(rid)
    assert len(trades) == 2                       # both legs, A replaced not duplicated
    assert {t["symbol"] for t in trades} == {"A", "B"}
    assert next(t for t in trades if t["symbol"] == "A")["notional"] == 999.0


def test_latest_live_run_returns_newest_matching_pair(tmp_path):
    store = Store(str(tmp_path / "s.db"))
    store.start_run("live", "A/B")
    store.start_run("backtest", "A/B")
    rid3 = store.start_run("live", "A/B")
    store.start_run("live", "C/D")
    assert store.latest_live_run("A/B") == rid3
    assert store.latest_live_run("X/Y") is None
