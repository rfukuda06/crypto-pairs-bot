# tests/test_cli_status.py
import pytest
import typer
from pairsbot.storage import Store
from pairsbot.cli import status_cmd


def _write_config(tmp_path, db_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "universe: [A, B]\nquote: USD\ntimeframe: 1h\n"
        "data:\n  start: '2024-01-01'\n  cache_dir: ./dc\n"
        "research:\n  train_window_days: 180\n  p_threshold: 0.05\n"
        "strategy:\n  z_window: 168\n  entry_z: 2.0\n  exit_z: 0.5\n  stop_z: 3.5\n  max_holding_bars: 168\n"
        "risk:\n  gross_exposure_pct: 0.5\n  max_drawdown_pct: 0.2\n"
        "costs:\n  fee_pct: 0.001\n  slippage_pct: 0.0005\n"
        "account:\n  starting_equity: 10000\n"
        f"storage:\n  db_path: {db_path}\n"
    )
    return str(p)


def test_status_shows_book_for_latest_run(tmp_path, capsys):
    db = str(tmp_path / "bot.db")
    store = Store(db)
    rid = store.start_run(mode="live", pair="A/B")
    store.record_equity(rid, "2024-01-01T00:00:00+00:00", 10000.0)
    store.record_equity(rid, "2024-01-01T01:00:00+00:00", 10120.0)
    store.upsert_position(rid, "A", 0.05, 60000.0)
    store.upsert_position(rid, "B", -1.0, 3000.0)
    status_cmd(config=_write_config(tmp_path, db))
    out = capsys.readouterr().out
    assert f"run #{rid} (live, A/B)" in out
    assert "equity $10,120" in out
    assert "Run summary" in out


def test_status_reports_no_runs_on_empty_db(tmp_path, capsys):
    cfg_path = _write_config(tmp_path, str(tmp_path / "empty.db"))
    with pytest.raises(typer.Exit):
        status_cmd(config=cfg_path)
    assert "no runs recorded" in capsys.readouterr().out
