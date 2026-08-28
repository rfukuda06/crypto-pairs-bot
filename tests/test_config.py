# tests/test_config.py
import pytest
from pairsbot.config import load_config, ConfigError


def test_load_config_parses_nested_values(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "universe: [ETH, SOL]\nquote: USDT\ntimeframe: 1h\n"
        "data:\n  start: '2024-01-01'\n  cache_dir: ./dc\n"
        "research:\n  train_window_days: 180\n  p_threshold: 0.05\n"
        "strategy:\n  z_window: 168\n  entry_z: 2.0\n  exit_z: 0.5\n  stop_z: 3.5\n  max_holding_bars: 168\n"
        "risk:\n  gross_exposure_pct: 0.5\n  max_drawdown_pct: 0.2\n"
        "costs:\n  fee_pct: 0.001\n  slippage_pct: 0.0005\n"
        "account:\n  starting_equity: 10000\n"
        "storage:\n  db_path: ./bot.db\n"
    )
    cfg = load_config(str(p))
    assert cfg.universe == ["ETH", "SOL"]
    assert cfg.strategy.entry_z == 2.0
    assert cfg.costs.fee_pct == 0.001
    assert cfg.account.starting_equity == 10000


def test_load_config_rejects_bad_thresholds(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "universe: [ETH, SOL]\nquote: USDT\ntimeframe: 1h\n"
        "data:\n  start: '2024-01-01'\n  cache_dir: ./dc\n"
        "research:\n  train_window_days: 180\n  p_threshold: 0.05\n"
        "strategy:\n  z_window: 168\n  entry_z: 0.5\n  exit_z: 2.0\n  stop_z: 3.5\n  max_holding_bars: 168\n"
        "risk:\n  gross_exposure_pct: 0.5\n  max_drawdown_pct: 0.2\n"
        "costs:\n  fee_pct: 0.001\n  slippage_pct: 0.0005\n"
        "account:\n  starting_equity: 10000\n"
        "storage:\n  db_path: ./bot.db\n"
    )
    with pytest.raises(ConfigError):
        load_config(str(p))  # entry_z must be > exit_z


def test_load_config_exchange_defaults_and_overrides(tmp_path):
    base = (
        "universe: [ETH, SOL]\ntimeframe: 1h\n"
        "data:\n  start: '2024-01-01'\n  cache_dir: ./dc\n"
        "research:\n  train_window_days: 180\n  p_threshold: 0.05\n"
        "strategy:\n  z_window: 168\n  entry_z: 2.0\n  exit_z: 0.5\n  stop_z: 3.5\n  max_holding_bars: 168\n"
        "risk:\n  gross_exposure_pct: 0.5\n  max_drawdown_pct: 0.2\n"
        "costs:\n  fee_pct: 0.001\n  slippage_pct: 0.0005\n"
        "account:\n  starting_equity: 10000\n"
        "storage:\n  db_path: ./bot.db\n"
    )
    # No exchange key -> defaults to bitstamp.
    p1 = tmp_path / "default.yaml"
    p1.write_text("quote: USD\n" + base)
    assert load_config(str(p1)).exchange == "bitstamp"
    # Explicit exchange is honored.
    p2 = tmp_path / "override.yaml"
    p2.write_text("exchange: kraken\nquote: USD\n" + base)
    assert load_config(str(p2)).exchange == "kraken"
