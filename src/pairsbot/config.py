# src/pairsbot/config.py
from __future__ import annotations

from dataclasses import dataclass

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class DataCfg:
    start: str
    cache_dir: str


@dataclass
class ResearchCfg:
    train_window_days: int
    p_threshold: float
    selection_path: str = "./selection.json"


@dataclass
class StrategyCfg:
    z_window: int
    entry_z: float
    exit_z: float
    stop_z: float
    max_holding_bars: int


@dataclass
class RiskCfg:
    gross_exposure_pct: float
    max_drawdown_pct: float


@dataclass
class CostsCfg:
    fee_pct: float
    slippage_pct: float


@dataclass
class AccountCfg:
    starting_equity: float


@dataclass
class StorageCfg:
    db_path: str


@dataclass
class Config:
    universe: list[str]
    quote: str
    timeframe: str
    data: DataCfg
    research: ResearchCfg
    strategy: StrategyCfg
    risk: RiskCfg
    costs: CostsCfg
    account: AccountCfg
    storage: StorageCfg
    exchange: str = "bitstamp"


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    try:
        cfg = Config(
            universe=list(raw["universe"]),
            quote=raw["quote"],
            timeframe=raw["timeframe"],
            data=DataCfg(**raw["data"]),
            research=ResearchCfg(**raw["research"]),
            strategy=StrategyCfg(**raw["strategy"]),
            risk=RiskCfg(**raw["risk"]),
            costs=CostsCfg(**raw["costs"]),
            account=AccountCfg(**raw["account"]),
            storage=StorageCfg(**raw["storage"]),
            exchange=raw.get("exchange", "bitstamp"),
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"malformed config: {e}") from e

    s = cfg.strategy
    if not (s.exit_z < s.entry_z < s.stop_z):
        raise ConfigError("require exit_z < entry_z < stop_z")
    if len(cfg.universe) < 2:
        raise ConfigError("universe needs >= 2 symbols")
    if not (0 < cfg.risk.gross_exposure_pct <= 1):
        raise ConfigError("gross_exposure_pct must be in (0, 1]")
    return cfg
