# src/pairsbot/cli.py
from __future__ import annotations

import typer

from pairsbot.config import load_config
from pairsbot.data.historical import HistoricalLoader, align_closes
from pairsbot.research.screen import screen, num_candidate_pairs, sidak_pvalue
from pairsbot.research.split import split_closes
from pairsbot.research.optimize import optimize_params
from pairsbot.research.selection_store import load_selection, save_selection
from pairsbot.strategy.pairs import PairsStrategy
from pairsbot.risk.manager import RiskManager
from pairsbot.backtest.engine import Backtester
from pairsbot.reporting.report import write_report, compute_metrics

app = typer.Typer(help="Crypto pairs-trading bot")


def _strategy_cfg(cfg) -> dict:
    s = cfg.strategy
    return dict(z_window=s.z_window, entry_z=s.entry_z, exit_z=s.exit_z,
                stop_z=s.stop_z, max_holding_bars=s.max_holding_bars)


def _frozen_selection(cfg, loader, force: bool = False):
    """Return the frozen pair, screening the IN-SAMPLE window once and persisting
    it. Both backtest and live call this so they trade the identical pair."""
    path = cfg.research.selection_path
    if not force:
        sel = load_selection(path)
        if sel is not None:
            return sel
    closes = align_closes(loader.load(cfg.universe, cfg.data.start))
    in_sample, _ = split_closes(closes, cfg.research.train_window_days, cfg.timeframe)
    sel = screen(in_sample, cfg.research.p_threshold)
    if sel is not None:
        save_selection(path, sel)
    return sel


@app.command("fetch-data")
def fetch_data(config: str = "config.yaml"):
    """Download and cache historical OHLCV for the universe."""
    cfg = load_config(config)
    loader = HistoricalLoader(cfg.data.cache_dir, cfg.quote, cfg.timeframe, exchange_name=cfg.exchange)
    data = loader.load(cfg.universe, cfg.data.start)
    typer.echo(f"Cached {len(data)} symbols to {cfg.data.cache_dir}")


@app.command("screen")
def screen_cmd(config: str = "config.yaml"):
    """Screen the in-sample window, FREEZE the pair to disk, and print it."""
    cfg = load_config(config)
    loader = HistoricalLoader(cfg.data.cache_dir, cfg.quote, cfg.timeframe, exchange_name=cfg.exchange)
    sel = _frozen_selection(cfg, loader, force=True)
    if sel is None:
        typer.echo("No cointegrated pair below threshold.")
        raise typer.Exit(code=0)
    # Be as honest about the pair search as `optimize` is about the param grid:
    # the winning p-value is the MINIMUM over every pair, so disclose the search
    # size and the multiple-testing-corrected p alongside it.
    in_sample, _ = split_closes(align_closes(loader.load(cfg.universe, cfg.data.start)),
                                cfg.research.train_window_days, cfg.timeframe)
    n_pairs = num_candidate_pairs(in_sample)
    typer.echo(f"Selected {sel.a}/{sel.b}  beta={sel.beta:.4f}  p={sel.pvalue:.4g}  "
               f"(min over {n_pairs} pairs; Šidák-adjusted p≈{sidak_pvalue(sel.pvalue, n_pairs):.2f}, "
               f"not significant after correction)  (frozen to {cfg.research.selection_path})")


@app.command("backtest")
def backtest_cmd(config: str = "config.yaml", out: str = "reports"):
    """Select the pair in-sample, backtest it out-of-sample, and write a report."""
    cfg = load_config(config)
    loader = HistoricalLoader(cfg.data.cache_dir, cfg.quote, cfg.timeframe, exchange_name=cfg.exchange)
    data = loader.load(cfg.universe, cfg.data.start)
    closes = align_closes(data)
    in_sample, out_sample = split_closes(closes, cfg.research.train_window_days, cfg.timeframe)
    sel = _frozen_selection(cfg, loader)                # frozen, in-sample pick
    if sel is None:
        typer.echo("No cointegrated pair in-sample; nothing to backtest.")
        raise typer.Exit(code=0)
    if len(out_sample) <= cfg.strategy.z_window:
        typer.echo("Not enough out-of-sample data to backtest after the training window.")
        raise typer.Exit(code=0)
    # Trade strictly out-of-sample: slice the raw OHLCV to the OOS timestamps.
    oos_start = out_sample.index[0]
    oos_data = {sym: df.loc[df.index >= oos_start] for sym, df in data.items()}
    bt = Backtester(PairsStrategy(),
                    RiskManager(cfg.risk.gross_exposure_pct, cfg.risk.max_drawdown_pct),
                    cfg.account.starting_equity, cfg.costs.fee_pct,
                    cfg.costs.slippage_pct, _strategy_cfg(cfg))
    result = bt.run(oos_data, sel)
    md = write_report(out, result.equity, result.trades, sel,
                      out_sample[[sel.a, sel.b]], sel.beta)
    typer.echo(f"Backtest complete (in-sample {len(in_sample)} bars, "
               f"out-of-sample {len(out_sample)} bars). Report: {md}")


@app.command("optimize")
def optimize_cmd(config: str = "config.yaml", out: str = "reports"):
    """Tune strategy params IN-SAMPLE via grid search, then evaluate OUT-OF-SAMPLE.

    Selection and tuning use only the training window; the out-of-sample numbers
    are an honest holdout. Prints tuned-vs-default OOS metrics for comparison.
    """
    cfg = load_config(config)
    loader = HistoricalLoader(cfg.data.cache_dir, cfg.quote, cfg.timeframe, exchange_name=cfg.exchange)
    data = loader.load(cfg.universe, cfg.data.start)
    closes = align_closes(data)
    in_sample, out_sample = split_closes(closes, cfg.research.train_window_days, cfg.timeframe)
    sel = _frozen_selection(cfg, loader)
    if sel is None:
        typer.echo("No cointegrated pair in-sample; nothing to optimize.")
        raise typer.Exit(code=0)
    if len(out_sample) <= cfg.strategy.z_window:
        typer.echo("Not enough out-of-sample data to evaluate after the training window.")
        raise typer.Exit(code=0)

    # 1. Grid-search strategy params on the IN-SAMPLE window only.
    ins_end = in_sample.index[-1]
    ins_data = {sym: df.loc[df.index <= ins_end] for sym, df in data.items()}
    best_cfg, ranked = optimize_params(
        ins_data, sel, z_window=cfg.strategy.z_window,
        gross_exposure_pct=cfg.risk.gross_exposure_pct,
        max_drawdown_pct=cfg.risk.max_drawdown_pct,
        starting_equity=cfg.account.starting_equity,
        fee_pct=cfg.costs.fee_pct, slippage_pct=cfg.costs.slippage_pct)
    ins_m = ranked[0]["metrics"]
    typer.echo(f"Selected {sel.a}/{sel.b} (p={sel.pvalue:.4g}); searched {len(ranked)} param sets in-sample.")
    typer.echo(f"Best in-sample params: entry_z={best_cfg['entry_z']} exit_z={best_cfg['exit_z']} "
               f"stop_z={best_cfg['stop_z']} max_holding_bars={best_cfg['max_holding_bars']} "
               f"(in-sample Sharpe {ins_m['sharpe']:.2f}, return {ins_m['total_return']:.2%})")

    # 2. Evaluate tuned vs default params OUT-OF-SAMPLE (true holdout).
    oos_start = out_sample.index[0]
    oos_data = {sym: df.loc[df.index >= oos_start] for sym, df in data.items()}

    def _oos(strategy_cfg):
        bt = Backtester(PairsStrategy(),
                        RiskManager(cfg.risk.gross_exposure_pct, cfg.risk.max_drawdown_pct),
                        cfg.account.starting_equity, cfg.costs.fee_pct,
                        cfg.costs.slippage_pct, strategy_cfg)
        return bt.run(oos_data, sel)

    tuned, default = _oos(best_cfg), _oos(_strategy_cfg(cfg))
    mt, mdflt = compute_metrics(tuned.equity, tuned.trades), compute_metrics(default.equity, default.trades)
    typer.echo("Out-of-sample comparison:")
    typer.echo(f"  tuned:   return {mt['total_return']:+.2%}  Sharpe {mt['sharpe']:+.2f}  "
               f"maxDD {mt['max_drawdown']:.2%}  fills {mt['num_trades']}")
    typer.echo(f"  default: return {mdflt['total_return']:+.2%}  Sharpe {mdflt['sharpe']:+.2f}  "
               f"maxDD {mdflt['max_drawdown']:.2%}  fills {mdflt['num_trades']}")
    report = write_report(out, tuned.equity, tuned.trades, sel,
                          out_sample[[sel.a, sel.b]], sel.beta)
    typer.echo(f"Tuned out-of-sample report: {report}")


@app.command("live")
def live_cmd(config: str = "config.yaml"):
    """Run the live paper-trading loop (Ctrl-C to stop)."""
    import time
    from pairsbot.data.live import LiveFeed
    from pairsbot.execution.broker import PaperBroker
    from pairsbot.storage import Store
    from pairsbot.live.runner import LiveRunner

    cfg = load_config(config)
    loader = HistoricalLoader(cfg.data.cache_dir, cfg.quote, cfg.timeframe, exchange_name=cfg.exchange)
    sel = _frozen_selection(cfg, loader)
    if sel is None:
        typer.echo("No cointegrated pair; not starting live loop.")
        raise typer.Exit(code=0)
    store = Store(cfg.storage.db_path)
    broker = PaperBroker(cfg.account.starting_equity, cfg.costs.fee_pct,
                         cfg.costs.slippage_pct)
    pair = f"{sel.a}/{sel.b}"
    prior = store.latest_live_run(pair)
    if prior is not None:
        LiveRunner.restore_broker(broker, store, prior)
        typer.echo(f"Resuming live run #{prior} for {pair}")
    seed = align_closes(loader.load([sel.a, sel.b], cfg.data.start)).tail(cfg.strategy.z_window)
    runner = LiveRunner(LiveFeed(cfg.quote, cfg.timeframe, exchange_name=cfg.exchange), broker,
                        PairsStrategy(),
                        RiskManager(cfg.risk.gross_exposure_pct, cfg.risk.max_drawdown_pct),
                        store, sel, [sel.a, sel.b], _strategy_cfg(cfg), time.sleep,
                        starting_equity=cfg.account.starting_equity, initial_closes=seed,
                        run_id=prior)
    typer.echo(f"Live paper trading {sel.a}/{sel.b}. Ctrl-C to stop.")
    runner.run()


@app.command("status")
def status_cmd(config: str = "config.yaml", run_id: int | None = None):
    """Show the current book + risk/PnL for a run (defaults to the latest)."""
    from pairsbot.storage import Store
    from pairsbot.monitor import build_status_snapshot, format_status_block

    cfg = load_config(config)
    store = Store(cfg.storage.db_path)
    run = store.get_run(run_id)
    if run is None:
        typer.echo("no runs recorded")
        raise typer.Exit(code=0)
    rid, _mode, _pair = run
    snap = build_status_snapshot(
        run=run,
        positions=store.load_positions(rid),
        equity_rows=store.load_equity(rid),
        trades=store.load_trades(rid),
        starting_equity=cfg.account.starting_equity,
        max_dd_pct=cfg.risk.max_drawdown_pct)
    typer.echo(format_status_block(snap))


if __name__ == "__main__":
    app()
