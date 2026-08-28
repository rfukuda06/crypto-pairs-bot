# Crypto Pairs-Trading Bot

A statistical-arbitrage bot for crypto: a custom backtesting engine that screens a
universe of coins for cointegration, plus a live paper-trading mode that runs the
**identical** strategy and risk logic against a live exchange feed.

## Architecture

```
      Strategy + Risk (shared core)  ──  spread → z-score → signals
                 │
     ┌───────────┴───────────┐
  BACKTEST                 LIVE PAPER
 historical bars          polled hourly feed
     └──────── same PaperBroker (fees + slippage) ────────┘
```

- **Cointegration screen** (`statsmodels`) picks the pair + hedge ratio.
- **Rolling z-score** of the log-price spread drives entries/exits.
- **No lookahead:** signals use only data up to the current bar; fills happen at the
  next bar's open. Dedicated tests lock this in — including an engine-level check that a
  backtest's equity path is unchanged when future bars are truncated.
- **In-sample / out-of-sample:** the pair and hedge ratio are selected on the training
  window (`research.train_window_days`); the backtest then trades strictly the
  out-of-sample bars, so selection never sees the data it is evaluated on.
- **Risk:** dollar-neutral legs, z-stop, time-stop, drawdown kill-switch.

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                    # all tests green
pairsbot fetch-data       # cache historical OHLCV
pairsbot screen           # show the cointegrated pair
pairsbot backtest         # writes reports/report.md + charts
pairsbot optimize         # tune params in-sample, evaluate out-of-sample
pairsbot live             # live paper trading (Ctrl-C to stop)
```

## How the strategy works

Two same-sector L1 coins whose log-price spread is cointegrated tend to mean-revert.
When the spread's rolling z-score exceeds +2, we short the spread (short the rich leg,
long the cheap leg); below −2 we long it. We exit when z returns toward 0, or on a
z-blowout stop (3.5), or a time stop (168 bars). Positions are dollar-neutral, so the
bot is market-neutral by construction.

## Results & honesty

Pair selection and any parameter tuning happen strictly **in-sample**; the reported
numbers are a true out-of-sample holdout. On real Bitstamp data (11 majors, Jan 2024→),
the screen picks **LTC/XLM** (cointegration p≈0.02 on an 18-month training window), and
the naive strategy is **not** profitable out-of-sample:

| params            | in-sample           | out-of-sample (~14 months)   |
|-------------------|---------------------|------------------------------|
| default           | —                   | −17.8% return, Sharpe −2.15  |
| tuned in-sample   | +61.9%, Sharpe 1.68 | −16.4% return, Sharpe −1.39  |

The gap between a stellar in-sample fit and a losing out-of-sample result is the point:
it's a live demonstration of overfitting, and exactly why the in/out-of-sample split and
no-lookahead properties are enforced *structurally* rather than trusted. Improving real
performance would mean better pair/feature selection — not tuning harder against the past.

## Configuration

All knobs live in `config.yaml`: universe, exchange, timeframe, z-score thresholds,
fees, and risk limits. The data source is any [`ccxt`](https://github.com/ccxt/ccxt)
exchange — set `exchange:` (default `bitstamp`, chosen because it serves deep hourly
history and is reachable without a VPN; Binance is geo-blocked in some regions). A
market the exchange lists but can't actually serve is skipped automatically rather than
crashing the run.

Real-money execution is intentionally a disabled stub (`LiveBroker`) — the architecture
supports it, but v1 runs paper-only by design.
