# Crypto Pairs-Trading Bot

**A statistical-arbitrage bot engineered to show its own strategy losing — honestly.** The same Strategy + Risk + PaperBroker core drives both the no-lookahead backtest and the live paper feed, so the numbers you see out-of-sample are the numbers live would trade.

```
      Strategy + Risk (shared core)  ──  spread → z-score → signals
                 │
     ┌───────────┴───────────┐
  BACKTEST                 LIVE PAPER
 historical bars          polled hourly feed
     └──────── same PaperBroker (fees + slippage) ────────┘
```

Python 3.13 · Typer CLI (`pairsbot`) · paper-only by design. This is a portfolio/engineering project: **returns are not a priority.** The value is structural correctness and honesty — and that one codebase trades the backtest and the live feed identically.

---

## The Honesty Demonstration

The headline result is a **loss**, and that is the entire point. Pair selection and every parameter tune happen strictly **in-sample**; the numbers below are a true out-of-sample holdout the strategy never saw.

On real Bitstamp hourly data (11 majors, from 2024-01-01), the screen freezes **LTC/XLM** (β = 0.2980, Engle-Granger cointegration p = 0.02166). In-sample window = 12,960 bars; out-of-sample = 10,269 bars.

| params            | in-sample                | out-of-sample                          |
|-------------------|--------------------------|----------------------------------------|
| default           | —                        | −17.76% return, Sharpe −2.15, 268 fills |
| tuned in-sample   | +61.89% return, Sharpe 1.68 | −16.37% return, Sharpe −1.39, 68 fills |

![Equity curve: $10k bleeding to ~$8.2k then flat](reports/equity.png)

The default backtest ends at **$8,223.62** from a $10,000 start (max drawdown −20.97%, 268 fills).

**Read this as a feature, not a bug.** A strategy that looks stellar in-sample (+61.89%, Sharpe 1.68) *loses* out-of-sample (−16.37%). We searched 144 parameter sets in-sample; the best of them barely beat the untuned default OOS. That is textbook overfitting — the grid fit noise, not signal — and it is only *visible* because selection and tuning never touch a single out-of-sample bar. Improving real performance would mean better pair/feature selection, not tuning harder against the past.

---

## Invariants & The Tests That Enforce Them

Every claim above rests on structural properties that are *enforced*, not trusted. 70 tests, all passing.

| Invariant | What it guarantees | Test |
|-----------|--------------------|------|
| **No-lookahead** | Bar `t` sees only `closes.iloc[:t+1]`; orders fill at the *next* bar's open. Truncating future bars leaves the equity path up to `t` identical (atol 1e-9). | `tests/test_no_lookahead.py` |
| **In / out-of-sample split** | Pair + β chosen on the training window only; exact non-overlapping partition with in-sample preceding out-of-sample. | `tests/test_split.py` |
| **Dollar-neutrality + kill-switch** | Entry legs net to zero; the drawdown kill-switch blocks entries at ≥ `max_drawdown_pct`. | `tests/test_risk.py` |
| **Backtest = live parity** | Identical bars replayed through `Backtester` and `LiveRunner` produce identical trades + equity. *(Caught a real regression during development.)* | `tests/test_live_backtest_parity.py` |
| **Determinism** | Same config twice → exactly equal final equity; > 0 trades; `len(equity) == len(input)`. | `tests/test_backtest_golden.py` |
| **Restart-resume** | Resumes the prior run id, restores positions/cash/state, keeps equity continuous, ignores post-flatten dust, never double-enters. | `tests/test_live_runner.py` |
| **Network resilience** | Retries a flaky feed then succeeds; raises a clean error on fewer than 2 bars. | `tests/test_live_feed.py` |

The no-lookahead proof is a *negative* test — it builds an alternate future that diverges from bar `t+1` on, and asserts the z-score at bar `t` is byte-identical across the two futures:

```python
# Alternate world: identical up to and including bar t, wildly different after.
alt = base.copy()
future = alt.index[t + 1:]
alt.loc[future, "A"] *= (2.0 + rng.normal(0, 1.0, len(future)))
alt.loc[future, "B"] *= (0.5 + rng.normal(0, 1.0, len(future)))
# The z-score at bar t is identical regardless of the (differing) future.
assert current_zscore(_ctx(alt.iloc[: t + 1])) == z_ref
```

If any decision peeked at the future, this assertion would fail.

---

## How It Works

**The strategy.** Two coins whose log-price spread is cointegrated tend to mean-revert. Define the spread as `spread = log(A) − β·log(B)`, then take a rolling z-score over `z_window = 168` bars:

- **Entry** when `z ≥ +2.0` (short the spread) or `z ≤ −2.0` (long the spread).
- **Exit** when `|z| ≤ exit_z = 0.5` (mean revert), or `|z| ≥ stop_z = 3.5` (blow-out stop), or held `≥ max_holding_bars = 168` (time stop).
- **Dollar-neutral** legs (gross/2 each side), so the book is market-neutral by construction.

Costs and limits: `gross_exposure_pct = 0.50`, `max_drawdown_pct = 0.20` (kill-switch), `fee_pct = 0.001`, `slippage_pct = 0.0005`, `starting_equity = $10,000`.

![Log spread LTC−XLM with rolling z-score and ±2 bands](reports/spread_zscore.png)

**The shared core.** One `PairsStrategy` + `RiskManager` + `PaperBroker` drives **both** paths identically. The contract is `StrategyContext → list[Signal] → Orders → Fills`. Backtest feeds a historical DataFrame; live feeds an hourly `ccxt` poll of the last *closed* bar. The only differences are the price source and the sink (in-memory report vs. SQLite + printed monitor).

Real-money execution (`LiveBroker`) is an **intentional disabled stub** that raises `NotImplementedError` — v1 is paper-only.

---

## A Real CLI Transcript

Screen freezes the pair, `live` resumes the prior paper run, and `status` reads the SQLite book:

```
$ pairsbot screen
Selected LTC/XLM  beta=0.2980  p=0.02166  (frozen to ./selection.json)

$ pairsbot live
Resuming live run #1 for LTC/XLM
06:00Z  eq $10,000  PnL $0 (+0.00%)  dd 0.00% (room 20.00% to 20% kill)  net $0  gross $0 (0.0%)  pos: flat

$ pairsbot status
── status · run #1 (live, LTC/XLM) · as of 2026-08-29 06:00 UTC ──
Positions (cost basis @ entry — not marked)
  (none)
Exposure @ entry   gross $0 (0.0%)   net $0 (0.0%)
Account            equity $10,000   PnL $0 (+0.00%) vs $10,000 start
Risk               peak $10,000   drawdown 0.00%   room 20.00% to 20% kill (kill equity $8,000)
Run summary        return +0.00%   Sharpe 0.00   max drawdown 0.00%   trades 0
```

---

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                    # 70 tests green
pairsbot fetch-data       # cache historical OHLCV
pairsbot screen           # screen in-sample & FREEZE the pair to selection.json
pairsbot backtest         # writes reports/report.md + charts
pairsbot optimize         # tune params in-sample, evaluate out-of-sample
pairsbot live             # live paper trading; resumes prior run (Ctrl-C to stop)
```

The pair is chosen **once** on the in-sample window and frozen to `selection.json`; `backtest`, `optimize`, and `live` all load that same frozen pair, so the backtest validates exactly what live trades. On startup `live` seeds its z-score window from the cache (so it can trade immediately) and, if a prior run for the pair exists, resumes it — restoring open positions, cash, and strategy state so the equity curve stays continuous across restarts.

## Configuration

All knobs live in `config.yaml`: universe, exchange, quote, timeframe, `research` (`train_window_days` = 540, `p_threshold` = 0.05, `selection_path`), strategy z-thresholds, risk limits, costs, and account. The data source is any [`ccxt`](https://github.com/ccxt/ccxt) exchange — set `exchange:` (default `bitstamp`, chosen because it serves deep hourly history and is reachable without a VPN). Historical OHLCV is cached as parquet in `data_cache/`.

**Tech stack:** pandas · numpy · statsmodels (Engle-Granger cointegration + OLS hedge ratio) · ccxt (Bitstamp) · typer · matplotlib · pyarrow · PyYAML · SQLite (stdlib). Entry point: `pairsbot = pairsbot.cli:app`, `src/` layout package `pairsbot`.
