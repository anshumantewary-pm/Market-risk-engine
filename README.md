# Market Risk Engine

Value-at-Risk, Expected Shortfall, backtesting, risk attribution and stress testing for a
multi-asset ETF portfolio, with an interactive Streamlit dashboard.

> **Status: in development.** The design is done ([PRD](docs/PRD.md), [architecture](docs/ARCHITECTURE.md));
> the code is being built step by step. Sections below marked *planned* describe what is being built,
> not what exists yet. No results have been published, and none will be until they come from a
> real-data run.

---

## What it does

A market-risk team needs to answer three questions about a portfolio every day:

1. **How much could we lose?** VaR and ES at 95% and 99%, 1-day and 10-day, from five methods.
2. **Can we trust that number?** Rolling out-of-sample backtests with Kupiec, Christoffersen and
   conditional-coverage tests, plus a Basel-style traffic light.
3. **Where does the risk come from, and what happens in a crisis?** Component VaR/ES by position and
   stress scenarios (historical replay and hypothetical shocks).

The portfolio is about $10M across 12 ETFs used as asset-class proxies:

| Asset class | ETFs |
|---|---|
| US equity | SPY, QQQ, IWM |
| International equity | EFA, EEM |
| Treasuries | TLT, IEF |
| Credit | LQD, HYG |
| Gold / oil | GLD, USO |
| Real estate | VNQ |

## Methods

Conventions: daily log returns; portfolio return `r_p = Σ wᵢ rᵢ`; P&L = notional × r_p;
**VaR and ES are positive loss numbers**; the forecast for day t uses data up to t−1 only.

| Method | VaR | ES |
|---|---|---|
| Historical simulation | empirical α-quantile of the last N losses | mean loss beyond VaR |
| Parametric normal | `z_α σ_p − μ_p`, with `σ_p = √(wᵀΣw)` | `σ_p φ(z_α)/(1−α) − μ_p` |
| Parametric Student-t | fitted ν, variance-matched scaling | closed-form t ES |
| EWMA filtered HS | RiskMetrics vol (λ = 0.94); returns rescaled to today's vol | mean rescaled loss beyond VaR |
| Monte Carlo | simulated losses from multivariate normal / t-copula | mean simulated loss beyond VaR |

**Backtesting:** Kupiec POF
`LR = −2 ln[(1−p)^{T−x} p^x] + 2 ln[(1−x/T)^{T−x} (x/T)^x] ~ χ²(1)`; Christoffersen independence
`~ χ²(1)`; conditional coverage `~ χ²(2)`; traffic light over 250 days at 99%
(green 0–4, yellow 5–9, red 10+ exceptions).

**Attribution:** marginal VaR `z_α (Σw)ᵢ / σ_p`; component VaR `wᵢ · MVaRᵢ` (sums to total);
component ES from historical tail days.

**Stress:** historical windows (2008 GFC, March 2020, 2022 rate shock) replayed on today's weights
from loaded data; hypothetical factor shocks from config, labelled illustrative.

10-day figures use √10 scaling, which assumes i.i.d. returns.

## Getting started *(planned)*

```bash
make install
```

```bash
make sample      # run the full pipeline on synthetic data
```

```bash
make dashboard   # streamlit run src/riskengine/dashboard/app.py
```

With real data (needs internet access to stooq.com):

```bash
python -m riskengine.download
```

```bash
python -m riskengine.pipeline
```

> Sample mode uses **synthetic** prices. The dashboard shows a banner and the run metadata is flagged
> whenever synthetic data is in use. Synthetic numbers are for demonstrating the engine only.

## Project structure *(planned)*

```
config/                 portfolio and stress scenario configs
src/riskengine/
  data/                 load, validate, returns, synthetic generator
  risk/                 VaR/ES estimators
  backtest/             rolling forecasts and statistical tests
  attribution/          marginal / component VaR and ES
  stress/               historical replay and hypothetical shocks
  pipeline.py           end-to-end run -> data/processed/
  dashboard/app.py      Streamlit app
tests/                  pytest suite
docs/                   PRD and architecture
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module responsibilities, data contracts and the
testing strategy.

## Results

*Not yet available.* This section will be filled in from a real-data run only, with the run date and
data range stated.

## Limitations

- Linear ETF positions only: no options, Greeks or non-linear payoffs.
- ETFs are proxies for asset classes, not real trading books.
- √10 scaling assumes i.i.d. returns; volatility clustering breaks it.
- Normal and t models are unconditional unless EWMA is used.
- Hypothetical stress shocks are illustrative.
- Real-data history begins when the youngest ETF in the universe started trading.

## Roadmap

1. Run on real data and publish results.
2. GARCH(1,1) volatility vs EWMA.
3. Stressed VaR (2008 calibration window).
4. Options book: delta-gamma VaR and full-revaluation Monte Carlo.
5. Liquidity-adjusted VaR.
6. Deploy on Streamlit Community Cloud with screenshots and a demo GIF.
7. FRTB-style ES at 97.5% with liquidity horizons.

## License

Licensed under the [Apache License 2.0](LICENSE).
