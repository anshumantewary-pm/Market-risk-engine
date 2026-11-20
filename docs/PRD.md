# Product Requirements: Market Risk Engine v1

| | |
|---|---|
| Owner | Anshuman Tewary |
| Status | Draft, pre-implementation |
| Related | [ARCHITECTURE.md](ARCHITECTURE.md), [README](../README.md) |
| License | Apache 2.0 |

---

## 1. Problem

A market-risk team has to answer three questions every day about a portfolio:

1. **How much could we lose?** VaR and Expected Shortfall at standard confidence levels and horizons.
2. **Can we trust that number?** Out-of-sample backtests with formal tests, not eyeballing.
3. **Where does the risk come from, and what happens in a crisis?** Attribution to positions and
   stress scenarios.

Most public VaR projects stop at one method on one chart. This project builds the full loop
(estimate, backtest, attribute, stress, present) the way a bank risk team would recognise, on a
transparent multi-asset ETF portfolio, with the maths tested.

## 2. Goals

- **G1:** Estimate 1-day and 10-day VaR and ES at 95% and 99% with five methods on one portfolio,
  using consistent conventions.
- **G2:** Backtest every method out of sample with Kupiec, Christoffersen and conditional coverage
  tests and a Basel-style traffic light.
- **G3:** Show which positions drive risk (component VaR and ES) and how that compares with their weight.
- **G4:** Run historical-replay and hypothetical stress scenarios on today's portfolio.
- **G5:** Present everything in an interactive dashboard, including a live what-if on weights.
- **G6:** Be reproducible and trustworthy: one command for sample data, one for real data, tests and
  lint green in CI, synthetic results never presented as real.

## 3. Non-goals (v1)

- Options, Greeks or any non-linear instrument (roadmap).
- Real trading books, intraday data or live feeds.
- Regulatory capital calculation; the traffic light is shown as an industry-standard diagnostic only.
- Production concerns such as auth, multi-user state or a database.

## 4. Users

| User | Needs |
|---|---|
| Hiring manager / interviewer (primary) | Understand in five minutes what was built and judge whether the risk reasoning is sound; README and dashboard must make this quick. |
| Technical reviewer | Read the code and tests and confirm the formulas, conventions and no-look-ahead claim. |
| The owner | A codebase that's easy to extend with the roadmap items (GARCH, stressed VaR, options). |

## 5. Portfolio and data

- **Universe (ETF proxies):** SPY, QQQ, IWM (US equity); EFA, EEM (international equity); TLT, IEF
  (Treasuries); LQD, HYG (credit); GLD (gold); USO (oil); VNQ (real estate).
- **Default portfolio:** a balanced multi-asset mix, notional about $10M, defined in
  `config/portfolio.yaml`.
- **Real data:** daily adjusted closes from Stooq via `python -m riskengine.download`, run on the
  owner's machine.
- **Synthetic data:** `--sample` generates the same file format (2005–2024, correlated, regime-switching
  volatility, fat tails, stress episodes). Clearly labelled everywhere it is used.

## 6. Functional requirements

### 6.1 Data

| ID | Requirement | Priority |
|---|---|---|
| D1 | Download adjusted closes for the configured tickers from Stooq into `data/raw/prices.csv`. | Must |
| D2 | Generate a synthetic prices file in the same format with a fixed seed. | Must |
| D3 | Validate prices (positive, numeric, sorted unique dates, no unexplained gaps) and fail with a clear message. | Must |
| D4 | Restrict to the common date range across tickers and report it. | Must |

### 6.2 Risk estimation

| ID | Requirement | Priority |
|---|---|---|
| R1 | Historical simulation VaR and ES. | Must |
| R2 | Parametric normal VaR and ES. | Must |
| R3 | Parametric Student-t VaR and ES with fitted ν. | Must |
| R4 | EWMA (λ = 0.94) filtered historical simulation VaR and ES. | Must |
| R5 | Monte Carlo VaR and ES (multivariate normal; t-copula). | Must (normal), Should (t-copula) |
| R6 | Confidence levels 95% and 99%; horizons 1-day and 10-day (√10 scaling, caveat shown). | Must |
| R7 | All VaR/ES reported as positive losses in $ and % of notional. | Must |
| R8 | Rolling window length configurable (default 500 days). | Must |

### 6.3 Backtesting

| ID | Requirement | Priority |
|---|---|---|
| B1 | Rolling out-of-sample 1-day 99% VaR forecasts per method; the forecast for day t uses data to t−1 only. | Must |
| B2 | Exception series and counts. | Must |
| B3 | Kupiec POF test with p-value. | Must |
| B4 | Christoffersen independence test and conditional coverage with p-values. | Must |
| B5 | Basel traffic light over the last 250 days. | Must |

### 6.4 Attribution

| ID | Requirement | Priority |
|---|---|---|
| A1 | Marginal and component VaR per position; components sum to total parametric VaR. | Must |
| A2 | Component ES from historical tail scenarios. | Must |
| A3 | Risk share vs weight share per position. | Must |

### 6.5 Stress testing

| ID | Requirement | Priority |
|---|---|---|
| S1 | Replay named historical windows (2008 GFC, March 2020 COVID crash, 2022 rate shock) on today's portfolio, computed only from loaded data. | Must |
| S2 | Skip a window with a stated reason if the data doesn't cover it. | Must |
| S3 | Hypothetical factor shocks from a config file, labelled "illustrative". | Must |

### 6.6 Pipeline and dashboard

| ID | Requirement | Priority |
|---|---|---|
| P1 | `python -m riskengine.pipeline [--sample]` runs everything and writes outputs to `data/processed/`. | Must |
| P2 | `run_metadata.json` records data source, synthetic flag, date range, config, seed, version and commit. | Must |
| UI1 | Streamlit dashboard with tabs: Overview, Method comparison, Backtest, Attribution, Stress, What-if. | Must |
| UI2 | Synthetic-data banner whenever sample data is in use. | Must |
| UI3 | What-if tab recomputes VaR/ES and attribution live from edited weights. | Should |

## 7. Quality requirements

- Correctness: unit tests for VaR on known distributions, component VaR summing to the total, Kupiec
  on constructed exception series, no look-ahead, end-to-end pipeline on sample data, and a Streamlit
  `AppTest` smoke test.
- `ruff check .` clean; CI runs lint, tests and a sample pipeline run on every push.
- Sample pipeline runs in under about 2 minutes on a laptop.
- Reproducible: same config and seed give identical outputs.

## 8. Honesty and branding rules

- No real bank's name, logo or branding anywhere in the project.
- Synthetic numbers are never presented as real results. The README results section is filled only
  from a real-data run, with the run date and data range stated.
- Historical stress move sizes are computed from loaded data, never typed in as facts.
- Every limitation in §10 is stated in the README.

## 9. Success criteria

1. A reviewer can clone the repo, run `make install sample dashboard` and see the full dashboard in a
   few minutes.
2. CI is green.
3. The backtest table shows, for each method, an exception count, test p-values and a traffic light,
   and the README explains in plain language which methods pass and why (from real data).
4. Attribution shows at least one clear concentration insight (risk share well above weight share),
   computed, not asserted.

## 10. Known limitations (to state in the README)

- Linear ETF positions only; no options or non-linear payoffs.
- ETFs are proxies for asset classes, not real trading books.
- √10 scaling assumes i.i.d. returns; volatility clustering breaks it.
- Normal and t models are unconditional unless EWMA is used.
- Hypothetical stress shocks are illustrative.
- Real data history starts when the youngest ETF launched, which shortens the backtest.

## 11. Milestones

| Milestone | Scope | Done when |
|---|---|---|
| M0 Docs | PRD, architecture, README skeleton | Merged |
| M1 Core | Scaffolding, config, data, historical/normal/t | Tests green in CI |
| M2 Full risk | EWMA/FHS, Monte Carlo | Tests green |
| M3 Validation | Backtest + tests, attribution, stress | Tests green |
| M4 Product | Pipeline, dashboard, AppTest | `make sample dashboard` works |
| M5 Real data | Downloader, real run, README results | Results section from real data |

## 12. Roadmap (after v1)

1. GARCH(1,1) volatility; compare its backtest with EWMA.
2. Stressed VaR (2008 calibration window).
3. Options book: delta-gamma VaR and full-revaluation Monte Carlo.
4. Liquidity-adjusted VaR.
5. Deploy on Streamlit Community Cloud with screenshots and a demo GIF.
6. FRTB-style ES at 97.5% with liquidity horizons.

## 13. Open questions

- Is Stooq's close dividend-adjusted for every ETF in the universe? To check on the first real run
  (it matters most for bond ETFs).
- Exact default weights for the balanced portfolio (to be set in `config/portfolio.yaml` and
  justified in the README).
