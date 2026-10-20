# Shifted MSV on one SPX chain (2026-10-16 close)

Fit of plain MSV (a0, a1, a2, lam, k; beta=0) and shifted MSV (+ beta in [0, 3*spot])
jointly over the six monthly expiries 2026-11-20 .. 2027-09-17, on OTM mid prices
(puts converted to calls by parity), vega-weighted residuals, 5 multi-starts.
Data: `spx_chain_2026-10-19.csv` (never committed; licensing). Run:
`uv run python experiments/shifted_spx.py`.

**Corrected 2026-10-19:** the first version computed market implied vols from the
discounted mid with an undiscounted Black formula, biasing them low by up to 0.73 vol
points at the 11-month expiry. Numbers below are from the fixed pipeline; the first
version reported IV RMSE 4.77 → 4.27 and 3–23% skew recovery.

## Output

```
chain: ~/.cache/opencode-fanout/article/20261019-1248-paper-experiments/spx_chain_2026-10-19.csv  spot=7659.44  asof=2026-10-16
Per-expiry parity forward/discount and OTM fit set (|ln(K/F)|<=0.3):
expiry           tau          F        D  n_options
2026-11-20    0.0959    7717.10  1.00000        379
2026-12-18    0.1726    7735.69  0.99594        359
2027-01-15    0.2493    7762.95  0.99525        328
2027-03-19    0.4219    7811.62  0.98014        131
2027-06-17    0.6685    7892.03  0.96860        133
2027-09-17    0.9205    7977.02  0.95577        122
Fitted parameters (joint over six expiries):
  plain MSV : a0=0.00000 a1=6.66572 a2=0.01638 lam=0.0041 k=0.2862  cost=1.2908  [17.7s]
  shifted   : a0=0.00000 a1=0.00970 a2=0.00102 lam=0.2803 k=0.5894 beta=22978.27  cost=1.0987  [5.7s]
IV RMSE (vol points):
expiry          n      MSV  shifted
2026-11-20    379   10.167    8.214
2026-12-18    359    7.665    6.167
2027-01-15    328    6.324    5.006
2027-03-19    131    6.562    4.961
2027-06-17    133    5.173    3.747
2027-09-17    122    4.381    3.036
overall      1452    7.648    6.079
ATM skew (vol points per log-moneyness), |ln(K/F)|<=0.05:
expiry         market      MSV  shifted   MSV/mkt  shift/mkt
2026-11-20    -83.738    0.066   -4.498     -0.00       0.05
2026-12-18    -67.819   -0.072   -5.249      0.00       0.08
2027-01-15    -57.687   -0.109   -5.592      0.00       0.10
2027-03-19    -45.774    0.026   -5.459     -0.00       0.12
2027-06-17    -38.486    0.002   -5.976     -0.00       0.16
2027-09-17    -33.497    0.005   -6.312     -0.00       0.19
```

## Interpretation

The market skew is steep and decays with maturity (about −84 vol points per unit
log-moneyness at one month, −33 at eleven months). Plain MSV recovers none of it (ratios
≈ 0), as its lognormal-level mixture smile is symmetric in log-moneyness. The one-parameter
shift helps only marginally: beta runs to its 3·spot bound and the shifted model recovers
5–19% of the ATM skew, least at the short end where the market is steepest. This is
structural: a displaced lognormal quoted in unshifted Black has an ATM slope that saturates
near −σ_ATM/2 as beta grows (about −5 to −6 vol points per unit here, against a market −84).
Overall IV RMSE improves only from 7.65 to 6.08 vol points. Shifted MSV cannot be presented
as a skew model; the correlated random-level model (`crl_spx.md`) can. Caveats: single
Friday-close yfinance snapshot; near-dated parity regressions are noisy; the call wing
beyond +0.1 log-moneyness is filtered out by the 25% spread cap.
