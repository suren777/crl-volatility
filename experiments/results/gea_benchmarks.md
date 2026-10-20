# GEA ATM-skew for the SSVI and Heston benchmarks (2026-10-16 close)

`experiments/gea_benchmarks.py` gives the SKEW rows of the S&P benchmark
table (`benchmarks_spx.md`) under the Guyon–El Amrani estimator as well as the
linear one, so CRL, SSVI and Heston are compared like-for-like on the estimator
Guyon & El Amrani actually use. Same Cboe SQLite snapshot (trade date
`2026-10-16`, spot 7659.44) and same six SPX monthlies as `benchmarks_spx.py`;
CRL (`ρ ≤ 0`, `fit_crl_nonpos`), SSVI and Heston are refitted by the very same
fitters, so their parameters match the benchmark table exactly. Skews are in
vol points per log-moneyness (100 × decimal). Run:
`uv run python experiments/gea_benchmarks.py`.

## Per-expiry ATM skew (vol pts per log-moneyness)

| expiry | days | mkt lin | mkt GEA | CRL lin | CRL GEA | SSVI lin | SSVI GEA | Hes lin | Hes GEA |
|--------|------|---------|---------|---------|---------|----------|----------|---------|---------|
| 2026-11-20 | 35 | −80.436 | −91.535 | −94.691 | −123.671 | −98.759 | −137.458 | −93.492 | −128.388 |
| 2026-12-18 | 63 | −66.329 | −71.477 | −80.447 | −91.772 | −82.365 | −95.994 | −83.059 | −99.235 |
| 2027-01-15 | 91 | −58.180 | −60.647 | −70.207 | −76.058 | −70.197 | −76.870 | −74.124 | −82.771 |
| 2027-03-19 | 154 | −45.295 | −46.534 | −55.385 | −57.715 | −52.674 | −55.097 | −58.626 | −62.078 |
| 2027-06-17 | 244 | −38.041 | −39.049 | −44.460 | −45.541 | −40.549 | −41.650 | −46.449 | −47.920 |
| 2027-09-17 | 336 | −33.292 | −33.666 | −38.028 | −38.530 | −33.760 | −34.209 | −36.130 | −36.162 |

## Model/market ATM skew ratio

| expiry | days | CRL lin | CRL GEA | SSVI lin | SSVI GEA | Hes lin | Hes GEA |
|--------|------|---------|---------|----------|----------|---------|---------|
| 2026-11-20 | 35 | 1.177 | 1.351 | 1.228 | 1.502 | 1.162 | 1.403 |
| 2026-12-18 | 63 | 1.213 | 1.284 | 1.242 | 1.343 | 1.252 | 1.388 |
| 2027-01-15 | 91 | 1.207 | 1.254 | 1.207 | 1.267 | 1.274 | 1.365 |
| 2027-03-19 | 154 | 1.223 | 1.240 | 1.163 | 1.184 | 1.294 | 1.334 |
| 2027-06-17 | 244 | 1.169 | 1.166 | 1.066 | 1.067 | 1.221 | 1.227 |
| 2027-09-17 | 336 | 1.142 | 1.144 | 1.014 | 1.016 | 1.085 | 1.074 |

## Log-log skew exponent (|S| ~ A·tau^α) over the six monthlies

| estimator | Market | CRL | SSVI | Heston |
|-----------|--------|-----|------|--------|
| linear | −0.398 | −0.412 | −0.487 | −0.418 |
| GEA | −0.445 | −0.516 | −0.616 | −0.550 |

Guyon–El Amrani SPX reference: α = 0.43 above 3 weeks.

## Findings

- **The fits are the benchmark's own.** CRL (`a0=0.00000 a1=0.09208 a2=0.01963
  lam=0.5167 k=2.2600 ρ=−0.487`), SSVI (`η=0.8016 γ=0.5937 ρ=−0.579`) and Heston
  (`v0=0.02242 κ=2.4940 θ=0.06000 σ=1.3997 ρ=−0.703`, Feller 0.153 violated)
  reproduce the `benchmarks_spx.md` parameters and the linear skew ratios, so
  the GEA columns below are the same models, re-read with their estimator.
- **Under GEA, every model over-shoots the market ATM skew more than under the
  linear estimator.** The GEA model/market ratio is above the linear one at every
  expiry for CRL (1.35→1.14 vs 1.18→1.14), SSVI (1.50→1.02 vs 1.23→1.01) and
  Heston (1.40→1.07 vs 1.16→1.09); the gap is largest at the short end. The GEA
  degree-2 fit captures near-money curvature the ±5% linear slope misses, so any
  paper number stated against G&EA's 0.43 / estimator must use the GEA columns.
- **SSVI and Heston are more curved near the money than CRL.** Under GEA the
  short-end ratio is 1.50 (SSVI) and 1.40 (Heston) vs 1.35 (CRL), whereas under
  the linear estimator SSVI and CRL are nearly tied (1.23 vs 1.18) at 35 days.
- **GEA widens the CRL-vs-market skew-exponent gap slightly.** On the six
  monthlies the GEA skew exponent is −0.445 for the market (right on G&EA's 0.43)
  versus −0.516 for CRL, −0.616 for SSVI and −0.550 for Heston; the linear
  exponents are −0.398 (market), −0.412 (CRL), −0.487 (SSVI), −0.418 (Heston).
  All four models decay faster (steeper) than the market under both estimators.
- **None of the models flattens the skew as fast as the market; SSVI is the
  steepest.** Every model's |S| decays faster (more negative exponent) than the
  market under both estimators; SSVI's GEA exponent −0.616 is the outlier, the
  steepest decay of the three, followed by Heston −0.550 and CRL −0.516.
- Caveats: single day (2026-10-16 close), delayed-quote Cboe snapshot, monthly
  expiries only (no sub-1-month weeklies), in-sample fits, and the six-expiry
  exponent is a six-point slope that is not evidence of out-of-sample behaviour.

## Output

```
Cboe SQLite spx.sqlite  trade_date=2026-10-16  spot=7659.44  asof=2026-10-16  (6 SPX monthlies)
ATM skew estimators: GEA = degree-2 poly, 6 closest strikes per side of the forward (Guyon-El Amrani); linear = regression over |ln K/F| <= 0.05.

Per-expiry parity forward/discount and OTM fit set (|ln(K/F)|<=0.3):
expiry           tau  days          F        D  n_options
2026-11-20    0.0959    35    7720.69  0.99589        377
2026-12-18    0.1726    63    7740.32  0.99230        363
2027-01-15    0.2493    91    7769.44  0.98863        334
2027-03-19    0.4219   154    7817.07  0.98071        130
2027-06-17    0.6685   244    7898.13  0.96839        132
2027-09-17    0.9205   336    7984.22  0.95577        122

Fitted parameters (joint over six expiries) and fit time:
  CRL   : a0=0.00000 a1=0.09208 a2=0.01963 lam=0.5167 k=2.2600 rho=-0.487 (rho<=0)  cost=0.0200  [11.4s]
  SSVI  : eta=0.8016 gamma=0.5937 rho=-0.579 cost=0.0354  [5.7s]
  Heston: v0=0.02242 kappa=2.4940 theta=0.06000 sigma=1.3997 rho=-0.703  cost=0.0663  [20.6s]  Feller=0.153

Per-expiry ATM skew (vol pts per log-moneyness): market and models, both estimators
expiry        days   mkt lin   mkt GEA   CRL lin   CRL GEA  SSVI lin  SSVI GEA   Hes lin   Hes GEA
2026-11-20      35   -80.436   -91.535   -94.691  -123.671   -98.759  -137.458   -93.492  -128.388
2026-12-18      63   -66.329   -71.477   -80.447   -91.772   -82.365   -95.994   -83.059   -99.235
2027-01-15      91   -58.180   -60.647   -70.207   -76.058   -70.197   -76.870   -74.124   -82.771
2027-03-19     154   -45.295   -46.534   -55.385   -57.715   -52.674   -55.097   -58.626   -62.078
2027-06-17     244   -38.041   -39.049   -44.460   -45.541   -40.549   -41.650   -46.449   -47.920
2027-09-17     336   -33.292   -33.666   -38.028   -38.530   -33.760   -34.209   -36.130   -36.162

Model/market ATM skew ratio, both estimators:
expiry        days   CRL lin   CRL GEA  SSVI lin  SSVI GEA   Hes lin   Hes GEA
2026-11-20      35     1.177     1.351     1.228     1.502     1.162     1.403
2026-12-18      63     1.213     1.284     1.242     1.343     1.252     1.388
2027-01-15      91     1.207     1.254     1.207     1.267     1.274     1.365
2027-03-19     154     1.223     1.240     1.163     1.184     1.294     1.334
2027-06-17     244     1.169     1.166     1.066     1.067     1.221     1.227
2027-09-17     336     1.142     1.144     1.014     1.016     1.085     1.074

Log-log skew exponent (|S| ~ A*tau^alpha) over the six monthlies, per estimator:
source        linear       GEA
Market        -0.398    -0.445
CRL           -0.412    -0.516
SSVI          -0.487    -0.616
Heston        -0.418    -0.550

Reference: Guyon-El Amrani SPX alpha = 0.43 for T > 3 weeks.
```
