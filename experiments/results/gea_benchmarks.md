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
| 2026-11-20 | 35 | −80.436 | −91.535 | −94.691 | −123.671 | −98.759 | −137.458 | −93.382 | −133.566 |
| 2026-12-18 | 63 | −66.329 | −71.477 | −80.447 | −91.772 | −82.365 | −95.994 | −81.661 | −94.691 |
| 2027-01-15 | 91 | −58.180 | −60.647 | −70.207 | −76.058 | −70.197 | −76.870 | −69.753 | −75.064 |
| 2027-03-19 | 154 | −45.295 | −46.534 | −55.385 | −57.715 | −52.674 | −55.097 | −50.729 | −52.230 |
| 2027-06-17 | 244 | −38.041 | −39.049 | −44.460 | −45.541 | −40.549 | −41.650 | −36.794 | −37.299 |
| 2027-09-17 | 336 | −33.292 | −33.666 | −38.028 | −38.530 | −33.760 | −34.209 | −28.896 | −29.064 |

## Model/market ATM skew ratio

| expiry | days | CRL lin | CRL GEA | SSVI lin | SSVI GEA | Hes lin | Hes GEA |
|--------|------|---------|---------|----------|----------|---------|---------|
| 2026-11-20 | 35 | 1.177 | 1.351 | 1.228 | 1.502 | 1.161 | 1.459 |
| 2026-12-18 | 63 | 1.213 | 1.284 | 1.242 | 1.343 | 1.231 | 1.325 |
| 2027-01-15 | 91 | 1.207 | 1.254 | 1.207 | 1.267 | 1.199 | 1.238 |
| 2027-03-19 | 154 | 1.223 | 1.240 | 1.163 | 1.184 | 1.120 | 1.122 |
| 2027-06-17 | 244 | 1.169 | 1.166 | 1.066 | 1.067 | 0.967 | 0.955 |
| 2027-09-17 | 336 | 1.142 | 1.144 | 1.014 | 1.016 | 0.868 | 0.863 |

## Log-log skew exponent (|S| ~ A·tau^α) over the six monthlies

| estimator | Market | CRL | SSVI | Heston |
|-----------|--------|-----|------|--------|
| linear | −0.398 | −0.412 | −0.487 | −0.534 |
| GEA | −0.445 | −0.516 | −0.616 | −0.676 |

Guyon–El Amrani SPX reference: α = 0.43 above 3 weeks.

## Findings

Corrected 2026-10-29: Heston refitted after the `heston_cf` branch-cut fix (commit 09d1e47);
the earlier Heston columns came from a characteristic function that crossed the branch cut of
the complex log at this surface's vol-of-vol. Refitted again 2026-10-30 after the integration
cutoff fix (`heston.cutoff`): the fixed cutoff u = 200 stopped short of where the CF had decayed
at the short expiries.

- **The fits are the benchmark's own.** CRL (`a0=0.00000 a1=0.09208 a2=0.01963
  lam=0.5168 k=2.2600 ρ=−0.487`), SSVI (`η=0.8016 γ=0.5937 ρ=−0.579`) and Heston
  (`v0=0.01780 κ=14.7219 θ=0.03871 σ=3.0682 ρ=−0.690`, Feller 0.121 violated)
  reproduce the `benchmarks_spx.md` parameters and the linear skew ratios, so
  the GEA columns are the same models, re-read with their estimator.
- **Under GEA, the models over-shoot the market ATM skew more than under the
  linear estimator at the short end.** At 35 days the GEA ratio is 1.35 (CRL),
  1.50 (SSVI) and 1.46 (Heston) against 1.18, 1.23 and 1.16 linear; at the longest
  expiry the two estimators agree (1.14, 1.02, 0.86–0.87). The GEA degree-2 fit
  captures near-money curvature the ±5% linear slope misses, so any paper number
  stated against G&EA's 0.43 / estimator must use the GEA columns.
- **SSVI and Heston are more curved near the money than CRL**, and Heston's skew
  falls below the market's beyond six months (ratio 0.95–0.97 at 244 days, 0.86–0.87
  at 336 days).
- **Skew exponents.** On the six monthlies the GEA skew exponent is −0.445 for the
  market (right on G&EA's 0.43) versus −0.516 for CRL, −0.616 for SSVI and −0.676
  for Heston; the linear exponents are −0.398 (market), −0.412 (CRL), −0.487 (SSVI),
  −0.534 (Heston). All three models decay faster than the market under both
  estimators; CRL is closest, Heston now the steepest.
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
  CRL   : a0=0.00000 a1=0.09208 a2=0.01963 lam=0.5168 k=2.2600 rho=-0.487 (rho<=0)  cost=0.0200  [11.0s]
  SSVI  : eta=0.8016 gamma=0.5937 rho=-0.579 cost=0.0354  [5.5s]
  Heston: v0=0.01780 kappa=14.7219 theta=0.03871 sigma=3.0682 rho=-0.690  cost=0.0296  [14.4s]  Feller=0.121

Per-expiry ATM skew (vol pts per log-moneyness): market and models, both estimators
expiry        days   mkt lin   mkt GEA   CRL lin   CRL GEA  SSVI lin  SSVI GEA   Hes lin   Hes GEA
2026-11-20      35   -80.436   -91.535   -94.691  -123.671   -98.759  -137.458   -93.382  -133.566
2026-12-18      63   -66.329   -71.477   -80.447   -91.772   -82.365   -95.994   -81.661   -94.691
2027-01-15      91   -58.180   -60.647   -70.207   -76.058   -70.197   -76.870   -69.753   -75.064
2027-03-19     154   -45.295   -46.534   -55.385   -57.715   -52.674   -55.097   -50.729   -52.230
2027-06-17     244   -38.041   -39.049   -44.460   -45.541   -40.549   -41.650   -36.794   -37.299
2027-09-17     336   -33.292   -33.666   -38.028   -38.530   -33.760   -34.209   -28.896   -29.064

Model/market ATM skew ratio, both estimators:
expiry        days   CRL lin   CRL GEA  SSVI lin  SSVI GEA   Hes lin   Hes GEA
2026-11-20      35     1.177     1.351     1.228     1.502     1.161     1.459
2026-12-18      63     1.213     1.284     1.242     1.343     1.231     1.325
2027-01-15      91     1.207     1.254     1.207     1.267     1.199     1.238
2027-03-19     154     1.223     1.240     1.163     1.184     1.120     1.122
2027-06-17     244     1.169     1.166     1.066     1.067     0.967     0.955
2027-09-17     336     1.142     1.144     1.014     1.016     0.868     0.863

Log-log skew exponent (|S| ~ A*tau^alpha) over the six monthlies, per estimator:
source        linear       GEA
Market        -0.398    -0.445
CRL           -0.412    -0.516
SSVI          -0.487    -0.616
Heston        -0.534    -0.676

Reference: Guyon-El Amrani SPX alpha = 0.43 for T > 3 weeks.
```
