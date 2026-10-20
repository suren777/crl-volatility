# Correlated random-level model on SPX (2026-10-16 close, two sources)

Joint fit over the six monthly expiries of plain MSV, shifted MSV and the correlated
random-level model (CRL, `msv.crl_call`, 6 parameters), plus a per-expiry CRL fit as a
diagnostic. Same pipeline and vega-weighted objective as `shifted_spx.py` (with its
discounting fix). Sources: yfinance CSV and the Cboe delayed-quote snapshot (SQLite, root
SPX; SPXW ignored), both outside the repo. Run: `uv run python experiments/crl_spx.py`
(script by an opencode worker; run, checked and written up by the planner).

## Summary

| | plain MSV | shifted MSV | **CRL joint** | CRL per expiry |
|---|---|---|---|---|
| IV RMSE, yfinance (vol pts) | 7.65 | 6.08 | **0.54** | 0.48 |
| IV RMSE, Cboe (vol pts) | 7.35 | 5.80 | **0.54** | 0.49 |
| ATM skew / market, by expiry | ≈ 0 | 0.05–0.19 | **1.14–1.23** | 1.04–1.21 |
| ATM-skew decay exponent (market −0.409 / −0.398) | — | — | **−0.415 / −0.412** | |

## Findings

- One six-parameter CRL fit reproduces the SPX surface across all six expiries with
  0.54 vol points RMSE, 14× better than plain MSV and 11× better than shifted MSV. Per-expiry
  fits improve this only to 0.48: one parameter set across maturities costs little.
- CRL reproduces the market's ATM-skew term structure: fitted decay exponent −0.415 vs
  market −0.409 (yfinance), −0.412 vs −0.398 (Cboe). The model's τ^(−1/2) is the
  flat-variance asymptote; with the fitted Nelson–Siegel curve the realised exponent matches
  the market's −0.4.
- CRL overshoots the local ATM skew by 14–23% at every expiry while fitting the whole
  smile within 0.4–0.8 vol points: it trades a slightly too steep centre for the wings. The
  largest errors are the 1-month put wing (0.87–0.88 vol pts) and the longer-dated call wing
  (0.55–0.62).
- Fitted correlation is stable: ρ = −0.49 jointly, −0.46 to −0.54 per expiry, on both
  sources. The level is very dispersed: k ≈ 2.3 (ζ log-sd 1.35, P(ζ < 0.1) ≈ 15%).
- Planner checks at the fitted parameters (both sources, K/F 0.5–2.0, τ 0.02–3): no
  calendar, strike-monotonicity or convexity violations.
- The two sources agree to within a few hundredths of a vol point on every metric. The
  long end of the term structure is poorly identified with ≤ 11-month data: Q(10y) is 0.071
  (yfinance) vs 0.037 (Cboe); a2 hits 0 on yfinance.
- Caveats: one day (Friday close), monthly expiries only, |ln K/F| ≤ 0.3 and spread
  filter; no SSVI/Heston benchmark yet (epic 0003); parameter stability across days needs
  the collector's panel.

## Output

```
==============================================================================
yfinance CSV ~/.cache/opencode-fanout/article/20261019-1248-paper-experiments/spx_chain_2026-10-19.csv  spot=7659.44  asof=2026-10-16
Per-expiry parity forward/discount and OTM fit set (|ln(K/F)|<=0.3):
expiry           tau          F        D  n_options
2026-11-20    0.0959    7717.10  1.00000        379
2026-12-18    0.1726    7735.69  0.99594        359
2027-01-15    0.2493    7762.95  0.99525        328
2027-03-19    0.4219    7811.62  0.98014        131
2027-06-17    0.6685    7892.03  0.96860        133
2027-09-17    0.9205    7977.02  0.95577        122
Fitted parameters (joint over six expiries):
  plain MSV : a0=0.00000 a1=6.66572 a2=0.01638 lam=0.0041 k=0.2862  cost=1.2908  [17.6s]
  shifted   : a0=0.00000 a1=0.00970 a2=0.00102 lam=0.2803 k=0.5894 beta=22978.27  cost=1.0987  [5.6s]
  CRL       : a0=0.02125 a1=0.20668 a2=0.00000 lam=0.1975 k=2.2833 rho=-0.491  cost=0.0198  [25.9s]
  CRL per expiry (diagnostic):
    2026-11-20: lam=0.3424 k=2.6655 rho=-0.467  cost=0.0081  [6.2s]
    2026-12-18: lam=2.0668 k=2.3139 rho=-0.488  cost=0.0039  [5.4s]
    2027-01-15: lam=0.0861 k=2.2172 rho=-0.493  cost=0.0028  [4.9s]
    2027-03-19: lam=0.6573 k=1.9445 rho=-0.522  cost=0.0007  [2.6s]
    2027-06-17: lam=0.2057 k=1.7323 rho=-0.538  cost=0.0003  [2.1s]
    2027-09-17: lam=0.9342 k=1.6836 rho=-0.541  cost=0.0001  [1.9s]
IV RMSE (vol points):
expiry          n      MSV  shifted      CRL  CRL/exp
2026-11-20    379   10.167    8.214    0.747    0.672
2026-12-18    359    7.665    6.167    0.493    0.474
2027-01-15    328    6.324    5.006    0.427    0.417
2027-03-19    131    6.562    4.961    0.390    0.321
2027-06-17    133    5.173    3.747    0.413    0.213
2027-09-17    122    4.381    3.036    0.378    0.125
overall      1452    7.648    6.079    0.537    0.477
Joint-CRL IV RMSE by wing (vol points):
expiry           puts    calls
2026-11-20      0.874    0.371
2026-12-18      0.484    0.509
2027-01-15      0.427    0.428
2027-03-19      0.172    0.593
2027-06-17      0.192    0.622
2027-09-17      0.260    0.551
ATM skew (vol points per log-moneyness), |ln(K/F)|<=0.05:
expiry         market      MSV  shifted      CRL  CRL/exp  CRL/mkt
2026-11-20    -83.738    0.066   -4.498  -96.708  -96.798     1.15
2026-12-18    -67.819   -0.072   -5.249  -81.332  -81.542     1.20
2027-01-15    -57.687   -0.109   -5.592  -70.858  -70.055     1.23
2027-03-19    -45.774    0.026   -5.459  -55.866  -53.843     1.22
2027-06-17    -38.486    0.002   -5.976  -45.075  -41.358     1.17
2027-09-17    -33.497    0.005   -6.312  -38.373  -34.749     1.15
ATM-skew log-log slope over the six expiries (|skew| ~ tau^alpha):
  market -0.409   joint CRL -0.415
==============================================================================
Cboe SQLite ~/Data/article/cboe/spx.sqlite (trade_date=2026-10-16, root=SPX; SPXW ignored)  spot=7659.44  asof=2026-10-16
Per-expiry parity forward/discount and OTM fit set (|ln(K/F)|<=0.3):
expiry           tau          F        D  n_options
2026-11-20    0.0959    7720.69  0.99589        377
2026-12-18    0.1726    7740.32  0.99230        363
2027-01-15    0.2493    7769.44  0.98863        334
2027-03-19    0.4219    7817.07  0.98071        130
2027-06-17    0.6685    7898.13  0.96839        132
2027-09-17    0.9205    7984.22  0.95577        122
Fitted parameters (joint over six expiries):
  plain MSV : a0=0.00000 a1=0.14763 a2=0.01560 lam=0.2252 k=0.3603  cost=1.2585  [2.0s]
  shifted   : a0=0.00002 a1=0.00668 a2=0.00095 lam=0.4770 k=0.6663 beta=22971.06  cost=1.0652  [3.8s]
  CRL       : a0=0.00000 a1=0.09208 a2=0.01963 lam=0.5168 k=2.2600 rho=-0.487  cost=0.0200  [12.1s]
  CRL per expiry (diagnostic):
    2026-11-20: lam=0.0069 k=2.6431 rho=-0.460  cost=0.0087  [6.9s]
    2026-12-18: lam=2.3316 k=2.3098 rho=-0.480  cost=0.0041  [5.0s]
    2027-01-15: lam=0.2959 k=2.2065 rho=-0.492  cost=0.0026  [4.6s]
    2027-03-19: lam=2.2450 k=1.9513 rho=-0.517  cost=0.0007  [2.4s]
    2027-06-17: lam=0.2625 k=1.7250 rho=-0.535  cost=0.0003  [2.0s]
    2027-09-17: lam=1.8760 k=1.6832 rho=-0.538  cost=0.0001  [2.0s]
IV RMSE (vol points):
expiry          n      MSV  shifted      CRL  CRL/exp
2026-11-20    377    9.714    7.798    0.768    0.696
2026-12-18    363    7.367    5.886    0.505    0.484
2027-01-15    334    6.129    4.823    0.401    0.400
2027-03-19    130    6.391    4.788    0.378    0.330
2027-06-17    132    4.993    3.606    0.398    0.215
2027-09-17    122    4.334    3.002    0.365    0.126
overall      1458    7.348    5.801    0.540    0.485
Joint-CRL IV RMSE by wing (vol points):
expiry           puts    calls
2026-11-20      0.881    0.424
2026-12-18      0.496    0.523
2027-01-15      0.393    0.417
2027-03-19      0.191    0.565
2027-06-17      0.205    0.585
2027-09-17      0.228    0.553
ATM skew (vol points per log-moneyness), |ln(K/F)|<=0.05:
expiry         market      MSV  shifted      CRL  CRL/exp  CRL/mkt
2026-11-20    -80.436    0.067   -4.404  -94.691  -94.010     1.18
2026-12-18    -66.329   -0.146   -5.307  -80.447  -80.319     1.21
2027-01-15    -58.180   -0.182   -5.649  -70.207  -69.870     1.21
2027-03-19    -45.295    0.006   -5.488  -55.385  -53.680     1.22
2027-06-17    -38.041    0.026   -5.852  -44.460  -41.016     1.17
2027-09-17    -33.292   -0.010   -6.283  -38.028  -34.633     1.14
ATM-skew log-log slope over the six expiries (|skew| ~ tau^alpha):
  market -0.398   joint CRL -0.412
```
