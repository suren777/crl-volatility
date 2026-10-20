# CRL on sub-1-month SPX expiries (2026-10-16 close)

The paper's main open empirical risk (epics/0010, round 3 task 01-short-maturity): CRL's
ATM skew grows like τ^(−1/2), while Guyon–El Amrani find the SPX ATM skew follows a power
law above ~1 month but may saturate below it (their extrapolated zero-maturity skew is
about −1.5 in decimal units, i.e. −150 in the vol-point tables below). Ten expiries from
the Cboe delayed-quote snapshot: the four weekly SPXW Fridays 2026-10-23, -10-30, -11-06,
-11-13 (root SPXW) plus the six monthlies of `shifted_spx.EXPIRIES` (root SPX), each
expiry loaded from its own root. Pipeline, vega-weighted CRL objective and IV metrics are
reused from `shifted_spx.py` and `crl_spx.py` by import; the CRL fit is repeated locally
with ρ bounded to [−0.99, 0] (`crl_call` is reliable only for ρ ≤ 0, and
`crl_spx.fit_crl` allows ρ up to +0.99). Skews are in vol points per log-moneyness
(100 × decimal). Run: `uv run python experiments/crl_short.py` (script and write-up by an
opencode worker; not yet planner-checked).

## Summary

| | CRL joint (all ten) | CRL monthlies-only (weeklies out of sample) |
|---|---|---|
| IV RMSE, all ten expiries (vol pts) | **1.186** | 1.292 |
| IV RMSE, four weeklies (vol pts) | 0.93–2.08 | 0.77–2.87 |
| ATM skew / market at the weeklies | 1.12–1.31 | 1.17–1.36 |
| ATM-skew exponent, all ten | −0.375 (market −0.374) | −0.510 on the predicted weeklies (market −0.399) |
| Implied ATM skew at 1d / 7d (vol pts) | −388.1 / −242.6 | −441.7 / −252.0 |

Market side: the pure power law fitted on all ten expiries gives A = 33.05, α = −0.374
(RMSE 5.1); the 7-day market skew is −153.7 vol pts = −1.54 decimal.

## Per-expiry IV RMSE and ATM skew (market vs CRL joint vs CRL monthlies-only)

| expiry | days | skew mkt | CRL joint | joint/mkt | CRL mon | mon/mkt | RMSE joint | RMSE mon |
|---|---|---|---|---|---|---|---|---|
| 2026-10-23 | 7 | −153.729 | −201.118 | 1.31 | −209.778 | 1.36 | 1.852 | 2.865 |
| 2026-10-30 | 14 | −98.882 | −116.835 | 1.18 | −126.342 | 1.28 | 2.081 | 2.016 |
| 2026-11-06 | 21 | −96.385 | −110.046 | 1.14 | −115.902 | 1.20 | 1.060 | 0.996 |
| 2026-11-13 | 28 | −87.304 | −97.852 | 1.12 | −102.443 | 1.17 | 0.926 | 0.772 |
| 2026-11-20 | 35 | −80.436 | −91.416 | 1.14 | −94.691 | 1.18 | 0.845 | 0.768 |
| 2026-12-18 | 63 | −66.329 | −80.759 | 1.22 | −80.447 | 1.21 | 0.509 | 0.505 |
| 2027-01-15 | 91 | −58.180 | −71.537 | 1.23 | −70.207 | 1.21 | 0.599 | 0.401 |
| 2027-03-19 | 154 | −45.295 | −56.768 | 1.25 | −55.385 | 1.22 | 0.716 | 0.378 |
| 2027-06-17 | 244 | −38.041 | −45.667 | 1.20 | −44.460 | 1.17 | 0.556 | 0.398 |
| 2027-09-17 | 336 | −33.292 | −39.189 | 1.18 | −38.028 | 1.14 | 0.831 | 0.365 |
| overall | | | | | | | 1.186 | 1.292 |

## Zero-maturity behaviour: implied |skew| at τ = 1, 2, 3, 7 days (vol pts)

| fit | 1d | 2d | 3d | 7d |
|---|---|---|---|---|
| power (all ten) | 300.666 | 231.969 | 199.312 | 145.153 |
| shifted (all ten) | 307.894 | 236.401 | 202.545 | 146.639 |
| capped (all ten) | 153.729 | 153.729 | 153.729 | 146.639 |
| power (monthlies) | 339.057 | 257.394 | 219.076 | 156.425 |
| shifted (monthlies) | 127.175 | 124.425 | 121.849 | 112.975 |
| capped (monthlies) | 80.436 | 80.436 | 80.436 | 80.436 |
| CRL joint (all ten, signed) | −388.091 | −335.052 | −305.388 | −242.634 |
| CRL monthlies-only (signed) | −441.681 | −374.284 | −334.952 | −251.991 |

Market ATM skew at 7 days: −153.729 (the observed anchor of this table).

## Findings

- The market ATM skew keeps growing below 1 month on this snapshot: −87.3 to −153.7 vol
  pts across the four weeklies, with the same power-law exponent as above 1 month
  (weeklies −0.399 vs monthlies −0.398). Fitted on all ten expiries, the pure power law
  gives α = −0.374 (RMSE 5.1), and neither finite-at-zero shape improves on it: the
  time-shifted shape drives τ0 to 0 and the capped shape puts the cap exactly at the 7-day
  maximum (153.7). No saturation toward a finite zero-maturity skew is visible down to
  7 days.
- The 7-day market skew of −153.7 vol pts (−1.54 decimal) is close to Guyon–El Amrani's
  extrapolated zero-maturity skew of about −1.5; the all-ten power law implies −145.2 at
  7 days (within 6%).
- Extrapolating saturation from the monthlies only is what fails: the monthlies-only
  time-shifted fit (τ0 = 0.056 y ≈ 21 days, α = −0.482) improves the in-sample RMSE
  (0.55 vs 1.20 for the power law) but crosses the weekly skews — under-predicting 7 days
  by 26% and over-predicting 28 days by 29% (RMSE 20.5) — and the capped fit (cap = 80.4,
  the 1-month value) under-predicts every weekly skew (RMSE 38.8). The pure power law is
  the best weekly extrapolator (RMSE 10.4).
- The joint CRL fit still works when the weeklies are included: overall IV RMSE 1.186 vol
  pts over all ten expiries, with k = 2.66, ρ = −0.452 — but it is a real degradation from
  the six-monthly fit (0.540 overall, k = 2.26, ρ = −0.487, cost 0.0200 → 0.1550), and the
  weeklies are the worst-fitted expiries (0.93–2.08 vol pts vs ≤ 0.85 at the monthlies).
- CRL over-shoots the market skew below 1 month: 1.12–1.31× market at the weeklies for the
  joint fit, 1.17–1.36× for the monthlies-only fit out of sample (weekly IV RMSE up to
  2.09 and 2.87 vol pts respectively).
- The over-shoot grows as τ → 0: from the fitted parameters CRL implies an ATM skew of
  −242.6 (joint) / −252.0 (monthlies-only) at 7 days against the market's −153.7, and
  −388.1 / −441.7 at 1 day where the market's own power-law trend implies about −300 and
  Guyon–El Amrani extrapolate −150. The monthlies-only fit's predicted weekly skew slope
  is −0.510 against the market's −0.399 — CRL is approaching its flat-Q τ^(−1/2) asymptote
  exactly where the market does not.
- Verdict: CRL's short-end behaviour contradicts the market in level, not in exponent —
  on the fitted Nelson–Siegel curve the joint fit's realised skew exponent over all ten
  expiries is −0.375 vs market −0.374, but the model's short-end skew level is 12–36% too
  steep at 7–28 days and roughly 1.6× the market at 7 days as τ → 0. Any claim that CRL
  "matches the SPX skew across maturities" must exclude or caveat the sub-monthly range.
- Caveats: one day (Friday 2026-10-16 close), delayed-quote snapshot, fit set
  |ln K/F| ≤ 0.3 with a spread filter, ATM skew over a ±5% window; the weeklies are one
  root (SPXW) and the monthlies another (SPX); the market-side power law could still bend
  between 1 and 7 days, which this snapshot cannot see.

## Output

```
Cboe SQLite ~/Data/article/cboe/spx.sqlite  trade_date=2026-10-16  spot=7659.44  asof=2026-10-16  (weeklies root SPXW, monthlies root SPX)

Per-expiry parity forward/discount and OTM fit set (|ln(K/F)|<=0.3):
expiry       root      tau  days          F        D  n_options
2026-10-23   SPXW   0.0192     7    7702.07  0.99931        227
2026-10-30   SPXW   0.0384    14    7707.80  0.99853        375
2026-11-06   SPXW   0.0575    21    7713.85  0.99776        235
2026-11-13   SPXW   0.0767    28    7717.64  0.99684        202
2026-11-20   SPX    0.0959    35    7720.69  0.99589        377
2026-12-18   SPX    0.1726    63    7740.32  0.99230        363
2027-01-15   SPX    0.2493    91    7769.44  0.98863        334
2027-03-19   SPX    0.4219   154    7817.07  0.98071        130
2027-06-17   SPX    0.6685   244    7898.13  0.96839        132
2027-09-17   SPX    0.9205   336    7984.22  0.95577        122

Market ATM skew (vol pts per log-moneyness), |ln(K/F)|<=0.05:
expiry       root      tau       skew
2026-10-23   SPXW   0.0192   -153.729
2026-10-30   SPXW   0.0384    -98.882
2026-11-06   SPXW   0.0575    -96.385
2026-11-13   SPXW   0.0767    -87.304
2026-11-20   SPX    0.0959    -80.436
2026-12-18   SPX    0.1726    -66.329
2027-01-15   SPX    0.2493    -58.180
2027-03-19   SPX    0.4219    -45.295
2027-06-17   SPX    0.6685    -38.041
2027-09-17   SPX    0.9205    -33.292

Market ATM-skew log-log slope (|skew| ~ tau^alpha): all ten -0.374, six monthlies -0.398, four weeklies -0.399

Skew shape fits, fitted on all ten expiries (weeklies in sample; RMSE in vol points):
shape     parameters                            RMSE fit set   RMSE weeklies
power     A=33.05 alpha=-0.374                         5.111           7.845
shifted   A=32.48 tau0=0.0000 alpha=-0.381             5.074           7.728
capped    A=32.48 alpha=-0.381 cap=153.73              5.074           7.728

Skew shape fits, fitted on the six monthlies only (weeklies out of sample; RMSE in vol points):
shape     parameters                            RMSE fit set   RMSE weeklies
power     A=32.48 alpha=-0.398                         1.203          10.393
shifted   A=32.51 tau0=0.0563 alpha=-0.482             0.547          20.508
capped    A=32.09 alpha=-0.417 cap=80.44               0.501          38.774

CRL fitted on all ten: a0=0.00000 a1=0.06809 a2=0.01415 lam=2.2220 k=2.6603 rho=-0.452  cost=0.1550  [18.3s]

CRL fitted on six monthlies: a0=0.00000 a1=0.09208 a2=0.01963 lam=0.5167 k=2.2600 rho=-0.487  cost=0.0200  [10.7s]

Per-expiry IV RMSE (vol pts) and ATM skew (vol pts per log-moneyness):
expiry          n  RMSE all  RMSE mon  skew mkt  skew all  all/mkt  skew mon  mon/mkt
2026-10-23    227     1.852     2.865  -153.729  -201.118     1.31  -209.778     1.36
2026-10-30    375     2.081     2.016   -98.882  -116.835     1.18  -126.342     1.28
2026-11-06    235     1.060     0.996   -96.385  -110.046     1.14  -115.902     1.20
2026-11-13    202     0.926     0.772   -87.304   -97.852     1.12  -102.443     1.17
2026-11-20    377     0.845     0.768   -80.436   -91.416     1.14   -94.691     1.18
2026-12-18    363     0.509     0.505   -66.329   -80.759     1.22   -80.447     1.21
2027-01-15    334     0.599     0.401   -58.180   -71.537     1.23   -70.207     1.21
2027-03-19    130     0.716     0.378   -45.295   -56.768     1.25   -55.385     1.22
2027-06-17    132     0.556     0.398   -38.041   -45.667     1.20   -44.460     1.17
2027-09-17    122     0.831     0.365   -33.292   -39.189     1.18   -38.028     1.14
overall      2497     1.186     1.292

CRL ATM-skew log-log slope (|skew| ~ tau^alpha):
  joint fit:  all ten -0.375 (market -0.374), six monthlies -0.386 (market -0.398)
  monthlies-only fit: on the six monthlies -0.412; on the four predicted weeklies -0.510 (market weeklies -0.399)

Zero-maturity behaviour: |skew| for shapes, signed model ATM skew for CRL (vol pts):
fit                                   1d        2d        3d        7d
power (all ten)                  300.666   231.969   199.312   145.153
shifted (all ten)                307.894   236.401   202.545   146.639
capped (all ten)                 153.729   153.729   153.729   146.639
power (monthlies)                339.057   257.394   219.076   156.425
shifted (monthlies)              127.175   124.425   121.849   112.975
capped (monthlies)                80.436    80.436    80.436    80.436
CRL joint (all ten)             -388.091  -335.052  -305.388  -242.634
CRL monthlies-only              -441.681  -374.284  -334.952  -251.991
```
