# Market and CRL ATM skews with the Guyon–El Amrani estimator (2026-10-16 close)

`experiments/gea_skew.py` recomputes the ATM skews of the prior notes
(`crl_short.md`, linear estimator over |ln K/F| ≤ 0.05) with the Guyon–El Amrani
estimator so the numbers are comparable to theirs: `S(T) = |dσ_BS/d ln K|` at `K = F_T`
"by fitting a polynomial of degree 2 locally around the money, using the 6 closest strikes
to the left and the 6 closest strikes to the right of the T-forward" (`msv.py` pipeline).
Ten expiries from the Cboe 2026-10-16 snapshot — four weekly SPXW Fridays plus the six SPX
monthlies, exactly as in `crl_short.py` (loader, IV pipeline, and the CRL joint fit bounded
to ρ ∈ [−0.99, 0] imported). Skews are in vol points per log-moneyness (100 × decimal). Run:
`uv run python experiments/gea_skew.py`.

## Summary

| | Market | CRL joint (all ten) |
|---|---|---|
| ATM-skew exponent, all ten | −0.412 | −0.511 |
| ATM-skew exponent, ≥ 3 weeks | **−0.443** | −0.515 |
| ATM-skew exponent, < 3 weeks | −0.353 (2 expiries) | −0.494 |
| ATM GEA skew at 7 days (vol pts / decimal) | −159.8 / −1.60 | −286.6 / −2.87 |
| CRL GEA / market GEA at 7 days | — | 1.79 |

Guyon–El Amrani SPX reference: α = 0.19 below / 0.43 above 3 weeks, extrapolated
zero-maturity skew ≈ 1.5 (decimal = 150 vol pts).

## Findings

- **Market, ≥ 3 weeks: −0.443 vs G&EA 0.43.** Over the eight expiries at or above 21 days the
  G&EA-estimated market skew decays as τ^(−0.443), within 3% of Guyon–El Amrani's SPX
  exponent 0.43 for maturities above 3 weeks. The prior linear estimator reported −0.398
  over the six monthlies; switching to the degree-2/12-strike estimator moves the long end
  essentially onto their number.
- **Market, all ten: −0.412.** Including the two sub-14-day weeklies flattens the exponent
  only slightly (−0.412 vs −0.443), consistent with a single regime here rather than G&EA's
  two-regime average across two years of monthly data.
- **The < 3-week exponent is not comparable to their 0.19.** Only two weeklies (7, 14 days)
  fall strictly below 21 days on this snapshot, so their −0.353 is a two-point slope we report
  as a single-date, two-point statistic, not evidence for a kink. One day cannot resolve the
  average two-regime shape G&EA see.
- **Market zero-maturity skew ≈ G&EA's 1.5.** The G&EA market skew at 7 days is −159.8 vol
  pts = −1.60 decimal, within 7% of their extrapolated zero-maturity value of about 1.5 (and
  closer than the linear −153.7); the all-ten power law extrapolates −145 at 7 days. The
  2-parameter power law still reaches the observed weeklies on this date.
- **CRL still overshoots, and more steeply under the G&EA estimator.** The joint CRL fit
  (k = 2.66, ρ = −0.452) reproduces the skew's shape — exponent −0.515 (≥ 3 weeks) vs market
  −0.443 — but its G&EA skew level is 1.19–1.79× the market's at every expiry, widest at the
  short end (1.79×, or −286.6 vs −159.8 vol pts at 7 days). CRL's degree-2 local skew rises
  faster than its linear one because the model's smile is more curved near the money than the
  market's.
- **Estimator effect.** The G&EA estimator reads 1.0–1.3× the linear one on the market (from
  1.01 long-dated to 1.27 at 14 days) and 1.2–1.7× on the CRL model, because the degree-2
  polynomial over the 12 nearest strikes captures near-money curvature the ±5% linear slope
  does not. Any paper comparison to G&EA's 0.43 / 1.5 must use their estimator.
- Caveats: one day (Friday 2026-10-16 close), delayed-quote snapshot, |ln K/F| ≤ 0.3 fit set
  with spread filter, weeklies one root (SPXW) and monthlies another (SPX); the sub-3-week
  exponent is a two-point fit.

## Output

```
Cboe SQLite ~/Data/article/cboe/spx.sqlite  trade_date=2026-10-16  spot=7659.44  asof=2026-10-16  (weeklies root SPXW, monthlies root SPX)
ATM skew estimators: GEA = degree-2 poly, 6 closest strikes per side of the forward (Guyon-El Amrani); linear = regression over |ln K/F| <= 0.05.

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

Market ATM skew (vol pts per log-moneyness), both estimators:
expiry       root      tau  days    linear       GEA  GEA/mkt lin
2026-10-23   SPXW   0.0192     7  -153.729  -159.820        1.040
2026-10-30   SPXW   0.0384    14   -98.882  -125.132        1.265
2026-11-06   SPXW   0.0575    21   -96.385  -112.147        1.164
2026-11-13   SPXW   0.0767    28   -87.304  -103.567        1.186
2026-11-20   SPX    0.0959    35   -80.436   -91.535        1.138
2026-12-18   SPX    0.1726    63   -66.329   -71.477        1.078
2027-01-15   SPX    0.2493    91   -58.180   -60.647        1.042
2027-03-19   SPX    0.4219   154   -45.295   -46.534        1.027
2027-06-17   SPX    0.6685   244   -38.041   -39.049        1.027
2027-09-17   SPX    0.9205   336   -33.292   -33.666        1.011

CRL fitted on all ten: a0=0.00000 a1=0.06809 a2=0.01415 lam=2.2220 k=2.6603 rho=-0.452  cost=0.1550  [19.8s]

Per-expiry ATM skew (vol pts per log-moneyness): both estimators, market vs CRL
expiry        days   mkt lin   mkt GEA   CRL lin   CRL GEA  CRL GEA/mkt GEA
2026-10-23       7  -153.729  -159.820  -201.118  -286.587            1.793
2026-10-30      14   -98.882  -125.132  -116.835  -203.424            1.626
2026-11-06      21   -96.385  -112.147  -110.046  -166.072            1.481
2026-11-13      28   -87.304  -103.567   -97.852  -143.673            1.387
2026-11-20      35   -80.436   -91.535   -91.416  -128.351            1.402
2026-12-18      63   -66.329   -71.477   -80.759   -95.175            1.332
2027-01-15      91   -58.180   -60.647   -71.537   -78.816            1.300
2027-03-19     154   -45.295   -46.534   -56.768   -59.683            1.283
2027-06-17     244   -38.041   -39.049   -45.667   -47.116            1.207
2027-09-17     336   -33.292   -33.666   -39.189   -39.925            1.186

Market: ATM-skew power-law exponent (|S| ~ A*tau^alpha), GEA estimator:
  all ten       -0.412   (fit offset ln A = 3.514)
  >=3 weeks     -0.443   (8 expiries, tau from 21d)
  <3 weeks      -0.353   (2 expiries, tau up to 14d)

CRL joint: ATM-skew power-law exponent (|S| ~ A*tau^alpha), GEA estimator:
  all ten       -0.511   (fit offset ln A = 3.651)
  >=3 weeks     -0.515   (8 expiries, tau from 21d)
  <3 weeks      -0.494   (2 expiries, tau up to 14d)

Guyon-El Amrani reference: SPX alpha = 0.19 if T < 3 weeks, 0.43 if T > 3 weeks; extrapolated zero-maturity skew about 1.5 decimal = 150 vol pts.
Market GEA skew at 7d = -159.8 vol pts = -1.60 decimal vs GEA ~1.5 decimal.
CRL  GEA skew at 7d = -286.6 vol pts (1.79x market).
```
