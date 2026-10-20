# Benchmarks: CRL vs SSVI vs Heston on SPX (2026-10-19)

Model comparison on a single-day SPX option surface. Two data sources for the
same trade date `2026-10-16` (spot 7659.44): the yfinance chain CSV
(`spx_chain_2026-10-19.csv`) and the Cboe SQLite snapshot
(`spx.sqlite`, trade_date `2026-10-16`, root `SPX`). Both use six monthly
expiries in an OTM fit set with `|ln(K/F)| <= 0.3`; prices are vega-weighted
price residuals inverted to IV and reported as RMSE in vol points. Forward
`F` and discount `D` come from put-call parity.

Models compared: plain MSV (Nelson–Siegel `Q(t)`, no correlation), CRL (the
paper's correlated random-level model, `rho <= 0` enforced), SSVI
(`arXiv:1204.0646v4`, power-law `phi = eta*theta^-gamma`, one free `theta`
per expiry, `eta`/`gamma`/`rho` shared), and Heston (5 params, multi-start
fit; Feller reported but not imposed).

## yfinance CSV

Per-expiry forward/discount and OTM fit set:

| expiry      | tau    | F      | D      | n_options |
|-------------|--------|--------|--------|-----------|
| 2026-11-20 | 0.0959 | 7717.10 | 1.00000 | 379 |
| 2026-12-18 | 0.1726 | 7735.69 | 0.99594 | 359 |
| 2027-01-15 | 0.2493 | 7762.95 | 0.99525 | 328 |
| 2027-03-19 | 0.4219 | 7811.62 | 0.98014 | 131 |
| 2027-06-17 | 0.6685 | 7892.03 | 0.96860 | 133 |
| 2027-09-17 | 0.9205 | 7977.02 | 0.95577 | 122 |

Fitted parameters (joint, six expiries) and fit time:

- plain MSV: `a0=0.00000 a1=6.66572 a2=0.01638 lam=0.0041 k=0.2862`,
  cost `1.2908` [16.6s]
- CRL: `a0=0.02125 a1=0.20668 a2=0.00000 lam=0.1975 k=2.2833 rho=-0.491`
  (`rho<=0`), cost `0.0198` [24.0s]
- SSVI: `eta=0.7807 gamma=0.5992 rho=-0.584`, cost `0.0340` [6.5s]
  - per-expiry `theta` (ATM total variance): `0.00158 0.00316 0.00471
   0.00919 0.01624 0.02437`
- Heston: `v0=0.02387 kappa=2.3931 theta=0.06166 sigma=1.4345 rho=-0.712`,
  cost `0.0692` [31.7s]
  - Feller `2*kappa*theta/sigma^2 = 0.143` (VIOLATED)

IV RMSE (vol points):

| expiry      | n   | MSV   | CRL   | SSVI | Heston |
|-------------|-----|-------|-------|------|--------|
| 2026-11-20 | 379 | 10.167 | 0.747 | 1.150 | 1.876 |
| 2026-12-18 | 359 | 7.665  | 0.493 | 0.679 | 0.756 |
| 2027-01-15 | 328 | 6.324  | 0.427 | 0.533 | 0.665 |
| 2027-03-19 | 131 | 6.562  | 0.390 | 0.372 | 0.737 |
| 2027-06-17 | 133 | 5.173  | 0.413 | 0.226 | 0.738 |
| 2027-09-17 | 122 | 4.381  | 0.378 | 0.157 | 1.072 |
| overall    | 1452 | 7.648 | 0.537 | 0.737 | 1.164 |

ATM skew ratio (model/market), `|ln(K/F)| <= 0.05`:

| expiry      | market  | MSV | CRL | SSVI | Heston |
|-------------|---------|-----|-----|------|--------|
| 2026-11-20 | -83.738 | -0.00 | 1.15 | 1.21 | 1.15 |
| 2026-12-18 | -67.819 | 0.00  | 1.20 | 1.23 | 1.25 |
| 2027-01-15 | -57.687 | 0.00  | 1.23 | 1.23 | 1.31 |
| 2027-03-19 | -45.774 | -0.00 | 1.22 | 1.16 | 1.31 |
| 2027-06-17 | -38.486 | -0.00 | 1.17 | 1.06 | 1.25 |
| 2027-09-17 | -33.497 | -0.00 | 1.15 | 1.01 | 1.14 |

## Cboe SQLite

Per-expiry forward/discount and OTM fit set:

| expiry      | tau    | F      | D      | n_options |
|-------------|--------|--------|--------|-----------|
| 2026-11-20 | 0.0959 | 7720.69 | 0.99589 | 377 |
| 2026-12-18 | 0.1726 | 7740.32 | 0.99230 | 363 |
| 2027-01-15 | 0.2493 | 7769.44 | 0.98863 | 334 |
| 2027-03-19 | 0.4219 | 7817.07 | 0.98071 | 130 |
| 2027-06-17 | 0.6685 | 7898.13 | 0.96839 | 132 |
| 2027-09-17 | 0.9205 | 7984.22 | 0.95577 | 122 |

Fitted parameters (joint, six expiries) and fit time:

- plain MSV: `a0=0.00000 a1=0.14763 a2=0.01560 lam=0.2252 k=0.3603`,
  cost `1.2585` [1.9s]
- CRL: `a0=0.00000 a1=0.09208 a2=0.01963 lam=0.5167 k=2.2600 rho=-0.487`
  (`rho<=0`), cost `0.0200` [10.7s]
- SSVI: `eta=0.8016 gamma=0.5937 rho=-0.579`, cost `0.0354` [5.6s]
  - per-expiry `theta` (ATM total variance): `0.00150 0.00306 0.00462
   0.00906 0.01606 0.02415`
- Heston: `v0=0.02242 kappa=2.4940 theta=0.06000 sigma=1.3997 rho=-0.703`,
  cost `0.0663` [19.9s]
  - Feller `2*kappa*theta/sigma^2 = 0.153` (VIOLATED)

IV RMSE (vol points):

| expiry      | n   | MSV   | CRL   | SSVI | Heston |
|-------------|-----|-------|-------|------|--------|
| 2026-11-20 | 377 | 9.714  | 0.768 | 1.192 | 1.913 |
| 2026-12-18 | 363 | 7.367  | 0.505 | 0.699 | 0.772 |
| 2027-01-15 | 334 | 6.129  | 0.401 | 0.523 | 0.599 |
| 2027-03-19 | 130 | 6.391  | 0.378 | 0.382 | 0.647 |
| 2027-06-17 | 132 | 4.993  | 0.398 | 0.231 | 0.641 |
| 2027-09-17 | 122 | 4.334  | 0.365 | 0.162 | 0.967 |
| overall    | 1458 | 7.348 | 0.540 | 0.756 | 1.153 |

ATM skew ratio (model/market), `|ln(K/F)| <= 0.05`:

| expiry      | market  | MSV | CRL | SSVI | Heston |
|-------------|---------|-----|-----|------|--------|
| 2026-11-20 | -80.436 | -0.00 | 1.18 | 1.23 | 1.16 |
| 2026-12-18 | -66.329 | 0.00  | 1.21 | 1.24 | 1.25 |
| 2027-01-15 | -58.180 | 0.00  | 1.21 | 1.21 | 1.27 |
| 2027-03-19 | -45.295 | -0.00 | 1.22 | 1.16 | 1.29 |
| 2027-06-17 | -38.041 | -0.00 | 1.17 | 1.07 | 1.22 |
| 2027-09-17 | -33.292 | 0.00  | 1.14 | 1.01 | 1.09 |

## Findings

On a single trading day (2026-10-16) across six monthly expiries, the CRL
model fits OTM SPX implied volatilities substantially better than both the
plain MSV and Heston models, and slightly better than SSVI on the overall
vega-weighted IV RMSE (0.54 vs 0.76 for SSVI, 1.15–1.16 for Heston, and 7.3–
7.6 for plain MSV). The CRL advantage is largest at the short end: at the
2026-11-20 expiry its IV RMSE is 0.75 vs 1.19 (SSVI) and 1.91 (Heston); the
plain MSV error decays into the longest expiry but stays far above the other
three.

SSVI overtakes CRL on the two longest expiries (IV RMSE 0.16–0.23 vs
0.37–0.41), where CRL's fitted smile is least flexible without a
per-expiry leverage parameter; both remain well under the ~0.6–1.9 RMSE of
Heston at those maturities. Heston, fitted freely, does not beat CRL
anywhere on this surface and shows a Feller-ratio violation of 0.14–0.15,
well below 1, so its variance process is not mean-reversion-stationary here.

For the ATM skew, CRL and SSVI both reproduce roughly 1.15–1.23x the market
skew magnitude at the short end, tightening toward ~1.0–1.2x at the longest
expiry; Heston is consistently a touch steeper, ~1.1–1.3x. This is a
single-day snapshot with monthly expiries only (no weeklies), so short-
maturity behaviour is under-sampled, and results are not evidence of
out-of-sample performance.

## Caveats

Single day (2026-10-16) and monthly expiries only; no weeklies at or below
one month are included, so the deep-OTM short-end fit is not stressed here.
Fits are in-sample on the vega-weighted price residuals; no temporal
validation. Heston's Feller violation is reported, not imposed, and its IV
RMSE is on the same OTM fit set as the others.
