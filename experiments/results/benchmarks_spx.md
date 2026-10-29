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
  cost `1.2908` [16.8s]
- CRL: `a0=0.02125 a1=0.20668 a2=0.00000 lam=0.1975 k=2.2833 rho=-0.491`
  (`rho<=0`), cost `0.0198` [27.3s]
- SSVI: `eta=0.7807 gamma=0.5992 rho=-0.584`, cost `0.0340` [9.9s]
  - per-expiry `theta` (ATM total variance): `0.00158 0.00316 0.00471
   0.00919 0.01624 0.02437`
- Heston: `v0=0.02061 kappa=15.4838 theta=0.03860 sigma=3.1942 rho=-0.696`,
  cost `0.0288` [6.6s]
  - Feller `2*kappa*theta/sigma^2 = 0.117` (VIOLATED)

IV RMSE (vol points):

| expiry      | n   | MSV   | CRL   | SSVI | Heston |
|-------------|-----|-------|-------|------|--------|
| 2026-11-20 | 379 | 10.167 | 0.747 | 1.150 | 0.918 |
| 2026-12-18 | 359 | 7.665  | 0.493 | 0.679 | 0.554 |
| 2027-01-15 | 328 | 6.324  | 0.427 | 0.533 | 0.502 |
| 2027-03-19 | 131 | 6.562  | 0.390 | 0.372 | 0.325 |
| 2027-06-17 | 133 | 5.173  | 0.413 | 0.226 | 0.414 |
| 2027-09-17 | 122 | 4.381  | 0.378 | 0.157 | 0.651 |
| overall    | 1452 | 7.648 | 0.537 | 0.737 | 0.643 |

ATM skew ratio (model/market), `|ln(K/F)| <= 0.05`:

| expiry      | market  | MSV | CRL | SSVI | Heston |
|-------------|---------|-----|-----|------|--------|
| 2026-11-20 | -83.738 | -0.00 | 1.15 | 1.21 | 1.14 |
| 2026-12-18 | -67.819 | 0.00  | 1.20 | 1.23 | 1.22 |
| 2027-01-15 | -57.687 | 0.00  | 1.23 | 1.23 | 1.22 |
| 2027-03-19 | -45.774 | -0.00 | 1.22 | 1.16 | 1.11 |
| 2027-06-17 | -38.486 | -0.00 | 1.17 | 1.06 | 0.96 |
| 2027-09-17 | -33.497 | -0.00 | 1.15 | 1.01 | 0.87 |

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
  cost `1.2585` [2.7s]
- CRL: `a0=0.00000 a1=0.09208 a2=0.01963 lam=0.5168 k=2.2600 rho=-0.487`
  (`rho<=0`), cost `0.0200` [14.6s]
- SSVI: `eta=0.8016 gamma=0.5937 rho=-0.579`, cost `0.0354` [6.0s]
  - per-expiry `theta` (ATM total variance): `0.00150 0.00306 0.00462
   0.00906 0.01606 0.02415`
- Heston: `v0=0.01779 kappa=14.7289 theta=0.03871 sigma=3.0697 rho=-0.690`,
  cost `0.0296` [6.2s]
  - Feller `2*kappa*theta/sigma^2 = 0.121` (VIOLATED)

IV RMSE (vol points):

| expiry      | n   | MSV   | CRL   | SSVI | Heston |
|-------------|-----|-------|-------|------|--------|
| 2026-11-20 | 377 | 9.714  | 0.768 | 1.192 | 0.956 |
| 2026-12-18 | 363 | 7.367  | 0.505 | 0.699 | 0.572 |
| 2027-01-15 | 334 | 6.129  | 0.401 | 0.523 | 0.470 |
| 2027-03-19 | 130 | 6.391  | 0.378 | 0.382 | 0.333 |
| 2027-06-17 | 132 | 4.993  | 0.398 | 0.231 | 0.395 |
| 2027-09-17 | 122 | 4.334  | 0.365 | 0.162 | 0.637 |
| overall    | 1458 | 7.348 | 0.540 | 0.756 | 0.653 |

ATM skew ratio (model/market), `|ln(K/F)| <= 0.05`:

| expiry      | market  | MSV | CRL | SSVI | Heston |
|-------------|---------|-----|-----|------|--------|
| 2026-11-20 | -80.436 | -0.00 | 1.18 | 1.23 | 1.16 |
| 2026-12-18 | -66.329 | 0.00  | 1.21 | 1.24 | 1.23 |
| 2027-01-15 | -58.180 | 0.00  | 1.21 | 1.21 | 1.20 |
| 2027-03-19 | -45.295 | -0.00 | 1.22 | 1.16 | 1.12 |
| 2027-06-17 | -38.041 | -0.00 | 1.17 | 1.07 | 0.97 |
| 2027-09-17 | -33.292 | 0.00  | 1.14 | 1.01 | 0.87 |

## Findings

Corrected 2026-10-29: the Heston rows were refitted after the `heston_cf` branch-cut fix
(commit 09d1e47); the earlier Heston numbers in this note came from a characteristic function
that crossed the branch cut of the complex log at this surface's vol-of-vol (sigma ~ 3).

On a single trading day (2026-10-16) across six monthly expiries, CRL has the lowest overall
vega-weighted IV RMSE (0.54, against 0.64–0.65 for Heston, 0.74–0.76 for SSVI and 7.3–7.6 for
plain MSV). The CRL advantage is largest at the short end: at the 2026-11-20 expiry its IV
RMSE is 0.75–0.77 vs 0.92–0.96 (Heston) and 1.15–1.19 (SSVI); the plain MSV error decays
into the longest expiry but stays far above the other three.

SSVI overtakes CRL on the two longest expiries (IV RMSE 0.16–0.23 vs 0.37–0.41). Heston beats
CRL at 2027-03-19 (0.33 vs 0.38–0.39) and ties it at 2027-06-17 (0.40–0.41 vs 0.40–0.41),
and is worst of the three at the longest expiry (0.64–0.65). Its Feller ratio is 0.12, well
below 1.

For the ATM skew, CRL and SSVI both reproduce roughly 1.15–1.23x the market skew magnitude
at the short end, tightening toward ~1.0–1.2x at the longest expiry; Heston matches them at
the short end (1.14–1.23x) and falls below the market at the two longest expiries
(0.87–0.97x). This is a single-day snapshot with monthly expiries only (no weeklies), so
short-maturity behaviour is under-sampled, and results are not evidence of out-of-sample
performance.

## Caveats

Single day (2026-10-16) and monthly expiries only; no weeklies at or below
one month are included, so the deep-OTM short-end fit is not stressed here.
Fits are in-sample on the vega-weighted price residuals; no temporal
validation. Heston's Feller violation is reported, not imposed, and its IV
RMSE is on the same OTM fit set as the others.
