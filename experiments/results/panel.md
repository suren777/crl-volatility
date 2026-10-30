# Day-to-day stability panel of the fitted models (preview)

For each trade date in `spx.sqlite`, six SPX monthly expiries in [3 weeks, 12 months] from that date (the paper's set: Nov/Dec/Jan then quarterly Mar/Jun/Sep, backfilled to six by the nearest remaining monthlies) are fitted jointly for plain MSV, CRL (`rho <= 0`), SSVI and Heston on the same OTM, vega-weighted price objective as `benchmarks_spx.py`. tau is measured from each trade date; `F`/`D` from put-call parity. All prices are Cboe SPX OTM mids; IV RMSE is in vol points. Derived numbers only are stored in `panel.csv`.

## 2026-10-16

Fitted parameters (joint over six expiries), cost and fit time:

| model | parameters | cost | time (s) | IV RMSE (vol pts) | n |
|-------|------------|------|----------|-------------------|-----|
| msv      | a0=0.00000 a1=0.14763 a2=0.01560 lam=0.2252 k=0.3603 | 1.2585 | 1.9 | 7.348 | 1458 |
| crl      | a0=0.00000 a1=0.09208 a2=0.01963 lam=0.5168 k=2.2600 rho=-0.487 (rho<=0)  AT_BOUND=a0 | 0.0200 | 11.0 | 0.540 | 1458 |
| ssvi     | eta=0.8016 gamma=0.5937 rho=-0.579 | 0.0354 | 5.5 | 0.756 | 1458 |
| heston   | v0=0.01780 kappa=14.7219 theta=0.03871 sigma=3.0682 rho=-0.690 feller=0.121 | 0.0296 | 14.4 | 0.653 | 1458 |

Per-expiry IV RMSE (vol points):

| expiry       |    n |     MSV |     CRL |    SSVI |  Heston |
|---|---|---|---|---|---|
| 2026-11-20   |  377 |   9.714 |   0.768 |   1.192 |   0.957 |
| 2026-12-18   |  363 |   7.367 |   0.505 |   0.699 |   0.572 |
| 2027-01-15   |  334 |   6.129 |   0.401 |   0.523 |   0.470 |
| 2027-03-19   |  130 |   6.391 |   0.378 |   0.382 |   0.333 |
| 2027-06-17   |  132 |   4.993 |   0.398 |   0.231 |   0.395 |
| 2027-09-17   |  122 |   4.334 |   0.365 |   0.162 |   0.637 |
| overall      | 1458 | 7.348 | 0.540 | 0.756 | 0.653 |

## 2026-10-19

Fitted parameters (joint over six expiries), cost and fit time:

| model | parameters | cost | time (s) | IV RMSE (vol pts) | n |
|-------|------------|------|----------|-------------------|-----|
| msv      | a0=0.00000 a1=6.43274 a2=0.01711 lam=0.0041 k=0.3606 | 1.3381 | 13.6 | 7.651 | 1449 |
| crl      | a0=0.02173 a1=0.19941 a2=0.00000 lam=0.2089 k=2.2959 rho=-0.491 (rho<=0)  AT_BOUND=a2 | 0.0226 | 22.3 | 0.575 | 1449 |
| ssvi     | eta=0.8031 gamma=0.5939 rho=-0.584 | 0.0400 | 7.0 | 0.806 | 1449 |
| heston   | v0=0.02153 kappa=16.6610 theta=0.03882 sigma=3.4043 rho=-0.691 feller=0.112 | 0.0333 | 14.7 | 0.693 | 1449 |

Per-expiry IV RMSE (vol points):

| expiry       |    n |     MSV |     CRL |    SSVI |  Heston |
|---|---|---|---|---|---|
| 2026-11-20   |  370 |  10.331 |   0.838 |   1.293 |   1.020 |
| 2026-12-18   |  361 |   7.578 |   0.520 |   0.730 |   0.598 |
| 2027-01-15   |  333 |   6.270 |   0.427 |   0.547 |   0.488 |
| 2027-03-19   |  129 |   6.554 |   0.392 |   0.386 |   0.352 |
| 2027-06-17   |  134 |   5.180 |   0.426 |   0.248 |   0.463 |
| 2027-09-17   |  122 |   4.417 |   0.375 |   0.159 |   0.683 |
| overall      | 1449 | 7.651 | 0.575 | 0.806 | 0.693 |

## 2026-10-20

Fitted parameters (joint over six expiries), cost and fit time:

| model | parameters | cost | time (s) | IV RMSE (vol pts) | n |
|-------|------------|------|----------|-------------------|-----|
| msv      | a0=0.00000 a1=0.18887 a2=0.01651 lam=0.1534 k=0.4107 | 1.1940 | 2.7 | 7.204 | 1458 |
| crl      | a0=0.00000 a1=0.07962 a2=0.01903 lam=0.6827 k=2.1887 rho=-0.473 (rho<=0)  AT_BOUND=a0 | 0.0247 | 10.6 | 0.599 | 1458 |
| ssvi     | eta=0.8871 gamma=0.5732 rho=-0.561 | 0.0444 | 10.4 | 0.861 | 1458 |
| heston   | v0=0.01693 kappa=14.3458 theta=0.03856 sigma=2.9770 rho=-0.669 feller=0.125 | 0.0385 | 14.8 | 0.756 | 1458 |

Per-expiry IV RMSE (vol points):

| expiry       |    n |     MSV |     CRL |    SSVI |  Heston |
|---|---|---|---|---|---|
| 2026-11-20   |  375 |   9.560 |   0.907 |   1.403 |   1.159 |
| 2026-12-18   |  364 |   7.112 |   0.530 |   0.738 |   0.627 |
| 2027-01-15   |  335 |   6.033 |   0.425 |   0.562 |   0.498 |
| 2027-03-19   |  129 |   6.281 |   0.375 |   0.405 |   0.354 |
| 2027-06-17   |  133 |   5.046 |   0.377 |   0.289 |   0.470 |
| 2027-09-17   |  122 |   4.358 |   0.346 |   0.234 |   0.687 |
| overall      | 1458 | 7.204 | 0.599 | 0.861 | 0.756 |

## 2026-10-22

Fitted parameters (joint over six expiries), cost and fit time:

| model | parameters | cost | time (s) | IV RMSE (vol pts) | n |
|-------|------------|------|----------|-------------------|-----|
| msv      | a0=0.00000 a1=17.39030 a2=0.01980 lam=0.0012 k=0.4721 | 1.3752 | 12.6 | 7.313 | 1439 |
| crl      | a0=0.02458 a1=0.30228 a2=0.00000 lam=0.1100 k=2.1532 rho=-0.492 (rho<=0)  AT_BOUND=a2 | 0.0214 | 41.0 | 0.557 | 1439 |
| ssvi     | eta=0.9362 gamma=0.5615 rho=-0.581 | 0.0397 | 5.5 | 0.807 | 1439 |
| heston   | v0=0.02720 kappa=14.5743 theta=0.04041 sigma=3.0726 rho=-0.685 feller=0.125 | 0.0370 | 13.5 | 0.739 | 1439 |

Per-expiry IV RMSE (vol points):

| expiry       |    n |     MSV |     CRL |    SSVI |  Heston |
|---|---|---|---|---|---|
| 2026-11-20   |  363 |   9.840 |   0.843 |   1.320 |   1.144 |
| 2026-12-18   |  358 |   7.185 |   0.497 |   0.704 |   0.608 |
| 2027-01-15   |  332 |   5.990 |   0.414 |   0.528 |   0.457 |
| 2027-03-19   |  129 |   6.413 |   0.354 |   0.388 |   0.377 |
| 2027-06-17   |  134 |   5.193 |   0.348 |   0.265 |   0.506 |
| 2027-09-17   |  123 |   4.509 |   0.305 |   0.209 |   0.676 |
| overall      | 1439 | 7.313 | 0.557 | 0.807 | 0.739 |

## CRL day-over-day parameter change

| date | a0 (Δ) | a1 (Δ) | a2 (Δ) | lam (Δ) | k (Δ) | rho (Δ) |
|------|--------|--------|--------|---------|-------|---------|
| 2026-10-16 | 0.0000 | 0.0921 | 0.0196 | 0.5168 | 2.2600 | -0.4874 |
| 2026-10-19 | 0.0217 (+0.0217) | 0.1994 (+0.1073) | 0.0000 (-0.0196) | 0.2089 (-0.3078) | 2.2959 (+0.0359) | -0.4905 (-0.0031) |
| 2026-10-20 | 0.0000 (-0.0217) | 0.0796 (-0.1198) | 0.0190 (+0.0190) | 0.6827 (+0.4738) | 2.1887 (-0.1072) | -0.4733 (+0.0172) |
| 2026-10-22 | 0.0246 (+0.0246) | 0.3023 (+0.2227) | 0.0000 (-0.0190) | 0.1100 (-0.5727) | 2.1532 (-0.0355) | -0.4921 (-0.0188) |

`(Δ)` is the change from the previous trade date.

## Caveats

- Single/paired trading days only — this is a preview, not a stability panel. The day-over-day CRL deltas over a two-date window say almost nothing yet; revisit once the collector has a few weeks.
- tau is measured per trade date and expiries are re-picked per date, so the near-term rollover is already reflected.
- CRL's rho is bounded to [-0.99, 0]; an optimiser-pinned parameter (see `at_bound`) would indicate an identification issue, not necessarily instability.
- Per-expiry option counts are in `panel.csv` (`n_e`); the md table repeats them through the MSV row's expiry order.

