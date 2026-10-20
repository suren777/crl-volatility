# Day-to-day stability panel of the fitted models (preview)

> **Preview, not a panel.** Only 2 trade date(s) are in the collector snapshot cache. Earlier scripts, benchmarks and the CRL short-end work used `2026-10-16`; this panel grows by one row per model per date as the collector adds trading days. Fewer than ~5-10 dates support only a provisional read of day-to-day stability.

For each trade date in `spx.sqlite`, six SPX monthly expiries in [3 weeks, 12 months] from that date (the paper's set: Nov/Dec/Jan then quarterly Mar/Jun/Sep, backfilled to six by the nearest remaining monthlies) are fitted jointly for plain MSV, CRL (`rho <= 0`), SSVI and Heston on the same OTM, vega-weighted price objective as `benchmarks_spx.py`. tau is measured from each trade date; `F`/`D` from put-call parity. All prices are Cboe SPX OTM mids; IV RMSE is in vol points. Derived numbers only are stored in `panel.csv`.

## 2026-10-16

Fitted parameters (joint over six expiries), cost and fit time:

| model | parameters | cost | time (s) | IV RMSE (vol pts) | n |
|-------|------------|------|----------|-------------------|-----|
| msv      | a0=0.00000 a1=0.14763 a2=0.01560 lam=0.2252 k=0.3603 | 1.2585 | 2.0 | 7.348 | 1458 |
| crl      | a0=0.00000 a1=0.09208 a2=0.01963 lam=0.5167 k=2.2600 rho=-0.487 (rho<=0)  AT_BOUND=a0 | 0.0200 | 10.8 | 0.540 | 1458 |
| ssvi     | eta=0.8016 gamma=0.5937 rho=-0.579 | 0.0354 | 5.9 | 0.756 | 1458 |
| heston   | v0=0.02242 kappa=2.4940 theta=0.06000 sigma=1.3997 rho=-0.703 feller=0.153 | 0.0663 | 20.1 | 1.153 | 1458 |

Per-expiry IV RMSE (vol points):

| expiry       |    n |     MSV |     CRL |    SSVI |  Heston |
|---|---|---|---|---|---|
| 2026-11-20   |  377 |   9.714 |   0.768 |   1.192 |   1.913 |
| 2026-12-18   |  363 |   7.367 |   0.505 |   0.699 |   0.772 |
| 2027-01-15   |  334 |   6.129 |   0.401 |   0.523 |   0.599 |
| 2027-03-19   |  130 |   6.391 |   0.378 |   0.382 |   0.647 |
| 2027-06-17   |  132 |   4.993 |   0.398 |   0.231 |   0.641 |
| 2027-09-17   |  122 |   4.334 |   0.365 |   0.162 |   0.967 |
| overall      | 1458 | 7.348 | 0.540 | 0.756 | 1.153 |

## 2026-10-19

Fitted parameters (joint over six expiries), cost and fit time:

| model | parameters | cost | time (s) | IV RMSE (vol pts) | n |
|-------|------------|------|----------|-------------------|-----|
| msv      | a0=0.00000 a1=6.43274 a2=0.01711 lam=0.0041 k=0.3606 | 1.3381 | 13.9 | 7.651 | 1449 |
| crl      | a0=0.02173 a1=0.19941 a2=0.00000 lam=0.2089 k=2.2959 rho=-0.491 (rho<=0)  AT_BOUND=a2 | 0.0226 | 20.4 | 0.575 | 1449 |
| ssvi     | eta=0.8031 gamma=0.5939 rho=-0.584 | 0.0400 | 7.2 | 0.806 | 1449 |
| heston   | v0=0.02450 kappa=2.4203 theta=0.06304 sigma=1.4946 rho=-0.710 feller=0.137 | 0.0794 | 27.1 | 1.258 | 1449 |

Per-expiry IV RMSE (vol points):

| expiry       |    n |     MSV |     CRL |    SSVI |  Heston |
|---|---|---|---|---|---|
| 2026-11-20   |  370 |  10.331 |   0.838 |   1.293 |   2.100 |
| 2026-12-18   |  361 |   7.578 |   0.520 |   0.730 |   0.797 |
| 2027-01-15   |  333 |   6.270 |   0.427 |   0.547 |   0.691 |
| 2027-03-19   |  129 |   6.554 |   0.392 |   0.386 |   0.765 |
| 2027-06-17   |  134 |   5.180 |   0.426 |   0.248 |   0.773 |
| 2027-09-17   |  122 |   4.417 |   0.375 |   0.159 |   0.970 |
| overall      | 1449 | 7.651 | 0.575 | 0.806 | 1.258 |

## CRL day-over-day parameter change

| date | a0 (Δ) | a1 (Δ) | a2 (Δ) | lam (Δ) | k (Δ) | rho (Δ) |
|------|--------|--------|--------|---------|-------|---------|
| 2026-10-16 | 0.0000 | 0.0921 | 0.0196 | 0.5167 | 2.2600 | -0.4874 |
| 2026-10-19 | 0.0217 (+0.0217) | 0.1994 (+0.1073) | 0.0000 (-0.0196) | 0.2089 (-0.3078) | 2.2959 (+0.0359) | -0.4905 (-0.0031) |

`(Δ)` is the change from the previous trade date.

## Caveats

- Single/paired trading days only — this is a preview, not a stability panel. The day-over-day CRL deltas over a two-date window say almost nothing yet; revisit once the collector has a few weeks.
- tau is measured per trade date and expiries are re-picked per date, so the near-term rollover is already reflected.
- CRL's rho is bounded to [-0.99, 0]; an optimiser-pinned parameter (see `at_bound`) would indicate an identification issue, not necessarily instability.
- Per-expiry option counts are in `panel.csv` (`n_e`); the md table repeats them through the MSV row's expiry order.

