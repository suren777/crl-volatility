# Sobol multi-start distribution of CRL optima, and parameter SEs, on SPX

Script: `experiments/multistart_se.py`. For each available Cboe SQLite snapshot
(`spx.sqlite`, root `SPX`) it fits CRL (ρ ≤ 0) jointly over the six monthly expiries from
20 seeded Sobol starting points over the CRL box, with the same vega-weighted residuals as
`crl_short.fit_crl_nonpos`. The Sobol box fixes a0, a1, a2 ∈ [0, hi_a] (hi_a = 5·long-end ATM
variance), λ ∈ [0.05, 5], k ∈ [0.05, 3], ρ ∈ [−0.99, 0]; each start is a full `least_squares`
with the same bounds as the existing CRL fit. τ is computed from each trade date (the
fitters' 2026-10-16 ASOF-derived τ is not reused). At the best optimum the Jacobian of the
residuals is computed by central differences and parameter standard errors are taken as
sqrt(diag((JᵀJ)⁻¹)·σ²) with σ² = RSS/(n − p), residuals treated as approximate implied-vol
errors (units: vol²). Trade dates available: 2026-10-16, 2026-10-19.

## 2026-10-16 (spot 7659.44, n_resid 1458)

Per-start results (20 Sobol starts; `*` = within 1e−6 relative of best cost):

| start | cost | a0 | a1 | a2 | λ | k | ρ |
|---|---|---|---|---|---|---|---|
| 0 | 9.114e−02 | 0.00000 | 0.00000 | 0.02490 | 34.27 | 1.9397 | −0.510 |
| 1 | 9.114e−02 | 0.00000 | 0.00000 | 0.02490 | 28.31 | 1.9397 | −0.510 |
| 2 | 2.005e−02 * | 0.00000 | 0.09208 | 0.01963 | 0.517 | 2.2600 | −0.487 |
| 3 | 9.114e−02 | 0.00000 | 0.00000 | 0.02490 | 46.76 | 1.9397 | −0.510 |
| 4 | 2.005e−02 * | 0.00000 | 0.09208 | 0.01963 | 0.517 | 2.2600 | −0.487 |
| 5 | 9.114e−02 | 0.00000 | 0.00000 | 0.02490 | 46.16 | 1.9397 | −0.510 |
| 6 | 2.005e−02 * | 0.00000 | 0.09208 | 0.01963 | 0.517 | 2.2600 | −0.487 |
| 7 | 2.005e−02 * | 0.00000 | 0.09207 | 0.01963 | 0.517 | 2.2600 | −0.487 |
| 8 | 9.114e−02 | 0.00000 | 0.00000 | 0.02490 | 46.41 | 1.9397 | −0.510 |
| 9 | 9.114e−02 | 0.00000 | 0.00000 | 0.02490 | 10.86 | 1.9397 | −0.510 |
| 10 | 2.005e−02 * | 0.00000 | 0.09208 | 0.01963 | 0.517 | 2.2600 | −0.487 |
| 11 | 9.114e−02 | 0.00000 | 0.00000 | 0.02490 | 43.38 | 1.9397 | −0.510 |
| 12 | 9.114e−02 | 0.00000 | 0.00000 | 0.02490 | 47.64 | 1.9397 | −0.510 |
| 13 | 2.005e−02 * | 0.00000 | 0.09208 | 0.01963 | 0.517 | 2.2600 | −0.487 |
| 14 | 2.005e−02 * | 0.00000 | 0.09208 | 0.01963 | 0.517 | 2.2600 | −0.487 |
| 15 | 2.005e−02 * | 0.00000 | 0.09208 | 0.01963 | 0.517 | 2.2600 | −0.487 |
| 16 | 2.005e−02 * | 0.00000 | 0.09208 | 0.01963 | 0.517 | 2.2600 | −0.487 |
| 17 | 9.114e−02 | 0.00000 | 0.00000 | 0.02490 | 46.53 | 1.9397 | −0.510 |
| 18 | 2.005e−02 * | 0.00000 | 0.09208 | 0.01963 | 0.517 | 2.2600 | −0.487 |
| 19 | 9.114e−02 | 0.00000 | 0.00000 | 0.02490 | 47.59 | 1.9397 | −0.510 |

**10 of 20 starts reach the best cost (2.005e−02) within 1e−6 relative.** Cost distribution:
min 2.005e−02, median 5.56e−02, max 9.114e−02.

Distinct optima (clustered by cost, relative tol 1e−3):

| cluster | n | cost | a0 | a1 | a2 | λ | k | ρ |
|---|---|---|---|---|---|---|---|---|
| 0 (best) | 10 | 2.005e−02 | 0.00000 | 0.09208 | 0.01963 | 0.517 | 2.2600 | −0.487 |
| 1 (local) | 10 | 9.114e−02 | 0.00000 | 0.00000 | 0.02490 | ~34 | 1.9397 | −0.510 |

Parameter standard errors, σ² = RSS/(n−p) = 2.762e−05 (vol²), residual sd 0.53 vol pts,
cond(J) = 8.58e+04:

| param | s.e. |
|---|---|
| a0 | 8.16e−01 |
| a1 | 1.29e+00 |
| a2 | 8.16e−01 |
| λ | 2.73e+00 |
| k | 2.08e−02 |
| ρ | 2.69e−03 |

Correlation matrix of (JᵀJ)⁻¹·σ²:

| | a0 | a1 | a2 | λ | k | ρ |
|---|---|---|---|---|---|---|
| a0 | 1.00 | 1.00 | −1.00 | −1.00 | −0.10 | −0.08 |
| a1 | 1.00 | 1.00 | −1.00 | −1.00 | −0.10 | −0.08 |
| a2 | −1.00 | −1.00 | 1.00 | 1.00 | 0.10 | 0.08 |
| λ | −1.00 | −1.00 | 1.00 | 1.00 | 0.11 | 0.08 |
| k | −0.10 | −0.10 | 0.10 | 0.11 | 1.00 | 0.67 |
| ρ | −0.08 | −0.08 | 0.08 | 0.08 | 0.67 | 1.00 |

## 2026-10-19 (spot 7690.73, n_resid 1449)

Per-start results (20 Sobol starts; `*` = within 1e−6 relative of best cost):

| start | cost | a0 | a1 | a2 | λ | k | ρ |
|---|---|---|---|---|---|---|---|
| 0 | 8.240e−02 | 0.00000 | 0.00000 | 0.02609 | 40.69 | 2.0058 | −0.509 |
| 1 | 8.240e−02 | 0.00000 | 0.00000 | 0.02609 | 47.31 | 2.0058 | −0.509 |
| 2 | 2.264e−02 * | 0.02173 | 0.19942 | 0.00000 | 0.209 | 2.2959 | −0.491 |
| 3 | 8.240e−02 | 0.00000 | 0.00000 | 0.02609 | 46.76 | 2.0058 | −0.509 |
| 4 | 2.264e−02 * | 0.02173 | 0.19937 | 0.00000 | 0.209 | 2.2959 | −0.491 |
| 5 | 8.240e−02 | 0.00000 | 0.00000 | 0.02609 | 46.16 | 2.0058 | −0.509 |
| 6 | 2.264e−02 * | 0.02173 | 0.19937 | 0.00000 | 0.209 | 2.2959 | −0.491 |
| 7 | 8.240e−02 | 0.00000 | 0.00000 | 0.02609 | 45.37 | 2.0058 | −0.509 |
| 8 | 8.240e−02 | 0.00000 | 0.00000 | 0.02609 | 17.38 | 2.0058 | −0.509 |
| 9 | 8.240e−02 | 0.00000 | 0.00000 | 0.02609 | 20.01 | 2.0058 | −0.509 |
| 10 | 2.264e−02 * | 0.02173 | 0.19942 | 0.00000 | 0.209 | 2.2959 | −0.491 |
| 11 | 8.240e−02 | 0.00000 | 0.00000 | 0.02609 | 30.14 | 2.0058 | −0.509 |
| 12 | 8.240e−02 | 0.00000 | 0.00000 | 0.02609 | 48.23 | 2.0058 | −0.509 |
| 13 | 2.264e−02 * | 0.02173 | 0.19940 | 0.00000 | 0.209 | 2.2959 | −0.491 |
| 14 | 2.264e−02 * | 0.02173 | 0.19955 | 0.00000 | 0.209 | 2.2959 | −0.491 |
| 15 | 2.264e−02 * | 0.02173 | 0.19941 | 0.00000 | 0.209 | 2.2959 | −0.491 |
| 16 | 2.264e−02 * | 0.02173 | 0.19933 | 0.00000 | 0.209 | 2.2959 | −0.491 |
| 17 | 8.240e−02 | 0.00000 | 0.00000 | 0.02609 | 46.53 | 2.0058 | −0.509 |
| 18 | 2.264e−02 * | 0.02172 | 0.19938 | 0.00000 | 0.209 | 2.2959 | −0.491 |
| 19 | 8.240e−02 | 0.00000 | 0.00000 | 0.02609 | 42.75 | 2.0058 | −0.509 |

**9 of 20 starts reach the best cost (2.264e−02) within 1e−6 relative.** Cost distribution:
min 2.264e−02, median 8.240e−02, max 8.240e−02.

Distinct optima (clustered by cost, relative tol 1e−3):

| cluster | n | cost | a0 | a1 | a2 | λ | k | ρ |
|---|---|---|---|---|---|---|---|---|
| 0 (best) | 9 | 2.264e−02 | 0.02173 | 0.19941 | 0.00000 | 0.209 | 2.2959 | −0.491 |
| 1 (local) | 11 | 8.240e−02 | 0.00000 | 0.00000 | 0.02609 | ~30–48 | 2.0058 | −0.509 |

Parameter standard errors, σ² = RSS/(n−p) = 3.138e−05 (vol²), residual sd 0.56 vol pts,
cond(J) = 6.78e+05:

| param | s.e. |
|---|---|
| a0 | 1.30e+01 |
| a1 | 1.92e+01 |
| a2 | 1.30e+01 |
| λ | 7.30e+00 |
| k | 2.23e−02 |
| ρ | 2.86e−03 |

Correlation matrix of (JᵀJ)⁻¹·σ²:

| | a0 | a1 | a2 | λ | k | ρ |
|---|---|---|---|---|---|---|
| a0 | 1.00 | 1.00 | −1.00 | −1.00 | −0.11 | −0.08 |
| a1 | 1.00 | 1.00 | −1.00 | −1.00 | −0.11 | −0.08 |
| a2 | −1.00 | −1.00 | 1.00 | 1.00 | 0.11 | 0.08 |
| λ | −1.00 | −1.00 | 1.00 | 1.00 | 0.11 | 0.08 |
| k | −0.11 | −0.11 | 0.11 | 0.11 | 1.00 | 0.68 |
| ρ | −0.08 | −0.08 | 0.08 | 0.08 | 0.68 | 1.00 |

## Findings

- Under 20 seeded Sobol starts the six-parameter CRL fit on 2026-10-16 has a clearly
  bimodal distribution of optima: ~half the starts land in a good basin (cost 2.0e−02,
  a1 = 0.092, λ = 0.52, k = 2.26, ρ = −0.49) and the other half converge to a poorer local
  optimum (cost 9.1e−02, a1 = 0, λ pushed to ~34, k = 1.94, ρ = −0.51). The same two-level
  structure appears on 2026-10-19 (good cost 2.26e−02 vs local 8.24e−02), so the objective
  is not unimodal and a single downhill start is not a reliable global-fit certificate.
- The poorer optimum on both days has a1 → 0 and a2 ≈ 0.025–0.026 with λ large, leaving
  Q(τ) ≈ a2 constant — effectively Black–Scholes plus a level — and costs ~4× the CRL
  optimum, so the correlated random level (k ≳ 2, ρ ≈ −0.5) never falls out: CRL beats the
  no-skew solution by a wide margin from any basin.
- Wherever a start reaches the good basin, the skew-setting parameters are tightly
  reproduced: k = 2.2600 and ρ = −0.487 on 2026-10-16, k = 2.2959 and ρ = −0.491 on
  2026-10-19, stable to 1e−4 across starts.
- The best 2026-10-16 optimum (a0 = 0, a1 = 0.09208, a2 = 0.01963, λ = 0.5168,
  k = 2.2600, ρ = −0.487) is exactly the optimum `crl_short.py` / `crl_spx.py` and
  `identifiability.md` report, and its JᵀJ SEs (a0 0.82, a1 1.29, λ 2.73, k 0.0208,
  ρ 2.69e−3; cond 8.6e4) reproduce those earlier values; the 2026-10-19 best
  (a0 = 0.02173, a1 = 0.19941, a2 = 0, λ = 0.209, k = 2.2959, ρ = −0.491) has the
  a1-heavy profile of the yfinance-column optimum in `identifiability.md`, with large
  (a0, a1, a2, λ) SEs (13–19) and cond 6.8e5.
- On both days the (a0, a1, a2, λ) block is mutually ±1-correlated in (JᵀJ)⁻¹ (a0, a1 up
  against a2, λ down) while k and ρ are only ~0.67-correlated with each other and nearly
  uncorrelated with the level parameters — the same flat (a0, a1, λ) direction that
  `identifiability.md` documents.
- Under the σ² = RSS/(n−p) noise model residuals are ≈ implied-vol errors in vol units;
  residual sd is 0.53 vol pts (2026-10-16) and 0.56 vol pts (2026-10-19), and k, ρ carry
  tight s.e. (~0.02 and ~3e−3) while a0, a1, a2, λ are poorly determined.

## Caveats

- The Sobol box upper end for a0, a1, a2 (hi_a = 5·long-end ATM variance ≈ 0.13) is a
  pragmatic choice from the ATM variance; refitting to the (large-λ, a2-only) branch is a
  genuine local optimum, not a clipping artifact, because λ stayed below the 50 upper bound
  on all starts.
- A Cluster is defined here by a ≥1e−3 relative cost separation; the good basin's starts
  differ in cost by <1e−6, so they form one cluster even though (a0, a1, λ) moves along
  the flat direction between them.
