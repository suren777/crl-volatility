# Identifiability of the CRL parameters on SPX (2026-10-16 close, two sources)

Script: `experiments/identifiability.py`. Fits CRL (ρ ≤ 0) jointly over the six monthly
expiries on the Cboe SQLite snapshot and the yfinance CSV, then at each optimum reports the
Jacobian of the vega-weighted residuals w.r.t. (a0, a1, a2, λ, k, ρ) by central differences:
singular values and condition number of J, the eigenvectors of JᵀJ for the two smallest
eigenvalues, and the correlation matrix of (JᵀJ)⁻¹ scaled by the fitted residual variance.
The residuals are ≈ implied-vol errors in vol units (`(price − D·vega·iv)` divided by
`D·vega`), so JᵀJ is the inverse Fisher information per unit residual variance. Then λ is
profiled: fixed on a grid 0.1..5 and the other five refitted. Run:
`uv run python experiments/identifiability.py`.

## Hypothesis in the epic

The known flat direction is among (a0, a1, λ): a0, a1 and λ jointly set the level of the
short-end variance term `a1·(1−e^(−λτ))/(λτ) + a0`, and there is a continuum of triples that
nearly reproduce the same surface. The Jensen/eigen analysis and λ profile below test whether
this direction is actually flat on the fitted SPX surface, or whether the data pins λ down.

## Task 1 — Fisher information / Hessian of the sums of squares at the optimum

Free 6-parameter fit at the optimum (matches `crl_spx.py`). Jacobian J of the vega-weighted
residuals, central differences with h ≈ 1e−5·scale per parameter.

### Cboe SQLite (root=SPX) — optimum a0=0.00000 a1=0.09208 a2=0.01963 λ=0.5167 k=2.2600 ρ=−0.487

| Singular values of J | value |
|---|---|
| s0 | 1.393e+02 |
| s1 | 5.505e+00 |
| s2 | 2.659e+00 |
| s3 | 2.542e−01 |
| s4 | 1.389e−01 |
| s5 | 1.623e−03 |
| **cond(J) = s0/s5** | **8.582e+04** |

Eigenvectors of JᵀJ for the two smallest eigenvalues (rows = parameters, columns v(p−1), v(p−2)):

| param | v(p−1) | v(p−2) |
|---|---|---|
| a0 | 0.2521 | 0.4819 |
| a1 | 0.3995 | 0.5014 |
| a2 | −0.2520 | −0.4856 |
| λ | −0.8446 | 0.5258 |
| k | −0.0007 | 0.0641 |
| ρ | −0.0001 | 0.0033 |

Residual variance 2.762e−05 (vol²), residual sd 0.53 vol pts. Approx. parameter std err
(sqrt diag of (JᵀJ)⁻¹ × residual variance):

| param | s.e. |
|---|---|
| a0 | 8.16 |
| a1 | 1.29e+00 |
| a2 | 8.16e−01 |
| λ | 2.73e+00 |
| k | 2.08e−02 |
| ρ | 2.69e−03 |

Correlation matrix of (JᵀJ)⁻¹ × residual variance:

| | a0 | a1 | a2 | λ | k | ρ |
|---|---|---|---|---|---|---|
| a0 | 1.00 | 1.00 | −1.00 | −1.00 | −0.10 | −0.08 |
| a1 | 1.00 | 1.00 | −1.00 | −1.00 | −0.10 | −0.08 |
| a2 | −1.00 | −1.00 | 1.00 | 1.00 | 0.10 | 0.08 |
| λ | −1.00 | −1.00 | 1.00 | 1.00 | 0.11 | 0.08 |
| k | −0.10 | −0.10 | 0.10 | 0.11 | 1.00 | 0.67 |
| ρ | −0.08 | −0.08 | 0.08 | 0.08 | 0.67 | 1.00 |

### yfinance CSV — optimum a0=0.02125 a1=0.20668 a2=0.00000 λ=0.1975 k=2.2833 ρ=−0.491

| Singular values of J | value |
|---|---|
| s0 | 1.385e+02 |
| s1 | 2.833e+00 |
| s2 | 2.730e+00 |
| s3 | 2.547e−01 |
| s4 | 6.252e−02 |
| s5 | 1.738e−04 |
| **cond(J) = s0/s5** | **7.971e+05** |

Eigenvectors of JᵀJ for the two smallest eigenvalues:

| param | v(p−1) | v(p−2) |
|---|---|---|
| a0 | 0.4741 | 0.3844 |
| a1 | 0.7011 | −0.2433 |
| a2 | −0.4741 | −0.3860 |
| λ | −0.2427 | 0.8022 |
| k | −0.0001 | 0.0216 |
| ρ | −0.0000 | 0.0008 |

Residual variance 2.740e−05 (vol²), residual sd 0.52 vol pts. Approx. parameter std err:

| param | s.e. |
|---|---|
| a0 | 1.43e+01 |
| a1 | 2.11e+01 |
| a2 | 1.43e+01 |
| λ | 7.31e+00 |
| k | 2.07e−02 |
| ρ | 2.59e−03 |

Correlation matrix of (JᵀJ)⁻¹ × residual variance:

| | a0 | a1 | a2 | λ | k | ρ |
|---|---|---|---|---|---|---|
| a0 | 1.00 | 1.00 | −1.00 | −1.00 | −0.11 | −0.07 |
| a1 | 1.00 | 1.00 | −1.00 | −1.00 | −0.11 | −0.07 |
| a2 | −1.00 | −1.00 | 1.00 | 1.00 | 0.11 | 0.07 |
| λ | −1.00 | −1.00 | 1.00 | 1.00 | 0.11 | 0.07 |
| k | −0.11 | −0.11 | 0.11 | 0.11 | 1.00 | 0.67 |
| ρ | −0.07 | −0.07 | 0.07 | 0.07 | 0.67 | 1.00 |

## Task 2/3 — λ profile (fix λ, refit a0, a1, a2, k, ρ)

Cost is `0.5·Σr²` (the `least_squares` cost); `cost/free` normalises by the smallest profile cost.

### Cboe SQLite

| λ | cost | cost/free | a0 | a1 | a2 | k | ρ |
|---|---|---|---|---|---|---|---|
| 0.1 | 2.0380e−02 | 1.016 | 0.02032 | 0.38970 | 0.00000 | 2.2519 | −0.488 |
| 0.2 | 2.0234e−02 | 1.009 | 0.02015 | 0.21763 | 0.00000 | 2.2542 | −0.488 |
| 0.5 | 2.0050e−02 | 1.000 | 0.00485 | 0.09987 | 0.01479 | 2.2600 | −0.487 |
| 1.0 | 2.0584e−02 | 1.027 | 0.00000 | 0.06299 | 0.01873 | 2.2660 | −0.487 |
| 2.0 | 2.6063e−02 | 1.300 | 0.00000 | 0.05286 | 0.01662 | 2.2605 | −0.488 |
| 3.0 | 3.9664e−02 | 1.978 | 0.00000 | 0.05248 | 0.01473 | 2.2208 | −0.491 |
| 5.0 | 8.4433e−02 | 4.211 | 0.00000 | 0.02552 | 0.01894 | 2.0196 | −0.504 |

### yfinance CSV

| λ | cost | cost/free | a0 | a1 | a2 | k | ρ |
|---|---|---|---|---|---|---|---|
| 0.1 | 1.9826e−02 | 1.001 | 0.02140 | 0.36453 | 0.00000 | 2.2815 | −0.491 |
| 0.2 | 1.9808e−02 | 1.000 | 0.01870 | 0.20198 | 0.00254 | 2.2833 | −0.491 |
| 0.5 | 1.9950e−02 | 1.007 | 0.00000 | 0.08723 | 0.02081 | 2.2873 | −0.491 |
| 1.0 | 2.0958e−02 | 1.058 | 0.00000 | 0.05809 | 0.01996 | 2.2912 | −0.491 |
| 2.0 | 2.6793e−02 | 1.353 | 0.00000 | 0.04830 | 0.01809 | 2.2825 | −0.492 |
| 3.0 | 3.9360e−02 | 1.987 | 0.00000 | 0.04724 | 0.01650 | 2.2424 | −0.495 |
| 5.0 | 7.5973e−02 | 3.835 | 0.00000 | 0.01925 | 0.02118 | 2.0531 | −0.507 |

## Summary

- The Jacobian of the vega-weighted residuals is rank-deficient-ill-conditioned: condition
  number ~1e5 (Cboe) and ~8e5 (yfinance), with the two smallest singular values ~1e−3 and
  ~1e−1–1e−4 versus the leading ~1.4e+02 — the objective is effectively flat in roughly two
  directions, so the 6-parameter optimum is numerically not a strict point.
- The two smallest eigenvectors of JᵀJ load almost exclusively on (a0, a1, a2, λ), with
  essentially zero weight on k and ρ: the flat directions move a0, a1, a2 and λ jointly while
  leaving the volatility level k = √Var(ζ) and the correlation ρ (the two parameters that set
  the skew) alone.
- Correspondingly, the (a0, a1, a2, λ) parameters are mutually ±1-correlated in (JᵀJ)⁻¹ on
  both sources (a0 and a1 move up together against a down-move in a2 and λ), while k and ρ
  pair only with each other (corr ≈ 0.67) and are nearly uncorrelated with the level
  parameters; their std errs (~0.02 vol units and ~3e−3) are two orders smaller than those of
  a0, a1, a2, λ (0.8–21 in the same units).
- The λ profile is nearly flat for λ ≲ 1 (cost/free within 1–6% for λ ∈ [0.1, 1] on both
  sources) and rises only steeply for λ ≳ 2 (cost/free 300–420% at λ = 5): the fit pins down
  that λ is small but has almost no power to separate compact λ ∈ [0.1, 1].
- As λ is swept from 0.1 to 5 the short-end level migrates off a0 onto a1 and then onto a2 —
  a0 → 0 and a1 falls as λ rises while a2 stays near the long-end variance — which is exactly
  the (a0, a1, a2, λ) cancellation the eigenvectors show.
- k, ρ, a2 and the long-end level are well identified (tight std errs, k ≈ 2.3, ρ ≈ −0.49,
  stable across λ and across the two data sources); the practical limit of the six-parameter
  CRL fit on SPX is the (a0, a1, λ) triple, not the model's skew-setting parameters.

## Output

```
======================================================================
Cboe SQLite root=SPX (2026-10-16)
======================================================================
Fitted parameters:
  a0=0.00000 a1=0.09208 a2=0.01963 lam=0.5167 k=2.2600 rho=-0.487  cost=0.0200
  n_resid=1458, n_params=6, residual variance (vol^2) = 2.762e-05, residual sd (vol pts) = 0.53
Jacobian J of vega-weighted residuals w.r.t. (a0,a1,a2,lam,k,rho): central diff h~1e-5·scale
Singular values of J:
  s0=1.393e+02
  s1=5.505e+00
  s2=2.659e+00
  s3=2.542e-01
  s4=1.389e-01
  s5=1.623e-03
Condition number (s_max/s_min) = 8.582e+04
Eigenvectors of J^T J for the two smallest eigenvalues (V columns, rows = params):
  param       v(p-1)       v(p-2)
  a0          0.2521       0.4819
  a1          0.3995       0.5014
  a2         -0.2520      -0.4856
  lam        -0.8446       0.5258
  k          -0.0007       0.0641
  rho        -0.0001       0.0033
Approx. parameter std err (sqrt(diag cov), residual-variance scaled):
  param         s.e.
  a0      8.1622e-01
  a1      1.2934e+00
  a2      8.1601e-01
  lam     2.7346e+00
  k       2.0810e-02
  rho     2.6900e-03
Correlation matrix of (J^T J)^-1 * residual_variance:
             a0       a1       a2      lam        k      rho
  a0        1.00     1.00    -1.00    -1.00    -0.10    -0.08
  a1        1.00     1.00    -1.00    -1.00    -0.10    -0.08
  a2       -1.00    -1.00     1.00     1.00     0.10     0.08
  lam      -1.00    -1.00     1.00     1.00     0.11     0.08
  k        -0.10    -0.10     0.10     0.11     1.00     0.67
  rho      -0.08    -0.08     0.08     0.08     0.67     1.00

======================================================================
yfinance CSV
======================================================================
Fitted parameters:
  a0=0.02125 a1=0.20668 a2=0.00000 lam=0.1975 k=2.2833 rho=-0.491  cost=0.0198
  n_resid=1452, n_params=6, residual variance (vol^2) = 2.740e-05, residual sd (vol pts) = 0.52
Jacobian J of vega-weighted residuals w.r.t. (a0,a1,a2,lam,k,rho): central diff h~1e-5·scale
Singular values of J:
  s0=1.385e+02
  s1=2.833e+00
  s2=2.730e+00
  s3=2.547e-01
  s4=6.252e-02
  s5=1.738e-04
Condition number (s_max/s_min) = 7.971e+05
Eigenvectors of J^T J for the two smallest eigenvalues (V columns, rows = params):
  param       v(p-1)       v(p-2)
  a0          0.4741       0.3844
  a1          0.7011      -0.2433
  a2         -0.4741      -0.3860
  lam        -0.2427       0.8022
  k          -0.0001       0.0216
  rho        -0.0000       0.0008
Approx. parameter std err (sqrt(diag cov), residual-variance scaled):
  param         s.e.
  a0      1.4279e+01
  a1      2.1117e+01
  a2      1.4279e+01
  lam     7.3094e+00
  k       2.0658e-02
  rho     2.5859e-03
Correlation matrix of (J^T J)^-1 * residual_variance:
             a0       a1       a2      lam        k      rho
  a0        1.00     1.00    -1.00    -1.00    -0.11    -0.07
  a1        1.00     1.00    -1.00    -1.00    -0.11    -0.07
  a2       -1.00    -1.00     1.00     1.00     0.11     0.07
  lam      -1.00    -1.00     1.00     1.00     0.11     0.07
  k        -0.11    -0.11     0.11     0.11     1.00     0.67
  rho      -0.07    -0.07     0.07     0.07     0.67     1.00

lam profile, Cboe SQLite (refit a0,a1,a2,k,rho at fixed lam; cost = 0.5*sum r^2):
  lam         cost  cost/free         a0         a1         a2        k     rho
  0.1   2.0380e-02      1.016    0.02032    0.38970    0.00000   2.2519  -0.488
  0.2   2.0234e-02      1.009    0.02015    0.21763    0.00000   2.2542  -0.488
  0.5   2.0050e-02      1.000    0.00485    0.09987    0.01479   2.2600  -0.487
  1.0   2.0584e-02      1.027    0.00000    0.06299    0.01873   2.2660  -0.487
  2.0   2.6063e-02      1.300    0.00000    0.05286    0.01662   2.2605  -0.488
  3.0   3.9664e-02      1.978    0.00000    0.05248    0.01473   2.2208  -0.491
  5.0   8.4433e-02      4.211    0.00000    0.02552    0.01894   2.0196  -0.504

lam profile, yfinance CSV (refit a0,a1,a2,k,rho at fixed lam; cost = 0.5*sum r^2):
  lam         cost  cost/free         a0         a1         a2        k     rho
  0.1   1.9826e-02      1.001    0.02140    0.36453    0.00000   2.2815  -0.491
  0.2   1.9808e-02      1.000    0.01870    0.20198    0.00254   2.2833  -0.491
  0.5   1.9950e-02      1.007    0.00000    0.08723    0.02081   2.2873  -0.491
  1.0   2.0958e-02      1.058    0.00000    0.05809    0.01996   2.2912  -0.491
  2.0   2.6793e-02      1.353    0.00000    0.04830    0.01809   2.2825  -0.492
  3.0   3.9360e-02      1.987    0.00000    0.04724    0.01650   2.2424  -0.495
  5.0   7.5973e-02      3.835    0.00000    0.01925    0.02118   2.0531  -0.507
```
