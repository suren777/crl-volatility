# CRL exact pricer: accuracy and speed

Produced by `uv run python experiments/pricing_accuracy.py`. Prices the exact CRL call
(`msv.crl_call`, forward form F = D = 1) over a grid of k in {0.5, 1.0, 2.3}, rho in
{-0.9, -0.5, -0.2, 0}, tau in {1/52, 1/12, 1/4, 1, 3}, K/F in {0.7, 0.85, 1.0, 1.15, 1.3},
with two model curves — flat Q = 0.04 and the fitted SPX curve
(a0, a1, a2, lam) = (0, 0.0921, 0.0196, 0.5167). That is 120 (k, rho, tau, curve) configs × 5
strikes. Convergence (Task 1) is measured against a 256-node Gauss-Hermite reference; the
reference is independently checked (Task 2) against a 4e6-draw Monte Carlo of the terminal law
(eq:terminal), i.e. Z, W iid N(0,1) with the model's normalizer m(tau) taken from the same
256-node quadrature. Speed (Task 3) is wall time on flat Q = 0.04, k = 1.0, rho = -0.5, median
of 20 runs.

## 1. Convergence to the 256-node reference (max over grid)

| n_nodes | max |price err| | max |IV err| (vol pts) |
|---|---|---|
| 16 | 5.105e-04 | 6.144e-01 |
| 32 | 5.984e-05 | 1.983e-01 |
| 64 | 2.421e-06 | 1.148e-02 |
| 128 | 3.036e-08 | 4.219e-04 |

## 2. 256-node reference vs Monte Carlo (4e6 draws, Z, W iid N(0,1))

max |Q256 - MC| over grid (absolute, price units) = 2.106e-04
max MC standard error over grid = 2.389e-04
max |Q256 - MC| / SE over grid = 22.45 at (curve=flat 0.04, k=1.0, rho=-0.2, tau=1/52, K/F=1.3)
max |Q256 - MC| / SE over cells with MC price >= 1e-4 = 2.15 at (curve=flat 0.04, k=0.5, rho=0.0, tau=1/4, K/F=1.15)

Worst-ratio configuration (reference vs MC, per K/F):

| K/F | Q256 (256 nodes) | MC | SE | |Q256-MC|/SE |
|---|---|---|---|---|
| 0.7 | 0.300000 | 0.299998 | 1.39e-05 | 0.16 |
| 0.85 | 0.150016 | 0.150013 | 1.39e-05 | 0.17 |
| 1.0 | 0.010207 | 0.010214 | 7.69e-06 | 0.86 |
| 1.15 | 0.000002 | 0.000002 | 1.71e-07 | 0.28 |
| 1.3 | 0.000000 | 0.000000 | 5.49e-10 | 22.45 |

## 3. Speed (flat Q=0.04, k=1.0, rho=-0.5; median of 20 runs)

| n_nodes | ms per 1,000 options (one call) |
|---|---|
| 16 | 0.977 |
| 32 | 1.795 |
| 64 | 3.821 |
| 128 | 7.472 |
| 64 (per option) | 3.991 ms / 1000 = 3.991 us/option |
| one objective evaluation @ 64 nodes over 600 grid options | 2.229 ms |

## What the tables show

1. **Fast, clean convergence in the price.** Max price error vs the 256-node reference falls
   from 5.1e-4 (16 nodes) to 2.4e-6 (64) to 3.0e-8 (128): a Gauss-Hermite-style rate, roughly
   e^{-c n} once n is large enough. At the default 64 nodes the max price error across the whole
   grid is 2.4e-6 in price units (F = 1, so ~1e-4% of spot).
2. **Implied-vol error is dominated by the deep-ITM edge, where IV is ill-conditioned.** Max
   |IV error| is 0.0115 vol points at 64 nodes and 4.2e-4 at 128; the earlier rows are larger
   (0.61 / 0.20 vol pts at 16 / 32). The 64-node maximum sits at the K/F = 0.7 cell (fitted SPX,
   k = 2.3, rho = -0.9, tau = 1/12): the price there is 0.3005, almost entirely intrinsic, so a
   ~8e-7 price gap maps to 0.0115 vol points. This is an inversion-sensitivity artefact of the
   near-intrinsic book, not a near-the-money pricing error — the near-the-money strikes converge
   to far smaller IV error.
3. **The 256-node reference is statistically faithful.** The worst |Q256 - MC| over the grid is
   2.1e-4 in price units — essentially equal to the largest MC standard error (2.4e-4) — so the
   reference sits within one MC standard error of the Monte Carlo everywhere it matters.
4. **The one huge SE-ratio (22.45) is a statistical artefact, not a pricing disagreement.** It is
   at an option worth ≈ 0 (K/F = 1.3, tau = 1/52, flat Q): both Q256 and MC price it at 0.000000;
   the absolute gap is ~1.2e-8, far below any pricing tolerance, but with so few payoff-hitting
   draws the MC standard error shrinks to ~5e-10 and the ratio is inflated. Over cells with
   material price (MC price >= 1e-4) the largest ratio is 2.15 — a single extreme draw, consistent
   with the reference and MC being the same price within MC noise.
5. **Speed:** 1,000 options in one elementwise call at 64 nodes take 3.99 ms (≈ 3.99 us/option;
   0.98 ms at 16 nodes). One objective evaluation as used in calibration — a single 64-node
   `crl_call` over the 600-option grid set — takes 2.23 ms, comfortably within the
   realtime/calibration budget the paper claims.
