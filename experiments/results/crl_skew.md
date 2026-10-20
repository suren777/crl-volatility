# CRL model: ATM-skew term structure and calendar check

Produced by `uv run python experiments/crl_skew.py` (crl_call at its default 64
Gauss-Hermite nodes). Skew part: F=100, D=1, flat Q = 0.04 (a0=a1=0, a2=0.04,
lam=1); skew = dIV/dln(K/F) at ln(K/F)=0 by central difference (+/-0.01), IV by
brentq inversion of forward Black. Calendar part: F=1, D=1 held constant,
K/F in [0.6, 1.6] (11 steps), tau in [0.02, 3] (30 points); a violation is a
negative increment C(K, tau_next) - C(K, tau), 319 per configuration.

## ATM skew term structure (flat Q = 0.04)

| (k, rho) | 1/52 | 1/12 | 2/12 | 3/12 | 6/12 | 1 | 2 | alpha (log-log fit, tau 1/12..2) |
|---|---|---|---|---|---|---|---|---|
| (0.5, -0.7) | -1.196 | -0.574 | -0.405 | -0.330 | -0.233 | -0.164 | -0.115 | -0.506 |
| (1.0, -0.7) | -2.115 | -1.015 | -0.715 | -0.582 | -0.409 | -0.286 | -0.200 | -0.511 |
| (1.0, -0.9) | -2.547 | -1.211 | -0.852 | -0.693 | -0.486 | -0.340 | -0.237 | -0.513 |
| (2.0, -0.9) | -3.631 | -1.711 | -1.200 | -0.975 | -0.682 | -0.475 | -0.331 | -0.517 |

## Calendar monotonicity

## Q: flat 0.04 (a0, a1, a2, lam) = (0.0, 0.0, 0.04, 1.0)
| k | rho | violations | min increment | min / C | at K/F | at tau |
|---|-----|------------|---------------|---------|--------|--------|
| 0.25 | -0.9 | 0 | 1.175e-56 | n/a | 1.6 | 0.020 |
| 0.25 | -0.5 | 0 | 2.719e-16 | n/a | 1.6 | 0.020 |
| 0.25 | 0.0 | 0 | 2.036e-11 | n/a | 0.6 | 0.020 |
| 0.25 | 0.5 | 1 | -5.551e-17 | -1.39e-16 | 0.6 | 0.020 |
| 0.5 | -0.9 | 0 | 5.950e-49 | n/a | 1.6 | 0.020 |
| 0.5 | -0.5 | 0 | 9.067e-13 | n/a | 1.6 | 0.020 |
| 0.5 | 0.0 | 0 | 1.625e-08 | n/a | 0.6 | 0.020 |
| 0.5 | 0.5 | 0 | 9.026e-14 | n/a | 0.6 | 0.020 |
| 1.0 | -0.9 | 0 | 1.078e-34 | n/a | 1.6 | 0.020 |
| 1.0 | -0.5 | 0 | 1.819e-09 | n/a | 1.6 | 0.020 |
| 1.0 | 0.0 | 0 | 2.280e-06 | n/a | 0.6 | 0.020 |
| 1.0 | 0.5 | 71 | -9.246e-02 | -4.22e-01 | 1.6 | 0.226 |
| 2.0 | -0.9 | 0 | 9.997e-27 | n/a | 1.6 | 0.020 |
| 2.0 | -0.5 | 0 | 1.121e-07 | n/a | 1.6 | 0.020 |
| 2.0 | 0.0 | 0 | 3.193e-05 | n/a | 0.6 | 0.020 |
| 2.0 | 0.5 | 5 | -1.502e-03 | -8.10e-03 | 1.6 | 0.226 |

## Q: decreasing short-end (a0, a1, a2, lam) = (0.03, 0.0, 0.03, 2.0)
| k | rho | violations | min increment | min / C | at K/F | at tau |
|---|-----|------------|---------------|---------|--------|--------|
| 0.25 | -0.9 | 0 | 3.243e-39 | n/a | 1.6 | 0.020 |
| 0.25 | -0.5 | 0 | 6.631e-13 | n/a | 1.6 | 0.020 |
| 0.25 | 0.0 | 0 | 1.528e-09 | n/a | 0.6 | 0.020 |
| 0.25 | 0.5 | 0 | 1.660e-14 | n/a | 0.6 | 0.020 |
| 0.5 | -0.9 | 0 | 3.467e-39 | n/a | 1.6 | 0.020 |
| 0.5 | -0.5 | 0 | 7.871e-11 | n/a | 1.6 | 0.020 |
| 0.5 | 0.0 | 0 | 1.794e-07 | n/a | 0.6 | 0.020 |
| 0.5 | 0.5 | 0 | 1.009e-11 | n/a | 0.6 | 0.020 |
| 1.0 | -0.9 | 0 | 8.501e-30 | n/a | 1.6 | 0.020 |
| 1.0 | -0.5 | 0 | 1.951e-08 | n/a | 1.6 | 0.020 |
| 1.0 | 0.0 | 0 | 8.471e-06 | n/a | 0.6 | 0.020 |
| 1.0 | 0.5 | 53 | -4.762e-02 | -2.25e-01 | 1.6 | 0.328 |
| 2.0 | -0.9 | 0 | 7.654e-24 | n/a | 1.6 | 0.020 |
| 2.0 | -0.5 | 0 | 4.805e-07 | n/a | 1.6 | 0.020 |
| 2.0 | 0.0 | 0 | 7.317e-05 | n/a | 0.6 | 0.020 |
| 2.0 | 0.5 | 2 | -2.646e-04 | -1.58e-03 | 1.6 | 0.020 |

## Q: humped (a0, a1, a2, lam) = (0.0, 0.06, 0.03, 1.5)
| k | rho | violations | min increment | min / C | at K/F | at tau |
|---|-----|------------|---------------|---------|--------|--------|
| 0.25 | -0.9 | 0 | 1.016e-64 | n/a | 1.6 | 0.020 |
| 0.25 | -0.5 | 0 | 7.849e-18 | n/a | 1.6 | 0.020 |
| 0.25 | 0.0 | 0 | 2.966e-12 | n/a | 0.6 | 0.020 |
| 0.25 | 0.5 | 1 | -1.665e-16 | -4.16e-16 | 0.6 | 0.020 |
| 0.5 | -0.9 | 0 | 3.911e-53 | n/a | 1.6 | 0.020 |
| 0.5 | -0.5 | 0 | 1.331e-13 | n/a | 1.6 | 0.020 |
| 0.5 | 0.0 | 0 | 5.803e-09 | n/a | 0.6 | 0.020 |
| 0.5 | 0.5 | 0 | 1.193e-14 | n/a | 0.6 | 0.020 |
| 1.0 | -0.9 | 0 | 9.958e-37 | n/a | 1.6 | 0.020 |
| 1.0 | -0.5 | 0 | 6.766e-10 | n/a | 1.6 | 0.020 |
| 1.0 | 0.0 | 0 | 1.318e-06 | n/a | 0.6 | 0.020 |
| 1.0 | 0.5 | 38 | -5.169e-02 | -2.77e-01 | 1.6 | 0.226 |
| 2.0 | -0.9 | 0 | 6.593e-28 | n/a | 1.6 | 0.020 |
| 2.0 | -0.5 | 0 | 6.176e-08 | n/a | 1.6 | 0.020 |
| 2.0 | 0.0 | 0 | 2.274e-05 | n/a | 0.6 | 0.020 |
| 2.0 | 0.5 | 8 | -3.039e-02 | -1.67e-01 | 1.6 | 0.020 |

## What the tables show

1. Under flat Q = 0.04 the ATM skew is negative for all four (k, rho) pairs and
   its log-log slope over tau in [1/12, 2] is alpha = -0.506 to -0.517,
   consistent with the expected tau^(-1/2) decay (and with tau^(H-1/2) at H = 0).
2. The skew level scales roughly linearly in k at fixed rho ((0.5, -0.7) vs
   (1.0, -0.7)) and grows in |rho| at fixed k.
3. At k = 1, rho = -0.7 the skew is -1.015 at tau = 1/12 and -0.286 at 1y,
   matching the feasibility spot checks in epics/0010 (-1.01, -0.29).
4. Calendar check: with F held constant, undiscounted call prices are
   non-decreasing in tau at all 319 (K, tau-pair) increments for rho = 0 and for
   both negative rhos (-0.5, -0.9) in every (k, curve) configuration — i.e.
   everywhere in the empirically relevant sign of rho on this grid.
5. Violations occur only for rho = +0.5, only at k = 1.0 and k = 2.0, and
   mostly at the OTM end (K/F = 1.6): up to 71 of 319 increments (flat Q,
   k = 1), with min/C down to -0.42.
6. Three single-count "violations" of order 1e-16 to 1e-17 (k = 0.25,
   rho = +0.5) are float noise at deep-OTM price levels, not arbitrage.
7. This is numerical evidence on a grid, not a proof: no violation was found
   for rho <= 0 anywhere on the grid, while positive rho at high k breaks
   calendar monotonicity.

## Planner review (2026-10-19): ρ > 0 results are numerical artefacts
The ρ > 0 "violations" cannot be trusted: for ρ > 0 and large k, `crl_call` does not converge
in the number of Gauss–Hermite nodes. At the worst case above (ρ = +0.0068, k = 2.61,
K/F = 2, τ = 0.1435) the price is 0.00007 (32 nodes), 0.244 (64), 0.092 (128), 0.097 (256),
while ρ = 0 and ρ = −0.0068 agree to every digit from 32 to 256 nodes. Cause: with ρ > 0 the
astronomically high-level states pair with large positive returns, so the integrand has a
far-tail peak fixed-node quadrature misses. The ρ ≤ 0 result stands (converged; MC-checked
in epic 0010). Whether ρ > 0 admits calendar arbitrage is **unknown**, not shown.
