# Moment monotonicity of `crl_call` marginals (calendar-arbitrage certificate)

Run 2026-10-19. The collapsed exact moment E[S^p] (F = 1) is
monotone in tau whenever the marginals are convex-ordered; a strict decrease
proves calendar arbitrage (calendar_proof.md). All integrals by dense
trapezoid on the weight-folded exponent; witnesses gated by resolution
doubling and a wider-clip re-run.

## Validation of the representation and the moment formula

- call prices, 2-D grid vs `msv.crl_call`: worst rel err 2.14e-03
- ln E[S^p], trapezoid (collapsed) vs 2-D grid, rho <= 0 (grid valid): worst abs err 4.96e-16
- ln E[S^p], rho = +0.3: worst abs err 1.97e+03 -- the far bump (Z ~ 55-65) is beyond any linear-space grid by construction, so this row is a structural limit, not an error; for rho > 0 the collapsed formula rests on exact algebra, the p = 1 identity below, and the trapezoid resolution/clip gates
- ln E[S^p], trapezoid at clip 60 vs 160, rho <= 0: worst abs err 3.33e-16; rho = +0.3: 5.35e+00 (clip 60 cuts the far bump at its peak)
- E[S] = 1 from the 2-D grid: worst |err| 0.00e+00
- p = 1 identity ln E[S] = 0 (trapezoid): worst residual 6.99e-14

## Moment-monotonicity scan (log-space trapezoid)

- finite-moment window per rho: p in (1, 1/(1-rho^2)); p at 25/50/75% of the window
- tau grid: 25 log-spaced points in [0.01, 5.0], trapezoid of the folded exponent on (-160, 160), violation = ln-increment < -1e-06
- rho > 0: 378 of 378 series violate
- rho < 0: 0 of 324 series violate

Worst witnesses (rho, k, NS, p, tau1 -> tau2, d ln E[S^p]):
```
rho=+0.9000 k=0.30 ns='humped' (0.0, 0.06, 0.03, 1.5) p=4.197368 tau 0.0100 -> 0.0130: -1.041e+03
rho=+0.9000 k=0.30 ns='flat 0.04' (0.0, 0.0, 0.04, 1.0) p=4.197368 tau 0.0100 -> 0.0130: -1.010e+03
rho=+0.9000 k=0.30 ns='decreasing short-end' (0.03, 0.0, 0.03, 2.0) p=4.197368 tau 0.0100 -> 0.0130: -9.835e+02
rho=+0.7000 k=0.30 ns='humped' (0.0, 0.06, 0.03, 1.5) p=1.720588 tau 0.0100 -> 0.0130: -3.978e+02
rho=+0.7000 k=0.30 ns='flat 0.04' (0.0, 0.0, 0.04, 1.0) p=1.720588 tau 0.0100 -> 0.0130: -3.861e+02
rho=+0.9000 k=0.50 ns='humped' (0.0, 0.06, 0.03, 1.5) p=4.197368 tau 0.0100 -> 0.0130: -3.839e+02
rho=+0.7000 k=0.30 ns='decreasing short-end' (0.03, 0.0, 0.03, 2.0) p=1.720588 tau 0.0100 -> 0.0130: -3.761e+02
rho=+0.9000 k=0.50 ns='flat 0.04' (0.0, 0.0, 0.04, 1.0) p=4.197368 tau 0.0100 -> 0.0130: -3.722e+02
rho=+0.9000 k=0.50 ns='decreasing short-end' (0.03, 0.0, 0.03, 2.0) p=4.197368 tau 0.0100 -> 0.0130: -3.620e+02
rho=+0.9000 k=0.30 ns='humped' (0.0, 0.06, 0.03, 1.5) p=3.131579 tau 0.0100 -> 0.0130: -3.224e+02
```

## Certification (resolution-doubling and clip-sensitivity gates)

- increment d = ln E[S^p](tau2) - ln E[S^p](tau1) at h / h/2 resolution (clip 160) / h/2 resolution (clip 240); gate: pairwise agreement < 1e-06 and both < -1e-06

- rho=+0.9000 k=0.30 ns='humped' p=4.197368 tau 0.0100 -> 0.0130: d = -1041.024059 / -1041.024059 / -1041.024059 PASS
- rho=+0.9000 k=0.30 ns='flat 0.04' p=4.197368 tau 0.0100 -> 0.0130: d = -1010.004705 / -1010.004705 / -1010.004705 PASS
- rho=+0.9000 k=0.30 ns='decreasing short-end' p=4.197368 tau 0.0100 -> 0.0130: d = -983.459674 / -983.459674 / -983.459674 PASS
- rho=+0.7000 k=0.30 ns='humped' p=1.720588 tau 0.0100 -> 0.0130: d = -397.766134 / -397.766134 / -397.766134 PASS
- rho=+0.7000 k=0.30 ns='flat 0.04' p=1.720588 tau 0.0100 -> 0.0130: d = -386.062697 / -386.062697 / -386.062697 PASS
- rho=+0.9000 k=0.50 ns='humped' p=4.197368 tau 0.0100 -> 0.0130: d = -383.876535 / -383.876535 / -383.876535 PASS
- rho=+0.2000 k=2.60 ns='humped' p=1.010417 tau 0.0100 -> 0.0130: d = -0.880106 / -0.880106 / -0.880106 PASS
- rho=+0.1000 k=2.60 ns='decreasing short-end' p=1.002525 tau 0.0100 -> 0.0130: d = -0.868508 / -0.868508 / -0.868508 PASS
- rho=+0.3000 k=2.60 ns='flat 0.04' p=1.024725 tau 0.0100 -> 0.0130: d = -0.857647 / -0.857647 / -0.857647 PASS
- rho=+0.2000 k=2.60 ns='flat 0.04' p=1.010417 tau 0.0100 -> 0.0130: d = -0.852433 / -0.852433 / -0.852433 PASS
- rho=+0.3000 k=2.60 ns='decreasing short-end' p=1.024725 tau 0.0100 -> 0.0130: d = -0.831859 / -0.831859 / -0.831859 PASS
- rho=+0.2000 k=2.60 ns='decreasing short-end' p=1.010417 tau 0.0100 -> 0.0130: d = -0.827858 / -0.827858 / -0.827858 PASS

- worst gate disagreement: 7.28e-12

## crl_call cross-check (K/F in [0.5, 2.0], F = D = 1)

- crl_call's own 64-node GH misses the far-tail bump for rho > 0 (see the
  msv.py docstring), so ~0 drops are expected here and say nothing about
  the certificate above

- rho=+0.9000 k=0.30 ns=(0.0, 0.06, 0.03, 1.5) tau 0.0100 -> 0.0130: worst drop -5.551e-17 at K/F=0.600 (C = 0.400000)
- rho=+0.9000 k=0.30 ns=(0.0, 0.0, 0.04, 1.0) tau 0.0100 -> 0.0130: worst drop -5.551e-17 at K/F=0.600 (C = 0.400000)
- rho=+0.9000 k=0.30 ns=(0.03, 0.0, 0.03, 2.0) tau 0.0100 -> 0.0130: worst drop 5.877e-18 at K/F=2.000 (C = 0.000000)
- rho=+0.7000 k=0.30 ns=(0.0, 0.06, 0.03, 1.5) tau 0.0100 -> 0.0130: worst drop 2.404e-25 at K/F=2.000 (C = 0.000000)
- rho=+0.7000 k=0.30 ns=(0.0, 0.0, 0.04, 1.0) tau 0.0100 -> 0.0130: worst drop 5.998e-23 at K/F=2.000 (C = 0.000000)
