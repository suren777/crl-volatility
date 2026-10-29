# Large-L' structure for rho < 0 (epic 0016, task 07-largeL)

Notation as in `calendar_remainder_rho.md`: r = |rho| > 0, lambda = L' > 0, sigma = s^2 =
ln(1 + k^2), mu = sqrt(2 lambda), c0 = ln(2 lambda/u^2) + sigma/2, v = ln y - ln(2 lambda),
q(v) = mu cosh(v/2) + r (v + c0)/s, E = {v : q(v) < 0}. Labels: PROVED (argument written out
below), VERIFIED NUMERICALLY (`calendar_largeL.py`, `test_calendar_largeL.py`), CONJECTURED.

## Result

**No inequality R'(L) < 1 for rho < 0 at large L' is proved here, and no threshold L0(k, delta)
is established.** The goal of this task is not delivered. What is delivered is the localisation
of the set where the potential's curvature bound fails (missing inequality 1 of
`calendar_remainder_rho.md` is reduced to a tail estimate on a known interval), together with
the numerical checks of these structural facts.

## Lemmas

**Lemma 1 (PROVED): E is an interval.** q is convex in v, since mu cosh(v/2) is convex and
r(v + c0)/s is affine. A sublevel set of a convex function is convex, so E = {q < 0} is an
interval (possibly empty).

**Lemma 2 (PROVED): E lies to the left of -c0 - mu s/r.** If q(v) < 0, then
mu cosh(v/2) < -r(v + c0)/s, so -(v + c0) > (mu s/r) cosh(v/2) >= mu s/r, because cosh >= 1.
Hence E is contained in (-inf, t0) with t0 = -c0 - mu s/r.

**Lemma 3 (PROVED): curvature off E.** From `calendar_remainder_rho.md`,
V'' = 1/sigma + (q'^2 + q q'')/(1 - r^2), with q' = mu sinh(v/2)/2 + r/s and
q'' = mu cosh(v/2)/4 >= 0. On the complement of E, q >= 0, so q q'' >= 0 and
V'' >= 1/sigma + q'^2/(1 - r^2) >= 1/sigma. So V is convex with V'' >= 1/sigma off E, and any
failure of the curvature bound used for Brascamp-Lieb is confined to E.

**Consequence.** Lemmas 1 to 3 localise the obstruction: the only place where the curvature
bound fails is the interval E, which lies in (-inf, t0). What is not shown is (i) that nu(E) is
small uniformly in u, and (ii) how to replace the convexity step on E. Neither is derived here,
so the localisation does not by itself give R' < 1.

**Not proved: a bound on nu(E).** One route would bound the normaliser from below by a window
near the mode of nu and bound the integral over E from above. Such a bound has not been carried
out. c0 ranges over all of R as u varies, so the window location and the constants depend on u,
and a bound uniform in u needs an argument that is not given here.

Review note: the complement of E is in general two rays, since q → +∞ again as v → −∞
(cosh dominates the affine term). ν restricted to the complement is therefore not log-concave
across the gap, so Brascamp–Lieb cannot be applied to it as one piece: a proof must either
show the left ray carries negligible mass or work on the right ray [sup E, ∞) only.

## Numerical checks (VERIFIED NUMERICALLY)

The scan below checks Lemmas 1 to 3 on the grid of `calendar_remainder_rho.py` with the widened
v-grid, and reports nu(E). The finite-difference check of the curvature formula is done on
|v| <= 5 off E, as a relative error, because absolute errors at V of order 1e6 are roundoff.

## Decisions

- The task asked for a rigorous R'(L) < 1 and an explicit L0. Neither is delivered. A partial
  result that is actually proved (Lemmas 1 to 3) is reported instead of an overclaimed bound.
- The uniform-in-u tail bound on nu(E) is left open because the naive bound fails as described.
- No file outside `derivations/calendar_largeL.*` and `derivations/test_calendar_largeL.py` is
  changed.

<!-- scan -->
# Structural lemmas for rho < 0 (scan)

Output of `uv run python derivations/calendar_largeL.py`. Grid: u in (0.1, 1.0, 5.0), k in (0.05, 0.3, 1.0, 2.0, 2.5), rho in (-0.05, -0.2, -0.5, -0.7, -0.9, -0.99), L' in (0.5, 2.0, 10.0, 50.0, 300.0, 3000.0); 540 points.

- E is an interval at every point: True.
- E lies left of -c0 - mu s/r at every point with E nonempty: True.
- min over the complement of E of (V'' - 1/sigma): 2.616e-02 (must be >= 0 up to rounding).
- max relative |finite-difference V'' - formula| off E on |v| <= 5: 9.898e-07.
- E nonempty at 69 of 540 points.
- nu-mass of E: max 5.022e-01; max at L' >= 300: 0.000e+00.

| u | k | rho | L' | E empty | E max | -c0 - mu s/r | nu(E) |
|---|---|---|---|---|---|---|---|
| 1.0 | 1.0 | -0.05 | 0.5 | True | nan | -16.9977 | 0.00e+00 |
| 1.0 | 1.0 | -0.05 | 2.0 | True | nan | -35.0351 | 0.00e+00 |
| 1.0 | 1.0 | -0.05 | 10.0 | True | nan | -77.8083 | 0.00e+00 |
| 1.0 | 1.0 | -0.05 | 50.0 | True | nan | -171.4627 | 0.00e+00 |
| 1.0 | 1.0 | -0.05 | 300.0 | True | nan | -414.6103 | 0.00e+00 |
| 1.0 | 1.0 | -0.05 | 3000.0 | True | nan | -1298.8341 | 0.00e+00 |
| 1.0 | 1.0 | -0.2 | 0.5 | True | nan | -4.5093 | 0.00e+00 |
| 1.0 | 1.0 | -0.2 | 2.0 | True | nan | -10.0584 | 0.00e+00 |
| 1.0 | 1.0 | -0.2 | 10.0 | True | nan | -21.9588 | 0.00e+00 |
| 1.0 | 1.0 | -0.2 | 50.0 | True | nan | -46.5795 | 0.00e+00 |
| 1.0 | 1.0 | -0.2 | 300.0 | True | nan | -108.7102 | 0.00e+00 |
| 1.0 | 1.0 | -0.2 | 3000.0 | True | nan | -331.4931 | 0.00e+00 |
| 1.0 | 1.0 | -0.5 | 0.5 | True | nan | -2.0117 | 0.00e+00 |
| 1.0 | 1.0 | -0.5 | 2.0 | True | nan | -5.0631 | 0.00e+00 |
| 1.0 | 1.0 | -0.5 | 10.0 | True | nan | -10.7889 | 0.00e+00 |
| 1.0 | 1.0 | -0.5 | 50.0 | True | nan | -21.6028 | 0.00e+00 |
| 1.0 | 1.0 | -0.5 | 300.0 | True | nan | -47.5302 | 0.00e+00 |
| 1.0 | 1.0 | -0.5 | 3000.0 | True | nan | -138.0249 | 0.00e+00 |
| 1.0 | 1.0 | -0.7 | 0.5 | True | nan | -1.5359 | 0.00e+00 |
| 1.0 | 1.0 | -0.7 | 2.0 | True | nan | -4.1116 | 0.00e+00 |
| 1.0 | 1.0 | -0.7 | 10.0 | True | nan | -8.6613 | 0.00e+00 |
| 1.0 | 1.0 | -0.7 | 50.0 | True | nan | -16.8454 | 0.00e+00 |
| 1.0 | 1.0 | -0.7 | 300.0 | True | nan | -35.8768 | 0.00e+00 |
| 1.0 | 1.0 | -0.7 | 3000.0 | True | nan | -101.1738 | 0.00e+00 |
| 1.0 | 1.0 | -0.9 | 0.5 | False | -1.5728 | -1.2716 | 3.09e-01 |
| 1.0 | 1.0 | -0.9 | 2.0 | True | nan | -3.5830 | 0.00e+00 |
| 1.0 | 1.0 | -0.9 | 10.0 | True | nan | -7.4793 | 0.00e+00 |
| 1.0 | 1.0 | -0.9 | 50.0 | True | nan | -14.2024 | 0.00e+00 |
| 1.0 | 1.0 | -0.9 | 300.0 | True | nan | -29.4028 | 0.00e+00 |
| 1.0 | 1.0 | -0.9 | 3000.0 | True | nan | -80.7010 | 0.00e+00 |
| 1.0 | 1.0 | -0.99 | 0.5 | False | -1.4032 | -1.1875 | 4.56e-01 |
| 1.0 | 1.0 | -0.99 | 2.0 | True | nan | -3.4148 | 0.00e+00 |
| 1.0 | 1.0 | -0.99 | 10.0 | True | nan | -7.1032 | 0.00e+00 |
| 1.0 | 1.0 | -0.99 | 50.0 | True | nan | -13.3614 | 0.00e+00 |
| 1.0 | 1.0 | -0.99 | 300.0 | True | nan | -27.3428 | 0.00e+00 |
| 1.0 | 1.0 | -0.99 | 3000.0 | True | nan | -74.1869 | 0.00e+00 |
