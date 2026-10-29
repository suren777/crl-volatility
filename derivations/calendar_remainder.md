# Rigorous bound on R'(L) at rho = 0 (epic 0016, task 03-remainder)

Notation as in `calendar_band.md`: r = |rho|, lambda = L' = L + ln m > 0, sigma = s^2,
P = 1/sigma + lambda/2, a = 2/lambda. Labels: PROVED (argument written out below),
VERIFIED NUMERICALLY (`calendar_remainder.py` and `test_calendar_remainder.py`), CONJECTURED
(stated, not established). The scope is rho = 0 only; rho < 0 is not treated.

## Result

**Theorem (rho = 0). PROVED.** For every k and every lambda > 0,

    R'(L) <= (lambda/2)(exp(1/P) - 1).

If sigma = ln(1 + k^2) <= 2, i.e. k <= sqrt(e^2 - 1) = 2.5277, then R'(L) < 1 for every
L' > 0. The explicit threshold is L0 = 0. The bound is independent of u.

For k > 2.5277 this route does not give R' < 1 (see "Not established").

## Proof

**Step 1 (PROVED): the conditional law at rho = 0.** At rho = 0, w = y and
d2 = -(y/2 + lambda)/sqrt(y). Put v = ln y - ln(2 lambda), so y = 2 lambda e^v, and
Z = (v + c0)/s with c0 = ln(2 lambda/u^2) + sigma/2. Then nu(dZ) is proportional to
phi(Z) phi(d2)/sqrt(y), and in v the density is proportional to exp(-V(v)) with

    V(v) = (v + c0)^2/(2 sigma) + v/2 + (lambda/2) cosh v.

The three terms come from phi(Z), from 1/sqrt(y) proportional to e^{-v/2}, and from
phi(d2), since y/8 + lambda^2/(2y) = (lambda/2) cosh v + lambda/2 (up to a constant).
Since cosh v >= 1, V''(v) = 1/sigma + (lambda/2) cosh v >= P > 0 for all v.

**Step 2 (PROVED): R' as a moment generating function.** By the definition of R' in
`calendar_band.md`, R' = Cov_nu(y/2, d2/sqrt(w)). Here y/2 = lambda e^v and
d2/sqrt(w) = -(1/2 + e^{-v}/2). So R' = -(lambda/2) Cov_nu(e^v, e^{-v}). Since
E_nu(e^v e^{-v}) = 1, this is R' = (lambda/2)(E e^v E e^{-v} - 1).

**Step 3 (PROVED, cited theorem): Brascamp-Lieb in one dimension.** For mu proportional to
e^{-W} on the real line with W in C^2 and W'' > 0, and f locally Lipschitz with finite
E_mu(f'^2), Var_mu(f) <= E_mu(f'^2/W'') (Brascamp and Lieb, J. Funct. Anal. 22 (1976), 366-389, the
one-dimensional variance inequality). The reference is cited from memory and has not been
checked against the paper; it must be verified before the note is cited elsewhere. Take the tilt nu_theta proportional to exp(-V(v) + theta v), so W = V - theta v,
W'' = V'' >= P, and f = v. Then Var_theta(v) <= E_theta[1/V''] <= 1/P. Finiteness holds
because V grows like (lambda/4)e^{|v|}, faster than any linear term.

**Step 4 (PROVED): the mgf integral.** Let K(theta) = log E_nu e^{theta v}. It is smooth on
R with K(0) = 0 and K''(theta) = Var_theta(v). Taylor's formula with integral remainder
gives K(1) + K(-1) = int_0^1 (1 - s)[K''(s) + K''(-s)] ds, because the K'(0) terms cancel.
By Step 3, the integrand is at most 2/P, so K(1) + K(-1) <= 1/P. Hence
E e^v E e^{-v} <= exp(1/P), which gives the theorem's bound on R' through Step 2.

**Step 5 (PROVED): comparison with 1.** With a = 2/lambda, 1/P = sigma a/(sigma + a). Two
elementary facts:

- log(1 + a) >= 2a/(2 + a) for a >= 0. The difference is 0 at a = 0 and has derivative
  a^2/((1 + a)(2 + a)^2) >= 0. It is strictly positive for a > 0.
- 2a/(2 + a) - sigma a/(sigma + a) = a^2 (2 - sigma)/((2 + a)(sigma + a)), which is >= 0
  when sigma <= 2.

So for sigma <= 2 and a > 0, 1/P < log(1 + a). Equivalently exp(1/P) - 1 < a = 2/lambda,
so R' < 1. This holds for every lambda > 0 and every u. The sympy checks cover each
identity above.

**Comparison with `calendar_band.md`.** The Proposition at rho = 0 there proves the same
inequality R' <= (L'/2)(e^{1/kappa} - 1), kappa = 1/s^2 + L'/2, by Caffarelli's contraction
theorem. That citation is marked there as "to be verified". The argument above gives the same
bound with Brascamp-Lieb, which is standard and whose hypotheses were checked in Step 3.

## Numerical checks (VERIFIED NUMERICALLY)

The scan below uses u in {0.1, 1, 5}, k in {0.05, ..., 10}, lambda in {0.5, ..., 3000}, and
tilts theta in {-1, ..., 1}. It prints the following.

- R' from `Model.r_slope` on a widened z-grid agrees with a quadrature in v to about 3e-9.
  This quadrature uses the v-form of V and does not reuse `Model`.
- Brascamp-Lieb holds on every tilt, and E_theta[1/V''] <= 1/P.
- The bound is above R' on every grid row, and below 1 on every row with sigma <= 2.

**Finding (not fixed here; outside this task's scope).** `Model.r_slope` builds its z-grid on
[-30, z_hi). For small k and large lambda, the conditional law sits near z = -c0/s, which lies
outside that range. The shipped `Model` then drops most of the mass: at k = 0.05 and
lambda = 3000 its value is 1e-6, against 0.408 from quadrature. The scan prints the
shipped-grid error. `calendar_band.py` uses the same grid, so its scans at large L' and small k
are affected.

## Not established

**(a) k > 2.5277 at rho = 0 (CONJECTURED route).** For sigma > 2 the bound equals
1 + (1 - 2/sigma)/lambda + O(lambda^-2) as lambda grows, so it exceeds 1 for large lambda. The
loss comes from dropping cosh v - 1 in V''. The sharper pieces are:

- Write V'' = P + b(v) with b = (lambda/2)(cosh v - 1) >= 0. Cauchy-Schwarz gives the
  Cramer-Rao bound Var_theta(v) >= 1/E_theta[V''] (PROVED). Together with
  cosh v - 1 >= v^2/2 this gives x := E_theta[cosh v - 1] >= x*, where x* is the positive
  root of lambda x^2/2 + P x - 1/2 = 0.
- Pointwise, 1/(P + b) <= 1/P - b/P^2 + b^2/P^3 for b >= 0. So
  Var_theta(v) <= 1/P - (lambda/(2P^2)) x* + E_theta[b^2]/P^3, and the same bound holds for
  -theta. Integrating as in Step 4 gives
  K(1) + K(-1) <= 1/P - lambda x*/(2P^2) + sup_theta E_theta[b^2]/P^3.
- Asymptotically the first two terms give 2/lambda - (2 + 4/sigma)/lambda^2, against the target
  2/lambda - 2/lambda^2 from Step 5. So the sharpened bound would leave a margin of order
  4/(sigma lambda^2). That margin survives if sup_theta E_theta[b^2] <= B with B independent
  of lambda, because then the remainder is O(lambda^-3).

The missing input is the tilted moment bound sup_{|theta| <= 1} E_theta[b^2] <= B, with explicit
B and no lambda dependence. It is not proved here. It is the only step needed beyond Steps 1-4.
Its numerical value is not printed by the script, so it is CONJECTURED.

**(b) rho < 0 (not done).** The potential has the extra terms r(v + c0)/s and
q^2/(2(1 - r^2)), with q = mu cosh(v/2) + r(v + c0)/s. The proof above needs V'' >= P with
P = 1/sigma + lambda/(2(1 - r^2)) or similar. That inequality is not established for r > 0.
`calendar_band.md` notes that nu can fail to be log-concave where d2 > 0. The expansion in
`calendar_laplace.md` gives a leading gap sqrt(2) r/(s sqrt(lambda)) (ASYMPTOTIC), which is
larger than the rho = 0 gap, but no uniform explicit remainder for r > 0 is proved here.

<!-- scan -->
# Rigorous bound on R'(L) at rho = 0

Output of `uv run python derivations/calendar_remainder.py`. Bound: R' <= (lambda/2)(exp(1/P) - 1), P = 1/s^2 + lambda/2, lambda = L + ln m. Grid: u in (0.1, 1.0, 5.0), k in (0.05, 0.3, 1.0, 2.0, 2.5, 3.0, 6.0, 10.0), lambda in (0.5, 2.0, 10.0, 50.0, 300.0, 3000.0), tilts theta in (-1.0, -0.5, 0.0, 0.5, 1.0). R' (quad) is the quadrature of (lambda/2)(E e^v E e^-v - 1) on v in [-40, 40]; R' (wide) is Model.r_slope on a z-grid widened to v in [-25, 25]; R' (default) is Model.r_slope as shipped.

- Identity: max |R'(wide) - R'(quad)| = 2.83e-09.
- Shipped Model grid: max |R'(default) - R'(quad)| = 0.991; rows off by more than 1e-6: 7 of 144 (small k, large lambda: the z-grid starts at -30 and the conditional law sits at z near -c0/s).
- Brascamp-Lieb on every tilt: max(Var_theta(v) - E_theta[1/V'']) = -2.02e-15 (must be <= 0 up to quadrature error); max(E_theta[1/V''] - 1/P) = -1.94e-09 (must be <= 0).
- Bound >= R'(quad) on every grid row: True.
- Bound < 1 on every row with sigma <= 2: True (90 rows).

Threshold L0 at rho = 0 (sigma = s^2): L0 = 0 (every L' > 0) for sigma <= 2, i.e. k <= sqrt(e^2 - 1) = 2.5277; no threshold from this bound for larger k.

| k | s^2 | L0 | max R' (quad) | max bound | bound - 1 at lambda = 3000 |
|---|---|---|---|---|---|
| 0.05 | 0.0025 | 0 | 0.5483 | 0.7895 | -0.2105 |
| 0.3 | 0.0862 | 0 | 0.9914 | 0.9927 | -0.0073 |
| 1.0 | 0.6931 | 0 | 0.9990 | 0.9994 | -0.0006 |
| 2.0 | 1.6094 | 0 | 0.9996 | 0.9999 | -0.0001 |
| 2.5 | 1.9810 | 0 | 0.9997 | 1.0000 | -0.0000 |
| 3.0 | 2.3026 | none | 0.9997 | 1.0102 | +0.0000 |
| 6.0 | 3.6109 | none | 0.9998 | 1.4177 | +0.0001 |
| 10.0 | 4.6151 | none | 0.9999 | 1.8808 | +0.0002 |

Full table at u = 1.0.

| k | lambda | R' (quad) | R' (wide) | R' (default) | bound | 1/P |
|---|---|---|---|---|---|---|
| 0.05 | 0.5 | 0.000625 | 0.000625 | 0.000625 | 0.000625 | 0.002495 |
| 0.05 | 2.0 | 0.002487 | 0.002487 | 0.002487 | 0.002494 | 0.002491 |
| 0.05 | 10.0 | 0.011236 | 0.011236 | 0.011236 | 0.012346 | 0.002466 |
| 0.05 | 50.0 | 0.030131 | 0.030131 | 0.030131 | 0.058824 | 0.002350 |
| 0.05 | 300.0 | 0.083342 | 0.083342 | 0.083342 | 0.272727 | 0.001817 |
| 0.05 | 3000.0 | 0.407621 | 0.407621 | 0.000001 | 0.789474 | 0.000526 |
| 0.3 | 0.5 | 0.021982 | 0.021982 | 0.021982 | 0.022005 | 0.084360 |
| 0.3 | 2.0 | 0.075888 | 0.075888 | 0.075888 | 0.082573 | 0.079340 |
| 0.3 | 10.0 | 0.187087 | 0.187087 | 0.187087 | 0.310386 | 0.060227 |
| 0.3 | 50.0 | 0.429708 | 0.429708 | 0.429708 | 0.692402 | 0.027319 |
| 0.3 | 300.0 | 0.847730 | 0.847730 | 0.847730 | 0.931073 | 0.006188 |
| 0.3 | 3000.0 | 0.990097 | 0.990097 | 0.000266 | 0.992652 | 0.000662 |
| 1.0 | 0.5 | 0.182690 | 0.182690 | 0.182690 | 0.201346 | 0.590774 |
| 1.0 | 2.0 | 0.362239 | 0.362239 | 0.362239 | 0.505890 | 0.409384 |
| 1.0 | 10.0 | 0.654111 | 0.654111 | 0.654111 | 0.839543 | 0.155215 |
| 1.0 | 50.0 | 0.913293 | 0.913293 | 0.913293 | 0.963545 | 0.037818 |
| 1.0 | 300.0 | 0.988308 | 0.988308 | 0.988308 | 0.993751 | 0.006603 |
| 1.0 | 3000.0 | 0.998999 | 0.998999 | 0.998999 | 0.999372 | 0.000666 |
| 2.0 | 0.5 | 0.348854 | 0.348854 | 0.348854 | 0.537706 | 1.147664 |
| 2.0 | 2.0 | 0.545262 | 0.545262 | 0.545262 | 0.852944 | 0.616776 |
| 2.0 | 10.0 | 0.827259 | 0.827259 | 0.827259 | 0.973492 | 0.177894 |
| 2.0 | 50.0 | 0.966438 | 0.966438 | 0.966438 | 0.995041 | 0.039030 |
| 2.0 | 300.0 | 0.995360 | 0.995360 | 0.995360 | 0.999188 | 0.006639 |
| 2.0 | 3000.0 | 0.999577 | 0.999577 | 0.999577 | 0.999919 | 0.000666 |
| 2.5 | 0.5 | 0.389413 | 0.389413 | 0.389413 | 0.690417 | 1.324863 |
| 2.5 | 2.0 | 0.588232 | 0.588232 | 0.588232 | 0.943601 | 0.664542 |
| 2.5 | 10.0 | 0.856493 | 0.856493 | 0.856493 | 0.996031 | 0.181660 |
| 2.5 | 50.0 | 0.973232 | 0.973232 | 0.973232 | 0.999678 | 0.039208 |
| 2.5 | 300.0 | 0.996276 | 0.996276 | 0.996276 | 0.999964 | 0.006644 |
| 2.5 | 3000.0 | 0.999657 | 0.999657 | 0.999657 | 0.999997 | 0.000666 |
| 3.0 | 0.5 | 0.416645 | 0.416645 | 0.416645 | 0.827954 | 1.461359 |
| 3.0 | 2.0 | 0.617708 | 0.617708 | 0.617708 | 1.008136 | 0.697207 |
| 3.0 | 10.0 | 0.874618 | 0.874618 | 0.874618 | 1.010178 | 0.184017 |
| 3.0 | 50.0 | 0.977221 | 0.977221 | 0.977221 | 1.002503 | 0.039317 |
| 3.0 | 300.0 | 0.996819 | 0.996819 | 0.996819 | 1.000435 | 0.006647 |
| 3.0 | 3000.0 | 0.999706 | 0.999706 | 0.999706 | 1.000044 | 0.000666 |
| 6.0 | 0.5 | 0.484348 | 0.484348 | 0.484348 | 1.417728 | 1.897757 |
| 6.0 | 2.0 | 0.695328 | 0.695328 | 0.695328 | 1.188297 | 0.783123 |
| 6.0 | 10.0 | 0.915961 | 0.915961 | 0.915961 | 1.043249 | 0.189504 |
| 6.0 | 50.0 | 0.985766 | 0.985766 | 0.985766 | 1.008869 | 0.039562 |
| 6.0 | 300.0 | 0.998001 | 0.998001 | 0.998001 | 1.001486 | 0.006654 |
| 6.0 | 3000.0 | 0.999813 | 0.999813 | 0.999813 | 1.000149 | 0.000667 |
| 10.0 | 0.5 | 0.512353 | 0.512353 | 0.512353 | 1.880818 | 2.142800 |
| 10.0 | 2.0 | 0.730286 | 0.730286 | 0.730286 | 1.274839 | 0.821909 |
| 10.0 | 10.0 | 0.931992 | 0.931992 | 0.931992 | 1.056492 | 0.191693 |
| 10.0 | 50.0 | 0.988897 | 0.988897 | 0.988897 | 1.011328 | 0.039656 |
| 10.0 | 300.0 | 0.998444 | 0.998444 | 0.998444 | 1.001889 | 0.006657 |
| 10.0 | 3000.0 | 0.999854 | 0.999854 | 0.999854 | 1.000189 | 0.000667 |
