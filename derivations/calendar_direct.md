# Direct enclosure of dC/du for rho <= 0 in the band (epic 0016, route 02-direct)

Notation as in `calendar_band.md`: F = D = 1, u = sqrt(tau Q), s^2 = ln(1+k^2),
zeta = e^{sZ - s^2/2}, x = ln K. Write M = e^{rho u sqrt(zeta) Z - rho^2 u^2 zeta/2},
m = E[M], A = M/m, h = d ln M/du, mbar = m'/m = E[A h], w = (1-rho^2) u^2 zeta,
sigma = sqrt((1-rho^2) zeta), d1 = (ln A - x + w/2)/sqrt(w), d2 = d1 - sqrt(w), and
G := dC/du(K). This route does not use the perturbation argument of the epic; it bounds G
directly from the one-dimensional Z-integral.

## 1. The integral formula (PROVED)

C(u, x) = E_Z[A N(d1) - e^x N(d2)] (Black mixture, E[A] = 1). Chain rule, using
dBlack/dF = N(d1), dBlack/dw = F phi(d1)/(2 sqrt w), d ln A/du = h - mbar and
dw/du = 2(1-rho^2) u zeta:

    G = E_Z[ A ( phi(d1) sigma + N(d1) (h - mbar) ) ]
      = (I1 + I2 - (mh/m) I3) / m,

with mh = E[M h], I1 = E[M phi(d1) sigma], I2 = E[M N(d1) h], I3 = E[M N(d1)].
At rho = 0: A = 1, h = 0, mbar = 0, so G = E_Z[sqrt(zeta) phi(d1)] > 0 (a mixture of
Black vegas), so the claim holds at rho = 0 for every x.

Checks: `test_black_derivatives_symbolic` and `test_dcdu_integrand_symbolic` (SymPy, the
chain-rule identities and d ln M/du = h); `test_g_matches_finite_difference` (G against a
central difference of C in u, on random points, independent of the formula for G);
`test_call_matches_msv` (C against `msv.crl_call`, which uses Gauss-Hermite with a
different normalisation path, with tau = 1 and Q(1) = u^2).

## 2. Perturbation from rho = 0 (NOT DONE)

The planned lemma dC/du(K; rho) >= dC/du(K; 0) - |rho| B(u, k, K) with an explicit B was not
derived in this route. No rho_0 is claimed. The certified box in section 4 is the only
rigorous statement here, and it covers only that box.

## 3. Rigorous enclosure (method)

`enclose_g` returns an arb enclosure of G at a point or over a box of (u, k, rho, ln K).
The Z-integrals m, mh, I1, I2, I3 are evaluated with `acb.integral` on [-12, 12], which
gives a rigorous ball for the integral when the integrand is evaluated on balls. The part
of the Z-line outside [-12, 12] is bounded in closed form (Gaussian integrals of
phi(Z) e^{aZ} and phi(Z) |Z| e^{aZ}), using M <= 1 for Z >= 0 (rho <= 0) and
M <= exp(2|rho| u/(e s)) for Z < 0. The tail bound is tested against a wider grid
(`test_tail_bounds_dominate_truncation`, three parameter sets, including small k).

Two implementation points, both caught by tests:
- `acb.integral` evaluates the integrand at complex points in analytic mode. An integrand
  that uses `.real` or sign-dependent ball operations is then wrong: the first version gave
  m = 0.98492746 against the correct 0.98493371 (mpmath). The integrand is now analytic in
  analytic mode, and the sign-aware square `_sq` is used only for real balls.
- A ball-valued d1 that straddles zero in a square `d1**2` overflows exp; `_sq` avoids this.

## 4. Certification of R1 (u in [0.1, 1], k in [0.3, 2], rho in [-0.9, -0.1])

Claim on a box B of (u, k, rho): G > 0 for ln K in [0, X(B)], where X(B) >= l2'(p) for all
p in B. Theorem A covers ln K <= l1 (l1 >= 0 is proved) and Theorem B' covers ln K >= l2',
so [0, X(B)] contains the band (l1, l2') at every point of B. Containment is therefore by
construction, not checked on a grid as in the previous round.

**Strike bound X(B) (rigorous, one float root).** l2'(p) = L'*(p) - ln m(p), with g increasing
in L'. Over B, g(L; p) >= g_lo(L) = ln(2L/u_hi^2) + s_lo^2/2 + s_lo |rho|_min sqrt(2L), since the
u term decreases and the s and |rho| terms increase, so L'*(p) <= L_hi, the root of g_lo.
For ln m, Jensen gives ln m >= E[ln M] = rho u (s/2) e^{-s^2/8} - rho^2 u^2 / 2 (using E zeta = 1
and E[sqrt(zeta) Z] = (s/2) e^{-s^2/8}); the box minimum uses sup_s (s/2) e^{-s^2/8} (at s = 2 if
it lies in the box). Where the arb enclosure of m is finite and positive, its lower endpoint of
ln m is also used (the larger of the two). So X(B) = L_hi - inf_B ln m. Float is used only for
the root L_hi (bisection, then a factor 1 + 1e-9, which dominates the float error of g_lo) and
the final sum (+1e-12). This step is therefore PROVED up to that margin argument, not an arb
computation. `test_x_upper_dominates_l2_sharp` checks it against `Model.l2_sharp` on random
points. (The first version used the arb enclosure of m alone; on wide boxes that enclosure is
NaN, which is why the Jensen bound is needed.)

**Cells.** A cell is a box B times an interval [xa, xb] of ln K, with xb <= X(B). `enclose_g`
gives an arb enclosure of G over the cell. If its lower end is positive, the cell is accepted.
Otherwise the dimension with the largest width relative to the top-level box is halved. An
x-split halves [xa, xb]. A split of (u, k, rho) gives children whose x-range is cut at
min(xb, X(child)); this only removes ln K above an upper bound of l2' on the child, and a child
whose range becomes empty is dropped. Every accepted cell lies inside the claim, so the
accepted cells certify the claim on their union.

**Not done.** Requirement 2 (centred forms with first-order Taylor models and derivative
enclosures) was not implemented; the cells use plain interval evaluation over the whole cell,
which is the fallback in the task. Widening beyond R1 was not attempted in the time budget.

<!-- scan -->
Output of `uv run python derivations/calendar_direct.py -j 8 --boxes 8`; boxes run: 8 of 36.

## Certified region R1

- Region: u in [0.1, 1], k in [0.3, 2], rho in [-0.9, -0.1]; split into 36 boxes (3 log-tiles in u, 3 in k, 4 linear in rho). On each box the claim is G > 0 for ln K in [0, X(box)], where X is the upper bound on l2' from `x_upper` (ln K <= l1 is Theorem A, ln K >= l2' is Theorem B', so this covers the band).
- Boxes certified (no unresolved cell): 8 of 36.
- Cells: 1888; unresolved at max depth or cell cap: 0.
- Smallest certified lower bound of G: 0.000107592.
- X over the boxes: [0.04060, 0.11430].
- Wall time 193.1 s with -j 8 (summed box times 482.9 s).
- Label: PROVED (computer-assisted) on each box with no unresolved cell; the pieces are the formula (section 1), the closed-form tails (section 3), the arb cell enclosures, and x_upper. x_upper takes the root of g in floating point and adds a 1e-9 relative margin; g > 0 at that point is then checked in arb with s shrunk below the float s_lo (g increases in s), so the bound is rigorous.

| u | k | rho | X | cells | unresolved | min lower bound | time (s) |
|---|---|---|---|---|---|---|---|
| [0.1, 0.2154] | [0.3, 0.5646] | [-0.9, -0.7] | 0.08937 | 353 | 0 | 0.0001076 | 74.75 |
| [0.1, 0.2154] | [0.3, 0.5646] | [-0.7, -0.5] | 0.07125 | 137 | 0 | 0.002534 | 33.71 |
| [0.1, 0.2154] | [0.3, 0.5646] | [-0.5, -0.3] | 0.05500 | 45 | 0 | 0.00344 | 15.59 |
| [0.1, 0.2154] | [0.3, 0.5646] | [-0.3, -0.1] | 0.04060 | 9 | 0 | 0.005137 | 5.53 |
| [0.1, 0.2154] | [0.5646, 1.063] | [-0.9, -0.7] | 0.11430 | 881 | 0 | 0.0006538 | 191.73 |
| [0.1, 0.2154] | [0.5646, 1.063] | [-0.7, -0.5] | 0.09021 | 297 | 0 | 0.0004161 | 90.44 |
| [0.1, 0.2154] | [0.5646, 1.063] | [-0.5, -0.3] | 0.06799 | 117 | 0 | 0.01356 | 46.65 |
| [0.1, 0.2154] | [0.5646, 1.063] | [-0.3, -0.1] | 0.04765 | 49 | 0 | 0.03934 | 24.48 |
