# Calendar no-arbitrage for ρ ≤ 0 outside a near-the-money band (epic 0016)

2026-10-28. Builds on `calendar_tail.md` (lemma: m(u) is decreasing for ρ < 0). Notation:
F = D = 1, u = √(τQ(τ)), s² = ln(1+k²), ζ = e^{sZ−s²/2}, y = u²ζ, w = (1−ρ²)y, m = E[M],
L = ln x, d2 = (ρ√y Z − y/2 − ln m − L)/√w, so P(S > x) = E_Z[N(d2)], and
p(x) = ∂P(S > x)/∂u. Calendar no-arbitrage in τ follows from ∂C/∂u ≥ 0 when τQ(τ) is
non-decreasing, as in the first paper.
**Outcome: the conjecture is proved at every strike outside a band (ℓ₁, ℓ₂) around the
money; inside the band it reduces to a single sign change of p, which holds at every one of
216 scanned parameter points.**

## Identity (all ρ ∈ (−1, 1))

At fixed Z the correlation term cancels from the u-derivative of d2:

    u ∂d2/∂u = (L̃ − y/2)/√w,   L̃(x) = ln x + ln m − u m'/m,
    u p(x) = E_Z[ φ(d2) (L̃(x) − y/2) / √w ].

(`test_calendar_band.py`: symbolic, and against central differences of P(S > x) computed by
independent Gauss–Hermite quadrature.) Since ∫₀^∞ p(x) dx = ∂E[S]/∂u = 0 and
C(K) = ∫_K^∞ P(S > x) dx,

    ∂C/∂u (K) = ∫_K^∞ p(x) dx = −∫₀^K p(x) dx.

## Theorem A (left wing)

If ln K ≤ ℓ₁ := −ln m(u) + u m'(u)/m(u), then ∂C/∂u(K) > 0.

*Proof.* For x ≤ K, L̃(x) ≤ L̃(K) ≤ 0, so the integrand of u p(x) is negative for every Z
and p(x) < 0. Hence ∂C/∂u(K) = −∫₀^K p > 0. ∎ (The sign of ρ is not used.)

## Theorem B (right wing, ρ ≤ 0)

If 2(ln K + ln m(u)) ≥ u² e^{−s²/2}, i.e. ln K ≥ ℓ₂ := u² e^{−s²/2}/2 − ln m(u), then
∂C/∂u(K) ≥ 0, strictly for ρ < 0.

*Proof.* Fix x with ln x ≥ ℓ₂ and put L' = ln x + ln m > 0. By the lemma m' ≤ 0, so
L̃ ≥ L' and it suffices that E_Z[ω(Z)(2L' − y)] ≥ 0 with ω = φ(Z)φ(d2)/√y. Change variable
to v = ln(y/(2L')), so Z = Z₀ + v/s with Z₀ = (ln(2L'/u²) + s²/2)/s ≥ 0 by assumption, and
write ω ∝ e^{f(v)}, f = −Z²/2 − B²/(2(1−ρ²)y) − ln y/2 with B = L' + y/2 + |ρ|√y Z
(= −d2 √w). Splitting at v = 0,

    ∫ e^{f(v)}(1 − e^v) dv = ∫₀^∞ (1 − e^{−v}) [e^{f(−v)} − e^{f(v)+v}] dv,

which is ≥ 0 if f(−v) − f(v) ≥ v for v > 0. Using (L' + y/2)²/y = L'(1 + cosh v) and
(L' + y/2)/√y = √(2L') cosh(v/2),

    f(−v) − f(v) = v + 2Z₀v/s + [ |ρ|√(2L') cosh(v/2) · 2v/s + ρ² · 2Z₀v/s ] / (1−ρ²),

(symbolic check in `test_calendar_band.py`), and every term after the first is ≥ 0 when
Z₀ ≥ 0, strictly > 0 when ρ < 0. So p(x) ≥ 0 for all ln x ≥ ℓ₂, and
∂C/∂u(K) = ∫_K^∞ p ≥ 0. ∎

## Lemma (ℓ₁ > 0 for ρ < 0)

By the Lambert form of `calendar_tail.md`, m = E[φ(βX)] with φ(q) = 1/(1 + W(q)),
X = e^{sV/2} > 0 and β ∝ u. Since (−ln(1 + W))'(q) = −e^{−W}/(1+W)², which increases with W
and so with q, φ is log-convex; a mixture of log-convex functions is log-convex, so ln m is
convex in u with ln m(0) = 0, which gives u (ln m)' ≥ ln m, i.e. ℓ₁ ≥ 0 (strict for ρ < 0).
Hence **Theorem A covers every strike K ≤ F.** (`test_calendar_band.py`: symbolic
derivative and monotonicity; ℓ₁ > 0 on 25 random points.)

## Theorem B′ (sharper right threshold)

In the proof of Theorem B, (f(−v) − f(v) − v)·s/(2v) = (Z₀ + |ρ|√(2L') cosh(v/2))/(1−ρ²),
which is ≥ 0 for every v > 0 as soon as Z₀ + |ρ|√(2L') ≥ 0, i.e.
g(L') := ln(2L'/u²) + s²/2 + s|ρ|√(2L') ≥ 0. g is increasing, so p(x) ≥ 0 and
∂C/∂u(K) ≥ 0 for ln K ≥ ℓ₂' := L'* − ln m, with L'* the root of g (ℓ₂' ≤ ℓ₂, equal at
ρ = 0). On the scan grid this removes up to 79% of the band (median about 2%).
(`test_calendar_band.py`: ℓ₂' ≤ ℓ₂ and p ≥ 0 beyond ℓ₂' on random points.)

## Reduction of the band to a slope bound

Write u p(x) = A(L)·h(L), with A(L) = E_Z[φ(d2)/√w] > 0 and
h(L) = L + c − R(L), where c = ln m − u m'/m and R(L) = E_ν[y/2], ν ∝ φ(Z)φ(d2)/√w.
Since ∂_L ln φ(d2) = d2/√w,

    R'(L) = Cov_ν(y/2, d2/√w).

By Theorem A, h(ℓ₁) ≤ 0; by Theorem B′, h(ℓ₂') ≥ 0. **If R'(L) < 1 on [ℓ₁, ℓ₂'] then h is
strictly increasing there, p changes sign exactly once, and the conjecture holds at every
strike.**

*Interpretation.* φ(Z)φ(d2)/√w is the joint density of (Z, ln S), so ν is the law of Z
given ln S = L and R(L) = E[y/2 | ln S = L]: the condition says the conditional mean of half
the conditional total variance grows more slowly than the log-price. With X' = ln S + ln m,
X' = √y W' − y/2 where W' = −|ρ|Z + √(1−ρ²) ε is standard normal.

*Sharpness.* The bound cannot be proved by crude estimates near ρ = 0: as u → ∞ with L' in
the band (L' ∝ u²), the conditional law concentrates where the y/8 and L'²/(2y) terms of
the exponent balance, at y ≈ 2L', so R(L) ≈ L' and R' → 1 from below. This is the
heuristic for ρ = 0; the correlation terms are of lower order in u, so it should extend to
every fixed ρ, but the approach is slow at strong correlation and is not verified there.
Numerically (scan below): max R' = 0.836 on the 216-point grid, always at the right end
ℓ₂'; on the stress grid it reaches 0.9978 at u = 50, ρ = −10⁻⁶ and 0.943 at u = 50,
ρ = −0.2, but only 0.78 at ρ = −0.49 and 0.18 at ρ = −0.9 (u = 50), and stays negative at
ρ = −0.99. Never ≥ 1. At ρ = 0 the conjecture is trivially true (mixture of Black prices in
their variance), but R' → 1 there too, so a proof of R' < 1 needs an exact argument where
ρ is small.

*Log-concavity.* In t = ln y (Z = (t − t₀)/s), ν ∝ e^{−V} with
V = Z²/2 + t/2 + q²/(2(1−ρ²)), q = L'e^{−t/2} + e^{t/2}/2 + |ρ|Z = −√(1−ρ²) d2, and
q'' = L'e^{−t/2}/4 + e^{t/2}/8 > 0. So s²V'' = 1 + s²(q'² + q q'')/(1−ρ²) ≥ 1 wherever
q ≥ 0 (d2 ≤ 0): the conditional law is at least as log-concave as the prior there, and can
lose log-concavity only where d2 > 0, which needs |ρ|Z < −√(2L'). Scan: the smallest s²V''
on the band is ≥ 0.70 for ρ ∈ {−0.01, −0.2} at every u up to 50, and negative from
ρ = −0.49 on (down to −102 at ρ = −0.99). The two difficulties are complementary: where the
margin of R' < 1 is thin (small |ρ|, large u), ν is strongly log-concave; where ν is not
log-concave (strong |ρ|), R' is far below 1.

*Comparison lemma (proved).* Let ν ∝ e^{−V} on ℝ with V' → ±∞ at ±∞, f' ≥ 0 and
g' ≤ M V''. Then Cov_ν(f, g) ≤ M E_ν[f']. *Proof.* Hoeffding's formula gives
Cov(f, g) = ∫ f'(t) K_g(t) dt with K_g(t) = ∫_t^∞ (g − E g) dν. With D = M V' − g,
D' ≥ 0, so ∫_t^∞ (D − E D) dν ≥ 0, i.e. K_g(t) ≤ K_{MV'}(t) = M ∫_t^∞ V' e^{−V}/Z_ν = M ν(t)
(E V' = 0). Hence Cov(f, g) ≤ M ∫ f' ν. ∎ (`test_calendar_band.py`: numeric check.)
For f = y/2 = e^t/2 and g = ∂_L ln ν = d2/√w this gives R' ≤ M·R(L) with
M = sup g'/V''. At ρ = 0 and the mode y = 2L', g'/V'' = 1/(L' + 2/s²) and R ≈ L', so the
bound reproduces the leading order R' ≈ L'/(L' + 2/s²) < 1, with the prior curvature 1/s²
as the margin. The global sup is too crude, though: as t → −∞, g'/V'' → 2/L', which would
only give R' ≲ 2. A proof along these lines needs a localised version (K_g(t) ≤ M(t) ν(t)
with M(t) from a comparison function adapted to the tail).

*Exact split of R' (all ρ ≤ 0, r = |ρ|).* Since d2/√w = −(L'/y + 1/2 + rZ/√y)/(1−ρ²),

    (1−ρ²) R' = (L'/2)(E_ν[y] E_ν[1/y] − 1) − (r/2) Cov_ν(y, Z/√y),

and with X, Y independent with law ν in t = ln y, E[y]E[1/y] − 1 = 2E[sinh²((X−Y)/2)], so
the first part is L'·E[sinh²((X−Y)/2)] (`test_calendar_band.py`: identity against
`r_slope`). Scan: Cov_ν(y, Z/√y) > 0 at every point (smallest 2e-5), so the correlation
term always lowers R'; but first/(1−ρ²) alone stays below 1 only near ρ = 0 (≤ 0.997 at
ρ ∈ {−10⁻⁶, −0.01}) and exceeds 1 from ρ = −0.2 at large u (1.027 at u = 50), up to 67 at
ρ = −0.99. So for ρ < 0 the sign of the covariance is not enough: the proof needs a
quantitative lower bound on Cov_ν(y, Z/√y) against the 1/(1−ρ²) loss.

*Proposition (ρ = 0; proved, a template).* At ρ = 0, V(t) = (t − t₀)²/(2s²) + t/2 +
L'²e^{−t}/2 + e^t/8, so V'' ≥ κ := 1/s² + |L'|/2 by AM–GM. By Caffarelli's contraction
theorem (in one dimension: the monotone map from N(0, 1/κ) to a measure with V'' ≥ κ is
1-Lipschitz; reference to be verified before citing) — superseded: `calendar_remainder.md` proves the same bound
without Caffarelli, by Brascamp–Lieb on the exponential tilts of ν (citation still to be
verified), |X − Y| is dominated by |G₁ − G₂| with
G₁ − G₂ ~ N(0, 2/κ), and sinh² is even and increasing in |·|, so

    R' = L' E[sinh²((X−Y)/2)] ≤ (L'/2)(e^{1/κ} − 1) = (κ − 1/s²)(e^{1/κ} − 1)   (L' ≥ 0).

With x = 1/κ and c = 1/s², (1 − cx)(e^x − 1) < x for c ≥ 1/2 (at c = 1/2 the difference
e^x(1 − x/2) − 1 − x/2 vanishes at 0 with zero slope and second derivative −x e^x/2 < 0).
Hence **at ρ = 0, R' < 1 for every L whenever s² ≤ 2 (k ≤ √(e² − 1) ≈ 2.53)**; for
L' < 0, R' ≤ 0. For s² > 2 the uniform κ is too weak at large L' (the bound tends to 1 from
above). The conjecture itself is trivial at ρ = 0; the point is the mechanism: the margin is
the prior curvature 1/s², exactly as in the leading-order heuristic. (Tests: the bound
against `r_slope` at ρ = 0, and the scalar inequality.)

*The unsplit form and a local margin.* Splitting R' loses information: at large u the
1/(1−ρ²) excess of the first part is cancelled by the covariance term. Write
R' = Cov_ν(y, G) with G = −(L'/y + |ρ|Z/√y)/(2(1−ρ²)). In t, with a = L'e^{−t/2},
b = e^{t/2}/2 (so q = a + b + rZ),

    Λ(t) := (1−ρ²)(V''(t) − e^t G'(t))
          = (1−ρ²)/s² + q'² + (a−b)²/4 + r b/s + r Z (a−b)/4

(symbolic check in `test_calendar_band.py`). Laplace's method gives R' ≈ e^{t*}G'(t*)/V''(t*)
at the mode t*, so R' < 1 at leading order iff Λ(t*) > 0. At large u the mode has a ≈ b
(e^{t*} ≈ 2L'), the only term that can be negative vanishes, and the margin is
(1−ρ²)/s² + r b/s > 0, which grows like |ρ|√L' instead of shrinking. The correlation helps;
only at ρ = 0 is the margin as thin as 1/s².

*Comparison in y.* In the variable y = e^t, f = y is linear and ν has potential
W(y) = V(ln y) + ln y. The Stein identity gives, for any ψ > 0 with Γ = ψW' − ψ',
∫_y^∞ Γ dν = ψ(y)ν(y); so if Γ' ≥ dG/dy everywhere, Hoeffding's formula gives
R' = Cov_ν(y, G) ≤ E_ν[ψ]. With ψ ≡ 1 the condition is e^t G' ≤ V'' − V' − 1, which is
Λ > 0 up to the −(1−ρ²)(V' + 1) term (≈ 0 near the mode). It fails in the right tail, where
V'' − V' − 1 < 0 while e^tG' > 0, and the exact Stein kernel τ = K_G/ν confirms that no
pointwise version works there: e^t τ(t) → −8(1−ρ²)E_ν[G] ≈ 2 as t → ∞ at large u. So a proof
needs a weight ψ that exceeds 1 in the right tail and falls below 1 in the bulk, with
E_ν[ψ] < 1, or a second-order Laplace expansion with an explicit remainder.

*Linear weights fail.* For ψ = β(y + c) the condition reads
β[e^t V'' + c(V'' − V' − 1)] ≥ e^t G'(t) for all t, giving R' ≤ β(E_ν[y] + c); the tails are
fine for any β, c > 0, and c ≈ e^{t*} removes the first-order term at the mode. An ad hoc
search over c (not scripted) found bounds above 1 at many small-|ρ| points (large u) and no
feasible (β, c) at many strong-|ρ| points, where V'' < 0 somewhere. The margin at large u is
O(1/L') relative, the same order as the second-order terms a linear weight ignores, so ψ
must track the curvature of ν; this is equivalent in difficulty to a second-order Laplace
expansion with remainder.

*A route that fails.* Integrating by parts in Z, using
(L' − y/2)/√y = (2/s)(√(1−ρ²) ∂_Z d2 + |ρ|),

    u p(x) = (2/s) E[Z N(d2)] + (2|ρ|/(s√(1−ρ²))) E[φ(d2)] − (u m'/m) E[φ(d2)/√w],

and integrating over x ≥ K, u ∂C/∂u(K) = (2/s) E[Z c(Z, K)] + (non-negative terms), with
c(Z, K) the conditional Black call. So Cov(Z, c(Z, K)) ≥ 0 would suffice, but an ad hoc
check (not scripted) found it negative at most grid strikes: the positive correlation terms
are essential, and this does not simplify the problem.

## Corollary and what remains

For ρ ≤ 0 the CRL call price is non-decreasing in u, hence in τ when τQ(τ) is
non-decreasing, at every strike with ln(K/F) ≤ ℓ₁ or ln(K/F) ≥ ℓ₂. Inside the band, the
function K ↦ ∂C/∂u(K) = ∫_K^∞ p vanishes at K = 0 and K = ∞; if p changes sign exactly once
(negative, then positive) this function is non-negative everywhere, which is the
Karlin–Novikoff cut criterion for convex order. **The conjecture therefore follows from a
single sign change of p(x) = ∂P(S > x)/∂u inside (ℓ₁, ℓ₂).** By Theorems A and B, p ≤ 0
below the band and p ≥ 0 above it; the open question is only what happens inside.

Evidence (scan below): on 216 points (u from 0.02 to 5, k from 0.05 to 6, ρ from −0.01 to
−0.99), p changes sign exactly once at every point, never has the wrong sign beyond ℓ₁ or
ℓ₂, and matches finite differences to 2e-6. The band width ℓ₂ − ℓ₁ ranges from 1.5e-4
(u = 0.02) to 12.6 (u = 5, k = 6). ℓ₁ ≥ 0 is now proved (lemma above).

Next step: prove R'(L) < 1 on [ℓ₁, ℓ₂'] from the unsplit form: construct ψ with
(ψW' − ψ')' ≥ dG/dy and E_ν[ψ] < 1 (bulk: ψ ≈ 1 − Λ-margin; right tail: ψ ≈ 2), or carry
the Laplace expansion of Cov_ν(y, G) to second order with an explicit remainder. The
leading-order margin Λ(t*) > 0 is the target the correction terms must not exceed.

*Grid caveat.* `Model` integrates on a fixed Z-grid ([−30, z_hi], step 5e-4). Inside the
band its R' agrees with a widened grid to 6e-11 at every scan and stress point (ad hoc
check), but far beyond the band at small k and large u the conditional mass can leave the
grid (found by the 03-remainder worker at L' = 3000, k = 0.05). The scan's sign-change counts
beyond ℓ₂ are therefore unreliable there; the sign of p beyond ℓ₂' is proved (Theorem B').

<!-- scan -->
# Sign structure of dP(S > x)/du for rho <= 0

Output of `uv run python derivations/calendar_band.py`; formulas in the module docstring. 216 parameter points (u in (0.02, 0.1, 0.3, 1.0, 2.0, 5.0), k in (0.05, 0.3, 1.0, 2.2, 3.0, 6.0), rho in (-0.01, -0.2, -0.49, -0.7, -0.9, -0.99)); p(x) on 400 log-strikes spanning the band [l1, l2] and ten band-widths (at least 2) either side; signs counted where |p| exceeds 1e-09 of its maximum.

- Points where p changes sign other than exactly once: 0.
- Log-strikes <= l1 with p > 0: 0; >= l2 with p < 0: 0.
- Formula vs finite difference of P at the band centre: max relative error 1.94e-06.
- Band width l2 - l1: from 1.54e-04 to 12.574; l1 from 0.0000 to 0.3803, l2 from 0.0002 to 12.5894.
- Sharp threshold l2' (calendar_band.md): band shrinks by a fraction 9.74e-06 to 0.791 of l2 - l1.
- Largest R'(L) on [l1, l2'] (60 points per parameter point): 0.8355; points with R' >= 1: 0.
- Smallest s^2 V'' (log-concavity of nu in t = ln y; 1 = prior curvature) on [l1, l2'] by rho: -0.01: 1.00, -0.2: 0.70, -0.49: -1.06, -0.7: -5.66, -0.9: -19.11, -0.99: -63.45.
- Split R' = (first - |rho| cov / 2) / (1 - rho^2) on [l1, l2'] (20 points): smallest cov = Cov(y, Z/sqrt y) 1.98e-05; largest first/(1 - rho^2) by rho: -0.01: 0.839, -0.2: 0.802, -0.49: 0.936, -0.7: 1.362, -0.9: 3.484, -0.99: 33.773.

| u | k | rho | l1 | l2 | l2' | max R' | sign changes |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.05 | -0.01 | 0.0000 | 0.0002 | 0.0002 | -0.000 | 1 |
| 0.02 | 0.05 | -0.2 | 0.0000 | 0.0003 | 0.0003 | -0.000 | 1 |
| 0.02 | 0.05 | -0.49 | 0.0000 | 0.0004 | 0.0004 | -0.000 | 1 |
| 0.02 | 0.05 | -0.7 | 0.0000 | 0.0005 | 0.0005 | -0.000 | 1 |
| 0.02 | 0.05 | -0.9 | 0.0000 | 0.0006 | 0.0006 | -0.000 | 1 |
| 0.02 | 0.05 | -0.99 | 0.0000 | 0.0007 | 0.0007 | -0.000 | 1 |
| 0.02 | 0.3 | -0.01 | 0.0000 | 0.0002 | 0.0002 | -0.000 | 1 |
| 0.02 | 0.3 | -0.2 | 0.0000 | 0.0008 | 0.0008 | -0.001 | 1 |
| 0.02 | 0.3 | -0.49 | 0.0000 | 0.0016 | 0.0016 | -0.001 | 1 |
| 0.02 | 0.3 | -0.7 | 0.0000 | 0.0022 | 0.0022 | -0.002 | 1 |
| 0.02 | 0.3 | -0.9 | 0.0000 | 0.0028 | 0.0028 | -0.003 | 1 |
| 0.02 | 0.3 | -0.99 | 0.0000 | 0.0031 | 0.0031 | -0.003 | 1 |
| 0.02 | 1.0 | -0.01 | 0.0000 | 0.0002 | 0.0002 | -0.000 | 1 |
| 0.02 | 1.0 | -0.2 | 0.0000 | 0.0017 | 0.0017 | -0.002 | 1 |
| 0.02 | 1.0 | -0.49 | 0.0000 | 0.0039 | 0.0039 | -0.004 | 1 |
| 0.02 | 1.0 | -0.7 | 0.0001 | 0.0054 | 0.0054 | -0.006 | 1 |
| 0.02 | 1.0 | -0.9 | 0.0001 | 0.0069 | 0.0069 | -0.007 | 1 |
| 0.02 | 1.0 | -0.99 | 0.0001 | 0.0076 | 0.0076 | -0.007 | 1 |
| 0.02 | 2.2 | -0.01 | 0.0000 | 0.0002 | 0.0002 | 0.000 | 1 |
| 0.02 | 2.2 | -0.2 | 0.0000 | 0.0022 | 0.0022 | -0.003 | 1 |
| 0.02 | 2.2 | -0.49 | 0.0001 | 0.0052 | 0.0052 | -0.007 | 1 |
| 0.02 | 2.2 | -0.7 | 0.0001 | 0.0074 | 0.0074 | -0.008 | 1 |
| 0.02 | 2.2 | -0.9 | 0.0002 | 0.0094 | 0.0094 | -0.009 | 1 |
| 0.02 | 2.2 | -0.99 | 0.0003 | 0.0104 | 0.0104 | -0.008 | 1 |
| 0.02 | 3.0 | -0.01 | 0.0000 | 0.0002 | 0.0002 | 0.000 | 1 |
| 0.02 | 3.0 | -0.2 | 0.0000 | 0.0023 | 0.0023 | -0.004 | 1 |
| 0.02 | 3.0 | -0.49 | 0.0001 | 0.0055 | 0.0055 | -0.008 | 1 |
| 0.02 | 3.0 | -0.7 | 0.0002 | 0.0078 | 0.0078 | -0.009 | 1 |
| 0.02 | 3.0 | -0.9 | 0.0003 | 0.0100 | 0.0100 | -0.009 | 1 |
| 0.02 | 3.0 | -0.99 | 0.0003 | 0.0110 | 0.0110 | -0.009 | 1 |
| 0.02 | 6.0 | -0.01 | 0.0000 | 0.0002 | 0.0002 | 0.000 | 1 |
| 0.02 | 6.0 | -0.2 | 0.0000 | 0.0024 | 0.0024 | -0.005 | 1 |
| 0.02 | 6.0 | -0.49 | 0.0001 | 0.0058 | 0.0058 | -0.010 | 1 |
| 0.02 | 6.0 | -0.7 | 0.0003 | 0.0082 | 0.0082 | -0.010 | 1 |
| 0.02 | 6.0 | -0.9 | 0.0004 | 0.0105 | 0.0105 | -0.008 | 1 |
| 0.02 | 6.0 | -0.99 | 0.0005 | 0.0115 | 0.0115 | -0.008 | 1 |
| 0.1 | 0.05 | -0.01 | 0.0000 | 0.0050 | 0.0050 | -0.000 | 1 |
| 0.1 | 0.05 | -0.2 | 0.0000 | 0.0055 | 0.0055 | -0.000 | 1 |
| 0.1 | 0.05 | -0.49 | 0.0000 | 0.0062 | 0.0062 | -0.001 | 1 |
| 0.1 | 0.05 | -0.7 | 0.0000 | 0.0067 | 0.0067 | -0.002 | 1 |
| 0.1 | 0.05 | -0.9 | 0.0000 | 0.0072 | 0.0072 | -0.002 | 1 |
| 0.1 | 0.05 | -0.99 | 0.0000 | 0.0075 | 0.0074 | -0.002 | 1 |
| 0.1 | 0.3 | -0.01 | 0.0000 | 0.0049 | 0.0049 | 0.000 | 1 |
| 0.1 | 0.3 | -0.2 | 0.0000 | 0.0077 | 0.0077 | -0.003 | 1 |
| 0.1 | 0.3 | -0.49 | 0.0001 | 0.0118 | 0.0118 | -0.007 | 1 |
| 0.1 | 0.3 | -0.7 | 0.0002 | 0.0148 | 0.0147 | -0.010 | 1 |
| 0.1 | 0.3 | -0.9 | 0.0003 | 0.0176 | 0.0175 | -0.013 | 1 |
| 0.1 | 0.3 | -0.99 | 0.0003 | 0.0189 | 0.0187 | -0.014 | 1 |
| 0.1 | 1.0 | -0.01 | 0.0000 | 0.0039 | 0.0039 | 0.001 | 1 |
| 0.1 | 1.0 | -0.2 | 0.0001 | 0.0111 | 0.0110 | -0.007 | 1 |
| 0.1 | 1.0 | -0.49 | 0.0006 | 0.0216 | 0.0215 | -0.019 | 1 |
| 0.1 | 1.0 | -0.7 | 0.0012 | 0.0290 | 0.0288 | -0.027 | 1 |
| 0.1 | 1.0 | -0.9 | 0.0018 | 0.0359 | 0.0357 | -0.032 | 1 |
| 0.1 | 1.0 | -0.99 | 0.0022 | 0.0389 | 0.0387 | -0.034 | 1 |
| 0.1 | 2.2 | -0.01 | 0.0000 | 0.0026 | 0.0026 | 0.004 | 1 |
| 0.1 | 2.2 | -0.2 | 0.0003 | 0.0124 | 0.0124 | -0.011 | 1 |
| 0.1 | 2.2 | -0.49 | 0.0014 | 0.0266 | 0.0265 | -0.032 | 1 |
| 0.1 | 2.2 | -0.7 | 0.0026 | 0.0363 | 0.0362 | -0.040 | 1 |
| 0.1 | 2.2 | -0.9 | 0.0040 | 0.0451 | 0.0450 | -0.042 | 1 |
| 0.1 | 2.2 | -0.99 | 0.0047 | 0.0490 | 0.0488 | -0.041 | 1 |
| 0.1 | 3.0 | -0.01 | 0.0000 | 0.0021 | 0.0021 | 0.006 | 1 |
| 0.1 | 3.0 | -0.2 | 0.0003 | 0.0126 | 0.0126 | -0.013 | 1 |
| 0.1 | 3.0 | -0.49 | 0.0017 | 0.0274 | 0.0274 | -0.037 | 1 |
| 0.1 | 3.0 | -0.7 | 0.0032 | 0.0375 | 0.0374 | -0.044 | 1 |
| 0.1 | 3.0 | -0.9 | 0.0048 | 0.0467 | 0.0465 | -0.043 | 1 |
| 0.1 | 3.0 | -0.99 | 0.0056 | 0.0506 | 0.0505 | -0.041 | 1 |
| 0.1 | 6.0 | -0.01 | 0.0000 | 0.0014 | 0.0014 | 0.012 | 1 |
| 0.1 | 6.0 | -0.2 | 0.0005 | 0.0124 | 0.0123 | -0.015 | 1 |
| 0.1 | 6.0 | -0.49 | 0.0023 | 0.0275 | 0.0275 | -0.048 | 1 |
| 0.1 | 6.0 | -0.7 | 0.0041 | 0.0376 | 0.0376 | -0.046 | 1 |
| 0.1 | 6.0 | -0.9 | 0.0061 | 0.0467 | 0.0466 | -0.041 | 1 |
| 0.1 | 6.0 | -0.99 | 0.0070 | 0.0506 | 0.0506 | -0.038 | 1 |
| 0.3 | 0.05 | -0.01 | 0.0000 | 0.0450 | 0.0450 | -0.000 | 1 |
| 0.3 | 0.05 | -0.2 | 0.0000 | 0.0464 | 0.0463 | -0.001 | 1 |
| 0.3 | 0.05 | -0.49 | 0.0000 | 0.0486 | 0.0483 | -0.004 | 1 |
| 0.3 | 0.05 | -0.7 | 0.0000 | 0.0501 | 0.0497 | -0.005 | 1 |
| 0.3 | 0.05 | -0.9 | 0.0001 | 0.0516 | 0.0510 | -0.007 | 1 |
| 0.3 | 0.05 | -0.99 | 0.0001 | 0.0523 | 0.0516 | -0.007 | 1 |
| 0.3 | 0.3 | -0.01 | 0.0000 | 0.0435 | 0.0435 | 0.001 | 1 |
| 0.3 | 0.3 | -0.2 | 0.0001 | 0.0517 | 0.0510 | -0.007 | 1 |
| 0.3 | 0.3 | -0.49 | 0.0006 | 0.0638 | 0.0620 | -0.020 | 1 |
| 0.3 | 0.3 | -0.7 | 0.0013 | 0.0722 | 0.0698 | -0.029 | 1 |
| 0.3 | 0.3 | -0.9 | 0.0020 | 0.0801 | 0.0770 | -0.037 | 1 |
| 0.3 | 0.3 | -0.99 | 0.0024 | 0.0836 | 0.0802 | -0.041 | 1 |
| 0.3 | 1.0 | -0.01 | 0.0000 | 0.0330 | 0.0329 | 0.014 | 1 |
| 0.3 | 1.0 | -0.2 | 0.0009 | 0.0538 | 0.0525 | -0.012 | 1 |
| 0.3 | 1.0 | -0.49 | 0.0044 | 0.0828 | 0.0799 | -0.050 | 1 |
| 0.3 | 1.0 | -0.7 | 0.0082 | 0.1021 | 0.0980 | -0.073 | 1 |
| 0.3 | 1.0 | -0.9 | 0.0123 | 0.1193 | 0.1142 | -0.090 | 1 |
| 0.3 | 1.0 | -0.99 | 0.0144 | 0.1267 | 0.1212 | -0.095 | 1 |
| 0.3 | 2.2 | -0.01 | 0.0000 | 0.0202 | 0.0202 | 0.038 | 1 |
| 0.3 | 2.2 | -0.2 | 0.0020 | 0.0483 | 0.0474 | -0.007 | 1 |
| 0.3 | 2.2 | -0.49 | 0.0090 | 0.0852 | 0.0831 | -0.075 | 1 |
| 0.3 | 2.2 | -0.7 | 0.0154 | 0.1086 | 0.1058 | -0.109 | 1 |
| 0.3 | 2.2 | -0.9 | 0.0222 | 0.1291 | 0.1256 | -0.115 | 1 |
| 0.3 | 2.2 | -0.99 | 0.0253 | 0.1378 | 0.1340 | -0.114 | 1 |
| 0.3 | 3.0 | -0.01 | 0.0000 | 0.0159 | 0.0159 | 0.049 | 1 |
| 0.3 | 3.0 | -0.2 | 0.0025 | 0.0454 | 0.0447 | -0.001 | 1 |
| 0.3 | 3.0 | -0.49 | 0.0104 | 0.0834 | 0.0818 | -0.084 | 1 |
| 0.3 | 3.0 | -0.7 | 0.0175 | 0.1072 | 0.1050 | -0.122 | 1 |
| 0.3 | 3.0 | -0.9 | 0.0248 | 0.1278 | 0.1252 | -0.118 | 1 |
| 0.3 | 3.0 | -0.99 | 0.0281 | 0.1365 | 0.1337 | -0.114 | 1 |
| 0.3 | 6.0 | -0.01 | 0.0000 | 0.0092 | 0.0092 | 0.075 | 1 |
| 0.3 | 6.0 | -0.2 | 0.0032 | 0.0395 | 0.0391 | 0.015 | 1 |
| 0.3 | 6.0 | -0.49 | 0.0124 | 0.0768 | 0.0761 | -0.094 | 1 |
| 0.3 | 6.0 | -0.7 | 0.0200 | 0.0999 | 0.0988 | -0.124 | 1 |
| 0.3 | 6.0 | -0.9 | 0.0276 | 0.1196 | 0.1183 | -0.112 | 1 |
| 0.3 | 6.0 | -0.99 | 0.0310 | 0.1279 | 0.1265 | -0.105 | 1 |
| 1.0 | 0.05 | -0.01 | 0.0000 | 0.4996 | 0.4994 | 0.000 | 1 |
| 1.0 | 0.05 | -0.2 | 0.0000 | 0.5043 | 0.4994 | -0.004 | 1 |
| 1.0 | 0.05 | -0.49 | 0.0002 | 0.5114 | 0.4995 | -0.012 | 1 |
| 1.0 | 0.05 | -0.7 | 0.0004 | 0.5164 | 0.4995 | -0.017 | 1 |
| 1.0 | 0.05 | -0.9 | 0.0007 | 0.5211 | 0.4996 | -0.022 | 1 |
| 1.0 | 0.05 | -0.99 | 0.0008 | 0.5232 | 0.4997 | -0.024 | 1 |
| 1.0 | 0.3 | -0.01 | 0.0000 | 0.4804 | 0.4790 | 0.020 | 1 |
| 1.0 | 0.3 | -0.2 | 0.0012 | 0.5067 | 0.4807 | -0.009 | 1 |
| 1.0 | 0.3 | -0.49 | 0.0060 | 0.5432 | 0.4841 | -0.053 | 1 |
| 1.0 | 0.3 | -0.7 | 0.0111 | 0.5673 | 0.4870 | -0.084 | 1 |
| 1.0 | 0.3 | -0.9 | 0.0169 | 0.5886 | 0.4900 | -0.113 | 1 |
| 1.0 | 0.3 | -0.99 | 0.0196 | 0.5977 | 0.4915 | -0.125 | 1 |
| 1.0 | 1.0 | -0.01 | 0.0000 | 0.3573 | 0.3549 | 0.138 | 1 |
| 1.0 | 1.0 | -0.2 | 0.0075 | 0.4209 | 0.3774 | 0.054 | 1 |
| 1.0 | 1.0 | -0.49 | 0.0306 | 0.4965 | 0.4060 | -0.080 | 1 |
| 1.0 | 1.0 | -0.7 | 0.0498 | 0.5411 | 0.4243 | -0.175 | 1 |
| 1.0 | 1.0 | -0.9 | 0.0685 | 0.5780 | 0.4405 | -0.245 | 1 |
| 1.0 | 1.0 | -0.99 | 0.0769 | 0.5932 | 0.4475 | -0.259 | 1 |
| 1.0 | 2.2 | -0.01 | 0.0001 | 0.2122 | 0.2104 | 0.253 | 1 |
| 1.0 | 2.2 | -0.2 | 0.0144 | 0.2933 | 0.2631 | 0.147 | 1 |
| 1.0 | 2.2 | -0.49 | 0.0486 | 0.3807 | 0.3194 | -0.044 | 1 |
| 1.0 | 2.2 | -0.7 | 0.0738 | 0.4296 | 0.3517 | -0.226 | 1 |
| 1.0 | 2.2 | -0.9 | 0.0968 | 0.4691 | 0.3785 | -0.304 | 1 |
| 1.0 | 2.2 | -0.99 | 0.1067 | 0.4852 | 0.3896 | -0.300 | 1 |
| 1.0 | 3.0 | -0.01 | 0.0001 | 0.1637 | 0.1624 | 0.283 | 1 |
| 1.0 | 3.0 | -0.2 | 0.0164 | 0.2475 | 0.2244 | 0.177 | 1 |
| 1.0 | 3.0 | -0.49 | 0.0524 | 0.3350 | 0.2883 | -0.018 | 1 |
| 1.0 | 3.0 | -0.7 | 0.0781 | 0.3835 | 0.3241 | -0.223 | 1 |
| 1.0 | 3.0 | -0.9 | 0.1012 | 0.4226 | 0.3534 | -0.307 | 1 |
| 1.0 | 3.0 | -0.99 | 0.1111 | 0.4384 | 0.3654 | -0.300 | 1 |
| 1.0 | 6.0 | -0.01 | 0.0001 | 0.0881 | 0.0875 | 0.319 | 1 |
| 1.0 | 6.0 | -0.2 | 0.0188 | 0.1712 | 0.1602 | 0.216 | 1 |
| 1.0 | 6.0 | -0.49 | 0.0550 | 0.2543 | 0.2317 | 0.032 | 1 |
| 1.0 | 6.0 | -0.7 | 0.0796 | 0.2998 | 0.2708 | -0.175 | 1 |
| 1.0 | 6.0 | -0.9 | 0.1013 | 0.3363 | 0.3024 | -0.294 | 1 |
| 1.0 | 6.0 | -0.99 | 0.1106 | 0.3511 | 0.3153 | -0.280 | 1 |
| 2.0 | 0.05 | -0.01 | 0.0000 | 1.9980 | 1.9960 | 0.002 | 1 |
| 2.0 | 0.05 | -0.2 | 0.0001 | 2.0073 | 1.9682 | -0.008 | 1 |
| 2.0 | 0.05 | -0.49 | 0.0008 | 2.0211 | 1.9279 | -0.022 | 1 |
| 2.0 | 0.05 | -0.7 | 0.0016 | 2.0307 | 1.9002 | -0.033 | 1 |
| 2.0 | 0.05 | -0.9 | 0.0026 | 2.0397 | 1.8749 | -0.043 | 1 |
| 2.0 | 0.05 | -0.99 | 0.0031 | 2.0436 | 1.8638 | -0.047 | 1 |
| 2.0 | 0.3 | -0.01 | 0.0000 | 1.9185 | 1.9076 | 0.076 | 1 |
| 2.0 | 0.3 | -0.2 | 0.0042 | 1.9691 | 1.7715 | 0.017 | 1 |
| 2.0 | 0.3 | -0.49 | 0.0193 | 2.0335 | 1.6117 | -0.073 | 1 |
| 2.0 | 0.3 | -0.7 | 0.0334 | 2.0731 | 1.5217 | -0.138 | 1 |
| 2.0 | 0.3 | -0.9 | 0.0479 | 2.1066 | 1.4505 | -0.199 | 1 |
| 2.0 | 0.3 | -0.99 | 0.0545 | 2.1206 | 1.4222 | -0.223 | 1 |
| 2.0 | 1.0 | -0.01 | 0.0001 | 1.4217 | 1.4022 | 0.370 | 1 |
| 2.0 | 1.0 | -0.2 | 0.0227 | 1.5358 | 1.2258 | 0.226 | 1 |
| 2.0 | 1.0 | -0.49 | 0.0759 | 1.6522 | 1.0728 | -0.011 | 1 |
| 2.0 | 1.0 | -0.7 | 0.1136 | 1.7145 | 1.0074 | -0.215 | 1 |
| 2.0 | 1.0 | -0.9 | 0.1469 | 1.7634 | 0.9649 | -0.394 | 1 |
| 2.0 | 1.0 | -0.99 | 0.1610 | 1.7829 | 0.9502 | -0.417 | 1 |
| 2.0 | 2.2 | -0.01 | 0.0003 | 0.8380 | 0.8241 | 0.514 | 1 |
| 2.0 | 2.2 | -0.2 | 0.0376 | 0.9774 | 0.7660 | 0.366 | 1 |
| 2.0 | 2.2 | -0.49 | 0.1056 | 1.1042 | 0.7237 | 0.112 | 1 |
| 2.0 | 2.2 | -0.7 | 0.1489 | 1.1691 | 0.7128 | -0.159 | 1 |
| 2.0 | 2.2 | -0.9 | 0.1855 | 1.2192 | 0.7102 | -0.471 | 1 |
| 2.0 | 2.2 | -0.99 | 0.2007 | 1.2391 | 0.7107 | -0.468 | 1 |
| 2.0 | 3.0 | -0.01 | 0.0003 | 0.6435 | 0.6329 | 0.536 | 1 |
| 2.0 | 3.0 | -0.2 | 0.0411 | 0.7854 | 0.6240 | 0.393 | 1 |
| 2.0 | 3.0 | -0.49 | 0.1100 | 0.9110 | 0.6205 | 0.151 | 1 |
| 2.0 | 3.0 | -0.7 | 0.1530 | 0.9750 | 0.6265 | -0.111 | 1 |
| 2.0 | 3.0 | -0.9 | 0.1891 | 1.0243 | 0.6356 | -0.476 | 1 |
| 2.0 | 3.0 | -0.99 | 0.2041 | 1.0438 | 0.6403 | -0.467 | 1 |
| 2.0 | 6.0 | -0.01 | 0.0005 | 0.3403 | 0.3353 | 0.547 | 1 |
| 2.0 | 6.0 | -0.2 | 0.0439 | 0.4783 | 0.4007 | 0.417 | 1 |
| 2.0 | 6.0 | -0.49 | 0.1096 | 0.5961 | 0.4536 | 0.202 | 1 |
| 2.0 | 6.0 | -0.7 | 0.1495 | 0.6560 | 0.4835 | -0.021 | 1 |
| 2.0 | 6.0 | -0.9 | 0.1828 | 0.7023 | 0.5087 | -0.457 | 1 |
| 2.0 | 6.0 | -0.99 | 0.1966 | 0.7207 | 0.5192 | -0.439 | 1 |
| 5.0 | 0.05 | -0.01 | 0.0000 | 12.4857 | 12.4546 | 0.014 | 1 |
| 5.0 | 0.05 | -0.2 | 0.0009 | 12.5085 | 11.9146 | -0.010 | 1 |
| 5.0 | 0.05 | -0.49 | 0.0045 | 12.5406 | 11.1790 | -0.047 | 1 |
| 5.0 | 0.05 | -0.7 | 0.0085 | 12.5620 | 10.7028 | -0.073 | 1 |
| 5.0 | 0.05 | -0.9 | 0.0130 | 12.5811 | 10.2872 | -0.099 | 1 |
| 5.0 | 0.05 | -0.99 | 0.0153 | 12.5894 | 10.1109 | -0.110 | 1 |
| 5.0 | 0.3 | -0.01 | 0.0001 | 11.9800 | 11.8105 | 0.339 | 1 |
| 5.0 | 0.3 | -0.2 | 0.0200 | 12.0926 | 9.4150 | 0.192 | 1 |
| 5.0 | 0.3 | -0.49 | 0.0721 | 12.2115 | 7.2305 | -0.030 | 1 |
| 5.0 | 0.3 | -0.7 | 0.1105 | 12.2756 | 6.2106 | -0.200 | 1 |
| 5.0 | 0.3 | -0.9 | 0.1449 | 12.3260 | 5.4877 | -0.370 | 1 |
| 5.0 | 0.3 | -0.99 | 0.1595 | 12.3461 | 5.2183 | -0.426 | 1 |
| 5.0 | 1.0 | -0.01 | 0.0006 | 8.8573 | 8.5583 | 0.758 | 1 |
| 5.0 | 1.0 | -0.2 | 0.0778 | 9.0801 | 5.4148 | 0.570 | 1 |
| 5.0 | 1.0 | -0.49 | 0.1954 | 9.2530 | 3.5805 | 0.240 | 1 |
| 5.0 | 1.0 | -0.7 | 0.2617 | 9.3338 | 2.9345 | -0.101 | 1 |
| 5.0 | 1.0 | -0.9 | 0.3142 | 9.3933 | 2.5401 | -0.609 | 1 |
| 5.0 | 1.0 | -0.99 | 0.3352 | 9.4163 | 2.4054 | -0.676 | 1 |
| 5.0 | 2.2 | -0.01 | 0.0015 | 5.1975 | 4.9856 | 0.831 | 1 |
| 5.0 | 2.2 | -0.2 | 0.1078 | 5.4526 | 3.0483 | 0.668 | 1 |
| 5.0 | 2.2 | -0.49 | 0.2373 | 5.6299 | 2.0664 | 0.382 | 1 |
| 5.0 | 2.2 | -0.7 | 0.3059 | 5.7107 | 1.7525 | 0.074 | 1 |
| 5.0 | 2.2 | -0.9 | 0.3592 | 5.7698 | 1.5718 | -0.650 | 1 |
| 5.0 | 2.2 | -0.99 | 0.3803 | 5.7926 | 1.5126 | -0.730 | 1 |
| 5.0 | 3.0 | -0.01 | 0.0018 | 3.9792 | 3.8174 | 0.836 | 1 |
| 5.0 | 3.0 | -0.2 | 0.1122 | 4.2348 | 2.3989 | 0.680 | 1 |
| 5.0 | 3.0 | -0.49 | 0.2399 | 4.4094 | 1.6875 | 0.411 | 1 |
| 5.0 | 3.0 | -0.7 | 0.3071 | 4.4890 | 1.4652 | 0.129 | 1 |
| 5.0 | 3.0 | -0.9 | 0.3593 | 4.5473 | 1.3402 | -0.556 | 1 |
| 5.0 | 3.0 | -0.99 | 0.3800 | 4.5698 | 1.3000 | -0.727 | 1 |
| 5.0 | 6.0 | -0.01 | 0.0024 | 2.0822 | 2.0059 | 0.823 | 1 |
| 5.0 | 6.0 | -0.2 | 0.1116 | 2.3255 | 1.4242 | 0.680 | 1 |
| 5.0 | 6.0 | -0.49 | 0.2296 | 2.4896 | 1.1235 | 0.442 | 1 |
| 5.0 | 6.0 | -0.7 | 0.2917 | 2.5652 | 1.0357 | 0.208 | 1 |
| 5.0 | 6.0 | -0.9 | 0.3402 | 2.6211 | 0.9909 | -0.324 | 1 |
| 5.0 | 6.0 | -0.99 | 0.3595 | 2.6427 | 0.9778 | -0.693 | 1 |

Stress grid (u in (5.0, 10.0, 20.0, 50.0), k in (1.0, 3.0, 6.0, 10.0), rho in (-1e-06, -0.01, -0.2, -0.49, -0.9, -0.99)): max R', smallest s^2 V'', largest first/(1 - rho^2) and smallest cov on [l1, l2'].

| u | k | rho | max R' | min s^2 V'' | max first/(1-rho^2) | min cov |
|---|---|---|---|---|---|---|
| 5.0 | 1.0 | -1e-06 | 0.7682 | 1.00 | 0.7626 | 0.883 |
| 5.0 | 1.0 | -0.01 | 0.7583 | 1.01 | 0.7572 | 0.906 |
| 5.0 | 1.0 | -0.2 | 0.5699 | 0.89 | 0.6894 | 1.24 |
| 5.0 | 1.0 | -0.49 | 0.2401 | 0.25 | 0.6821 | 1.31 |
| 5.0 | 1.0 | -0.9 | -0.6093 | -4.78 | 0.7272 | 0.259 |
| 5.0 | 1.0 | -0.99 | -0.6760 | -0.33 | 0.5882 | 0.0258 |
| 5.0 | 3.0 | -1e-06 | 0.8437 | 1.00 | 0.8402 | 0.777 |
| 5.0 | 3.0 | -0.01 | 0.8355 | 1.03 | 0.8360 | 0.803 |
| 5.0 | 3.0 | -0.2 | 0.6800 | 0.92 | 0.7977 | 1.19 |
| 5.0 | 3.0 | -0.49 | 0.4113 | 0.06 | 0.8812 | 1.5 |
| 5.0 | 3.0 | -0.9 | -0.5563 | -13.10 | 2.4146 | 0.266 |
| 5.0 | 3.0 | -0.99 | -0.7273 | -50.53 | 9.5406 | 0.0256 |
| 5.0 | 6.0 | -1e-06 | 0.8310 | 1.00 | 0.8280 | 0.82 |
| 5.0 | 6.0 | -0.01 | 0.8235 | 1.05 | 0.8245 | 0.844 |
| 5.0 | 6.0 | -0.2 | 0.6804 | 0.93 | 0.7996 | 1.2 |
| 5.0 | 6.0 | -0.49 | 0.4422 | -0.32 | 0.9352 | 1.45 |
| 5.0 | 6.0 | -0.9 | -0.3235 | -17.09 | 3.4926 | 0.238 |
| 5.0 | 6.0 | -0.99 | -0.6934 | -65.85 | 33.5623 | 0.0226 |
| 5.0 | 10.0 | -1e-06 | 0.8103 | 1.00 | 0.8075 | 0.876 |
| 5.0 | 10.0 | -0.01 | 0.8030 | 1.06 | 0.8046 | 0.898 |
| 5.0 | 10.0 | -0.2 | 0.6669 | 0.94 | 0.7883 | 1.21 |
| 5.0 | 10.0 | -0.49 | 0.4449 | -1.21 | 1.0138 | 1.29 |
| 5.0 | 10.0 | -0.9 | -0.2035 | -22.72 | 4.9202 | 0.211 |
| 5.0 | 10.0 | -0.99 | -0.6539 | -75.35 | 57.1002 | 0.0199 |
| 10.0 | 1.0 | -1e-06 | 0.9263 | 1.01 | 0.9240 | 0.543 |
| 10.0 | 1.0 | -0.01 | 0.9190 | 1.01 | 0.9195 | 0.577 |
| 10.0 | 1.0 | -0.2 | 0.7642 | 0.98 | 0.8713 | 1.09 |
| 10.0 | 1.0 | -0.49 | 0.4469 | 0.20 | 0.9209 | 1.52 |
| 10.0 | 1.0 | -0.9 | -0.6724 | -5.40 | 1.4113 | 0.341 |
| 10.0 | 1.0 | -0.99 | -0.8745 | -21.03 | 1.1736 | 0.0341 |
| 10.0 | 3.0 | -1e-06 | 0.9506 | 1.00 | 0.9491 | 0.454 |
| 10.0 | 3.0 | -0.01 | 0.9447 | 1.05 | 0.9455 | 0.489 |
| 10.0 | 3.0 | -0.2 | 0.8190 | 0.90 | 0.9202 | 1.01 |
| 10.0 | 3.0 | -0.49 | 0.5694 | -0.41 | 1.0438 | 1.49 |
| 10.0 | 3.0 | -0.9 | -0.3739 | -12.72 | 3.0164 | 0.354 |
| 10.0 | 3.0 | -0.99 | -0.9214 | -56.95 | 14.4702 | 0.0342 |
| 10.0 | 6.0 | -1e-06 | 0.9425 | 1.00 | 0.9409 | 0.492 |
| 10.0 | 6.0 | -0.01 | 0.9366 | 1.08 | 0.9375 | 0.525 |
| 10.0 | 6.0 | -0.2 | 0.8147 | 1.04 | 0.9170 | 1.02 |
| 10.0 | 6.0 | -0.49 | 0.5884 | -0.30 | 1.1128 | 1.45 |
| 10.0 | 6.0 | -0.9 | -0.1467 | -15.29 | 4.2002 | 0.332 |
| 10.0 | 6.0 | -0.99 | -0.8892 | -68.29 | 38.2000 | 0.0314 |
| 10.0 | 10.0 | -1e-06 | 0.9304 | 1.00 | 0.9287 | 0.544 |
| 10.0 | 10.0 | -0.01 | 0.9243 | 1.10 | 0.9253 | 0.575 |
| 10.0 | 10.0 | -0.2 | 0.8032 | 0.91 | 0.9076 | 1.04 |
| 10.0 | 10.0 | -0.49 | 0.5883 | -0.15 | 1.2088 | 1.42 |
| 10.0 | 10.0 | -0.9 | -0.0401 | -23.32 | 5.8579 | 0.306 |
| 10.0 | 10.0 | -0.99 | -0.8507 | -80.23 | 65.6816 | 0.0287 |
| 20.0 | 1.0 | -1e-06 | 0.9801 | 1.02 | 0.9795 | 0.289 |
| 20.0 | 1.0 | -0.01 | 0.9760 | 1.03 | 0.9768 | 0.327 |
| 20.0 | 1.0 | -0.2 | 0.8702 | 1.20 | 0.9617 | 0.913 |
| 20.0 | 1.0 | -0.49 | 0.6043 | 0.26 | 1.0765 | 1.51 |
| 20.0 | 1.0 | -0.9 | -0.6040 | -6.07 | 2.1610 | 0.413 |
| 20.0 | 1.0 | -0.99 | -1.0510 | -29.10 | 2.0676 | 0.0415 |
| 20.0 | 3.0 | -1e-06 | 0.9867 | 1.00 | 0.9862 | 0.238 |
| 20.0 | 3.0 | -0.01 | 0.9832 | 1.09 | 0.9841 | 0.277 |
| 20.0 | 3.0 | -0.2 | 0.8951 | 1.16 | 0.9814 | 0.854 |
| 20.0 | 3.0 | -0.49 | 0.6818 | -1.09 | 1.1673 | 1.45 |
| 20.0 | 3.0 | -0.9 | -0.2201 | -12.88 | 3.5076 | 0.43 |
| 20.0 | 3.0 | -0.99 | -1.0911 | -64.39 | 18.1194 | 0.0418 |
| 20.0 | 6.0 | -1e-06 | 0.9839 | 1.00 | 0.9834 | 0.262 |
| 20.0 | 6.0 | -0.01 | 0.9802 | 1.13 | 0.9811 | 0.3 |
| 20.0 | 6.0 | -0.2 | 0.8912 | 1.00 | 0.9785 | 0.862 |
| 20.0 | 6.0 | -0.49 | 0.6928 | -0.26 | 1.2413 | 1.41 |
| 20.0 | 6.0 | -0.9 | -0.0195 | -18.64 | 4.6814 | 0.416 |
| 20.0 | 6.0 | -0.99 | -1.0640 | -77.95 | 39.6330 | 0.0395 |
| 20.0 | 10.0 | -1e-06 | 0.9796 | 1.00 | 0.9790 | 0.295 |
| 20.0 | 10.0 | -0.01 | 0.9756 | 1.16 | 0.9766 | 0.333 |
| 20.0 | 10.0 | -0.2 | 0.8839 | 0.87 | 0.9728 | 0.878 |
| 20.0 | 10.0 | -0.49 | 0.6921 | -1.29 | 1.3395 | 1.39 |
| 20.0 | 10.0 | -0.9 | 0.0765 | -22.39 | 6.3540 | 0.397 |
| 20.0 | 10.0 | -0.99 | -1.0306 | -88.50 | 67.1195 | 0.0372 |
| 50.0 | 1.0 | -1e-06 | 0.9967 | 1.06 | 0.9966 | 0.118 |
| 50.0 | 1.0 | -0.01 | 0.9950 | 1.06 | 0.9956 | 0.157 |
| 50.0 | 1.0 | -0.2 | 0.9337 | 1.20 | 1.0100 | 0.745 |
| 50.0 | 1.0 | -0.49 | 0.7375 | 0.09 | 1.2035 | 1.43 |
| 50.0 | 1.0 | -0.9 | -0.4259 | -7.58 | 2.9896 | 0.49 |
| 50.0 | 1.0 | -0.99 | -1.2382 | -36.98 | 3.7381 | 0.0494 |
| 50.0 | 3.0 | -1e-06 | 0.9978 | 1.00 | 0.9977 | 0.0967 |
| 50.0 | 3.0 | -0.01 | 0.9963 | 1.15 | 0.9969 | 0.136 |
| 50.0 | 3.0 | -0.2 | 0.9431 | 1.50 | 1.0214 | 0.712 |
| 50.0 | 3.0 | -0.49 | 0.7780 | -0.50 | 1.2906 | 1.38 |
| 50.0 | 3.0 | -0.9 | -0.0668 | -14.76 | 4.0295 | 0.508 |
| 50.0 | 3.0 | -0.99 | -1.2695 | -80.56 | 21.7884 | 0.0498 |
| 50.0 | 6.0 | -1e-06 | 0.9973 | 1.00 | 0.9972 | 0.107 |
| 50.0 | 6.0 | -0.01 | 0.9957 | 1.22 | 0.9963 | 0.146 |
| 50.0 | 6.0 | -0.2 | 0.9408 | 1.72 | 1.0236 | 0.718 |
| 50.0 | 6.0 | -0.49 | 0.7827 | -0.77 | 1.3623 | 1.35 |
| 50.0 | 6.0 | -0.9 | 0.0971 | -20.63 | 5.0990 | 0.502 |
| 50.0 | 6.0 | -0.99 | -1.2497 | -92.10 | 38.7100 | 0.0482 |
| 50.0 | 10.0 | -1e-06 | 0.9966 | 1.00 | 0.9964 | 0.121 |
| 50.0 | 10.0 | -0.01 | 0.9947 | 1.26 | 0.9954 | 0.16 |
| 50.0 | 10.0 | -0.2 | 0.9371 | 1.49 | 1.0268 | 0.73 |
| 50.0 | 10.0 | -0.49 | 0.7816 | -1.61 | 1.4460 | 1.34 |
| 50.0 | 10.0 | -0.9 | 0.1805 | -22.83 | 6.5584 | 0.492 |
| 50.0 | 10.0 | -0.99 | -1.2248 | -101.84 | 62.1433 | 0.0465 |
