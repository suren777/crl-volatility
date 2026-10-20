# Calendar no-arbitrage in the CRL model — proof attempt

Attempt 2026-10-19 (requested). Status: **ρ = 0 proven** (Theorem 1); **ρ < 0 open**
(numerically clean); **ρ > 0 open** (numerics unreliable, §4). Numeric witness for
everything below: `search_calendar.py` → `search_calendar.md` (1.12 M checks, seed
20261019). This note contains no new numbers.

## 1. Setup — the exact terminal law

From `msv.crl_call` (forward form, D = 1): conditional on the level driver Z ~ N(0,1),
with ζ = exp(sZ − s²/2), s² = ln(1+k²), the terminal law is lognormal with forward
F·M(Z)/m(τ) and variance rate (1−ρ²)·ζ(Z)·Q(τ), where

    M(Z,τ) = exp(ρ·√(ζ(Z)Q(τ)τ)·Z − (ρ²/2)·ζ(Z)Q(τ)τ),   m(τ) = E_quad[M(·,τ)].

Writing H(τ) = τ·Q(τ) = ∫₀^τ g(t) dt (total variance per unit ζ; `msv.g`), the
continuum terminal law is

    ln(S_τ/F) = −ln m(τ) − ζ(Z)H(τ)/2 + ρ·√(ζ(Z)H(τ))·Z + √((1−ρ²)ζ(Z)H(τ))·W,

with W ~ N(0,1) independent of Z. Forward exactness: E[S_τ/F] = E[M(Z)]/m(τ) = 1
(the W-moment function exp((1−ρ²)ζH/2) cancels exp(−(1−ρ²)ζH/2) inside M).

## 2. Theorem 1 (ρ = 0, i.e. the MSV mixture): no calendar arbitrage

**Statement.** Let g(t) ≥ 0 on [τ1, τ2] (so H(τ1) ≤ H(τ2)). Then for every K,
C(τ1, K) ≤ C(τ2, K) — the per-maturity marginals are increasing in convex order.

**Proof.** At ρ = 0, M ≡ 1 and m ≡ 1, so conditional on Z,

    S_τ/F = Y_w,   Y_w := exp(−w/2 + √w·W),   w = ζ(Z)·H(τ).

Fix the Z-realisation (equivalently fix ζ > 0). The family (Y_w)_{w≥0} is a geometric
Brownian in the "time" parameter w with zero drift, hence a martingale in w. Marginals
of a martingale are increasing in convex order: for w1 ≤ w2 and convex f,

    E[f(Y_{w2}) | Y_{w1}] ≥ f(E[Y_{w2} | Y_{w1}]) = f(Y_{w1})   (Jensen),

and taking expectations gives E[f(Y_{w1})] ≤ E[f(Y_{w2})]. With f(x) = (x − K)⁺ and
w_i = ζ·H(τ_i), this is C(τ1, K | Z) ≤ C(τ2, K | Z) pointwise in Z. Integrating over Z
(any common law of ζ) preserves the inequality. Note E[Y_w] = 1 for every w, so the
marginals share the forward F — the equal-mean requirement of convex order holds
pointwise, without using E[ζ] = 1. ∎

**Corollary (the search domain is covered).** The search's parameter box satisfies
g(t) = a2 + (a0 + a1λt)e^{−λt} > 0 everywhere (a0, a1, a2 ≥ 0, a2 > 0 in the box and in
all three named curves — including the "decreasing short-end" curve, whose *average*
Q(τ) decreases but whose total variance H(τ) = τQ(τ) increases). Theorem 1 therefore
applies to every τ-pair and every draw searched at ρ = 0, and the 0/1.12 M violations
are what the theorem predicts. The right monotonicity object is H = τQ, not Q — Q
itself may decrease (NS humps) without harming calendar consistency.

## 3. Why the proof breaks for ρ ≠ 0

For ρ ≠ 0 the Z-conditional forward F·M(Z,τ)/m(τ) depends on τ. Two lognormals are
convex-ordered **iff** their means are equal and variances ordered, and
M(Z,τ1) ≠ M(Z,τ2) whenever H(τ1) ≠ H(τ2), so the pointwise-in-Z route fails for every
ρ ≠ 0 — including ρ < 0, where no violation was found in 1.12 M converged checks.

The natural coupling (Z revealed at t = 0, W unfolding in H-time) gives

    E[S_τ2 | Z, F_{H1}] / S_τ1 = (m(τ1)/m(τ2)) · exp(ρ√ζ·(√H2−√H1)·Z − (ρ²/2)ζ(H2−H1)),

which is ≠ 1 in general (the Z-dependent factor is exactly the M-ratio between the two
maturities), so the interpolation is not a martingale, super- or sub-martingale — the
mixture must compensate across Z. Convex order for ρ < 0 therefore needs an argument
that uses the joint law of (Z, ζ) and not the coupling above. **Open.**

## 4. ρ > 0 — status unknown, and how to decide it

The 1,234/2,000 "violations" in `search_calendar.md` are fixed-node Gauss–Hermite
artifacts (planner review there: prices at the worst case do not converge from 32 to
256 nodes, while ρ = 0 and ρ = −0.0068 agree to every digit). Two independent ways to
settle ρ > 0:

1. **Perturbative certificate.** The calendar spread S(K) = C(τ2,K) − C(τ1,K) satisfies
   S(K) ≥ 0 at ρ = 0 (Theorem 1). If ∂S/∂ρ|_{ρ=0} < 0 at some K, convex order fails for
   small ρ > 0. First order: ∂ln S_τ/∂ρ|₀ = √(ζH(τ))·Z, with an m(τ,ρ) correction —
   note ∂m/∂ρ|₀ = √H·E[√ζ·Z] = √H·(s/2)e^{s²/8} > 0, so the normaliser also moves.
   What must be signed: E[1(S⁰_K)·S⁰·√ζ·Z] for the OTM region (S⁰ = the ρ = 0
   terminal; OTM outcomes favour *small* ζ, i.e. small Z, suggesting a negative sign —
   this is the sign to make rigorous, handling the m-correction exactly).
2. **Paired Monte-Carlo certificate.** Sample the §1 continuum law directly (draw Z, W;
   no quadrature, converges for all ρ) with common random numbers across τ, and test
   ε = C(τ2,K) − C(τ1,K) with a confidence interval at specific parameter points. This
   is quadrature-free and either certifies a real violation or certifies consistency at
   the tested points — the independent second method `derivations/README.md` asks for.

## 5. What the paper can claim

- ρ = 0 (MSV): calendar no-arbitrage **proven** whenever g ≥ 0 on the maturity interval
  (state as a proposition with the §2 proof; H = τQ is the object that must increase).
- ρ < 0: no arbitrage found in 1.12 M converged, MC-checked checks over a box where
  g > 0 everywhere; proof open.
- ρ > 0: **unknown** — do not claim either way; the fixed-node quadrature cannot decide
  it (see `search_calendar.md`, planner review). If the paper stays in the equity region
  ρ ≤ 0 (its empirical fit), state ρ > 0 as a limitation.
