# Calendar arbitrage in the CRL model: moment route

Companion to `calendar_noarb.md` (Theorem 1: no arbitrage for ρ = 0) and the two numerical
certificates `calendar_moments.py` → `calendar_moments.md` (moments, log-space trapezoid) and
`calendar_mc.py` → `calendar_mc.md` (paired MC on call/put prices). This note records the
exact identities the moment certificate rests on, the Kellerer bridge from moment decreases
to arbitrage, and why fixed-node Gauss–Hermite cannot certify anything here.

## 1. Setup

CRL terminal law at expiry τ (F = 1, D = 1): with variance time t = τ·Q(τ), s² = ln(1+k²),
ζ(Z) = exp(sZ − s²/2), and U = ρZ + √(1−ρ²)·N ~ N(0,1) correlated with Z ~ N(0,1),

    S_τ = (F/m) · exp( √(t·ζ(Z)) · U − t·ζ(Z)/2 ),
    m   = E[ exp( ρ√(t·ζ) Z − ρ² t·ζ/2 ) ].

`msv.crl_call` integrates the inner Black–Scholes function over Z by Gauss–Hermite; the
discrete forward is kept exact by construction of m.

## 2. Exact collapsed moments

Condition on Z and integrate out U | Z ~ N(ρZ, 1−ρ²) exactly: for any p,

    E[S^p] = (F/m)^p · E_Z[ exp( pρ√(t·ζ) Z + (t·ζ/2)(p²(1−ρ²) − p) ) ],

finite iff p < 1/(1−ρ²); p = 1 returns E[S] = F identically (the drift term p²(1−ρ²) − p
vanishes and m^(-1) cancels E[exp(ρ√(tζ)Z)]·... by construction of m). The identity is a
one-dimensional dense-trapezoid integral of the weight-folded exponent (log space — no
node grid, no underflow). `calendar_moments.py` validates it (2-D grid cross-check for
ρ ≤ 0, clip sensitivity, and the p = 1 identity, exact to 7e-14).

## 3. The Kellerer bridge

For unit-discount martingale terminal laws, calendar no-arbitrage across [τ₁, τ₂] holds iff
S_{τ₁} ≤cx S_{τ₂} in convex order (Kellerer 1972: convex order ⟺ martingale coupling).
Convex order with equal means implies E[f(S_{τ₁})] ≤ E[f(S_{τ₂})] for every convex f, in
particular f(x) = x^p, p ≥ 1. Hence:

**A certified strict decrease of E[S^p] in τ at any p ∈ (1, 1/(1−ρ²)) proves calendar
arbitrage exists somewhere on the strike line** — not necessarily at the moderate strikes
probed by `calendar_mc.py` (K/F ∈ {0.5, 1, 1.5, 2}), so a negative MC scan and a certified
moment decrease are logically consistent; the arbitrage would live in the far tail.

Converse caveat: moment monotonicity for all p is necessary but not sufficient for convex
order, so non-decreasing moments do not prove no-arbitrage.

## 4. Why Gauss–Hermite fails for ρ > 0 (and why the moment route still works)

Two independent structural problems, both specific to ρ > 0:

1. **Weight underflow.** `roots_hermitenorm(n)` for n ≥ 512 returns extreme-node weights
   underflowed to exact 0.0; clamping them up (e.g. to 1e-300) resurrects dead nodes as
   phantom quadrature terms — the source of the earlier garbage (ln m = 24.3 at n = 512).
   Filtering zero weights truncates the integral instead.
2. **Second Laplace bump.** The E[S^p] integrand for p > 1, ρ > 0 has its mass at a Z-peak
   where pρ√(tζ)Z overtakes the Gaussian drag — empirically Z ≈ 40–65 and, for extreme
   parameters, beyond any tested clip. The standard normal weight exp(−Z²/2) that GH
   integrates against is essentially zero there. No node count rescues this; the fix is to
   fold the weight into the integrand, which removes the Gaussian entirely. Quadrature of
   the folded exponent then needs care in the far tail: adaptive Gauss–Kronrod (quad) steps
   over the Laplace bumps — in the first scan every extreme-tail witness failed the
   clip-240 sensitivity gate, changing by up to ~75 nats between clips — while a dense
   trapezoid in log space on the folded exponent certifies cleanly (the final scheme in
   `calendar_moments.py`).

Consequence: the moment integrand is evaluated at Z values where its value is astronomically
large (E[S^p] ~ e^{±10³} is real mathematics, not overflow), so certificate claims must be
gated on (a) resolution doubling (h vs h/2) and (b) clip sensitivity. With the final
log-space trapezoid scheme all 12 witnesses (six mildest + six worst, spanning ρ ∈
{0.1, …, 0.9}, k ∈ {0.3, 0.5, 2.6}, p from just above 1 to 75% of the finite window) pass
both gates; worst pairwise gate disagreement 7.3e-12.

## 5. What can be claimed

- **ρ = 0:** calendar no-arbitrage is proven (Theorem 1 in `calendar_noarb.md`; the
  monotonicity object is H = τQ = ∫g, not Q).
- **ρ < 0:** no moment decrease found anywhere in the scan (`calendar_moments.md`), consistent
  with the ρ = 0 proof; no formal proof beyond the ρ = 0 theorem.
- **ρ > 0:** calendar arbitrage exists — certified. Twelve witnesses pass all gates
  (resolution doubling h/h·2 at clip 160 and the clip-240 re-run agree pairwise to
  < 1e-6; worst disagreement observed 7.3e-12), from mild (ρ = 0.1–0.3, k = 2.60, p just
  above 1, τ ≈ 0.01: ln-decreases ≈ 0.83–0.88) to extreme (ρ = 0.9, k = 0.3, p = 4.197:
  ln-decrease ≈ 1041) — exact values as printed in `calendar_moments.md`. Via §3 a strict
  decrease at any p ∈ (1, 1/(1−ρ²)) with equal means proves calendar arbitrage at some
  strike. (The earlier all-FAIL certification was the adaptive-quad artifact of §4, not a
  property of the witnesses; the trapezoid certificate certifies all twelve.)
- **Coherence with `calendar_mc.md`:** the MC certificate found 0 violations in 1,008 paired
  comparisons at K/F ∈ {0.5, 1, 1.5, 2}. No contradiction: convex-order failure implies a
  call-dominance violation at *some* strike, which the far-tail moments localize beyond the
  moderate-strike grid the MC scan probed.
- Paper claim: the CRL model is calendar-arbitrage-free for ρ ≤ 0 (proven at ρ = 0; no
  violation found for ρ < 0) and calendar-arbitrageable for ρ > 0 (certified witness
  decreases), with the arbitrage living in the far tail.
