# eq_skew: short-maturity ATM skew and level of the CRL model

Written 2026-10-19 from the planner's sketch and numerics; verified by
`test_skew_asymptotics.py` (label `eq_skew`: symbolic checks plus numeric cross-checks
against `msv.crl_call` through the independent `checks.black_fwd`). The numerics are
reliable for ρ ≤ 0 only (`msv.crl_call` docstring); the formulas themselves hold for any
|ρ| < 1 and k > 0.

## 1. Result

As τ → 0, forward form F = D = 1, x = ln K, ε = √τ:

    S(τ) = dI/dx(0) = C(k,ρ)/√τ + O(1),
    C(k,ρ) = √(2π)·[ 1/2 − E_Z N(d_Z) ],
    d_Z = ρ_t·( Z − (s/2)·exp(s²/8 − sZ/2) ),    s² = ln(1+k²),   ρ_t = ρ/√(1−ρ²),

with the ATM level tending to the Bachelier mixture

    I(0) → √(2π)·E_Z[ u_Z·N(u_Z/σ_Z) + σ_Z·φ(u_Z/σ_Z) ],
    u_Z = ρ·(a(Z)·Z − E[aZ]),   σ_Z = √(1−ρ²)·a(Z),   a(Z) = √(ζ(Z)·Q).

Corollaries (§5, §6):

- **C does not depend on Q** — the skew blow-up coefficient is level-independent. In the
  rough-volatility heuristic (ATM skew ~ τ^(H−1/2)) this is an **H = 0 law**.
- C = 0 iff ρ = 0 or k = 0; for ρ < 0, C < 0.
- Small s: C = ρs/2 + c3(ρ)s³ + O(s⁴) with c3(ρ) = ρ(3−5ρ²)/24 (the s² term vanishes
  identically, §5).

## 2. Setup and assumptions

From `msv.crl_call` (forward form, D = 1): conditional on the level driver Z ~ N(0,1),
with ζ(Z) = exp(sZ − s²/2), s² = ln(1+k²), V = ζ(Z)·Q(τ), the terminal law is lognormal,
ln S_T | Z ~ N(ln F_Z, (1−ρ²)·ζ(Z)·Q(τ)·τ), with forward F_Z = M(Z)/m and

    M(Z) = exp(ρ·√(Vτ)·Z − ρ²·Vτ/2),    m = E[M(Z)].

Write a(Z) = √(ζ(Z)·Q), so √(Vτ) = a(Z)·ε. Conditioned on Z the price is a Black price
with

    ln F_Z = ρ·a·ε·Z − ρ²·a²·ε²/2 − ln m,    total sd = √(1−ρ²)·a·ε.        (2.1)

Assumptions: Q(τ) = Q(0+) + O(τ) with Q(0+) = a0 + a2 (Nelson–Siegel, smooth at 0);
0 < k; |ρ| < 1. The only place ρ ≤ 0 enters is the reliability of the numeric pricer.

Strike delta of the undiscounted forward Black: ∂C/∂K = −N(d2), so at K = e^x with F = 1,

    dC/dx(0) = −N(d2_Z),    d2_Z = (ln F_Z − (1−ρ²)a²ε²/2) / (√(1−ρ²)·a·ε).     (2.2)

## 3. E[aZ] and ln m to first order

Using E[Z·e^{cZ}] = c·e^{c²/2} with c = s/2:

    E[aZ] = √Q·E[Z·e^{sZ/2 − s²/4}] = √Q·(s/2)·e^{−s²/8}.                       (3.1)

Expanding the normaliser m(ε) = E[exp(ρε·a(Z)·Z − ρ²ε²·a(Z)²/2)] to second order — the ε²
coefficient is (E[a²Z²] − E[a²])/2 = Q·(E[Z²ζ] − E[ζ])/2 = Q·s²/2, since E[ζ] = 1 and
E[Z²·e^{sZ}] = (1+s²)·e^{s²/2} —

    m = 1 + ρε·E[aZ] + ρ²ε²·Q·s²/2 + O(ε³)    ⟹    ln m = ρε·E[aZ] + O(ε²).     (3.2)

## 4. The d2_Z limit: Q cancels

The numerator of (2.2) is ρaεZ − a²ε²/2 − ln m = ρε·(aZ − E[aZ]) + O(ε²) = ε·u_Z + O(ε²)
by (3.2); the denominator is ε·σ_Z. Hence, pointwise in Z,

    d2_Z → u_Z/σ_Z ≡ d_Z,    E[aZ]/a(Z) = √Q(s/2)e^{−s²/8} / (√Q·e^{sZ/2−s²/4})
                                 = (s/2)·e^{s²/8 − sZ/2},                       (4.1)

which is (1): d_Z = ρ_t·(Z − (s/2)·e^{s²/8 − sZ/2}). Q enters (2.1) only through √Q
factors that cancel in every ratio — the limiting d2 contains no Q.

## 5. The skew: strike delta + Black identity

By (2.2) and bounded convergence (N(d2_Z) → N(d_Z) pointwise, |N| ≤ 1):

    dC/dx(0) = −E_Z[N(d2_Z)] → −E_Z[N(d_Z)].                                    (5.1)

Writing the model price as Black_forward(x, I(x)) and differentiating at x = 0 — there
d1 = Iε/2, d2 = −Iε/2, vega = φ(d1)·ε —

    dC/dx(0) = −N(−I(0)ε/2) + φ(I(0)ε/2)·ε·I'(0),
    I'(0) = ( dC/dx(0) + N(−I(0)ε/2) ) / ( ε·φ(I(0)ε/2) ).                      (5.2)

The numerator of (5.2) tends to 1/2 − E_Z N(d_Z) + o(1), the denominator is εφ(0)(1 + O(ε²));
dividing gives the result

    S(τ) = C(k,ρ)/√τ + O(1),   C(k,ρ) = √(2π)·[ 1/2 − E_Z N(ρ_t(Z − (s/2)e^{s²/8 − sZ/2})) ].

- **Zeros.** ρ = 0: ρ_t = 0 so d_Z ≡ 0 and E N = 1/2. k = 0: s = 0 so d_Z = ρ_t·Z, whose
  law is symmetric, so E N(ρ_t Z) = 1/2. For ρ < 0 the coefficient is negative.
- **Small s.** Linearise g(Z) = (s/2)·e^{s²/8 − sZ/2} = s/2 + O(s²), so d_Z = ρ_t Z − ρ_t s/2 + O(s²).
  Use E[N(αZ+β)] = N(β/√(1+α²)) — both sides have β-derivative
  E[φ(αZ+β)] = φ(β/√(1+α²))/√(1+α²) and both → 0 as β → −∞ (checked symbolically and
  numerically in the test module) — with α = ρ_t, β = −ρ_t s/2, and √(1+ρ_t²) = 1/√(1−ρ²):

      E_Z N(d_Z) = N(−ρs/2) + O(s²)
      C = √(2π)·[1/2 − N(−ρs/2)] + O(s²) = √(2π)·φ(0)·(ρs/2) + O(s²) = ρs/2 + O(s²).

  One can do better: expand d_Z in s to third order and Taylor `N` about d0 = ρ_t Z
  (derivatives of N are φ and polynomials-in-Z·φ) with Gaussian expectations term by
  term (only even moments E[z^2m·φ(αz)] survive; the s² term is proportional to
  E[z·φ(ρ_t z)] = 0). Writing α = ρ_t = ρ/√(1−ρ²), this gives

      E_Z N(d_Z) = 1/2 − (α/2)·E[φ(αz)]·s + c3(s) s³ + ...,   α = ρ/√(1−ρ²),
      C = ρs/2 + c3(ρ)·s³ + O(s⁴),                            (s² term vanishes)
      c3(ρ) = ρ(3 − 5ρ²)/24,

  since E[φ(αz)] = φ(0)/√(1+α²) and √(1+α²) = 1/√(1−ρ²), the s coefficient of E_Z N is
  −(α/2)·φ(0)/√(1+α²), so C's s coefficient is √(2π)·(α/2)·φ(0)·√(1−ρ²) = ρ/2; and
  substituting the s³ coefficient from c3(α) = α(3−2α²)/(24(1+α²)^{3/2}) at
  α = ρ_t (1+α² = 1/(1−ρ²)) yields c3(ρ) above. Checked symbolically and numerically
  (`test_eq_skew_small_s_cubic`, label `eq_skew`): (C − ρs/2)/s³ → c3(ρ) to 3e-3
  relative already at s = 0.05 for ρ ∈ {−0.3, −0.7, −0.9}. The stronger statement
  used everywhere is therefore C = ρs/2 + O(s³), with O(s³) coefficient
  c3(ρ) = ρ(3−5ρ²)/24.

## 6. The ATM level: Bachelier limit of the conditional Black prices

Conditional on Z, ln S_T | Z is normal with mean μ_Z = ln F_Z = ε·u_Z + O(ε²) and sd
s_Z = ε·σ_Z. As ε → 0 the event {e^{μ+sY} > 1} = {μ + sY > 0} has O(1) probability, on it
e^{μ+sY} − 1 = μ + sY + O(ε²), so the conditional Black price converges to the Bachelier
price of the normal(μ_Z, s_Z) variable, with moneyness ratio μ_Z/s_Z = u_Z/σ_Z = O(1):

    C_Z = μ_Z·N(u_Z/σ_Z) + s_Z·φ(u_Z/σ_Z) + O(ε²)
        = ε·[ u_Z·N(u_Z/σ_Z) + σ_Z·φ(u_Z/σ_Z) ] + O(ε²).                        (6.1)

Total price C(0) = E_Z[C_Z]; the implied form gives C(0) = 2N(Iε/2) − 1 = φ(0)·I(0)·ε + O(ε³).
Equating the O(ε) terms:

    I(0) → √(2π)·E_Z[ u_Z·N(u_Z/σ_Z) + σ_Z·φ(u_Z/σ_Z) ].                        (6.2)

Check, ρ = 0: u ≡ 0, so I(0) → E_Z[σ_Z] = √(1−ρ²)·√Q·E[√ζ] = √Q·e^{−s²/8} — the MSV
mixture short-end vol (E[√ζ] = e^{−s²/8}); the level scales like √Q, unlike the skew.

## 7. Numerics

**Strike-step lesson (planner round 1).** The central difference must use
h = 0.02·√(Qτ): a fixed h is a large fraction of the implied sd at tiny τ and biases the
difference. With h = 0.02√(Qτ), h stays ≈ 2–4% of the conditional sd for the sets below.

Test module `test_skew_asymptotics.py` (label `eq_skew`): prices from `msv.crl_call`
(n_nodes = 128, three strikes in one elementwise call), IVs inverted with the independent
`checks.black_fwd` (Brent), C(k,ρ) and the level (6.2) by Gauss–Hermite quadrature of the
closed forms; Q made exact and constant by a0 = a1 = 0, a2 = Q (`msv.Q(τ)` = a2 for every
τ, so Q(0+) = a0 + a2 = Q exactly). All numbers below are that module's own printout.

| (k, ρ, Q) | C(k,ρ) | S√τ at τ=1e-4 | |S√τ−C| slope | I(0) at τ=1e-4 | level (6.2) |
|---|---|---|---|---|---|
| (1, −0.7, 0.04) | −0.29642 | −0.29631 | 0.49 | 0.19928 | 0.19937 |
| (0.5, −0.3, 0.04) | −0.07424 | −0.07423 | 0.45 | 0.19548 | 0.19550 |
| (2, −0.9, 0.01) | −0.50164 | −0.50148 | 0.51 | 0.10781 | 0.10790 |
| (1, −0.7, 0.16) | −0.29642 | −0.29620 | 0.49 | 0.39837 | 0.39874 |

- Convergence grid τ ∈ {1e-4, 4e-4, 1.6e-3, 6.4e-3, 2.56e-2}: measured slopes 0.49, 0.45,
  0.51, 0.49 — the predicted 1/2.
- Q-independence: rows 1 and 4 share (k, ρ) with Q 0.04 vs 0.16 — same C, matched to 4e-4
  and 7e-4 relative; the level scales like √Q (0.199 vs 0.398 ≈ ×2 = √4).
- Small-s check: ρs/2 = −0.291, −0.071, −0.571 vs C = −0.296, −0.074, −0.502 on rows 1–3;
  the O(s²) correction is largest at k = 2 (s = 1.27), as expected.

## 8. What it means

The CRL ATM skew explodes like τ^(−1/2) as τ → 0 with a coefficient C(k,ρ) that does not
involve the variance-curve level Q. In the rough-volatility heuristic (ATM skew ~
τ^(H−1/2)) this is an H = 0 law: the random level is fully persistent at the short end,
and no Nelson–Siegel shape can bend the exponent — (k, ρ) set only the coefficient's size
and sign. The market, on current evidence, does not do this: Guyon–El Amrani extrapolate a
finite zero-maturity SPX ATM skew of about −1.5 in decimal units
(`docs/literature/claims-check.md`, Claim 4), and our 2026-10-16 snapshot sees the market
power law continue at exponent −0.374..−0.399 down to 7 days with no saturation visible
(`experiments/results/crl_short.md`). Said plainly: this derivation is the mechanism
behind the short-end overshoot documented there — the fitted CRL skew keeps steepening
toward its flat-Q τ^(−1/2) asymptote exactly where the market does not (CRL −242.6 vol
pts at 7 d and −388.1 at 1 d against the market's −153.7 observed and ≈ −300 power-law
implied; fitted weekly exponent −0.510). Any claim that CRL matches the SPX skew term
structure must exclude or caveat the sub-monthly range.
