# Calendar no-arbitrage for ρ < 0: the maturity derivative (route 1)

Attempt (2026-10-23) to prove conjecture `conj:noarb-neg` by signing ∂C/∂τ directly.
**Outcome: no proof.** The derivative splits exactly into a non-negative volatility term and
a correlation covariance; the covariance is negative over much of the strike line, and in
the far right tail it cancels the volatility term to leading order. The identity is checked
in `test_calendar_derivative.py`; the scan below is the evidence for the tail statement.

## Identity

F = D = 1, t = τQ(τ) (non-decreasing in τ when g ≥ 0, so it suffices to sign ∂/∂u with
u = √t), s² = ln(1+k²), ζ = exp(sZ − s²/2), U = ρZ + √(1−ρ²)N:

    S = exp(u√ζ U − u²ζ/2) / m(u),        A(Z) = E[S | Z] = exp(ρu√ζ Z − ρ²u²ζ/2) / m(u).

1. Share measure dP*/dP = S. Since E[S] = 1 for every u, E*[∂_u ln S] = 0, so with
   D = √ζ U − uζ (= ∂_u ln S + m'/m), for every convex f

       d/du E[f(S)] = E*[f'(S) (D − E*[D])] = Cov*(f'(S), D),

   and for a call (f' = 1{S > K}) ∂C/∂u = Cov*(1{S > K}, D).
2. Under P*, Z has density φ(z)A(z), and given Z the tilt e^{u√ζ U} shifts
   U | Z ~ N(ρZ, 1−ρ²) to mean ρZ + (1−ρ²)u√ζ. Hence

       h(Z) := E*[D | Z] = ρ√ζ Z − ρ²uζ,    D − h = √((1−ρ²)ζ) N',
       ln S = ln A + w/2 + √w N',   w = (1−ρ²)u²ζ,   P*(S > K | Z) = N(d1),

   with N' standard normal under P* and d1 = (ln(A/K) + w/2)/√w.
3. Law of total covariance, with Cov(1{N' > −d1}, σN') = σφ(d1):

       ∂C/∂u = T1 + T2,
       T1 = E*[√((1−ρ²)ζ) φ(d1)] = E_Z[A √((1−ρ²)ζ) φ(d1)] ≥ 0,
       T2 = Cov*(N(d1(Z)), h(Z)).

At ρ = 0, h ≡ 0 and T2 = 0, which re-proves Theorem `thm:noarb`(i). The proof of the
conjecture reduces to T2 ≥ −T1 for ρ < 0.

## Why it does not close

- T2 is not signed: in a 6,090-point scan (ρ ∈ [−0.95, −0.05], k ∈ [0.1, 3],
  u ∈ [0.02, 1.5], ln(K/F) ∈ [−3, 4]) T2 < 0 at 2,681 points, so T1 must carry the proof.
  h is hump-shaped in Z (zero at −∞, maximum near Z = −2/s, then → −∞) and N(d1(Z)) is
  not monotone (A pulls d1 down as Z grows, the variance w pushes it up for OTM calls), so
  neither a Chebyshev nor an FKG argument applies: the (Z, D) density under P* is not MTP2
  (its mixed log-derivative is [ρ + (s/2)(2U − ρZ − 2u√ζ)] / ((1−ρ²)√ζ), unsigned).
- Left wing: (T1+T2)/T1 exceeds 1 at every scanned point with ln K/F ≤ −3 — the
  correlation helps there. Near the money its sign varies (e.g. 0.997 at ln K/F = −1,
  ρ = −0.05, k = 2.2, u = 2).
- Right wing: T2 < 0 and the ratio stays positive but decays towards zero (table below;
  e.g. ρ = −0.49, k = 2.2, u = 0.3: 0.73 at ln K/F = 0.5, 0.33 at 10, 0.22 at 40, 0.12
  at 200; "underflow" marks points where T1 itself underflows double precision). T2 cancels T1 at leading
  order, so a proof must control a second-order Laplace term in the far right tail,
  uniformly in u and k. That is the hard core of the conjecture.
- For ρ > 0 a first scan (Z grid [−14, 14]) gave negative ratios at moderate strikes
  (e.g. ln K/F = 3, u = 0.4, k = 3, ρ = 0.95), consistent with Theorem `thm:noarb`(ii);
  a fixed grid is unreliable for ρ > 0, so these are leads, not certified witnesses.

## Partial results within reach

- The identity itself, as a remark (`rem:calendar-derivative` in `paper/noarbitrage.tex`):
  it isolates the far right wing as the open case.
- A right-tail asymptotic (Laplace) showing (T1+T2)/T1 → 0⁺ rather than crossing zero —
  long, and still not a proof for finite strikes.

<!-- scan -->
# CRL calendar derivative: tail scan of (T1 + T2) / T1

Output of `uv run python derivations/calendar_derivative.py`; derivation in the
section below. Positive ratio = no calendar arbitrage at that point; ratio < 1 means
the correlation term T2 is negative. Columns: log-strike x = ln(K/F).

| rho | k | u | x=-10 | x=-3 | x=-1 | x=0 | x=0.5 | x=1 | x=2 | x=4 | x=10 | x=40 | x=200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -0.05 | 0.1 | 0.3 | 1.0433 | 1.0228 | 1.0102 | 1.0021 | 0.9980 | 0.9939 | 0.9867 | 0.9760 | 0.9598 | underflow | underflow |
| -0.05 | 0.1 | 2.0 | 1.0116 | 1.0037 | 1.0013 | 1.0000 | 0.9994 | 0.9988 | 0.9975 | 0.9950 | 0.9879 | 0.9657 | 0.9330 |
| -0.05 | 2.2 | 0.3 | 1.0473 | 1.0523 | 1.0465 | 0.9974 | 0.9449 | 0.9284 | 0.9100 | 0.8874 | 0.8465 | 0.7503 | 0.5857 |
| -0.05 | 2.2 | 2.0 | 1.0186 | 1.0112 | 0.9972 | 0.9732 | 0.9544 | 0.9393 | 0.9185 | 0.8920 | 0.8466 | 0.7479 | 0.5843 |
| -0.05 | 3.0 | 0.3 | 1.0446 | 1.0516 | 1.0483 | 0.9966 | 0.9355 | 0.9180 | 0.8978 | 0.8725 | 0.8267 | 0.7222 | 0.5515 |
| -0.05 | 3.0 | 2.0 | 1.0184 | 1.0124 | 0.9989 | 0.9701 | 0.9444 | 0.9264 | 0.9029 | 0.8737 | 0.8242 | 0.7185 | 0.5497 |
| -0.49 | 0.1 | 0.3 | 1.8159 | 1.5539 | 1.4099 | 1.3127 | 1.2573 | 1.1991 | 1.0855 | 0.9234 | 0.7386 | underflow | underflow |
| -0.49 | 0.1 | 2.0 | 1.4194 | 1.3316 | 1.3025 | 1.2873 | 1.2795 | 1.2717 | 1.2556 | 1.2225 | 1.1195 | 0.8127 | underflow |
| -0.49 | 2.2 | 0.3 | 1.9312 | 1.9772 | 1.8921 | 1.4649 | 0.7327 | 0.6013 | 0.5042 | 0.4244 | 0.3327 | 0.2166 | 0.1192 |
| -0.49 | 2.2 | 2.0 | 1.5191 | 1.4501 | 1.3513 | 1.2139 | 1.0667 | 0.8939 | 0.7037 | 0.5528 | 0.4033 | 0.2436 | 0.1268 |
| -0.49 | 3.0 | 0.3 | 1.9014 | 1.9798 | 1.9286 | 1.5135 | 0.6836 | 0.5620 | 0.4711 | 0.3951 | 0.3074 | 0.1974 | 0.1070 |
| -0.49 | 3.0 | 2.0 | 1.5220 | 1.4692 | 1.3776 | 1.2299 | 1.0363 | 0.8318 | 0.6462 | 0.5064 | 0.3680 | 0.2200 | 0.1132 |
| -0.9 | 0.1 | 0.3 | 8.8044 | 6.8994 | 5.9261 | 5.2486 | 4.8274 | 4.3266 | 3.0232 | 1.3156 | underflow | underflow | underflow |
| -0.9 | 0.1 | 2.0 | 5.9525 | 5.3756 | 5.1770 | 5.0704 | 5.0151 | 4.9584 | 4.8405 | 4.5844 | 3.6069 | underflow | underflow |
| -0.9 | 2.2 | 0.3 | 10.0163 | 10.2796 | 9.6214 | 7.1011 | 0.8332 | 0.5011 | 0.3546 | 0.2682 | 0.1925 | 0.1174 | underflow |
| -0.9 | 2.2 | 2.0 | 6.6843 | 6.2130 | 5.6433 | 4.9384 | 4.1896 | 2.3909 | 0.8778 | 0.4854 | 0.2817 | 0.1449 | underflow |
| -0.9 | 3.0 | 0.3 | 9.8331 | 10.3809 | 9.9904 | 7.6037 | 0.7205 | 0.4488 | 0.3224 | 0.2452 | 0.1760 | 0.1066 | underflow |
| -0.9 | 3.0 | 2.0 | 6.7305 | 6.3636 | 5.8299 | 5.0811 | 4.1446 | 1.7972 | 0.7197 | 0.4189 | 0.2490 | 0.1291 | underflow |
| -0.99 | 0.1 | 0.3 | 87.1748 | 67.2213 | 57.1586 | 50.1189 | 45.6592 | 40.1458 | 21.3821 | underflow | underflow | underflow | underflow |
| -0.99 | 0.1 | 2.0 | 57.3495 | 51.4256 | 49.3720 | 48.2642 | 47.6874 | 47.0941 | 45.8541 | 43.1254 | 31.6234 | underflow | underflow |
| -0.99 | 2.2 | 0.3 | 100.6845 | 103.3272 | 96.3921 | 70.9659 | 0.9259 | 0.4927 | 0.3329 | underflow | underflow | underflow | underflow |
| -0.99 | 2.2 | 2.0 | 65.0228 | 60.1284 | 54.3806 | 47.3984 | 40.0614 | 14.1056 | 1.0319 | 0.4864 | underflow | underflow | underflow |
| -0.99 | 3.0 | 0.3 | 98.8466 | 104.5876 | 100.5224 | 76.6068 | 0.7747 | 0.4361 | 0.3009 | underflow | underflow | underflow | underflow |
| -0.99 | 3.0 | 2.0 | 65.5586 | 61.7344 | 56.3371 | 48.9212 | 39.8405 | 4.4316 | 0.7958 | 0.4123 | underflow | underflow | underflow |
