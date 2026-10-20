# Calendar-arbitrage random search of `crl_call`

Run 2026-10-19, seed 20261019, `msv.crl_call` (F = D = 1, 64-node Gauss-Hermite). Grid per draw: K/F in [0.5, 2.0] (8 pts), tau in [0.01, 5.0] (8 log-spaced pts). Parameters per draw: k in [0.05, 3]; Nelson-Siegel = one of the three `experiments/crl_skew.py` curves (flat 0.04, decreasing short-end, humped) or a uniform draw in the box ((0.0, 0.03), (0.0, 0.06), (0.03, 0.04), (1.0, 2.0)) they span. A violation is a price increment across tau below -1e-10 in absolute price (prices are O(1) at most; quadrature round-off is order 1e-14). This is numerical evidence over a random box, not a proof.

## Main search: rho in [-0.99, 0] (equity region), 20,000 draws
- draws: 20000; increments checked: 1120000
- draws with a violation: 0; violating increments: 0
- worst price drop seen anywhere: >= -TOL (no violation)

## Sanity search: rho in (0, 0.99], 2,000 draws (violations expected here)
- draws: 2000; increments checked: 112000
- draws with a violation: 1234; violating increments: 10647
- worst violation: rho=+0.0068 k=2.6129 ns=(uniform box) (0.0099, 0.0067, 0.0356, 1.1559) K/F=2.000 tau 0.1435->0.3486: drop -2.298e-01 (C = 0.250028)

First violations (parameters):
```
rho=+0.3785 k=1.2604 ns=(uniform box) (0.0109, 0.044, 0.0397, 1.9226) K/F=2.000 tau 0.0243->0.0590: drop -1.569e-01 (C = 0.289561)
rho=+0.8661 k=1.3107 ns=(uniform box) (0.0063, 0.0184, 0.0314, 1.9182) K/F=2.000 tau 0.0100->0.0243: drop -1.321e-01 (C = 0.229887)
rho=+0.2612 k=1.3060 ns=(flat 0.04) (0.0, 0.0, 0.04, 1.0) K/F=2.000 tau 0.0590->0.1435: drop -2.212e-02 (C = 0.085286)
rho=+0.9521 k=2.9978 ns=(humped) (0.0, 0.06, 0.03, 1.5) K/F=2.000 tau 0.0243->0.0590: drop -2.615e-03 (C = 0.173421)
rho=+0.9500 k=2.4200 ns=(decreasing short-end) (0.03, 0.0, 0.03, 2.0) K/F=2.000 tau 0.0100->0.0243: drop -1.442e-02 (C = 0.174245)
```

## Outcome
- rho <= 0: no violation found; consistent with the convex-order conjecture (epics/0010), within the box and tolerance searched.
- rho > 0: search finds the expected violations (10647 increments), so it is sensitive enough.

## Planner review (2026-10-19): ρ > 0 results are numerical artefacts
The ρ > 0 "violations" cannot be trusted: for ρ > 0 and large k, `crl_call` does not converge
in the number of Gauss–Hermite nodes. At the worst case above (ρ = +0.0068, k = 2.61,
K/F = 2, τ = 0.1435) the price is 0.00007 (32 nodes), 0.244 (64), 0.092 (128), 0.097 (256),
while ρ = 0 and ρ = −0.0068 agree to every digit from 32 to 256 nodes. Cause: with ρ > 0 the
astronomically high-level states pair with large positive returns, so the integrand has a
far-tail peak fixed-node quadrature misses. The ρ ≤ 0 result stands (converged; MC-checked
in epic 0010). Whether ρ > 0 admits calendar arbitrage is **unknown**, not shown.
Follow-up: `calendar_noarb.md` — proof attempt; Theorem 1 proves the ρ = 0 case
(g ≥ 0 on the interval covers the whole search box), ρ ≠ 0 remains open.
