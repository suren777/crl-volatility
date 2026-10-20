# Calendar-arbitrage Monte-Carlo certificate for the CRL model

Run 2026-10-19. Quadrature-free: samples the exact continuum terminal law
(calendar_noarb.md section 1) with common random numbers and antithetic pairs;
500,000 antithetic pairs per comparison. Witness = lower CI of the paired drop
C(tau1,K) - C(tau2,K) above 0 (drop > 0 means the shorter maturity is strictly
dearer: a calendar violation), estimated via put-call parity so payoffs are bounded
by K. m computed per maturity by adaptive quadrature of E[M(Z)] (deterministic).

## Validation vs `msv.crl_call` (trusted for rho <= 0)
- m_exact at rho=0: 1.000000000000 (exact 1: OK)
- worst |MC - crl_call| / tolerance: 0.37 (PASS, gate 1.0)

## Paired scan: rho > 0, three NS curves, adjacent tau pairs

| rho | k | NS | tau1 -> tau2 | K/F | drop (C1-C2) | SE | verdict |
|---|---|---|---|---|---|---|---|

 violations (lower CI > 0): **0** of 1008 comparisons

## Star witnesses from the failed-quadrature scans, re-tested here

- rho=+0.0068 k=2.6129 ns=(0.0099, 0.0067, 0.0356, 1.1559) K/F=2.00 tau 0.1435 -> 0.3486: drop -4.226e-04 (CI [-5.240e-04, -3.212e-04]) -> consistent with no arbitrage
- rho=+0.0500 k=0.5000 ns=(0.0, 0.0, 0.04, 1.0) K/F=2.00 tau 0.6300 -> 0.8162: drop -7.716e-05 (CI [-1.081e-04, -4.624e-05]) -> consistent with no arbitrage
- rho=+0.0500 k=0.5000 ns=(0.0, 0.0, 0.04, 1.0) K/F=0.50 tau 0.6300 -> 0.8162: drop -1.883e-05 (CI [-2.138e-05, -1.629e-05]) -> consistent with no arbitrage

## Reproduction gate (independent seed)

- 0 witnesses re-run on seed 20261020: PASS (same sign, overlapping CIs)

## Outcome

- No calendar violation found at any tested rho > 0 point: the
  fixed-node quadrature 'violations' are artifacts, and the convex-order
  question stays open (calendar_noarb.md section 4) — but now with a
  quadrature-free negative result over the same box.
