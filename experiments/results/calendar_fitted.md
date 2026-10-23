# Calendar check of `crl_call` at the fitted S&P 500 parameter sets

Run 2026-10-23. For each CRL fit in `results/panel.csv`, undiscounted call prices with F = D = 1 on K/F in [0.3, 3] (200 log-spaced points) and tau in [1/365, 5] (200 log-spaced points), 64-node Gauss-Hermite. Checked: the increments C(K, tau_{j+1}) - C(K, tau_j) as tau increases, at every strike; a violation is an increment below -1e-10. A date with a violation is re-priced at 128 nodes. No hand-edited numbers; this file is the script's output.

## 2026-10-16: a0=5.9243e-09 a1=0.092082 a2=0.019633 lam=0.51674 k=2.26 rho=-0.4874 (64 nodes)
- min increment +9.089e-24 at K/F=3.0000, tau 0.00273973 -> 0.00284509
- increments below -1e-10: 0 of 39800

## 2026-10-19: a0=0.021726 a1=0.19941 a2=6.0497e-10 lam=0.20895 k=2.2959 rho=-0.4905 (64 nodes)
- min increment +2.369e-23 at K/F=3.0000, tau 0.00273973 -> 0.00284509
- increments below -1e-10: 0 of 39800

## 2026-10-20: a0=1.0875e-09 a1=0.079621 a2=0.019032 lam=0.68275 k=2.1887 rho=-0.4733 (64 nodes)
- min increment +9.480e-24 at K/F=3.0000, tau 0.00273973 -> 0.00284509
- increments below -1e-10: 0 of 39800

## 2026-10-22: a0=0.024584 a1=0.30228 a2=3.568e-10 lam=0.11 k=2.1532 rho=-0.49208 (64 nodes)
- min increment +9.814e-24 at K/F=3.0000, tau 0.00273973 -> 0.00284509
- increments below -1e-10: 0 of 39800

## Conclusion
- no calendar increment is negative below -1e-10 at any of the four fits; consistent with the no-arbitrage conjecture for rho <= 0 at the fitted parameters.