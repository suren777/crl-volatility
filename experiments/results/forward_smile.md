# Forward-start smiles under MSV

Produced by `uv run python experiments/forward_smile.py` (S0=100, r=0.02,
(a0,a1,a2,lam)=(0.05,0.03,0.02,1.2), k in {0.35, 0.6}, T1 in {0.25, 1.0},
T2 = T1 + 0.25, m in {0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2}). Prices are exact
Gauss-Hermite: `S0 * E_zeta[BS(1, m, r, T2-T1, zeta*vf)]` with forward variance
rate `vf = (T2*Q(T2) - T1*Q(T1)) / (T2 - T1)`. Spot comparison uses strikes
K = m*F(tau), tau = T2-T1 (same K/F moneyness).
## k=0.35, T1=0.25 (T2=0.5), forward variance rate vf=0.060498

| m | forward IV | spot IV (tau=T2-T1, K=m*F) |
|---|------------|-------------------------------|
| 0.80 | 0.253645 | 0.265257 |
| 0.90 | 0.245279 | 0.257397 |
| 0.95 | 0.243182 | 0.255509 |
| 1.00 | 0.242428 | 0.254908 |
| 1.05 | 0.242883 | 0.255452 |
| 1.10 | 0.244354 | 0.256954 |
| 1.20 | 0.249487 | 0.262031 |

Symmetry in forward log-moneyness ln(K/F_fwd) = ln(m) - r*(T2-T1) = +/-x:

| x | IV at m=exp(r*dt+x) | IV at m=exp(r*dt-x) | abs diff |
|---|---------------------|---------------------|----------|
| 0.05 | 0.243022 | 0.243022 | 3.61e-16 |
| 0.10 | 0.244780 | 0.244780 | 1.39e-16 |
| 0.20 | 0.251252 | 0.251252 | 4.44e-16 |

Claim (b): given zeta, the T1-conditional smile expiring at T2 is flat:

| zeta | sqrt(zeta*vf) | min IV | max IV | spread |
|------|---------------|--------|--------|--------|
| 0.5396 | 0.180678 | 0.180678 | 0.180678 | 7.44e-15 |
| 0.7505 | 0.213076 | 0.213076 | 0.213076 | 4.15e-14 |
| 0.9439 | 0.238959 | 0.238959 | 0.238959 | 1.83e-14 |
| 1.1871 | 0.267985 | 0.267985 | 0.267985 | 2.33e-14 |
| 1.6510 | 0.316039 | 0.316039 | 0.316039 | 2.71e-14 |
max spread across strikes = 4.15e-14 (< 1e-8: True)

Claim (c): MC vs closed form (n_paths=200000, n_steps=200):
note: msv.simulate applies forward variance rate 0.060498 over [T1,T2] (+0.00% vs model vf=0.060498) -- exact per-step g-integrals (epic 0009), so its MC must match the closed form up to noise.

| m | closed | MC simulate | diff | 3*SE | inside |
|---|--------|-------------|------|------|--------|
| 0.90 | 11.613355 | 11.630987 | 0.017632 | 0.073282 | True |
| 1.00 | 5.074198 | 5.069937 | 0.004260 | 0.054160 | True |
| m | closed | MC exact | diff | 3*SE | inside |
|---|--------|----------|------|------|--------|
| 0.90 | 11.613355 | 11.623674 | 0.010319 | 0.073266 | True |
| 1.00 | 5.074198 | 5.070895 | 0.003303 | 0.054121 | True |

## k=0.35, T1=1.0 (T2=1.25), forward variance rate vf=0.043491

| m | forward IV | spot IV (tau=T2-T1, K=m*F) |
|---|------------|-------------------------------|
| 0.80 | 0.218305 | 0.265257 |
| 0.90 | 0.208878 | 0.257397 |
| 0.95 | 0.206439 | 0.255509 |
| 1.00 | 0.205553 | 0.254908 |
| 1.05 | 0.206089 | 0.255452 |
| 1.10 | 0.207807 | 0.256954 |
| 1.20 | 0.213672 | 0.262031 |

Symmetry in forward log-moneyness ln(K/F_fwd) = ln(m) - r*(T2-T1) = +/-x:

| x | IV at m=exp(r*dt+x) | IV at m=exp(r*dt-x) | abs diff |
|---|---------------------|---------------------|----------|
| 0.05 | 0.206252 | 0.206252 | 7.49e-16 |
| 0.10 | 0.208301 | 0.208301 | 1.64e-15 |
| 0.20 | 0.215650 | 0.215650 | 5.55e-17 |

Claim (b): given zeta, the T1-conditional smile expiring at T2 is flat:

| zeta | sqrt(zeta*vf) | min IV | max IV | spread |
|------|---------------|--------|--------|--------|
| 0.5396 | 0.153191 | 0.153191 | 0.153191 | 5.69e-14 |
| 0.7505 | 0.180661 | 0.180661 | 0.180661 | 8.41e-15 |
| 0.9439 | 0.202606 | 0.202606 | 0.202606 | 6.33e-15 |
| 1.1871 | 0.227217 | 0.227217 | 0.227217 | 3.97e-15 |
| 1.6510 | 0.267960 | 0.267960 | 0.267960 | 2.23e-14 |
max spread across strikes = 5.69e-14 (< 1e-8: True)

Claim (c): MC vs closed form (n_paths=200000, n_steps=200):
note: msv.simulate applies forward variance rate 0.043491 over [T1,T2] (+0.00% vs model vf=0.043491) -- exact per-step g-integrals (epic 0009), so its MC must match the closed form up to noise.

| m | closed | MC simulate | diff | 3*SE | inside |
|---|--------|-------------|------|------|--------|
| 0.90 | 11.186520 | 11.185751 | 0.000769 | 0.067014 | True |
| 1.00 | 4.342370 | 4.348420 | 0.006050 | 0.047084 | True |
| m | closed | MC exact | diff | 3*SE | inside |
|---|--------|----------|------|------|--------|
| 0.90 | 11.186520 | 11.184066 | 0.002454 | 0.067367 | True |
| 1.00 | 4.342370 | 4.346501 | 0.004130 | 0.047032 | True |

## k=0.6, T1=0.25 (T2=0.5), forward variance rate vf=0.060498

| m | forward IV | spot IV (tau=T2-T1, K=m*F) |
|---|------------|-------------------------------|
| 0.80 | 0.263351 | 0.273803 |
| 0.90 | 0.244205 | 0.255474 |
| 0.95 | 0.238741 | 0.250497 |
| 1.00 | 0.236669 | 0.248838 |
| 1.05 | 0.237928 | 0.250341 |
| 1.10 | 0.241842 | 0.254333 |
| 1.20 | 0.254239 | 0.266635 |

Symmetry in forward log-moneyness ln(K/F_fwd) = ln(m) - r*(T2-T1) = +/-x:

| x | IV at m=exp(r*dt+x) | IV at m=exp(r*dt-x) | abs diff |
|---|---------------------|---------------------|----------|
| 0.05 | 0.238308 | 0.238308 | 2.78e-16 |
| 0.10 | 0.242939 | 0.242939 | 1.39e-16 |
| 0.20 | 0.258186 | 0.258186 | 2.11e-15 |

Claim (b): given zeta, the T1-conditional smile expiring at T2 is flat:

| zeta | sqrt(zeta*vf) | min IV | max IV | spread |
|------|---------------|--------|--------|--------|
| 0.3444 | 0.144353 | 0.144353 | 0.144353 | 5.19e-14 |
| 0.5899 | 0.188916 | 0.188916 | 0.188916 | 2.05e-14 |
| 0.8575 | 0.227764 | 0.227764 | 0.227764 | 2.36e-15 |
| 1.2464 | 0.274600 | 0.274600 | 0.274600 | 1.59e-14 |
| 2.1348 | 0.359371 | 0.359371 | 0.359371 | 1.62e-14 |
max spread across strikes = 5.19e-14 (< 1e-8: True)

Claim (c): MC vs closed form (n_paths=200000, n_steps=200):
note: msv.simulate applies forward variance rate 0.060498 over [T1,T2] (+0.00% vs model vf=0.060498) -- exact per-step g-integrals (epic 0009), so its MC must match the closed form up to noise.

| m | closed | MC simulate | diff | 3*SE | inside |
|---|--------|-------------|------|------|--------|
| 0.90 | 11.599884 | 11.557619 | 0.042265 | 0.073366 | True |
| 1.00 | 4.959923 | 4.943610 | 0.016313 | 0.054574 | True |
| m | closed | MC exact | diff | 3*SE | inside |
|---|--------|----------|------|------|--------|
| 0.90 | 11.599884 | 11.593551 | 0.006333 | 0.073332 | True |
| 1.00 | 4.959923 | 4.940755 | 0.019168 | 0.054708 | True |

## k=0.6, T1=1.0 (T2=1.25), forward variance rate vf=0.043491

| m | forward IV | spot IV (tau=T2-T1, K=m*F) |
|---|------------|-------------------------------|
| 0.80 | 0.230002 | 0.273803 |
| 0.90 | 0.209319 | 0.255474 |
| 0.95 | 0.203100 | 0.250497 |
| 1.00 | 0.200679 | 0.248838 |
| 1.05 | 0.202155 | 0.250341 |
| 1.10 | 0.206656 | 0.254333 |
| 1.20 | 0.220315 | 0.266635 |

Symmetry in forward log-moneyness ln(K/F_fwd) = ln(m) - r*(T2-T1) = +/-x:

| x | IV at m=exp(r*dt+x) | IV at m=exp(r*dt-x) | abs diff |
|---|---------------------|---------------------|----------|
| 0.05 | 0.202598 | 0.202598 | 4.44e-16 |
| 0.10 | 0.207897 | 0.207897 | 1.44e-15 |
| 0.20 | 0.224539 | 0.224539 | 2.22e-16 |

Claim (b): given zeta, the T1-conditional smile expiring at T2 is flat:

| zeta | sqrt(zeta*vf) | min IV | max IV | spread |
|------|---------------|--------|--------|--------|
| 0.3444 | 0.122393 | 0.122393 | 0.122393 | 7.19e-14 |
| 0.5899 | 0.160176 | 0.160176 | 0.160176 | 4.84e-14 |
| 0.8575 | 0.193114 | 0.193114 | 0.193114 | 2.23e-14 |
| 1.2464 | 0.232825 | 0.232825 | 0.232825 | 7.27e-15 |
| 2.1348 | 0.304700 | 0.304700 | 0.304700 | 2.14e-14 |
max spread across strikes = 7.19e-14 (< 1e-8: True)

Claim (c): MC vs closed form (n_paths=200000, n_steps=200):
note: msv.simulate applies forward variance rate 0.043491 over [T1,T2] (+0.00% vs model vf=0.043491) -- exact per-step g-integrals (epic 0009), so its MC must match the closed form up to noise.

| m | closed | MC simulate | diff | 3*SE | inside |
|---|--------|-------------|------|------|--------|
| 0.90 | 11.191282 | 11.222420 | 0.031138 | 0.068081 | True |
| 1.00 | 4.245641 | 4.223485 | 0.022156 | 0.048508 | True |
| m | closed | MC exact | diff | 3*SE | inside |
|---|--------|----------|------|------|--------|
| 0.90 | 11.191282 | 11.177969 | 0.013313 | 0.067693 | True |
| 1.00 | 4.245641 | 4.279103 | 0.033462 | 0.048487 | True |

ALL CHECKS PASSED
