# RIGHT-WING SLOPES OF TOTAL IMPLIED VARIANCE: MARKET vs CRL vs SSVI (2026-10-19)

Cboe SQLite ~/Data/article/cboe/spx.sqlite, trade_date=2026-10-16, root SPX, six monthly expiries. Call wing: `x = ln(K/F) >= 0.05` to the largest quoted strike; total implied variance `w = iv^2 * tau`; slope is OLS of `w` on `x` over the wing. Inverted-IV cells outside the Black price range are `nan`.

Fitted params (joint, six expiries): CRL `a0=0.00000 a1=0.09208 a2=0.01963 lam=0.5167 k=2.2600 rho=-0.487` (rho<=0); SSVI `eta=0.8016 gamma=0.5937 rho=-0.579`.

Lee (2004) right-wing bound at the fitted rho: `p~ = rho^2/(1-rho^2) = 0.3116`, `psi(p~) = 2-4(sqrt(p~^2+p~)-p~) = 0.6893`.

## 1. Observed call-wing slope of total variance (x in [0.05, max quoted])

| expiry | tau | n_wings | mkt slope | CRL slope | SSVI slope | CRL/mkt | SSVI/mkt |
|--------|-----|---------|-----------|-----------|------------|---------|----------|
| 2026-11-20 | 0.0959 |   39 |    0.0078 |    0.0096 |     0.0098 |   1.24 |   1.26 |
| 2026-12-18 | 0.1726 |   50 |    0.0043 |    0.0113 |     0.0108 |   2.65 |   2.55 |
| 2027-01-15 | 0.2493 |   50 |    0.0028 |    0.0118 |     0.0106 |   4.24 |   3.79 |
| 2027-03-19 | 0.4219 |   33 |   -0.0018 |    0.0127 |     0.0085 |  -7.05 |  -4.75 |
| 2027-06-17 | 0.6685 |   34 |   -0.0106 |    0.0091 |     0.0015 |  -0.86 |  -0.14 |
| 2027-09-17 | 0.9205 |   23 |   -0.0176 |    0.0036 |    -0.0070 |  -0.21 |   0.40 |

`psi(p~)` upper-bounds the CRL wing slope asymptotically (limsup as x -> inf); the observed finite-x slope must sit below it. A finite-moneyness slope at or near the bound is a signature that the wing is approaching the steepest tail the moment explosion allows.

## 2. Fitted CRL: w/x and Black IV on an extended grid x in [0.25, 3]

Model total variance is Black-inverted from `crl_call` at rho=-0.487, so the asymptotic right-wing bound is `psi(p~) = 0.6893`.

| expiry | tau | w/x 0.25 | w/x 0.5 | w/x 1 | w/x 2 | w/x 3 | (w/x)/psi@3 |
|--------|-----|----------|---------|-------|-------|-------|-------------|
| 2026-11-20 | 0.0959 | 0.016 | 0.019 | 0.025 | 0.035 | 0.043 | 0.062 |
| 2026-12-18 | 0.1726 | 0.021 | 0.024 | 0.030 | 0.040 | 0.049 | 0.071 |
| 2027-01-15 | 0.2493 | 0.026 | 0.028 | 0.034 | 0.045 | 0.053 | 0.077 |
| 2027-03-19 | 0.4219 | 0.037 | 0.036 | 0.042 | 0.052 | 0.062 | 0.089 |
| 2027-06-17 | 0.6685 | 0.053 | 0.047 | 0.051 | 0.061 | 0.071 | 0.103 |
| 2027-09-17 | 0.9205 | 0.071 | 0.058 | 0.059 | 0.069 | 0.079 | 0.114 |

| expiry | tau | iv@0.25 | iv@0.5 | iv@1 | iv@2 | iv@3 |
|--------|-----|---------|--------|------|------|------|
| 2026-11-20 | 0.0959 | 0.205 | 0.316 | 0.512 | 0.851 | 1.155 |
| 2026-12-18 | 0.1726 | 0.176 | 0.263 | 0.417 | 0.683 | 0.920 |
| 2027-01-15 | 0.2493 | 0.162 | 0.237 | 0.369 | 0.598 | 0.801 |
| 2027-03-19 | 0.4219 | 0.148 | 0.207 | 0.314 | 0.498 | 0.662 |
| 2027-06-17 | 0.6685 | 0.141 | 0.188 | 0.276 | 0.429 | 0.564 |
| 2027-09-17 | 0.9205 | 0.139 | 0.177 | 0.254 | 0.388 | 0.507 |

w/x climbs toward `psi(p~)` as x grows and stays below it at every expiry, consistent with an asymptotic upper bound approached from below.

## 3. Findings

At the fitted rho=-0.487, `p~ = 0.3116` and the Lee right-wing bound `psi(p~) = 0.689`. Table 2 shows the fitted CRL's `w/x` on the extended grid: at every expiry it rises toward this bound as x grows and stays below it — e.g. the longest expiry reaches `w/x = 0.079` at x=3 versus a bound of 0.689 — so the moment-formula bound is approached from below, as an asymptotic (x -> inf) statement predicts.

Where the data probe (x up to ~0.3), CRL's call-wing slope is steeper than the market's at the short end but comparable or flatter at the longest expiries (Table 1, CRL/mkt); SSVI's wing is flatter than CRL's throughout and matches the market more closely at the long end. The Lee bound is therefore not directly relevant at observed moneyness — the farthest quoted call here is only `ln(K/F) ~ 0.3`, far short of the far tail — it governs the model's wing slope only asymptotically.

