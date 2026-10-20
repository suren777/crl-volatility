# Calibration objective weighting — CRL (ρ ≤ 0) on the six SPX monthlies

Epic 0002 task 5. Compares the current vega-weighted price residual used in our
SPX fits with the 2014 paper's squared-price-error objective weighted by
1/spread⁴ and with unweighted price residuals. One joint CRL fit (a0,a1,a2,λ,k,ρ),
ρ ∈ [−0.99, 0], over the six OTM fit sets at the 2026-10-16 Cboe close, same
fit sets and starts as `experiments/crl_short.py`. Data read-only from
`spx.sqlite`; nothing copied into the repo.

Run: `uv run python experiments/objective_weighting.py`

## Objectives and implied noise models

Same price residual `r = px − mid` in all three; only the weight differs. We
write `resid = r / w`. If the optimisation assumes iid standard-normal
residuals, then `Var(price error) ∝ w²`, giving the noise model.

| key | residual w | weight on square | implied price-error sd |
|-----|-----------|------------------|------------------------|
| `vega` | `D·vega` | 1/(D·vega)² | ∝ vega (≈ homoscedastic in implied vol) |
| `spread2014` | `spread²` | 1/spread⁴ | ∝ spread² (2014 paper) |
| `none` | 1 | 1 | constant (homoscedastic in price) |

`vega` is the current objective: dividing the price error by per-option vega
makes the objective approximately an implied-vol fit, so its noise model is
“error sd ∝ vega”, i.e. residuals roughly homoscedastic in volatility space —
matching the metric (IV RMSE) the paper reports. `spread2014` down-weights an
option the wider its spread (sd ∝ spread²): it de-emphasises the wider,
longer-dated options. `none` treats every strike's price error equally, which
lets deep-OTM wings (tiny prices, largest relative IV error) dominate the vol
fit.

## Fit set (1458 options)

| expiry | τ | F | D | n | median spread |
|--------|------|---------|--------|-----|-----------|
| 2026-11-20 | 0.0959 | 7720.69 | 0.99589 | 377 | 0.600 |
| 2026-12-18 | 0.1726 | 7740.32 | 0.99230 | 363 | 1.000 |
| 2027-01-15 | 0.2493 | 7769.44 | 0.98863 | 334 | 1.100 |
| 2027-03-19 | 0.4219 | 7817.07 | 0.98071 | 130 | 1.100 |
| 2027-06-17 | 0.6685 | 7898.13 | 0.96839 | 132 | 1.500 |
| 2027-09-17 | 0.9205 | 7984.22 | 0.95577 | 122 | 1.600 |

## Market ATM skew (vol pts per log-moneyness), |ln(K/F)| ≤ 0.05

| expiry | skew |
|--------|------|
| 2026-11-20 | −80.436 |
| 2026-12-18 | −66.329 |
| 2027-01-15 | −58.180 |
| 2027-03-19 | −45.295 |
| 2027-06-17 | −38.041 |
| 2027-09-17 | −33.292 |

## Fits and metrics (numbers as printed)

### Objective `vega` — a0=0.00000 a1=0.09208 a2=0.01963 λ=0.5167 k=2.2600 ρ=−0.487, cost=0.0200, [11.3 s]

| expiry | IV RMSE | px RMSE | skew mkt | skew mod | mod/mkt |
|--------|---------|---------|----------|----------|---------|
| 2026-11-20 | 0.768 | 2.9107 | −80.436 | −94.691 | 1.177 |
| 2026-12-18 | 0.505 | 3.4726 | −66.329 | −80.447 | 1.213 |
| 2027-01-15 | 0.401 | 3.7395 | −58.180 | −70.207 | 1.207 |
| 2027-03-19 | 0.378 | 4.8564 | −45.295 | −55.385 | 1.223 |
| 2027-06-17 | 0.398 | 6.6345 | −38.041 | −44.460 | 1.169 |
| 2027-09-17 | 0.365 | 7.6978 | −33.292 | −38.028 | 1.142 |
| **overall** | **0.540** | **4.4094** | | | |

1458 options, 6.9% inside bid-ask spread.

### Objective `spread2014` — a0=0.02075 a1=0.43609 a2=0.00002 λ=0.0788 k=2.3527 ρ=−0.496, cost=33139.8657, [48.2 s]

| expiry | IV RMSE | px RMSE | skew mkt | skew mod | mod/mkt |
|--------|---------|---------|----------|----------|---------|
| 2026-11-20 | 0.771 | 3.7067 | −80.436 | −98.143 | 1.220 |
| 2026-12-18 | 0.507 | 4.0843 | −66.329 | −83.108 | 1.253 |
| 2027-01-15 | 0.423 | 4.3894 | −58.180 | −72.433 | 1.245 |
| 2027-03-19 | 0.480 | 7.0792 | −45.295 | −57.168 | 1.262 |
| 2027-06-17 | 0.531 | 10.4990 | −38.041 | −45.908 | 1.207 |
| 2027-09-17 | 0.456 | 10.9696 | −33.292 | −39.250 | 1.179 |
| **overall** | **0.568** | **6.0528** | | | |

1458 options, 6.7% inside bid-ask spread.

### Objective `none` — a0=0.01938 a1=0.12205 a2=0.00000 λ=0.4848 k=1.9843 ρ=−0.463, cost=4172.0215, [10.4 s]

| expiry | IV RMSE | px RMSE | skew mkt | skew mod | mod/mkt |
|--------|---------|---------|----------|----------|---------|
| 2026-11-20 | 1.101 | 1.8688 | −80.436 | −85.770 | 1.066 |
| 2026-12-18 | 0.734 | 2.4335 | −66.329 | −72.623 | 1.095 |
| 2027-01-15 | 0.535 | 2.6388 | −58.180 | −63.350 | 1.089 |
| 2027-03-19 | 0.475 | 2.9262 | −45.295 | −49.828 | 1.100 |
| 2027-06-17 | 0.344 | 2.2429 | −38.041 | −40.016 | 1.052 |
| 2027-09-17 | 0.310 | 2.5204 | −33.292 | −34.293 | 1.030 |
| **overall** | **0.743** | **2.3923** | | | |

1458 options, 16.7% inside bid-ask spread.

All objective definitions and the per-expiry IV/price RMSE, skew ratio and
inside-spread fraction are what the script printed.

## Reading

- **IV RMSE:** `vega` 0.540 is the best overall and per expiry it is best or
  near-best everywhere; `spread2014` 0.568 is close; `none` 0.743 is clearly
  worst, and its worst errors are the short-dated expiries (11-20: 1.101,
  12-18: 0.734).
- **Price RMSE:** the reverse ordering — `none` 2.39 < `vega` 4.41 <
  `spread2014` 6.05. Unweighted price errors concentrate the fit where prices
  are large (ATM and long-dated), so deep-OTM wings (tiny prices) are left
  mispriced in IV terms — hence the high IV RMSE. `spread2014` is worst in
  price: its 1/spread⁴ weighting de-emphasises exactly the wide, long-dated
  options, whose prices it then misses by ~10 $ at 06-17/09-17.
- **ATM skew ratio (model/market):** `none` is closest to 1 (1.03–1.10),
  `vega` overshoots ~14–22%, `spread2014` overshoots most (~18–26%). So pure
  ATM-skew recovery favours `none`, at the cost of the worst all-round vol fit.
- **Inside spread:** only 6.9% / 6.7% / 16.7% of fit options are priced within
  their own bid–ask. The 11-20/12-18 medians are 0.6/1.0 index points while the
  joint 6-parameter surface has IV RMSE ~0.5 vol pts, i.e. price errors exceed
  a typical tight quote for most strikes. A single joint CRL surface does not
  replicate each quoted mid within its (often sub-point) spread — expected for
  any stiff 6-parameter fit on these spreads.

## Recommendation

**Keep the vega-weighted objective (`vega`) for the paper's SPX calibration.**
Reasons:

1. Its noise model (price-error sd ∝ vega, i.e. approximately homoscedastic in
   implied-vol space) matches the quantity the paper actually reports and
   compares (IV RMSE). This is the statistically coherent choice: the objective
   and the headline metric live in the same space.
2. It gives the best IV RMSE (0.540 overall) and is best or near-best per
   expiry; it also reproduces the crl_spx result (STATUS: IV RMSE 0.54 vol
   pts), so adopting it keeps continuity with prior evidence.
3. The 2014 1/spread⁴ scheme (`spread2014`) is dominated in vol terms (0.568),
   is ~4× slower (48 s vs 11 s), pushes the fitted surface to a harsher shape
   (a1 = 0.436, λ = 0.079 pinned near its lower bound) and de-emphasises the
   long-dated expiries where it then misses prices by ~10 $. Nothing in the
   current evidence earns its heavier weighting.

`none` (unweighted price) is a defensible contrast but dominated for vol
purposes: it best matches ATM skew (ratio closest to 1) and tick-marks the most
options inside the spread, but it marginally fits all short-dated wings in vol
and is the worst on the paper's reported IV metric. If approving the paper's
skew narrative (not its RMSE), `none`'s near-ATM advantage is a useful
sensitivity to cite, but not the headline fit.

**Noise-model caveat for parameter standard errors.** Under `vega` the residuals
imply price-error sd ∝ vega. JᵀJ-based standard errors must be computed against
that noise model (error sd ∝ vega), not against a constant-sd assumption; the
SE task (0002 round 5) should adopt this. Conversely, parameter SEs quoted
under a constant-sd (unweighted) assumption would be meaningless for the vega
fit.

Out of scope here: the flat-(a0,a1,λ) direction of §0002 task 6; per-parameter
SEs are deferred to round 5.
