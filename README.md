# Correlated random-level volatility model

Code accompanying the preprint *A correlated random-level volatility model: exact
one-dimensional pricing with skew* (S. Islyaev, 2026).

A single Gaussian factor drives a static lognormal volatility level, which multiplies a
Nelson–Siegel variance term structure, and the returns of the underlying, with correlation ρ.
European options are priced exactly by Gauss–Hermite quadrature. At ρ = 0 the model is the
MSV model of Date & Islyaev (2015), *European Journal of Operational Research* 243(2), 599–606.

## Layout

| Path | Contents |
|---|---|
| `msv.py` | Pricers: `crl_call` (exact, ρ ≤ 0), `gh_call`, the MSV Taylor pricer, calibration |
| `derivations/` | Symbolic and numerical checks of the paper's formulas (skew asymptotics, moments, calendar arbitrage) |
| `experiments/` | S&P 500 fits (CRL, SSVI, Heston, MSV), skew estimators, figures; results in `experiments/results/` |
| `replication/` | Port of the 2014 pricer and replication of the published 2012 results (derived metrics only) |
| `collector/` | Daily snapshot of the delayed Cboe S&P 500 option chain |

## Usage

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv sync
uv run pytest -q
uv run ruff check . && uv run mypy msv.py test_msv.py experiments
```

## Data

No market data is distributed here. The S&P 500 experiments read a SQLite database built by
`collector/cboe_snapshot.py` from Cboe's public delayed quotes; set `CRL_CBOE_DB` to its path
(default `~/Data/article/cboe/spx.sqlite`). The replication scripts need the original 2012–2013
data, which is licensed and not included; their recorded results are in `replication/results/`.
Use of market data is subject to the provider's terms.

## Citation

Paper: Islyaev, S., *A Correlated Random-level Volatility Model: Exact One-dimensional Pricing
with Skew* (October 23, 2026). SSRN: [abstract 7664360](https://ssrn.com/abstract=7664360),
[doi:10.2139/ssrn.7664360](https://doi.org/10.2139/ssrn.7664360).

Code archived on Zenodo: [doi:10.5281/zenodo.23600840](https://doi.org/10.5281/zenodo.23600840)
(all versions; v1.0.0 is [10.5281/zenodo.23600841](https://doi.org/10.5281/zenodo.23600841)).
See `CITATION.cff` for the citation entry.

## Licence

MIT; see `LICENSE`.
