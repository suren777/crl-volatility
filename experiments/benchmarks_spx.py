"""Benchmark CRL against SSVI and Heston on the same SPX data and machine (epic 0003).

Usage: uv run python experiments/benchmarks_spx.py

Same data and pipeline as crl_spx.py: yfinance CSV + Cboe SQLite (2026-10-16),
six monthly expiries, OTM fit set, vega-weighted price residuals. Rows: plain
MSV, CRL (rho <= 0), SSVI (power-law phi), Heston. Reports IV RMSE overall and
per expiry, ATM skew ratio per expiry, parameter count and fit time per model.

SSVI (Gatheral & Jacquier 2014, DOI 10.1080/14697688.2013.819986, arXiv:1204.0646):

    w(k, theta) = theta/2 * {1 + rho*phi(theta)*k
                             + sqrt((phi(theta)*k + rho)^2 + 1 - rho^2)},

power-law phi(theta) = eta*theta^(-gamma). theta_t is the per-expiry ATM total
variance (fit theta per expiry, six free thetas). No-arbitrage enforced (their
Corollary 4.1): for power-law phi, condition 2 (theta*(theta*phi)' <= bound)
holds for any gamma in [0,1], and calendar monotonicity (condition 1, theta non-
decreasing in t) is enforced by parameterising the six thetas as cumulative
positive increments; the two butterfly bounds (conditions 3 and 4) are enforced
as penalty residuals. rho is bounded to |rho|<1.

Heston: 5 parameters (v0, kappa, theta, sigma, rho), the fast heston_call_vec
path. Feller condition 2*kappa*theta/sigma^2 is reported, not imposed.
Both new models use multi-start likes the existing MSV/CRL fits.

eparameter counts used in the table: plain MSV 5, CRL 6, SSVI 9 (eta, gamma,
rho + six thetas), Heston 5.
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import crl_spx  # noqa: E402
import heston  # noqa: E402
import numpy as np  # noqa: E402
import shifted_spx  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.optimize import OptimizeResult, least_squares  # noqa: E402
from scipy.stats import norm  # noqa: E402
from shifted_spx import (  # noqa: E402
    EXPIRIES,
    SKEW_LN_MONEYNESS,
    ExpiryData,
    atm_skew,
    black_vega,
    load_expiry,
    rmse_volpts,
)

import msv  # noqa: E402

BUTTERFLY_WEIGHT = 50.0  # penalty multiplier for the two Corollary-4.1 butterfly bounds

# --------------------------------------------------------------------------- CRL


def fit_crl_rho0(data: list[ExpiryData]) -> tuple[OptimizeResult, float]:
    """crl_spx.fit_crl with the rho upper bound clamped to 0.

    The paper's exact pricer msv.crl_call is documented reliable only for
    rho <= 0, so every CRL fit constrains rho to [-0.99, 0]. Copy of
    crl_spx.fit_crl with ub[-1] changed from 0.99 to 0.0.
    """
    F = np.concatenate([np.full(d.K.size, d.F) for d in data])
    D = np.concatenate([np.full(d.K.size, d.D) for d in data])
    tau = np.concatenate([np.full(d.K.size, d.tau) for d in data])
    K = np.concatenate([d.K for d in data])
    mid = np.concatenate([d.mid for d in data])
    vega = np.array(
        [black_vega(d.F, k, d.tau, iv) for d in data for k, iv in zip(d.K, d.iv_mkt, strict=True)]
    )
    denom = D * vega

    def resid(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        a0, a1, a2, lam, k, rho = (float(v) for v in theta)
        px = msv.crl_call(F, K, D, tau, a0, a1, a2, lam, k, rho)
        return np.asarray((px - mid) / denom, dtype=float)

    v_atm = [float(d.iv_mkt[np.argmin(np.abs(np.log(d.K / d.F)))]) ** 2 for d in data if d.K.size]
    v_short, v_long = min(v_atm), max(v_atm)
    base = [max(v_short - v_long, 0.0), max(v_long - v_short, 0.0) * 0.5, v_long]
    lam_k = [(1.0, 0.3), (0.5, 0.15), (2.0, 0.3), (4.0, 0.5), (0.5, 0.5)]
    lb = [0.0, 0.0, 0.0, 1e-3, 1e-3, -0.99]
    ub = [np.inf, np.inf, np.inf, 50.0, 3.0, 0.0]  # rho upper bound clamped to 0
    best: OptimizeResult | None = None
    t0 = time.perf_counter()
    for lam0, k0 in lam_k:
        for rho0 in crl_spx.RHO_STARTS:
            res = least_squares(
                resid, np.array(base + [lam0, k0, rho0]), bounds=(lb, ub), xtol=1e-10, ftol=1e-10
            )
            if best is None or res.cost < best.cost:
                best = res
    assert best is not None
    return best, time.perf_counter() - t0


# --------------------------------------------------------------------------- SSVI


def ssvi_total_var(
    k: NDArray[np.float64], theta: float, eta: float, gamma: float, rho: float
) -> NDArray[np.float64]:
    """SSVI total implied variance w(k, theta) with power-law phi = eta*theta^-gamma."""
    phi = eta * theta ** (-gamma)
    arg = phi * k + rho
    w = 0.5 * theta * (1.0 + rho * phi * k + np.sqrt(arg * arg + 1.0 - rho * rho))
    return np.asarray(w, dtype=np.float64)


def black_px_vec(
    F: NDArray[np.float64],
    K: NDArray[np.float64],
    tau: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:  # noqa: E501
    """Undiscounted forward Black call price, vectorised (shifted_spx.black_px is scalar)."""
    v = sigma * sigma * tau
    sd = np.sqrt(v)
    d1 = (np.log(F / K) + v / 2.0) / sd
    return F * np.asarray(norm.cdf(d1)) - K * np.asarray(norm.cdf(d1 - sd))


def ssvi_penalties(
    thetas: NDArray[np.float64], eta: float, gamma: float, rho: float
) -> NDArray[np.float64]:
    """Penalty rows for the two Corollary-4.1 butterfly bounds at each fitted theta."""
    out = []
    for th in thetas:
        phi = eta * th ** (-gamma)
        fac = 1.0 + abs(rho)
        out.append(BUTTERFLY_WEIGHT * max(0.0, th * phi * fac - 4.0))
        out.append(BUTTERFLY_WEIGHT * max(0.0, th * phi * phi * fac - 4.0))
    return np.array(out)


def fit_ssvi(data: list[ExpiryData]) -> tuple[OptimizeResult, float]:
    """Joint SSVI fit: (eta, gamma, rho, theta_1..theta_6), vega-weighted price residuals."""
    F = np.concatenate([np.full(d.K.size, d.F) for d in data])
    D = np.concatenate([np.full(d.K.size, d.D) for d in data])
    tau = np.concatenate([np.full(d.K.size, d.tau) for d in data])
    K = np.concatenate([d.K for d in data])
    mid = np.concatenate([d.mid for d in data])
    vega = np.array(
        [black_vega(d.F, k, d.tau, iv) for d in data for k, iv in zip(d.K, d.iv_mkt, strict=True)]
    )
    denom = D * vega
    thetas0 = np.array(
        [float(d.iv_mkt[np.argmin(np.abs(np.log(d.K / d.F)))]) ** 2 * d.tau for d in data]
    )
    thetas0 = np.maximum(thetas0, 1e-4)

    def unpack(x: NDArray[np.float64]) -> tuple[float, float, float, NDArray[np.float64]]:
        eta, gamma, rho, lth = float(x[0]), float(x[1]), float(x[2]), x[3:]
        iinc = np.zeros_like(lth)
        iinc[1:] = np.exp(np.minimum(lth[1:], 30.0))
        thetas = np.exp(lth[0]) + np.cumsum(
            np.concatenate([[0.0], np.exp(np.minimum(lth[1:], 30.0))])
        )
        return eta, gamma, rho, thetas[: len(data)]

    def resid(x: NDArray[np.float64]) -> NDArray[np.float64]:
        eta, gamma, rho, thetas = unpack(x)
        iv = np.concatenate(
            [
                np.sqrt(ssvi_total_var(np.log(d.K / d.F), th, eta, gamma, rho) / d.tau)
                for d, th in zip(data, thetas, strict=True)
            ]
        )
        px = D * black_px_vec(F, K, tau, iv)
        return np.concatenate(
            [np.asarray((px - mid) / denom), ssvi_penalties(thetas, eta, gamma, rho)]
        )

    starts: list[tuple[float, float, float]] = []
    for eta0 in (0.2, 0.8, 2.0):
        for gamma0 in (0.3, 0.5, 0.7):
            for rho0 in (-0.9, -0.5, 0.0):
                starts.append((eta0, gamma0, rho0))
    lb = [1e-4, 1e-3, -0.99, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0]
    ub = [10.0, 1.0, 0.99, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
    best: OptimizeResult | None = None
    t0 = time.perf_counter()
    for eta0, gamma0, rho0 in starts:
        x0 = np.concatenate([[eta0, gamma0, rho0], np.log(np.maximum(thetas0 - thetas0[0], 1e-3))])
        x0[3] = np.log(thetas0[0])
        res = least_squares(resid, x0, bounds=(lb, ub), xtol=1e-10, ftol=1e-10, max_nfev=4000)
        if best is None or res.cost < best.cost:
            best = res
    assert best is not None
    return best, time.perf_counter() - t0


def ssvi_model_ivs(res: OptimizeResult, data: list[ExpiryData]) -> list[NDArray[np.float64]]:
    """SSVI IV per expiry from a fitted result."""
    eta, gamma, rho, lth = float(res.x[0]), float(res.x[1]), float(res.x[2]), res.x[3:]
    thetas = np.exp(lth[0]) + np.cumsum(np.concatenate([[0.0], np.exp(np.minimum(lth[1:], 30.0))]))
    thetas = thetas[: len(data)]
    return [
        np.sqrt(ssvi_total_var(np.log(d.K / d.F), float(th), eta, gamma, rho) / d.tau)
        for d, th in zip(data, thetas, strict=True)
    ]


# ------------------------------------------------------------------------- Heston


def fit_heston(data: list[ExpiryData]) -> tuple[OptimizeResult, float]:
    """Joint Heston fit (v0, kappa, theta, sigma, rho) via the fast heston_call_vec."""
    D = np.concatenate([np.full(d.K.size, d.D) for d in data])
    mid = np.concatenate([d.mid for d in data])
    vega = np.array(
        [black_vega(d.F, k, d.tau, iv) for d in data for k, iv in zip(d.K, d.iv_mkt, strict=True)]
    )
    denom = D * vega

    def resid(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        v0, kappa, th, sigma, rho = (float(v) for v in theta)
        px = np.concatenate(
            [heston.heston_call_vec(d.F, d.K, d.tau, v0, kappa, th, sigma, rho) for d in data]
        )
        return np.asarray((D * px - mid) / denom, dtype=float)

    atm = [float(d.iv_mkt[np.argmin(np.abs(np.log(d.K / d.F)))]) ** 2 for d in data if d.K.size]
    v0_0 = min(atm)
    th_0 = max(atm)
    starts: list[tuple[float, float, float, float, float]] = []
    for kappa0 in (0.8, 2.0, 5.0):
        for sigma0 in (0.3, 0.8):
            for rho0 in (-0.9, -0.5):
                starts.append((v0_0, kappa0, th_0, sigma0, rho0))
    lb = [1e-4, 1e-3, 1e-4, 1e-4, -0.99]
    ub = [1.0, 20.0, 1.0, 5.0, 0.99]
    best: OptimizeResult | None = None
    t0 = time.perf_counter()
    for st in starts:
        res = least_squares(resid, np.array(st), bounds=(lb, ub), xtol=1e-10, ftol=1e-10)
        if best is None or res.cost < best.cost:
            best = res
    assert best is not None
    return best, time.perf_counter() - t0


def heston_model_ivs(res: OptimizeResult, data: list[ExpiryData]) -> list[NDArray[np.float64]]:
    """Heston Back-Black IV of fitted prices (market-comparable quote)."""
    v0, kappa, th, sigma, rho = (float(v) for v in res.x)
    out = []
    for d in data:
        px = heston.heston_call_vec(d.F, d.K, d.tau, v0, kappa, th, sigma, rho)
        ivs = [
            shifted_spx.implied_iv(d.F, float(kk), d.tau, float(p))
            for kk, p in zip(d.K, px, strict=True)
        ]
        out.append(np.array([np.nan if iv is None else iv for iv in ivs]))
    return out


# --------------------------------------------------------------------- reporting


def run_source(name: str, raw: list[dict[str, str]], spot: float) -> None:
    data = [load_expiry(raw, e, spot) for e in EXPIRIES]
    print(f"\n{'=' * 78}\n{name}  spot={spot:.2f}  asof={shifted_spx.ASOF}")
    print(
        f"\nPer-expiry parity forward/discount and OTM fit set "
        f"(|ln(K/F)|<={shifted_spx.MAX_LN_MONEYNESS}):"
    )
    print(f"{'expiry':<12} {'tau':>7} {'F':>10} {'D':>8} {'n_options':>10}")
    for d in data:
        print(f"{d.expiry:<12} {d.tau:>7.4f} {d.F:>10.2f} {d.D:>8.5f} {d.K.size:>10d}")

    plain, t_plain = shifted_spx.fit_model(data, shifted=False, spot=spot)
    crl, t_crl = fit_crl_rho0(data)
    ssvi_res, t_ssvi = fit_ssvi(data)
    hres, t_heston = fit_heston(data)
    p, c, s, h = plain.x, crl.x, ssvi_res.x, hres.x

    print("\nFitted parameters (joint over six expiries) and fit time:")
    print(
        f"  plain MSV : a0={p[0]:.5f} a1={p[1]:.5f} a2={p[2]:.5f} lam={p[3]:.4f} "
        f"k={p[4]:.4f}  cost={plain.cost:.4f}  [{t_plain:.1f}s]"
    )
    print(
        f"  CRL       : a0={c[0]:.5f} a1={c[1]:.5f} a2={c[2]:.5f} lam={c[3]:.4f} "
        f"k={c[4]:.4f} rho={c[5]:.3f} (rho<=0)  cost={crl.cost:.4f}  [{t_crl:.1f}s]"
    )
    s_thetas = np.exp(s[3]) + np.cumsum(np.concatenate([[0.0], np.exp(np.minimum(s[4:], 30.0))]))
    print(
        f"  SSVI      : eta={s[0]:.4f} gamma={s[1]:.4f} rho={s[2]:.3f} "
        f"cost={ssvi_res.cost:.4f}  [{t_ssvi:.1f}s]"
    )
    print("              per-expiry theta (ATM total variance):")
    print("             " + " ".join(f"{t:.5f}" for t in s_thetas[: len(data)]))
    print(
        f"  Heston    : v0={h[0]:.5f} kappa={h[1]:.4f} theta={h[2]:.5f} "
        f"sigma={h[3]:.4f} rho={h[4]:.3f}  cost={hres.cost:.4f}  [{t_heston:.1f}s]"
    )
    feller = 2.0 * h[1] * h[2] / h[3] ** 2
    print(
        f"              Feller condition 2*kappa*theta/sigma^2 = {feller:.3f} "
        f"({'HOLDS' if feller >= 1.0 else 'VIOLATED'})"
    )

    ivs_plain = shifted_spx.model_ivs(plain, data, shifted=False)
    ivs_crl = crl_spx.crl_model_ivs(crl, data)
    ivs_ssvi = ssvi_model_ivs(ssvi_res, data)
    ivs_hes = heston_model_ivs(hres, data)

    print("\nIV RMSE (vol points):")
    print(f"{'expiry':<12} {'n':>4} {'MSV':>7} {'CRL':>7} {'SSVI':>7} {'Heston':>7}")
    for d, a, b, c_, e in zip(data, ivs_plain, ivs_crl, ivs_ssvi, ivs_hes, strict=True):
        print(
            f"{d.expiry:<12} {d.K.size:>4} {rmse_volpts(a, d.iv_mkt):>7.3f} "
            f"{rmse_volpts(b, d.iv_mkt):>7.3f} {rmse_volpts(c_, d.iv_mkt):>7.3f} "
            f"{rmse_volpts(e, d.iv_mkt):>7.3f}"
        )
    all_mkt = np.concatenate([d.iv_mkt for d in data])
    print(
        f"{'overall':<12} {all_mkt.size:>4} "
        f"{rmse_volpts(np.concatenate(ivs_plain), all_mkt):>7.3f} "
        f"{rmse_volpts(np.concatenate(ivs_crl), all_mkt):>7.3f} "
        f"{rmse_volpts(np.concatenate(ivs_ssvi), all_mkt):>7.3f} "
        f"{rmse_volpts(np.concatenate(ivs_hes), all_mkt):>7.3f}"
    )

    print(f"\nATM skew ratio (model/market), |ln(K/F)|<={SKEW_LN_MONEYNESS}:")
    print(f"{'expiry':<12} {'market':>8} {'MSV':>6} {'CRL':>6} {'SSVI':>6} {'Heston':>6}")
    for d, a, b, c_, e in zip(data, ivs_plain, ivs_crl, ivs_ssvi, ivs_hes, strict=True):
        x = np.log(d.K / d.F)
        s_m = 100.0 * atm_skew(x, d.iv_mkt)
        r_p = 100.0 * atm_skew(x, a) / s_m
        r_c = 100.0 * atm_skew(x, b) / s_m
        r_s = 100.0 * atm_skew(x, c_) / s_m
        r_h = 100.0 * atm_skew(x, e) / s_m
        print(f"{d.expiry:<12} {s_m:>8.3f} {r_p:>6.2f} {r_c:>6.2f} {r_s:>6.2f} {r_h:>6.2f}")


def main() -> None:
    with open(shifted_spx.DEFAULT_CSV, newline="") as f:
        raw = list(csv.DictReader(f))
    run_source(f"yfinance CSV {shifted_spx.DEFAULT_CSV}", raw, float(raw[0]["spot"]))

    raw_db, spot_db = crl_spx.load_cboe_raw(crl_spx.CBOE_DB)
    run_source(
        f"Cboe SQLite {crl_spx.CBOE_DB} (trade_date={crl_spx.TRADE_DATE}, root={crl_spx.ROOT})",
        raw_db,
        spot_db,
    )


if __name__ == "__main__":
    main()
