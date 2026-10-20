"""Sobol multi-start distribution of optima + parameter SEs for the CRL fit to SPX.

Usage: uv run python experiments/multistart_se.py

For each available Cboe snapshot trade date (2026-10-16, then 2026-10-19, ... whatever
exists in the DB), fits CRL (rho <= 0) jointly over the six monthly expiries from 20
seeded Sobol starting points over the CRL box, with the same vega-weighted residuals as
crl_short.fit_crl_nonpos (local copy). Reports the distribution of final costs and
parameters, how many starts reach the best cost within 1e-6 relative, and clusters
distinct local optima by cost and parameters. At the best optimum reports parameter
standard errors from (J^T J)^-1 with sigma^2 = RSS/(n - p), residuals treated as
approximate implied-vol errors, plus the parameter correlation matrix. tau is computed
from each trade date (the fitters' ASOF-derived tau is not reused here).
"""

import os
import sqlite3
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.optimize import OptimizeResult, least_squares  # noqa: E402
from scipy.stats.qmc import Sobol  # noqa: E402
from shifted_spx import (  # noqa: E402
    MAX_LN_MONEYNESS,
    MAX_REL_SPREAD,
    ExpiryData,
    black_vega,
    implied_iv,
)

import msv  # noqa: E402

CBOE_DB = os.environ.get("CRL_CBOE_DB", str(Path.home() / "Data/article/cboe/spx.sqlite"))
ROOT = "SPX"
EXPIRIES = ("2026-11-20", "2026-12-18", "2027-01-15", "2027-03-19", "2027-06-17", "2027-09-17")
PARAMS = ("a0", "a1", "a2", "lam", "k", "rho")
N_STARTS = 20
SOBOL_SEED = 0  # Sobol(scramble=False) is a fixed deterministic sequence; seed kept for clarity
COST_TOL = 1e-6  # "reaches the best cost" relative tolerance
CLUSTER_TOL = 1e-3  # relative cost tolerance for distinct-local-optimum clustering


def available_dates(db: str) -> list[str]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT DISTINCT trade_date FROM snapshots ORDER BY trade_date")
        return [r[0] for r in rows]
    finally:
        con.close()


def load_raw(db: str, trade_date: str) -> tuple[list[dict[str, str]], float]:
    """Quotes for one trade date/root in load_expiry's dict format; spot from snapshots."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT spot FROM snapshots WHERE trade_date = ?", (trade_date,)
        ).fetchone()
        assert row is not None, f"no snapshot for {trade_date} in {db}"
        spot = float(row[0])
        rows = con.execute(
            "SELECT expiry, cp, strike, bid, ask FROM quotes WHERE trade_date = ? AND root = ?",
            (trade_date, ROOT),
        )
        raw = []
        for expiry, cp, strike, bid, ask in rows:
            bid, ask = float(bid), float(ask)
            if bid <= 0 or ask <= bid:
                continue
            raw.append(
                {
                    "expiry": expiry,
                    "type": cp,
                    "strike": str(strike),
                    "bid": str(bid),
                    "ask": str(ask),
                }
            )
    finally:
        con.close()
    return raw, spot


def load_expiry_asof(raw: list[dict[str, str]], expiry: str, spot: float, asof: date) -> ExpiryData:
    """shifted_spx.load_expiry with tau computed from the given trade-date asof."""
    rows = [r for r in raw if r["expiry"] == expiry]
    tau = (date.fromisoformat(expiry) - asof).days / 365.0
    quotes = {
        t: {float(r["strike"]): (float(r["bid"]), float(r["ask"])) for r in rows if r["type"] == t}
        for t in ("C", "P")
    }
    both = [k for k in sorted(set(quotes["C"]) & set(quotes["P"])) if abs(k / spot - 1.0) <= 0.10]
    K_pc = np.array(both)
    y = np.array(
        [
            (quotes["C"][k][0] + quotes["C"][k][1] - quotes["P"][k][0] - quotes["P"][k][1]) / 2
            for k in both
        ]
    )
    slope, intercept = (float(v) for v in np.polyfit(K_pc, y, 1))
    D = -slope
    F = intercept / D

    ks, mids, ivs = [], [], []
    for k in sorted(set(quotes["C"]) | set(quotes["P"])):
        if abs(np.log(k / F)) > MAX_LN_MONEYNESS:
            continue
        side = "P" if k < F else "C"
        if k not in quotes[side]:
            continue
        bid, ask = quotes[side][k]
        quoted_mid = (bid + ask) / 2
        if ask <= 0 or quoted_mid <= 0 or (ask - bid) / quoted_mid > MAX_REL_SPREAD:
            continue
        mid = quoted_mid + D * (F - k) if side == "P" else quoted_mid
        iv = implied_iv(F, k, tau, mid / D)
        if iv is None:
            continue
        ks.append(k)
        mids.append(mid)
        ivs.append(iv)
    return ExpiryData(expiry, tau, F, D, np.array(ks), np.array(mids), np.array(ivs))


def load_data(trade_date: str) -> tuple[list[ExpiryData], float]:
    raw, spot = load_raw(CBOE_DB, trade_date)
    asof = date.fromisoformat(trade_date)
    return [load_expiry_asof(raw, e, spot, asof) for e in EXPIRIES], spot


ResidFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def build_resid(data: list[ExpiryData]) -> ResidFn:
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

    return resid


def crl_box(data: list[ExpiryData]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    v_atm = [float(d.iv_mkt[np.argmin(np.abs(np.log(d.K / d.F)))]) ** 2 for d in data if d.K.size]
    v_long = float(max(v_atm))
    hi_a = max(5.0 * v_long, 0.05)  # a0, a1, a2 upper from the long-end ATM variance
    low = np.array([0.0, 0.0, 0.0, 0.05, 0.05, -0.99])
    high = np.array([hi_a, hi_a, hi_a, 5.0, 3.0, 0.0])
    return low, high


def sobol_starts(low: NDArray[np.float64], high: NDArray[np.float64]) -> NDArray[np.float64]:
    sampler = Sobol(d=6, scramble=False, seed=SOBOL_SEED)
    u = sampler.random(N_STARTS + 1)[1:]  # drop the all-zeros corner point
    return low + u * (high - low)


def fit_one(
    starts: NDArray[np.float64], low: NDArray[np.float64], high: NDArray[np.float64], resid: ResidFn
) -> list[OptimizeResult[Any]]:
    del low, high  # least_squares bounds are fixed constants here
    lb = [0.0, 0.0, 0.0, 1e-3, 1e-3, -0.99]
    ub = [np.inf, np.inf, np.inf, 50.0, 3.0, 0.0]
    out: list[OptimizeResult[Any]] = []
    for x0 in starts:
        res = least_squares(resid, x0, bounds=(lb, ub), xtol=1e-10, ftol=1e-10, max_nfev=2000)
        out.append(res)
    return out


def cluster_optima(results: list[OptimizeResult[Any]]) -> list[dict[str, Any]]:
    """Group starts by roughly equal best cost (relative tol); return clusters sorted by cost."""
    best = min(float(r.cost) for r in results)
    clusters: list[dict[str, Any]] = []
    for r in sorted(results, key=lambda r: r.cost):
        c = float(r.cost)
        label = "best" if abs(c - best) / best <= COST_TOL else "local"
        placed = False
        for cl in clusters:
            if float(cl["cost"]) > 0 and abs(c - cl["cost"]) / cl["cost"] <= CLUSTER_TOL:
                cl["n"] += 1
                if c < cl["cost"]:
                    cl["cost"], cl["x"] = c, np.array(r.x)
                placed = True
                break
        if not placed:
            clusters.append({"label": label, "n": 1, "cost": c, "x": np.array(r.x)})
    clusters.sort(key=lambda cl: cl["cost"])
    return clusters


def jacobian_central(resid: ResidFn, theta: NDArray[np.float64]) -> NDArray[np.float64]:
    n = int(resid(theta).size)
    p = theta.size
    eps = float(np.sqrt(np.finfo(float).eps))
    J = np.empty((n, p))
    for j in range(p):
        h = eps * max(1.0, abs(float(theta[j])))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        J[:, j] = (resid(tp) - resid(tm)) / (2.0 * h)
    return J


def std_errs(resid: ResidFn, theta: NDArray[np.float64], n_resid: int) -> dict[str, Any]:
    """SEs and correlation matrix at theta, sigma^2 = RSS/(n - p); residuals ~ IV errors."""
    p = theta.size
    J = jacobian_central(resid, theta)
    r = resid(theta)
    rss = float(np.sum(r * r))
    sigma2 = rss / (n_resid - p)
    cov = sigma2 * np.linalg.inv(J.T @ J)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / np.outer(se, se)
    cond = float(np.linalg.cond(J))
    return {
        "J": J,
        "rss": rss,
        "sigma2": sigma2,
        "resid_sd_volpts": float(100.0 * np.sqrt(sigma2)),
        "cov": cov,
        "se": se,
        "corr": corr,
        "cond": cond,
    }


def run_date(trade_date: str) -> None:
    data, spot = load_data(trade_date)
    resid = build_resid(data)
    n_resid = sum(d.K.size for d in data)
    low, high = crl_box(data)
    starts = sobol_starts(low, high)
    results = fit_one(starts, low, high, resid)

    print(f"\n{'=' * 78}")
    print(
        f"Cboe SQLite {CBOE_DB} trade_date={trade_date} root={ROOT}  "
        f"spot={spot:.2f}  n_resid={n_resid}"
    )
    print(
        f"Sobol box (a0,a1,a2 upper hi_a={high[0]:.4f}): "
        f"a0,a1,a2 in [0,{high[0]:.4f}], lam in [{low[3]},{high[3]}], "
        f"k in [{low[4]},{high[4]}], rho in [{low[5]},{high[5]}]"
    )

    best = min(results, key=lambda r: r.cost)
    costs = np.array([float(r.cost) for r in results])
    n_best = int(np.sum(costs <= float(best.cost) * (1.0 + COST_TOL)))

    print("\nPer-start results (20 Sobol starts):")
    print(f"{'start':>5} {'cost':>12} {'a0':>9} {'a1':>9} {'a2':>9} {'lam':>7} {'k':>7} {'rho':>7}")
    for i, r in enumerate(results):
        x = r.x
        tag = "*" if float(r.cost) <= float(best.cost) * (1.0 + COST_TOL) else " "
        print(
            f"{i:>5} {float(r.cost):>12.6e} {x[0]:>9.5f} {x[1]:>9.5f} {x[2]:>9.5f} "
            f"{x[3]:>7.4f} {x[4]:>7.4f} {x[5]:>7.3f} {tag}"
        )

    print(
        f"\nBest cost {float(best.cost):.6e}; {n_best} of {len(results)} starts within "
        f"{COST_TOL:.0e} relative of it."
    )
    print(
        f"Cost distribution: min {float(costs.min()):.3e}  "
        f"median {float(np.median(costs)):.3e}  max {float(costs.max()):.3e}"
    )

    clusters = cluster_optima(results)
    print(f"\nDistinct optima clustered by cost (relative tol {CLUSTER_TOL:.0e}):")
    hdr = f"{'cluster':>7} {'n':>4} {'cost':>12}"
    print(hdr + "".join(f"{p:>9}" if p in ("a0", "a1", "a2") else f"{p:>8}" for p in PARAMS[3:]))
    for j, cl in enumerate(clusters):
        x = cl["x"]
        print(
            f"{j:>7} {cl['n']:>4} {cl['cost']:>12.6e} {x[0]:>9.5f} {x[1]:>9.5f} "
            f"{x[2]:>9.5f} {x[3]:>7.4f} {x[4]:>7.4f} {x[5]:>7.3f}"
        )

    xb = best.x
    se = std_errs(resid, xb, n_resid)
    print(
        f"\nBest optimum: a0={xb[0]:.5f} a1={xb[1]:.5f} a2={xb[2]:.5f} lam={xb[3]:.4f} "
        f"k={xb[4]:.4f} rho={xb[5]:.3f}  cost={float(best.cost):.4f}  "
        f"resid sd={se['resid_sd_volpts']:.2f} vol pts  cond(J)={se['cond']:.3e}"
    )
    print(
        "Parameter standard errors, sigma^2 = RSS/(n-p) = "
        f"{se['sigma2']:.4e} (vol^2), residuals "
        "treated as approx. implied-vol errors:"
    )
    for i, name in enumerate(PARAMS):
        print(f"  {name:<4} {se['se'][i]:>10.3e}")
    print("Correlation matrix of (J^T J)^-1 * sigma^2:")
    print("        " + "".join(f"{p:>7}" for p in PARAMS))
    for i, name in enumerate(PARAMS):
        print(f"  {name:<4} " + "".join(f"{se['corr'][i, j]:>7.2f}" for j in range(6)))


def main() -> None:
    dates = available_dates(CBOE_DB)
    print(f"trade dates available: {dates}")
    for td in dates:
        run_date(td)


if __name__ == "__main__":
    main()
