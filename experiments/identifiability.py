"""Identifiability of the CRL parameters (a0, a1, a2, lam, k, rho) on SPX.

Usage: uv run python experiments/identifiability.py

Fits CRL (rho <= 0) jointly over the six SPX monthlies (Cboe SQLite and the
yfinance CSV), then, at each optimum, reports the Jacobian of the
vega-weighted residuals w.r.t. the six parameters (central differences):
singular values / condition number of J, eigenvectors of J^T J for the two
smallest eigenvalues, and the correlation matrix of (J^T J)^-1 scaled by the
fitted residual variance. Then profiles lam: fix it on a grid (0.1..5) and
refit the other five, reporting the cost profile and how (a0, a1, a2) move.
"""

import csv
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import shifted_spx  # noqa: E402
from crl_short import fit_crl_nonpos, load_raw  # noqa: E402
from crl_spx import RHO_STARTS  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.optimize import OptimizeResult, least_squares  # noqa: E402
from shifted_spx import EXPIRIES, ExpiryData, black_vega, load_expiry  # noqa: E402

import msv  # noqa: E402

PARAMS = ("a0", "a1", "a2", "lam", "k", "rho")
LAM_GRID = (0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0)
LAM_K_STARTS = (0.3, 0.15, 0.3, 0.5, 0.5)


def concat(data: list[ExpiryData]) -> tuple[NDArray[np.float64], ...]:
    """Vega-weighted-residual arrays, matching fit_crl_nonpos's objective."""
    F = np.concatenate([np.full(d.K.size, d.F) for d in data])
    D = np.concatenate([np.full(d.K.size, d.D) for d in data])
    tau = np.concatenate([np.full(d.K.size, d.tau) for d in data])
    K = np.concatenate([d.K for d in data])
    mid = np.concatenate([d.mid for d in data])
    vega = np.array(
        [black_vega(d.F, k, d.tau, iv) for d in data for k, iv in zip(d.K, d.iv_mkt, strict=True)]
    )
    denom = D * vega
    return F, D, tau, K, mid, denom


ResidFunc = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def make_resid(data: list[ExpiryData], fixed_lam: float | None = None) -> ResidFunc:
    F, D, tau, K, mid, denom = concat(data)

    def resid(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        if fixed_lam is None:
            a0, a1, a2, lam, k, rho = (float(v) for v in theta)
        else:
            a0, a1, a2, k, rho = (float(v) for v in theta)
            lam = float(fixed_lam)
        px = msv.crl_call(F, K, D, tau, a0, a1, a2, lam, k, rho)
        return np.asarray((px - mid) / denom, dtype=float)

    return resid


def jac_central(f: ResidFunc, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Finite-difference Jacobian of residual vector f at x (central, h~1e-5·scale)."""
    f0 = f(x)
    J = np.empty((f0.size, x.size))
    for i in range(x.size):
        h = 1e-5 * max(1.0, abs(float(x[i])))
        xp = np.array(x, dtype=float)
        xm = np.array(x, dtype=float)
        xp[i] += h
        xm[i] -= h
        J[:, i] = (f(xp) - f(xm)) / (2.0 * h)
    return J


def ident(fit: OptimizeResult, data: list[ExpiryData], label: str) -> None:
    x = fit.x
    resid = make_resid(data)
    J = jac_central(resid, x)
    N = J.shape[0]
    p = x.size
    r = resid(x)
    # residuals are approximately IV errors in vol units; J^T J is inverse covariance per
    # unit residual variance. Multiply by fitted residual variance for absolute SEs.
    var = float(np.dot(r, r) / (N - p))  # residual variance (units: vol^2)
    JtJ = J.T @ J
    cov = np.linalg.inv(JtJ) * var
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    s = np.linalg.svd(J, compute_uv=False)
    cond = float(s[0] / s[-1])
    _U, _s, Vt = np.linalg.svd(J, full_matrices=False)

    print(f"\n{'=' * 70}\n{label}\n" + "=" * 70)
    print("Fitted parameters:")
    print(
        f"  a0={x[0]:.5f} a1={x[1]:.5f} a2={x[2]:.5f} lam={x[3]:.4f} "
        f"k={x[4]:.4f} rho={x[5]:.3f}  cost={fit.cost:.4f}"
    )
    print(
        f"  n_resid={N}, n_params={p}, residual variance (vol^2) = {var:.3e}, "
        f"residual sd (vol pts) = {100 * np.sqrt(var):.2f}"
    )
    print(
        "\nJacobian J of vega-weighted residuals w.r.t. "
        "(a0,a1,a2,lam,k,rho): central diff h~1e-5·scale"
    )
    print("Singular values of J:")
    for i, si in enumerate(s):
        print(f"  s{i}={si:.3e}")
    print(f"Condition number (s_max/s_min) = {cond:.3e}")
    print("\nEigenvectors of J^T J for the two smallest eigenvalues (V columns, rows = params):")
    print(f"  {'param':<5} {'v(p-1)':>12} {'v(p-2)':>12}")
    for i, name in enumerate(PARAMS):
        print(f"  {name:<5} {Vt[-1, i]:>12.4f} {Vt[-2, i]:>12.4f}")
    print("\nApprox. parameter std err (sqrt(diag cov), residual-variance scaled):")
    print(f"  {'param':<5} {'s.e.':>12}")
    for name, di in zip(PARAMS, d, strict=True):
        print(f"  {name:<5} {di:>12.4e}")
    print("\nCorrelation matrix of (J^T J)^-1 * residual_variance:")
    print("       " + " ".join(f"{n:>8}" for n in PARAMS))
    for i, name in enumerate(PARAMS):
        print(f"  {name:<5} " + " ".join(f"{corr[i, j]:>8.2f}" for j in range(p)))


def profile_lam(data: list[ExpiryData], label: str) -> None:
    v_atm = [float(d.iv_mkt[np.argmin(np.abs(np.log(d.K / d.F)))]) ** 2 for d in data if d.K.size]
    v_short, v_long = min(v_atm), max(v_atm)
    base = [max(v_short - v_long, 0.0), max(v_long - v_short, 0.0) * 0.5, v_long]
    lb = [0.0, 0.0, 0.0, 1e-3, -0.99]
    ub = [np.inf, np.inf, np.inf, 3.0, 0.0]

    best_for_lam: dict[float, tuple[float, NDArray[np.float64]]] = {}
    for lam in LAM_GRID:
        resid = make_resid(data, fixed_lam=lam)
        best: OptimizeResult | None = None
        for k0 in LAM_K_STARTS:
            for rho0 in RHO_STARTS:
                res = least_squares(
                    resid, np.array(base + [k0, rho0]), bounds=(lb, ub), xtol=1e-10, ftol=1e-10
                )
                if best is None or res.cost < best.cost:
                    best = res
        assert best is not None
        best_for_lam[lam] = (best.cost, best.x)
    free = min(c for c, _ in best_for_lam.values())
    print(f"\nlam profile, {label} (refit a0,a1,a2,k,rho at fixed lam; cost = 0.5*sum r^2):")
    hdr = (
        f"{'lam':>5} {'cost':>12} {'cost/free':>10} {'a0':>10} "
        f"{'a1':>10} {'a2':>10} {'k':>8} {'rho':>7}"
    )
    print(hdr)
    for lam in LAM_GRID:
        c, x = best_for_lam[lam]
        print(
            f"{lam:>5.1f} {c:>12.4e} {c / free:>10.3f} {x[0]:>10.5f} {x[1]:>10.5f} "
            f"{x[2]:>10.5f} {x[3]:>8.4f} {x[4]:>7.3f}"
        )


def main() -> None:
    raw_spx, spot = load_raw("SPX")
    cboe_data = [load_expiry(raw_spx, e, spot) for e in EXPIRIES]
    with open(shifted_spx.DEFAULT_CSV, newline="") as f:
        raw = list(csv.DictReader(f))
    yf_spot = float(raw[0]["spot"])
    yf_data = [load_expiry(raw, e, yf_spot) for e in EXPIRIES]

    cboe_fit, _t_c = fit_crl_nonpos(cboe_data)
    yf_fit, _t_y = fit_crl_nonpos(yf_data)
    print(f"Cboe fit time {_t_c:.1f}s, yfinance fit time {_t_y:.1f}s")
    ident(cboe_fit, cboe_data, "Cboe SQLite root=SPX (2026-10-16)")
    ident(yf_fit, yf_data, "yfinance CSV")
    profile_lam(cboe_data, "Cboe SQLite")
    profile_lam(yf_data, "yfinance CSV")


if __name__ == "__main__":
    main()
