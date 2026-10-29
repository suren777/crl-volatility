"""Ratio form dC/du = T1 (1 + Q) on the band for rho <= 0 (epic 0016, task 08-ratio).

Usage: uv run python derivations/calendar_ratio.py [-j N]   (writes calendar_ratio.md)

F = D = 1, u = sqrt(tau Q), s^2 = ln(1+k^2), zeta = e^{sZ - s^2/2}, y = u^2 zeta, x = ln K,
M = e^{rho sqrt(y) Z - rho^2 y/2}, m = E[M], A = M/m (a probability weight on Z), h = d ln M/du
= rho sqrt(zeta) Z - rho^2 u zeta, so E_A[h] = m'/m. With w = (1-rho^2) y, d1 = (ln A - x +
w/2)/sqrt(w), d2 = d1 - sqrt(w), the Black formula conditional on Z gives

    C = E_A[N(d1)] - e^x E[N(d2)],
    dC/du = E_A[phi(d1) sigma] + E_A[N(d1) (h - m'/m)] = T1 + Cov_A(N(d1), h),
    T1 = E_A[phi(d1) sigma], sigma = sqrt((1-rho^2) zeta),  Q = Cov_A(N(d1), h) / T1.

Every sum is taken on a Z grid in log space; N(d1) is scaled by its maximum so that deep
out-of-the-money strikes do not underflow. Band thresholds: l1 = -ln m + u m'/m and l2' from
calendar_band.md (Theorem B'). Exploration and verification, not a proof; the proof status
is in calendar_ratio.md.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from calendar_band import KS, RHOS, STRESS, US
from numpy.typing import NDArray
from scipy.special import log_ndtr, logsumexp
from scipy.stats import norm

OUT = __file__.replace(".py", ".md")
STEP = 1e-3
RESOLVE = 1e-8
N_X = 25
LADDER_U = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
LADDER_K = (0.05, 1.0, 3.0, 6.0)
LADDER_RHO = (-0.01, -0.2, -0.5, -0.7, -0.9, -0.99)
Arr = NDArray[np.float64]


def _log_terms(u: float, k: float, rho: float, z: Arr) -> tuple[Arr, Arr, Arr, float]:
    """(log(phi dz) + log M, h, zeta, s) with M unnormalised."""
    s = float(np.sqrt(np.log1p(k * k)))
    zeta = np.exp(s * z - s * s / 2)
    y = u * u * zeta
    lm_z = rho * np.sqrt(y) * z - rho * rho * y / 2
    return (
        norm.logpdf(z) + np.log(STEP) + lm_z,
        rho * np.sqrt(zeta) * z - rho * rho * u * zeta,
        zeta,
        s,
    )


def z_grid(u: float, k: float, rho: float) -> Arr:
    """Z grid on which the weight A is negligible (< e^-60 of its maximum) at both ends."""
    s = float(np.sqrt(np.log1p(k * k)))
    lo, hi = -40.0, max(14.0, (np.log(50.0 / u**2) + s * s / 2) / s + 12.0)
    for _ in range(60):
        z = np.arange(lo, hi, STEP)
        g, *_ = _log_terms(u, k, rho, z)
        top = g.max()
        left, right = g[0] < top - 60, g[-1] < top - 60
        if left and right:
            return z
        if not left:
            lo *= 2.0
        if not right:
            hi += 10.0
    raise RuntimeError(f"Z grid did not close at u={u}, k={k}, rho={rho}")


def _weights(u: float, k: float, rho: float, z: Arr) -> tuple[Arr, Arr, Arr, float, float, Arr]:
    """(log A, h, zeta, s, log m, log F) with A = phi M/m normalised, F = M/m."""
    g, h, zeta, s = _log_terms(u, k, rho, z)
    lm = float(logsumexp(g))
    return g - lm, h, zeta, s, lm, g - lm - norm.logpdf(z) - np.log(STEP)


def thresholds(u: float, k: float, rho: float, z: Arr) -> tuple[float, float, float]:
    """(l1, l2', m'/m) with l1 = -ln m + u m'/m and l2' from calendar_band.md."""
    log_a, h, _, s, lm, _ = _weights(u, k, rho, z)
    mbar = float(np.sum(np.exp(log_a) * h))
    l1 = -lm + u * mbar

    def g(lp: float) -> float:
        return float(np.log(2 * lp / u**2) + s * s / 2 + s * abs(rho) * np.sqrt(2 * lp))

    lo, hi = 1e-300, u * u * np.exp(-s * s / 2) / 2
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
    return l1, float(hi) - lm, mbar


def evaluate(
    u: float, k: float, rho: float, x: float, z: Arr
) -> tuple[float, float, float, float, float]:
    """(dC/du, T1, Q, Z-centre of the negative part of Q, its share of the total |integrand|).

    Q is NaN when the covariance is below 1e-8 of the sum of |integrand| terms: N(d1) is then
    numerically 1 on the mass and the centred sum cannot be resolved in float64.
    """
    log_a, h, zeta, s, _, log_f = _weights(u, k, rho, z)
    a = np.exp(log_a)
    a /= a.sum()
    w = (1 - rho * rho) * u * u * zeta
    sw = np.sqrt(w)
    d1 = (log_f - x + w / 2) / sw
    log_n = log_ndtr(d1)
    log_t1_terms = log_a + norm.logpdf(d1) + 0.5 * np.log((1 - rho * rho) * zeta)
    log_t1 = float(logsumexp(log_t1_terms))
    lmax = float(np.max(log_a + log_n))
    an = np.exp(log_a + log_n - lmax)
    hbar = float(np.sum(a * h))
    integrand = an * (h - hbar)
    scaled = float(np.sum(integrand))
    neg = integrand < 0
    neg_mass = float(np.sum(np.abs(integrand[neg])))
    total = float(np.sum(np.abs(integrand)))
    resolved = abs(scaled) >= RESOLVE * total
    with np.errstate(over="ignore"):
        t1 = float(np.exp(log_t1))
        q = scaled * float(np.exp(lmax - log_t1)) if resolved else np.nan
        cov = scaled * float(np.exp(lmax))
    z_neg = float(np.sum(np.abs(integrand[neg]) * z[neg]) / neg_mass) if neg_mass > 0 else np.nan
    share = neg_mass / total if total > 0 else 0.0
    return t1 + cov, t1, q, z_neg, share


def call_price(u: float, k: float, rho: float, x: float, z: Arr) -> float:
    log_a, _, zeta, _, _, log_f = _weights(u, k, rho, z)
    a = np.exp(log_a)
    a /= a.sum()
    w = (1 - rho * rho) * u * u * zeta
    sw = np.sqrt(w)
    d1 = (log_f - x + w / 2) / sw
    d2 = d1 - sw
    p_d2 = float(np.sum(norm.pdf(z) * STEP * norm.cdf(d2)))
    return float(np.sum(a * norm.cdf(d1)) - np.exp(x) * p_d2)


def band_point(p: tuple[float, float, float]) -> dict[str, float]:
    u, k, rho = p
    z = z_grid(u, k, rho)
    l1, l2, mbar = thresholds(u, k, rho, z)
    xs = np.linspace(max(l1, 0.0), l2, N_X)
    rows = [evaluate(u, k, rho, float(x), z) for x in xs]
    q_arr = np.array([r[2] for r in rows])
    resolved = ~np.isnan(q_arr)
    out = {
        "u": u,
        "k": k,
        "rho": rho,
        "l1": l1,
        "l2": l2,
        "mbar": mbar,
        "n_res": float(resolved.sum()),
        "min_one_plus_q": np.nan,
        "x_min": np.nan,
        "z_neg": np.nan,
        "share_neg": np.nan,
        "max_abs_q": np.nan,
    }
    if resolved.any():
        i = int(np.nanargmin(np.where(resolved, 1 + q_arr, np.inf)))
        out.update(
            min_one_plus_q=float(1 + q_arr[i]),
            x_min=float(xs[i]),
            z_neg=rows[i][3],
            share_neg=rows[i][4],
            max_abs_q=float(np.nanmax(np.abs(q_arr))),
        )
    return out


def fd_check(p: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """(analytic dC/du, central difference in u, relative error) at moderate x."""
    u, k, rho, x = p
    h = 1e-5 * u
    z = z_grid(u, k, rho)
    d_analytic = evaluate(u, k, rho, x, z)[0]
    fd = (call_price(u + h, k, rho, x, z) - call_price(u - h, k, rho, x, z)) / (2 * h)
    return d_analytic, fd, abs(d_analytic - fd) / max(abs(fd), 1e-300)


def step_check(p: tuple[float, float, float]) -> float:
    """Max |1+Q| change when the Z step is halved, on the band (quadrature convergence)."""
    global STEP
    u, k, rho = p
    z = z_grid(u, k, rho)
    l1, l2, _ = thresholds(u, k, rho, z)
    xs = np.linspace(max(l1, 0.0), l2, 5)
    coarse = [1 + evaluate(u, k, rho, float(x), z)[2] for x in xs]
    fine_step = STEP / 2
    old, STEP = STEP, fine_step
    try:
        zf = z_grid(u, k, rho)
        fine = [1 + evaluate(u, k, rho, float(x), zf)[2] for x in xs]
    finally:
        STEP = old
    return float(np.max(np.abs(np.array(coarse) - np.array(fine))))


def _nanmin(rows: list[dict[str, float]], key: str) -> str:
    vals = np.array([r[key] for r in rows], dtype=float)
    return "n/a" if np.all(np.isnan(vals)) else f"{np.nanmin(vals):.4f}"


def _nanmed(rows: list[dict[str, float]], key: str) -> str:
    vals = np.array([r[key] for r in rows], dtype=float)
    return "n/a" if np.all(np.isnan(vals)) else f"{np.nanmedian(vals):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", type=int, default=8, help="worker processes")
    args = parser.parse_args()
    scan = list(product(US, KS, RHOS))
    stress = list(product(*STRESS))
    ladder = list(product(LADDER_U, LADDER_K, LADDER_RHO))
    fd_pts = [(1.0, 0.3, -0.5, 0.2), (2.0, 1.0, -0.2, -0.5), (5.0, 3.0, -0.9, 1.0)]
    with ProcessPoolExecutor(args.j) as pool:
        scan_rows = list(pool.map(band_point, scan))
        stress_rows = list(pool.map(band_point, stress))
        ladder_rows = list(pool.map(band_point, ladder))
        fd_rows = list(pool.map(fd_check, fd_pts))
        step_err = max(
            pool.map(step_check, [(1.0, 1.0, -0.5), (5.0, 3.0, -0.2), (10.0, 0.3, -0.9)])
        )
    all_rows = scan_rows + stress_rows + ladder_rows
    rho0 = [r for r in all_rows if r["rho"] == 0]
    unres = [r for r in ladder_rows if r["n_res"] == 0]
    lines = [
        "# Ratio form of the band derivative for rho <= 0",
        "",
        "Output of `uv run python derivations/calendar_ratio.py`; formulas in the module "
        f"docstring. x runs over {N_X} points of [max(l1, 0), l2'] at each parameter point; "
        f"Z step {STEP:g}, float64. Q is reported only where the covariance is at least "
        f"{RESOLVE:g} of the sum of absolute integrand terms (else it is not resolved in "
        "float64; see the last section). Columns: min(1 + Q) over the resolved band points, "
        "the x where it is attained, the Z-centre of the negative part of the covariance "
        "integrand there, and the share of the absolute integrand that is negative; n_res "
        "counts resolved points out of N_X.",
        "",
        "## Finite differences of C in u (VERIFIED NUMERICALLY)",
        "",
        "| u | k | rho | x | dC/du (analytic) | central difference | relative error |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {p[0]} | {p[1]} | {p[2]} | {p[3]} | {d:.10g} | {f:.10g} | {e:.2e} |"
            for p, (d, f, e) in zip(fd_pts, fd_rows, strict=True)
        ),
        "",
        f"Quadrature: max change in (1 + Q) when the Z step is halved, over three points at "
        f"five band strikes: {step_err:.2e}.",
        "",
        "## Minimum of 1 + Q on the band (resolved points)",
        "",
        f"- Scan grid (u, k, rho), {len(scan_rows)} points: min(1+Q) = "
        f"{_nanmin(scan_rows, 'min_one_plus_q')}.",
        f"- Stress grid, {len(stress_rows)} points: min(1+Q) = "
        f"{_nanmin(stress_rows, 'min_one_plus_q')}.",
        f"- Ladder (u up to {LADDER_U[-1]:g}, k {LADDER_K}, rho {LADDER_RHO}), "
        f"{len(ladder_rows)} points: min(1+Q) = {_nanmin(ladder_rows, 'min_one_plus_q')}; "
        f"points with no resolved band strike: {len(unres)}.",
        f"- The minimum over x sits at the right endpoint l2' on "
        f"{sum(abs(r['x_min'] - r['l2']) <= 1e-9 * max(1.0, r['l2']) for r in scan_rows)} of "
        f"{len(scan_rows)} scan points.",
        f"- rho = 0 (exactly 1): max |Q| = "
        f"{max((r['max_abs_q'] for r in rho0 if not np.isnan(r['max_abs_q'])), default=0.0):.2e}.",
        "",
        "Ladder: minimum over k and rho of 1+Q on the resolved band, by u:",
        "",
        "| u | min(1+Q) | resolved x (of " + f"{len(LADDER_K) * len(LADDER_RHO) * N_X}) |",
        "|---|---|---|",
        *(
            f"| {u:g} | "
            f"{_nanmin([r for r in ladder_rows if r['u'] == u], 'min_one_plus_q')} | "
            f"{int(sum(r['n_res'] for r in ladder_rows if r['u'] == u))} |"
            for u in LADDER_U
        ),
        "",
        "Where the negative part of Q sits (at the minimising x of each ladder point with a "
        "resolved band): median Z-centre "
        f"{_nanmed(ladder_rows, 'z_neg')}, median share of the absolute integrand that is "
        f"negative {_nanmed(ladder_rows, 'share_neg')}.",
        "",
        "## Per-point table (scan grid)",
        "",
        "| u | k | rho | l1 | l2' | n_res | x at min | min(1+Q) | Z-centre (neg) | neg share | "
        "m'/m |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *(
            f"| {r['u']} | {r['k']} | {r['rho']} | {r['l1']:.4f} | {r['l2']:.4f} | "
            f"{int(r['n_res'])} | {r['x_min']:.4f} | {r['min_one_plus_q']:.4f} | "
            f"{r['z_neg']:.3f} | {r['share_neg']:.3f} | {r['mbar']:.4f} |"
            for r in scan_rows
        ),
        "",
        "## Stress grid",
        "",
        "| u | k | rho | n_res | min(1+Q) | x at min |",
        "|---|---|---|---|---|---|",
        *(
            f"| {r['u']} | {r['k']} | {r['rho']:g} | {int(r['n_res'])} | "
            f"{r['min_one_plus_q']:.4f} | {r['x_min']:.4f} |"
            for r in stress_rows
        ),
        "",
        "## Ladder",
        "",
        "| u | k | rho | n_res | min(1+Q) | x at min | max abs Q |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {r['u']:g} | {r['k']} | {r['rho']:g} | {int(r['n_res'])} | "
            f"{r['min_one_plus_q']:.4f} | {r['x_min']:.4g} | {r['max_abs_q']:.4f} |"
            for r in ladder_rows
        ),
        "",
    ]
    try:
        with open(OUT) as f:
            old = f.read()
    except FileNotFoundError:
        old = ""
    head = old.split("<!-- scan -->")[0] if "<!-- scan -->" in old else ""
    with open(OUT, "w") as f:
        f.write(head + "<!-- scan -->\n" + "\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
