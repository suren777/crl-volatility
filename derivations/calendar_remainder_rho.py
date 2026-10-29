"""Remainder for rho < 0 (epic 0016, task 05-remainder-rho).

Usage: uv run python derivations/calendar_remainder_rho.py   (writes calendar_remainder_rho.md)

Notation as in calendar_remainder.py: r = |rho|, lambda = L' > 0, sigma = s^2, mu = sqrt(2 lambda),
c0 = ln(2 lambda / u^2) + sigma/2, v = ln y - ln(2 lambda), so Z = (v + c0)/s and y = 2 lambda e^v.
In v the conditional law is nu proportional to exp(-V), with

    V(v) = (v + c0)^2 / (2 sigma) + v/2 + q^2 / (2 (1 - r^2)),   q = mu cosh(v/2) + r (v + c0)/s,

and R' = [ (lambda/2)(E e^v E e^{-v} - 1)
    - (lambda r / (s mu)) Cov(e^v, (v + c0) e^{-v/2}) ] / (1 - r^2).

This module evaluates that covariance form by quadrature (independent of Model), compares it with
Model.r_slope on a widened z-grid, and scans R' < 1 for sigma <= 2. It proves no inequality: the
scan is VERIFIED NUMERICALLY only. The analytic obstruction to Brascamp-Lieb (V'' fails where q < 0)
is measured here as the nu-mass of {q < 0} and {V'' < 1/sigma + lambda/(2(1 - r^2))}.
"""

from itertools import product

import numpy as np
from calendar_band import Model
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import norm

OUT = __file__.replace(".py", ".md")
US = (0.1, 1.0, 5.0)
KS = (0.05, 0.3, 1.0, 2.0, 2.5)
LAMS = (0.5, 2.0, 10.0, 50.0, 300.0, 3000.0)
RS = (0.05, 0.2, 0.5, 0.7, 0.9, 0.99)
V_GRID: NDArray[np.float64] = np.arange(-40.0, 40.0, 2e-4)


def potential(lam: float, u: float, k: float, r: float) -> tuple[float, float, NDArray[np.float64]]:
    """(sigma, c0, V on V_GRID)."""
    sigma = float(np.log1p(k * k))
    c0 = float(np.log(2 * lam / u**2) + sigma / 2)
    v = V_GRID
    mu = np.sqrt(2 * lam)
    q = mu * np.cosh(v / 2) + r * (v + c0) / np.sqrt(sigma)
    big_v = (v + c0) ** 2 / (2 * sigma) + v / 2 + q**2 / (2 * (1 - r * r))
    return sigma, c0, big_v


def quad_rprime(lam: float, u: float, k: float, r: float) -> dict[str, float]:
    """R' by the covariance form on V_GRID, plus the mass where log-concavity of nu can fail."""
    sigma, c0, big_v = potential(lam, u, k, r)
    v = V_GRID
    mu = np.sqrt(2 * lam)
    w = np.exp(-big_v - logsumexp(-big_v))
    e_p, e_m = float(np.sum(w * np.exp(v))), float(np.sum(w * np.exp(-v)))
    cov_h = float(np.sum(w * np.exp(v) * (v + c0) * np.exp(-v / 2)))
    cov_h -= e_p * float(np.sum(w * (v + c0) * np.exp(-v / 2)))
    rp = (lam / 2 * (e_p * e_m - 1) - lam * r / (np.sqrt(sigma) * mu) * cov_h) / (1 - r * r)
    q = mu * np.cosh(v / 2) + r * (v + c0) / np.sqrt(sigma)
    q1 = mu * np.sinh(v / 2) / 2 + r / np.sqrt(sigma)
    q2 = mu * np.cosh(v / 2) / 4
    vpp = 1 / sigma + (q1**2 + q * q2) / (1 - r * r)
    prec = 1 / sigma + lam / (2 * (1 - r * r))
    return {
        "rprime": float(rp),
        "mass_q_neg": float(np.sum(w[q < 0])),
        "mass_vpp_low": float(np.sum(w[vpp < prec])),
        "min_vpp_over_prec": float(np.min(vpp[w > 1e-12 * w.max()]) / prec),
    }


def wide_model(u: float, k: float, lam: float, rho: float) -> Model:
    """Model with its z-grid widened to the v-range [-25, 25]; the shipped grid can miss mass."""
    mod = Model(u, k, rho)
    c0 = np.log(2 * lam / u**2) + mod.s**2 / 2
    mod.z = np.arange((-25.0 + c0) / mod.s, (25.0 + c0) / mod.s, 1e-3)
    mod.lw = norm.logpdf(mod.z) + np.log(1e-3)
    mod.lm, mod.mm = mod.normaliser(u)
    return mod


def main() -> None:
    rows = []
    for u, k, r, lam in product(US, KS, RS, LAMS):
        q = quad_rprime(lam, u, k, r)
        mod = wide_model(u, k, lam, -r)
        truth = mod.r_slope(lam - mod.lm)
        sigma = mod.s**2
        mu = np.sqrt(2 * lam)
        rows.append(
            {
                "u": u,
                "k": k,
                "r": r,
                "lam": lam,
                "sigma": sigma,
                "id_err": abs(truth - q["rprime"]),
                "rprime": q["rprime"],
                "margin": 1 - q["rprime"],
                "asym": 2 * r / (np.sqrt(sigma) * mu),
                "mass_q_neg": q["mass_q_neg"],
                "mass_vpp_low": q["mass_vpp_low"],
                "min_vpp": q["min_vpp_over_prec"],
            }
        )
    covered = [row for row in rows if row["sigma"] <= 2.0]
    lines = [
        "# Remainder for rho < 0 (scan)",
        "",
        "Output of `uv run python derivations/calendar_remainder_rho.py`. R' is the covariance "
        "form on a v-quadrature (v in [-40, 40]); the asymptotic margin is 2r/(s mu), "
        "mu = sqrt(2 L') "
        "(calendar_laplace.md gives 1 - R' ~ sqrt(2) r/(s sqrt(L')), the same quantity).",
        f"Grid: u in {US}, k in {KS}, rho in {tuple(-r for r in RS)}, L' in {LAMS}; "
        f"{len(rows)} points, {len(covered)} with sigma <= 2.",
        "",
        f"- Identity: max |R'(Model.r_slope on wide grid) - R'(covariance form)| = "
        f"{max(row['id_err'] for row in rows):.2e}.",
        f"- R' < 1 on every point with sigma <= 2: {all(row['rprime'] < 1 for row in covered)}; "
        f"min of 1 - R' = {min(row['margin'] for row in covered):.3e}.",
        f"- Ratio (1 - R') / (2r/(s mu)) over sigma <= 2: min "
        f"{min(row['margin'] / row['asym'] for row in covered):.3f}, max "
        f"{max(row['margin'] / row['asym'] for row in covered):.3f}.",
        f"- nu-mass of {{q < 0}}: max {max(row['mass_q_neg'] for row in rows):.3e}.",
        f"- nu-mass of {{V'' < 1/sigma + lambda/(2(1-r^2))}}: max "
        f"{max(row['mass_vpp_low'] for row in rows):.3e}; min over support of V''/P_r: "
        f"{min(row['min_vpp'] for row in rows):.4f}.",
        "",
        "| u | k | rho | L' | R' | 1 - R' | 2r/(s mu) | mass q<0 | mass V''<P |",
        "|---|---|---|---|---|---|---|---|---|",
        *(
            f"| {row['u']} | {row['k']} | {-row['r']:g} | {row['lam']} | {row['rprime']:.6f} | "
            f"{row['margin']:.6f} | {row['asym']:.6f} | {row['mass_q_neg']:.2e} | "
            f"{row['mass_vpp_low']:.2e} |"
            for row in rows
            if row["u"] == 1.0
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
    print(f"Wrote {OUT}; {len(rows)} rows")


if __name__ == "__main__":
    main()
