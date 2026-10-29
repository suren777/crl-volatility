"""Rigorous bound on R'(L) at rho = 0 (epic 0016, task 03-remainder).

Usage: uv run python derivations/calendar_remainder.py   (writes calendar_remainder.md)

At rho = 0, with lambda = L' = L + ln m > 0, sigma = s^2 and P = 1/sigma + lambda/2, the law of
v = ln y - ln(2 lambda) is nu proportional to exp(-V), V'' = 1/sigma + (lambda/2) cosh v >= P,
and R' = (lambda/2)(E e^v E e^{-v} - 1). Brascamp-Lieb on the tilts gives E e^v E e^{-v} <=
exp(1/P), so R' <= bound := (lambda/2)(exp(1/P) - 1). The bound is u-free and, since
1/P = sigma a/(sigma + a) with a = 2/lambda, is < 1 whenever sigma <= 2.

The script checks the identity R' = (lambda/2)(E e^v E e^{-v} - 1) against Model.r_slope,
the Brascamp-Lieb inequality on each tilt by independent quadrature in v, and the bound
against R'. It also reports where the bound fails to be below 1 (sigma > 2).
"""

from itertools import product

import numpy as np
from calendar_band import Model
from scipy.special import logsumexp
from scipy.stats import norm

OUT = __file__.replace(".py", ".md")
US = (0.1, 1.0, 5.0)
KS = (0.05, 0.3, 1.0, 2.0, 2.5, 3.0, 6.0, 10.0)
LAMS = (0.5, 2.0, 10.0, 50.0, 300.0, 3000.0)
THETAS = (-1.0, -0.5, 0.0, 0.5, 1.0)
V_GRID = np.arange(-40.0, 40.0, 2e-4)


def precision(lam: float, sigma: float) -> float:
    return 1.0 / sigma + lam / 2.0


def bound_rho0(lam: float, sigma: float) -> float:
    return float(lam / 2.0 * np.expm1(1.0 / precision(lam, sigma)))


def l0_rho0(sigma: float) -> float | None:
    """Explicit threshold in L' at rho = 0: 0 (every L' > 0) when sigma <= 2, else none."""
    return 0.0 if sigma <= 2.0 else None


def tilted_law(lam: float, u: float, k: float, theta: float) -> tuple[float, float, float]:
    """(E_theta v, Var_theta v, E_theta[1/V'']) for nu_theta proportional to exp(-V + theta v)."""
    s2 = float(np.log1p(k * k))
    c0 = np.log(2 * lam / u**2) + s2 / 2
    v = V_GRID
    vpp = 1.0 / s2 + lam / 2 * np.cosh(v)
    big_v = (v + c0) ** 2 / (2 * s2) + v / 2 + lam / 2 * np.cosh(v)
    lw = -big_v + theta * v
    lw -= logsumexp(lw)
    w = np.exp(lw)
    mean = float(np.sum(w * v))
    var = float(np.sum(w * (v - mean) ** 2))
    return mean, var, float(np.sum(w / vpp))


def quad_rprime(lam: float, u: float, k: float) -> tuple[float, float]:
    """(R' from (lambda/2)(E e^v E e^{-v} - 1), sum_{theta=+-1} K(theta)) by quadrature in v."""
    s2 = float(np.log1p(k * k))
    c0 = np.log(2 * lam / u**2) + s2 / 2
    v = V_GRID
    big_v = (v + c0) ** 2 / (2 * s2) + v / 2 + lam / 2 * np.cosh(v)
    lw = -big_v
    lw -= logsumexp(lw)

    def log_mgf(theta: float) -> float:
        return float(logsumexp(lw + theta * v))

    k_sum = log_mgf(1.0) + log_mgf(-1.0)
    return lam / 2 * np.expm1(k_sum), k_sum


def wide_model(u: float, k: float, lam: float) -> Model:
    """Model with its z-grid widened to the v-range [-25, 25]; the shipped grid can miss mass."""
    mod = Model(u, k, 0.0)
    c0 = np.log(2 * lam / u**2) + mod.s**2 / 2
    mod.z = np.arange((-25.0 + c0) / mod.s, (25.0 + c0) / mod.s, 1e-3)
    mod.lw = norm.logpdf(mod.z) + np.log(1e-3)
    mod.lm, mod.mm = mod.normaliser(u)
    return mod


def main() -> None:
    rows = []
    for u, k, lam in product(US, KS, LAMS):
        mod = wide_model(u, k, lam)
        truth = mod.r_slope(lam - mod.lm)
        default = Model(u, k, 0.0).r_slope(lam - Model(u, k, 0.0).lm)
        quad, _ = quad_rprime(lam, u, k)
        tilts = [tilted_law(lam, u, k, th) for th in THETAS]
        rows.append(
            {
                "u": u,
                "k": k,
                "lam": lam,
                "sigma": mod.s**2,
                "truth": truth,
                "default": default,
                "quad": quad,
                "id_err": abs(truth - quad),
                "default_err": abs(default - quad),
                "bound": bound_rho0(lam, mod.s**2),
                "bl_gap": max(var - (ev) for _, var, ev in tilts),
                "bl_tight": max(ev - 1.0 / precision(lam, mod.s**2) for _, _, ev in tilts),
            }
        )
    covered = [r for r in rows if l0_rho0(r["sigma"]) is not None]
    lines = [
        "# Rigorous bound on R'(L) at rho = 0",
        "",
        "Output of `uv run python derivations/calendar_remainder.py`. Bound: "
        "R' <= (lambda/2)(exp(1/P) - 1), P = 1/s^2 + lambda/2, lambda = L + ln m. "
        f"Grid: u in {US}, k in {KS}, lambda in {LAMS}, tilts theta in {THETAS}. "
        "R' (quad) is the quadrature of (lambda/2)(E e^v E e^-v - 1) on v in [-40, 40]; "
        "R' (wide) is Model.r_slope on a z-grid widened to v in [-25, 25]; R' (default) is "
        "Model.r_slope as shipped.",
        "",
        f"- Identity: max |R'(wide) - R'(quad)| = {max(r['id_err'] for r in rows):.2e}.",
        f"- Shipped Model grid: max |R'(default) - R'(quad)| = "
        f"{max(r['default_err'] for r in rows):.3g}; rows off by more than 1e-6: "
        f"{sum(r['default_err'] > 1e-6 for r in rows)} of {len(rows)} (small k, large lambda: "
        "the z-grid starts at -30 and the conditional law sits at z near -c0/s).",
        f"- Brascamp-Lieb on every tilt: max(Var_theta(v) - E_theta[1/V'']) = "
        f"{max(r['bl_gap'] for r in rows):.2e} (must be <= 0 up to quadrature error); "
        f"max(E_theta[1/V''] - 1/P) = {max(r['bl_tight'] for r in rows):.2e} (must be <= 0).",
        f"- Bound >= R'(quad) on every grid row: "
        f"{all(r['bound'] >= r['quad'] - 1e-9 for r in rows)}.",
        f"- Bound < 1 on every row with sigma <= 2: "
        f"{all(r['bound'] < 1 for r in covered)} ({len(covered)} rows).",
        "",
        "Threshold L0 at rho = 0 (sigma = s^2): L0 = 0 (every L' > 0) for sigma <= 2, i.e. "
        f"k <= sqrt(e^2 - 1) = {np.sqrt(np.e**2 - 1):.4f}; no threshold from this bound "
        "for larger k.",
        "",
        "| k | s^2 | L0 | max R' (quad) | max bound | bound - 1 at lambda = 3000 |",
        "|---|---|---|---|---|---|",
    ]
    for k in KS:
        sub = [r for r in rows if r["k"] == k]
        sig = sub[0]["sigma"]
        l0 = l0_rho0(sig)
        big = next(r for r in sub if r["lam"] == max(LAMS))
        l0_txt = "none" if l0 is None else f"{l0:g}"
        lines.append(
            f"| {k} | {sig:.4f} | {l0_txt} | {max(r['quad'] for r in sub):.4f} | "
            f"{max(r['bound'] for r in sub):.4f} | {big['bound'] - 1.0:+.4f} |"
        )
    lines += [
        "",
        "Full table at u = 1.0.",
        "",
        "| k | lambda | R' (quad) | R' (wide) | R' (default) | bound | 1/P |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {r['k']} | {r['lam']} | {r['quad']:.6f} | {r['truth']:.6f} | "
            f"{r['default']:.6f} | {r['bound']:.6f} | {1.0 / precision(r['lam'], r['sigma']):.6f} |"
            for r in rows
            if r["u"] == 1.0
        ),
        "",
    ]
    try:
        with open(OUT) as f:
            old = f.read()
    except FileNotFoundError:
        old = ""
    head_md = old.split("<!-- scan -->")[0] if "<!-- scan -->" in old else ""
    with open(OUT, "w") as f:
        f.write(head_md + "<!-- scan -->\n" + "\n".join(lines))
    print(
        f"Wrote {OUT}; {len(rows)} rows; bound >= quad: "
        f"{all(r['bound'] >= r['quad'] - 1e-9 for r in rows)}"
    )


if __name__ == "__main__":
    main()
