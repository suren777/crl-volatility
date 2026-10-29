"""Far right wing of the rho < 0 calendar conjecture (epic 0016, route 3).

Usage: uv run python derivations/calendar_tail.py [-j N]   (writes calendar_tail.md)

F = D = 1, u = sqrt(tau Q(tau)), s^2 = ln(1+k^2), zeta = exp(s Z - s^2/2), y = u^2 zeta,
L = ln K, M = exp(rho sqrt(y) Z - rho^2 y/2), m = E[M], A = M/m, w = (1-rho^2) y,
h = d ln M / du = rho sqrt(zeta) Z - rho^2 u zeta, so m'/m = E[A h]. Differentiating
C = E_Z[Black(A, K, w)] under the integral,

    dC/du = E[A phi(d1) sqrt((1-rho^2) zeta)] + E[A N(d1) h] - E[A N(d1)] E[A h],

and with P = P(S > K) = E_Z[N(d2)], d2 = d1 - sqrt(w),

    dP/du = E[phi(d2) ((h - m'/m - (1-rho^2) u zeta) / sqrt(w) - d2/u)].

Every expectation is a log-space sum on a dense Z grid around the saddle, split by sign, so
nothing underflows. Laplace prediction: with B = L + ln m + y/2 - rho sqrt(y) Z and Z* the
maximiser of Psi = -Z^2/2 - B^2 / (2 (1-rho^2) y), the envelope theorem gives

    d ln P / du ~ G = (2/(s u)) (Z* - rho B* / ((1-rho^2) sqrt(y*))) - B* (m'/m) / ((1-rho^2) y*),

whose first term is positive when rho <= 0, Z* > 0 and B* > 0. Exploration, not a test.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.special import log_ndtr, logsumexp
from scipy.stats import norm

OUT = __file__.replace(".py", ".md")
RHOS = (-0.05, -0.49, -0.9, -0.99)
KS = (0.1, 2.2, 3.0)
US = (0.3, 2.0)
XS = (0.5, 2.0, 10.0, 40.0, 200.0, 1000.0)
STEP = 2e-4
Arr = NDArray[np.float64]


def signed_log_sum(logs: Arr, b: Arr) -> tuple[float, float]:
    """(log |sum b e^logs|, sign)."""
    val, sign = logsumexp(logs, b=b, return_sign=True)
    return float(val), float(sign)


def point(p: tuple[float, float, float, float]) -> dict[str, float]:
    rho, k, u, x = p
    s = float(np.sqrt(np.log1p(k * k)))
    z_hi = (np.log(2.0 * max(x, 1.0) / u**2) + s * s / 2) / s + 20.0
    z = np.arange(-40.0, z_hi, STEP)
    lw = norm.logpdf(z) + np.log(STEP)
    zeta = np.exp(s * z - s * s / 2)
    y = u * u * zeta
    log_m_z = rho * np.sqrt(y) * z - rho * rho * y / 2
    log_m = float(logsumexp(lw + log_m_z))
    log_a = log_m_z - log_m
    h = rho * np.sqrt(zeta) * z - rho * rho * u * zeta
    mm = float(np.sum(np.exp(lw + log_a) * h))
    w = (1 - rho * rho) * y
    sw = np.sqrt(w)
    d1 = (log_a - x + w / 2) / sw
    d2 = d1 - sw

    base = lw + log_a
    log_t1 = float(logsumexp(base + norm.logpdf(d1) + 0.5 * np.log((1 - rho * rho) * zeta)))
    log_pstar = float(logsumexp(base + log_ndtr(d1)))
    log_ah, sign_ah = signed_log_sum(base + log_ndtr(d1), h)
    terms = np.array([0.0, log_ah - log_t1, log_pstar - log_t1])
    coefs = np.array([1.0, sign_ah, -mm])
    log_tot, sign_tot = signed_log_sum(terms, coefs)

    log_p = float(logsumexp(lw + log_ndtr(d2)))
    dd2 = (h - mm - (1 - rho * rho) * u * zeta) / sw - d2 / u
    log_dp, sign_dp = signed_log_sum(lw + norm.logpdf(d2), dd2)
    dlnp = sign_dp * float(np.exp(log_dp - log_p))

    b = x + log_m + y / 2 - rho * np.sqrt(y) * z
    i = int(np.argmax(-z * z / 2 - b * b / (2 * (1 - rho * rho) * y)))
    zs, bs, ys = float(z[i]), float(b[i]), float(y[i])
    g_main = 2 / (s * u) * (zs - rho * bs / ((1 - rho * rho) * np.sqrt(ys)))
    g_m = -bs * mm / ((1 - rho * rho) * ys)
    return {
        "rho": rho,
        "k": k,
        "u": u,
        "x": x,
        "ratio_c": sign_tot * float(np.exp(log_tot)),
        "log_t1": log_t1,
        "dlnp": dlnp,
        "g": g_main + g_m,
        "mm": mm,
        "zs": zs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", type=int, default=4, help="worker processes")
    args = parser.parse_args()
    grid = list(product(RHOS, KS, US, XS))
    with ProcessPoolExecutor(args.j) as pool:
        rows = list(pool.map(point, grid))
    neg = [r for r in rows if r["ratio_c"] <= 0]
    far = [r for r in rows if r["x"] >= 10]
    lines = [
        "# Far right wing of the rho < 0 calendar conjecture",
        "",
        "Output of `uv run python derivations/calendar_tail.py`; formulas in the module "
        f"docstring. Log-space sums on a Z grid of step {STEP}, F = D = 1, x = ln(K/F), "
        "u = sqrt(tau Q(tau)). Columns: ratio_c = (dC/du) / T1 with T1 the volatility term "
        "(same normalisation as calendar_derivative.md; positive = no calendar arbitrage); "
        "ln T1; dlnp = d ln P(S > K) / du; G = Laplace prediction of dlnp; mm = m'(u)/m(u); "
        "Z* = saddle.",
        "",
        f"- Points: {len(rows)}; points with dC/du <= 0: {len(neg)}.",
        f"- Far wing (x >= 10): dlnp / G from {min(r['dlnp'] / r['g'] for r in far):.3f} to "
        f"{max(r['dlnp'] / r['g'] for r in far):.3f}; m'/m from "
        f"{min(r['mm'] for r in rows):.4f} to {max(r['mm'] for r in rows):.4f}.",
        "",
        "| rho | k | u | x | ratio_c | ln T1 | dlnp | G | dlnp/G | m'/m | Z* |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *(
            f"| {r['rho']} | {r['k']} | {r['u']} | {r['x']:g} | {r['ratio_c']:.4f} | "
            f"{r['log_t1']:.1f} | {r['dlnp']:.4g} | {r['g']:.4g} | {r['dlnp'] / r['g']:.3f} | "
            f"{r['mm']:.4f} | {r['zs']:.2f} |"
            for r in rows
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
    print(f"Wrote {OUT}; dC/du <= 0 at {len(neg)} of {len(rows)} points")


if __name__ == "__main__":
    main()
