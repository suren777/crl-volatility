"""Exact maturity derivative of CRL call prices: a positive volatility term plus a
correlation covariance (route-1 attempt at the rho < 0 calendar conjecture).

Usage: uv run python derivations/calendar_derivative.py   (writes calendar_derivative.md)

With u = sqrt(t), t = tau Q(tau), F = D = 1 (see calendar_derivative.md for the derivation):

    dC/du = T1 + T2,
    T1 = E_Z[A sqrt((1-rho^2) zeta) phi(d1)] >= 0,
    T2 = Cov_{P*}(N(d1(Z)), h(Z)),   h = rho sqrt(zeta) Z - rho^2 u zeta,

A(Z) = E[S | Z], w = (1-rho^2) u^2 zeta, d1 = (ln(A/K) + w/2) / sqrt(w), and P* the share
measure, under which Z has density phi(z) A(z). All sums are in log space on a dense
linear Z grid, valid for rho <= 0 (for rho > 0 the mass sits beyond any fixed grid).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import norm

Z = np.linspace(-40.0, 120.0, 400_001)
LW = norm.logpdf(Z) + np.log(Z[1] - Z[0])
OUT = __file__.replace(".py", ".md")


def _setup(
    u: float, k: float, rho: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    s = np.sqrt(np.log1p(k * k))
    zeta = np.exp(s * Z - s * s / 2)
    la = rho * u * np.sqrt(zeta) * Z - rho**2 * u * u * zeta / 2
    l_a = la - logsumexp(LW + la)
    h = rho * np.sqrt(zeta) * Z - rho**2 * u * zeta
    return zeta, l_a, h, LW + l_a


def parts(x: float, u: float, k: float, rho: float) -> tuple[float, float]:
    """(T1, T2) at log-strike x = ln(K/F)."""
    zeta, l_a, h, lwa = _setup(u, k, rho)
    w = (1 - rho**2) * u * u * zeta
    d1 = (l_a - x + w / 2) / np.sqrt(w)
    t1 = float(np.exp(logsumexp(lwa + 0.5 * np.log((1 - rho**2) * zeta) + norm.logpdf(d1))))
    hb = float(np.sum(np.exp(lwa - logsumexp(lwa)) * h))
    # T2 = g_bar (E_{A g}[h] - E_A[h]); for x < 0 use the put side to avoid cancellation
    lg, sgn = (lwa + norm.logcdf(d1), 1.0) if x >= 0 else (lwa + norm.logsf(d1), -1.0)
    lgb = logsumexp(lg)
    hg = float(np.sum(np.exp(lg - lgb) * h))
    return t1, sgn * float(np.exp(lgb)) * (hg - hb)


def main() -> None:
    xs = (-10.0, -3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 10.0, 40.0, 200.0)
    lines = [
        "# CRL calendar derivative: tail scan of (T1 + T2) / T1",
        "",
        "Output of `uv run python derivations/calendar_derivative.py`; derivation in the",
        "section below. Positive ratio = no calendar arbitrage at that point; ratio < 1 means",
        "the correlation term T2 is negative. Columns: log-strike x = ln(K/F).",
        "",
        "| rho | k | u | " + " | ".join(f"x={x:g}" for x in xs) + " |",
        "|" + "---|" * (3 + len(xs)),
    ]
    for rho in (-0.05, -0.49, -0.9, -0.99):
        for k in (0.1, 2.2, 3.0):
            for u in (0.3, 2.0):
                cells = []
                for x in xs:
                    t1, t2 = parts(x, u, k, rho)
                    cells.append(f"{(t1 + t2) / t1:.4f}" if t1 > 0 else "underflow")
                lines.append(f"| {rho} | {k} | {u} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(OUT) as f:
        old = f.read()
    head = old.split("<!-- scan -->")[0] if "<!-- scan -->" in old else ""
    with open(OUT, "w") as f:
        f.write(head + "<!-- scan -->\n" + text)


if __name__ == "__main__":
    main()
