"""Structural lemmas for rho < 0 at large L' (epic 0016, task 07-largeL).

Usage: uv run python derivations/calendar_largeL.py   (writes calendar_largeL.md)

With q(v) = mu cosh(v/2) + r (v + c0)/s and E = {q < 0} (calendar_remainder_rho.md), the
PROVED lemmas in calendar_largeL.md say: E is an interval, E lies in (-inf, -c0 - mu s/r), and
V'' >= 1/sigma off E. This script checks them on a grid and prints the scan. It proves no bound
on R' and gives no threshold L0.
"""

from itertools import product

import numpy as np
from calendar_remainder_rho import V_GRID, potential

OUT = __file__.replace(".py", ".md")
US = (0.1, 1.0, 5.0)
KS = (0.05, 0.3, 1.0, 2.0, 2.5)
RS = (0.05, 0.2, 0.5, 0.7, 0.9, 0.99)
LAMS = (0.5, 2.0, 10.0, 50.0, 300.0, 3000.0)
WINDOW = np.abs(V_GRID) <= 5.0


def check(lam: float, u: float, k: float, r: float) -> dict[str, float | bool]:
    sigma, c0, big_v = potential(lam, u, k, r)
    v = V_GRID
    mu = np.sqrt(2 * lam)
    s = np.sqrt(sigma)
    q = mu * np.cosh(v / 2) + r * (v + c0) / s
    neg = q < 0
    idx = np.flatnonzero(neg)
    is_interval = idx.size == 0 or idx[-1] - idx[0] + 1 == idx.size
    t0 = -c0 - mu * s / r
    inside_left = idx.size == 0 or bool(v[idx[-1]] < t0)
    q1 = mu * np.sinh(v / 2) / 2 + r / s
    q2 = mu * np.cosh(v / 2) / 4
    vpp = 1 / sigma + (q1**2 + q * q2) / (1 - r * r)
    off = ~neg
    fd = (big_v[2:] - 2 * big_v[1:-1] + big_v[:-2]) / (V_GRID[1] - V_GRID[0]) ** 2
    sel = WINDOW[1:-1] & off[1:-1]
    ref = vpp[1:-1][sel]
    fd_err = (
        float(np.max(np.abs(fd[sel] - ref) / np.maximum(1.0, np.abs(ref)))) if sel.any() else 0.0
    )
    curv_low = float(np.min(vpp[off])) - 1 / sigma if off.any() else 0.0
    w = np.exp(-(big_v - big_v.min()))
    mass = float(np.sum(w[neg]) / np.sum(w))
    return {
        "sigma": sigma,
        "E_empty": idx.size == 0,
        "E_interval": bool(is_interval),
        "E_left_of_t0": inside_left,
        "t0": float(t0),
        "E_max": float(v[idx[-1]]) if idx.size else float("nan"),
        "mass": mass,
        "curv_low": curv_low,
        "fd_err": fd_err,
    }


def main() -> None:
    rows = []
    for u, k, r, lam in product(US, KS, RS, LAMS):
        rows.append({"u": u, "k": k, "r": r, "lam": lam, **check(lam, u, k, r)})
    lines = [
        "# Structural lemmas for rho < 0 (scan)",
        "",
        "Output of `uv run python derivations/calendar_largeL.py`. Grid: u in "
        f"{US}, k in {KS}, rho in {tuple(-r for r in RS)}, L' in {LAMS}; {len(rows)} points.",
        "",
        f"- E is an interval at every point: {all(row['E_interval'] for row in rows)}.",
        f"- E lies left of -c0 - mu s/r at every point with E nonempty: "
        f"{all(row['E_left_of_t0'] for row in rows)}.",
        f"- min over the complement of E of (V'' - 1/sigma): "
        f"{min(row['curv_low'] for row in rows):.3e} (must be >= 0 up to rounding).",
        f"- max relative |finite-difference V'' - formula| off E on |v| <= 5: "
        f"{max(row['fd_err'] for row in rows):.3e}.",
        f"- E nonempty at {sum(not row['E_empty'] for row in rows)} of {len(rows)} points.",
        f"- nu-mass of E: max {max(row['mass'] for row in rows):.3e}; "
        f"max at L' >= 300: {max(row['mass'] for row in rows if row['lam'] >= 300.0):.3e}.",
        "",
        "| u | k | rho | L' | E empty | E max | -c0 - mu s/r | nu(E) |",
        "|---|---|---|---|---|---|---|---|",
        *(
            f"| {row['u']} | {row['k']} | {-row['r']:g} | {row['lam']} | {row['E_empty']} | "
            f"{row['E_max']:.4f} | {row['t0']:.4f} | {row['mass']:.2e} |"
            for row in rows
            if row["u"] == 1.0 and row["k"] == 1.0
        ),
        "",
    ]
    with open(OUT) as f:
        head = f.read().split("<!-- scan -->")[0]
    with open(OUT, "w") as f:
        f.write(head + "<!-- scan -->\n" + "\n".join(lines))
    print(f"Wrote {OUT}; {len(rows)} rows")


if __name__ == "__main__":
    main()
