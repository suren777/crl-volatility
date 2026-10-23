"""Calendar-arbitrage check of crl_call at the fitted S&P 500 parameter sets.

For each CRL fit in results/panel.csv (trade dates 2026-10-16, -19, -20, -22),
prices undiscounted calls with F = D = 1 on a grid of 200 strikes K/F
log-spaced in [0.3, 3] and 200 maturities tau log-spaced in [1/365, 5] years,
and counts increments C(K, tau_{j+1}) - C(K, tau_j) below -TOL: calendar
no-arbitrage requires them non-negative at every strike (paper/noarbitrage.tex,
conjecture conj:noarb-neg for rho < 0). A date with a violation is re-priced at
128 Gauss-Hermite nodes to tell quadrature noise from a genuine violation.

Run: uv run python experiments/calendar_fitted.py  -> writes results/calendar_fitted.md
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date  # noqa: E402

import numpy as np  # noqa: E402

import msv  # noqa: E402

PARAM_COLS = ("a0", "a1", "a2", "lam", "k", "rho")
N_K = 200
N_TAU = 200
KF = np.geomspace(0.3, 3.0, N_K)
TAUS = np.geomspace(1.0 / 365.0, 5.0, N_TAU)
TOL = 1e-10  # price drop counted as a violation (F = 1, so prices are O(1) at most)
CSV = Path(__file__).with_name("results") / "panel.csv"
MD = Path(__file__).with_name("results") / "calendar_fitted.md"


def fitted_crl() -> list[dict[str, str]]:
    """The CRL rows of the panel, one per trade date, in date order."""
    with CSV.open() as f:
        rows = [r for r in csv.DictReader(f) if r["model"] == "crl"]
    return sorted(rows, key=lambda r: r["trade_date"])


def increments(params: tuple[float, ...], n_nodes: int) -> np.ndarray:
    """Call-price increments across tau, shape (n_tau - 1, n_k)."""
    a0, a1, a2, lam, k, rho = params
    px = np.array(
        [
            msv.crl_call(1.0, KF, 1.0, float(t), a0, a1, a2, lam, k, rho, n_nodes=n_nodes)
            for t in TAUS
        ]
    )
    return np.diff(px, axis=0)


def check(row: dict[str, str], n_nodes: int) -> tuple[list[str], bool]:
    """Price the grid at one fit; returns (report lines, whether a violation was flagged)."""
    date_ = row["trade_date"]
    params = tuple(float(row[c]) for c in PARAM_COLS)
    d = increments(params, n_nodes)
    i_t, i_k = (int(v) for v in np.unravel_index(int(np.argmin(d)), d.shape))
    worst = float(d[i_t, i_k])
    n_bad = int((d < -TOL).sum())
    a0, a1, a2, lam, k, rho = params
    lines = [
        f"## {date_}: a0={a0:.5g} a1={a1:.5g} a2={a2:.5g} lam={lam:.5g} k={k:.5g} rho={rho:.5g}"
        f" ({n_nodes} nodes)",
        f"- min increment {worst:+.3e} at K/F={KF[i_k]:.4f},"
        f" tau {TAUS[i_t]:.6g} -> {TAUS[i_t + 1]:.6g}",
        f"- increments below -{TOL:g}: {n_bad} of {d.size}",
    ]
    if n_bad:
        d2 = increments(params, 128)
        i2 = int(np.argmin(d2))
        worst2 = float(d2.flat[i2])
        n_bad2 = int((d2 < -TOL).sum())
        lines += [
            f"- re-priced at 128 nodes: min increment {worst2:+.3e},"
            f" increments below -{TOL:g}: {n_bad2} of {d2.size}",
            "- violation persists: genuine, not quadrature noise"
            if n_bad2
            else "- violation does not persist at 128 nodes: quadrature noise",
        ]
    lines.append("")
    return lines, bool(n_bad)


def main() -> None:
    rows = fitted_crl()
    blocks = [check(row, 64) for row in rows]
    lines = [
        "# Calendar check of `crl_call` at the fitted S&P 500 parameter sets",
        "",
        f"Run {date.today().isoformat()}. For each CRL fit in `results/panel.csv`, undiscounted "
        f"call prices with F = D = 1 on K/F in [0.3, 3] ({N_K} log-spaced points) and tau in "
        f"[1/365, 5] ({N_TAU} log-spaced points), 64-node Gauss-Hermite. Checked: the "
        "increments C(K, tau_{j+1}) - C(K, tau_j) as tau increases, at every strike; a "
        f"violation is an increment below -{TOL:g}. A date with a violation is re-priced at "
        "128 nodes. No hand-edited numbers; this file is the script's output.",
        "",
        *(line for lines_i, _ in blocks for line in lines_i),
        "## Conclusion",
    ]
    if any(flagged for _, flagged in blocks):
        lines.append(
            f"- {sum(flagged for _, flagged in blocks)} date(s) show increments below "
            f"-{TOL:g}; see the 128-node re-pricing above for whether the violation persists."
        )
    else:
        lines.append(
            f"- no calendar increment is negative below -{TOL:g} at any of the four fits; "
            "consistent with the no-arbitrage conjecture for rho <= 0 at the fitted parameters."
        )
    text = "\n".join(lines)
    print(text)
    MD.write_text(text)
    print(f"wrote {MD}")


if __name__ == "__main__":
    main()
