"""Random search for calendar-arbitrage violations of crl_call (exploration, NOT a test).

Calendar no-arbitrage: for fixed K and F the undiscounted call price must be
non-decreasing in tau. Each draw fixes (rho, k, a0, a1, a2, lam), prices a
K/F x tau grid with msv.crl_call (F = D = 1) and counts increments
C(tau2) - C(tau1) below -TOL. Main search: rho in [-0.99, 0], the equity
region conjectured arbitrage-free (epics/0010). Sanity search: rho in
(0, 0.99], where violations are known to exist (epic grid: rho = +0.5, k >= 1)
-- it must find them, or the search is not looking hard enough.

Nelson-Siegel shapes: the three curves of experiments/crl_skew.py, plus uniform
draws in the (a0, a1, a2, lam) box those curves span.

Run: uv run python derivations/search_calendar.py  -> writes search_calendar.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date  # noqa: E402

import numpy as np  # noqa: E402

import msv  # noqa: E402

SEED = 20261019
N_DRAWS = 20_000
N_SANITY = 2_000
KF = np.linspace(0.5, 2.0, 8)
TAUS = np.geomspace(0.01, 5.0, 8)
TOL = 1e-10  # price drop counted as a violation (F = 1, so prices are O(1) at most)
N_EXAMPLES = 20
NS_CURVES: dict[str, tuple[float, float, float, float]] = {
    "flat 0.04": (0.0, 0.0, 0.04, 1.0),
    "decreasing short-end": (0.03, 0.0, 0.03, 2.0),
    "humped": (0.0, 0.06, 0.03, 1.5),
}
NS_BOX = ((0.0, 0.03), (0.0, 0.06), (0.03, 0.04), (1.0, 2.0))  # spanned by the three curves
MD = Path(__file__).with_name("search_calendar.md")


def search(
    rho_lo: float, rho_hi: float, n_draws: int, rng: np.random.Generator
) -> tuple[int, int, int, float, str, list[str]]:
    """Random parameter draws; returns summary of calendar violations found."""
    viol_draws = 0
    viol_incr = 0
    worst = 0.0
    worst_desc = "n/a"
    examples: list[str] = []
    curve_names = list(NS_CURVES)
    for _ in range(n_draws):
        if rng.random() < 0.5:
            name = curve_names[int(rng.integers(len(curve_names)))]
            ns = NS_CURVES[name]
        else:
            a0, a1, a2, lam = (float(rng.uniform(lo, hi)) for lo, hi in NS_BOX)
            name, ns = "uniform box", (a0, a1, a2, lam)
        k = float(rng.uniform(0.05, 3.0))
        rho = float(rng.uniform(rho_lo, rho_hi))
        # one call per tau: crl_call pairs equal-length K/tau arrays (see crl_skew.py)
        px = np.array([msv.crl_call(1.0, KF, 1.0, float(t), *ns, k, rho) for t in TAUS])
        d = np.diff(px, axis=0)  # (n_taus-1, n_k): price change as tau increases
        bad = d < -TOL
        if not bad.any():
            continue
        viol_draws += 1
        viol_incr += int(bad.sum())
        i_t, i_k = (
            int(v) for v in np.unravel_index(int(np.argmin(np.where(bad, d, np.inf))), d.shape)
        )
        drop = float(d[i_t, i_k])
        desc = (
            f"rho={rho:+.4f} k={k:.4f} ns=({name}) {tuple(round(v, 4) for v in ns)} "
            f"K/F={KF[i_k]:.3f} tau {TAUS[i_t]:.4f}->{TAUS[i_t + 1]:.4f}: "
            f"drop {drop:.3e} (C = {float(px[i_t, i_k]):.6f})"
        )
        if drop < worst:
            worst = drop
            worst_desc = desc
        if len(examples) < N_EXAMPLES:
            examples.append(desc)
    return n_draws, viol_draws, viol_incr, worst, worst_desc, examples


def report(title: str, res: tuple[int, int, int, float, str, list[str]]) -> list[str]:
    n_draws, viol_draws, viol_incr, worst, worst_desc, examples = res
    lines = [
        f"## {title}",
        f"- draws: {n_draws}; increments checked: {n_draws * (TAUS.size - 1) * KF.size}",
        f"- draws with a violation: {viol_draws}; violating increments: {viol_incr}",
    ]
    if viol_draws:
        lines += [f"- worst violation: {worst_desc}", "", "First violations (parameters):", "```"]
        lines += examples[:5] or [worst_desc]
        lines += ["```"]
    else:
        lines.append("- worst price drop seen anywhere: >= -TOL (no violation)")
    lines.append("")
    return lines


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"Calendar-arbitrage random search of crl_call, {date.today().isoformat()}, seed {SEED}")
    print(
        f"grid: K/F in [{KF[0]}, {KF[-1]}] ({KF.size} pts), tau in [{TAUS[0]}, {TAUS[-1]}] "
        f"({TAUS.size} log-spaced pts), F = D = 1, 64-node Gauss-Hermite"
    )
    print(
        f"NS: 3 crl_skew.py curves + uniform box (a0, a1, a2, lam) in {NS_BOX}; "
        "k in [0.05, 3]; violation = price drop < -1e-10\n"
    )

    res_main = search(-0.99, 0.0, N_DRAWS, rng)
    res_sanity = search(0.0, 0.99, N_SANITY, rng)

    lines = [
        "# Calendar-arbitrage random search of `crl_call`",
        "",
        f"Run {date.today().isoformat()}, seed {SEED}, `msv.crl_call` (F = D = 1, 64-node "
        "Gauss-Hermite). Grid per draw: K/F in [0.5, 2.0] (8 pts), tau in [0.01, 5.0] "
        "(8 log-spaced pts). Parameters per draw: k in [0.05, 3]; Nelson-Siegel = one of the "
        "three `experiments/crl_skew.py` curves (flat 0.04, decreasing short-end, humped) or a "
        f"uniform draw in the box {NS_BOX} they span. A violation is a price increment across "
        "tau below -1e-10 in absolute price (prices are O(1) at most; quadrature round-off is "
        "order 1e-14). This is numerical evidence over a random box, not a proof.",
        "",
        *report(f"Main search: rho in [-0.99, 0] (equity region), {N_DRAWS:,} draws", res_main),
        *report(
            f"Sanity search: rho in (0, 0.99], {N_SANITY:,} draws (violations expected here)",
            res_sanity,
        ),
        "## Outcome",
        "- rho <= 0: "
        + (
            f"calendar arbitrage found ({res_main[2]} violating increments); see worst above."
            if res_main[1]
            else "no violation found; consistent with the convex-order conjecture (epics/0010), "
            "within the box and tolerance searched."
        ),
        "- rho > 0: "
        + (
            f"search finds the expected violations ({res_sanity[2]} increments), so it is "
            "sensitive enough."
            if res_sanity[1]
            else "NO violations found -- the search failed "
            "its own sanity check; do not trust the rho <= 0 result."
        ),
        "",
    ]
    text = "\n".join(lines)
    print(text)
    MD.write_text(text)
    print(f"wrote {MD}")


if __name__ == "__main__":
    main()
