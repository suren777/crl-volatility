"""Paired Monte-Carlo calendar-arbitrage certificate for the CRL model (writes calendar_mc.md).

Why: fixed-node Gauss-Hermite does not converge for rho > 0 (search_calendar.md planner
review; calendar_moments.md node-doubling FAIL) — the call integrand peaks far in the Z
tail. This certificate samples the exact continuum terminal law directly, no quadrature:

    ln S_tau/F = -ln m - zeta*H/2 + rho*sqrt(zeta*H)*Z + sqrt((1-rho^2)*zeta*H)*W,

with zeta = exp(s Z - s^2/2), s^2 = ln(1+k^2), H(tau) = tau*Q(tau) = int_0^tau g
(terminal-law derivation in calendar_noarb.md section 1). m is estimated on the same
draws, so the MC forward is exact by construction (mean_i[M(Z_i)/m_hat] = 1). Common
random numbers plus antithetic pairs across tau give tight paired differences
D(K) = C(tau1, K) - C(tau2, K); a witness is lower-CI(D) > 0.

The paired difference is estimated via put-call parity (C - P = F - K with the same
exact forward and D = 1, so C(tau1,K) - C(tau2,K) = P(tau1,K) - P(tau2,K)): the put
payoff (K - S)+ is bounded by K, so the estimator has finite moments of every order
even where E[S^2] explodes (rho^2 -> 1/2), which plain call payoffs do not.

m is computed once per maturity by adaptive quadrature of E[M(Z)] — deterministic and
seed-independent. This matters: at rho > 0 and large k, M(Z) has a far-tail bump
(calendar_noarb.md section 3), and an MC-estimated m_hat carries ~1e-3 relative
seed noise that propagates into deep-ITM put prices an order of magnitude above the
paired SEs — the earlier reproduction-gate failures. The bump afflicts only the
*call* integrand (M times payoff), never M alone: M is bounded and decays both ends,
so adaptive quadrature nails it.

Validation: at rho in {0, -0.3} (where msv.crl_call is trusted) MC must agree within
4 standard errors plus a 1e-6 absolute floor (extreme-OTM points have price ~1e-7
carried by quadrature tail nodes no MC sample reaches, with SE exactly 0). Every
witness is re-run on an independent seed; sign and CI must reproduce.

Run: uv run python derivations/calendar_mc.py  ->  writes calendar_mc.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date  # noqa: E402

import numpy as np  # noqa: E402
from scipy.integrate import quad  # noqa: E402
from scipy.stats import norm  # noqa: E402
from search_calendar import NS_CURVES  # noqa: E402

import msv  # noqa: E402

N_PAIRS = 500_000
K_GRID = np.array([0.5, 1.0, 1.5, 2.0])
TAU_GRID = np.array(np.geomspace(0.01, 5.0, 8))
RHO_GRID = (0.05, 0.3, 0.7, 0.95)
K_GRID_VALID = (0.5, 1.3, 2.6)

Witness = tuple[float, float, tuple[float, float, float, float], float, float, float]
Lines = list[str]


def draws(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z0 = np.asarray(rng.standard_normal(N_PAIRS), dtype=float)
    w0 = np.asarray(rng.standard_normal(N_PAIRS), dtype=float)
    return (np.concatenate([z0, -z0]), np.concatenate([w0, -w0]))  # antithetic


def paired(x: np.ndarray) -> tuple[float, float]:
    """Antithetic pair means -> (estimate, se)."""
    half = 0.5 * (x[:N_PAIRS] + x[N_PAIRS:])
    return float(half.mean()), float(half.std(ddof=1) / np.sqrt(N_PAIRS))


def terminal(
    z: np.ndarray,
    w: np.ndarray,
    tau: float,
    ns: tuple[float, float, float, float],
    k: float,
    rho: float,
) -> np.ndarray:
    """One draw-batch of the exact continuum terminal law S_tau/F (calendar_noarb.md §1)."""
    q = float(msv.Q(tau, *ns))
    s2 = np.log1p(k * k)
    zeta = np.exp(np.sqrt(s2) * z - s2 / 2.0)
    h = tau * q
    mz = np.exp(rho * np.sqrt(zeta * h) * z - rho**2 * zeta * h / 2.0)
    sd = np.sqrt((1.0 - rho**2) * zeta * h)
    st = (mz / m_exact(tau, ns, k, rho)) * np.exp(-(1.0 - rho**2) * zeta * h / 2.0 + sd * w)
    return np.asarray(st, dtype=float)


def m_exact(
    tau: float,
    ns: tuple[float, float, float, float],
    k: float,
    rho: float,
) -> float:
    """E[M(Z)] by adaptive quadrature — deterministic, seed-independent (see module doc)."""
    s2 = float(np.log1p(k * k))
    s = float(np.sqrt(s2))
    h = tau * float(msv.Q(tau, *ns))

    def integrand(z: float) -> float:
        zeta = float(np.exp(s * z - s2 / 2.0))
        log_m = rho * np.sqrt(zeta * h) * z - rho**2 * zeta * h / 2.0
        return float(np.exp(log_m) * norm.pdf(z))

    val, _ = quad(integrand, -12.0, 12.0, limit=200)
    return float(val)


def price(
    z: np.ndarray,
    w: np.ndarray,
    tau: float,
    ns: tuple[float, float, float, float],
    k: float,
    rho: float,
    strike: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """MC call prices (per strike) and SEs."""
    st = terminal(z, w, tau, ns, k, rho)
    est, ses = [], []
    for kf in strike:
        e, s = paired(np.maximum(st - kf, 0.0))
        est.append(e)
        ses.append(s)
    return np.array(est), np.array(ses)


def call_diff(
    z: np.ndarray,
    w: np.ndarray,
    tau1: float,
    tau2: float,
    ns: tuple[float, float, float, float],
    k: float,
    rho: float,
    strike: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Paired difference C(tau1,K) - C(tau2,K) with CRN; positive = calendar violation."""
    st1 = terminal(z, w, tau1, ns, k, rho)
    st2 = terminal(z, w, tau2, ns, k, rho)
    est, ses = [], []
    for kf in strike:
        d = np.maximum(kf - st1, 0.0) - np.maximum(kf - st2, 0.0)  # P1 - P2 = C1 - C2
        e, s = paired(d)
        est.append(e)
        ses.append(s)
    return np.array(est), np.array(ses)


def validate(z: np.ndarray, w: np.ndarray, lines: Lines) -> None:
    """MC vs msv.crl_call at rho in {0, -0.3}: 4*SE + 1e-6 tolerance; report worst point."""
    lines.append("## Validation vs `msv.crl_call` (trusted for rho <= 0)")
    m0 = m_exact(float(TAU_GRID[-1]), NS_CURVES["flat 0.04"], 1.3, 0.0)
    m_ok = abs(m0 - 1.0) < 1e-9
    lines.append(f"- m_exact at rho=0: {m0:.12f} (exact 1: {'OK' if m_ok else 'FAIL'})")
    worst = (0.0, "")
    for rho in (0.0, -0.3):
        for k in K_GRID_VALID:
            ns = NS_CURVES["flat 0.04"]
            for tau in TAU_GRID:
                est, ses = price(z, w, float(tau), ns, k, rho, K_GRID)
                for j, kf in enumerate(K_GRID):
                    ref = float(msv.crl_call(1.0, float(kf), 1.0, float(tau), *ns, k, rho)[0])
                    # 1e-6 absolute floor: extreme-OTM prices (~1e-7) ride on quadrature
                    # tail nodes no MC sample reaches, with SE exactly 0.
                    zsc = abs(est[j] - ref) / (4.0 * ses[j] + 1e-6)
                    if zsc > worst[0]:
                        worst = (
                            zsc,
                            f"rho={rho} k={k} tau={tau:.4f} K/F={kf:.2f}: "
                            f"mc={est[j]:.6f} crl_call={ref:.6f} se={ses[j]:.2e}",
                        )
    ok = m_ok and worst[0] < 1.0
    verdict = "PASS" if ok else "FAIL"
    lines.append(f"- worst |MC - crl_call| / tolerance: {worst[0]:.2f} ({verdict}, gate 1.0)")
    if not ok:
        lines.append(f"- worst point: {worst[1]}")
    lines.append("")


def main() -> None:
    lines: Lines = [
        "# Calendar-arbitrage Monte-Carlo certificate for the CRL model",
        "",
        f"Run {date.today():%Y-%m-%d}. Quadrature-free: samples the exact continuum terminal law",
        "(calendar_noarb.md section 1) with common random numbers and antithetic pairs;",
        f"{N_PAIRS:,} antithetic pairs per comparison. Witness = lower CI of the paired drop",
        "C(tau1,K) - C(tau2,K) above 0 (drop > 0 means the shorter maturity is strictly",
        "dearer: a calendar violation), estimated via put-call parity so payoffs are bounded",
        "by K. m computed per maturity by adaptive quadrature of E[M(Z)] (deterministic).",
        "",
    ]
    z, w = draws(20261019)
    validate(z, w, lines)

    lines.append("## Paired scan: rho > 0, three NS curves, adjacent tau pairs")
    lines.append("")
    lines.append("| rho | k | NS | tau1 -> tau2 | K/F | drop (C1-C2) | SE | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")
    n_wit = 0
    witnesses: list[Witness] = []
    n_comp = len(RHO_GRID) * len(K_GRID_VALID) * len(NS_CURVES) * (len(TAU_GRID) - 1) * len(K_GRID)
    for rho in RHO_GRID:
        for k in K_GRID_VALID:
            for ns_name, ns in NS_CURVES.items():
                pairs = zip(TAU_GRID[:-1], TAU_GRID[1:], strict=True)
                for t1, t2 in pairs:
                    est, ses = call_diff(z, w, float(t1), float(t2), ns, k, rho, K_GRID)
                    for j, kf in enumerate(K_GRID):
                        if est[j] - 3.0 * ses[j] > 0.0:
                            n_wit += 1
                            witnesses.append((rho, k, ns, float(t1), float(t2), float(kf)))
                            lines.append(
                                f"| {rho:.2f} | {k:.2f} | {ns_name} | {t1:.4f} -> {t2:.4f} "
                                f"| {kf:.2f} | {est[j]:+.3e} | {ses[j]:.1e} | **witness** |"
                            )
    lines.append("")
    lines.append(f" violations (lower CI > 0): **{n_wit}** of {n_comp} comparisons")
    lines.append("")

    lines.append("## Star witnesses from the failed-quadrature scans, re-tested here")
    lines.append("")
    stars: list[Witness] = [
        (0.0068, 2.6129, (0.0099, 0.0067, 0.0356, 1.1559), 0.1435, 0.3486, 2.0),
        (0.05, 0.5, (0.0, 0.0, 0.04, 1.0), 0.63, 0.8162, 2.0),
        (0.05, 0.5, (0.0, 0.0, 0.04, 1.0), 0.6300, 0.8162, 0.5),
    ]
    for rho, k, ns, t1, t2, kf in stars:
        est, ses = call_diff(z, w, t1, t2, ns, k, rho, np.array([kf]))
        lo, hi = est[0] - 3 * ses[0], est[0] + 3 * ses[0]
        verdict = "violation" if lo > 0 else "consistent with no arbitrage"
        lines.append(
            f"- rho={rho:+.4f} k={k:.4f} ns={ns} K/F={kf:.2f} tau {t1:.4f} -> {t2:.4f}: "
            f"drop {est[0]:+.3e} (CI [{lo:+.3e}, {hi:+.3e}]) -> {verdict}"
        )
    lines.append("")

    lines.append("## Reproduction gate (independent seed)")
    lines.append("")
    z2, w2 = draws(20261020)
    ok = True
    for rho, k, ns, t1, t2, kf in witnesses:
        e1, s1 = call_diff(z, w, t1, t2, ns, k, rho, np.array([kf]))
        e2, s2 = call_diff(z2, w2, t1, t2, ns, k, rho, np.array([kf]))
        if not (np.sign(e1[0]) == np.sign(e2[0]) and abs(e1[0] - e2[0]) < 3 * (s1[0] + s2[0])):
            ok = False
    gate_ok = ok
    lines.append(
        f"- {len(witnesses)} witnesses re-run on seed 20261020: "
        f"{'PASS (same sign, overlapping CIs)' if ok else 'FAIL'}"
    )
    lines.append("")

    lines.append("## Outcome")
    lines.append("")
    if n_wit == 0:
        lines.append("- No calendar violation found at any tested rho > 0 point: the")
        lines.append("  fixed-node quadrature 'violations' are artifacts, and the convex-order")
        lines.append("  question stays open (calendar_noarb.md section 4) — but now with a")
        lines.append("  quadrature-free negative result over the same box.")
    elif gate_ok:
        lines.append(f"- {n_wit} certified violations at rho > 0 (reproduction-gated): the CRL")
        lines.append("  model admits calendar arbitrage there (paired MC on the exact continuum")
        lines.append("  law, bounded put payoffs via put-call parity).")
    else:
        lines.append(f"- {n_wit} apparent violations FAILED the independent-seed reproduction")
        lines.append("  gate: no conclusion — the estimator is seed-sensitive at those points;")
        lines.append("  rho > 0 stays unresolved (calendar_noarb.md section 4).")
    lines.append("")

    out = Path(__file__).with_suffix(".md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines[-8:]))


if __name__ == "__main__":
    main()
