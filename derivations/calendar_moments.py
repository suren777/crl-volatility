"""Moment route to calendar arbitrage for `crl_call` (writes calendar_moments.md).

Exact representation of the CRL marginals (proof in calendar_proof.md, verified here
against `msv.crl_call`): with t = tau * Q(tau) (variance time), zeta = exp(s Z - s^2/2),
s^2 = ln(1+k^2), U = rho Z + sqrt(1-rho^2) N ~ N(0,1) correlated with Z,

    S_tau = (F / m) exp( sqrt(t zeta(Z)) U  -  t zeta(Z) / 2 ),
    m = E[ exp( rho sqrt(t zeta) Z - rho^2 t zeta / 2 ) ].

Collapsing the U integral (U | Z ~ N(rho Z, 1-rho^2)) gives the exact p-th moment

    E[S^p] = (F/m)^p E_Z[ exp( p rho sqrt(t zeta) Z + (t zeta / 2)(p^2 (1-rho^2) - p) ) ],

finite iff p < 1/(1-rho^2); p = 1 returns F identically. If the per-maturity laws were
convex-ordered (necessary for calendar no-arbitrage: Kellerer 1972), a martingale
coupling exists and E[S^p] must be non-decreasing in tau for every p >= 1 (Jensen).
A certified strict decrease at some p in (1, 1/(1-rho^2)) therefore proves calendar
arbitrage exists. rho = 0 is arbitrage-free outright: the representation is a single
GBM in variance time with the level zeta drawn at t = 0 (calendar_proof.md, Thm 1).

Method. Every E[exp(phi(Z))] integral is a dense trapezoid on the weight-folded
exponent, g = phi - x^2/2 - ln sqrt(2 pi), over a uniform grid on (-CLIP, CLIP),
accumulated in log space; certification doubles the grid resolution and re-runs on
a wider clip. Fixed-node quadrature in log space is the third scheme tried here and
the first that is sound: (1) plain Gauss-Hermite misses the second Laplace bump of
the integrand (where rho sqrt(t zeta(Z)) catches up with Z, far out in the tail for
rho > 0) and scipy's hermitenorm weights underflow to exact zero at extreme nodes
for n >= 512 -- clamping them to a tiny weight creates phantom terms of size
exp(phi(x_node) - 690) (ln m came out 24 nats too big at 512 nodes and diverged
with node count); (2) adaptive quad with a `points` hint at the grid peak is
defeated by bumps far from the subinterval centre (its Gauss-Kronrod nodes step
over a bump of width ~0.1: clip-160 and clip-240 disagreed by up to 76 nats while
loose and tight tolerance at the same clip agreed to 9 decimals -- the same miss
reproduced exactly). A dense grid samples every bump by construction. The 2-D
(Z, N) tensor grid (zero weights dropped), accumulated in log space, validates the
U|Z collapse and ties the representation to `msv.crl_call`; it is only valid where
the integrand stays representable in linear space (rho <= 0.3, k = 0.3, p up to
half the finite-moment window -- for rho > 0 the p-moment integrand peaks at
Z ~ 40-70, beyond any linear-space grid). Witnesses are gated by resolution
doubling and a wider-clip re-run.

Run: uv run python derivations/calendar_moments.py  ->  writes calendar_moments.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections.abc import Callable  # noqa: E402
from datetime import date  # noqa: E402
from functools import lru_cache  # noqa: E402

import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from scipy.special import roots_hermitenorm  # noqa: E402
from search_calendar import NS_CURVES  # noqa: E402

import msv  # noqa: E402

FloatArray = npt.NDArray[np.float64]
NS = tuple[float, float, float, float]
Witness = tuple[float, float, str, NS, float, float, float, float]

TAUS = np.geomspace(0.01, 5.0, 25)
RHOS_POS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
RHOS_NEG = (-0.9, -0.7, -0.5, -0.3, -0.2, -0.1)
KS = (0.3, 0.5, 1.0, 1.5, 2.0, 2.6)
P_FRACS = (0.25, 0.5, 0.75)  # of the finite-moment window (1, 1/(1-rho^2)), above 1
TOL = 1e-6  # ln E[S^p] decrease counted as a violation (certified d are O(100))
N_NODES = 512
GATE = 1e-6  # resolution-doubling agreement required of certified increments
CLIP = 60.0
CLIP_WIDE = 160.0  # certificate integrals run here: the second Laplace bump of the
# rho > 0 integrands sits at Z ~ 45-70, sometimes beyond CLIP (p ~ 4.2, rho = 0.9)
CLIP_XWIDE = 240.0  # truncation-sensitivity check for certification
N_QUAD = 400_001  # trapezoid grid: validation and certification coarse pass
N_QUAD_FINE = 800_001  # resolution-doubled certification pass
N_SCAN = 200_001  # lighter grid for the tau sweep
SQRT_2PI = float(np.sqrt(2.0 * np.pi))
MD = Path(__file__).with_name("calendar_moments.md")


@lru_cache(maxsize=4)
def gh(n_nodes: int) -> tuple[FloatArray, FloatArray]:
    """Gauss-Hermite nodes/weights for the N(0,1) weight, exactly normalised.

    scipy's weights underflow to exact 0 at extreme nodes once the node range
    outgrows float64 (n >= 512 reaches |x| ~ 44 where exp(-x^2/2) < 1e-308); those
    nodes are dropped. Clamping them to a tiny weight instead resurrects them as
    phantom quadrature terms wherever phi is large.
    """
    x, w = roots_hermitenorm(n_nodes)
    keep = w > 0.0
    x, w = x[keep], w[keep]
    w = w / w.sum()
    return x, w


def lse(v: FloatArray) -> float:
    """log-sum-exp; -inf (all-zero payoff) maps to -inf."""
    top = float(v.max())
    if not np.isfinite(top):
        return top
    return top + float(np.log(np.exp(v - top).sum()))


def var_time(tau: float, ns: NS) -> float:
    """Variance time t = tau Q(tau)."""
    return tau * float(msv.Q(tau, *ns))


def phi_m(t: float, s: float, rho: float) -> Callable[[float | FloatArray], FloatArray]:
    """phi with exp(phi(Z)) = M(Z) = exp(rho sqrt(t zeta) Z - rho^2 t zeta / 2).

    Unclipped on purpose: truncation is owned by ln_eexp's fold (it clips x before
    calling phi), so the integration clip can never mismatch the integrand.
    """

    def f(x: float | FloatArray) -> FloatArray:
        zeta = np.exp(s * x - 0.5 * s * s)
        return rho * np.sqrt(t * zeta) * x - 0.5 * rho**2 * t * zeta

    return f


def phi_p(t: float, s: float, rho: float, p: float) -> Callable[[float | FloatArray], FloatArray]:
    """phi with exp(phi(Z)) = M(Z)^p E[exp(p sqrt((1-rho^2) t zeta) N - p(1-rho^2) t zeta/2)]."""

    def f(x: float | FloatArray) -> FloatArray:
        zeta = np.exp(s * x - 0.5 * s * s)
        c = p * p * (1.0 - rho**2) - p
        return p * rho * np.sqrt(t * zeta) * x + 0.5 * t * zeta * c

    return f


def ln_eexp(
    phi: Callable[[float | FloatArray], FloatArray],
    clip: float = CLIP,
    n_points: int = N_QUAD,
) -> float:
    """ln E[exp(phi(Z))] for Z ~ N(0,1): trapezoid on the weight-folded exponent.

    g = phi(x) - x^2/2 - ln sqrt(2 pi) on a dense uniform grid over (-clip, clip),
    accumulated in log space. Fixed-node quadrature in log space is the only sound
    scheme found here: GH weights underflow at tail nodes (phantom terms when
    clamped), and adaptive quad's Gauss-Kronrod nodes step over far-tail Laplace
    bumps of width ~0.1 that hug a `points` break (clip 160 vs 240 disagreed by up
    to 76 nats). A dense grid samples every bump by construction; accuracy is
    checked by resolution doubling at the call sites.
    """
    x = np.linspace(-clip, clip, n_points)
    gv = np.asarray(phi(x) - 0.5 * x * x - np.log(SQRT_2PI), dtype=float)
    top = float(gv.max())
    val = float(np.trapezoid(np.exp(gv - top), dx=2.0 * clip / (n_points - 1)))
    return top + float(np.log(val))


def s_loggrid(
    tau: float, ns: NS, k: float, rho: float, n_nodes: int
) -> tuple[FloatArray, FloatArray]:
    """ln S_tau on the (Z, N) tensor grid and the 2-D log weights (F = 1).

    Independent of the U|Z collapse: the N integral is done by tensor quadrature.
    """
    x, w = gh(n_nodes)
    s = float(np.sqrt(np.log1p(k * k)))
    t = var_time(tau, ns)
    zeta = np.exp(s * x - 0.5 * s * s)
    tz = t * zeta
    lw = np.log(w)
    lm = lse((rho * np.sqrt(tz) * x - 0.5 * rho**2 * tz) + lw)
    drift = rho * np.sqrt(tz) * x - 0.5 * tz - lm
    load = np.sqrt((1.0 - rho**2) * tz)
    ln_s = drift[:, None] + load[:, None] * x[None, :]
    return ln_s, lw[:, None] + lw[None, :]


def call_2d(K: float, tau: float, ns: NS, k: float, rho: float) -> float:
    """E[(S_tau - K)+] from the 2-D grid: an independent check of msv.crl_call."""
    ln_s, lw2 = s_loggrid(tau, ns, k, rho, N_NODES)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # exp(-ln_s) overflows on the discarded branch; cap it -- the selected
        # branch has ln_s > ln(K), so exp(-ln_s) < 1/K and the cap never binds.
        t_ = np.where(ln_s > np.log(K), np.exp(np.minimum(-ln_s, 700.0)), 1.0)
        lpay = np.where(ln_s > np.log(K), ln_s + np.log1p(-K * t_), -np.inf) + lw2
    return float(np.exp(lse(lpay.ravel())))


def moment_2d(p: float, tau: float, ns: NS, k: float, rho: float) -> float:
    """E[S_tau^p] from the 2-D grid: an independent check of the collapsed formula."""
    ln_s, lw2 = s_loggrid(tau, ns, k, rho, N_NODES)
    return float(np.exp(lse((p * ln_s + lw2).ravel())))


def validate() -> list[str]:
    """Collapsed formula vs 2-D grid, 2-D call vs crl_call, E[S] = 1, p = 1 identity.

    The 2-D tensor grid is only valid where everything stays representable in
    linear space: its nodes reach |Z| ~ 38 and the N-load
    sqrt((1-rho^2) t zeta(Z)) must stay below ~36. For rho > 0 and p > 1 the
    p-moment integrand peaks at Z ~ 40-65, so the 2-D box uses rho <= 0.3,
    k = 0.3; rho > 0 accuracy rests on the quad clip-consistency line instead.
    """
    lines = ["## Validation of the representation and the moment formula", ""]
    worst_call = worst_p1 = worst_mean = 0.0
    worst_mom: dict[str, float] = {"le": 0.0, "pos": 0.0}
    worst_clip: dict[str, float] = {"le": 0.0, "pos": 0.0}
    for rho in (-0.5, 0.0, 0.3):
        tag = "le" if rho <= 0 else "pos"
        top = 1.0 / (1.0 - rho**2)
        for k in (0.3,):
            s = float(np.sqrt(np.log1p(k * k)))
            for _name, ns in list(NS_CURVES.items())[:2]:
                for tau in (0.1, 0.25):
                    t = var_time(tau, ns)
                    for K in (0.8, 1.0, 1.2):
                        px = call_2d(K, tau, ns, k, rho)
                        ref = float(msv.crl_call(1.0, K, 1.0, tau, *ns, k, rho)[0])
                        worst_call = max(worst_call, abs(px - ref) / max(abs(ref), 1e-300))
                    ln_s, lw2 = s_loggrid(tau, ns, k, rho, N_NODES)
                    es = float(np.exp(lse((ln_s + lw2).ravel())))
                    worst_mean = max(worst_mean, abs(es - 1.0))
                    lm_q = ln_eexp(phi_m(t, s, rho))
                    # p-fracs kept <= 0.5 here: at 0.75 the p-moment integrand grows a
                    # far peak (Z ~ 60) that a linear-space 2-D grid cannot represent.
                    for frac in (0.1, 0.25, 0.5):
                        p = 1.0 + frac * (top - 1.0)
                        lp_q = ln_eexp(phi_p(t, s, rho, p))
                        m2 = moment_2d(p, tau, ns, k, rho)
                        worst_mom[tag] = max(worst_mom[tag], abs((lp_q - p * lm_q) - np.log(m2)))
                        lp_w = ln_eexp(phi_p(t, s, rho, p), clip=CLIP_WIDE)
                        lm_w = ln_eexp(phi_m(t, s, rho), clip=CLIP_WIDE)
                        worst_clip[tag] = max(
                            worst_clip[tag], abs((lp_q - p * lm_q) - (lp_w - p * lm_w))
                        )
                    lp1 = ln_eexp(phi_p(t, s, rho, 1.0))
                    worst_p1 = max(worst_p1, abs(lp1 - lm_q))
    lines += [
        f"- call prices, 2-D grid vs `msv.crl_call`: worst rel err {worst_call:.2e}",
        f"- ln E[S^p], trapezoid (collapsed) vs 2-D grid, rho <= 0 (grid valid): "
        f"worst abs err {worst_mom['le']:.2e}",
        f"- ln E[S^p], rho = +0.3: worst abs err {worst_mom['pos']:.2e} -- the far bump "
        "(Z ~ 55-65) is beyond any linear-space grid by construction, so this row is a "
        "structural limit, not an error; for rho > 0 the collapsed formula rests on "
        "exact algebra, the p = 1 identity below, and the trapezoid resolution/clip gates",
        f"- ln E[S^p], trapezoid at clip {CLIP:g} vs {CLIP_WIDE:g}, rho <= 0: worst abs err "
        f"{worst_clip['le']:.2e}; rho = +0.3: {worst_clip['pos']:.2e} (clip {CLIP:g} cuts "
        "the far bump at its peak)",
        f"- E[S] = 1 from the 2-D grid: worst |err| {worst_mean:.2e}",
        f"- p = 1 identity ln E[S] = 0 (trapezoid): worst residual {worst_p1:.2e}",
        "",
    ]
    return lines


def scan() -> tuple[list[str], list[Witness]]:
    """Sweep (rho, k, NS, p) series over TAUS with adaptive quadrature."""
    witnesses: list[Witness] = []
    n_series = {"pos": 0, "neg": 0}
    viol = {"pos": 0, "neg": 0}
    for tag, rhos in (("pos", RHOS_POS), ("neg", RHOS_NEG)):
        for rho in rhos:
            top = 1.0 / (1.0 - rho**2)
            ps = [1.0 + f * (top - 1.0) for f in P_FRACS]
            for k in KS:
                s = float(np.sqrt(np.log1p(k * k)))
                for name, ns in NS_CURVES.items():
                    ts = [var_time(tau, ns) for tau in TAUS]
                    lms = [ln_eexp(phi_m(t, s, rho), clip=CLIP_WIDE, n_points=N_SCAN) for t in ts]
                    for p in ps:
                        n_series[tag] += 1
                        ln_e = [
                            ln_eexp(phi_p(t, s, rho, p), clip=CLIP_WIDE, n_points=N_SCAN) - p * lm
                            for t, lm in zip(ts, lms, strict=True)
                        ]
                        d = np.diff(ln_e)
                        if d.min() < -TOL:
                            i = int(d.argmin())
                            viol[tag] += 1
                            wit: Witness = (
                                rho,
                                k,
                                name,
                                ns,
                                p,
                                float(TAUS[i]),
                                float(TAUS[i + 1]),
                                float(d[i]),
                            )
                            witnesses.append(wit)
    lines = [
        "## Moment-monotonicity scan (log-space trapezoid)",
        "",
        "- finite-moment window per rho: p in (1, 1/(1-rho^2)); p at 25/50/75% of the window",
        f"- tau grid: {TAUS.size} log-spaced points in [{TAUS[0]}, {TAUS[-1]}], "
        f"trapezoid of the folded exponent on (-{CLIP_WIDE:g}, {CLIP_WIDE:g}), "
        f"violation = ln-increment < -{TOL:g}",
        f"- rho > 0: {viol['pos']} of {n_series['pos']} series violate",
        f"- rho < 0: {viol['neg']} of {n_series['neg']} series violate",
        "",
    ]
    if witnesses:
        lines += ["Worst witnesses (rho, k, NS, p, tau1 -> tau2, d ln E[S^p]):", "```"]
        for rho, k, name, ns, p, t1, t2, dv in sorted(witnesses, key=lambda e: e[7])[:10]:
            lines.append(
                f"rho={rho:+.4f} k={k:.2f} ns={name!r} {ns} p={p:.6f} "
                f"tau {t1:.4f} -> {t2:.4f}: {dv:.3e}"
            )
        lines += ["```", ""]
    else:
        lines += ["- no violation found in the scan", ""]
    return lines, witnesses


def certify(witnesses: list[Witness]) -> tuple[list[str], list[Witness]]:
    """Re-integrate up to 12 witnesses at tight tolerance and a wider clip."""
    if not witnesses:
        return ["## Certification", "", "- no witnesses to certify", ""], []
    by_d = sorted(witnesses, key=lambda e: e[7])
    # 6 mildest + 6 worst: tame parameters may certify where the extreme tail is
    # clip-dominated (the worst witnesses sit beyond any tested clip).
    picks = list(dict.fromkeys(by_d[:6] + by_d[-6:]))[:12]
    lines = [
        "## Certification (resolution-doubling and clip-sensitivity gates)",
        "",
        "- increment d = ln E[S^p](tau2) - ln E[S^p](tau1) at h / h/2 resolution "
        f"(clip {CLIP_WIDE:g}) / h/2 resolution (clip {CLIP_XWIDE:g}); gate: pairwise "
        f"agreement < {GATE:g} and both < -{TOL:g}",
        "",
    ]
    certified: list[Witness] = []
    worst_gate = 0.0
    for rho, k, name, ns, p, t1, t2, _d in picks:
        s = float(np.sqrt(np.log1p(k * k)))
        incs = []
        for n in (N_QUAD, N_QUAD_FINE):
            vals = []
            for tau in (t1, t2):
                t = var_time(tau, ns)
                lp = ln_eexp(phi_p(t, s, rho, p), clip=CLIP_WIDE, n_points=n)
                lm = ln_eexp(phi_m(t, s, rho), clip=CLIP_WIDE, n_points=n)
                vals.append(lp - p * lm)
            incs.append(vals[1] - vals[0])
        vals = []
        for tau in (t1, t2):  # truncation sensitivity: even wider clip
            t = var_time(tau, ns)
            lp = ln_eexp(phi_p(t, s, rho, p), clip=CLIP_XWIDE, n_points=N_QUAD_FINE)
            lm = ln_eexp(phi_m(t, s, rho), clip=CLIP_XWIDE, n_points=N_QUAD_FINE)
            vals.append(lp - p * lm)
        inc_wide = vals[1] - vals[0]
        ok = (
            abs(incs[0] - incs[1]) < GATE
            and abs(incs[0] - inc_wide) < GATE
            and incs[0] < -TOL
            and incs[1] < -TOL
        )
        worst_gate = max(worst_gate, abs(incs[0] - incs[1]), abs(incs[0] - inc_wide))
        if ok:
            certified.append((rho, k, name, ns, p, t1, t2, incs[0]))
        lines.append(
            f"- rho={rho:+.4f} k={k:.2f} ns={name!r} p={p:.6f} tau {t1:.4f} -> {t2:.4f}: "
            f"d = {incs[0]:.6f} / {incs[1]:.6f} / {inc_wide:.6f} "
            f"{'PASS' if ok else 'FAIL'}"
        )
    lines += ["", f"- worst gate disagreement: {worst_gate:.2e}", ""]
    return lines, certified


def crosscheck(certified: list[Witness]) -> list[str]:
    """For certified witnesses, find the actual crl_call price drop over a K grid."""
    if not certified:
        return ["## crl_call cross-check", "", "- no certified witnesses", ""]
    kf = np.linspace(0.5, 2.0, 16)
    lines = [
        "## crl_call cross-check (K/F in [0.5, 2.0], F = D = 1)",
        "",
        "- crl_call's own 64-node GH misses the far-tail bump for rho > 0 (see the",
        "  msv.py docstring), so ~0 drops are expected here and say nothing about",
        "  the certificate above",
        "",
    ]
    for rho, k, _name, ns, _p, t1, t2, _d in certified[:5]:
        p1 = np.array([float(msv.crl_call(1.0, K, 1.0, t1, *ns, k, rho)[0]) for K in kf])
        p2 = np.array([float(msv.crl_call(1.0, K, 1.0, t2, *ns, k, rho)[0]) for K in kf])
        drop = p2 - p1
        i = int(drop.argmin())
        lines.append(
            f"- rho={rho:+.4f} k={k:.2f} ns={ns} tau {t1:.4f} -> {t2:.4f}: "
            f"worst drop {drop[i]:.3e} at K/F={kf[i]:.3f} (C = {p1[i]:.6f})"
        )
    lines.append("")
    return lines


def main() -> None:
    print(f"Moment route to calendar arbitrage, {date.today().isoformat()}")
    print(
        "all integrals: dense trapezoid on the weight-folded exponent; "
        f"tau grid {TAUS.size} pts in [{TAUS[0]}, {TAUS[-1]}]"
    )
    scan_lines, witnesses = scan()
    cert_lines, certified = certify(witnesses)
    lines = [
        "# Moment monotonicity of `crl_call` marginals (calendar-arbitrage certificate)",
        "",
        f"Run {date.today().isoformat()}. The collapsed exact moment E[S^p] (F = 1) is",
        "monotone in tau whenever the marginals are convex-ordered; a strict decrease",
        "proves calendar arbitrage (calendar_proof.md). All integrals by dense",
        "trapezoid on the weight-folded exponent; witnesses gated by resolution",
        "doubling and a wider-clip re-run.",
        "",
        *validate(),
        *scan_lines,
        *cert_lines,
        *crosscheck(certified),
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {MD}")


if __name__ == "__main__":
    main()
