"""Direct enclosure of dC/du(K) for rho <= 0 in the band (epic 0016, route 02-direct).

Usage: uv run python derivations/calendar_direct.py [-j N]   (writes calendar_direct.md)

F = D = 1, u = sqrt(tau Q), s^2 = ln(1+k^2), zeta = e^{sZ - s^2/2}, x = ln K.
M = e^{rho u sqrt(zeta) Z - rho^2 u^2 zeta/2}, m = E[M], A = M/m, h = d ln M/du,
mbar = m'/m = E[A h], w = (1-rho^2) u^2 zeta, sigma = sqrt((1-rho^2) zeta),
d1 = (ln A - x + w/2)/sqrt(w), d2 = d1 - sqrt(w). Then

    C(u, x) = E[A N(d1) - e^x N(d2)],
    G := dC/du = E[A (phi(d1) sigma + N(d1) (h - mbar))] = (I1 + I2 - (mh/m) I3) / m,

with I1 = E[M phi(d1) sigma], I2 = E[M N(d1) h], I3 = E[M N(d1)], mh = E[M h].
The enclosure integrates over Z in [-12, 12] with acb.integral and adds explicit tail
bounds for |Z| > 12, using M <= 1 for Z >= 0 and M <= exp(2|rho| u / (e s)) for Z < 0.
"""

import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any

import numpy as np
from flint import acb, arb
from numpy.typing import NDArray
from scipy.special import logsumexp, ndtr
from scipy.stats import norm

sys.path.insert(0, __file__.rsplit("/", 1)[0])

OUT = __file__.replace(".py", ".md")
ZMAX = 12.0
NZ = 24001
MAX_CELLS = 2000
Arr = NDArray[np.float64]


def _grid() -> tuple[Arr, Arr]:
    z = np.linspace(-ZMAX, ZMAX, NZ)
    wt = norm.pdf(z) * (z[1] - z[0])
    wt[0] *= 0.5
    wt[-1] *= 0.5
    return z, wt


def point(u: float, k: float, rho: float, x: float) -> tuple[float, float, float]:
    """(G, C, mbar) in float from the Z-grid sums."""
    z, wt = _grid()
    s = np.sqrt(np.log1p(k * k))
    sq = np.exp(s * z / 2 - s * s / 4)
    zeta = sq * sq
    log_m = rho * u * sq * z - rho * rho * u * u * zeta / 2
    h = rho * sq * z - rho * rho * u * zeta
    lm = float(logsumexp(np.log(wt) + log_m))
    a = np.exp(log_m - lm)
    mbar = float(np.sum(wt * a * h))
    w = (1 - rho * rho) * u * u * zeta
    sw = np.sqrt(w)
    d1 = ((log_m - lm) - x + w / 2) / sw
    d2 = d1 - sw
    sig = np.sqrt(1 - rho * rho) * sq
    g = float(np.sum(wt * a * (norm.pdf(d1) * sig + ndtr(d1) * (h - mbar))))
    c = float(np.sum(wt * (a * ndtr(d1) - np.exp(x) * ndtr(d2))))
    return g, c, mbar


def _sq(v: Any) -> Any:
    if isinstance(v, acb):
        return v * v
    lo, hi = v.lower(), v.upper()
    if lo > 0:
        return arb(lo * lo).union(hi * hi)
    if hi < 0:
        return arb(hi * hi).union(lo * lo)
    return arb(0).union(max(lo * lo, hi * hi))


def _phi(v: Any) -> Any:
    return (-_sq(v) / 2).exp() / (2 * arb.pi()).sqrt()


def _cdf(v: Any) -> Any:
    return (-v / arb(2).sqrt()).erfc() / 2


def _ccdf(v: arb) -> arb:
    return (v / arb(2).sqrt()).erfc() / 2


def _tail(kind: str, rho: arb, u: arb, s: arb) -> Any:
    """Bound on |int_{|Z| > ZMAX} phi(Z) M q(Z) dZ| for the integrand of `kind`."""
    c = arb(ZMAX)
    e_s = (-(s * s) / 8).exp()
    sqrt2pi = (2 * arb.pi()).sqrt()
    abs_rho = -rho
    cm = (2 * abs_rho * u / (arb.const_e() * s)).exp()
    # Z >= c: |h| <= |rho| sqrt(zeta) Z + rho^2 u zeta, sqrt(zeta) <= e^{sZ/2}
    h_up = abs_rho * e_s * (_phi(c - s / 2) + s / 2 * _ccdf(c - s / 2)) + (
        rho * rho * u * _ccdf(c - s)
    )
    sig_up = (1 - rho * rho).sqrt() / sqrt2pi * e_s * _ccdf(c - s / 2)
    # Z <= -c: sqrt(zeta) |Z| <= 2/(e s), zeta <= e^{-sc}, sqrt(zeta) <= e^{-sc/2}
    sc = s * c
    h_low = abs_rho * 2 / (arb.const_e() * s) + rho * rho * u * (-sc).exp()
    sig_low = (-sc / 2).exp() / sqrt2pi
    if kind == "m":
        return _ccdf(c) + cm * _cdf(-c)
    if kind == "mh":
        return h_up + cm * _cdf(-c) * h_low
    if kind == "I1":
        return sig_up + cm * _cdf(-c) * sig_low
    if kind == "I2":
        return h_up + cm * _cdf(-c) * h_low
    if kind == "I3":
        return _ccdf(c) + cm * _cdf(-c)
    raise ValueError(kind)


def _integrand(kind: str, z: Any, lm: arb, u: arb, rho: arb, s: arb, x: arb) -> Any:
    sq = (s * z / 2 - s * s / 4).exp()
    zeta = sq * sq
    log_m = rho * u * sq * z - rho * rho * u * u * zeta / 2
    mm = log_m.exp()
    h = rho * sq * z - rho * rho * u * zeta
    if kind == "m":
        return _phi(z) * mm
    if kind == "mh":
        return _phi(z) * mm * h
    w = (1 - rho * rho) * u * u * zeta
    d1 = ((log_m - lm) - x + w / 2) / (u * (1 - rho * rho).sqrt() * sq)
    if kind == "I1":
        return _phi(z) * mm * _phi(d1) * (1 - rho * rho).sqrt() * sq
    if kind == "I2":
        return _phi(z) * mm * _cdf(d1) * h
    if kind == "I3":
        return _phi(z) * mm * _cdf(d1)
    raise ValueError(kind)


def _enclose_int(kind: str, lm: arb, u: arb, rho: arb, s: arb, x: arb) -> Any:
    def f(z: acb, analytic: bool) -> acb:
        return acb(_integrand(kind, z if analytic else z.real, lm, u, rho, s, x))

    val = acb.integral(f, -ZMAX, ZMAX, rel_tol=1e-13, abs_tol=1e-15).real
    t = _tail(kind, rho, u, s)
    return val + (-t).union(t)


def enclose_g(u: arb, k: arb, rho: arb, x: arb) -> arb:
    """Rigorous enclosure of dC/du at (u, k, rho, x = ln K), F = D = 1."""
    s = (1 + k * k).log().sqrt()
    m = _enclose_int("m", arb(0), u, rho, s, x)
    lm = m.log()
    mh = _enclose_int("mh", lm, u, rho, s, x)
    i1 = _enclose_int("I1", lm, u, rho, s, x)
    i2 = _enclose_int("I2", lm, u, rho, s, x)
    i3 = _enclose_int("I3", lm, u, rho, s, x)
    g: arb = (i1 + i2 - mh / m * i3) / m
    return g


def _enclose_cell(cell: list[tuple[float, float]]) -> arb:
    return enclose_g(*(arb(lo).union(arb(hi)) for lo, hi in cell))


Box = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
Cell = tuple[Box, float, float, float, int]


def _iv(lo: float, hi: float) -> arb:
    return arb(lo).union(arb(hi))


def _s_arb(k: arb) -> arb:
    return (1 + k * k).log().sqrt()


def _root_lo(u_hi: float, s_lo: float, r_lo: float) -> float:
    """Upper bound of L'* over a box: root of g with the box-worst constants (g increasing)."""

    def g(lp: float) -> float:
        return math.log(2 * lp / u_hi**2) + s_lo**2 / 2 + s_lo * r_lo * math.sqrt(2 * lp)

    lo, hi = 1e-300, 1.0
    while g(hi) < 0:
        hi *= 2
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
    root = hi * (1 + 1e-9)
    s = arb(s_lo * (1 - 1e-12))  # below the true s_lo despite float rounding; g increases in s
    two_l = 2 * arb(root)
    g_arb = (two_l / arb(u_hi) ** 2).log() + s**2 / 2 + s * arb(r_lo) * two_l.sqrt()
    if not g_arb > 0:
        raise ArithmeticError("root bound of g not certified")
    return root


def _jensen_ln_m(box: Box) -> float:
    """Lower bound on ln m over the box, from ln m >= E[ln M] (Jensen)."""
    (ua, ub), (ka, kb), (ra, rb) = box
    s_lo, s_hi = math.sqrt(math.log1p(ka * ka)), math.sqrt(math.log1p(kb * kb))
    s_pts = [s_lo, s_hi] + ([2.0] if s_lo < 2.0 < s_hi else [])
    c_sup = max((s / 2) * math.exp(-(s * s) / 8) for s in s_pts)
    r_max = -ra
    return -r_max * ub * c_sup - (r_max * ub) ** 2 / 2


def x_upper(box: Box) -> float:
    """Upper bound on l2' = L'* - ln m over the box (ln m bounded below by Jensen and arb)."""
    (ua, ub), (ka, kb), (ra, rb) = box
    lnm = _jensen_ln_m(box)
    m = _enclose_int("m", arb(0), _iv(ua, ub), _iv(ra, rb), _s_arb(_iv(ka, kb)), arb(0))
    if m.is_finite() and float(m.lower()) > 0:
        lnm = max(lnm, float(m.log().lower()) - 1e-12)
    lp = _root_lo(ub, math.sqrt(math.log1p(ka * ka)), -rb)
    return lp - lnm + 1e-12


def certify_box(box: Box, max_depth: int = 40, max_cells: int = 20000) -> dict[str, Any]:
    """Adaptive bisection of box x [0, x_upper(box)] until G's enclosure is positive."""
    t0 = time.perf_counter()
    x0 = x_upper(box)
    res: dict[str, Any] = {"box": box, "x0": x0, "cells": 0, "bad": 0, "min_lo": float("inf")}
    if x0 <= 0:
        res["time"] = time.perf_counter() - t0
        return res
    roots = [hi - lo for lo, hi in box] + [x0]
    stack: list[Cell] = [(box, x0, 0.0, x0, 0)]
    while stack:
        if res["cells"] >= max_cells:
            res["bad"] += len(stack)
            break
        b, xcap, xa, xb, d = stack.pop()
        g = enclose_g(_iv(*b[0]), _iv(*b[1]), _iv(*b[2]), _iv(xa, xb))
        res["cells"] += 1
        lo = float(g.lower())
        if lo > 0:
            res["min_lo"] = min(res["min_lo"], lo)
            continue
        if d >= max_depth:
            res["bad"] += 1
            continue
        rel = [(hi - a) / r for (a, hi), r in zip(b, roots[:3], strict=True)] + [(xb - xa) / x0]
        j = max(range(4), key=lambda i: rel[i])
        if j == 3:
            mid = (xa + xb) / 2
            stack += [(b, xcap, xa, mid, d + 1), (b, xcap, mid, xb, d + 1)]
            continue
        a, c = b[j]
        mid = (a + c) / 2
        for half in ((a, mid), (mid, c)):
            sides = [b[0], b[1], b[2]]
            sides[j] = half
            child: Box = (sides[0], sides[1], sides[2])
            xc = min(xcap, x_upper(child))
            xbc = min(xb, xc)
            if xa < xbc:
                stack.append((child, xc, xa, xbc, d + 1))
    res["time"] = time.perf_counter() - t0
    return res


def tiles(lo: float, hi: float, n: int, log: bool) -> list[tuple[float, float]]:
    edges = np.geomspace(lo, hi, n + 1) if log else np.linspace(lo, hi, n + 1)
    return [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


R1_BOXES: list[Box] = [
    (bu, bk, br)
    for bu in tiles(0.1, 1.0, 3, log=True)
    for bk in tiles(0.3, 2.0, 3, log=True)
    for br in tiles(-0.9, -0.1, 4, log=False)
]


def _row(r: dict[str, Any]) -> str:
    (ua, ub), (ka, kb), (ra, rb) = r["box"]
    return (
        f"| [{ua:.4g}, {ub:.4g}] | [{ka:.4g}, {kb:.4g}] | [{ra:.3g}, {rb:.3g}] | "
        f"{r['x0']:.5f} | {r['cells']} | {r['bad']} | {r['min_lo']:.4g} | {r['time']:.2f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", type=int, default=8, help="worker processes")
    parser.add_argument("--boxes", type=int, default=0, help="run only the first N R1 boxes")
    args = parser.parse_args()
    boxes = R1_BOXES[: args.boxes] if args.boxes else R1_BOXES
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.j) as pool:
        rows = []
        for r in pool.map(partial(certify_box, max_cells=MAX_CELLS), boxes):
            rows.append(r)
            print(
                f"{len(rows)}/{len(boxes)} cells={r['cells']} bad={r['bad']} "
                f"min_lo={r['min_lo']:.4g} t={r['time']:.1f}s",
                flush=True,
            )
    wall = time.perf_counter() - t0
    cells = sum(r["cells"] for r in rows)
    bad = sum(r["bad"] for r in rows)
    ok = sum(r["bad"] == 0 for r in rows)
    min_lo = min(r["min_lo"] for r in rows)
    cpu = sum(r["time"] for r in rows)
    x0s = [r["x0"] for r in rows]
    lines = [
        f"Output of `uv run python derivations/calendar_direct.py -j {args.j}"
        + (f" --boxes {args.boxes}`" if args.boxes else "`")
        + f"; boxes run: {len(boxes)} of {len(R1_BOXES)}.",
        "",
        "## Certified region R1",
        "",
        "- Region: u in [0.1, 1], k in [0.3, 2], rho in [-0.9, -0.1]; split into "
        f"{len(R1_BOXES)} boxes (3 log-tiles in u, 3 in k, 4 linear in rho). On each box "
        "the claim is G > 0 for ln K in [0, X(box)], where X is the upper bound on l2' from "
        "`x_upper` (ln K <= l1 is Theorem A, ln K >= l2' is Theorem B', so this covers the band).",
        f"- Boxes certified (no unresolved cell): {ok} of {len(R1_BOXES)}.",
        f"- Cells: {cells}; unresolved at max depth or cell cap: {bad}.",
        f"- Smallest certified lower bound of G: {min_lo:.6g}.",
        f"- X over the boxes: [{min(x0s):.5f}, {max(x0s):.5f}].",
        f"- Wall time {wall:.1f} s with -j {args.j} (summed box times {cpu:.1f} s).",
        "- Label: PROVED (computer-assisted) on each box with no unresolved cell; the pieces "
        "are the formula (section 1), the closed-form tails (section 3), the arb cell "
        "enclosures, and x_upper. x_upper takes the root of g in floating point and adds a "
        "1e-9 relative margin; g > 0 at that point is then checked in arb with s shrunk below the "
        "float s_lo (g increases in s), so the bound is rigorous.",
        "",
        "| u | k | rho | X | cells | unresolved | min lower bound | time (s) |",
        "|---|---|---|---|---|---|---|---|",
        *(_row(r) for r in rows),
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
    print("\n".join(lines))


if __name__ == "__main__":
    main()
