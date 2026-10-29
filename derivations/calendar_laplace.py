"""Large-L' expansion of R'(L) = Cov_nu(y, G) for rho <= 0 (epic 0016, task 01).

Usage: uv run python derivations/calendar_laplace.py   (writes calendar_laplace.md)

With L' = L + ln m, lambda = L', mu = sqrt(2 lambda), c0 = ln(2 lambda/u^2) + s^2/2 and
v = ln y - ln(2 lambda), the conditional law is nu proportional to exp(-V(v)) with
V = (v + c0)^2 / (2 s^2) + v/2 + q^2 / (2 (1 - r^2)), q = mu cosh(v/2) + r (v + c0) / s.
The predicted gap 1 - R' is 2 / (s^2 lambda) + O(lambda^-2) at r = 0 and
sqrt(2) r / (s sqrt(lambda)) + O(lambda^-1) at r > 0. The scan compares the predictions
with Model.r_slope and fits the decay exponent of the error.
"""

import argparse
from itertools import product

import numpy as np
from calendar_band import Model

OUT = __file__.replace(".py", ".md")
US = (5.0, 10.0, 20.0, 50.0, 100.0)
KS = (0.05, 0.3, 1.0, 3.0, 6.0, 10.0)
RHOS = (0.0, -0.2, -0.7)
FRACS = (1.0, 0.5, 0.2)
ASYMPTOTIC_MIN = 20.0


def gap_leading(lp: float, s: float, rho: float) -> float:
    """Leading term of 1 - R' at L' = lp -> infinity with c0 = ln(frac) fixed."""
    if rho == 0.0:
        return 2.0 / (s * s * lp)
    return float(np.sqrt(2.0) * abs(rho) / (s * np.sqrt(lp)))


def point(u: float, k: float, rho: float, frac: float) -> tuple[float, float, float, float]:
    """(lambda, gap, predicted gap, error) at L' = frac e^{-s^2/2} u^2 / 2, so c0 = ln frac."""
    mod = Model(u, k, rho)
    s = mod.s
    lp = frac * np.exp(-s * s / 2) / 2 * u * u
    gap = 1.0 - mod.r_slope(lp - mod.lm)
    pred = gap_leading(lp, s, rho)
    return lp, gap, pred, gap - pred


def decay_exponent(lams: list[float], errs: list[float]) -> float:
    slope, _ = np.polyfit(np.log(lams), np.log(np.abs(errs)), 1)
    return float(slope)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    rows = []
    for rho, k, frac in product(RHOS, KS, FRACS):
        pts = [point(u, k, rho, frac) for u in US]
        rows.append((rho, k, frac, pts))
    table: list[str] = [
        "| rho | k | frac | u | L' | 1 - R' | predicted | err | err L'^p | p |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    slopes_zero: list[float] = []
    slopes_neg: list[float] = []
    for rho, k, frac, pts in rows:
        p = 2 if rho == 0.0 else 1
        s2 = np.log1p(k * k)
        tail = [pt for pt, u in zip(pts, US, strict=True) if u >= 20]
        if all(s2 * pt[0] >= ASYMPTOTIC_MIN for pt in tail):
            slope = decay_exponent([pt[0] for pt in tail], [pt[3] for pt in tail])
            (slopes_zero if rho == 0.0 else slopes_neg).append(slope)
            verdict = f"{slope:.3f}"
        else:
            verdict = f"n/a (s^2 L' < {ASYMPTOTIC_MIN} at u = 20)"
        for u, (lp, gap, pred, err) in zip(US, pts, strict=True):
            table.append(
                f"| {rho:g} | {k:g} | {frac:g} | {u:g} | {lp:.4g} | {gap:.6g} | {pred:.6g} | "
                f"{err:.4g} | {err * lp**p:.4g} | {p} |"
            )
        table.append(f"| {rho:g} | {k:g} | {frac:g} | slope (u >= 20) | | | | | | {verdict} |")
    lines = [
        "# Large-L' expansion of R'(L) for rho <= 0",
        "",
        "Output of `uv run python derivations/calendar_laplace.py`. Predicted gap: "
        "2/(s^2 L') at rho = 0 and sqrt(2)|rho|/(s sqrt(L')) at rho < 0 (c0 = ln frac, "
        f"u in {US}, k in {KS}, rho in {RHOS}, frac in {FRACS}).",
        "",
        "Columns: err = (1 - R') - predicted; the slope rows give the least-squares exponent "
        "of |err| against L' over u in (20, 50, 100), fitted only where s^2 L' >= "
        f"{ASYMPTOTIC_MIN:g} (the expansion parameter is 1/(s^2 L')). The predicted decay is "
        "L'^-2 at rho = 0 and L'^-1 at rho < 0.",
        "",
        f"- rho = 0: decay exponent of |err| over all (k, frac): min {min(slopes_zero):.3f}, "
        f"max {max(slopes_zero):.3f} (predicted -2).",
        f"- rho < 0: decay exponent of |err| over all (rho, k, frac): min "
        f"{min(slopes_neg):.3f}, max {max(slopes_neg):.3f} (predicted -1).",
        "",
        *table,
    ]
    try:
        with open(OUT) as f:
            old = f.read()
    except FileNotFoundError:
        old = ""
    head = old.split("<!-- scan -->")[0] if "<!-- scan -->" in old else ""
    with open(OUT, "w") as f:
        f.write(head + "<!-- scan -->\n" + "\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
