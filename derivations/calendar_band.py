"""Calendar no-arbitrage for rho <= 0 outside a near-the-money band (epic 0016).

Usage: uv run python derivations/calendar_band.py [-j N]   (writes calendar_band.md)

F = D = 1, u = sqrt(tau Q(tau)), L = ln x, y = u^2 zeta(Z), w = (1-rho^2) y,
d2 = (rho sqrt(y) Z - y/2 - ln m - L) / sqrt(w). With p(x) = dP(S > x)/du,

    u p(x) = E_Z[phi(d2) (Lt(x) - y/2) / sqrt(w)],   Lt(x) = ln x + ln m - u m'/m,

so p(x) <= 0 wherever Lt(x) <= 0 (left threshold l1 = -ln m + u m'/m), and the symmetric
pairing in calendar_band.md gives p(x) >= 0 for ln x >= l2 = u^2 e^{-s^2/2} / 2 - ln m when
rho <= 0. Since the integral of p over x is d E[S]/du = 0, dC/du(K) = int_K^inf p >= 0 at
every K if p changes sign once (Karlin-Novikoff cut criterion). The script checks the
identity against finite differences of P, the signs of p beyond l1 and l2, and counts the
sign changes of p on a dense log-strike grid. Exploration, not a regression test.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.special import log_ndtr, logsumexp
from scipy.stats import norm

OUT = __file__.replace(".py", ".md")
RHOS = (-0.01, -0.2, -0.49, -0.7, -0.9, -0.99)
KS = (0.05, 0.3, 1.0, 2.2, 3.0, 6.0)
US = (0.02, 0.1, 0.3, 1.0, 2.0, 5.0)
STRESS = ((5.0, 10.0, 20.0, 50.0), (1.0, 3.0, 6.0, 10.0), (-1e-6, -0.01, -0.2, -0.49, -0.9, -0.99))
N_X = 400
REL_TOL = 1e-9
Arr = NDArray[np.float64]


class Model:
    def __init__(self, u: float, k: float, rho: float) -> None:
        self.u, self.rho = u, rho
        self.s = float(np.sqrt(np.log1p(k * k)))
        z_hi = max(12.0, (np.log(50.0 / u**2) + self.s**2 / 2) / self.s + 12.0)
        self.z = np.arange(-30.0, z_hi, 5e-4)
        self.lw = norm.logpdf(self.z) + np.log(5e-4)
        self.lm, self.mm = self.normaliser(u)

    def normaliser(self, u: float) -> tuple[float, float]:
        zeta = np.exp(self.s * self.z - self.s**2 / 2)
        y = u * u * zeta
        lmz = self.rho * np.sqrt(y) * self.z - self.rho**2 * y / 2
        lm = float(logsumexp(self.lw + lmz))
        h = self.rho * np.sqrt(zeta) * self.z - self.rho**2 * u * zeta
        return lm, float(np.sum(np.exp(self.lw + lmz - lm) * h))

    def d2(self, u: float, lm: float, big_l: float) -> tuple[Arr, Arr]:
        y = u * u * np.exp(self.s * self.z - self.s**2 / 2)
        w = (1 - self.rho**2) * y
        return (self.rho * np.sqrt(y) * self.z - y / 2 - lm - big_l) / np.sqrt(w), y

    def log_prob(self, u: float, big_l: float) -> float:
        lm, _ = self.normaliser(u)
        d2, _ = self.d2(u, lm, big_l)
        return float(logsumexp(self.lw + log_ndtr(d2)))

    def p(self, big_l: float) -> tuple[float, float]:
        """(log |u p(x)|, sign) at L = ln x."""
        d2, y = self.d2(self.u, self.lm, big_l)
        lt = big_l + self.lm - self.u * self.mm
        val, sign = logsumexp(
            self.lw + norm.logpdf(d2) - 0.5 * np.log((1 - self.rho**2) * y),
            b=lt - y / 2,
            return_sign=True,
        )
        return float(val), float(sign)

    def thresholds(self) -> tuple[float, float]:
        l1 = -self.lm + self.u * self.mm
        l2 = self.u**2 * np.exp(-(self.s**2) / 2) / 2 - self.lm
        return l1, float(l2)

    def l2_sharp(self) -> float:
        """Root in L' of ln(2L'/u^2) + s^2/2 + s |rho| sqrt(2L') (increasing), minus ln m."""

        def g(lp: float) -> float:
            return float(
                np.log(2 * lp / self.u**2)
                + self.s**2 / 2
                + self.s * abs(self.rho) * np.sqrt(2 * lp)
            )

        lo, hi = 1e-300, self.u**2 * np.exp(-(self.s**2) / 2) / 2
        for _ in range(200):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
        return float(hi) - self.lm

    def r_slope(self, big_l: float) -> float:
        """R'(L) = Cov_nu(y/2, d2/sqrt(w)), nu ~ phi(Z) phi(d2)/sqrt(w)."""
        d2, y = self.d2(self.u, self.lm, big_l)
        lw = self.lw + norm.logpdf(d2) - 0.5 * np.log((1 - self.rho**2) * y)
        nu = np.exp(lw - logsumexp(lw))
        g = d2 / np.sqrt((1 - self.rho**2) * y)
        return float(np.sum(nu * y / 2 * g) - np.sum(nu * y / 2) * np.sum(nu * g))

    def split(self, big_l: float) -> tuple[float, float]:
        """(first, cov) with R' (1-rho^2) = first - |rho| cov / 2, first = L'/2 (E y E 1/y - 1)."""
        d2, y = self.d2(self.u, self.lm, big_l)
        lw = self.lw + norm.logpdf(d2) - 0.5 * np.log(y)
        nu = np.exp(lw - logsumexp(lw))
        ey, h = float(np.sum(nu * y)), self.z / np.sqrt(y)
        first = (big_l + self.lm) / 2 * (ey * float(np.sum(nu / y)) - 1)
        return first, float(np.sum(nu * y * h)) - ey * float(np.sum(nu * h))

    def curvature(self, big_l: float) -> float:
        """min s^2 V''(t) on the support of nu; t = ln y, V = Z^2/2 + t/2 + q^2/(2(1-rho^2))."""
        d2, y = self.d2(self.u, self.lm, big_l)
        lw = self.lw + norm.logpdf(d2) - 0.5 * np.log(y)
        r, lp, sy = -self.rho, big_l + self.lm, np.sqrt(y)
        q = lp / sy + sy / 2 + r * self.z
        qp = -lp / (2 * sy) + sy / 4 + r / self.s
        qpp = lp / (4 * sy) + sy / 8
        vpp = 1 + self.s**2 * (qp**2 + q * qpp) / (1 - r * r)
        return float(vpp[lw > lw.max() + np.log(1e-12)].min())


def check(p: tuple[float, float, float]) -> dict[str, float]:
    u, k, rho = p
    mod = Model(u, k, rho)
    l1, l2 = mod.thresholds()
    half = max(l2 - l1, 0.05)
    xs = np.linspace(l1 - 10 * half - 2.0, l2 + 10 * half + 2.0, N_X)
    vals = [mod.p(x) for x in xs]
    logs = np.array([v[0] for v in vals])
    signs = np.array([v[1] for v in vals])
    big = logs > logs.max() + np.log(REL_TOL)
    s_big = signs[big]
    changes = int(np.sum(s_big[1:] != s_big[:-1]))
    left_bad = int(np.sum((xs <= l1) & (signs > 0) & big))
    right_bad = int(np.sum((xs >= l2) & (signs < 0) & big))
    l2s = mod.l2_sharp()
    max_slope = max(mod.r_slope(x) for x in np.linspace(l1, l2s, 60))
    x_mid = 0.5 * (l1 + l2)
    hh = 1e-5 * u
    fd = (np.exp(mod.log_prob(u + hh, x_mid)) - np.exp(mod.log_prob(u - hh, x_mid))) / (2 * hh)
    lp, sp = mod.p(x_mid)
    formula = sp * np.exp(lp) / u
    return {
        "u": u,
        "k": k,
        "rho": rho,
        "l1": l1,
        "l2": l2,
        "l2s": l2s,
        "max_slope": max_slope,
        "curv": min(mod.curvature(x) for x in np.linspace(l1, l2s, 20)),
        "first": max(mod.split(x)[0] for x in np.linspace(l1, l2s, 20)) / (1 - rho**2),
        "cov": min(mod.split(x)[1] for x in np.linspace(l1, l2s, 20)),
        "changes": changes,
        "left_bad": left_bad,
        "right_bad": right_bad,
        "fd_err": abs(formula - fd) / max(abs(fd), 1e-300),
    }


def stress(p: tuple[float, float, float]) -> tuple[float, ...]:
    mod = Model(*p)
    band = np.linspace(mod.thresholds()[0], mod.l2_sharp(), 60)
    parts = [mod.split(x) for x in band[::3]]
    return (
        p[0],
        p[1],
        p[2],
        max(mod.r_slope(x) for x in band),
        min(mod.curvature(x) for x in band[::3]),
        max(a for a, _ in parts) / (1 - p[2] ** 2),
        min(c for _, c in parts),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", type=int, default=4, help="worker processes")
    args = parser.parse_args()
    grid = list(product(US, KS, RHOS))
    with ProcessPoolExecutor(args.j) as pool:
        rows = list(pool.map(check, grid))
        hard = list(pool.map(stress, product(*STRESS)))
    multi = [r for r in rows if r["changes"] != 1]
    lines = [
        "# Sign structure of dP(S > x)/du for rho <= 0",
        "",
        "Output of `uv run python derivations/calendar_band.py`; formulas in the module "
        f"docstring. {len(rows)} parameter points (u in {US}, k in {KS}, rho in {RHOS}); "
        f"p(x) on {N_X} log-strikes spanning the band [l1, l2] and ten band-widths (at least "
        f"2) either side; signs counted where |p| exceeds {REL_TOL:g} of its maximum.",
        "",
        f"- Points where p changes sign other than exactly once: {len(multi)}.",
        f"- Log-strikes <= l1 with p > 0: {sum(r['left_bad'] for r in rows)}; "
        f">= l2 with p < 0: {sum(r['right_bad'] for r in rows)}.",
        f"- Formula vs finite difference of P at the band centre: max relative error "
        f"{max(r['fd_err'] for r in rows):.2e}.",
        f"- Band width l2 - l1: from {min(r['l2'] - r['l1'] for r in rows):.2e} to "
        f"{max(r['l2'] - r['l1'] for r in rows):.3f}; l1 from {min(r['l1'] for r in rows):.4f} "
        f"to {max(r['l1'] for r in rows):.4f}, l2 from {min(r['l2'] for r in rows):.4f} to "
        f"{max(r['l2'] for r in rows):.4f}.",
        f"- Sharp threshold l2' (calendar_band.md): band shrinks by a fraction "
        f"{min((r['l2'] - r['l2s']) / (r['l2'] - r['l1']) for r in rows):.2e} to "
        f"{max((r['l2'] - r['l2s']) / (r['l2'] - r['l1']) for r in rows):.3f} of l2 - l1.",
        f"- Largest R'(L) on [l1, l2'] (60 points per parameter point): "
        f"{max(r['max_slope'] for r in rows):.4f}; points with R' >= 1: "
        f"{sum(r['max_slope'] >= 1 for r in rows)}.",
        "- Smallest s^2 V'' (log-concavity of nu in t = ln y; 1 = prior curvature) on [l1, l2'] "
        "by rho: "
        + ", ".join(f"{rho}: {min(r['curv'] for r in rows if r['rho'] == rho):.2f}" for rho in RHOS)
        + ".",
        "- Split R' = (first - |rho| cov / 2) / (1 - rho^2) on [l1, l2'] (20 points): smallest "
        f"cov = Cov(y, Z/sqrt y) {min(r['cov'] for r in rows):.3g}; largest first/(1 - rho^2) "
        "by rho: "
        + ", ".join(
            f"{rho}: {max(r['first'] for r in rows if r['rho'] == rho):.3f}" for rho in RHOS
        )
        + ".",
        "",
        "| u | k | rho | l1 | l2 | l2' | max R' | sign changes |",
        "|---|---|---|---|---|---|---|---|",
        *(
            f"| {r['u']} | {r['k']} | {r['rho']} | {r['l1']:.4f} | {r['l2']:.4f} | "
            f"{r['l2s']:.4f} | "
            f"{r['max_slope']:.3f} | {r['changes']} |"
            for r in rows
        ),
        "",
        f"Stress grid (u in {STRESS[0]}, k in {STRESS[1]}, rho in {STRESS[2]}): max R', "
        "smallest s^2 V'', largest first/(1 - rho^2) and smallest cov on [l1, l2'].",
        "",
        "| u | k | rho | max R' | min s^2 V'' | max first/(1-rho^2) | min cov |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {u} | {k} | {rho:g} | {v:.4f} | {c:.2f} | {a:.4f} | {cv:.3g} |"
            for u, k, rho, v, c, a, cv in hard
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
    print(f"Wrote {OUT}; {len(multi)} points without a single sign change")


if __name__ == "__main__":
    main()
