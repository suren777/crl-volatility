"""CRL model: ATM-skew term structure and calendar no-arbitrage check.

The correlated random-level model (epics/0010) is priced exactly by msv.crl_call.
This script measures the two properties the paper needs:

1. ATM skew dIV/dln(K/F) at ln(K/F)=0 (central difference, +/-0.01) for taus
   1/52..2 under flat Q = 0.04, for four (k, rho) pairs, plus the log-log slope
   alpha of |skew| vs tau over [1/12, 2] (paper expects ~ -0.5, the rough-vol
   benchmark tau^(H-1/2) with H=0).
2. Calendar monotonicity: undiscounted call prices C(K, tau) (F constant) must
   be non-decreasing in tau at every K, checked on a grid of k, rho, three
   Nelson-Siegel curves, K/F in [0.6, 1.6], tau in [0.02, 3]. This is numerical
   evidence, not a proof.

Run: uv run python experiments/crl_skew.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.stats import norm  # noqa: E402

import msv  # noqa: E402

F = 100.0
H = 0.01  # central-difference half-width in ln(K/F)
TAUS = (1 / 52, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0, 2.0)
TAU_LABELS = ("1/52", "1/12", "2/12", "3/12", "6/12", "1", "2")
PAIRS = ((0.5, -0.7), (1.0, -0.7), (1.0, -0.9), (2.0, -0.9))
FLAT_NS = (0.0, 0.0, 0.04, 1.0)  # a0, a1, a2, lam: Q(tau) = 0.04 for all tau
NS_CURVES = {
    "flat 0.04": FLAT_NS,
    "decreasing short-end": (0.03, 0.0, 0.03, 2.0),
    "humped": (0.0, 0.06, 0.03, 1.5),
}
KS = (0.25, 0.5, 1.0, 2.0)
RHOS = (-0.9, -0.5, 0.0, 0.5)
KF = np.linspace(0.6, 1.6, 11)
TAUS_CAL = np.linspace(0.02, 3.0, 30)


def black_fwd(F: float, K: float, tau: float, sigma: float) -> float:
    v = sigma * sigma * tau
    sd = np.sqrt(v)
    d1 = (np.log(F / K) + v / 2.0) / sd
    return float(F * norm.cdf(d1) - K * norm.cdf(d1 - sd))


def crl_iv(ln_m: float, tau: float, k: float, rho: float) -> float:
    """Implied vol of the CRL call at ln(K/F)=ln_m, flat Q, D=1."""
    K = F * float(np.exp(ln_m))
    price = float(msv.crl_call(F, K, 1.0, tau, *FLAT_NS, k, rho)[0])
    return float(brentq(lambda s: black_fwd(F, K, tau, s) - price, 1e-8, 5.0, xtol=1e-13))


def atm_skew(tau: float, k: float, rho: float) -> float:
    """dIV/dln(K/F) at ln(K/F)=0 by central difference +/-H."""
    return (crl_iv(H, tau, k, rho) - crl_iv(-H, tau, k, rho)) / (2.0 * H)


def skew_table() -> None:
    print(
        f"ATM skew term structure: F={F}, D=1, flat Q=0.04 (a0=a1=0, a2=0.04, lam=1); "
        f"skew = [IV(+{H}) - IV(-{H})] / {2 * H} at ln(K/F)=0"
    )
    print("| (k, rho) | " + " | ".join(TAU_LABELS) + " | alpha (log-log fit, tau 1/12..2) |")
    print("|" + "---|" * (len(TAU_LABELS) + 2))
    for k, rho in PAIRS:
        skews = [atm_skew(t, k, rho) for t in TAUS]
        fit_t = np.log(np.asarray(TAUS[1:]))  # drop 1/52, fit over [1/12, 2]
        fit_s = np.log(np.abs(np.asarray(skews[1:])))
        alpha = float(np.polyfit(fit_t, fit_s, 1)[0])
        cells = " | ".join(f"{s:.3f}" for s in skews)
        print(f"| ({k}, {rho}) | {cells} | {alpha:.3f} |")


def calendar_table() -> None:
    n_incr = KF.size * (TAUS_CAL.size - 1)
    print(
        "\nCalendar monotonicity: undiscounted C(K, tau) (D=1) with F=1 constant, "
        f"K/F in [{KF[0]}, {KF[-1]}] ({KF.size} steps), tau in "
        f"[{TAUS_CAL[0]}, {TAUS_CAL[-1]}] ({TAUS_CAL.size} points); violations = negative "
        f"increments of {n_incr} per configuration; min increment located at "
        "(K/F, lower tau of the pair); min/C shown only when the min increment is negative"
    )
    for name, ns in NS_CURVES.items():
        print(f"\n## Q: {name} (a0, a1, a2, lam) = {ns}")
        print("| k | rho | violations | min increment | min / C | at K/F | at tau |")
        print("|---|-----|------------|---------------|---------|--------|--------|")
        for k in KS:
            for rho in RHOS:
                px = np.array([msv.crl_call(1.0, KF, 1.0, float(t), *ns, k, rho) for t in TAUS_CAL])
                d = np.diff(px, axis=0)  # (n_taus-1, n_strikes)
                i, j = (int(v) for v in np.unravel_index(int(np.argmin(d)), d.shape))
                min_incr = float(d[i, j])
                ratio = f"{min_incr / float(px[i, j]):.2e}" if min_incr < 0.0 else "n/a"
                print(
                    f"| {k} | {rho} | {int((d < 0.0).sum())} | {min_incr:.3e} "
                    f"| {ratio} | {KF[j]:.1f} | {TAUS_CAL[i]:.3f} |"
                )


def main() -> None:
    skew_table()
    calendar_table()


if __name__ == "__main__":
    main()
