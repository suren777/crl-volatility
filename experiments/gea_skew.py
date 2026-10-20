"""ATM skews (market and CRL) with the Guyon-El Amrani estimator (epics/0010).

Usage: uv run python experiments/gea_skew.py

Guyon & El Amrani (*Does the Term-Structure of Equity At-the-Money Skew Really
Follow a Power Law?*, 2022/23) define the ATM skew S(T) = |d sigma_BS / d ln K|
at K = F_T "by fitting a polynomial of degree 2 locally around the money, using
the 6 closest strikes to the left and the 6 closest strikes to the right of the
T-forward". Our previous notes (`crl_short.md`) used a linear regression over
|ln K/F| <= 0.05 (`atm_skew`). This script recomputes both market and CRL-model
skews with the G&EA degree-2/12-strike estimator so the numbers are comparable
to their SPX exponents 0.19 below / 0.43 above 3 weeks and their extrapolated
zero-maturity skew of about 1.5 (decimal).

Ten expiries from the Cboe 2026-10-16 snapshot (four weekly SPXW Fridays
2026-10-23..-11-13 plus the six SPX monthlies), each expiry loaded from its own
root, exactly as in `crl_short.py` (whose loader, CRL fit bounded to
rho in [-0.99, 0], and IV pipeline are imported). CRL is fitted jointly on all
ten expiries (crl_call is reliable only for rho <= 0). For each of market and
fitted CRL we report the skew per expiry under BOTH estimators, then fit the
pure power law |S| ~ A*tau^alpha (Guyon-El Amrani's shape) over all ten, over
tau >= 3 weeks and over tau < 3 weeks, and compare with their 0.19 / 0.43 and
~1.5.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from crl_short import MONTHLIES, ROOTS, WEEKLIES, fit_crl_nonpos, load_raw  # noqa: E402
from crl_spx import CBOE_DB, TRADE_DATE, crl_model_ivs, loglog_alpha  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from shifted_spx import (  # noqa: E402
    ASOF,
    SKEW_LN_MONEYNESS,
    atm_skew,
    load_expiry,
)

THREE_WEEKS = 21.0 / 365.0  # 21 days = 3 weeks: Guyon-El Amrani's kink
N_SIDE = 6
DEG = 2


def gea_skew(x: NDArray[np.float64], iv: NDArray[np.float64], n: int = N_SIDE) -> float:
    """Guyon-El Amrani ATM skew: fit degree-2 poly through the n closest strikes on
    each side of the T-forward, return the derivative d sigma_BS/d ln K at x = 0
    (= the linear coefficient, negative for SPX)."""
    xs = np.asarray(x, dtype=float)
    ivs = np.asarray(iv, dtype=float)
    left = np.flatnonzero(xs < 0)
    right = np.flatnonzero(xs > 0)
    if left.size == 0 or right.size == 0:
        raise ValueError("no strikes on both sides of the forward")
    left = left[np.argsort(xs[left])][-n:]  # n largest negatives = closest left
    right = right[np.argsort(xs[right])][:n]  # n smallest positives = closest right
    idx = np.concatenate([left, right])
    if idx.size < DEG + 1:
        raise ValueError("not enough strikes around the forward for a degree-2 fit")
    coef = np.polyfit(xs[idx], ivs[idx], DEG)  # highest degree first
    return float(coef[1])


def main() -> None:
    raw_spx, spot = load_raw("SPX")
    raw_spxw, spot_w = load_raw("SPXW")
    data = [load_expiry(raw_spxw, e, spot_w) for e in WEEKLIES] + [
        load_expiry(raw_spx, e, spot) for e in MONTHLIES
    ]
    print(
        f"Cboe SQLite {CBOE_DB}  trade_date={TRADE_DATE}  spot={spot:.2f}  "
        f"asof={ASOF}  (weeklies root SPXW, monthlies root SPX)"
    )
    print(
        f"ATM skew estimators: GEA = degree-{DEG} poly, {N_SIDE} closest strikes per side "
        f"of the forward (Guyon-El Amrani); linear = regression over |ln K/F| <= "
        f"{SKEW_LN_MONEYNESS}."
    )

    print("\nPer-expiry parity forward/discount and OTM fit set (|ln(K/F)|<=0.3):")
    print(f"{'expiry':<12} {'root':<5} {'tau':>7} {'days':>5} {'F':>10} {'D':>8} {'n_options':>10}")
    taus = np.array([d.tau for d in data])
    days = np.array([round(d.tau * 365) for d in data])
    for d, root in zip(data, ROOTS, strict=True):
        print(
            f"{d.expiry:<12} {root:<5} {d.tau:>7.4f} {round(d.tau * 365):>5d} {d.F:>10.2f} "
            f"{d.D:>8.5f} {d.K.size:>10d}"
        )

    s_mkt_lin = np.array([100.0 * atm_skew(np.log(d.K / d.F), d.iv_mkt) for d in data])
    s_mkt_gea = np.array([100.0 * gea_skew(np.log(d.K / d.F), d.iv_mkt) for d in data])

    print("\nMarket ATM skew (vol pts per log-moneyness), both estimators:")
    print(
        f"{'expiry':<12} {'root':<5} {'tau':>7} {'days':>5} "
        f"{'linear':>9} {'GEA':>9} {'GEA/mkt lin':>12}"
    )
    for d, root, s_lin, s_gea, dy in zip(data, ROOTS, s_mkt_lin, s_mkt_gea, days, strict=True):
        print(
            f"{d.expiry:<12} {root:<5} {d.tau:>7.4f} {dy:>5d} "
            f"{s_lin:>9.3f} {s_gea:>9.3f} {s_gea / s_lin:>12.3f}"
        )

    res, t_fit = fit_crl_nonpos(data)
    x = res.x
    print(
        f"\nCRL fitted on all ten: a0={x[0]:.5f} a1={x[1]:.5f} a2={x[2]:.5f} "
        f"lam={x[3]:.4f} k={x[4]:.4f} rho={x[5]:.3f}  cost={res.cost:.4f}  [{t_fit:.1f}s]"
    )

    ivs_crl = crl_model_ivs(res, data)
    s_crl_lin = np.array(
        [100.0 * atm_skew(np.log(d.K / d.F), iv) for d, iv in zip(data, ivs_crl, strict=True)]
    )
    s_crl_gea = np.array(
        [100.0 * gea_skew(np.log(d.K / d.F), iv) for d, iv in zip(data, ivs_crl, strict=True)]
    )

    print("\nPer-expiry ATM skew (vol pts per log-moneyness): both estimators, market vs CRL")
    print(
        f"{'expiry':<12} {'days':>5} {'mkt lin':>9} {'mkt GEA':>9} {'CRL lin':>9} "
        f"{'CRL GEA':>9} {'CRL GEA/mkt GEA':>16}"
    )
    for d, dy, ml, mg, cl, cg in zip(
        data, days, s_mkt_lin, s_mkt_gea, s_crl_lin, s_crl_gea, strict=True
    ):
        print(
            f"{d.expiry:<12} {dy:>5d} {ml:>9.3f} {mg:>9.3f} {cl:>9.3f} {cg:>9.3f} {cg / mg:>16.3f}"
        )

    def powers(name: str, skews: NDArray[np.float64]) -> None:
        below = taus < THREE_WEEKS
        above = ~below
        print(f"\n{name}: ATM-skew power-law exponent (|S| ~ A*tau^alpha), GEA estimator:")
        print(
            f"  all ten      {loglog_alpha(taus, skews):>7.3f}   "
            f"(fit offset ln A = {np.polyfit(np.log(taus), np.log(np.abs(skews)), 1)[1]:.3f})"
        )
        print(
            f"  >=3 weeks    {loglog_alpha(taus[above], skews[above]):>7.3f}   "
            f"({above.sum()} expiries, tau from {days[above].min()}d)"
        )
        print(
            f"  <3 weeks     {loglog_alpha(taus[below], skews[below]):>7.3f}   "
            f"({below.sum()} expiries, tau up to {days[below].max()}d)"
        )

    powers("Market", s_mkt_gea)
    powers("CRL joint", s_crl_gea)

    print(
        "\nGuyon-El Amrani reference: SPX alpha = 0.19 if T < 3 weeks, 0.43 if T > 3 weeks; "
        "extrapolated zero-maturity skew about 1.5 decimal = 150 vol pts."
    )
    i7 = int(np.argmin(np.abs(days - 7)))
    print(
        f"Market GEA skew at {days[i7]}d = {s_mkt_gea[i7]:.1f} vol pts = "
        f"{s_mkt_gea[i7] / 100.0:.2f} decimal vs GEA ~1.5 decimal."
    )
    print(
        f"CRL  GEA skew at {days[i7]}d = {s_crl_gea[i7]:.1f} vol pts "
        f"({s_crl_gea[i7] / s_mkt_gea[i7]:.2f}x market)."
    )


if __name__ == "__main__":
    main()
