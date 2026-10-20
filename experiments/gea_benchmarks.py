"""GEA ATM-skew for CRL, SSVI and Heston fits on SPX so the comparison is like-for-like.

Usage: uv run python experiments/gea_benchmarks.py

Same Cboe 2026-10-16 snapshot and six SPX monthlies as the benchmarks
(`benchmarks_spx.py`); CRL (rho <= 0), SSVI and Heston fitted by the very same
fitters. Per expiry we report the market and each model's ATM skew under BOTH
estimators — the 5% linear one (`shifted_spx.atm_skew`, as in the benchmark
table) and the Guyon-El Amrani degree-2/12-strike one (`gea_skew.gea_skew`) —
so the paper's S&P table can state model/market skew ratios under the estimator
Guyon & El Amrani use, comparable to their SPX alpha = 0.43 above 3 weeks.
Also reports the log-log skew exponent |S| ~ A*tau^alpha over the six monthlies,
per estimator, for market and each model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from benchmarks_spx import fit_heston, fit_ssvi, heston_model_ivs, ssvi_model_ivs  # noqa: E402
from crl_short import fit_crl_nonpos, load_raw  # noqa: E402
from crl_spx import crl_model_ivs, loglog_alpha  # noqa: E402
from gea_skew import gea_skew  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from shifted_spx import ASOF, EXPIRIES, atm_skew, load_expiry  # noqa: E402


def per_expiry(
    x: list[NDArray[np.float64]], ivl: list[NDArray[np.float64]]
) -> tuple[list[float], list[float]]:
    lin = [100.0 * atm_skew(xi, iv) for xi, iv in zip(x, ivl, strict=True)]
    g = [100.0 * gea_skew(xi, iv) for xi, iv in zip(x, ivl, strict=True)]
    return lin, g


def main() -> None:
    raw_spx, spot = load_raw("SPX")
    data = [load_expiry(raw_spx, e, spot) for e in EXPIRIES]
    taus = np.array([d.tau for d in data])
    days = np.array([round(d.tau * 365) for d in data])
    x = [np.log(d.K / d.F) for d in data]

    print(
        f"Cboe SQLite spx.sqlite  trade_date=2026-10-16  spot={spot:.2f}  asof={ASOF}  "
        f"({len(EXPIRIES)} SPX monthlies)"
    )
    print(
        "ATM skew estimators: GEA = degree-2 poly, 6 closest strikes per side of the "
        "forward (Guyon-El Amrani); linear = regression over |ln K/F| <= 0.05."
    )

    print("\nPer-expiry parity forward/discount and OTM fit set (|ln(K/F)|<=0.3):")
    print(f"{'expiry':<12} {'tau':>7} {'days':>5} {'F':>10} {'D':>8} {'n_options':>10}")
    for d, dy in zip(data, days, strict=True):
        print(f"{d.expiry:<12} {d.tau:>7.4f} {dy:>5d} {d.F:>10.2f} {d.D:>8.5f} {d.K.size:>10d}")

    mkt_lin, mkt_gea = per_expiry(x, [d.iv_mkt for d in data])

    crl, t_crl = fit_crl_nonpos(data)
    c = crl.x
    ssvi, t_ssvi = fit_ssvi(data)
    s = ssvi.x
    hes, t_hes = fit_heston(data)
    h = hes.x

    print("\nFitted parameters (joint over six expiries) and fit time:")
    print(
        f"  CRL   : a0={c[0]:.5f} a1={c[1]:.5f} a2={c[2]:.5f} lam={c[3]:.4f} "
        f"k={c[4]:.4f} rho={c[5]:.3f} (rho<=0)  cost={crl.cost:.4f}  [{t_crl:.1f}s]"
    )
    print(
        f"  SSVI  : eta={s[0]:.4f} gamma={s[1]:.4f} rho={s[2]:.3f} "
        f"cost={ssvi.cost:.4f}  [{t_ssvi:.1f}s]"
    )
    feller = 2.0 * h[1] * h[2] / h[3] ** 2
    print(
        f"  Heston: v0={h[0]:.5f} kappa={h[1]:.4f} theta={h[2]:.5f} sigma={h[3]:.4f} "
        f"rho={h[4]:.3f}  cost={hes.cost:.4f}  [{t_hes:.1f}s]  Feller={feller:.3f}"
    )

    ivs = {
        "CRL": crl_model_ivs(crl, data),
        "SSVI": ssvi_model_ivs(ssvi, data),
        "Heston": heston_model_ivs(hes, data),
    }
    model_skews = {k: per_expiry(x, iv) for k, iv in ivs.items()}

    print("\nPer-expiry ATM skew (vol pts per log-moneyness): market and models, both estimators")
    print(
        f"{'expiry':<12} {'days':>5} {'mkt lin':>9} {'mkt GEA':>9} "
        f"{'CRL lin':>9} {'CRL GEA':>9} {'SSVI lin':>9} {'SSVI GEA':>9} "
        f"{'Hes lin':>9} {'Hes GEA':>9}"
    )
    for d, dy, ml, mg in zip(data, days, mkt_lin, mkt_gea, strict=True):
        cells = [f"{d.expiry:<12} {dy:>5d} {ml:>9.3f} {mg:>9.3f}"]
        i = data.index(d)
        for k in ("CRL", "SSVI", "Heston"):
            lv, gv = model_skews[k]
            cells.append(f"{lv[i]:>9.3f} {gv[i]:>9.3f}")
        print(" ".join(cells))

    print("\nModel/market ATM skew ratio, both estimators:")
    print(
        f"{'expiry':<12} {'days':>5} "
        f"{'CRL lin':>9} {'CRL GEA':>9} {'SSVI lin':>9} {'SSVI GEA':>9} "
        f"{'Hes lin':>9} {'Hes GEA':>9}"
    )
    for d, dy, ml, mg in zip(data, days, mkt_lin, mkt_gea, strict=True):
        cells = [f"{d.expiry:<12} {dy:>5d}"]
        i = data.index(d)
        for k in ("CRL", "SSVI", "Heston"):
            lv, gv = model_skews[k]
            cells.append(f"{lv[i] / ml:>9.3f} {gv[i] / mg:>9.3f}")
        print(" ".join(cells))

    print("\nLog-log skew exponent (|S| ~ A*tau^alpha) over the six monthlies, per estimator:")
    print(f"{'source':<10} {'linear':>9} {'GEA':>9}")
    print(
        f"{'Market':<10} {loglog_alpha(taus, np.array(mkt_lin)):>9.3f} "
        f"{loglog_alpha(taus, np.array(mkt_gea)):>9.3f}"
    )
    for k in ("CRL", "SSVI", "Heston"):
        lv, gv = model_skews[k]
        print(
            f"{k:<10} {loglog_alpha(taus, np.array(lv)):>9.3f} "
            f"{loglog_alpha(taus, np.array(gv)):>9.3f}"
        )
    print("\nReference: Guyon-El Amrani SPX alpha = 0.43 for T > 3 weeks.")


if __name__ == "__main__":
    main()
