"""CRL on sub-1-month SPX expiries: the Guyon–El Amrani short-end risk (epics/0010).

Usage: uv run python experiments/crl_short.py

Guyon–El Amrani find the SPX ATM skew follows a power law above ~1 month but may
not blow up below it (extrapolated zero-maturity skew about -1.5 in decimal
units, i.e. -150 in the vol-point tables below); the CRL skew grows like
tau^(-1/2). This script tests CRL against the four sub-1-month weekly SPXW
Fridays (2026-10-23, -10-30, -11-06, -11-13, root SPXW) plus the six monthly
expiries (root SPX) of the Cboe 2026-10-16 close, each expiry loaded from its
own root. Pipeline and metrics are reused from shifted_spx.py and crl_spx.py by
import. Reports: (1) market ATM skew over all ten expiries, fitted to a pure
power law and to two shapes that stay finite as tau -> 0 (time-shifted and
capped), on all ten and on the six monthlies only; (2) CRL fitted jointly on
all ten and on the six monthlies only (weeklies then out of sample), with the
same vega-weighted objective as crl_spx.py but rho bounded to [-0.99, 0] —
crl_call is reliable only for rho <= 0, and crl_spx.fit_crl allows rho up to
+0.99; (3) per-expiry IV RMSE and ATM skew, the fitted models' skew at tau =
1, 2, 3 days, and whether CRL over/under-shoots the market skew below 1 month.
"""

import sqlite3
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import crl_spx  # noqa: E402
import numpy as np  # noqa: E402
import shifted_spx  # noqa: E402
from crl_spx import crl_model_ivs, loglog_alpha  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.optimize import OptimizeResult, least_squares  # noqa: E402
from shifted_spx import (  # noqa: E402
    SKEW_LN_MONEYNESS,
    atm_skew,
    black_vega,
    implied_iv,
    load_expiry,
    rmse_volpts,
)

import msv  # noqa: E402

WEEKLIES = ("2026-10-23", "2026-10-30", "2026-11-06", "2026-11-13")
MONTHLIES = shifted_spx.EXPIRIES
ROOTS = ("SPXW", "SPXW", "SPXW", "SPXW") + ("SPX",) * len(MONTHLIES)
SHAPE_EVAL = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def load_raw(root: str) -> tuple[list[dict[str, str]], float]:
    """Quotes for one trade date/root in load_expiry's dict format; bid > 0, ask > bid."""
    con = sqlite3.connect(f"file:{crl_spx.CBOE_DB}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT spot FROM snapshots WHERE trade_date = ?", (crl_spx.TRADE_DATE,)
        ).fetchone()
        assert row is not None, f"no snapshot for {crl_spx.TRADE_DATE} in {crl_spx.CBOE_DB}"
        rows = con.execute(
            "SELECT expiry, cp, strike, bid, ask FROM quotes WHERE trade_date = ? AND root = ?",
            (crl_spx.TRADE_DATE, root),
        )
        raw = []
        for expiry, cp, strike, bid, ask in rows:
            bid, ask = float(bid), float(ask)
            if bid <= 0 or ask <= bid:
                continue
            raw.append(
                {
                    "expiry": expiry,
                    "type": cp,
                    "strike": str(strike),
                    "bid": str(bid),
                    "ask": str(ask),
                }
            )
    finally:
        con.close()
    return raw, float(row[0])


def fit_crl_nonpos(data: list[shifted_spx.ExpiryData]) -> tuple[OptimizeResult, float]:
    """Joint 6-parameter CRL fit with rho in [-0.99, 0].

    Same vega-weighted residuals, starts and other bounds as crl_spx.fit_crl,
    whose rho upper bound is +0.99; crl_call is reliable only for rho <= 0.
    """
    F = np.concatenate([np.full(d.K.size, d.F) for d in data])
    D = np.concatenate([np.full(d.K.size, d.D) for d in data])
    tau = np.concatenate([np.full(d.K.size, d.tau) for d in data])
    K = np.concatenate([d.K for d in data])
    mid = np.concatenate([d.mid for d in data])
    vega = np.array(
        [black_vega(d.F, k, d.tau, iv) for d in data for k, iv in zip(d.K, d.iv_mkt, strict=True)]
    )
    denom = D * vega

    def resid(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        a0, a1, a2, lam, k, rho = (float(v) for v in theta)
        px = msv.crl_call(F, K, D, tau, a0, a1, a2, lam, k, rho)
        return np.asarray((px - mid) / denom, dtype=float)

    v_atm = [float(d.iv_mkt[np.argmin(np.abs(np.log(d.K / d.F)))]) ** 2 for d in data if d.K.size]
    v_short, v_long = min(v_atm), max(v_atm)
    base = [max(v_short - v_long, 0.0), max(v_long - v_short, 0.0) * 0.5, v_long]
    lam_k = [(1.0, 0.3), (0.5, 0.15), (2.0, 0.3), (4.0, 0.5), (0.5, 0.5)]
    lb = [0.0, 0.0, 0.0, 1e-3, 1e-3, -0.99]
    ub = [np.inf, np.inf, np.inf, 50.0, 3.0, 0.0]
    best: OptimizeResult | None = None
    t0 = time.perf_counter()
    for lam0, k0 in lam_k:
        for rho0 in crl_spx.RHO_STARTS:
            res = least_squares(
                resid, np.array(base + [lam0, k0, rho0]), bounds=(lb, ub), xtol=1e-10, ftol=1e-10
            )
            if best is None or res.cost < best.cost:
                best = res
    assert best is not None
    return best, time.perf_counter() - t0


def fit_shapes(
    taus: NDArray[np.float64], s: NDArray[np.float64]
) -> list[tuple[str, str, SHAPE_EVAL]]:
    """Fit three shapes to market |skew| (vol points): returns (name, parameters, eval).

    Pure power law A*tau^alpha by log-log least squares (Guyon–El Amrani's
    shape); time-shifted A*(tau+tau0)^alpha and capped min(cap, A*tau^alpha) by
    nonlinear least squares in vol points — both stay finite as tau -> 0.
    """
    slope, intercept = np.polyfit(np.log(taus), np.log(s), 1)
    A0, a0_ = float(np.exp(intercept)), float(slope)

    def resid_shifted(p: NDArray[np.float64]) -> NDArray[np.float64]:
        A, t0, a = (float(v) for v in p)
        return np.asarray(A * (taus + t0) ** a - s, dtype=float)

    def resid_capped(p: NDArray[np.float64]) -> NDArray[np.float64]:
        A, a, cap = (float(v) for v in p)
        return np.asarray(np.minimum(cap, A * taus**a) - s, dtype=float)

    shifted = least_squares(
        resid_shifted,
        np.array([A0, 0.01, a0_]),
        bounds=([0.0, 1e-8, -3.0], [np.inf, 2.0, 0.0]),
        xtol=1e-12,
        ftol=1e-12,
    )
    capped = least_squares(
        resid_capped,
        np.array([A0, a0_, float(np.max(s))]),
        bounds=([0.0, -3.0, 5.0], [np.inf, 0.0, 500.0]),
        xtol=1e-12,
        ftol=1e-12,
    )
    Ap, ap = A0, a0_
    As, ts, ash = (float(v) for v in shifted.x)
    Ac, ac, cap = (float(v) for v in capped.x)
    return [
        ("power", f"A={Ap:.2f} alpha={ap:.3f}", lambda t: Ap * t**ap),
        ("shifted", f"A={As:.2f} tau0={ts:.4f} alpha={ash:.3f}", lambda t: As * (t + ts) ** ash),
        (
            "capped",
            f"A={Ac:.2f} alpha={ac:.3f} cap={cap:.2f}",
            lambda t: np.minimum(cap, Ac * t**ac),
        ),
    ]


def print_shape_fits(
    heading: str,
    taus: NDArray[np.float64],
    abs_s: NDArray[np.float64],
    fit_on: NDArray[np.bool_],
    weekly: NDArray[np.bool_],
) -> list[tuple[str, str, SHAPE_EVAL]]:
    """Fit all three shapes on fit_on and report RMSE on the fit set and the weeklies."""
    tf, sf = taus[fit_on], abs_s[fit_on]
    shapes = fit_shapes(tf, sf)
    wk_in = bool(fit_on[weekly].all())
    print(
        f"\nSkew shape fits, {heading} "
        f"(weeklies {'in' if wk_in else 'out of'} sample; RMSE in vol points):"
    )
    print(f"{'shape':<9} {'parameters':<36} {'RMSE fit set':>13} {'RMSE weeklies':>15}")
    for name, params, ev in shapes:
        rm_fit = float(np.sqrt(np.mean((ev(tf) - sf) ** 2)))
        rm_wk = float(np.sqrt(np.mean((ev(taus[weekly]) - abs_s[weekly]) ** 2)))
        print(f"{name:<9} {params:<36} {rm_fit:>13.3f} {rm_wk:>15.3f}")
    return shapes


def crl_atm_skew(theta: NDArray[np.float64], tau: float) -> float:
    """CRL ATM skew (vol pts per log-moneyness) at tau by central difference +/-0.01.

    F = 100, D = 1, the fitted Nelson–Siegel parameters; same estimator as
    crl_skew.py (its H = 0.01).
    """
    a0, a1, a2, lam, k, rho = (float(v) for v in theta)
    h = 0.01
    ks = 100.0 * np.exp(np.array([-h, h]))
    px = msv.crl_call(100.0, ks, 1.0, np.array([tau, tau]), a0, a1, a2, lam, k, rho)
    iv_lo = implied_iv(100.0, float(ks[0]), tau, float(px[0]))
    iv_hi = implied_iv(100.0, float(ks[1]), tau, float(px[1]))
    assert iv_lo is not None and iv_hi is not None, f"IV inversion failed at tau={tau}"
    return 100.0 * (iv_hi - iv_lo) / (2.0 * h)


def main() -> None:
    raw_spx, spot = load_raw("SPX")
    raw_spxw, spot_w = load_raw("SPXW")
    data = [load_expiry(raw_spxw, e, spot_w) for e in WEEKLIES] + [
        load_expiry(raw_spx, e, spot) for e in MONTHLIES
    ]
    print(
        f"Cboe SQLite {crl_spx.CBOE_DB}  trade_date={crl_spx.TRADE_DATE}  "
        f"spot={spot:.2f}  asof={shifted_spx.ASOF}  (weeklies root SPXW, monthlies root SPX)"
    )

    print(
        f"\nPer-expiry parity forward/discount and OTM fit set "
        f"(|ln(K/F)|<={shifted_spx.MAX_LN_MONEYNESS}):"
    )
    print(f"{'expiry':<12} {'root':<5} {'tau':>7} {'days':>5} {'F':>10} {'D':>8} {'n_options':>10}")
    for d, root in zip(data, ROOTS, strict=True):
        print(
            f"{d.expiry:<12} {root:<5} {d.tau:>7.4f} {round(d.tau * 365):>5d} {d.F:>10.2f} "
            f"{d.D:>8.5f} {d.K.size:>10d}"
        )

    taus = np.array([d.tau for d in data])
    s_mkt = np.array([100.0 * atm_skew(np.log(d.K / d.F), d.iv_mkt) for d in data])
    wk: NDArray[np.bool_] = np.array([True] * len(WEEKLIES) + [False] * len(MONTHLIES))
    mon = ~wk
    print(f"\nMarket ATM skew (vol pts per log-moneyness), |ln(K/F)|<={SKEW_LN_MONEYNESS}:")
    print(f"{'expiry':<12} {'root':<5} {'tau':>7} {'skew':>10}")
    for d, root, s in zip(data, ROOTS, s_mkt, strict=True):
        print(f"{d.expiry:<12} {root:<5} {d.tau:>7.4f} {s:>10.3f}")
    print(
        "\nMarket ATM-skew log-log slope (|skew| ~ tau^alpha): "
        f"all ten {loglog_alpha(taus, s_mkt):.3f}, "
        f"six monthlies {loglog_alpha(taus[mon], s_mkt[mon]):.3f}, "
        f"four weeklies {loglog_alpha(taus[wk], s_mkt[wk]):.3f}"
    )

    abs_mkt = np.abs(s_mkt)
    shapes_all = print_shape_fits("fitted on all ten expiries", taus, abs_mkt, wk | mon, wk)
    shapes_mon = print_shape_fits("fitted on the six monthlies only", taus, abs_mkt, mon, wk)

    data_mon = [d for d, m in zip(data, mon, strict=True) if m]
    res_all, t_all = fit_crl_nonpos(data)
    res_mon, t_mon = fit_crl_nonpos(data_mon)
    for name, res, t in [("all ten", res_all, t_all), ("six monthlies", res_mon, t_mon)]:
        x = res.x
        print(
            f"\nCRL fitted on {name}: a0={x[0]:.5f} a1={x[1]:.5f} a2={x[2]:.5f} lam={x[3]:.4f} "
            f"k={x[4]:.4f} rho={x[5]:.3f}  cost={res.cost:.4f}  [{t:.1f}s]"
        )
    ivs_all = crl_model_ivs(res_all, data)
    ivs_mon = crl_model_ivs(res_mon, data)  # weeklies are out of sample here

    print("\nPer-expiry IV RMSE (vol pts) and ATM skew (vol pts per log-moneyness):")
    print(
        f"{'expiry':<12}{'n':>5}{'RMSE all':>10}{'RMSE mon':>10}{'skew mkt':>10}"
        f"{'skew all':>10}{'all/mkt':>9}{'skew mon':>10}{'mon/mkt':>9}"
    )
    for d, i_a, i_m, s_m in zip(data, ivs_all, ivs_mon, s_mkt, strict=True):
        x = np.log(d.K / d.F)
        s_a = 100.0 * atm_skew(x, i_a)
        s_mn = 100.0 * atm_skew(x, i_m)
        print(
            f"{d.expiry:<12}{d.K.size:>5d}{rmse_volpts(i_a, d.iv_mkt):>10.3f}"
            f"{rmse_volpts(i_m, d.iv_mkt):>10.3f}{s_m:>10.3f}{s_a:>10.3f}{s_a / s_m:>9.2f}"
            f"{s_mn:>10.3f}{s_mn / s_m:>9.2f}"
        )
    all_mkt = np.concatenate([d.iv_mkt for d in data])
    print(
        f"{'overall':<12}{all_mkt.size:>5d}"
        f"{rmse_volpts(np.concatenate(ivs_all), all_mkt):>10.3f}"
        f"{rmse_volpts(np.concatenate(ivs_mon), all_mkt):>10.3f}"
    )

    s_all = np.array(
        [100.0 * atm_skew(np.log(d.K / d.F), iv) for d, iv in zip(data, ivs_all, strict=True)]
    )
    s_mon = np.array(
        [100.0 * atm_skew(np.log(d.K / d.F), iv) for d, iv in zip(data, ivs_mon, strict=True)]
    )
    print("\nCRL ATM-skew log-log slope (|skew| ~ tau^alpha):")
    print(
        f"  joint fit:  all ten {loglog_alpha(taus, s_all):.3f} "
        f"(market {loglog_alpha(taus, s_mkt):.3f}), "
        f"six monthlies {loglog_alpha(taus[mon], s_all[mon]):.3f} "
        f"(market {loglog_alpha(taus[mon], s_mkt[mon]):.3f})"
    )
    print(
        f"  monthlies-only fit: on the six monthlies "
        f"{loglog_alpha(taus[mon], s_mon[mon]):.3f}; "
        f"on the four predicted weeklies {loglog_alpha(taus[wk], s_mon[wk]):.3f} "
        f"(market weeklies {loglog_alpha(taus[wk], s_mkt[wk]):.3f})"
    )

    day_taus = np.array([d / 365.0 for d in (1, 2, 3, 7)])
    print("\nZero-maturity behaviour: |skew| for shapes, signed model ATM skew for CRL (vol pts):")
    print(f"{'fit':<30} {'1d':>9} {'2d':>9} {'3d':>9} {'7d':>9}")
    for label, (_name, _params, ev) in [
        ("power (all ten)", shapes_all[0]),
        ("shifted (all ten)", shapes_all[1]),
        ("capped (all ten)", shapes_all[2]),
        ("power (monthlies)", shapes_mon[0]),
        ("shifted (monthlies)", shapes_mon[1]),
        ("capped (monthlies)", shapes_mon[2]),
    ]:
        print(f"{label:<30} " + " ".join(f"{v:>9.3f}" for v in ev(day_taus)))
    for label, res in [("CRL joint (all ten)", res_all), ("CRL monthlies-only", res_mon)]:
        vals = [crl_atm_skew(res.x, float(t)) for t in day_taus]
        print(f"{label:<30} " + " ".join(f"{v:>9.3f}" for v in vals))


if __name__ == "__main__":
    main()
