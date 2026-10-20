"""Fit plain, shifted, and correlated random-level (CRL) MSV to SPX quotes on two sources.

Usage: uv run python experiments/crl_spx.py

Quotes are the Friday 2026-10-16 close, from yfinance (CSV) and Cboe delayed
quotes (SQLite, root SPX; SPXW also lists these dates and is ignored). Fits
plain MSV, shifted MSV and CRL (a0, a1, a2, lam, k, rho) jointly over the six
monthly expiries by vega-weighted least squares on OTM mids — pipeline and
metrics reused from shifted_spx.py — plus a per-expiry CRL fit as a diagnostic.
"""

import csv
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import shifted_spx  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.optimize import OptimizeResult, least_squares  # noqa: E402
from shifted_spx import (  # noqa: E402
    EXPIRIES,
    SKEW_LN_MONEYNESS,
    ExpiryData,
    atm_skew,
    black_vega,
    implied_iv,
    load_expiry,
    rmse_volpts,
)

import msv  # noqa: E402

CBOE_DB = os.environ.get("CRL_CBOE_DB", str(Path.home() / "Data/article/cboe/spx.sqlite"))
TRADE_DATE = "2026-10-16"
ROOT = "SPX"  # AM-settled monthlies; SPXW also lists these dates and is ignored
RHO_STARTS = (-0.9, -0.6, -0.3)


def load_cboe_raw(db: str) -> tuple[list[dict[str, str]], float]:
    """Quotes for one trade date/root in load_expiry's dict format; bid > 0, ask > bid."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT spot FROM snapshots WHERE trade_date = ?", (TRADE_DATE,)
        ).fetchone()
        assert row is not None, f"no snapshot for {TRADE_DATE} in {db}"
        spot = float(row[0])
        rows = con.execute(
            "SELECT expiry, cp, strike, bid, ask FROM quotes WHERE trade_date = ? AND root = ?",
            (TRADE_DATE, ROOT),
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
    return raw, spot


def fit_crl(data: list[ExpiryData]) -> tuple[OptimizeResult, float]:
    """6-parameter CRL fit: same vega-weighted residuals and starts as fit_model."""
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

    # same base/lam/k starts as fit_model (base from ATM IV^2), crossed with rho starts
    v_atm = [float(d.iv_mkt[np.argmin(np.abs(np.log(d.K / d.F)))]) ** 2 for d in data if d.K.size]
    v_short, v_long = min(v_atm), max(v_atm)
    base = [max(v_short - v_long, 0.0), max(v_long - v_short, 0.0) * 0.5, v_long]
    lam_k = [(1.0, 0.3), (0.5, 0.15), (2.0, 0.3), (4.0, 0.5), (0.5, 0.5)]
    lb = [0.0, 0.0, 0.0, 1e-3, 1e-3, -0.99]
    ub = [np.inf, np.inf, np.inf, 50.0, 3.0, 0.99]
    best: OptimizeResult | None = None
    t0 = time.perf_counter()
    for lam0, k0 in lam_k:
        for rho0 in RHO_STARTS:
            res = least_squares(
                resid, np.array(base + [lam0, k0, rho0]), bounds=(lb, ub), xtol=1e-10, ftol=1e-10
            )
            if best is None or res.cost < best.cost:
                best = res
    assert best is not None
    return best, time.perf_counter() - t0


def crl_model_ivs(res: OptimizeResult, data: list[ExpiryData]) -> list[NDArray[np.float64]]:
    """Unshifted-Black IV of the CRL fit's own prices (same convention as model_ivs)."""
    a0, a1, a2, lam, k, rho = (float(v) for v in res.x)
    out = []
    for d in data:
        px = msv.crl_call(
            d.F, d.K, np.full(d.K.size, d.D), np.full(d.K.size, d.tau), a0, a1, a2, lam, k, rho
        )
        ivs = [
            implied_iv(d.F, float(kk), d.tau, float(p) / d.D) for kk, p in zip(d.K, px, strict=True)
        ]
        out.append(np.array([np.nan if iv is None else iv for iv in ivs]))
    return out


def loglog_alpha(taus: NDArray[np.float64], skews: NDArray[np.float64]) -> float:
    """Slope of log|skew| vs log tau (paper claim: CRL ATM skew ~ tau^(-1/2))."""
    a = np.abs(skews)
    sel = (a > 0) & np.isfinite(a)
    return float(np.polyfit(np.log(taus[sel]), np.log(a[sel]), 1)[0])


def rmse_wing(
    sel: NDArray[np.bool_], iv_m: NDArray[np.float64], iv_k: NDArray[np.float64]
) -> float:
    return (
        float(100.0 * np.sqrt(np.mean((iv_m[sel] - iv_k[sel]) ** 2))) if sel.any() else float("nan")
    )


def run_source(name: str, raw: list[dict[str, str]], spot: float) -> None:
    data = [load_expiry(raw, e, spot) for e in EXPIRIES]
    print(f"\n{'=' * 78}\n{name}  spot={spot:.2f}  asof={shifted_spx.ASOF}")
    print(
        f"\nPer-expiry parity forward/discount and OTM fit set "
        f"(|ln(K/F)|<={shifted_spx.MAX_LN_MONEYNESS}):"
    )
    print(f"{'expiry':<12} {'tau':>7} {'F':>10} {'D':>8} {'n_options':>10}")
    for d in data:
        print(f"{d.expiry:<12} {d.tau:>7.4f} {d.F:>10.2f} {d.D:>8.5f} {d.K.size:>10d}")

    plain, t_plain = shifted_spx.fit_model(data, shifted=False, spot=spot)
    shifted_res, t_shift = shifted_spx.fit_model(data, shifted=True, spot=spot)
    crl, t_crl = fit_crl(data)
    per_exp = [fit_crl([d]) for d in data]
    p, q, c = plain.x, shifted_res.x, crl.x
    print("\nFitted parameters (joint over six expiries):")
    print(
        f"  plain MSV : a0={p[0]:.5f} a1={p[1]:.5f} a2={p[2]:.5f} lam={p[3]:.4f} "
        f"k={p[4]:.4f}  cost={plain.cost:.4f}  [{t_plain:.1f}s]"
    )
    print(
        f"  shifted   : a0={q[0]:.5f} a1={q[1]:.5f} a2={q[2]:.5f} lam={q[3]:.4f} "
        f"k={q[4]:.4f} beta={q[5]:.2f}  cost={shifted_res.cost:.4f}  [{t_shift:.1f}s]"
    )
    print(
        f"  CRL       : a0={c[0]:.5f} a1={c[1]:.5f} a2={c[2]:.5f} lam={c[3]:.4f} "
        f"k={c[4]:.4f} rho={c[5]:.3f}  cost={crl.cost:.4f}  [{t_crl:.1f}s]"
    )
    print("  CRL per expiry (diagnostic):")
    for d, (res, t) in zip(data, per_exp, strict=True):
        x = res.x
        print(
            f"    {d.expiry}: lam={x[3]:.4f} k={x[4]:.4f} rho={x[5]:.3f}  "
            f"cost={res.cost:.4f}  [{t:.1f}s]"
        )

    ivs_plain = shifted_spx.model_ivs(plain, data, shifted=False)
    ivs_shift = shifted_spx.model_ivs(shifted_res, data, shifted=True)
    ivs_crl = crl_model_ivs(crl, data)
    ivs_pe = [crl_model_ivs(res, [d])[0] for d, (res, _t) in zip(data, per_exp, strict=True)]

    print("\nIV RMSE (vol points):")
    print(f"{'expiry':<12} {'n':>4} {'MSV':>8} {'shifted':>8} {'CRL':>8} {'CRL/exp':>8}")
    for d, iv_p, iv_s, iv_c, iv_ce in zip(data, ivs_plain, ivs_shift, ivs_crl, ivs_pe, strict=True):
        print(
            f"{d.expiry:<12} {d.K.size:>4} {rmse_volpts(iv_p, d.iv_mkt):>8.3f} "
            f"{rmse_volpts(iv_s, d.iv_mkt):>8.3f} {rmse_volpts(iv_c, d.iv_mkt):>8.3f} "
            f"{rmse_volpts(iv_ce, d.iv_mkt):>8.3f}"
        )
    all_mkt = np.concatenate([d.iv_mkt for d in data])
    print(
        f"{'overall':<12} {all_mkt.size:>4} "
        f"{rmse_volpts(np.concatenate(ivs_plain), all_mkt):>8.3f} "
        f"{rmse_volpts(np.concatenate(ivs_shift), all_mkt):>8.3f} "
        f"{rmse_volpts(np.concatenate(ivs_crl), all_mkt):>8.3f} "
        f"{rmse_volpts(np.concatenate(ivs_pe), all_mkt):>8.3f}"
    )

    print("\nJoint-CRL IV RMSE by wing (vol points):")
    print(f"{'expiry':<12} {'puts':>8} {'calls':>8}")
    for d, iv_c in zip(data, ivs_crl, strict=True):
        x = np.log(d.K / d.F)
        fin = np.isfinite(iv_c)
        rm_p = rmse_wing((x < 0) & fin, iv_c, d.iv_mkt)
        rm_c = rmse_wing((x >= 0) & fin, iv_c, d.iv_mkt)
        print(f"{d.expiry:<12} {rm_p:>8.3f} {rm_c:>8.3f}")

    print(f"\nATM skew (vol points per log-moneyness), |ln(K/F)|<={SKEW_LN_MONEYNESS}:")
    print(
        f"{'expiry':<12} {'market':>8} {'MSV':>8} {'shifted':>8} {'CRL':>8} "
        f"{'CRL/exp':>8} {'CRL/mkt':>8}"
    )
    sk_m, sk_c = [], []
    for d, iv_p, iv_s, iv_c, iv_ce in zip(data, ivs_plain, ivs_shift, ivs_crl, ivs_pe, strict=True):
        x = np.log(d.K / d.F)
        s_m = 100.0 * atm_skew(x, d.iv_mkt)
        s_p, s_s = 100.0 * atm_skew(x, iv_p), 100.0 * atm_skew(x, iv_s)
        s_c, s_ce = 100.0 * atm_skew(x, iv_c), 100.0 * atm_skew(x, iv_ce)
        sk_m.append(s_m)
        sk_c.append(s_c)
        print(
            f"{d.expiry:<12} {s_m:>8.3f} {s_p:>8.3f} {s_s:>8.3f} {s_c:>8.3f} "
            f"{s_ce:>8.3f} {s_c / s_m:>8.2f}"
        )

    taus = np.array([d.tau for d in data])
    a_m = loglog_alpha(taus, np.array(sk_m))
    a_c = loglog_alpha(taus, np.array(sk_c))
    print("\nATM-skew log-log slope over the six expiries (|skew| ~ tau^alpha):")
    print(f"  market {a_m:.3f}   joint CRL {a_c:.3f}")


def main() -> None:
    with open(shifted_spx.DEFAULT_CSV, newline="") as f:
        raw = list(csv.DictReader(f))
    run_source(f"yfinance CSV {shifted_spx.DEFAULT_CSV}", raw, float(raw[0]["spot"]))

    raw_db, spot_db = load_cboe_raw(CBOE_DB)
    run_source(
        f"Cboe SQLite {CBOE_DB} (trade_date={TRADE_DATE}, root={ROOT}; SPXW ignored)",
        raw_db,
        spot_db,
    )


if __name__ == "__main__":
    main()
