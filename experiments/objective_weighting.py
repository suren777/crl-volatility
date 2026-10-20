"""Compare calibration objectives for CRL (rho <= 0) on the six SPX monthlies.

Epic 0002, task 5. The current SPX fits use a vega-weighted price residual
(approximately an implied-vol objective); the 2014 paper used squared price
errors weighted by 1/spread^4, i.e. a price residual divided by spread^2. We
fit CRL jointly over the six 2026-10-16 Cboe monthlies under three objectives
on the same OTM fit set:

  (a) vega-weighted   resid = (px - mid) / (D * vega)   [current SPX fits]
  (b) 1/spread^4      resid = (px - mid) / spread^2     [2014 paper]
  (c) unweighted      resid = (px - mid)

For each objective we report parameters, IV RMSE (overall and per expiry),
price RMSE, ATM skew ratio per expiry, and the fraction of fit options priced
inside their own bid-ask spread. Rho is bounded to [-0.99, 0] because
crl_call is reliable only for rho <= 0 (experiments/crl_short.py).

Usage: uv run python experiments/objective_weighting.py
"""

import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import crl_short  # noqa: E402
import crl_spx  # noqa: E402
import numpy as np  # noqa: E402
import shifted_spx  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.optimize import OptimizeResult, least_squares  # noqa: E402
from shifted_spx import (  # noqa: E402
    MAX_LN_MONEYNESS,
    MAX_REL_SPREAD,
    SKEW_LN_MONEYNESS,
    atm_skew,
    black_vega,
    implied_iv,
    rmse_volpts,
)

import msv  # noqa: E402

Weight = Literal["vega", "spread2014", "none"]
START_SPECS = [  # same (lam, k) crosses and rho starts as crl_spx.fit_crl
    (1.0, 0.3),
    (0.5, 0.15),
    (2.0, 0.3),
    (4.0, 0.5),
    (0.5, 0.5),
]


@dataclass
class ExpiryData:
    expiry: str
    tau: float
    F: float
    D: float
    K: NDArray[np.float64]
    mid: NDArray[np.float64]
    iv_mkt: NDArray[np.float64]
    side: NDArray[np.str_]
    bid: NDArray[np.float64]
    ask: NDArray[np.float64]
    spread: NDArray[np.float64]


def load_expiry_w(raw: list[dict[str, str]], expiry: str, spot: float) -> ExpiryData:
    """Same OTM fit set and parity F/D as shifted_spx.load_expiry, plus side/bid/ask/spread.

    The fit set keeps the OTM option per strike (put for K < F, else call);
    bid/ask/spread stored for that listed option. mid and iv_mkt are the
    call-equivalent values (put mids converted by parity), as in load_expiry.
    """
    rows = [r for r in raw if r["expiry"] == expiry]
    tau = (date.fromisoformat(expiry) - shifted_spx.ASOF).days / 365.0
    quotes = {
        t: {float(r["strike"]): (float(r["bid"]), float(r["ask"])) for r in rows if r["type"] == t}
        for t in ("C", "P")
    }
    both = [k for k in sorted(set(quotes["C"]) & set(quotes["P"])) if abs(k / spot - 1.0) <= 0.10]
    K_pc = np.array(both)
    y = np.array(
        [
            (quotes["C"][k][0] + quotes["C"][k][1] - quotes["P"][k][0] - quotes["P"][k][1]) / 2
            for k in both
        ]
    )
    slope, intercept = (float(v) for v in np.polyfit(K_pc, y, 1))
    D = -slope
    F = intercept / D

    ks, mids, ivs, sides, bids, asks = [], [], [], [], [], []
    for k in sorted(set(quotes["C"]) | set(quotes["P"])):
        if abs(np.log(k / F)) > MAX_LN_MONEYNESS:
            continue
        side = "P" if k < F else "C"
        if k not in quotes[side]:
            continue
        bid, ask = quotes[side][k]
        quoted_mid = (bid + ask) / 2
        if ask <= 0 or quoted_mid <= 0 or (ask - bid) / quoted_mid > MAX_REL_SPREAD:
            continue
        mid = quoted_mid + D * (F - k) if side == "P" else quoted_mid
        iv = implied_iv(F, k, tau, mid / D)
        if iv is None:
            continue
        ks.append(k)
        mids.append(mid)
        ivs.append(iv)
        sides.append(side)
        bids.append(bid)
        asks.append(ask)
    return ExpiryData(
        expiry,
        tau,
        F,
        D,
        np.array(ks),
        np.array(mids),
        np.array(ivs),
        np.array(sides),
        np.array(bids),
        np.array(asks),
        np.array(asks) - np.array(bids),
    )


def fit_crl_weighted(data: list[ExpiryData], weight: Weight) -> tuple[OptimizeResult, float]:
    """Joint 6-parameter CRL fit (rho in [-0.99, 0]) under one weighting."""
    F = np.concatenate([np.full(d.K.size, d.F) for d in data])
    D = np.concatenate([np.full(d.K.size, d.D) for d in data])
    tau = np.concatenate([np.full(d.K.size, d.tau) for d in data])
    K = np.concatenate([d.K for d in data])
    mid = np.concatenate([d.mid for d in data])
    spread = np.concatenate([d.spread for d in data])

    if weight == "vega":
        vega = np.array(
            [
                black_vega(d.F, k, d.tau, iv)
                for d in data
                for k, iv in zip(d.K, d.iv_mkt, strict=True)
            ]
        )
        denom = D * vega
    elif weight == "spread2014":
        denom = spread**2
    else:
        denom = np.ones_like(mid)

    def resid(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        a0, a1, a2, lam, k, rho = (float(v) for v in theta)
        px = msv.crl_call(F, K, D, tau, a0, a1, a2, lam, k, rho)
        return np.asarray((px - mid) / denom, dtype=float)

    v_atm = [float(d.iv_mkt[np.argmin(np.abs(np.log(d.K / d.F)))]) ** 2 for d in data if d.K.size]
    v_short, v_long = min(v_atm), max(v_atm)
    base = [max(v_short - v_long, 0.0), max(v_long - v_short, 0.0) * 0.5, v_long]
    lb = [0.0, 0.0, 0.0, 1e-3, 1e-3, -0.99]
    ub = [np.inf, np.inf, np.inf, 50.0, 3.0, 0.0]
    best: OptimizeResult | None = None
    t0 = time.perf_counter()
    for lam0, k0 in START_SPECS:
        for rho0 in crl_spx.RHO_STARTS:
            res = least_squares(
                resid, np.array(base + [lam0, k0, rho0]), bounds=(lb, ub), xtol=1e-10, ftol=1e-10
            )
            if best is None or res.cost < best.cost:
                best = res
    assert best is not None
    return best, time.perf_counter() - t0


def model_ivs(res: OptimizeResult, data: list[ExpiryData]) -> list[NDArray[np.float64]]:
    """Unshifted-Black IV of the fitted CRL prices (same convention as crl_spx)."""
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


def model_prices(res: OptimizeResult, data: list[ExpiryData]) -> list[NDArray[np.float64]]:
    a0, a1, a2, lam, k, rho = (float(v) for v in res.x)
    return [
        msv.crl_call(
            d.F, d.K, np.full(d.K.size, d.D), np.full(d.K.size, d.tau), a0, a1, a2, lam, k, rho
        )
        for d in data
    ]


def frac_in_spread(res: OptimizeResult, data: list[ExpiryData]) -> float:
    """Fraction of fit options whose model-implied OTM price sits in its own bid-ask.

    For put-derived rows the model call price is converted back to the listed
    put by parity (px - D*(F - K)) before comparison with the put's bid/ask.
    """
    hit = tot = 0
    for d, px in zip(data, model_prices(res, data), strict=True):
        pm = np.where(d.side == "P", px - d.D * (d.F - d.K), px)
        hit += int(np.sum((pm >= d.bid) & (pm <= d.ask)))
        tot += d.K.size
    return hit / tot


def run_objective(
    weight: Weight, data: list[ExpiryData], market_skews: list[float]
) -> tuple[OptimizeResult, list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    res, t = fit_crl_weighted(data, weight)
    x = res.x
    print(
        f"\nObjective '{weight}': a0={x[0]:.5f} a1={x[1]:.5f} a2={x[2]:.5f} "
        f"lam={x[3]:.4f} k={x[4]:.4f} rho={x[5]:.3f}  cost={res.cost:.4f}  [{t:.1f}s]"
    )
    ivs = model_ivs(res, data)
    prices = model_prices(res, data)

    all_mkt = np.concatenate([d.iv_mkt for d in data])
    iv_rmse = rmse_volpts(np.concatenate(ivs), all_mkt)
    px_all = np.concatenate(prices)
    mid_all = np.concatenate([d.mid for d in data])
    px_rmse = float(np.sqrt(np.mean((px_all - mid_all) ** 2)))

    print(
        f"{'expiry':<12}{'iv_rmse':>9}{'px_rmse':>9}{'skew_mkt':>10}{'skew_mod':>10}{'mod/mkt':>9}"
    )
    for d, ivc, px, s_m in zip(data, ivs, prices, market_skews, strict=True):
        xrow = np.log(d.K / d.F)
        s_mod = 100.0 * atm_skew(xrow, ivc)
        pr = float(np.sqrt(np.mean((px - d.mid) ** 2)))
        ratio = s_mod / s_m if s_m else float("nan")
        print(
            f"{d.expiry:<12}{rmse_volpts(ivc, d.iv_mkt):>9.3f}{pr:>9.4f}{s_m:>10.3f}"
            f"{s_mod:>10.3f}{ratio:>9.3f}"
        )
    frac = frac_in_spread(res, data)
    print(
        f"{'overall':<12}{iv_rmse:>9.3f}{px_rmse:>9.4f}{'':>10}{'':>10}"
        f"  {len(all_mkt)} opts, {frac * 100:.1f}% inside bid-ask spread"
    )
    return res, ivs, prices


def main() -> None:
    raw, spot = crl_short.load_raw("SPX")
    data = [load_expiry_w(raw, e, spot) for e in shifted_spx.EXPIRIES]
    print(
        f"Cboe SQLite {crl_spx.CBOE_DB}  trade_date={crl_spx.TRADE_DATE}  "
        f"spot={spot:.2f}  asof={shifted_spx.ASOF}"
    )
    print(f"\nPer-expiry OTM fit set (|ln(K/F)|<={MAX_LN_MONEYNESS}):")
    print(f"{'expiry':<12} {'tau':>7} {'F':>10} {'D':>8} {'n':>4} {'median spread':>14}")
    for d in data:
        print(
            f"{d.expiry:<12} {d.tau:>7.4f} {d.F:>10.2f} {d.D:>8.5f} {d.K.size:>4} "
            f"{np.median(d.spread):>14.3f}"
        )

    market_skews = [100.0 * atm_skew(np.log(d.K / d.F), d.iv_mkt) for d in data]
    print(f"\nMarket ATM skew (vol pts per log-moneyness), |ln(K/F)|<= {SKEW_LN_MONEYNESS:.2f}:")
    for d, s in zip(data, market_skews, strict=True):
        print(f"  {d.expiry}: {s:.3f}")

    for weight in ("vega", "spread2014", "none"):
        run_objective(weight, data, market_skews)


if __name__ == "__main__":
    main()
