"""Fit plain and shifted (displaced-diffusion) MSV to one SPX chain.

Usage: uv run python experiments/shifted_spx.py [spx_chain.csv]

Quotes are the Friday 2026-10-16 close. Fits both models jointly over the six
monthly expiries (one parameter set per surface) by vega-weighted least squares
on OTM mids, then reports IV RMSE and ATM-skew recovery.
"""

import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.optimize import OptimizeResult, brentq, least_squares  # noqa: E402
from scipy.stats import norm  # noqa: E402

import msv  # noqa: E402

DEFAULT_CSV = os.environ.get("CRL_SPX_CSV", "spx_chain_2026-10-19.csv")
ASOF = date(2026, 10, 16)  # quotes are this Friday close
EXPIRIES = ("2026-11-20", "2026-12-18", "2027-01-15", "2027-03-19", "2027-06-17", "2027-09-17")
MAX_LN_MONEYNESS = 0.3
MAX_REL_SPREAD = 0.25
SKEW_LN_MONEYNESS = 0.05


def black_px(F: float, K: float, tau: float, sigma: float) -> float:
    v = sigma * sigma * tau
    sd = np.sqrt(v)
    d1 = (np.log(F / K) + v / 2.0) / sd
    return float(F * norm.cdf(d1) - K * norm.cdf(d1 - sd))


def black_vega(F: float, K: float, tau: float, sigma: float) -> float:
    v = sigma * sigma * tau
    d1 = (np.log(F / K) + v / 2.0) / np.sqrt(v)
    return float(F * np.sqrt(tau) * norm.pdf(d1))


def implied_iv(F: float, K: float, tau: float, price: float) -> float | None:
    try:
        return float(brentq(lambda s: black_px(F, K, tau, s) - price, 1e-6, 5.0, xtol=1e-12))
    except ValueError:  # quote outside Black price bounds: stale/crossed
        return None


@dataclass
class ExpiryData:
    expiry: str
    tau: float
    F: float
    D: float
    K: NDArray[np.float64]
    mid: NDArray[np.float64]
    iv_mkt: NDArray[np.float64]


def load_expiry(raw: list[dict[str, str]], expiry: str, spot: float) -> ExpiryData:
    """Parity F, D plus the OTM fit set (OTM puts converted to calls by parity)."""
    rows = [r for r in raw if r["expiry"] == expiry]
    tau = (date.fromisoformat(expiry) - ASOF).days / 365.0
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

    ks, mids, ivs = [], [], []
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
    return ExpiryData(expiry, tau, F, D, np.array(ks), np.array(mids), np.array(ivs))


def fit_model(data: list[ExpiryData], shifted: bool, spot: float) -> tuple[OptimizeResult, float]:
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
        beta = 0.0
        if shifted:
            a0, a1, a2, lam, k, beta = (float(v) for v in theta)
        else:
            a0, a1, a2, lam, k = (float(v) for v in theta)
        px = msv.gh_call_fwd(F, K, D, tau, a0, a1, a2, lam, k, beta=beta)
        return np.asarray((px - mid) / denom, dtype=float)

    # start points from ATM IV^2: Q(0)=a0+a2 ~ short-end var, Q(inf)=a2 ~ long-end var
    v_atm = [float(d.iv_mkt[np.argmin(np.abs(np.log(d.K / d.F)))]) ** 2 for d in data if d.K.size]
    v_short, v_long = min(v_atm), max(v_atm)
    base = [max(v_short - v_long, 0.0), max(v_long - v_short, 0.0) * 0.5, v_long]
    starts = [(1.0, 0.3, 0.0), (0.5, 0.15, 0.05), (2.0, 0.3, 0.1), (4.0, 0.5, 0.0), (0.5, 0.5, 0.2)]
    lb: list[float] = [0.0, 0.0, 0.0, 1e-3, 1e-3]
    ub: list[float] = [np.inf, np.inf, np.inf, 50.0, 2.0]
    if shifted:
        lb.append(0.0)
        ub.append(3.0 * spot)
    best: OptimizeResult | None = None
    t0 = time.perf_counter()
    for lam0, k0, b0 in starts:
        x0 = base + [lam0, k0] + ([b0 * spot] if shifted else [])
        res = least_squares(resid, np.array(x0), bounds=(lb, ub), xtol=1e-10, ftol=1e-10)
        if best is None or res.cost < best.cost:
            best = res
    assert best is not None
    return best, time.perf_counter() - t0


def model_ivs(
    res: OptimizeResult, data: list[ExpiryData], shifted: bool
) -> list[NDArray[np.float64]]:
    """Unshifted-Black IV of the fitted model's own prices (market-comparable quote)."""
    a0, a1, a2, lam, k = (float(v) for v in res.x[:5])
    beta = float(res.x[5]) if shifted else 0.0
    out = []
    for d in data:
        px = msv.gh_call_fwd(
            d.F,
            d.K,
            np.full(d.K.size, d.D),
            np.full(d.K.size, d.tau),
            a0,
            a1,
            a2,
            lam,
            k,
            beta=beta,
        )
        ivs = []
        for kk, p in zip(d.K, px, strict=True):
            iv = implied_iv(d.F, float(kk), d.tau, float(p) / d.D)
            ivs.append(np.nan if iv is None else iv)
        out.append(np.array(ivs))
    return out


def rmse_volpts(iv_model: NDArray[np.float64], iv_mkt: NDArray[np.float64]) -> float:
    sel = np.isfinite(iv_model)
    return float(100.0 * np.sqrt(np.mean((iv_model[sel] - iv_mkt[sel]) ** 2)))


def atm_skew(x: NDArray[np.float64], iv: NDArray[np.float64]) -> float:
    sel = np.abs(x) <= SKEW_LN_MONEYNESS
    return float(np.polyfit(x[sel], iv[sel], 1)[0])


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    spot = float(raw[0]["spot"])
    data = [load_expiry(raw, e, spot) for e in EXPIRIES]

    print(f"chain: {path}  spot={spot:.2f}  asof={ASOF}")
    print(f"\nPer-expiry parity forward/discount and OTM fit set (|ln(K/F)|<={MAX_LN_MONEYNESS}):")
    print(f"{'expiry':<12} {'tau':>7} {'F':>10} {'D':>8} {'n_options':>10}")
    for d in data:
        print(f"{d.expiry:<12} {d.tau:>7.4f} {d.F:>10.2f} {d.D:>8.5f} {d.K.size:>10d}")

    plain, t_plain = fit_model(data, shifted=False, spot=spot)
    shifted_res, t_shift = fit_model(data, shifted=True, spot=spot)
    p, q = plain.x, shifted_res.x
    print("\nFitted parameters (joint over six expiries):")
    print(
        f"  plain MSV : a0={p[0]:.5f} a1={p[1]:.5f} a2={p[2]:.5f} lam={p[3]:.4f} "
        f"k={p[4]:.4f}  cost={plain.cost:.4f}  [{t_plain:.1f}s]"
    )
    print(
        f"  shifted   : a0={q[0]:.5f} a1={q[1]:.5f} a2={q[2]:.5f} lam={q[3]:.4f} "
        f"k={q[4]:.4f} beta={q[5]:.2f}  cost={shifted_res.cost:.4f}  [{t_shift:.1f}s]"
    )

    ivs_plain = model_ivs(plain, data, shifted=False)
    ivs_shift = model_ivs(shifted_res, data, shifted=True)
    print("\nIV RMSE (vol points):")
    print(f"{'expiry':<12} {'n':>4} {'MSV':>8} {'shifted':>8}")
    for d, iv_p, iv_s in zip(data, ivs_plain, ivs_shift, strict=True):
        print(
            f"{d.expiry:<12} {d.K.size:>4} {rmse_volpts(iv_p, d.iv_mkt):>8.3f} "
            f"{rmse_volpts(iv_s, d.iv_mkt):>8.3f}"
        )
    all_mkt = np.concatenate([d.iv_mkt for d in data])
    rmse_p = rmse_volpts(np.concatenate(ivs_plain), all_mkt)
    rmse_s = rmse_volpts(np.concatenate(ivs_shift), all_mkt)
    print(f"{'overall':<12} {all_mkt.size:>4} {rmse_p:>8.3f} {rmse_s:>8.3f}")

    print(f"\nATM skew (vol points per log-moneyness), |ln(K/F)|<={SKEW_LN_MONEYNESS}:")
    print(
        f"{'expiry':<12} {'market':>8} {'MSV':>8} {'shifted':>8} {'MSV/mkt':>9} {'shift/mkt':>10}"
    )
    for d, iv_p, iv_s in zip(data, ivs_plain, ivs_shift, strict=True):
        x = np.log(d.K / d.F)
        s_m = 100.0 * atm_skew(x, d.iv_mkt)
        s_p, s_s = 100.0 * atm_skew(x, iv_p), 100.0 * atm_skew(x, iv_s)
        print(
            f"{d.expiry:<12} {s_m:>8.3f} {s_p:>8.3f} {s_s:>8.3f}"
            f" {s_p / s_m:>9.2f} {s_s / s_m:>10.2f}"
        )


if __name__ == "__main__":
    main()
