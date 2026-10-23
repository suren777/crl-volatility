"""Fixed-lambda CRL refits across the panel dates: is the (a0,a1,a2,lam) drift a flat direction?

Usage: uv run python experiments/panel_fixed_lambda.py

The free six-parameter CRL fits in `panel.csv` jump between trade dates mostly
in (a0, a1, a2, lam), with a0 or a2 at its zero bound. This script tests
whether that jump is the known flat direction of the objective rather than
model instability: it refits CRL on every panel trade date with lam held fixed
at one pooled value, and compares the remaining parameters, the fitted
Nelson–Siegel variance curve and the IV RMSE against the free fits.

Step 1 profiles lam on a grid: for each grid value the five remaining
parameters (a0, a1, a2, k, rho) are fitted on each panel date with the same
vega-weighted residuals, bounds and starts as crl_short.fit_crl_nonpos (k
starts {0.15, 0.3, 0.5} crossed with crl_spx.RHO_STARTS); lam* is the argmin
of the cost summed over dates. Step 2 reports the lam* fits per date next to
the free fits read from panel.csv (read-only). Step 3 compares sqrt(Q(tau))
(vol points) of the free and fixed-lam fits. Trade dates come from the same
snapshot query as panel.py, restricted to dates with a free CRL row in
panel.csv. Reads the DB read-only; writes derived numbers only to
experiments/results/panel_fixed_lambda.md.
"""

import csv
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import crl_spx  # noqa: E402
import numpy as np  # noqa: E402
import panel  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.optimize import OptimizeResult, least_squares  # noqa: E402
from shifted_spx import ExpiryData, black_vega, implied_iv, rmse_volpts  # noqa: E402

import msv  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
MD_PATH = RESULTS / "panel_fixed_lambda.md"
LAM_GRID = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
K_STARTS = (0.15, 0.3, 0.5)
NAMES5 = ("a0", "a1", "a2", "k", "rho")
LB5 = (0.0, 0.0, 0.0, 1e-3, -0.99)
UB5 = (np.inf, np.inf, np.inf, 3.0, 0.0)
FREE_NAMES = ("a0", "a1", "a2", "lam", "k", "rho", "cost", "rmse_ov")
CURVE_TAUS = (1.0 / 12.0, 0.25, 0.5, 1.0)


def trade_dates() -> list[str]:
    """Same snapshot query as panel.main."""
    con = sqlite3.connect(f"file:{panel.DB}?mode=ro", uri=True)
    try:
        return [
            str(r[0]) for r in con.execute("SELECT trade_date FROM snapshots ORDER BY trade_date")
        ]
    finally:
        con.close()


def free_fits(dates: list[str]) -> dict[str, dict[str, float]]:
    """Free CRL rows of panel.csv for the given dates (read-only)."""
    with open(panel.CSV_PATH, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["model"] == "crl" and r["trade_date"] in dates]
    return {r["trade_date"]: {n: float(r[n]) for n in FREE_NAMES} for r in rows}


def fit_crl_fixed_lam(data: list[ExpiryData], lam: float) -> tuple[OptimizeResult, float]:
    """5-parameter CRL fit with lam held fixed: residuals, bounds and starts as crl_short."""
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
        a0, a1, a2, k, rho = (float(v) for v in theta)
        px = msv.crl_call(F, K, D, tau, a0, a1, a2, lam, k, rho)
        return np.asarray((px - mid) / denom, dtype=float)

    v_atm = [float(d.iv_mkt[np.argmin(np.abs(np.log(d.K / d.F)))]) ** 2 for d in data if d.K.size]
    v_short, v_long = min(v_atm), max(v_atm)
    base = [max(v_short - v_long, 0.0), max(v_long - v_short, 0.0) * 0.5, v_long]
    best: OptimizeResult | None = None
    t0 = time.perf_counter()
    for k0 in K_STARTS:
        for rho0 in crl_spx.RHO_STARTS:
            res = least_squares(
                resid, np.array(base + [k0, rho0]), bounds=(LB5, UB5), xtol=1e-10, ftol=1e-10
            )
            if best is None or res.cost < best.cost:
                best = res
    assert best is not None
    return best, time.perf_counter() - t0


def fixed_lam_ivs(
    res: OptimizeResult, data: list[ExpiryData], lam: float
) -> list[NDArray[np.float64]]:
    """Unshifted-Black IVs of the fixed-lam fit's own prices (as crl_spx.crl_model_ivs)."""
    a0, a1, a2, k, rho = (float(v) for v in res.x)
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


def main() -> None:
    out: list[str] = []

    def say(line: str = "") -> None:
        print(line)
        out.append(line)

    t_start = time.perf_counter()
    all_dates = trade_dates()
    free = free_fits(all_dates)
    dates = [td for td in all_dates if td in free]
    assert dates, "no trade date has a free CRL row in panel.csv"

    data: dict[str, list[ExpiryData]] = {}
    for td in dates:
        raw, spot = panel.load_raw(panel.DB, td)
        asof = date.fromisoformat(td)
        expiries = panel.pick_expiries(raw, asof)
        data[td] = [panel.load_expiry_asof(raw, e, spot, asof) for e in expiries]

    say(f"Panel dates: {', '.join(dates)} (from panel.DB snapshot query x panel.csv CRL rows)")

    # Step 1: pooled lam profile — 5 free params per (date, lam).
    say("")
    say("Step 1: lam profile, per-date cost of the 5-parameter fit (a0,a1,a2,k,rho; lam fixed):")
    profile: dict[float, dict[str, OptimizeResult]] = {}
    for lam in LAM_GRID:
        profile[lam] = {}
        costs = []
        for td in dates:
            res, _t = fit_crl_fixed_lam(data[td], lam)
            profile[lam][td] = res
            costs.append(res.cost)
        total = sum(costs)
        say(
            f"  lam={lam:.2f}: "
            + "  ".join(f"{td}={c:.4f}" for td, c in zip(dates, costs, strict=True))
            + f"  sum={total:.4f}"
        )
    lam_star = min(LAM_GRID, key=lambda lm: sum(profile[lm][td].cost for td in dates))
    say(f"lam* = {lam_star:.2f} (argmin of the summed cost)")

    # Step 2: lam* fits per date, next to the free fits.
    say("")
    say(f"Step 2: fixed-lam fits at lam*={lam_star:.2f} vs free six-parameter fits (panel.csv):")
    say("")
    say(
        "  date        a0       a1       a2       k      rho     cost   IV RMSE (fixed)   "
        "IV RMSE (free)   at bound"
    )
    fixed: dict[str, OptimizeResult] = {td: profile[lam_star][td] for td in dates}
    for td in dates:
        res = fixed[td]
        x = res.x
        all_mkt = np.concatenate([d.iv_mkt for d in data[td]])
        rm = rmse_volpts(np.concatenate(fixed_lam_ivs(res, data[td], lam_star)), all_mkt)
        b = panel.at_bound(NAMES5, x, LB5, UB5)
        say(
            f"  {td} {x[0]:.5f} {x[1]:.5f} {x[2]:.5f} {x[3]:.4f} {x[4]:.3f} {res.cost:.4f}   "
            f"{rm:.3f}             {free[td]['rmse_ov']:.3f}            {b or '-'}"
        )
    say("")
    say("  day-over-day change at lam* (fixed-lam fits):")
    say("    param     " + "  ".join(f"{dates[i - 1]} -> {dates[i]}" for i in range(1, len(dates))))
    for j, name in enumerate(NAMES5):
        cells = [
            f"{fixed[dates[i]].x[j] - fixed[dates[i - 1]].x[j]:+.4f}" for i in range(1, len(dates))
        ]
        say(f"    {name:<5}   " + "  ".join(cells))
    for name in ("a0", "a1", "a2", "k", "rho"):
        vals = [float(fixed[td].x[NAMES5.index(name)]) for td in dates]
        say(f"  {name} range across dates: [{min(vals):.5f}, {max(vals):.5f}]")

    # Step 3: curve stability, free vs fixed-lam.
    say("")
    say("Step 3: sqrt(Q(tau)) in vol points per date, free fits (panel.csv) vs fixed-lam fits:")
    taus = np.array(CURVE_TAUS)
    header = "  date        role    " + "  ".join(f"tau={t:>6.4f}" for t in CURVE_TAUS)
    say(header)
    q_free: dict[str, NDArray[np.float64]] = {}
    q_fixed: dict[str, NDArray[np.float64]] = {}
    for td in dates:
        fr = free[td]
        q_free[td] = np.asarray(
            100.0 * np.sqrt(msv.Q(taus, fr["a0"], fr["a1"], fr["a2"], fr["lam"])),
            dtype=float,
        )
        fx = fixed[td].x
        q_fixed[td] = np.asarray(
            100.0 * np.sqrt(msv.Q(taus, float(fx[0]), float(fx[1]), float(fx[2]), lam_star)),
            dtype=float,
        )
        say(f"  {td} free   " + "  ".join(f"{v:8.3f}" for v in q_free[td]))
        say(f"  {td} fixed  " + "  ".join(f"{v:8.3f}" for v in q_fixed[td]))
    say(
        "  range free  "
        + "  ".join(
            f"{max(q_free[td][i] for td in dates) - min(q_free[td][i] for td in dates):8.3f}"
            for i in range(len(CURVE_TAUS))
        )
    )
    say(
        "  range fixed "
        + "  ".join(
            f"{max(q_fixed[td][i] for td in dates) - min(q_fixed[td][i] for td in dates):8.3f}"
            for i in range(len(CURVE_TAUS))
        )
    )
    say("")
    say(f"total runtime {time.perf_counter() - t_start:.0f}s")

    RESULTS.mkdir(exist_ok=True)
    MD_PATH.write_text(
        "# Fixed-lambda CRL refits across the panel dates\n\n"
        "Verbatim output of `uv run python experiments/panel_fixed_lambda.py` "
        "(derived numbers only; free fits read from `panel.csv`):\n\n"
        "```\n" + "\n".join(out) + "\n```\n"
    )
    print(f"\nWrote {MD_PATH}")


if __name__ == "__main__":
    main()
