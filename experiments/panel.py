"""Day-to-day stability panel of the fitted models across SPX trade dates.

Usage: uv run python experiments/panel.py

For every trade date present in the Cboe SQLite snapshots table, builds the six
SPX monthly expiries in [3 weeks, 12 months] and fits plain MSV, CRL (rho <= 0),
SSVI and Heston on the same OTM, vega-weighted objective as benchmarks_spx.py.
Reports per-date parameters, IV RMSE (overall and per expiry) and fit times;
for CRL, the day-over-day parameter changes and any parameter at a bound.

tau is measured from each trade date (shifted_spx.load_expiry and its ASOF are
for 2026-10-16 only, so a local loader rebuilds ExpiryData with the correct
tau). Reads the DB read-only; writes only derived numbers to
experiments/results/panel.csv (one row per trade_date x model) and summarises
in experiments/results/panel.md. New dates extend the CSV automatically.
"""

import csv
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmarks_spx  # noqa: E402
import crl_short  # noqa: E402
import crl_spx  # noqa: E402
import numpy as np  # noqa: E402
import shifted_spx  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.optimize import OptimizeResult  # noqa: E402
from shifted_spx import ExpiryData, implied_iv, rmse_volpts  # noqa: E402

DB = crl_spx.CBOE_DB
ROOT = "SPX"
MIN_DAYS, MAX_DAYS = 21, 365
# Paper's six monthlies (shifted_spx.EXPIRIES): Nov, Dec, Jan then quarterly
# Mar/Jun/Sep. Kept among monthlies in range; backfilled to six by nearest.
MV_MONTHS = (1, 3, 6, 9, 11, 12)
N_EXP = 6
RESULTS = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS / "panel.csv"
MD_PATH = RESULTS / "panel.md"
CRL_LB = (0.0, 0.0, 0.0, 1e-3, 1e-3, -0.99)
CRL_UB = (np.inf, np.inf, np.inf, 50.0, 3.0, 0.0)
CRL_NAMES = ("a0", "a1", "a2", "lam", "k", "rho")
MODELS = ("msv", "crl", "ssvi", "heston")


def load_raw(db: str, trade_date: str) -> tuple[list[dict[str, str]], float]:
    """SPX quotes for one trade date in load_expiry's dict format; bid > 0, ask > bid."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT spot FROM snapshots WHERE trade_date = ?", (trade_date,)
        ).fetchone()
        assert row is not None, f"no snapshot for {trade_date} in {db}"
        spot = float(row[0])
        rows = con.execute(
            "SELECT expiry, cp, strike, bid, ask FROM quotes WHERE trade_date = ? AND root = ?",
            (trade_date, ROOT),
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


def pick_expiries(raw: list[dict[str, str]], asof: date) -> list[str]:
    """Six SPX monthlies in [3 weeks, 12 months], preferring the paper's set."""
    expiries = []
    for r in raw:
        days = (date.fromisoformat(r["expiry"]) - asof).days
        if MIN_DAYS <= days <= MAX_DAYS and r["expiry"] not in expiries:
            expiries.append(r["expiry"])
    expiries.sort(key=lambda e: (date.fromisoformat(e) - asof).days)
    preferred = [e for e in expiries if date.fromisoformat(e).month in MV_MONTHS]
    selected = preferred[:N_EXP]
    for e in expiries:
        if len(selected) >= N_EXP:
            break
        if e not in selected:
            selected.append(e)
    return sorted(selected)


def load_expiry_asof(raw: list[dict[str, str]], expiry: str, spot: float, asof: date) -> ExpiryData:
    """Same filters as shifted_spx.load_expiry but tau measured from asof, not ASOF."""
    rows = [r for r in raw if r["expiry"] == expiry]
    tau = (date.fromisoformat(expiry) - asof).days / 365.0
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
        if abs(np.log(k / F)) > shifted_spx.MAX_LN_MONEYNESS:
            continue
        side = "P" if k < F else "C"
        if k not in quotes[side]:
            continue
        bid, ask = quotes[side][k]
        quoted_mid = (bid + ask) / 2
        if ask <= 0 or quoted_mid <= 0 or (ask - bid) / quoted_mid > shifted_spx.MAX_REL_SPREAD:
            continue
        mid = quoted_mid + D * (F - k) if side == "P" else quoted_mid
        iv = implied_iv(F, k, tau, mid / D)
        if iv is None:
            continue
        ks.append(k)
        mids.append(mid)
        ivs.append(iv)
    return ExpiryData(expiry, tau, F, D, np.array(ks), np.array(mids), np.array(ivs))


def at_bound(
    names: tuple[str, ...], x: NDArray[np.float64], lb: tuple[float, ...], ub: tuple[float, ...]
) -> str:
    """Comma-separated params within an abs tol of a finite bound, or ''."""
    hit = []
    for name, v, lo, hi in zip(names, x.tolist(), lb, ub, strict=True):
        if (lo > -np.inf and abs(v - lo) < 1e-6) or (hi < np.inf and abs(v - hi) < 1e-6):
            hit.append(name)
    return ",".join(hit)


def fit_all(data: list[ExpiryData], spot: float) -> dict[str, tuple[OptimizeResult, float]]:
    plain, t_p = shifted_spx.fit_model(data, shifted=False, spot=spot)
    crl, t_c = crl_short.fit_crl_nonpos(data)
    ssvi, t_s = benchmarks_spx.fit_ssvi(data)
    heston, t_h = benchmarks_spx.fit_heston(data)
    return {"msv": (plain, t_p), "crl": (crl, t_c), "ssvi": (ssvi, t_s), "heston": (heston, t_h)}


def model_ivs(name: str, res: OptimizeResult, data: list[ExpiryData]) -> list[NDArray[np.float64]]:
    if name == "msv":
        return shifted_spx.model_ivs(res, data, shifted=False)
    if name == "crl":
        return crl_spx.crl_model_ivs(res, data)
    if name == "ssvi":
        return benchmarks_spx.ssvi_model_ivs(res, data)
    return benchmarks_spx.heston_model_ivs(res, data)


def param_text(name: str, x: list[float], res: OptimizeResult, extra: str = "") -> str:
    if name == "msv":
        return f"a0={x[0]:.5f} a1={x[1]:.5f} a2={x[2]:.5f} lam={x[3]:.4f} k={x[4]:.4f}"
    if name == "crl":
        b = at_bound(CRL_NAMES, np.array(x), CRL_LB, CRL_UB)
        flag = f"  AT_BOUND={b}" if b else ""
        return (
            f"a0={x[0]:.5f} a1={x[1]:.5f} a2={x[2]:.5f} lam={x[3]:.4f} k={x[4]:.4f} "
            f"rho={x[5]:.3f} (rho<=0){flag}"
        )
    if name == "ssvi":
        return f"eta={x[0]:.4f} gamma={x[1]:.4f} rho={x[2]:.3f}"
    return (
        f"v0={x[0]:.5f} kappa={x[1]:.4f} theta={x[2]:.5f} sigma={x[3]:.4f} "
        f"rho={x[4]:.3f} feller={2 * x[1] * x[2] / x[3] ** 2:.3f}"
    )


@dataclass
class PanelRow:
    """One fitted model on one trade date. Model-specific params are None when absent."""

    trade_date: str
    model: str
    expiries: str
    n_e: str
    n_options: int
    cost: float
    fit_time_s: float
    rmse_ov: float
    rmse_e: str
    params: str
    at_bound: str = ""
    feller: float | None = None
    a0: float | None = None
    a1: float | None = None
    a2: float | None = None
    lam: float | None = None
    k: float | None = None
    rho: float | None = None
    eta: float | None = None
    gamma: float | None = None
    v0: float | None = None
    kappa: float | None = None
    theta: float | None = None
    sigma: float | None = None
    theta_1: float | None = None
    theta_2: float | None = None
    theta_3: float | None = None
    theta_4: float | None = None
    theta_5: float | None = None
    theta_6: float | None = None

    def param(self, name: str) -> float:
        return float(getattr(self, name))


def make_row(
    td: str,
    name: str,
    res: OptimizeResult,
    ftime: float,
    data: list[ExpiryData],
    expiries: list[str],
    ssvi_thetas: list[float] | None,
    feller: float | None,
) -> PanelRow:
    ivs = model_ivs(name, res, data)
    all_mkt = np.concatenate([d.iv_mkt for d in data])
    row = PanelRow(
        trade_date=td,
        model=name,
        expiries=";".join(expiries),
        n_e=";".join(str(d.K.size) for d in data),
        n_options=int(all_mkt.size),
        cost=float(res.cost),
        fit_time_s=float(ftime),
        rmse_ov=float(rmse_volpts(np.concatenate(ivs), all_mkt)),
        rmse_e=";".join(
            f"{rmse_volpts(iv, d.iv_mkt):.3f}" for iv, d in zip(ivs, data, strict=True)
        ),
        params=param_text(name, res.x.tolist(), res),
    )
    if name in ("msv", "crl"):
        row.a0, row.a1, row.a2, row.lam, row.k = (float(v) for v in res.x[:5])
        if name == "crl":
            row.rho = float(res.x[5])
            row.at_bound = at_bound(CRL_NAMES, res.x, CRL_LB, CRL_UB)
    elif name == "ssvi":
        row.eta, row.gamma, row.rho = (float(v) for v in res.x[:3])
        if ssvi_thetas is not None:
            for i, th in enumerate(ssvi_thetas):
                setattr(row, f"theta_{i + 1}", float(th))
    else:
        row.v0, row.kappa, row.theta, row.sigma, row.rho = (float(v) for v in res.x[:5])
        row.feller = feller
    return row


def main() -> None:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        dates = [
            str(r[0]) for r in con.execute("SELECT trade_date FROM snapshots ORDER BY trade_date")
        ]
    finally:
        con.close()
    RESULTS.mkdir(exist_ok=True)

    rows: list[PanelRow] = []
    crl_prev: dict[str, float] | None = None
    prev_date: str | None = None

    for td in dates:
        raw, spot = load_raw(DB, td)
        asof = date.fromisoformat(td)
        expiries = pick_expiries(raw, asof)
        data = [load_expiry_asof(raw, e, spot, asof) for e in expiries]
        t0 = time.perf_counter()
        fits = fit_all(data, spot)
        fit_total = time.perf_counter() - t0

        per_exp_head = {e: data[expiries.index(e)] for e in expiries}
        print(
            f"\n{'=' * 78}\ntrade_date {td}  spot={spot:.2f}  ({len(expiries)} expiries, "
            f"all fits {fit_total:.1f}s)"
        )
        print(
            f"\nPer-expiry parity forward/discount and OTM fit set "
            f"(|ln(K/F)|<={shifted_spx.MAX_LN_MONEYNESS}):"
        )
        print(f"{'expiry':<12} {'tau':>7} {'F':>10} {'D':>8} {'n_options':>10}")
        for e in expiries:
            d = per_exp_head[e]
            print(f"{e:<12} {d.tau:>7.4f} {d.F:>10.2f} {d.D:>8.5f} {d.K.size:>10d}")

        print("\nFitted parameters (joint over expiries) and fit time:")
        for name in MODELS:
            res, ftime = fits[name]
            print(
                f"  {name:7}: {param_text(name, res.x.tolist(), res)}  "
                f"cost={res.cost:.4f}  [{ftime:.1f}s]"
            )

        print("\nIV RMSE (vol points):")
        print(f"{'expiry':<12} {'n':>4} {'MSV':>7} {'CRL':>7} {'SSVI':>7} {'Heston':>7}")
        for d in data:
            vals = [
                rmse_volpts(model_ivs(n, fits[n][0], data)[data.index(d)], d.iv_mkt) for n in MODELS
            ]
            print(
                f"{d.expiry:<12} {d.K.size:>4} {vals[0]:>7.3f} {vals[1]:>7.3f} "
                f"{vals[2]:>7.3f} {vals[3]:>7.3f}"
            )
        all_mkt = np.concatenate([d.iv_mkt for d in data])
        ov = [rmse_volpts(np.concatenate(model_ivs(n, fits[n][0], data)), all_mkt) for n in MODELS]
        print(
            f"{'overall':<12} {all_mkt.size:>4} {ov[0]:>7.3f} {ov[1]:>7.3f} "
            f"{ov[2]:>7.3f} {ov[3]:>7.3f}"
        )

        # assemble rows
        ssvi_res, _ = fits["ssvi"]
        s_x = ssvi_res.x
        thetas = np.exp(s_x[3]) + np.cumsum(
            np.concatenate([[0.0], np.exp(np.minimum(s_x[4:], 30.0))])
        )
        h_res, _ = fits["heston"]
        h_x = h_res.x.tolist()
        feller = 2 * h_x[1] * h_x[2] / h_x[3] ** 2
        for name in MODELS:
            res, ftime = fits[name]
            rows.append(
                make_row(
                    td,
                    name,
                    res,
                    ftime,
                    data,
                    expiries,
                    thetas[: len(data)].tolist() if name == "ssvi" else None,
                    feller if name == "heston" else None,
                )
            )

        crl_res = fits["crl"][0].x
        crl_cur = {n: float(v) for n, v in zip(CRL_NAMES, crl_res.tolist(), strict=True)}
        print("\nCRL day-over-day parameter change:")
        if crl_prev is None:
            print("  (only one date so far — no delta)")
        else:
            print(f"  delta {prev_date} -> {td}:")
            for n in CRL_NAMES:
                print(
                    f"    {n:<4} {crl_prev[n]:+.4f} -> {crl_cur[n]:+.4f} "
                    f"(d={crl_cur[n] - crl_prev[n]:+.4f})"
                )
        crl_prev = crl_cur
        prev_date = td

    fieldnames = [
        "trade_date",
        "model",
        "expiries",
        "n_e",
        "n_options",
        "cost",
        "fit_time_s",
        "rmse_ov",
        "rmse_e",
        "params",
        "a0",
        "a1",
        "a2",
        "lam",
        "k",
        "rho",
        "eta",
        "gamma",
        "v0",
        "kappa",
        "theta",
        "sigma",
        "at_bound",
        "feller",
    ] + [f"theta_{i}" for i in range(1, N_EXP + 1)]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"\nWrote {CSV_PATH} ({len(rows)} rows) across dates {dates}")

    write_md(dates, rows)
    print(f"Wrote {MD_PATH}")


def _row(rows: list[PanelRow], td: str, model: str) -> PanelRow:
    return [r for r in rows if r.trade_date == td and r.model == model][0]


def write_md(dates: list[str], rows: list[PanelRow]) -> None:
    lines: list[str] = []
    lines.append("# Day-to-day stability panel of the fitted models (preview)")
    lines.append("")
    if len(dates) < 3:
        lines.append(
            f"> **Preview, not a panel.** Only {len(dates)} trade date(s) are in the collector "
            "snapshot cache. Earlier scripts, benchmarks and the CRL short-end work used "
            "`2026-10-16`; this panel grows by one row per model per date as the collector adds "
            "trading days. Fewer than ~5-10 dates support only a provisional read of "
            "day-to-day stability."
        )
        lines.append("")
    lines.append(
        "For each trade date in `spx.sqlite`, six SPX monthly expiries in [3 weeks, 12 months] "
        "from that date (the paper's set: Nov/Dec/Jan then quarterly Mar/Jun/Sep, backfilled to "
        "six by the nearest remaining monthlies) are fitted jointly for plain MSV, CRL "
        "(`rho <= 0`), SSVI and Heston on the same OTM, vega-weighted price objective as "
        "`benchmarks_spx.py`. tau is measured from each trade date; `F`/`D` from put-call "
        "parity. All prices are Cboe SPX OTM mids; IV RMSE is in vol points. Derived numbers "
        "only are stored in `panel.csv`."
    )
    lines.append("")

    crl_by_date = [r for r in rows if r.model == "crl"]
    for td in dates:
        lines.append(f"## {td}")
        lines.append("")
        lines.append("Fitted parameters (joint over six expiries), cost and fit time:")
        lines.append("")
        lines.append("| model | parameters | cost | time (s) | IV RMSE (vol pts) | n |")
        lines.append("|-------|------------|------|----------|-------------------|-----|")
        for model in MODELS:
            sub = _row(rows, td, model)
            lines.append(
                f"| {model:8} | {sub.params} | {sub.cost:.4f} | "
                f"{sub.fit_time_s:.1f} | {sub.rmse_ov:.3f} | {sub.n_options} |"
            )
        lines.append("")
        lines.append("Per-expiry IV RMSE (vol points):")
        lines.append("")
        msv_row = _row(rows, td, "msv")
        exps = msv_row.expiries.split(";")
        ns = [int(v) for v in msv_row.n_e.split(";")]
        rmbys = {
            model: [float(v) for v in _row(rows, td, model).rmse_e.split(";")] for model in MODELS
        }
        lines.append(
            f"| {'expiry':<12} | {'n':>4} | {'MSV':>7} | {'CRL':>7} | {'SSVI':>7} | {'Heston':>7} |"
        )
        lines.append("|" + "---|" * 6)
        for i, (e, n) in enumerate(zip(exps, ns, strict=True)):
            lines.append(
                f"| {e:<12} | {n:>4} | {rmbys['msv'][i]:>7.3f} "
                f"| {rmbys['crl'][i]:>7.3f} | {rmbys['ssvi'][i]:>7.3f} "
                f"| {rmbys['heston'][i]:>7.3f} |"
            )
        n_tot = sum(ns)
        ovl = [_row(rows, td, model).rmse_ov for model in MODELS]
        lines.append(
            f"| {'overall':<12} | {n_tot:>4} | {ovl[0]:.3f} | {ovl[1]:.3f} "
            f"| {ovl[2]:.3f} | {ovl[3]:.3f} |"
        )
        lines.append("")

    if len(crl_by_date) >= 2:
        lines.append("## CRL day-over-day parameter change")
        lines.append("")
        lines.append("| date | a0 (Δ) | a1 (Δ) | a2 (Δ) | lam (Δ) | k (Δ) | rho (Δ) |")
        lines.append("|------|--------|--------|--------|---------|-------|---------|")
        prev: dict[str, float] | None = None
        for r in crl_by_date:
            cur = {n: r.param(n) for n in CRL_NAMES}
            if prev is None:
                cells = [f"{cur[n]:.4f}" for n in CRL_NAMES]
            else:
                cells = [f"{cur[n]:.4f} ({cur[n] - prev[n]:+.4f})" for n in CRL_NAMES]
            lines.append(f"| {r.trade_date} | " + " | ".join(cells) + " |")
            prev = cur
        lines.append("")
        lines.append("`(Δ)` is the change from the previous trade date.")

    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Single/paired trading days only — this is a preview, not a stability panel. "
        "The day-over-day CRL deltas over a two-date window say almost nothing yet; "
        "revisit once the collector has a few weeks."
    )
    lines.append(
        "- tau is measured per trade date and expiries are re-picked per date, so the "
        "near-term rollover is already reflected."
    )
    lines.append(
        "- CRL's rho is bounded to [-0.99, 0]; an optimiser-pinned parameter (see "
        "`at_bound`) would indicate an identification issue, not necessarily "
        "instability."
    )
    lines.append(
        "- Per-expiry option counts are in `panel.csv` (`n_e`); the md table repeats "
        "them through the MSV row's expiry order."
    )
    lines.append("")
    MD_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
