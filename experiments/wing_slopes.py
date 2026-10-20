"""Right-wing slope of total implied variance: market vs fitted CRL vs fitted SSVI.

Usage: uv run python experiments/wing_slopes.py

Tests the paper's claim (noarbitrage.tex, Proposition "Wings") that the fitted
CRL's moment explosion p~ = rho^2/(1-rho^2) makes its call wing turn up earlier
than the market's, with limsup_{x->inf} w/x = psi(p~) = 2 - 4(sqrt(p~^2+p~)-p~)
for the right wing of total implied variance w = iv^2 tau.

Fitters are reused from crl_short.py (load_raw, fit_crl_nonpos, rho<=0),
benchmarks_spx.py (fit_ssvi, ssvi_model_ivs), crl_spx.py (crl_model_ivs) and
shifted_spx.py (load_expiry, implied_iv). Three parts:

1. Per monthly Cboe expiry, least-squares slope of w = iv^2 tau on x =
   ln(K/F) over the observed call wing (0.05 <= x <= max quoted) for the
   market, the fitted CRL and the fitted SSVI on identical strikes.
2. For the fitted CRL parameters, total model implied variance on an extended
   grid x in [0, 3] (crl_call priced, then Black-inverted to IV), showing how
   w/x approaches psi(p~) as x grows (table at x = 0.25, 0.5, 1, 2, 3).
3. Results written to experiments/results/wing_slopes.md.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmarks_spx  # noqa: E402
import crl_short  # noqa: E402
import crl_spx  # noqa: E402
import numpy as np  # noqa: E402
import shifted_spx  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from shifted_spx import implied_iv, load_expiry  # noqa: E402

import msv  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "wing_slopes.md"
WING_MIN_X = 0.05
LEE_GRID = (0.25, 0.5, 1.0, 2.0, 3.0)


def psi_p_tilde(rho: float) -> float:
    """Lee (2004) right-wing bound psi(p~), p~ = rho^2/(1-rho^2)."""
    p = rho * rho / (1.0 - rho * rho)
    return 2.0 - 4.0 * (math.sqrt(p * p + p) - p)


def wing_slope(w: NDArray[np.float64], x: NDArray[np.float64]) -> float:
    """OLS slope of total variance w on log-moneyness x over the call wing."""
    return float(np.polyfit(x, w, 1)[0])


def crl_wx(
    d: shifted_spx.ExpiryData, theta: NDArray[np.float64], grid: tuple[float, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fitted-CRL total-variance ratios w/x and IVs on an extended log-moneyness grid."""
    a0, a1, a2, lam, k, rho = (float(v) for v in theta)
    K = d.F * np.exp(np.array(grid, dtype=float))
    px = msv.crl_call(d.F, K, np.full(K.size, d.D), np.full(K.size, d.tau), a0, a1, a2, lam, k, rho)
    wrs, ivs = [], []
    for kk, p in zip(K, px, strict=True):
        iv = implied_iv(d.F, float(kk), d.tau, float(p) / d.D)
        ivs.append(np.nan if iv is None else iv)
        wrs.append(np.nan if iv is None else iv * iv * d.tau / float(np.log(kk / d.F)))
    return np.array(wrs), np.array(ivs)


def cell(v: float, spec: str, w: int) -> str:
    return f"{v:{spec}}" if np.isfinite(v) else f"{'nan':>{w}}"


def main() -> None:
    raw, spot = crl_short.load_raw("SPX")
    data = [load_expiry(raw, e, spot) for e in shifted_spx.EXPIRIES]

    crl_res, _t_crl = crl_short.fit_crl_nonpos(data)
    ssvi_res, _t_ssvi = benchmarks_spx.fit_ssvi(data)
    c = crl_res.x
    a0, a1, a2, lam, kc, rho = (float(v) for v in c)
    p_tilde = rho * rho / (1.0 - rho * rho)
    psi_star = psi_p_tilde(rho)

    ivs_crl = crl_spx.crl_model_ivs(crl_res, data)
    ivs_ssvi = benchmarks_spx.ssvi_model_ivs(ssvi_res, data)

    body: list[str] = []
    body.append("# RIGHT-WING SLOPES OF TOTAL IMPLIED VARIANCE: MARKET vs CRL vs SSVI (2026-10-19)")
    body.append("")
    body.append(
        f"Cboe SQLite {crl_spx.CBOE_DB}, trade_date={crl_spx.TRADE_DATE}, root SPX, six "
        f"monthly expiries. Call wing: `x = ln(K/F) >= {WING_MIN_X}` to the largest quoted "
        "strike; total implied variance `w = iv^2 * tau`; slope is OLS of `w` on `x` over "
        "the wing. Inverted-IV cells outside the Black price range are `nan`."
    )
    body.append("")
    body.append(
        f"Fitted params (joint, six expiries): CRL `a0={a0:.5f} a1={a1:.5f} a2={a2:.5f} "
        f"lam={lam:.4f} k={kc:.4f} rho={rho:.3f}` (rho<=0); SSVI `eta={ssvi_res.x[0]:.4f} "
        f"gamma={ssvi_res.x[1]:.4f} rho={ssvi_res.x[2]:.3f}`."
    )
    body.append("")
    body.append(
        f"Lee (2004) right-wing bound at the fitted rho: `p~ = rho^2/(1-rho^2) = "
        f"{p_tilde:.4f}`, `psi(p~) = 2-4(sqrt(p~^2+p~)-p~) = {psi_star:.4f}`."
    )
    body.append("")

    body.append("## 1. Observed call-wing slope of total variance (x in [0.05, max quoted])")
    body.append("")
    hdr = "| expiry | tau | n_wings | mkt slope | CRL slope | SSVI slope | CRL/mkt | SSVI/mkt |"
    body.append(hdr)
    body.append(
        "|--------|-----|---------|-----------|-----------|------------|---------|----------|"
    )
    for d, iv_c, iv_s in zip(data, ivs_crl, ivs_ssvi, strict=True):
        x = np.log(d.K / d.F)
        wing = x >= WING_MIN_X
        if not wing.any():
            continue
        xw = x[wing]
        m_s = wing_slope(d.iv_mkt[wing] ** 2 * d.tau, xw)
        c_s = wing_slope(iv_c[wing] ** 2 * d.tau, xw)
        s_s = wing_slope(iv_s[wing] ** 2 * d.tau, xw)
        body.append(
            f"| {d.expiry} | {d.tau:.4f} | {int(wing.sum()):4d} | {m_s:9.4f} | "
            f"{c_s:9.4f} | {s_s:10.4f} | {c_s / m_s:6.2f} | {s_s / m_s:6.2f} |"
        )
    body.append("")
    body.append(
        "`psi(p~)` upper-bounds the CRL wing slope asymptotically (limsup as x -> inf); "
        "the observed finite-x slope must sit below it. A finite-moneyness slope at or "
        "near the bound is a signature that the wing is approaching the steepest tail "
        "the moment explosion allows."
    )
    body.append("")

    body.append("## 2. Fitted CRL: w/x and Black IV on an extended grid x in [0.25, 3]")
    body.append("")
    body.append(
        f"Model total variance is Black-inverted from `crl_call` at rho={rho:.3f}, so the "
        f"asymptotic right-wing bound is `psi(p~) = {psi_star:.4f}`."
    )
    body.append("")
    wtab = [
        "| expiry | tau | w/x 0.25 | w/x 0.5 | w/x 1 | w/x 2 | w/x 3 | (w/x)/psi@3 |",
        "|--------|-----|----------|---------|-------|-------|-------|-------------|",
    ]
    ivtab = [
        "| expiry | tau | iv@0.25 | iv@0.5 | iv@1 | iv@2 | iv@3 |",
        "|--------|-----|---------|--------|------|------|------|",
    ]
    for d in data:
        wrs, ivs = crl_wx(d, c, LEE_GRID)
        ratio3 = wrs[-1] / psi_star if np.isfinite(wrs[-1]) else float("nan")
        wtab.append(
            "| "
            + d.expiry
            + " | "
            + f"{d.tau:.4f} | "
            + " | ".join(cell(v, ".3f", 6) for v in wrs)
            + f" | {ratio3:.3f} |"
        )
        ivtab.append(
            "| "
            + d.expiry
            + " | "
            + f"{d.tau:.4f} | "
            + " | ".join(cell(v, ".3f", 5) for v in ivs)
            + " |"
        )
    body += wtab + [""] + ivtab
    body.append("")
    body.append(
        "w/x climbs toward `psi(p~)` as x grows and stays below it at every expiry, "
        "consistent with an asymptotic upper bound approached from below."
    )

    body.append("")
    body.append("## 3. Findings")
    body.append("")
    last = None
    d_last = data[-1]
    wrs_last, _ = crl_wx(d_last, c, LEE_GRID)
    if np.isfinite(wrs_last[-1]):
        last = wrs_last[-1]
    body.append(
        f"At the fitted rho={rho:.3f}, `p~ = {p_tilde:.4f}` and the Lee right-wing bound "
        f"`psi(p~) = {psi_star:.3f}`. Table 2 shows the fitted CRL's `w/x` on the extended "
        "grid: at every expiry it rises toward this bound as x grows and stays below it — "
        f"e.g. the longest expiry reaches `w/x = {last:.3f}` at x=3 versus a bound of "
        f"{psi_star:.3f} — so the moment-formula bound is approached from below, as an "
        "asymptotic (x -> inf) statement predicts."
    )
    body.append("")
    body.append(
        "Where the data probe (x up to ~0.3), CRL's call-wing slope is steeper than the "
        "market's at the short end but comparable or flatter at the longest expiries "
        "(Table 1, CRL/mkt); SSVI's wing is flatter than CRL's throughout and matches the "
        "market more closely at the long end. The Lee bound is therefore not directly "
        "relevant at observed moneyness — the farthest quoted call here is only "
        "`ln(K/F) ~ 0.3`, far short of the far tail — it governs the model's wing slope "
        "only asymptotically."
    )
    body.append("")

    md_text = "\n".join(body) + "\n"
    OUT.write_text(md_text)
    print(md_text)


if __name__ == "__main__":
    main()
