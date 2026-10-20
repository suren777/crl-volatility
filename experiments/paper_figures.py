"""Paper figures (script-generated, vector PDF) for the CRL model.

Usage: uv run python experiments/paper_figures.py

Writes paper/figures/skew_asymptote.pdf, spx_smiles.pdf and spx_skew_term.pdf.
Data: the Cboe delayed-quote snapshot (spx.sqlite, trade_date 2026-10-16, roots
SPX/SPXW), read-only and outside the repo. Curve/IV helpers are reused by import
from crl_short.py / crl_spx.py / shifted_spx.py; the short-maturity skew
coefficient C(k, rho) of derivations.skew_asymptotics is reimplemented here
(decimal-units dI/dx, matching test_skew_asymptotics._c_coef).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import crl_short  # noqa: E402
import crl_spx  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import shifted_spx  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.special import ndtr, roots_hermitenorm  # noqa: E402

import msv  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

FIGDIR = Path(__file__).resolve().parents[1] / "paper" / "figures"
MAX_H = 0.02  # central-difference log-moneyness step coefficient: h = MAX_H sqrt(Q tau)


def c_coef(k: float, rho: float) -> float:
    """C(k, rho) = dI/dx(0)*sqrt(tau) (decimal units), by self-normalised quadrature."""
    s2 = float(np.log1p(k * k))
    sq = float(np.sqrt(s2))
    rho_t = rho / float(np.sqrt(1.0 - rho * rho))
    zz, ww = roots_hermitenorm(256)
    ww_ = ww / float(np.sum(ww))
    dz = rho_t * (zz - sq / 2.0 * np.exp(s2 / 8.0 - sq * zz / 2.0))
    return float(np.sqrt(2.0 * np.pi) * (0.5 - float(ww_ @ ndtr(dz))))


def crl_atm_skew_decimal(tau: float, k: float, rho: float, q: float) -> float:
    """ATM skew dI/dx(0) (decimal) of crl_call at tau, central difference +/-(max_h sqrt(Q tau))."""
    h = MAX_H * float(np.sqrt(q * tau))
    strikes = np.array([np.exp(-h), np.exp(h)])
    px = msv.crl_call(
        1.0,
        strikes,
        1.0,
        np.array([tau, tau]),
        0.0,
        0.0,
        q,
        1.0,
        k,
        rho,
        n_nodes=128,
    )
    iv_lo = shifted_spx.implied_iv(1.0, float(strikes[0]), tau, float(px[0]))
    iv_hi = shifted_spx.implied_iv(1.0, float(strikes[1]), tau, float(px[1]))
    assert iv_lo is not None and iv_hi is not None, f"IV inversion failed at tau={tau}"
    return float((iv_hi - iv_lo) / (2.0 * h))


def power_law(taus: NDArray[np.float64], s: NDArray[np.float64]) -> tuple[float, float]:
    """A, alpha of the log-log least-squares fit s ~= A tau^alpha (s > 0)."""
    slope, intercept = np.polyfit(np.log(taus), np.log(s), 1)
    return float(np.exp(intercept)), float(slope)


def fig_skew_asymptote() -> None:
    """|ATM skew| vs tau (log-log) for three (k, rho) sets, with the |C|/sqrt(tau) asymptote."""
    sets = ((1.0, -0.7), (0.5, -0.3), (2.0, -0.9))
    q = 0.04
    taus = np.geomspace(1e-4, 2.0, 40)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    markers = ("o", "s", "^")
    for (k, rho), mk in zip(sets, markers, strict=True):
        skew = np.array([crl_atm_skew_decimal(float(t), k, rho, q) for t in taus[::3]])
        c = abs(c_coef(k, rho))
        ax.loglog(taus, c / np.sqrt(taus), ls="-", lw=0.9, color="0.55")
        ax.loglog(
            taus[::3],
            np.abs(skew),
            ls="none",
            marker=mk,
            ms=4.5,
            mfc="none",
            color="k",
            label=f"$k={k}$, $\\rho={rho}$",
        )
    ax.plot([], [], ls="-", lw=0.9, color="0.55", label=r"asymptote $|\mathcal{C}|/\sqrt{\tau}$")
    ax.set_xlabel(r"$\tau$ (years)")
    ax.set_ylabel(r"$|\partial\sigma_{\mathrm{imp}}/\partial x|$ at the money")
    ax.legend(frameon=False, loc="lower left", title="markers: exact model", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "skew_asymptote.pdf")
    plt.close(fig)


def fig_spx_smiles() -> None:
    """Market vs joint-CRL implied-vol smiles, one panel per monthly expiry (Cboe)."""
    raw, spot = crl_short.load_raw("SPX")
    data_mon = [shifted_spx.load_expiry(raw, e, spot) for e in shifted_spx.EXPIRIES]
    res_mon, _t = crl_short.fit_crl_nonpos(data_mon)
    ivs = crl_spx.crl_model_ivs(res_mon, data_mon)
    months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.6), sharex=True)
    for ax, d, iv in zip(axes.flat, data_mon, ivs, strict=True):
        x = np.log(d.K / d.F)
        ax.plot(x, d.iv_mkt, ls="none", marker=".", ms=2.0, color="0.6", label="market", zorder=1)
        ax.plot(x, iv, ls="-", lw=1.3, color="k", label="CRL", zorder=2)
        label = f"{months[int(d.expiry[5:7]) - 1]} {d.expiry[:4]}"
        ax.text(0.97, 0.95, label, transform=ax.transAxes, va="top", ha="right", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlim(-0.3, 0.3)
    for ax in axes[1]:
        ax.set_xlabel(r"$\ln(K/F)$", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("implied volatility", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="center right", markerscale=3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "spx_smiles.pdf")
    plt.close(fig)


def fig_spx_skew_term() -> None:
    """|ATM skew| vs tau (log-log): market (ten expiries), CRL monthlies-fit, market power law."""
    raw_spx, spot = crl_short.load_raw("SPX")
    raw_spxw, spot_w = crl_short.load_raw("SPXW")
    data = [shifted_spx.load_expiry(raw_spxw, e, spot_w) for e in crl_short.WEEKLIES] + [
        shifted_spx.load_expiry(raw_spx, e, spot) for e in shifted_spx.EXPIRIES
    ]
    taus = np.array([d.tau for d in data])
    s_mkt = np.array([100.0 * shifted_spx.atm_skew(np.log(d.K / d.F), d.iv_mkt) for d in data])
    n_wk = len(crl_short.WEEKLIES)
    res_mon, _t = crl_short.fit_crl_nonpos(data[n_wk:])
    ivs_all = crl_spx.crl_model_ivs(res_mon, data)
    s_crl = np.array(
        [
            100.0 * shifted_spx.atm_skew(np.log(d.K / d.F), iv)
            for d, iv in zip(data, ivs_all, strict=True)
        ]
    )
    a, alpha = power_law(taus, np.abs(s_mkt))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(
        taus[:n_wk],
        np.abs(s_mkt[:n_wk]),
        ls="none",
        marker="^",
        ms=5,
        mfc="none",
        color="k",
        label="market (SPXW weeklies)",
    )
    ax.loglog(
        taus[n_wk:],
        np.abs(s_mkt[n_wk:]),
        ls="none",
        marker="o",
        ms=5,
        mfc="none",
        color="k",
        label="market (monthlies)",
    )
    ax.loglog(
        np.geomspace(taus.min(), taus.max(), 100),
        a * np.geomspace(taus.min(), taus.max(), 100) ** alpha,
        ls="-",
        color="0.5",
        label=f"market power law ($\\alpha$={alpha:.2f})",
    )
    ax.loglog(
        taus,
        np.abs(s_crl),
        ls="--",
        color="0.3",
        marker="s",
        ms=4,
        mfc="none",
        label="CRL (fitted to monthlies; weeklies out of sample)",
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([30, 50, 100, 200])
    ax.set_xlabel(r"$\tau$ (years)")
    ax.set_ylabel(r"$|\,\mathrm{ATM\ skew}\,|$ (vol points per unit $\ln K$)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "spx_skew_term.pdf")
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_skew_asymptote()
    fig_spx_smiles()
    fig_spx_skew_term()


if __name__ == "__main__":
    main()
