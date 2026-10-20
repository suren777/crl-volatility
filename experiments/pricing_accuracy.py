"""CRL exact pricer: accuracy and speed of msv.crl_call vs high-node quadrature and MC.

The exact CRL call (msv.crl_call) is a one-dimensional Gauss-Hermite expectation.
This script measures, over a grid of (k, rho, tau, model curve, K/F):

1. Convergence to a 256-node reference: for n_nodes in {16, 32, 64, 128}, max
   absolute price error and max implied-vol error (vol points) across the grid.
2. That the 256-node reference is itself accurate: max |quadrature - MC| over the
   grid in units of the Monte-Carlo standard error, where MC samples the terminal
   law (eq:terminal), i.e. Z and W standard normal, 4e6 draws, with the model's
   normalizer m(tau) taken from the same 256-node quadrature.
3. Speed: wall time to price 1,000 options in one elementwise call at each node
   count (median of 20 runs), and the time of one objective evaluation as used in
   calibration (one crl_call at 64 nodes over the whole grid option set in a
   single call, median of 20 runs).

Run: uv run python experiments/pricing_accuracy.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from numpy.polynomial.hermite_e import hermegauss  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

import msv  # noqa: E402
from derivations.checks import black_fwd  # noqa: E402

F = 1.0
D = 1.0
NV_REF = 256
NODES = (16, 32, 64, 128)
N_MC = 4_000_000
KS = (0.5, 1.0, 2.3)
RHOS = (-0.9, -0.5, -0.2, 0.0)
TAUS = (1 / 52, 1 / 12, 1 / 4, 1.0, 3.0)
KFR = (0.7, 0.85, 1.0, 1.15, 1.3)
CURVES: dict[str, tuple[float, float, float, float]] = {
    "flat 0.04": (0.0, 0.0, 0.04, 1.0),
    "fitted SPX": (0.0, 0.0921, 0.0196, 0.5167),
}
CONFIG_COUNT = len(KS) * len(RHOS) * len(TAUS) * len(CURVES)
# An option's implied vol is only meaningful when it carries material time value;
# a deep-ITM short-dated call is worth near its intrinsic payoff, where tiny price
# changes swing IV enormously. The IV-error metric therefore counts cells whose
# 256-node price has at least this (relative) discount from the intrinsic bound.
TV_TOL = 1e-3
# For the |Q256-MC|/SE comparison, the MC standard error is only a valid disagreement
# scale when the option has material price (enough payoff-hitting paths): for a call
# worth ~0 the SE shrinks to ~1e-10 and the ratio is inflated by an immaterial ~1e-8
# price difference. MC-price floor above which the ratio is reported as meaningful.
MC_PRICE_FLOOR = 1e-4

CurveTuple = tuple[float, float, float, float]
QKey = tuple[str, float, float, float]  # curve name, k, rho, tau
LocTuple = tuple[str, float, float, float, float]  # ... plus K/F
MCResult = dict[str, float | LocTuple | NDArray[np.float64] | None]


def option_arrays(
    tau: float, kf: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Elementwise-paired F, K, D, tau arrays for one (tau) set of K/F values."""
    n = kf.size
    return np.full(n, F), kf, np.full(n, D), np.full(n, tau)


def crl_price(
    curve: CurveTuple,
    k: float,
    rho: float,
    tau: float,
    kfr: NDArray[np.float64],
    n_nodes: int,
) -> NDArray[np.float64]:
    Fa, Kc, Da, tc = option_arrays(tau, kfr)
    return msv.crl_call(Fa, Kc, Da, tc, *curve, k, rho, n_nodes=n_nodes)


def implied_vol(K: float, tau: float, price: float) -> float | None:
    """Volatility inverting forward Black (D cancels; D=1 here). None if the price
    is outside Black's reachable range (deep-ITM short-dated options sit at their
    intrinsic value, where IV is degenerate) - same convention as shifted_spx.implied_iv."""
    try:
        return float(
            brentq(lambda s: float(black_fwd(F, K, tau, s)) - price, 1e-6, 5.0, xtol=1e-12)
        )
    except ValueError:
        return None


def reference_prices() -> dict[QKey, NDArray[np.float64]]:
    ref: dict[QKey, NDArray[np.float64]] = {}
    for name, curve in CURVES.items():
        for k in KS:
            for rho in RHOS:
                for tau in TAUS:
                    ref[(name, k, rho, tau)] = crl_price(curve, k, rho, tau, np.array(KFR), NV_REF)
    return ref


def node_errors(ref: dict[QKey, NDArray[np.float64]]) -> dict[int, tuple[float, float]]:
    """Per node count: (max |price err|, max |IV err| in vol points) over the grid."""
    out: dict[int, tuple[float, float]] = {}
    for n in NODES:
        max_pe, max_ive = 0.0, 0.0
        for (name, k, rho, tau), pr in ref.items():
            pn = crl_price(CURVES[name], k, rho, tau, np.array(KFR), n)
            max_pe = max(max_pe, float(np.max(np.abs(pn - pr))))
            for j, kf in enumerate(KFR):
                pr_j = float(pr[j])
                extr = max(F - kf, 0.0)  # intrinsic bound for a call
                # only IV-meaningful cells (see TV_TOL): positive price with material time value
                if pr_j > 0.0 and (pr_j - extr) / pr_j >= TV_TOL:
                    iv_n = implied_vol(kf, tau, float(pn[j]))
                    iv_r = implied_vol(kf, tau, pr_j)
                    if iv_n is not None and iv_r is not None:
                        max_ive = max(max_ive, abs(iv_n - iv_r) * 100.0)
        out[n] = (max_pe, max_ive)
    return out


def normaliser(k: float, rho: float, tau: float, curve: CurveTuple) -> float:
    """E_quad[M(Z,tau)] at NV_REF nodes, the normalizer m(tau) used by the model."""
    x, w = hermegauss(NV_REF)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    s2 = np.log1p(k * k)
    ss = np.sqrt(s2)
    zeta = np.exp(ss * x - s2 / 2.0)
    Vt = zeta * float(msv.Q(tau, *curve)) * tau
    M = np.exp(rho * np.sqrt(Vt) * x - 0.5 * rho * rho * Vt)
    return float((w / np.sqrt(2.0 * np.pi)) @ M)


def mc_vs_quad(ref: dict[QKey, NDArray[np.float64]]) -> MCResult:
    """Sample the terminal law; return max ratio |quad-MC|/SE, max SE, MC max err."""
    rng = np.random.default_rng(20261019)
    Z = rng.standard_normal(N_MC)
    W = rng.standard_normal(N_MC)

    max_ratio = 0.0
    max_ratio_loc: LocTuple | None = None
    max_ratio_mat = 0.0
    max_ratio_mat_loc: LocTuple | None = None
    max_abs = 0.0
    max_se = 0.0
    ill_ref: NDArray[np.float64] | None = None
    ill_mc: NDArray[np.float64] | None = None
    ill_se: NDArray[np.float64] | None = None
    ill_ratio: NDArray[np.float64] | None = None

    for name, curve in CURVES.items():
        for k in KS:
            s2 = np.log1p(k * k)
            ss = np.sqrt(s2)
            for rho in RHOS:
                for tau in TAUS:
                    qv = float(msv.Q(tau, *curve))
                    zeta = np.exp(ss * Z - s2 / 2.0)
                    Vt = zeta * qv * tau
                    m_val = normaliser(k, rho, tau, curve)
                    Mfac = np.exp(rho * np.sqrt(Vt) * Z - 0.5 * rho * rho * Vt)
                    St = (
                        F
                        * (Mfac / m_val)
                        * np.exp(
                            -0.5 * (1.0 - rho * rho) * Vt + np.sqrt((1.0 - rho * rho) * Vt) * W
                        )
                    )
                    mc = np.empty(len(KFR))
                    se = np.empty(len(KFR))
                    ratio = np.empty(len(KFR))
                    for j, kf in enumerate(KFR):
                        payoff = np.maximum(St - kf, 0.0)
                        mc[j] = float(np.mean(payoff))
                        se[j] = float(np.std(payoff) / np.sqrt(N_MC))
                        # skip cells with no empirical variance (price exactly 0 across all
                        # draws): there the ratio is ill-defined, not a disagreement.
                        ratio[j] = (
                            abs(float(ref[(name, k, rho, tau)][j]) - mc[j]) / se[j]
                            if se[j] > 0.0
                            else 0.0
                        )
                    jmat = [j for j in range(len(KFR)) if mc[j] >= MC_PRICE_FLOOR]
                    if jmat:
                        jm = jmat[int(np.argmax([ratio[j] for j in jmat]))]
                        if ratio[jm] > max_ratio_mat:
                            max_ratio_mat = float(ratio[jm])
                            max_ratio_mat_loc = (name, k, rho, tau, KFR[jm])
                    jmax = int(np.argmax(ratio))
                    if ratio[jmax] > max_ratio:
                        max_ratio = float(ratio[jmax])
                        max_ratio_loc = (name, k, rho, tau, KFR[jmax])
                        ill_ref = ref[(name, k, rho, tau)]
                        ill_mc = mc
                        ill_se = se
                        ill_ratio = ratio
                    max_abs = max(max_abs, float(np.max(np.abs(ref[(name, k, rho, tau)] - mc))))
                    max_se = max(max_se, float(np.max(se)))

    assert max_ratio_loc is not None and max_ratio_mat_loc is not None
    assert ill_ref is not None
    return {
        "max_ratio": max_ratio,
        "max_ratio_loc": max_ratio_loc,
        "max_ratio_mat": max_ratio_mat,
        "max_ratio_mat_loc": max_ratio_mat_loc,
        "max_abs": max_abs,
        "max_se": max_se,
        "ill_ref": ill_ref,
        "ill_mc": ill_mc,
        "ill_se": ill_se,
        "ill_ratio": ill_ratio,
    }


def timing_arrays(
    n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """n elementwise-paired options spanning the grid (F = D = 1)."""
    kf = np.array(KFR)[np.random.default_rng(0).integers(0, len(KFR), n)]
    tau = np.array(TAUS)[np.random.default_rng(1).integers(0, len(TAUS), n)]
    return np.full(n, F), kf, np.full(n, D), tau.astype(float)


def time_call(
    n_nodes: int,
    n_opts: int,
    curve: CurveTuple,
    k: float,
    rho: float,
    reps: int = 20,
) -> float:
    Fa, Kc, Da, tc = timing_arrays(n_opts)
    times = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter()
        msv.crl_call(Fa, Kc, Da, tc, *curve, k, rho, n_nodes=n_nodes)
        times[i] = time.perf_counter() - t0
    return float(np.median(times))


def _f(mc: MCResult, key: str) -> float:
    v = mc[key]
    assert isinstance(v, float)
    return v


def _loc(mc: MCResult, key: str) -> LocTuple:
    v = mc[key]
    assert isinstance(v, tuple)
    return v


def _arr(mc: MCResult, key: str) -> NDArray[np.float64]:
    v = mc[key]
    assert isinstance(v, np.ndarray)
    return v


def _all_grid_options() -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Every (curve, k, rho, tau, K/F) grid option as one elementwise-paired set."""
    per = len(KFR)
    Fa = np.empty(CONFIG_COUNT * per)
    Kc = np.empty(CONFIG_COUNT * per)
    Da = np.empty(CONFIG_COUNT * per)
    tc = np.empty(CONFIG_COUNT * per)
    i = 0
    for _curve in CURVES:
        for _k in KS:
            for _rho in RHOS:
                for tau in TAUS:
                    aF, aK, aD, at = option_arrays(tau, np.array(KFR))
                    Fa[i : i + per] = aF
                    Kc[i : i + per] = aK
                    Da[i : i + per] = aD
                    tc[i : i + per] = at
                    i += per
    return Fa, Kc, Da, tc


def print_main() -> None:
    print(
        "CRL pricing accuracy and speed (msv.crl_call, forward form F=D=1). "
        "Grid: k in "
        + ", ".join(str(x) for x in KS)
        + ", rho in "
        + ", ".join(str(x) for x in RHOS)
        + ", tau in [1/52, 1/12, 1/4, 1, 3], K/F in "
        + ", ".join(str(x) for x in KFR)
        + ", model curves: flat Q=0.04 and fitted SPX "
        "(a0,a1,a2,lam)=(0, 0.0921, 0.0196, 0.5167); "
        f"{CONFIG_COUNT} (k,rho,tau,curve) configs x {len(KFR)} strikes."
    )

    ref = reference_prices()

    print("\n## 1. Convergence to the 256-node reference (max over grid)")
    print("| n_nodes | max |price err| | max |IV err| (vol pts) |")
    print("|---|---|---|")
    for n, (pe, ive) in node_errors(ref).items():
        print(f"| {n} | {pe:.3e} | {ive:.3e} |")

    mc = mc_vs_quad(ref)
    mr = _loc(mc, "max_ratio_loc")
    mrm = _loc(mc, "max_ratio_mat_loc")
    name, k, rho, tau, kf = mr
    name_m, k_m, rho_m, tau_m, kf_m = mrm
    ill_ref = _arr(mc, "ill_ref")
    ill_mc = _arr(mc, "ill_mc")
    ill_se = _arr(mc, "ill_se")
    ill_ratio = _arr(mc, "ill_ratio")

    print(
        "\n## 2. 256-node reference vs Monte Carlo of the terminal law (4e6 draws, Z, W iid N(0,1))"
    )
    print(f"max |Q256 - MC| over grid (absolute, price units) = {_f(mc, 'max_abs'):.3e}")
    print(f"max MC standard error over grid = {_f(mc, 'max_se'):.3e}")
    print(
        f"max |Q256 - MC| / SE over grid = {_f(mc, 'max_ratio'):.2f} "
        f"at (curve={name}, k={k}, rho={rho}, tau={tau}, K/F={kf})"
    )
    print(
        f"max |Q256 - MC| / SE over cells with MC price >= {MC_PRICE_FLOOR} = "
        f"{_f(mc, 'max_ratio_mat'):.2f} "
        f"at (curve={name_m}, k={k_m}, rho={rho_m}, tau={tau_m}, K/F={kf_m})"
    )
    print("\nWorst-ratio configuration (reference, MC, SE, |Q256-MC|/SE per K/F):")
    print("| K/F | Q256 (256 nodes) | MC | SE | |Q256-MC|/SE |")
    print("|---|---|---|---|---|")
    for j, kk in enumerate(KFR):
        print(
            f"| {kk} | {ill_ref[j]:.6f} | {ill_mc[j]:.6f} | {ill_se[j]:.2e} | {ill_ratio[j]:.2f} |"
        )

    # speed (64 nodes per the spec)
    flat = (0.0, 0.0, 0.04, 1.0)
    t1000_64 = time_call(64, 1000, flat, 1.0, -0.5) * 1000.0  # ms
    t1000s = {n: time_call(n, 1000, flat, 1.0, -0.5) * 1000.0 for n in NODES}

    # one objective evaluation as used in calibration: one crl_call (64 nodes) over
    # every grid option in a single elementwise call (cf. crl_spx.fit_crl.resid).
    Fa, Kc, Da, tc = _all_grid_options()
    obj_times = np.empty(20)
    for i in range(20):
        t0 = time.perf_counter()
        msv.crl_call(Fa, Kc, Da, tc, *flat, 1.0, -0.5, n_nodes=64)
        obj_times[i] = time.perf_counter() - t0
    t_obj = float(np.median(obj_times)) * 1000.0  # ms

    print("\n## 3. Speed (flat Q=0.04, k=1.0, rho=-0.5; median of 20 runs)")
    print("| n_nodes | ms per 1,000 options (one call) |")
    print("|---|---|")
    for n in NODES:
        print(f"| {n} | {t1000s[n]:.3f} |")
    print(f"| 64 (per option) | {t1000_64:.3f} ms / 1000 = {t1000_64:.3f} us/option |")
    print(f"| one objective evaluation @ 64 nodes over {Fa.size} grid options | {t_obj:.3f} ms |")


def main() -> None:
    print_main()


if __name__ == "__main__":
    main()
