"""Python port of the 2014 MSV MATLAB pricer and calibration loop, plus exact references.

Ported from legacy/MATLAB/Options/logaddedparam/mmpricer.m, mmpricer1.m, minfunc*.m and
NEW_DATA/MSV/minfunc.m. Parameter vector order follows the MATLAB code:
theta = (k, s0, s1, s2, c5) with lambda = c5**2; the 7-parameter variant appends (ca, cc)
with a = ca**2, c = cc**2. The 2014 code uses Q(tau) as the Black-Scholes *volatility*.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize
from scipy.stats import norm

FloatArray = npt.NDArray[np.float64]
Theta = Sequence[float] | FloatArray
Pricer = Callable[[Theta, FloatArray, FloatArray, float, FloatArray], FloatArray]


@dataclass(frozen=True)
class Dataset:
    """One 2014 calibration set: in-sample/out-of-sample split exactly as the script had it."""

    name: str
    S: float
    K_in: FloatArray
    T_in: FloatArray
    C_in: FloatArray
    r_in: FloatArray
    w_in: FloatArray
    K_out: FloatArray
    T_out: FloatArray
    C_out: FloatArray
    r_out: FloatArray


def q_code(theta: Theta, T: FloatArray) -> FloatArray:
    """Nelson-Siegel level Q(T) exactly as mmpricer.m computes it."""
    s0, s1, s2, lam = theta[1], theta[2], theta[3], theta[4] ** 2
    lT = lam * T
    e = np.exp(-lT)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(((s0**2 + s1**2) / lT + s1**2 + (s2**2 - s1**2) / (1 - e)) * (1 - e))


def _taylor(
    k: float, q: FloatArray, K: FloatArray, T: FloatArray, S: float, r: FloatArray, pdf_fix: bool
) -> FloatArray:
    v4 = k**2
    s6 = k**4 * (3 + k**2)
    p8 = k**4 * ((1 + k**2) ** 4 + 2 * (1 + k**2) ** 3 + 3 * (1 + k**2) ** 2 - 3)
    d1 = (np.log(S / K) + (r + q**2 / 2) * T) / (q * np.sqrt(T))
    d2 = d1 - q * np.sqrt(T)
    f = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    # mmpricer.m uses exp(-d2^2)/sqrt(2 pi): the 2014 pdf bug (0007 item 11)
    pdf = norm.pdf(d2) if pdf_fix else np.exp(-(d2**2)) / np.sqrt(2 * np.pi)
    det = K * np.exp(-r * T) * pdf * np.sqrt(T)
    vo = det * d1 * d2 / q
    u = -det / q**2 * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2)
    fo = (
        det
        * (12 * d1 * d2 + 3 * T * q**2 - 3 * q**2 * T * d1 * d2 - 9 * d1**2 * d2**2 + d1**3 * d2**3)
        / q**3
    )
    return np.asarray(f + vo * v4 / 2 * q**2 + u * s6 / 6 * q**3 + fo * p8 / 24 * q**4)


def mmpricer(
    theta: Theta,
    K: FloatArray,
    T: FloatArray,
    S: float,
    r: FloatArray,
    pdf_fix: bool = False,
) -> FloatArray:
    """5-parameter 2014 pricer (Taylor-4 in volatility). pdf_fix=False reproduces the bug."""
    return _taylor(theta[0], q_code(theta, T), K, T, S, r, pdf_fix)


def mmpricer1(
    theta: Theta,
    K: FloatArray,
    T: FloatArray,
    S: float,
    r: FloatArray,
    pdf_fix: bool = False,
) -> FloatArray:
    """7-parameter variant: Q scaled by a*ln(S/K) + c (not in the paper)."""
    a, c = theta[5] ** 2, theta[6] ** 2
    q = (a * np.log(S / K) + c) * q_code(theta, T)
    return _taylor(theta[0], q, K, T, S, r, pdf_fix)


def _mixture(
    theta: Theta,
    K: FloatArray,
    T: FloatArray,
    S: float,
    r: FloatArray,
    vol_convention: bool,
    n_nodes: int,
) -> FloatArray:
    k = theta[0]
    q = q_code(theta, T)
    s2 = np.log1p(k * k)
    x, w = hermegauss(n_nodes)
    zeta = np.exp(np.sqrt(s2) * np.asarray(x) - s2 / 2)
    v = zeta[:, None] * q[None, :]
    sig = v if vol_convention else np.sqrt(v)
    d1 = (np.log(S / K) + (r + sig**2 / 2) * T) / (sig * np.sqrt(T))
    px = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sig * np.sqrt(T))
    return np.asarray((np.asarray(w) / np.sqrt(2 * np.pi)) @ px)


def vol_mixture_call(
    theta: Theta, K: FloatArray, T: FloatArray, S: float, r: FloatArray, n_nodes: int = 64
) -> FloatArray:
    """Exact E[BS(vol = zeta*Q)]: the model the 2014 code approximates."""
    return _mixture(theta, K, T, S, r, True, n_nodes)


def var_mixture_call(
    theta: Theta, K: FloatArray, T: FloatArray, S: float, r: FloatArray, n_nodes: int = 64
) -> FloatArray:
    """Exact E[BS(variance = zeta*Q)]: the paper text's model, as in msv.gh_call."""
    return _mixture(theta, K, T, S, r, False, n_nodes)


FloorKind = Literal["none", "abs_0.001", "plain_0.01"]


def objective(theta: Theta, ds: Dataset, pricer: Pricer, floor: FloorKind = "none") -> float:
    """minfunc.m: mean(w*(C - model)^2) plus the 2014 penalty on k, if any."""
    model = pricer(theta, ds.K_in, ds.T_in, ds.S, ds.r_in)
    f = float(np.sum((ds.C_in - model) ** 2 * ds.w_in) / ds.C_in.size)
    if floor == "abs_0.001" and abs(theta[0]) < 0.001:
        f += 1e7
    if floor == "plain_0.01" and theta[0] < 0.01:
        f += 1e7
    return f if np.isfinite(f) else 1e30


def fminsearch(
    fun: Callable[[FloatArray], float], x0: Theta, max_evals: int = 20_000
) -> FloatArray:
    """MATLAB fminsearch equivalent (same 5% initial simplex, TolX = TolFun = 1e-4).

    The 2014 scripts loop fminsearch from the same x0 until fval stops changing, which
    terminates after the second identical run; one run is equivalent.
    """
    res = minimize(
        fun,
        np.asarray(x0, dtype=float),
        method="Nelder-Mead",
        options={"maxfev": max_evals, "maxiter": max_evals, "xatol": 1e-4, "fatol": 1e-4},
    )
    return np.asarray(res.x, dtype=float)


def metrics(C: FloatArray, model: FloatArray) -> dict[str, float]:
    """MRAE (%), RMSE and max relative error (%) as the 2014 scripts print them."""
    rel = np.abs(C - model) / C
    return {
        "MRAE": float(np.mean(rel) * 100),
        "RMSE": float(np.sqrt(np.mean((C - model) ** 2))),
        "MaxRE": float(np.max(rel) * 100),
    }


def _fixed_k0(pricer: Pricer) -> Pricer:
    def p(t: Theta, K: FloatArray, T: FloatArray, S: float, r: FloatArray) -> FloatArray:
        return pricer((0.0, *t), K, T, S, r)

    return p


def _taylor_pdf_fixed(
    t: Theta, K: FloatArray, T: FloatArray, S: float, r: FloatArray
) -> FloatArray:
    return mmpricer(t, K, T, S, r, pdf_fix=True)


def replicate(
    ds: Dataset,
    x0: Theta,
    seven_param: bool,
    floor: FloorKind,
    n_starts: int = 20,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Run the six standard variants on one dataset; one result row per variant.

    1. faithful: the script's own pricer, penalty and x0, one fminsearch run;
    2. faithful without the k penalty;
    3. k fixed at 0 (Black-Scholes with the Nelson-Siegel volatility term structure);
    4. Taylor-4 with the pdf fixed; 5. exact volatility mixture; 6. exact variance mixture.
    Variants 3-6 are 5-parameter (4 for variant 3) and multi-start: x0 plus n_starts
    Gaussian perturbations (sd 0.5), keeping the lowest objective.
    """
    rng = np.random.default_rng(seed)
    x5 = np.asarray(x0[:5], dtype=float)
    starts = [x5] + [x5 + rng.normal(0.0, 0.5, 5) for _ in range(n_starts)]
    faithful: Pricer = mmpricer1 if seven_param else mmpricer
    rows: list[dict[str, object]] = []

    def record(variant: str, pricer: Pricer, th: FloatArray, k: float) -> None:
        rows.append(
            {
                "dataset": ds.name,
                "variant": variant,
                "k": k,
                "theta": th.tolist(),
                "in": metrics(ds.C_in, pricer(th, ds.K_in, ds.T_in, ds.S, ds.r_in)),
                "out": metrics(ds.C_out, pricer(th, ds.K_out, ds.T_out, ds.S, ds.r_out)),
            }
        )

    def best(pricer: Pricer, xs: list[FloatArray]) -> FloatArray:
        fits = [fminsearch(lambda t: objective(t, ds, pricer), x) for x in xs]
        return min(fits, key=lambda t: objective(t, ds, pricer))

    th = fminsearch(lambda t: objective(t, ds, faithful, floor), x0)
    record("1 faithful 2014", faithful, th, float(th[0]))
    th = fminsearch(lambda t: objective(t, ds, faithful), x0)
    record("2 faithful, no k penalty", faithful, th, float(th[0]))
    k0 = _fixed_k0(mmpricer)
    th = best(k0, [s[1:] for s in starts])
    record("3 k=0 (BS + NS vol term structure)", k0, th, 0.0)
    variants: list[tuple[str, Pricer]] = [
        ("4 Taylor-4, pdf fixed", _taylor_pdf_fixed),
        ("5 exact vol mixture (GH)", vol_mixture_call),
        ("6 exact variance mixture (GH)", var_mixture_call),
    ]
    for name, pricer in variants:
        th = best(pricer, starts)
        record(name, pricer, th, float(th[0]))
    return rows
