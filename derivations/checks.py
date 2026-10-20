"""Shared helpers for derivation checks.

Independence rule for cross-checks: do not reuse the code under test. The Black
formula here is plain numpy written from the textbook expression — `msv.bs_call`,
`msv.gh_call_fwd` and the `black_fwd` copies in test_msv.py / experiments/ are
off limits for cross-checks.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

FloatArray = npt.NDArray[np.float64]


def relerr(a: float | npt.NDArray[np.float64], b: float | npt.NDArray[np.float64]) -> float:
    """Relative error |a - b| / |b|."""
    return float(np.abs(a - b) / max(abs(float(np.asarray(b))), 1e-300))


def lognormal_moment(k: float, n: int) -> float:
    """E[zeta^n] for zeta = exp(s Z - s^2/2), s^2 = ln(1+k^2): (1+k^2)^(n(n-1)/2)."""
    return float(np.exp(n * (n - 1) * np.log1p(k * k) / 2.0))


def loglog_slope(x: npt.ArrayLike, err: npt.ArrayLike) -> float:
    """Slope of log(err) vs log(x) by least squares: the convergence order.

    Only finite, strictly positive entries are used.
    """
    xs = np.asarray(x, dtype=float).ravel()
    es = np.asarray(err, dtype=float).ravel()
    mask = np.isfinite(xs) & np.isfinite(es) & (xs > 0.0) & (es > 0.0)
    return float(np.polyfit(np.log(xs[mask]), np.log(es[mask]), 1)[0])


def black_fwd(F: float, K: npt.ArrayLike, tau: float, sigma: float) -> FloatArray:
    """Undiscounted Black call price, forward form (F, K, tau in years, sigma vol)."""
    v = sigma * sigma * tau
    sd = np.sqrt(v)
    k_a = np.asarray(K, dtype=float)
    with np.errstate(divide="ignore"):
        d1 = (np.log(F / k_a) + v / 2.0) / sd
    return np.asarray(F * ndtr(d1) - k_a * ndtr(d1 - sd), dtype=float)
