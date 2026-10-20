import legacy_pricer as lp
import numpy as np
from scipy.stats import norm

S = 6600.0
theta = (0.385043, 0.161195, -0.403900, 0.247768, 2.945836)  # FTSE, 2014 hedging script
K = np.array([5940.0, 6600.0, 7260.0])
T = np.array([0.1, 0.5, 1.5])
r = np.full(3, 0.005)


def bs(sig: lp.FloatArray) -> lp.FloatArray:
    d1 = (np.log(S / K) + (r + sig**2 / 2) * T) / (sig * np.sqrt(T))
    return np.asarray(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sig * np.sqrt(T)))


def test_small_k_collapses_to_black_scholes() -> None:
    th = (1e-9, *theta[1:])
    q = lp.q_code(th, T)
    assert np.allclose(lp.mmpricer(th, K, T, S, r), bs(q), rtol=1e-8)
    assert np.allclose(lp.vol_mixture_call(th, K, T, S, r), bs(q), rtol=1e-8)
    assert np.allclose(lp.var_mixture_call(th, K, T, S, r), bs(np.sqrt(q)), rtol=1e-8)


def test_pdf_fix_moves_taylor_towards_exact() -> None:
    bug = lp.mmpricer(theta, K, T, S, r)
    fixed = lp.mmpricer(theta, K, T, S, r, pdf_fix=True)
    exact = lp.vol_mixture_call(theta, K, T, S, r)
    assert abs(exact[0] - 664.676) < 1e-2
    assert np.all(np.abs(fixed - exact) <= np.abs(bug - exact) + 1e-6)


def test_seven_param_with_unit_scale_matches_five() -> None:
    th7 = (*theta, 0.0, 1.0)
    assert np.allclose(lp.mmpricer1(th7, K, T, S, r), lp.mmpricer(theta, K, T, S, r))
