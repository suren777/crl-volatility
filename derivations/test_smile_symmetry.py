"""smile_symmetry: at rho = 0 the implied-vol smile is even in forward log-moneyness.

For rho = 0 the CRL model is the lognormal variance mixture
    price(K) = E_zeta[ Black(F, K, tau, zeta*Q(tau)) ]          (eq_crl_rho0),
so ln S_T = ln F - V tau/2 + sqrt(V tau) eps per component with V = zeta Q(tau).
The precise reason the implied vol is even in x = ln(K/F):

(i) Black put-call symmetry holds per component: for any total variance v,
        C_BS(F, Fe^x, v) = e^x P_BS(F, Fe^{-x}, v).
    Taking the expectation over zeta (E[S_T] = F keeps the model a martingale)
    gives the same relation at the model level,
        C_model(F, Fe^x) = e^x P_model(F, Fe^{-x}).

(ii) Call and put at the same strike share the same Black IV (put-call parity),
    so the IV of the model call at log-strike +x equals the IV of the model put
    at -x, which in turn equals the IV of the model call at -x, i.e.
        sigma_BS(+x) = sigma_BS(-x).

We verify the Black put-call symmetry symbolically (SymPy, the two normal-argument
alignments -d2' = d1 and -d1' = d2 carry the whole identity) and the IV evenness
numerically to 1e-8. IV inversion uses the independent black_fwd from
derivations/checks.py (never the code under test).

Numerical box: a2 >= 0.03 and tau >= 0.5 (so Q*tau isn't tiny). When the total
variance is small the deep-OTM point (x = 0.4: K/F ~ 1.49) prices collapse
toward zero and Black IV inversion from a ~1e-9 price is ill-conditioned, so
an innocuous quadrature/inversion residual would exceed 1e-8 even though the
exact model is still symmetric. The bound is a well-conditioning range, not
the claim failing; within it the two-sided agreement is ~1e-14.
"""

import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import checks  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sp  # type: ignore[import-untyped]  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

import msv  # noqa: E402

MONEYNESS = (0.05, 0.1, 0.2, 0.4)

Pricer = Callable[[float, np.ndarray, float, float, float, float, float, float, float], np.ndarray]


def iv_from_price(F: float, K: float, tau: float, px_undisc: float) -> float:
    """Black implied vol inverting the undiscounted price with checks.black_fwd."""
    return float(
        brentq(lambda s: float(checks.black_fwd(F, K, tau, s)) - px_undisc, 1e-9, 4.0, xtol=1e-12)
    )


def assert_even(
    pricer: Pricer, a0: float, a1: float, a2: float, lam: float, k: float, tau: float
) -> None:
    """sigma_BS(+x) == sigma_BS(-x) to 1e-8 for each x in MONEYNESS."""
    F0, D = 100.0, 1.0
    for x in MONEYNESS:
        px_p = float(pricer(F0, F0 * np.exp(x), D, tau, a0, a1, a2, lam, k)[0]) / D
        px_m = float(pricer(F0, F0 * np.exp(-x), D, tau, a0, a1, a2, lam, k)[0]) / D
        iv_p = iv_from_price(F0, F0 * np.exp(x), tau, px_p)
        iv_m = iv_from_price(F0, F0 * np.exp(-x), tau, px_m)
        assert abs(iv_p - iv_m) <= 1e-8, (x, a0, a1, a2, lam, k, tau, iv_p, iv_m)
    print(f"ok  smile symmetry rho=0  |iv(+x)-iv(-x)| <= 1e-8  (k={k:.2f}, tau={tau:.3f})")


def _crl0(
    F: float,
    K: np.ndarray,
    D: float,
    tau: float,
    a0: float,
    a1: float,
    a2: float,
    lam: float,
    k: float,
) -> np.ndarray:
    return msv.crl_call(F, K, D, tau, a0, a1, a2, lam, k, 0.0, n_nodes=129)


def _gh0(
    F: float,
    K: np.ndarray,
    D: float,
    tau: float,
    a0: float,
    a1: float,
    a2: float,
    lam: float,
    k: float,
) -> np.ndarray:
    return msv.gh_call_fwd(F, K, D, tau, a0, a1, a2, lam, k, beta=0.0, n_nodes=129)


@settings(max_examples=10, deadline=None)
@given(
    a0=st.floats(0.0, 0.1),
    a1=st.floats(0.0, 0.1),
    a2=st.floats(0.03, 0.1),
    lam=st.floats(0.5, 4.0),
    k=st.floats(0.2, 2.0),
    tau=st.floats(0.5, 3.0),
)
def test_eq_smile_symmetry_crl(
    a0: float, a1: float, a2: float, lam: float, k: float, tau: float
) -> None:
    """crl_call(rho = 0): implied vol even in ln(K/F) to 1e-8."""
    assert_even(_crl0, a0, a1, a2, lam, k, tau)


@settings(max_examples=10, deadline=None)
@given(
    a0=st.floats(0.0, 0.1),
    a1=st.floats(0.0, 0.1),
    a2=st.floats(0.03, 0.1),
    lam=st.floats(0.5, 4.0),
    k=st.floats(0.2, 2.0),
    tau=st.floats(0.5, 3.0),
)
def test_eq_smile_symmetry_gh(
    a0: float, a1: float, a2: float, lam: float, k: float, tau: float
) -> None:
    """gh_call_fwd(beta = 0): implied vol even in ln(K/F) to 1e-8."""
    assert_even(_gh0, a0, a1, a2, lam, k, tau)


def test_eq_smile_symmetry_symbolic() -> None:
    """Symbolic proof of Black put-call symmetry  C(F, Fe^x, v) = e^x P(F, Fe^{-x}, v).

    C = F N(d1) - F e^x N(d2),  d1 = (-x+v/2)/sqrt(v),  d2 = (-x-v/2)/sqrt(v).
    P(F, Fe^{-x}, v) = F e^{-x} N(-d2') - F N(-d1'),  d1' = (x+v/2)/sqrt(v),
    d2' = (x-v/2)/sqrt(v). The identity reduces to the two argument alignments
    -d2' = d1 and -d1' = d2 (verified below), so C - e^x P = 0. Averaging over
    zeta keeps E[S_T] = F, so C_model(F, Fe^x) = e^x P_model(F, Fe^{-x}); call
    and put at the same strike share a Black IV, giving sigma_BS(x) = sigma_BS(-x).
    """
    x, v = sp.symbols("x v", positive=True)
    sd = sp.sqrt(v)
    d1 = (-x + v / 2) / sd
    d2 = (-x - v / 2) / sd
    d1p = (x + v / 2) / sd
    d2p = (x - v / 2) / sd
    assert sp.simplify((-d2p) - d1) == 0  # put first-term normal argument = call d1
    assert sp.simplify((-d1p) - d2) == 0  # put second-term normal argument = call d2
    # Cross-check the concrete identity numerically.
    F0, x0, tau0, sig0 = 100.0, 0.3, 0.5, 0.2
    v0 = sig0 * sig0 * tau0
    c = float(msv.bs_call(F0, F0 * np.exp(x0), 0.0, tau0, v0))
    p_c = float(msv.bs_call(F0, F0 * np.exp(-x0), 0.0, tau0, v0))
    p_put = p_c - (F0 - F0 * np.exp(-x0))  # put-call parity
    assert checks.relerr(c, np.exp(x0) * p_put) < 1e-9, (c, p_put)
    print("ok  smile symmetry symbolic: C(F,Fe^x) = e^x P(F,Fe^-x) (put-call symmetry)")


if __name__ == "__main__":
    test_eq_smile_symmetry_symbolic()
    test_eq_smile_symmetry_crl()
    test_eq_smile_symmetry_gh()
