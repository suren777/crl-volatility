"""eq_crl_m_lambert: m(u) = E_V[1 / (1 + W(beta e^{s V / 2}))] (calendar_tail.md).

Symbolic: the tilted density phi(Z) M(Z) is phi(Z - c g(Z)), and V = Z - c g(Z) gives
1 + (s/2)(V - Z) = 1 + W(beta e^{sV/2}). Numeric: the Lambert-W form against direct
quadrature of E_Z[M] (independent code path), and m'(u) < 0 for rho < 0 by finite
differences of the direct quadrature. Limit: m = 1 at rho = 0.
"""

import numpy as np
import sympy as sp  # type: ignore[import-untyped]
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import lambertw, logsumexp
from scipy.stats import norm

Z = np.linspace(-40.0, 120.0, 320_001)
LW = norm.logpdf(Z) + np.log(Z[1] - Z[0])
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(200)


def m_direct(u: float, k: float, rho: float) -> float:
    s = np.sqrt(np.log1p(k * k))
    y = u * u * np.exp(s * Z - s * s / 2)
    return float(np.exp(logsumexp(LW + rho * np.sqrt(y) * Z - rho * rho * y / 2)))


def m_lambert(u: float, k: float, rho: float) -> float:
    s = np.sqrt(np.log1p(k * k))
    beta = s / 2 * abs(rho) * u * np.exp(-s * s / 4)
    w = lambertw(beta * np.exp(s * GH_X / 2)).real
    return float(np.sum(GH_W / (1.0 + w)) / np.sqrt(2 * np.pi))


def test_eq_crl_m_lambert_symbolic() -> None:
    z, c, s, t = sp.symbols("z c s t", real=True)
    g = sp.exp(s * z / 2)
    exponent = -(z**2) / 2 + c * z * g - c**2 * g**2 / 2
    assert sp.simplify(exponent + (z - c * g) ** 2 / 2) == 0
    # V = Z + a e^{sZ/2} with a = -c > 0; put V - Z = t: t e^{st/2} = a e^{sV/2}, so
    # (s/2) t = W((s a / 2) e^{sV/2}), and the Jacobian dV/dZ = 1 + (s/2) t.
    a = -c
    jac = sp.diff(z + a * g, z)
    assert sp.simplify(jac - (1 + s / 2 * a * g)) == 0
    vv = z + a * g
    assert sp.simplify((t * sp.exp(s * t / 2)).subs(t, vv - z) - a * sp.exp(s * vv / 2)) == 0


@settings(max_examples=40, deadline=None)
@given(
    u=st.floats(0.01, 3.0),
    k=st.floats(0.05, 3.0),
    rho=st.floats(-0.99, -0.01),
)
def test_eq_crl_m_lambert(u: float, k: float, rho: float) -> None:
    assert abs(m_lambert(u, k, rho) - m_direct(u, k, rho)) < 1e-8
    h = 1e-4 * u
    assert m_direct(u + h, k, rho) < m_direct(u - h, k, rho)


def test_eq_crl_m_lambert_rho_zero() -> None:
    assert abs(m_lambert(1.0, 1.0, 0.0) - 1.0) < 1e-14
    assert abs(m_direct(1.0, 1.0, 0.0) - 1.0) < 1e-10
