"""Ratio form dC/du = T1 (1 + Q) of calendar_ratio.py (epic 0016, task 08-ratio).

Symbolic: the pointwise u-derivative of the Black call conditional on Z (sympy), and the
identity d ln M / du = h. Numeric: the grid evaluation against a central difference of the
price, against a 60-digit mpmath evaluation at resolved band points, and the rho = 0 limit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mpmath as mp  # type: ignore[import-untyped]  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sp  # type: ignore[import-untyped]  # noqa: E402
from calendar_ratio import call_price, evaluate, z_grid  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


def test_dln_m_du_is_h_symbolic() -> None:
    u = sp.Symbol("u", positive=True)
    s, z, rho = sp.symbols("s Z rho", real=True)
    zeta = sp.exp(s * z - s**2 / 2)
    y = u**2 * zeta
    log_m = rho * sp.sqrt(y) * z - rho**2 * y / 2
    h = rho * sp.sqrt(zeta) * z - rho**2 * u * zeta
    assert sp.simplify(sp.diff(log_m, u) - h) == 0


def test_conditional_black_derivative_symbolic() -> None:
    """d/du [F N(d1) - K N(d2)] = F N(d1)(h - m'/m) + F phi(d1) sigma, with F = M/m."""
    u, s, z, rho, big_k = sp.symbols("u s Z rho K", positive=True)
    zeta = sp.exp(s * z - s**2 / 2)
    y = u**2 * zeta
    w = (1 - rho**2) * y
    sw = sp.sqrt(w)
    m = sp.Function("m")(u)
    log_m = rho * sp.sqrt(y) * z - rho**2 * y / 2
    fwd = sp.exp(log_m) / m
    d1 = (sp.log(fwd) - sp.log(big_k) + w / 2) / sw
    d2 = d1 - sw

    def big_n(t: sp.Expr) -> sp.Expr:
        return (1 + sp.erf(t / sp.sqrt(2))) / 2

    def phi(t: sp.Expr) -> sp.Expr:
        return sp.exp(-(t**2) / 2) / sp.sqrt(2 * sp.pi)

    price = fwd * big_n(d1) - big_k * big_n(d2)
    h = rho * sp.sqrt(zeta) * z - rho**2 * u * zeta
    mbar = sp.Symbol("mbar")
    sigma = sp.sqrt((1 - rho**2) * zeta)
    target = fwd * big_n(d1) * (h - mbar) + fwd * phi(d1) * sigma
    mm = sp.Symbol("mm", positive=True)
    dprice = sp.diff(price, u).subs(sp.Derivative(m, u), mbar * m).subs(m, mm)
    expr = dprice - target.subs(m, mm)
    r = sp.Rational
    values = {u: r(7, 10), s: r(9, 10), z: r(-2, 5), rho: r(-3, 10), big_k: r(8, 5)}
    values |= {mm: r(113, 100), mbar: r(-2, 25)}
    assert abs(sp.N(expr.subs(values), 40)) < 1e-30


@given(
    u=st.floats(0.2, 3.0),
    k=st.floats(0.2, 2.0),
    rho=st.floats(-0.9, -0.05),
    x=st.floats(-0.5, 1.5),
)
@settings(max_examples=20, deadline=None)
def test_ratio_matches_central_difference(u: float, k: float, rho: float, x: float) -> None:
    z = z_grid(u, k, rho)
    analytic = evaluate(u, k, rho, x, z)[0]
    h = 1e-5 * u
    fd = (call_price(u + h, k, rho, x, z) - call_price(u - h, k, rho, x, z)) / (2 * h)
    assert abs(analytic - fd) <= 1e-6 * max(abs(fd), 1e-4)


def test_q_vanishes_at_rho_zero() -> None:
    z = z_grid(5.0, 1.0, 0.0)
    for x in np.linspace(0.0, 3.0, 4):
        _, t1, q, _, _ = evaluate(5.0, 1.0, 0.0, float(x), z)
        assert q == 0.0
        assert t1 > 0


def mp_ratio(u: float, k: float, rho: float, x: float, step: float = 0.01) -> float:
    """Q = Cov_A(N(d1), h) / T1 in 60-digit arithmetic on the same Z range as the float code."""
    mp.mp.dps = 60
    uu, kk, rr, xx = mp.mpf(u), mp.mpf(k), mp.mpf(rho), mp.mpf(x)
    s = mp.sqrt(mp.log(1 + kk * kk))
    stp = mp.mpf(step)
    zs = [mp.mpf(-40) + i * stp for i in range(int(round(54 / step)) + 1)]
    lw0, lM, zeta, hh = [], [], [], []
    for zz in zs:
        ze = mp.exp(s * zz - s * s / 2)
        y = uu * uu * ze
        lw0.append(-zz * zz / 2 - mp.log(2 * mp.pi) / 2 + mp.log(stp))
        lM.append(rr * mp.sqrt(y) * zz - rr * rr * y / 2)
        zeta.append(ze)
        hh.append(rr * mp.sqrt(ze) * zz - rr * rr * uu * ze)
    g = [lw0[i] + lM[i] for i in range(len(zs))]
    gm = max(g)
    lmz = gm + mp.log(mp.fsum(mp.exp(gi - gm) for gi in g))
    a = [mp.exp(gi - lmz) for gi in g]
    hbar = mp.fsum(a[i] * hh[i] for i in range(len(zs)))
    t1 = mp.mpf(0)
    cov = mp.mpf(0)
    for i in range(len(zs)):
        logf = lM[i] - lmz
        w = (1 - rr * rr) * uu * uu * zeta[i]
        sw = mp.sqrt(w)
        d1 = (logf - xx + w / 2) / sw
        t1 += a[i] * mp.npdf(d1) * mp.sqrt((1 - rr * rr) * zeta[i])
        cov += a[i] * mp.ncdf(d1) * (hh[i] - hbar)
    return float(cov / t1)


def test_float_ratio_matches_high_precision() -> None:
    u, k, rho, x = 50.0, 3.0, -0.2, 34.05
    z = z_grid(u, k, rho)
    _, t1, q, _, _ = evaluate(u, k, rho, x, z)
    assert abs(q - mp_ratio(u, k, rho, x)) < 1e-6
    assert abs(t1 * (1 + q) - evaluate(u, k, rho, x, z)[0]) < 1e-12 * t1
