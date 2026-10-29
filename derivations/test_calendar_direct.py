"""dC/du of calendar_direct.py: integral formula, enclosure, tail bounds.

Symbolic: the Black-derivative identities behind G = E[A (phi(d1) sigma + N(d1)(h - mbar))],
and d ln M / du = h. Numeric: G against a central finite difference of C (which uses only
the price), C against msv.crl_call (independent Gauss-Hermite code), the rigorous enclosure
against the float value, and the tail bounds against a wider Z-grid.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import sympy as sp  # type: ignore[import-untyped]  # noqa: E402
from calendar_band import Model  # noqa: E402
from calendar_direct import ZMAX, _enclose_cell, _tail, certify_box, point, x_upper  # noqa: E402
from flint import arb  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from scipy.stats import norm  # noqa: E402

import msv  # noqa: E402


def test_black_derivatives_symbolic() -> None:
    f, big_k, v = sp.symbols("F K v", positive=True)
    d1 = (sp.log(f / big_k) + v / 2) / sp.sqrt(v)
    d2 = d1 - sp.sqrt(v)
    black = f * sp.Rational(1, 2) * (1 + sp.erf(d1 / sp.sqrt(2))) - big_k * sp.Rational(1, 2) * (
        1 + sp.erf(d2 / sp.sqrt(2))
    )
    phi = sp.exp(-(d1**2) / 2) / sp.sqrt(2 * sp.pi)
    assert sp.simplify(sp.diff(black, f) - (1 + sp.erf(d1 / sp.sqrt(2))) / 2) == 0
    assert sp.simplify(sp.diff(black, v) - f * phi / (2 * sp.sqrt(v))) == 0


def test_dcdu_integrand_symbolic() -> None:
    u, rho, s, z, x, mbar, a = sp.symbols("u rho s Z x mbar A", real=True)
    zeta = sp.exp(s * z - s**2 / 2)
    sq = sp.sqrt(zeta)
    h = rho * sq * z - rho**2 * u * zeta
    log_m = rho * u * sq * z - rho**2 * u**2 * zeta / 2
    assert sp.simplify(sp.diff(log_m, u) - h) == 0
    w = (1 - rho**2) * u**2 * zeta
    d1 = (sp.log(a) - x + w / 2) / sp.sqrt(w)
    d2 = d1 - sp.sqrt(w)
    price = a * (1 + sp.erf(d1 / sp.sqrt(2))) / 2 - sp.exp(x) * (1 + sp.erf(d2 / sp.sqrt(2))) / 2
    total = sp.diff(price, u) + sp.diff(price, a) * a * (h - mbar)
    phi_d1 = sp.exp(-(d1**2) / 2) / sp.sqrt(2 * sp.pi)
    claim = a * (
        phi_d1 * sp.sqrt((1 - rho**2) * zeta) + (1 + sp.erf(d1 / sp.sqrt(2))) / 2 * (h - mbar)
    )
    diff = total - claim
    for point_vals in (
        {u: 0.37, rho: -0.4, s: 0.9, z: 0.2, x: 0.05, mbar: 0.3, a: 1.1},
        {u: 1.2, rho: -0.8, s: 0.5, z: -0.6, x: -0.2, mbar: -0.1, a: 0.7},
    ):
        assert abs(float(diff.subs(point_vals).evalf())) < 1e-12


@settings(max_examples=12, deadline=None)
@given(
    u=st.floats(0.05, 3.0),
    k=st.floats(0.1, 3.0),
    rho=st.floats(-0.95, -0.02),
    x=st.floats(-0.5, 0.5),
)
def test_g_matches_finite_difference(u: float, k: float, rho: float, x: float) -> None:
    g, _, _ = point(u, k, rho, x)
    h = 1e-5 * u
    fd = (point(u + h, k, rho, x)[1] - point(u - h, k, rho, x)[1]) / (2 * h)
    assert abs(g - fd) < 1e-6 + 1e-5 * abs(fd)


@settings(max_examples=6, deadline=None)
@given(
    u=st.floats(0.2, 1.5),
    k=st.floats(0.2, 1.5),
    rho=st.floats(-0.6, -0.05),
    x=st.floats(-0.3, 0.3),
)
def test_call_matches_msv(u: float, k: float, rho: float, x: float) -> None:
    lam = 1.0
    a0 = u * u / (1 - np.exp(-lam)) * lam
    price = msv.crl_call(1.0, np.exp(x), 1.0, 1.0, a0, 0.0, 0.0, lam, k, rho, n_nodes=256)
    assert abs(point(u, k, rho, x)[1] - float(np.ravel(price)[0])) < 1e-6


@settings(max_examples=6, deadline=None)
@given(
    u=st.floats(0.1, 2.0),
    k=st.floats(0.1, 2.0),
    rho=st.floats(-0.9, -0.05),
    x=st.floats(-0.2, 0.2),
)
def test_enclosure_contains_float(u: float, k: float, rho: float, x: float) -> None:
    g, _, _ = point(u, k, rho, x)
    enc = _enclose_cell([(u, u), (k, k), (rho, rho), (x, x)])
    assert float(enc.lower()) - 1e-9 <= g <= float(enc.upper()) + 1e-9


@pytest.mark.parametrize(("u", "k", "rho"), [(0.7, 1.2, -0.6), (0.5, 0.1, -0.3), (3.0, 0.2, -0.95)])
def test_tail_bounds_dominate_truncation(u: float, k: float, rho: float) -> None:
    s = np.sqrt(np.log1p(k * k))
    z = np.linspace(-60.0, 60.0, 240001)
    wt = norm.pdf(z) * (z[1] - z[0])
    sq = np.exp(s * z / 2 - s * s / 4)
    zeta = sq * sq
    big_m = np.exp(rho * u * sq * z - rho * rho * u * u * zeta / 2)
    h = rho * sq * z - rho * rho * u * zeta
    outside = np.abs(z) > ZMAX
    for kind, q in (("m", np.ones_like(z)), ("mh", h)):
        exact_tail = abs(float(np.sum((wt * big_m * q)[outside])))
        bound = float(_tail(kind, arb(rho), arb(u), arb(s)).upper())
        assert exact_tail <= bound


def test_x_upper_dominates_l2_sharp() -> None:
    box = ((0.2, 0.5), (0.5, 1.5), (-0.8, -0.2))
    bound = x_upper(box)
    rng = np.random.default_rng(0)
    for _ in range(8):
        u = rng.uniform(*box[0])
        k = rng.uniform(*box[1])
        rho = rng.uniform(*box[2])
        assert Model(u, k, rho).l2_sharp() <= bound + 1e-6


def test_small_region_certifies_quickly() -> None:
    res = certify_box(((0.3, 0.31), (1.0, 1.01), (-0.5, -0.49)))
    assert res["bad"] == 0
    assert res["cells"] <= 10
    assert res["min_lo"] > 0
