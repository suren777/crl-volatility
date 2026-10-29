"""Checks for calendar_remainder.md: the rho = 0 Brascamp-Lieb bound on R'(L).

Symbolic: the algebra of 1/P = sigma a/(sigma + a), the sigma <= 2 comparison, the derivative
of log(1+a) - 2a/(2+a), the large-lambda expansion of the bound, and the even-part identity
K(1) + K(-1) = int_0^1 (1-s)(K''(s) + K''(-s)) ds for a polynomial cumulant.
Numeric: R' from Model.r_slope on a widened grid against quadrature in v, the tilted
Brascamp-Lieb inequality, and bound >= R' for every k.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import sympy as sp  # type: ignore[import-untyped]  # noqa: E402
from calendar_remainder import (  # noqa: E402
    bound_rho0,
    l0_rho0,
    precision,
    quad_rprime,
    tilted_law,
    wide_model,
)
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


def test_inverse_precision_symbolic() -> None:
    sigma, lam = sp.symbols("sigma lambda", positive=True)
    a = 2 / lam
    inv_p = 1 / (1 / sigma + lam / 2)
    assert sp.simplify(inv_p - sigma * a / (sigma + a)) == 0


def test_comparison_with_two_over_two_plus_a_symbolic() -> None:
    sigma, a = sp.symbols("sigma a", positive=True)
    lhs = 2 * a / (2 + a) - sigma * a / (sigma + a)
    assert sp.simplify(lhs - a**2 * (2 - sigma) / ((2 + a) * (sigma + a))) == 0


def test_log_lower_bound_derivative_symbolic() -> None:
    a = sp.symbols("a", positive=True)
    diff = sp.log(1 + a) - 2 * a / (2 + a)
    assert sp.simplify(sp.diff(diff, a) - a**2 / ((1 + a) * (2 + a) ** 2)) == 0
    assert sp.simplify(diff.subs(a, 0)) == 0


def test_bound_large_lambda_expansion_symbolic() -> None:
    sigma, x = sp.symbols("sigma x", positive=True)
    lam = 1 / x
    inv_p = 1 / (1 / sigma + lam / 2)
    bound = lam / 2 * (sp.exp(inv_p) - 1)
    ser = sp.series(bound, x, 0, 2).removeO()
    assert sp.simplify(ser - (1 + (1 - 2 / sigma) * x)) == 0


def test_even_part_identity_symbolic() -> None:
    s = sp.symbols("s", real=True)
    c1, c2, c3, c4 = sp.symbols("c1 c2 c3 c4", real=True)
    kap = c1 * s + c2 * s**2 + c3 * s**3 + c4 * s**4
    lhs = kap.subs(s, 1) + kap.subs(s, -1)
    integrand = (1 - s) * (sp.diff(kap, s, 2) + sp.diff(kap, s, 2).subs(s, -s))
    rhs = sp.integrate(integrand, (s, 0, 1))
    assert sp.simplify(lhs - rhs) == 0


@settings(max_examples=200, deadline=None)
@given(
    sigma=st.floats(min_value=1e-3, max_value=2.0),
    lam=st.floats(min_value=1e-3, max_value=1e4),
)
def test_bound_below_one_for_sigma_le_two(sigma: float, lam: float) -> None:
    assert bound_rho0(lam, sigma) < 1.0
    assert bound_rho0(lam, sigma) >= 0.0


def test_threshold_only_for_sigma_le_two() -> None:
    assert l0_rho0(np.log1p(2.5**2)) == 0.0
    assert l0_rho0(np.log1p(3.0**2)) is None


@settings(max_examples=12, deadline=None)
@given(
    u=st.floats(min_value=0.2, max_value=5.0),
    k=st.floats(min_value=0.05, max_value=3.0),
    lam=st.floats(min_value=0.5, max_value=100.0),
)
def test_quadrature_matches_model_on_widened_grid(u: float, k: float, lam: float) -> None:
    mod = wide_model(u, k, lam)
    truth = mod.r_slope(lam - mod.lm)
    quad, _ = quad_rprime(lam, u, k)
    assert abs(truth - quad) < 1e-6


@settings(max_examples=12, deadline=None)
@given(
    u=st.floats(min_value=0.2, max_value=5.0),
    k=st.floats(min_value=0.05, max_value=10.0),
    lam=st.floats(min_value=0.5, max_value=300.0),
)
def test_bound_dominates_truth_for_all_k(u: float, k: float, lam: float) -> None:
    quad, _ = quad_rprime(lam, u, k)
    sigma = float(np.log1p(k * k))
    assert quad <= bound_rho0(lam, sigma) + 1e-9


@settings(max_examples=12, deadline=None)
@given(
    u=st.floats(min_value=0.2, max_value=5.0),
    k=st.floats(min_value=0.05, max_value=10.0),
    lam=st.floats(min_value=0.5, max_value=300.0),
    theta=st.sampled_from([-1.0, -0.5, 0.5, 1.0]),
)
def test_brascamp_lieb_on_tilts(u: float, k: float, lam: float, theta: float) -> None:
    sigma = float(np.log1p(k * k))
    mean, var, ev_inv = tilted_law(lam, u, k, theta)
    assert var <= ev_inv + 1e-9
    assert ev_inv <= 1.0 / precision(lam, sigma) + 1e-9
