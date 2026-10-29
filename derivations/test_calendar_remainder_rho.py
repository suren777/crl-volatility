"""Checks for calendar_remainder_rho.md: the rho < 0 potential, its curvature, and R' identities.

Symbolic: the exponent of nu in (v, r) equals -V(v) - ln(mu) exactly, and the second derivative of
V. Numeric: the covariance form of R' against Model.r_slope on a widened grid, and the sign
structure used in the scan (R' < 1 at the sample points; no bound is claimed for rho < 0).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sympy as sp  # type: ignore[import-untyped]  # noqa: E402
from calendar_remainder_rho import quad_rprime, wide_model  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

R = sp.Rational


def _exponent_and_potential() -> tuple[
    sp.Expr, sp.Expr, sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol
]:
    v, c0, s, mu = sp.symbols("v c0 s mu", real=True)
    r = sp.symbols("r", positive=True)
    sigma = s**2
    lam = mu**2 / 2
    z = (v + c0) / s
    y = mu**2 * sp.exp(v)
    rt = sp.sqrt(y)
    expo = (
        -(z**2) / (2 * (1 - r**2))
        - r * z * (y / 2 + lam) / ((1 - r**2) * rt)
        - (y / 2 + lam) ** 2 / (2 * (1 - r**2) * y)
        - sp.log(y) / 2
    )
    cosh_half = (sp.exp(v / 2) + sp.exp(-v / 2)) / 2
    q = mu * cosh_half + r * (v + c0) / s
    big_v = (v + c0) ** 2 / (2 * sigma) + v / 2 + q**2 / (2 * (1 - r**2))
    return expo, big_v, v, c0, s, mu, r


def test_exponent_equals_minus_potential_symbolic() -> None:
    expo, big_v, v, c0, s, mu, r = _exponent_and_potential()
    diff = sp.expand(expo + big_v + sp.log(mu))
    for vals in (
        {v: R(3, 10), c0: R(-6, 5), s: R(7, 10), mu: R(5, 2), r: R(2, 5)},
        {v: R(-17, 10), c0: R(2), s: R(13, 10), mu: R(4, 5), r: R(9, 10)},
        {v: R(11, 5), c0: R(1, 10), s: R(1, 5), mu: R(11), r: R(1, 20)},
    ):
        assert abs(float(diff.subs(vals).evalf(40))) < 1e-30


def test_curvature_formula_symbolic() -> None:
    _, big_v, v, c0, s, mu, r = _exponent_and_potential()
    q = mu * (sp.exp(v / 2) + sp.exp(-v / 2)) / 2 + r * (v + c0) / s
    q1 = mu * (sp.exp(v / 2) - sp.exp(-v / 2)) / 4 + r / s
    q2 = mu * (sp.exp(v / 2) + sp.exp(-v / 2)) / 8
    claimed = 1 / s**2 + (q1**2 + q * q2) / (1 - r**2)
    lhs = sp.diff(big_v, v, 2)
    for vals in (
        {v: R(3, 10), c0: R(-6, 5), s: R(7, 10), mu: R(5, 2), r: R(2, 5)},
        {v: R(-17, 10), c0: R(2), s: R(13, 10), mu: R(4, 5), r: R(9, 10)},
    ):
        assert abs(float((lhs - claimed).subs(vals).evalf(40))) < 1e-30


@settings(max_examples=12, deadline=None)
@given(
    u=st.sampled_from([0.1, 1.0, 5.0]),
    k=st.sampled_from([0.3, 1.0, 2.5]),
    r=st.sampled_from([0.2, 0.5, 0.9]),
    lam=st.sampled_from([2.0, 10.0, 50.0]),
)
def test_covariance_form_matches_model(u: float, k: float, r: float, lam: float) -> None:
    mod = wide_model(u, k, lam, -r)
    truth = mod.r_slope(lam - mod.lm)
    assert abs(truth - quad_rprime(lam, u, k, r)["rprime"]) < 1e-4


def test_scan_points_below_one_for_sigma_at_most_two() -> None:
    for u, k, r, lam in ((1.0, 1.0, 0.5, 10.0), (1.0, 2.5, 0.99, 300.0), (5.0, 0.05, 0.5, 3000.0)):
        assert quad_rprime(lam, u, k, r)["rprime"] < 1.0
