"""Checks for calendar_laplace.md: sympy expansion coefficients and large-L' rates.

Symbolic: the v-form of the potential (ln y = v + ln 2L'), its quadratic coefficient, the
Gaussian perturbation coefficients (-12 g sigma^6 for the variance, -96 g sigma^8 for E v^4),
and the even-cumulant identity E e^v E e^{-v} = exp(kappa2 + kappa4/12 + ...) for a Gaussian.
Numeric: the v-form of R' against Model.r_slope, and hypothesis rate checks at large L'.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import sympy as sp  # type: ignore[import-untyped]  # noqa: E402
from calendar_band import Model  # noqa: E402
from calendar_laplace import point  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


def test_v_form_potential_numeric() -> None:
    u, s, lam, r = sp.Rational(23, 10), sp.Rational(7, 10), sp.Rational(31, 10), sp.Rational(2, 5)
    v_val = sp.Rational(37, 100)
    t = v_val + sp.log(2 * lam)
    t0 = sp.log(u**2) - s**2 / 2
    z = (t - t0) / s
    a = lam * sp.exp(-t / 2)
    b = sp.exp(t / 2) / 2
    q = a + b + r * z
    v_t = z**2 / 2 + t / 2 + q**2 / (2 * (1 - r**2))
    mu = sp.sqrt(2 * lam)
    c0 = sp.log(2 * lam / u**2) + s**2 / 2
    v_v = (
        (v_val + c0) ** 2 / (2 * s**2)
        + v_val / 2
        + (mu * sp.cosh(v_val / 2) + r * (v_val + c0) / s) ** 2 / (2 * (1 - r**2))
    )
    assert abs(float(sp.N(v_t - v_v - sp.log(2 * lam) / 2, 30))) < 1e-20


def test_quadratic_coefficient_symbolic() -> None:
    v, s, lam, r, c0 = sp.symbols("v s lambda r c0", positive=True)
    mu = sp.sqrt(2 * lam)
    v_v = (
        (v + c0) ** 2 / (2 * s**2)
        + v / 2
        + (mu * sp.cosh(v / 2) + r * (v + c0) / s) ** 2 / (2 * (1 - r**2))
    )
    ser = sp.series(v_v, v, 0, 3).removeO()
    quad = sp.simplify(ser.coeff(v, 2))
    expected = (
        1 / (2 * s**2)
        + lam / (4 * (1 - r**2))
        + r**2 / (2 * s**2 * (1 - r**2))
        + mu * r * c0 / (8 * s * (1 - r**2))
    )
    assert sp.simplify(quad - expected) == 0
    linear = sp.simplify(ser.coeff(v, 1))
    assert (
        sp.simplify(
            linear
            - (
                c0 / s**2
                + sp.Rational(1, 2)
                + mu * r / (s * (1 - r**2))
                + r**2 * c0 / (s**2 * (1 - r**2))
            )
        )
        == 0
    )


def test_gaussian_perturbation_symbolic() -> None:
    p, g = sp.symbols("P g", positive=True)

    def gauss_moment(k: int) -> sp.Expr:
        return sp.factorial2(k - 1) / p ** (k // 2) if k % 2 == 0 else sp.Integer(0)

    z0 = gauss_moment(0)
    e2 = (gauss_moment(2) - g * gauss_moment(6)) / (z0 - g * gauss_moment(4))
    e4 = (gauss_moment(4) - g * gauss_moment(8)) / (z0 - g * gauss_moment(4))
    var_first = sp.series(e2, g, 0, 2).removeO()
    assert sp.simplify(var_first.coeff(g, 1) + 12 / p**3) == 0
    kappa4 = sp.series(e4 - 3 * e2**2, g, 0, 2).removeO()
    assert sp.simplify(kappa4.coeff(g, 1) + 24 / p**4) == 0


def test_even_cumulant_identity_gaussian() -> None:
    m, sig2 = sp.symbols("m sigma2", positive=True)
    lhs = sp.exp(m + sig2 / 2) * sp.exp(-m + sig2 / 2)
    assert sp.simplify(lhs - sp.exp(sig2)) == 0


def v_form_density(u: float, k: float, rho: float, lam: float) -> tuple[np.ndarray, float, float]:
    s = np.sqrt(np.log1p(k * k))
    r = -rho
    c0 = np.log(2 * lam / u**2) + s**2 / 2
    mu = np.sqrt(2 * lam)
    v = np.linspace(-12, 12, 400001)
    vv = (
        (v + c0) ** 2 / (2 * s**2)
        + v / 2
        + (mu * np.cosh(v / 2) + r * (v + c0) / s) ** 2 / (2 * (1 - r**2))
    )
    w = np.exp(-(vv - vv.min()))
    w /= w.sum()
    return v, w, s


def test_v_form_r_slope_matches_model() -> None:
    u, k, rho, frac = 20.0, 1.0, -0.4, 0.5
    s = np.sqrt(np.log1p(k * k))
    mod = Model(u, k, rho)
    lam = frac * np.exp(-s * s / 2) / 2 * u * u
    v, w, s = v_form_density(u, k, rho, lam)
    r = -rho
    c0 = np.log(2 * lam / u**2) + s**2 / 2
    mu = np.sqrt(2 * lam)
    ey = np.sum(w * np.exp(v))
    ey_rev = np.sum(w * np.exp(-v))
    ez = np.sum(w * (v + c0) * np.exp(-v / 2))
    cov_ey_eym = np.sum(w * np.exp(v) * np.exp(-v)) - ey * ey_rev
    cov_ey_z = np.sum(w * np.exp(v) * (v + c0) * np.exp(-v / 2)) - ey * ez
    r_v = -(lam / (1 - r**2)) * (0.5 * cov_ey_eym + r / (s * mu) * cov_ey_z)
    assert abs(r_v - mod.r_slope(lam - mod.lm)) < 1e-6 * max(1.0, abs(r_v))


@settings(max_examples=12, deadline=None)
@given(
    k=st.floats(0.5, 8.0),
    frac=st.floats(0.2, 1.0),
    u=st.floats(20.0, 100.0),
)
def test_rho0_error_is_order_lambda_minus_two(k: float, frac: float, u: float) -> None:
    lp, gap, pred, err = point(u, k, 0.0, frac)
    s2 = np.log1p(k * k)
    if s2 * lp < 20:
        return
    assert abs(err) * lp**2 * s2**2 <= 40.0


@settings(max_examples=12, deadline=None)
@given(
    k=st.floats(0.5, 8.0),
    frac=st.floats(0.2, 1.0),
    u=st.floats(20.0, 100.0),
    rho=st.floats(-0.7, -0.05),
)
def test_rho_negative_error_is_order_lambda_minus_one(
    k: float, frac: float, u: float, rho: float
) -> None:
    lp, gap, pred, err = point(u, k, rho, frac)
    s = np.sqrt(np.log1p(k * k))
    # the rho < 0 expansion is not uniform as rho -> 0; stay clear of r ~ 1/(s sqrt(L'))
    if s * s * lp < 20 or abs(rho) * s * np.sqrt(lp) < 6:
        return
    assert abs(err) * lp * s <= 6.0
    assert abs(err) < abs(pred)


@settings(max_examples=20, deadline=None)
@given(
    k=st.floats(0.5, 8.0),
    frac=st.floats(0.2, 1.0),
    lp=st.floats(50.0, 1e5),
)
def test_second_order_agrees_with_leading_laplace(k: float, frac: float, lp: float) -> None:
    s2 = np.log1p(k * k)
    expansion = 1.0 - 2.0 / (s2 * lp)
    leading_laplace = lp / (lp + 2.0 / s2)
    assert abs(expansion - leading_laplace) * lp**2 <= 8.0 * (1 + 1 / s2) ** 2
