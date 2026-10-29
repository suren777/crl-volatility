"""eq_crl_tail_derivative and the wing theorems of calendar_band.md.

Symbolic: u d(d2)/du = (Lt - y/2) / sqrt(w) with Lt = ln x + ln m - u m'/m, and the pairing
identity f(-v) - f(v) = v + 2 Z0 v / s + (|rho| sqrt(2L') cosh(v/2) 2v/s + rho^2 2 Z0 v/s)
/ (1-rho^2). Numeric: u p(x) against a central difference of P(S > x) computed with
Gauss-Hermite in Z (independent of calendar_band.Model), and the signs of p beyond the left
threshold l1 and the right threshold l2 on random parameters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import sympy as sp  # type: ignore[import-untyped]  # noqa: E402
from calendar_band import Model  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from scipy.special import logsumexp, ndtr  # noqa: E402
from scipy.stats import norm  # noqa: E402

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(300)


def test_eq_crl_tail_derivative_symbolic() -> None:
    u = sp.Symbol("u", positive=True)
    s, z, big_l = sp.symbols("s z L", real=True)
    rho = sp.Symbol("rho", real=True)
    m = sp.Function("m")(u)
    zeta = sp.exp(s * z - s**2 / 2)
    y = u**2 * zeta
    w = (1 - rho**2) * y
    d2 = (rho * sp.sqrt(y) * z - y / 2 - sp.log(m) - big_l) / sp.sqrt(w)
    lt = big_l + sp.log(m) - u * sp.diff(m, u) / m
    assert sp.simplify(u * sp.diff(d2, u) - (lt - y / 2) / sp.sqrt(w)) == 0


def test_pairing_identity_symbolic() -> None:
    v, s, lp, r = sp.symbols("v s Lp r", positive=True)
    z0 = sp.Symbol("Z0", real=True)

    def f(t: sp.Expr) -> sp.Expr:
        zz = z0 + t / s
        y = 2 * lp * sp.exp(t)
        b = lp + y / 2 + r * sp.sqrt(y) * zz
        return -(zz**2) / 2 - b**2 / (2 * (1 - r**2) * y) - sp.log(y) / 2

    target = (
        v
        + 2 * z0 * v / s
        + (r * sp.sqrt(2 * lp) * sp.cosh(v / 2) * 2 * v / s + r**2 * 2 * z0 * v / s) / (1 - r**2)
    )
    diff = sp.simplify((f(-v) - f(v) - target).rewrite(sp.exp))
    assert diff == 0


def gh_prob(u: float, k: float, rho: float, big_l: float) -> float:
    s = np.sqrt(np.log1p(k * k))
    y = u * u * np.exp(s * GH_X - s * s / 2)
    m = np.sum(GH_W * np.exp(rho * np.sqrt(y) * GH_X - rho * rho * y / 2)) / np.sqrt(2 * np.pi)
    d2 = (rho * np.sqrt(y) * GH_X - y / 2 - np.log(m) - big_l) / np.sqrt((1 - rho * rho) * y)
    return float(np.sum(GH_W * ndtr(d2)) / np.sqrt(2 * np.pi))


@settings(max_examples=15, deadline=None)
@given(
    u=st.floats(0.05, 1.5),
    k=st.floats(0.1, 2.0),
    rho=st.floats(-0.9, -0.05),
    x=st.floats(-1.0, 1.5),
)
def test_eq_crl_tail_derivative(u: float, k: float, rho: float, x: float) -> None:
    lp, sign = Model(u, k, rho).p(x)
    h = 1e-5 * u
    fd = (gh_prob(u + h, k, rho, x) - gh_prob(u - h, k, rho, x)) / (2 * h)
    assert abs(sign * np.exp(lp) / u - fd) < 1e-6 + 1e-4 * abs(fd)


@settings(max_examples=15, deadline=None)
@given(
    u=st.floats(0.05, 3.0),
    k=st.floats(0.05, 4.0),
    rho=st.floats(-0.99, -0.01),
    dl=st.floats(0.0, 5.0),
)
def test_wing_signs(u: float, k: float, rho: float, dl: float) -> None:
    mod = Model(u, k, rho)
    l1, l2 = mod.thresholds()
    assert mod.p(l1 - dl)[1] <= 0
    assert mod.p(l2 + dl)[1] >= 0


def test_l1_lemma_symbolic() -> None:
    # h(q) = -ln(1 + W(q)) has h'(q) = -e^{-W}/(1+W)^2, which increases with W (hence q),
    # so h is convex: 1/(1 + W(q)) is log-convex, m(u) is log-convex and ln m(0) = 0.
    q = sp.Symbol("q", positive=True)
    w = sp.LambertW(q)
    hp = sp.diff(-sp.log(1 + w), q)
    ww = sp.Symbol("W", positive=True)
    resid = (hp + sp.exp(-w) / (1 + w) ** 2).subs(w, ww).subs(q, ww * sp.exp(ww))
    assert sp.simplify(resid) == 0
    dd = sp.diff(sp.exp(-ww) / (1 + ww) ** 2, ww)
    assert sp.simplify(dd + sp.exp(-ww) * (3 + ww) / (1 + ww) ** 3) == 0


@settings(max_examples=25, deadline=None)
@given(u=st.floats(0.01, 5.0), k=st.floats(0.05, 6.0), rho=st.floats(-0.99, -0.01))
def test_l1_positive(u: float, k: float, rho: float) -> None:
    assert Model(u, k, rho).thresholds()[0] > 0


@settings(max_examples=15, deadline=None)
@given(
    u=st.floats(0.05, 3.0),
    k=st.floats(0.05, 4.0),
    rho=st.floats(-0.99, -0.01),
    dl=st.floats(0.0, 2.0),
)
def test_sharp_right_wing(u: float, k: float, rho: float, dl: float) -> None:
    mod = Model(u, k, rho)
    l2s = mod.l2_sharp()
    assert l2s <= mod.thresholds()[1] + 1e-12
    assert mod.p(l2s + dl)[1] >= 0


def r_of_l(mod: Model, big_l: float) -> float:
    d2, y = mod.d2(mod.u, mod.lm, big_l)
    lw = mod.lw + norm.logpdf(d2) - 0.5 * np.log((1 - mod.rho**2) * y)
    return float(np.sum(np.exp(lw - logsumexp(lw)) * y / 2))


@settings(max_examples=10, deadline=None)
@given(u=st.floats(0.1, 3.0), k=st.floats(0.1, 4.0), rho=st.floats(-0.95, -0.01))
def test_r_slope(u: float, k: float, rho: float) -> None:
    mod = Model(u, k, rho)
    l1, l2 = mod.thresholds()
    x = 0.5 * (l1 + l2)
    h = 1e-5
    fd = (r_of_l(mod, x + h) - r_of_l(mod, x - h)) / (2 * h)
    assert abs(mod.r_slope(x) - fd) < 1e-5 + 1e-4 * abs(fd)


@settings(max_examples=10, deadline=None)
@given(u=st.floats(0.1, 5.0), k=st.floats(0.1, 6.0), rho=st.floats(-0.99, -0.01))
def test_curvature(u: float, k: float, rho: float) -> None:
    # s^2 V''(t) = -d^2/dZ^2 ln nu(Z), by finite differences on the model's Z grid
    mod = Model(u, k, rho)
    x = 0.5 * sum(mod.thresholds())
    d2, y = mod.d2(mod.u, mod.lm, x)
    lw = mod.lw + norm.logpdf(d2) - 0.5 * np.log(y)
    dz = mod.z[1] - mod.z[0]
    fd = -(lw[2:] - 2 * lw[1:-1] + lw[:-2]) / dz**2
    keep = lw[1:-1] > lw.max() + np.log(1e-12)
    assert abs(fd[keep].min() - mod.curvature(x)) < 1e-3 * (1 + abs(fd[keep].min()))


@settings(max_examples=30, deadline=None)
@given(
    a=st.floats(0.01, 3.0),
    b=st.floats(0.01, 3.0),
    c=st.floats(0.1, 2.0),
    m=st.floats(0.1, 2.0),
    e=st.floats(0.0, 1.0),
)
def test_comparison_lemma(a: float, b: float, c: float, m: float, e: float) -> None:
    # nu ~ e^{-V}, V = c t^2/2 + a e^t + b e^{-t}; g = m V' - D with D' >= 0, so g' <= m V''
    t = np.linspace(-15.0, 15.0, 200_001)
    v = c * t**2 / 2 + a * np.exp(t) + b * np.exp(-t)
    nu = np.exp(-(v - v.min()))
    nu /= nu.sum()
    g = m * (c * t + a * np.exp(t) - b * np.exp(-t)) - e * np.tanh(t)
    f = np.exp(t / 2)
    cov = np.sum(nu * f * g) - np.sum(nu * f) * np.sum(nu * g)
    assert cov <= m * np.sum(nu * f / 2) + 1e-9


@settings(max_examples=15, deadline=None)
@given(u=st.floats(0.05, 5.0), k=st.floats(0.05, 6.0), rho=st.floats(-0.99, 0.0))
def test_split(u: float, k: float, rho: float) -> None:
    mod = Model(u, k, rho)
    x = 0.5 * sum(mod.thresholds())
    first, cov = mod.split(x)
    rp = (first + rho / 2 * cov) / (1 - rho**2)
    assert abs(rp - mod.r_slope(x)) < 1e-8 * (1 + abs(rp))


@settings(max_examples=20, deadline=None)
@given(u=st.floats(0.05, 20.0), k=st.floats(0.05, 2.5), frac=st.floats(0.0, 1.0))
def test_caffarelli_rho_zero(u: float, k: float, frac: float) -> None:
    # rho = 0: R' = L' E sinh^2((X-Y)/2) <= (kappa - 1/s^2)(e^{1/kappa} - 1) < 1 for s^2 <= 2
    mod = Model(u, k, 0.0)
    big_l = frac * mod.thresholds()[1]
    kappa = big_l / 2 + 1 / mod.s**2
    bound = (kappa - 1 / mod.s**2) * np.expm1(1 / kappa)
    assert mod.r_slope(big_l) <= bound + 1e-9
    assert mod.s**2 > 2 or bound < 1


@settings(max_examples=50, deadline=None)
@given(c=st.floats(0.5, 50.0), frac=st.floats(1e-6, 1.0))
def test_caffarelli_scalar(c: float, frac: float) -> None:
    x = frac / c
    assert (1 - c * x) * np.expm1(x) < x


def test_local_margin_symbolic() -> None:
    # (1-r^2)(V'' - e^t G') = (1-r^2)/s^2 + q'^2 + (a-b)^2/4 + r b/s + r Z (a-b)/4,
    # a = L' e^{-t/2}, b = e^{t/2}/2, Z = (t - t0)/s, G = -(L'/y + r Z/sqrt y)/(2(1-r^2))
    t, t0, s, lp, r = sp.symbols("t t0 s Lp r", positive=True)
    zz = (t - t0) / s
    q = lp * sp.exp(-t / 2) + sp.exp(t / 2) / 2 + r * zz
    v = zz**2 / 2 + t / 2 + q**2 / (2 * (1 - r**2))
    g = -(lp * sp.exp(-t) + r * zz * sp.exp(-t / 2)) / (2 * (1 - r**2))
    a, b = lp * sp.exp(-t / 2), sp.exp(t / 2) / 2
    lam = (
        (1 - r**2) / s**2 + sp.diff(q, t) ** 2 + (a - b) ** 2 / 4 + r * b / s + r * zz * (a - b) / 4
    )
    lhs = (1 - r**2) * (sp.diff(v, t, 2) - sp.exp(t) * sp.diff(g, t))
    assert sp.simplify(sp.expand(lhs - lam)) == 0
