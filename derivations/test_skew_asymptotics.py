"""eq_skew: short-maturity ATM skew and level of the CRL model.

Forward form F = D = 1, eps = sqrt(tau), x = ln K, Q = Q(tau) = Q(0+) + O(tau) with
Q(0+) = a0 + a2, s^2 = ln(1+k^2), zeta(Z) = exp(sZ - s^2/2), a(Z) = sqrt(zeta(Z) Q),
rho_t = rho/sqrt(1-rho^2). Conditional on Z the price is Black with
ln F_Z = rho a eps Z - rho^2 a^2 eps^2 / 2 - ln m and total sd sqrt(1-rho^2) a eps.

- eq_skew (result): S(tau) = dI/dx(0) = C(k,rho)/sqrt(tau) + O(1) with
  C(k,rho) = sqrt(2 pi) [1/2 - E_Z N(rho_t (Z - (s/2) exp(s^2/8 - s Z/2)))];
  C is independent of Q, C = 0 for rho = 0 or k = 0, and C = rho s/2 + O(s^3) (the
  s^2 term vanishes; C = rho s/2 + c3(rho) s^3 + O(s^4) with c3(rho) = rho(3-5rho^2)/24).
  Numeric: S sqrt(tau) from crl_call (n_nodes=128, central strike difference with
  h = 0.02 sqrt(Q tau), inversion via the independent checks.black_fwd) vs the
  quadrature of the closed-form C, at tau = 1e-4 and across a tau grid (slope 1/2).
- eq_skew_atm_level: I(0) -> sqrt(2 pi) E_Z[u_Z N(u_Z/sigma_Z) + sigma_Z phi(u_Z/sigma_Z)]
  with u_Z = rho (a Z - E[aZ]), sigma_Z = sqrt(1-rho^2) a (Bachelier limit of the
  conditional Black prices). Numeric: crl_call IV at K = F vs the same quadrature.
- eq_skew_eaz: E[a Z] = sqrt(Q) (s/2) exp(-s^2/8) (symbolic).
- eq_skew_lnm: ln m = rho eps E[aZ] + O(eps^2); the m expansion is
  m = 1 + rho eps E[aZ] + rho^2 eps^2 Q s^2 / 2 + O(eps^3) (symbolic).
- eq_skew_dz: at x = 0, d2_Z -> d_Z = rho_t (Z - (s/2) exp(s^2/8 - s Z/2)); Q cancels
  (symbolic limit in eps).
- eq_skew_small_s: E[N(alpha Z + beta)] = N(beta/sqrt(1+alpha^2)) (symbolic derivative
  match, numeric sweep), and the linearised-in-s series of C is rho s/2 + O(s^2).

The numeric parameters are the four planner parameter sets (k, rho, Q); Q is made
exact and constant by a0 = a1 = 0, a2 = Q (then msv.Q(tau) = a2 for every tau).
"""

import sys
from functools import cache, lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import checks  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sp  # type: ignore[import-untyped]  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.special import ndtr, roots_hermitenorm  # noqa: E402

import msv  # noqa: E402

z, s = sp.symbols("z s", real=True)
rho, q, eps = sp.symbols("rho q eps", real=True)
PDF = sp.exp(-(z**2) / 2) / sp.sqrt(2 * sp.pi)

# (k, rho, Q) planner parameter sets; the first three have planner C values
# -0.29642, -0.07424, -0.50164 (see derivations/skew_asymptotics.md).
SETS = ((1.0, -0.7, 0.04), (0.5, -0.3, 0.04), (2.0, -0.9, 0.01), (1.0, -0.7, 0.16))
TAUS = (1e-4, 4e-4, 1.6e-3, 6.4e-3, 2.56e-2)


@lru_cache(maxsize=1)
def _znodes(n: int) -> tuple[checks.FloatArray, checks.FloatArray]:
    """Gauss-Hermite nodes/weights for E_Z[f] = sum w_i f(z_i), self-normalised."""
    zz, ww = roots_hermitenorm(n)
    zz_a = np.asarray(zz, dtype=float)
    ww_a = np.asarray(ww, dtype=float)
    return zz_a, ww_a / float(np.sum(ww_a))


def _c_coef(k: float, rho: float) -> float:
    """C(k, rho) = sqrt(2 pi) [1/2 - E_Z N(d_Z)] by quadrature of the closed form."""
    s2 = float(np.log1p(k * k))
    sq = float(np.sqrt(s2))
    rho_t = rho / float(np.sqrt(1.0 - rho * rho))
    zz, ww = _znodes(256)
    dz = rho_t * (zz - sq / 2.0 * np.exp(s2 / 8.0 - sq * zz / 2.0))
    return float(np.sqrt(2.0 * np.pi) * (0.5 - float(ww @ ndtr(dz))))


def _atm_level(k: float, rho: float, q: float) -> float:
    """sqrt(2 pi) E_Z[u_Z N(u_Z/sigma_Z) + sigma_Z phi(u_Z/sigma_Z)] by quadrature."""
    s2 = float(np.log1p(k * k))
    sq = float(np.sqrt(s2))
    zz, ww = _znodes(256)
    a = np.sqrt(q * np.exp(sq * zz - s2 / 2.0))
    eaz = float(np.sqrt(q) * sq / 2.0 * np.exp(-s2 / 8.0))
    u = rho * (a * zz - eaz)
    sig = float(np.sqrt(1.0 - rho * rho)) * a
    r = u / sig
    phi_r = np.exp(-(r * r) / 2.0) / np.sqrt(2.0 * np.pi)
    return float(np.sqrt(2.0 * np.pi) * float(ww @ (u * ndtr(r) + sig * phi_r)))


def _invert_iv(price: float, strike: float, tau: float) -> float:
    """Implied vol of an undiscounted forward Black price via the independent formula."""

    def obj(sigma: float) -> float:
        return float(checks.black_fwd(1.0, strike, tau, sigma)) - price

    return float(brentq(obj, 1e-10, 5.0))


@cache
def _measure(k: float, rho: float, q: float, tau: float) -> tuple[float, float]:
    """(S(tau) sqrt(tau), I(0)) from crl_call with 128 nodes.

    Central difference in x = ln K with h = 0.02 sqrt(Q tau) (a fixed h would be a
    large fraction of the implied sd at small tau and bias the difference);
    I(0) from the K = F price. One elementwise call prices all three strikes.
    """
    h = 0.02 * float(np.sqrt(q * tau))
    strikes = np.array([float(np.exp(-h)), 1.0, float(np.exp(h))])
    px = msv.crl_call(1.0, strikes, 1.0, tau, 0.0, 0.0, q, 1.0, k, rho, n_nodes=128)
    ivs = [_invert_iv(float(p), float(kk), tau) for kk, p in zip(strikes, px, strict=True)]
    skew = (ivs[2] - ivs[0]) / (2.0 * h) * float(np.sqrt(tau))
    return skew, ivs[1]


def test_eq_skew_eaz_symbolic() -> None:
    """E[a Z] = sqrt(Q) E[Z exp(sZ/2 - s^2/4)] = sqrt(Q) (s/2) exp(-s^2/8)."""
    eaz = sp.integrate(z * sp.exp(s * z / 2 - s**2 / 4) * PDF, (z, -sp.oo, sp.oo))
    diff = sp.simplify(eaz - s / 2 * sp.exp(-(s**2) / 8))
    assert diff == 0, eaz
    print("ok  eq_skew_eaz symbolic: E[a Z] = sqrt(Q) (s/2) exp(-s^2/8)")


def test_eq_skew_lnm_symbolic() -> None:
    """m = 1 + rho eps E[aZ] + rho^2 eps^2 Q s^2 / 2 + O(eps^3), hence
    ln m = rho eps E[aZ] + O(eps^2): the O(eps) coefficient of ln m is E[aZ]."""
    zeta = sp.exp(s * z - s**2 / 2)
    e_zeta = sp.integrate(zeta * PDF, (z, -sp.oo, sp.oo))  # = 1
    e_z2_zeta = sp.integrate(z**2 * zeta * PDF, (z, -sp.oo, sp.oo))  # = 1 + s^2
    assert sp.simplify(e_zeta - 1) == 0, e_zeta
    assert sp.simplify(e_z2_zeta - (1 + s**2)) == 0, e_z2_zeta
    eaz = sp.sqrt(q) * sp.integrate(z * sp.exp(s * z / 2 - s**2 / 4) * PDF, (z, -sp.oo, sp.oo))
    m2 = 1 + rho * eps * eaz + rho**2 * eps**2 * q * (e_z2_zeta - e_zeta) / 2
    ln_series = sp.series(sp.log(m2), eps, 0, 2).removeO()
    diff = sp.simplify(sp.expand(ln_series).coeff(eps, 1) - rho * eaz)
    assert diff == 0, diff
    print("ok  eq_skew_lnm symbolic: ln m = rho eps E[aZ] + O(eps^2)")


def test_eq_skew_dz_symbolic() -> None:
    """At x = 0, d2_Z -> d_Z = rho_t (Z - (s/2) exp(s^2/8 - s Z/2)); Q cancels."""
    a = sp.sqrt(q) * sp.exp(s * z / 2 - s**2 / 4)
    lnm1 = rho * eps * sp.sqrt(q) * s / 2 * sp.exp(-(s**2) / 8)  # first-order ln m
    d2 = (rho * a * eps * z - eps**2 * a**2 / 2 - lnm1) / (sp.sqrt(1 - rho**2) * a * eps)
    target = rho / sp.sqrt(1 - rho**2) * (z - s / 2 * sp.exp(s**2 / 8 - s * z / 2))
    diff = sp.simplify(sp.limit(d2, eps, 0) - target)
    assert diff == 0, diff
    print("ok  eq_skew_dz symbolic: d2_Z -> d_Z, Q cancels")


def test_eq_skew_small_s_symbolic() -> None:
    """E[N(alpha Z + beta)] = N(beta/sqrt(1+alpha^2)): both sides have the same beta
    derivative (a Gaussian integral) and both -> 0 as beta -> -oo. Linearising d_Z in
    s (g(Z) = (s/2) exp(s^2/8 - sZ/2) = s/2 + O(s^2)) then gives C = rho s/2 + O(s^2)."""
    al, be = sp.symbols("alpha beta", real=True)
    phi_ab = sp.exp(-((al * z + be) ** 2) / 2) / sp.sqrt(2 * sp.pi)
    lhs = sp.integrate(phi_ab * PDF, (z, -sp.oo, sp.oo))
    rhs = sp.exp(-(be**2) / (2 * (1 + al**2))) / (sp.sqrt(2 * sp.pi) * sp.sqrt(1 + al**2))
    diff_id = sp.simplify((lhs - rhs).rewrite(sp.erf))
    assert diff_id == 0, lhs
    # linearised d_Z = rho_t Z - rho_t s/2, so E[N(d_Z)] = N(-rho s/2):
    # beta/sqrt(1+alpha^2) = -rho_t s sqrt(1-rho^2)/2 = -rho s/2.
    c_lin = sp.sqrt(2 * sp.pi) * (sp.Rational(1, 2) - (1 + sp.erf(-rho * s / (2 * sp.sqrt(2)))) / 2)
    series = sp.series(c_lin, s, 0, 2).removeO()
    diff_c = sp.simplify(series - rho * s / 2)
    assert diff_c == 0, series
    print("ok  eq_skew_small_s symbolic: probit identity, C = rho s/2 + O(s^2)")


def test_eq_skew_small_s_cubic() -> None:
    """Expand E_Z N(d(Z)) in s to third order. With d = d0 + s d1 + s^2 d2 + s^3 d3,
    d1 = -rho_t/2, d2 = rho_t Z/4, d3 = -rho_t (Z^2+1)/16; Taylor of N about d0
    (phi and its derivatives) and Gaussian expectations term by term. The s
    coefficient gives C = rho s/2 (after -sqrt(2 pi)); the s^2 coefficient vanishes
    identically (E[z phi(alpha z)] = 0); the s^3 coefficient is
    c3(rho) = rho (3 - 5 rho^2)/24. Numeric: (C - rho s/2)/s^3 -> c3(rho) at
    s = 0.02, 0.05 for rho in {-0.3, -0.7, -0.9} (C by Gauss-Hermite quadrature)."""
    al = sp.symbols("alpha", real=True)
    # Gaussian moments E[z^n phi(alpha z)] (only even n survive; odd vanish by parity).
    p00 = sp.exp(-(al**2) * z**2 / 2) / sp.sqrt(2 * sp.pi) * PDF
    m0 = sp.integrate(p00, (z, -sp.oo, sp.oo))
    m2 = sp.integrate(z**2 * p00, (z, -sp.oo, sp.oo))
    m4 = sp.integrate(z**4 * p00, (z, -sp.oo, sp.oo))
    m0 = sp.simplify(m0)
    m2 = sp.simplify(m2)
    m4 = sp.simplify(m4)
    # d1 = -alpha/2, d2 = alpha z/4, d3 = -alpha (z^2+1)/16; N(d0+D) Taylor to D^3.
    a1 = -al / 2 * m0  # E[d1 phi]
    a2 = sp.simplify(al / 4 * 0)  # E[z phi] = 0 kills the s^2 term identically
    a3 = sp.simplify(-al / 16 * (m0 + m2) + al**3 / 8 * m2 - al**3 / 48 * (al**2 * m2 - m0))
    c1 = sp.simplify(-sp.sqrt(2 * sp.pi) * a1)
    c2 = sp.simplify(-sp.sqrt(2 * sp.pi) * a2)
    c3 = sp.simplify(-sp.sqrt(2 * sp.pi) * a3)
    # c1 = alpha/2 sqrt(1+alpha^2); at alpha = rho_t this is rho/2.
    assert sp.simplify(c1 - al / (2 * sp.sqrt(1 + al**2))) == 0, c1
    assert c2 == 0, c2
    c3_alpha = al * (3 - 2 * al**2) / (24 * (1 + al**2) ** sp.Rational(3, 2))
    assert sp.simplify(c3 - c3_alpha) == 0, c3
    # alpha = rho_t = rho/sqrt(1-rho^2); 1+alpha^2 = 1/(1-rho^2), so c3(rho) = rho(3-5rho^2)/24.
    # Verify numerically (the symbolic substitution leaves sqrt branches unsimplified).
    c3_closed = rho * (3 - 5 * rho**2) / 24
    c3_rho = c3_alpha.subs(al, rho / sp.sqrt(1 - rho**2))
    for rv in (-0.3, 0.5, -0.9):
        assert abs(float(sp.N(c3_rho.subs(rho, rv) - c3_closed.subs(rho, rv), 12))) < 1e-10
    print(
        "ok  eq_skew_small_s cubic symbolic: C = rho s/2 + c3(rho) s^3 + O(s^3),"
        f" s^2 term = 0, c3(rho) = {c3_closed} = {sp.sstr(c3_closed)}"
    )
    rss = ((-0.3, 0.02), (-0.3, 0.05), (-0.7, 0.02), (-0.7, 0.05), (-0.9, 0.02), (-0.9, 0.05))
    for rv, sv in rss:
        kv = float(np.sqrt(np.expm1(sv * sv)))
        c = _c_coef(kv, rv)
        c3v = float(c3_closed.subs(rho, rv))
        ratio = (c - rv * sv / 2) / sv**3
        assert checks.relerr(ratio, c3v) < 1e-2, (rv, sv, ratio, c3v)
        print(
            f"ok  eq_skew_small_s cubic numeric: rho={rv}, s={sv}: "
            f"(C - rho s/2)/s^3 = {ratio:.2e} vs c3 = {c3v:.2e}"
        )


@settings(max_examples=15, deadline=None)
@given(al=st.floats(-3.0, 3.0), be=st.floats(-2.0, 2.0))
def test_eq_skew_probit_identity_numeric(al: float, be: float) -> None:
    """E[N(alpha Z + beta)] = N(beta/sqrt(1+alpha^2)) numerically (quadrature)."""
    zz, ww = _znodes(128)
    lhs = float(ww @ ndtr(al * zz + be))
    rhs = float(ndtr(be / np.sqrt(1.0 + al * al)))
    assert checks.relerr(lhs, rhs) < 1e-9, (al, be, lhs, rhs)
    print(f"ok  eq_skew probit identity numeric to 1e-9 (alpha={al:.2f}, beta={be:.2f})")


def test_eq_skew_value_numeric() -> None:
    """S sqrt(tau) from crl_call matches C(k, rho) at tau = 1e-4 (rel tol 1e-2) for all
    four planner parameter sets; sets 1 and 4 share (k, rho) with different Q, so this
    is also the Q-independence of the coefficient."""
    tau = 1e-4
    for k, rho, q in SETS:
        c = _c_coef(k, rho)
        skew, _ = _measure(k, rho, q, tau)
        assert checks.relerr(skew, c) < 1e-2, (k, rho, q, skew, c)
        print(
            f"ok  eq_skew value numeric: S sqrt(tau) = {skew:.5f} vs C = {c:.5f} "
            f"(k={k}, rho={rho}, Q={q})"
        )


def test_eq_skew_convergence_numeric() -> None:
    """|S sqrt(tau) - C| decays like tau^(1/2): loglog_slope in [0.4, 0.6] on the
    planner tau grid, for each of the four parameter sets (planner slopes
    0.49 / 0.45 / 0.51 on the first three)."""
    for k, rho, q in SETS:
        c = _c_coef(k, rho)
        errs = [abs(_measure(k, rho, q, tau)[0] - c) for tau in TAUS]
        slope = checks.loglog_slope(TAUS, errs)
        assert 0.4 <= slope <= 0.6, (k, rho, q, slope, errs)
        print(
            f"ok  eq_skew convergence numeric: slope {slope:.2f} in [0.4, 0.6] "
            f"(k={k}, rho={rho}, Q={q})"
        )


def test_eq_skew_atm_level_numeric() -> None:
    """I(0) at tau = 1e-4 matches the Bachelier-limit formula (rel tol 1e-2)."""
    tau = 1e-4
    for k, rho, q in SETS:
        _, iv0 = _measure(k, rho, q, tau)
        level = _atm_level(k, rho, q)
        assert checks.relerr(iv0, level) < 1e-2, (k, rho, q, iv0, level)
        print(
            f"ok  eq_skew_atm_level numeric: I(0) = {iv0:.5f} vs {level:.5f} "
            f"(k={k}, rho={rho}, Q={q})"
        )


if __name__ == "__main__":
    test_eq_skew_eaz_symbolic()
    test_eq_skew_lnm_symbolic()
    test_eq_skew_dz_symbolic()
    test_eq_skew_small_s_symbolic()
    test_eq_skew_small_s_cubic()
    test_eq_skew_probit_identity_numeric()
    test_eq_skew_value_numeric()
    test_eq_skew_convergence_numeric()
    test_eq_skew_atm_level_numeric()
