"""eq_crl_moment, eq_crl_calendar_witness: the collapsed CRL moments and the certified
calendar-arbitrage witness.

Model as in test_crl_identities.py: S_tau = (F/m) exp(sqrt(t zeta) U - t zeta / 2),
t = tau Q(tau), zeta = exp(s Z - s^2/2), s^2 = ln(1+k^2), U = rho Z + sqrt(1-rho^2) eps.

- eq_crl_moment: E[S^p] = (F/m)^p E_Z[exp(p rho sqrt(t zeta) Z + (t zeta / 2)
  (p^2 (1-rho^2) - p))], finite iff p < 1/(1-rho^2). Checked symbolically (the Gaussian
  mgf collapses U | Z), against the independent 2-D (Z, N) tensor grid, and through the
  p = 1 identity E[S] = F. For rho > 0 the p-moment integrand has a far-tail Laplace bump
  beyond any linear-space grid, so the 2-D cross-check is restricted to rho <= 0 (see
  calendar_moments.md); for rho > 0 the formula rests on the algebra, the p = 1 identity,
  and the resolution/clip gates of the certificate.
- eq_crl_calendar_witness: for rho > 0, ln E[S^p] strictly decreases in tau (certified in
  calendar_moments.md, 12 witnesses) — by the Kellerer bridge (calendar_proof.md section 3)
  this proves calendar arbitrage exists. The mildest witness is recomputed here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import checks  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sp  # type: ignore[import-untyped]  # noqa: E402
from calendar_moments import (  # noqa: E402
    CLIP_WIDE,
    N_NODES,
    N_SCAN,
    TAUS,
    call_2d,
    ln_eexp,
    lse,
    moment_2d,
    phi_m,
    phi_p,
    s_loggrid,
    var_time,
)
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import msv  # noqa: E402

x = sp.symbols("x", real=True)
PDF = sp.exp(-(x**2) / 2) / sp.sqrt(2 * sp.pi)
a, b = sp.symbols("a b", real=True)
Z, Vt, rho_s, p_s = sp.symbols("Z Vt rho p", positive=True)

NS_FLAT = (0.0, 0.0, 0.04, 1.0)
NS_HUMPED = (0.0, 0.06, 0.03, 1.5)


def test_eq_crl_moment_symbolic() -> None:
    """Conditioning S^p on Z and integrating U | Z ~ N(rho Z, (1-rho^2) t zeta) by the
    Gaussian mgf reproduces the collapsed exponent p rho sqrt(tz) Z + (tz / 2)
    (p^2 (1-rho^2) - p); at p = 1 the t zeta-coefficient reduces it to ln M(Z), so
    E[S | Z] = F M(Z)/m and E[S] = F."""
    MGF = sp.simplify(sp.integrate(sp.exp(a + b * x) * PDF, (x, -sp.oo, sp.oo)))
    assert sp.simplify(MGF - sp.exp(a + b**2 / 2)) == 0, MGF
    b_sub = p_s * sp.sqrt((1 - rho_s**2) * Vt)
    a_sub = -p_s * Vt / 2 + p_s * rho_s * sp.sqrt(Vt) * Z
    exponent = a_sub + b_sub**2 / 2
    expected = p_s * rho_s * sp.sqrt(Vt) * Z + Vt / 2 * (p_s**2 * (1 - rho_s**2) - p_s)
    assert sp.simplify(exponent - expected) == 0, (exponent, expected)
    ln_m = rho_s * sp.sqrt(Vt) * Z - rho_s**2 * Vt / 2
    assert sp.simplify(exponent.subs(p_s, 1) - ln_m) == 0
    print("ok  eq_crl_moment symbolic: collapse exponent == phi_p; p = 1 reduces to ln M(Z)")


@settings(max_examples=10, deadline=None)
@given(
    a0=st.floats(0.0, 0.1),
    a1=st.floats(0.0, 0.1),
    a2=st.floats(0.01, 0.1),
    lam=st.floats(0.2, 4.0),
    k=st.floats(0.1, 2.0),
    rho=st.floats(-0.95, 0.95),
    tau=st.floats(0.05, 3.0),
)
def test_eq_crl_moment_p1_identity(
    a0: float, a1: float, a2: float, lam: float, k: float, rho: float, tau: float
) -> None:
    """p = 1 identity: phi_p(., 1.0) and phi_m assemble the same integrand, so the
    collapsed quadrature returns ln E[S] = 0 exactly for every rho (window boundary)."""
    t = var_time(tau, (a0, a1, a2, lam))
    s = float(np.sqrt(np.log1p(k * k)))
    lp1 = ln_eexp(phi_p(t, s, rho, 1.0), n_points=100_001)
    lm = ln_eexp(phi_m(t, s, rho), n_points=100_001)
    # float noise: c = (1-rho^2) - 1 differs from -rho^2 by ~1e-16, amplified by
    # 0.5 t zeta ~ 1e4 at the bump region for small rho and large t — 1e-9 is still
    # 9 orders below any real formula bug.
    assert abs(lp1 - lm) < 1e-9, (a0, a1, a2, lam, k, rho, tau, lp1, lm)
    print(f"ok  eq_crl_moment p=1 identity exact (rho={rho:.2f}, k={k:.2f}, tau={tau:.2f})")


def test_eq_crl_moment_p1_grid() -> None:
    """E[S] = 1 through the independent 2-D (Z, N) tensor grid at tame parameters."""
    for ns in (NS_FLAT, NS_HUMPED):
        for rho in (-0.5, 0.0):
            ln_s, lw2 = s_loggrid(0.25, ns, 0.3, rho, N_NODES)
            es = float(np.exp(lse((ln_s + lw2).ravel())))
            assert abs(es - 1.0) < 1e-12, (ns, rho, es)
    print("ok  eq_crl_moment E[S] = 1 from the 2-D grid to 1e-12 (rho <= 0)")


def test_eq_crl_moment_collapsed() -> None:
    """Collapsed E[S^p] (one-dimensional quadrature after the U | Z collapse) vs the
    independent 2-D (Z, N) tensor grid, at parameters where the 2-D grid is valid
    (rho <= 0: no far-tail Laplace bump; see calendar_moments.md)."""
    for rho, fracs in ((-0.8, (0.1, 0.25, 0.5)), (-0.5, (0.25, 0.5)), (0.0, (0.25,))):
        top = 1.0 / (1.0 - rho**2)
        for frac in fracs:
            p = 1.0 + frac * (top - 1.0)
            for ns in (NS_FLAT, NS_HUMPED):
                for tau in (0.1, 0.25):
                    t = var_time(tau, ns)
                    s = float(np.sqrt(np.log1p(0.3 * 0.3)))
                    lm = ln_eexp(phi_m(t, s, rho))
                    lp = ln_eexp(phi_p(t, s, rho, p))
                    collapsed = lp - p * lm
                    m2 = moment_2d(p, tau, ns, 0.3, rho)
                    assert abs(collapsed - np.log(m2)) < 1e-9, (
                        rho,
                        frac,
                        ns,
                        tau,
                        p,
                        collapsed,
                        m2,
                    )
    print("ok  eq_crl_moment collapsed vs 2-D grid to 1e-9 (rho <= 0, k = 0.3)")


def test_eq_crl_moment_call() -> None:
    """2-D-grid call prices vs msv.crl_call (independent code paths; crl_call's own
    64-node GH is the coarser side for rho > 0, hence the 5e-3 tolerance)."""
    for rho in (-0.5, 0.0, 0.3):
        for ns in (NS_FLAT, NS_HUMPED):
            for kf in (0.8, 1.0, 1.2):
                px = call_2d(kf, 0.1, ns, 0.3, rho)
                ref = float(msv.crl_call(1.0, kf, 1.0, 0.1, *ns, 0.3, rho)[0])
                assert checks.relerr(px, ref) < 5e-3, (rho, ns, kf, px, ref)
    print("ok  eq_crl_moment call_2d vs crl_call to 5e-3")


def test_eq_crl_calendar_witness() -> None:
    """The mildest certified witness of calendar_moments.md (rho = +0.2, k = 2.6,
    humped, p = 1.010417, tau 0.0100 -> 0.0130 as printed = TAUS[0] -> TAUS[1],
    d = -0.880106): recomputed at scan resolution, the strict decrease of ln E[S^p]
    in tau persists — the calendar-arbitrage certificate via the Kellerer bridge
    (calendar_proof.md)."""
    s = float(np.sqrt(np.log1p(2.6 * 2.6)))
    p = 1.010417
    t1, t2 = float(TAUS[0]), float(TAUS[1])
    d = []
    for tau in (t1, t2):
        t = var_time(tau, NS_HUMPED)
        lm = ln_eexp(phi_m(t, s, 0.2), clip=CLIP_WIDE, n_points=N_SCAN)
        lp = ln_eexp(phi_p(t, s, 0.2, p), clip=CLIP_WIDE, n_points=N_SCAN)
        d.append(lp - p * lm)
    inc = d[1] - d[0]
    assert inc < -0.5, inc
    assert abs(inc - (-0.880106)) < 5e-3, inc
    print(f"ok  eq_crl_calendar_witness: d ln E[S^p] = {inc:.6f} < 0 (certified -0.880106)")


if __name__ == "__main__":
    test_eq_crl_moment_symbolic()
    test_eq_crl_moment_p1_identity()
    test_eq_crl_moment_p1_grid()
    test_eq_crl_moment_collapsed()
    test_eq_crl_moment_call()
    test_eq_crl_calendar_witness()
