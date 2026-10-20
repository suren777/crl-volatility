"""eq_crl_conditional_forward, eq_crl_rho0, eq_crl_k0: the CRL martingale identities.

Model: ln S_T = ln F - ln m + rho sqrt(V tau) Z - V tau / 2 + sqrt((1-rho^2) V tau) eps,
eps perp Z, V = zeta Q(tau), M(Z) = exp(rho sqrt(V tau) Z - rho^2 V tau / 2), m = E[M].

- eq_crl_conditional_forward: E[S_T | Z] = F M(Z)/m, Var(ln S_T | Z) = (1-rho^2) V tau,
  hence the unconditional mean E[S_T] = F (numeric: crl_call at K -> 0 equals D F).
- eq_crl_rho0: rho = 0 gives M = 1, m = 1, conditional variance V tau: the plain MSV
  variance mixture (numeric: crl_call == gh_call_fwd(beta=0)).
- eq_crl_k0: k -> 0 gives zeta -> 1, so ln S_T is normal with variance Q(tau) tau for ANY
  rho (numeric: crl_call with k = 1e-6 equals the independent Black formula).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import checks  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sp  # type: ignore[import-untyped]  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import msv  # noqa: E402

x, a, b = sp.symbols("x a b", real=True)
PDF = sp.exp(-(x**2) / 2) / sp.sqrt(2 * sp.pi)
MGF = sp.simplify(sp.integrate(sp.exp(a + b * x) * PDF, (x, -sp.oo, sp.oo)))
F, m, Z, Vt, rho = sp.symbols("F m Z Vt rho", positive=True)
M = sp.exp(rho * sp.sqrt(Vt) * Z - rho**2 * Vt / 2)


def test_eq_crl_conditional_forward_symbolic() -> None:
    """E[exp(a + b eps)] = exp(a + b^2/2) gives E[S_T|Z] = F M(Z)/m;
    Var(ln S_T | Z) = (1-rho^2) V tau."""
    assert sp.simplify(MGF - sp.exp(a + b**2 / 2)) == 0, MGF
    a_sub = sp.log(F) - sp.log(m) + rho * sp.sqrt(Vt) * Z - Vt / 2
    cond_mean = MGF.subs({a: a_sub, b: sp.sqrt((1 - rho**2) * Vt)})
    diff = sp.simplify(sp.powsimp(sp.expand(cond_mean - F * M / m), force=True))
    assert diff == 0, diff
    var = sp.simplify(
        sp.integrate((a + b * x) ** 2 * PDF, (x, -sp.oo, sp.oo))
        - sp.integrate((a + b * x) * PDF, (x, -sp.oo, sp.oo)) ** 2
    )
    assert sp.simplify(var - b**2) == 0  # so Var(ln S_T | Z) = (1-rho^2) V tau
    print(
        "ok  eq_crl_conditional_forward symbolic: E[S_T|Z] = F M(Z)/m, cond var = (1-rho^2) V tau"
    )


@settings(max_examples=15, deadline=None)
@given(
    a0=st.floats(0.0, 0.1),
    a1=st.floats(0.0, 0.1),
    a2=st.floats(0.01, 0.1),
    lam=st.floats(0.2, 4.0),
    k=st.floats(0.1, 2.0),
    rho=st.floats(-0.95, 0.95),
    tau=st.floats(0.05, 3.0),
)
def test_eq_crl_conditional_forward_numeric(
    a0: float, a1: float, a2: float, lam: float, k: float, rho: float, tau: float
) -> None:
    """Unconditional mean E[S_T] = F: crl_call at K -> 0 recovers D F (the -K Phi(d2)
    floor puts the relative error at K/F, so K = 1e-9 F; see test_msv.py)."""
    F0 = 100.0 * float(np.exp(0.02 * tau))
    px = float(msv.crl_call(F0, 1e-9 * F0, 1.0, tau, a0, a1, a2, lam, k, rho)[0])
    assert checks.relerr(px, F0) < 1e-8, (a0, a1, a2, lam, k, rho, tau, px, F0)
    print(
        f"ok  eq_crl_conditional_forward numeric: lim K->0 crl_call == D F to 1e-8 (rho={rho:.2f})"
    )


def test_eq_crl_rho0_symbolic() -> None:
    """rho = 0: M = 1 (so m = 1) and the conditional variance is V tau: the MSV mixture."""
    assert sp.simplify(M.subs(rho, 0) - 1) == 0
    assert sp.simplify(((1 - rho**2) * Vt).subs(rho, 0) - Vt) == 0
    print("ok  eq_crl_rho0 symbolic: M = 1, m = 1, conditional variance V tau")


@settings(max_examples=15, deadline=None)
@given(
    a0=st.floats(0.0, 0.1),
    a1=st.floats(0.0, 0.1),
    a2=st.floats(0.01, 0.1),
    lam=st.floats(0.2, 4.0),
    k=st.floats(0.1, 2.0),
    tau=st.floats(0.05, 3.0),
)
def test_eq_crl_rho0_numeric(
    a0: float, a1: float, a2: float, lam: float, k: float, tau: float
) -> None:
    """crl_call(rho=0) == gh_call_fwd(beta=0) for random parameters."""
    r = 0.02
    F0 = 100.0 * float(np.exp(r * tau))
    D = float(np.exp(-r * tau))
    ks = F0 * np.array([0.8, 1.0, 1.2])
    # 129 nodes on both sides: the defaults (64 vs 32) differ by ~3e-8 in quadrature
    # residual at k=2, which would swamp the 1e-9 identity tolerance.
    crl = msv.crl_call(F0, ks, D, tau, a0, a1, a2, lam, k, 0.0, n_nodes=129)
    gh = msv.gh_call_fwd(F0, ks, D, tau, a0, a1, a2, lam, k, beta=0.0, n_nodes=129)
    assert np.all(np.abs(crl - gh) < 1e-9), (a0, a1, a2, lam, k, tau, crl, gh)
    print(f"ok  eq_crl_rho0 numeric: crl_call(rho=0) == gh_call_fwd(beta=0) to 1e-9 (k={k:.2f})")


def test_eq_crl_k0_symbolic() -> None:
    """k = 0: zeta -> 1, m = E[M] = 1 by the Gaussian mgf, so Var(ln S_T) = rho^2 Q tau
    + (1-rho^2) Q tau = Q tau with mean ln F: Black-Scholes for any rho."""
    q, tau = sp.symbols("q tau", positive=True)
    mm = sp.integrate(
        sp.exp(rho * sp.sqrt(q * tau) * x - rho**2 * q * tau / 2) * PDF, (x, -sp.oo, sp.oo)
    )
    assert sp.simplify(mm - 1) == 0, mm
    total_var = rho**2 * q * tau + (1 - rho**2) * q * tau
    assert sp.simplify(total_var - q * tau) == 0
    print("ok  eq_crl_k0 symbolic: m = 1 and rho^2 Q tau + (1-rho^2) Q tau = Q tau")


@settings(max_examples=15, deadline=None)
@given(
    a0=st.floats(0.0, 0.1),
    a1=st.floats(0.0, 0.1),
    a2=st.floats(0.01, 0.1),
    lam=st.floats(0.2, 4.0),
    rho=st.floats(-0.95, 0.95),
    tau=st.floats(0.05, 3.0),
    kf=st.floats(0.7, 1.5),
)
def test_eq_crl_k0_numeric(
    a0: float, a1: float, a2: float, lam: float, rho: float, tau: float, kf: float
) -> None:
    """crl_call with k = 1e-6 matches the independent plain-numpy Black formula.

    At k = 1e-6 the model is not exactly Black-Scholes: the conditional forward
    F M(Z)/m jiggles at O(k) across quadrature nodes (worst measured 2.2e-4
    relative, deep OTM + short tau + |rho| large), so the tolerance is 1e-3; the
    sharp statement that the limit itself is Black is test_eq_crl_k0_limit_linear.
    """
    F0 = 100.0
    q = float(msv.Q(tau, a0, a1, a2, lam))
    px = float(msv.crl_call(F0, kf * F0, 1.0, tau, a0, a1, a2, lam, 1e-6, rho)[0])
    ref = float(checks.black_fwd(F0, kf * F0, tau, np.sqrt(q)))
    assert checks.relerr(px, ref) < 1e-3, (a0, a1, a2, lam, rho, tau, kf, px, ref)
    print(f"ok  eq_crl_k0 numeric: crl_call(k=1e-6) == independent Black to 1e-3 (rho={rho:.2f})")


def test_eq_crl_k0_limit_linear() -> None:
    """The deviation from Black shrinks linearly in k (ratio ~ 10 for k 1e-6 -> 1e-7),
    so the k -> 0 limit is exactly Black-Scholes with variance rate Q(tau)."""
    a0, a1, a2, lam, rho, tau, kf = 0.05, 0.05, 0.04, 1.5, 0.7, 0.5, 1.3
    q = float(msv.Q(tau, a0, a1, a2, lam))
    ref = float(checks.black_fwd(100.0, kf * 100.0, tau, np.sqrt(q)))

    def relerr_at(k: float) -> float:
        px = float(msv.crl_call(100.0, kf * 100.0, 1.0, tau, a0, a1, a2, lam, k, rho)[0])
        return checks.relerr(px, ref)

    e6, e7 = relerr_at(1e-6), relerr_at(1e-7)
    assert e7 > 1e-13, (e6, e7)  # deviation must be measurable at k = 1e-7
    assert checks.relerr(e6, 10.0 * e7) < 0.2, (e6, e7)
    print(f"ok  eq_crl_k0 limit: deviation linear in k (relerr {e6:.2e} -> {e7:.2e})")


if __name__ == "__main__":
    test_eq_crl_conditional_forward_symbolic()
    test_eq_crl_conditional_forward_numeric()
    test_eq_crl_rho0_symbolic()
    test_eq_crl_rho0_numeric()
    test_eq_crl_k0_symbolic()
    test_eq_crl_k0_numeric()
    test_eq_crl_k0_limit_linear()
