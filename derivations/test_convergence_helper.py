"""test_convergence_helper: loglog_slope recovers the O(tau) error of the ATM short-maturity
Black approximation C ~ F sigma sqrt(tau) / sqrt(2 pi).

Derivation (SymPy series below): at K = F, w = sigma sqrt(tau),
C = F (2 Phi(w/2) - 1) = F erf(w / (2 sqrt(2)))
  = F sigma sqrt(tau)/sqrt(2 pi) (1 - w^2/24 + O(w^4)),
so the relative error is -w^2/24 = -sigma^2 tau / 24: order tau.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import checks  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sp  # type: ignore[import-untyped]  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


def test_loglog_slope_recovers_power_laws() -> None:
    """The helper returns the exact exponent on synthetic power laws."""
    rng = np.random.default_rng(0)
    xs = np.float64(10.0) ** rng.uniform(-3.0, 1.0, 12)
    for p in (0.5, 1.0, 2.0, -0.5):
        assert abs(checks.loglog_slope(xs, 3.0 * xs**p) - p) < 1e-8, p
    print("ok  loglog_slope recovers exact power laws")


def test_eq_atm_black_order_symbolic() -> None:
    """SymPy series: relative error of the leading-term ATM call is -w^2/24, i.e. O(tau)."""
    w = sp.symbols("w", positive=True)
    lead = w / sp.sqrt(2 * sp.pi)
    exact = sp.series(sp.erf(w / (2 * sp.sqrt(2))), w, 0, 5).removeO()
    rel = sp.series(exact / lead - 1, w, 0, 3).removeO()
    assert sp.simplify(rel - (-(w**2) / 24)) == 0, rel
    print(
        "ok  eq_atm_black order symbolic: C = F sigma sqrt(tau)/sqrt(2 pi)"
        " (1 - sigma^2 tau/24 + O(tau^2))"
    )


@settings(max_examples=10, deadline=None)
@given(sigma=st.floats(0.05, 1.5))
def test_eq_atm_black_order_numeric(sigma: float) -> None:
    """loglog_slope recovers order ~1 for the relative error over tau in [1e-4, 1e-2]."""
    taus = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    w = sigma * np.sqrt(taus)
    lead = 100.0 * w / np.sqrt(2.0 * np.pi)
    exact = np.array([float(checks.black_fwd(100.0, 100.0, float(t), sigma)) for t in taus])
    rel_err = np.abs(exact / lead - 1.0)
    assert np.all(rel_err > 0.0), rel_err  # error must be measurable at these scales
    assert np.max(np.abs(rel_err - sigma**2 * taus / 24.0)) < 1e-4, rel_err
    slope = checks.loglog_slope(taus, rel_err)
    assert abs(slope - 1.0) < 0.05, (sigma, slope)
    print(f"ok  eq_atm_black order numeric: log-log slope {slope:.4f} ~ 1 (sigma={sigma:.2f})")


if __name__ == "__main__":
    test_loglog_slope_recovers_power_laws()
    test_eq_atm_black_order_symbolic()
    test_eq_atm_black_order_numeric()
