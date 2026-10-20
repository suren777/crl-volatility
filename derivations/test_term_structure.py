"""eq_inst_variance: instantaneous variance g(t) = d(t Q(t))/dt = a2 + (a0 + a1 lam t) e^{-lam t}.

Variance follows v_t = zeta * g(t) with zeta the static random level; t Q(t) is the
integrated mean variance. Symbolic: differentiate t*Q(t). Numeric: integral_0^T g dt
== T Q(T) by scipy quad on random Nelson-Siegel parameters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import checks  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sp  # type: ignore[import-untyped]  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from scipy.integrate import quad  # noqa: E402

import msv  # noqa: E402

t, lam, a0, a1, a2 = sp.symbols("t lam a0 a1 a2", positive=True)
f = (1 - sp.exp(-lam * t)) / (lam * t)
Q = a2 + a0 * f + a1 * (f - sp.exp(-lam * t))
G_EXACT = a2 + (a0 + a1 * lam * t) * sp.exp(-lam * t)


def test_eq_inst_variance_symbolic() -> None:
    """d(t Q)/dt simplifies to a2 + (a0 + a1 lam t) e^{-lam t}; g(0) = Q(0) = a0 + a2."""
    assert sp.simplify(sp.diff(t * Q, t) - G_EXACT) == 0
    assert sp.simplify(sp.limit(G_EXACT, t, 0) - (a0 + a2)) == 0
    print("ok  eq_inst_variance symbolic: d(tQ)/dt == a2 + (a0 + a1 lam t) e^{-lam t}")


@settings(max_examples=25, deadline=None)
@given(
    a0=st.floats(0.0, 0.2),
    a1=st.floats(0.0, 0.2),
    a2=st.floats(0.0, 0.2),
    lam=st.floats(0.05, 5.0),
    big_t=st.floats(0.01, 10.0),
)
def test_eq_inst_variance_numeric(
    a0: float, a1: float, a2: float, lam: float, big_t: float
) -> None:
    """integral_0^T g dt == T Q(T) via scipy quad on random parameters."""
    g, _ = quad(lambda s: a2 + (a0 + a1 * lam * s) * np.exp(-lam * s), 0.0, big_t, limit=200)
    tq = float(big_t * msv.Q(big_t, a0, a1, a2, lam))
    assert checks.relerr(g, tq) < 1e-9, (a0, a1, a2, lam, big_t, g, tq)
    print(f"ok  eq_inst_variance numeric: quad == T*Q(T) to 1e-9 (last T={big_t:.3f})")


if __name__ == "__main__":
    test_eq_inst_variance_symbolic()
    test_eq_inst_variance_numeric()
