"""Checks for calendar_largeL.md: the structural lemmas for E = {q < 0} at rho < 0.

Symbolic: the derivatives of q used in the curvature bound. Numeric: E is an interval, E lies left
of -c0 - mu s/r, and V'' >= 1/sigma off E. No bound on R' is claimed here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import sympy as sp  # type: ignore[import-untyped]  # noqa: E402
from calendar_largeL import check  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


def test_q_derivatives_symbolic() -> None:
    v, c0, s, mu, r = sp.symbols("v c0 s mu r", real=True)
    q = mu * sp.cosh(v / 2) + r * (v + c0) / s
    q1 = mu * sp.sinh(v / 2) / 2 + r / s
    q2 = mu * sp.cosh(v / 2) / 4
    assert sp.simplify(sp.diff(q, v) - q1) == 0
    assert sp.simplify(sp.diff(q, v, 2) - q2) == 0


@settings(max_examples=25, deadline=None)
@given(
    u=st.sampled_from([0.1, 1.0, 5.0]),
    k=st.sampled_from([0.05, 0.3, 1.0, 2.5]),
    r=st.sampled_from([0.05, 0.5, 0.99]),
    lam=st.sampled_from([0.5, 10.0, 300.0]),
)
def test_lemmas_hold(u: float, k: float, r: float, lam: float) -> None:
    res = check(lam, u, k, r)
    assert res["E_interval"]
    assert res["E_empty"] or res["E_left_of_t0"]
    assert res["curv_low"] >= -1e-9
    assert res["fd_err"] < 1e-5


def test_E_empty_at_large_lambda_for_moderate_rho() -> None:
    res = check(300.0, 1.0, 1.0, 0.5)
    assert res["E_empty"]
    assert np.isfinite(res["t0"])
