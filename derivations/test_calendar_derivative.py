"""eq_crl_calendar_derivative: dC/du = T1 + T2 (calendar_derivative.md).

Symbolic: the share-measure conditional mean E*[D | Z] = h(Z). Numeric: T1 + T2 against a
central finite difference of msv.crl_call in u = sqrt(t) (independent code path:
Gauss-Hermite on the Black mixture). Limit: T2 = 0 at rho = 0.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import sympy as sp  # type: ignore[import-untyped]  # noqa: E402
from calendar_derivative import parts  # noqa: E402

import msv  # noqa: E402

A2 = 0.04


def test_eq_crl_calendar_derivative_symbolic() -> None:
    uu, z, sz, x = sp.symbols("u z sqrtzeta x", real=True)
    rho = sp.Symbol("rho", real=True)
    var = sp.Symbol("v", positive=True)
    dens = sp.exp(-((x - rho * z) ** 2) / (2 * var) + uu * sz * x)
    num = sp.integrate(x * dens, (x, -sp.oo, sp.oo), conds="none")
    den = sp.integrate(dens, (x, -sp.oo, sp.oo), conds="none")
    e_u = sp.simplify(num / den)
    h = (sz * e_u - uu * sz**2).subs(var, 1 - rho**2)
    assert sp.simplify(h - (rho * sz * z - rho**2 * uu * sz**2)) == 0


def _call(x: float, u: float, k: float, rho: float) -> float:
    px = msv.crl_call(1.0, np.exp(x), 1.0, u * u / A2, 0.0, 0.0, A2, 1.0, k, rho, n_nodes=200)
    return float(np.asarray(px).ravel()[0])


@pytest.mark.parametrize(
    "x,u,k,rho",
    [
        (0.0, 0.2, 1.0, -0.5),
        (0.26, 0.3, 2.2, -0.49),
        (-0.36, 0.15, 2.2, -0.9),
        (0.4, 0.4, 0.5, -0.3),
    ],
)
def test_eq_crl_calendar_derivative(x: float, u: float, k: float, rho: float) -> None:
    t1, t2 = parts(x, u, k, rho)
    e = 1e-5
    fd = (_call(x, u + e, k, rho) - _call(x, u - e, k, rho)) / (2 * e)
    assert t1 >= 0
    assert abs(t1 + t2 - fd) < 1e-6 * max(1.0, abs(fd))


def test_eq_crl_calendar_derivative_rho0() -> None:
    t1, t2 = parts(0.2, 0.3, 1.5, 0.0)
    assert abs(t2) < 1e-12 and t1 > 0
