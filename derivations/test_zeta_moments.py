"""eq_zeta_moments: central moments of the lognormal level zeta = exp(s Z - s^2/2).

E[zeta] = 1, Var zeta = k^2, E[(zeta-1)^3] = k^4 (3 + k^2),
E[(zeta-1)^4] = k^4 {(1+k^2)^4 + 2(1+k^2)^3 + 3(1+k^2)^2 - 3}.
Symbolic: raw moments E[zeta^n] = exp(n(n-1) s^2/2) expanded by the binomial theorem.
Numeric: high-order Gauss-Hermite quadrature over Z.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import checks  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sp  # type: ignore[import-untyped]  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from numpy.polynomial.hermite_e import hermegauss  # noqa: E402

s2, u = sp.symbols("s2 u", positive=True)


def raw(n: int) -> sp.Expr:
    """E[zeta^n] = exp(n(n-1) s^2/2), with u = exp(s^2) = 1 + k^2."""
    return sp.exp(n * (n - 1) * s2 / 2)


def test_eq_zeta_moments_symbolic() -> None:
    """Binomial expansion of (zeta - 1)^3, (zeta-1)^4 from the raw lognormal moments."""
    m1, m2, m3, m4 = (sp.simplify(raw(n).subs(s2, sp.log(u))) for n in range(1, 5))
    assert sp.simplify(m1 - 1) == 0 and sp.simplify(m2 - u) == 0  # mean 1, E[z^2] = 1+k^2
    third = sp.simplify(m3 - 3 * m2 + 3 * m1 - 1 - (u - 1) ** 2 * (u + 2))
    fourth = sp.simplify(
        m4 - 4 * m3 + 6 * m2 - 4 * m1 + 1 - (u - 1) ** 2 * (u**4 + 2 * u**3 + 3 * u**2 - 3)
    )
    assert third == 0 and fourth == 0, (third, fourth)
    print(
        "ok  eq_zeta_moments symbolic: mu3 = k^4(3+k^2),"
        " mu4 = k^4{(1+k^2)^4 + 2(1+k^2)^3 + 3(1+k^2)^2 - 3}"
    )


@settings(max_examples=20, deadline=None)
@given(k=st.floats(0.05, 3.0))
def test_eq_zeta_moments_numeric(k: float) -> None:
    """129-node Gauss-Hermite of (zeta-1)^3, (zeta-1)^4 matches the closed forms."""
    x, w = hermegauss(129)
    w = w / np.sqrt(2.0 * np.pi)
    zeta = np.exp(np.sqrt(np.log1p(k * k)) * x - np.log1p(k * k) / 2.0)
    mu3 = float(w @ (zeta - 1.0) ** 3)
    mu4 = float(w @ (zeta - 1.0) ** 4)
    k2 = k * k
    k4 = k2 * k2
    assert checks.relerr(mu3, k4 * (3.0 + k2)) < 1e-9, (k, mu3)
    ref4 = k4 * ((1 + k2) ** 4 + 2 * (1 + k2) ** 3 + 3 * (1 + k2) ** 2 - 3.0)
    assert checks.relerr(mu4, ref4) < 1e-9, (k, mu4)
    # raw-moment helper agrees too: E[zeta^3] = (1+k^2)^3
    assert checks.relerr(checks.lognormal_moment(k, 3), (1 + k2) ** 3) < 1e-12
    print(f"ok  eq_zeta_moments numeric: Gauss-Hermite == closed forms to 1e-9 (k={k:.3f})")


if __name__ == "__main__":
    test_eq_zeta_moments_symbolic()
    test_eq_zeta_moments_numeric()
