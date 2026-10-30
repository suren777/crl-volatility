"""Heston call pricer: characteristic function + Lewis (2000) single integral.

Under the T-forward measure the forward F follows dF = sqrt(v) F dW1 with
dv = kappa (theta - v) dt + sigma sqrt(v) dW2, corr(dW1, dW2) = rho, so the
forward log-return X_T = ln(F_T/F) is Heston with zero drift. Its
characteristic function uses the "little Heston trap" parametrization
(Albrecher, Mayer, Schoutens, Tistaert 2007, Wilmott; u_2 = -1/2, b = kappa),
for which the principal complex sqrt and log are correct without branch
switching. The undiscounted (forward) call price is the Lewis single integral
  C = F - sqrt(F K)/pi * int_0^U Re[ e^{i u k} phi(u - i/2) ] du / (u^2 + 1/4),
computed either by adaptive scipy quad (`heston_call`, the reference pricer) or
on a fixed Gauss-Legendre grid (`heston_call_vec`, same integral, fast enough
to calibrate on ~1.5k quotes; agreement is tested in test_heston.py). The cutoff U is
UMAX unless |phi(u - i/2)| is still above CF_TOL there (low v0 at short tau), in which case
it is the first point of a geometric grid where it is below; beyond UMAX the grid pricer
adds composite Gauss-Legendre panels of width PANEL.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.integrate import quad  # noqa: E402

UMAX = 200.0
NNODES = 400
CF_TOL = 1e-8
U_CAP = 1e5
PANEL = 8.0

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(NNODES)
U_GRID: NDArray[np.float64] = 0.5 * UMAX * (_NODES + 1.0)
_WQ: NDArray[np.float64] = 0.5 * UMAX * _WEIGHTS / (U_GRID**2 + 0.25)
_PX, _PW = np.polynomial.legendre.leggauss(12)

ComplexArg = complex | NDArray[np.complex128]


def heston_cf(
    z: ComplexArg, tau: float, v0: float, kappa: float, theta: float, sigma: float, rho: float
) -> ComplexArg:
    """Heston CF of the forward log-return, E[exp(i z X_T)] (little Heston trap)."""
    a = kappa * theta
    b = kappa
    d = np.sqrt((rho * sigma * 1j * z - b) ** 2 + sigma**2 * (z * z + 1j * z))
    e = b - rho * sigma * 1j * z
    g = (e - d) / (e + d)
    ed = np.exp(-d * tau)
    c = (e - d) / sigma**2 * (1.0 - ed) / (1.0 - g * ed)
    return np.exp(
        c * v0 + (a / sigma**2) * ((e - d) * tau - 2.0 * np.log((1.0 - g * ed) / (1.0 - g)))
    )


def cutoff(tau: float, v0: float, kappa: float, theta: float, sigma: float, rho: float) -> float:
    """UMAX, or the first u of a geometric grid above it where |phi(u - i/2)| < CF_TOL."""
    u = np.geomspace(UMAX, U_CAP, 200)
    small = np.abs(
        heston_cf(np.asarray(u - 0.5j, np.complex128), tau, v0, kappa, theta, sigma, rho)
    )
    small_idx = np.flatnonzero(small < CF_TOL)
    return float(u[small_idx[0]]) if small_idx.size else U_CAP


def heston_call(
    F: float, K: float, tau: float, v0: float, kappa: float, theta: float, sigma: float, rho: float
) -> float:
    """Undiscounted Heston call on the forward F (multiply by D for a spot price)."""
    k = float(np.log(F / K))

    def integrand(u: float) -> float:
        phi = heston_cf(u - 0.5j, tau, v0, kappa, theta, sigma, rho)
        return float(np.real(np.exp(1j * u * k) * phi)) / (u * u + 0.25)

    umax = cutoff(tau, v0, kappa, theta, sigma, rho)
    edges = np.append(np.arange(0.0, umax, UMAX), umax)
    val = sum(
        quad(integrand, a, b, limit=200, epsabs=1e-12, epsrel=1e-10)[0]
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    )
    return float(F - np.sqrt(F * K) / np.pi * float(val))


def heston_call_vec(
    F: float,
    K: NDArray[np.float64],
    tau: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> NDArray[np.float64]:
    """heston_call on a strike array: one CF evaluation per expiry, then matmuls."""
    k = np.log(F / K)[:, None]
    umax = cutoff(tau, v0, kappa, theta, sigma, rho)
    u, wq = U_GRID, _WQ
    if umax > UMAX:
        left = np.arange(UMAX, umax, PANEL)
        tail = (left[:, None] + 0.5 * PANEL * (_PX + 1.0)).ravel()
        u = np.concatenate([U_GRID, tail])
        wq = np.concatenate([_WQ, np.tile(0.5 * PANEL * _PW, left.size) / (tail**2 + 0.25)])
    z = np.asarray(u - 0.5j, dtype=np.complex128)
    phi = heston_cf(z, tau, v0, kappa, theta, sigma, rho)
    re = np.asarray(np.real(phi), dtype=np.float64)
    im = np.asarray(np.imag(phi), dtype=np.float64)
    integ = np.cos(k * u) @ (re * wq) - np.sin(k * u) @ (im * wq)
    px = np.asarray(F - np.sqrt(F * K) / np.pi * integ, dtype=np.float64)
    return np.asarray(px, dtype=np.float64)
