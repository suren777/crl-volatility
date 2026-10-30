"""Self-checks for heston.py: run with `uv run pytest`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import heston  # noqa: E402
import numpy as np  # noqa: E402
import shifted_spx  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.integrate import quad  # noqa: E402

F: float = 100.0


def test_black_scholes_limit() -> None:
    """vol-of-vol -> 0 with v0 = theta, rho = 0: flat variance v0 -> Black-Scholes.

    rho must be 0 here: for rho != 0 the gap is a first-order rho*sigma effect
    (checked separately in test_bs_limit_gap_linear_in_sigma), so a tight
    threshold would be violated at this sigma.
    """
    v0 = 0.04
    for tau in (0.1, 1.0):
        for m in (0.8, 0.9, 1.0, 1.1, 1.2):
            K = F * m
            px = heston.heston_call(F, K, tau, v0, 2.0, v0, 1e-4, 0.0)
            bs = shifted_spx.black_px(F, K, tau, np.sqrt(v0))
            assert abs(px - bs) < 1e-4 * max(bs, 1.0), (tau, m, px, bs)


def test_bs_limit_gap_linear_in_sigma() -> None:
    """For rho != 0 the BS gap is a first-order rho*sigma effect, so ~ linear in sigma."""
    v0 = 0.04
    K = F * 1.2
    for tau in (0.1, 1.0):
        bs = shifted_spx.black_px(F, K, tau, np.sqrt(v0))
        for rho in (-0.8, 0.6):
            g1 = heston.heston_call(F, K, tau, v0, 2.0, v0, 1e-4, rho) - bs
            g4 = heston.heston_call(F, K, tau, v0, 2.0, v0, 4e-4, rho) - bs
            # 4x sigma -> ~4x gap (allow a little slack for the O(sigma^2) tail)
            assert 2.5 * abs(g1) < abs(g4) < 5.5 * abs(g1), (tau, rho, g1, g4)


def _mc_call(
    tau: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    K: NDArray[np.float64],
    n_paths: int,
    n_steps: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Full-truncation Euler MC of the Heston forward; returns (prices, standard errors)."""
    rng = np.random.default_rng(seed)
    dt = tau / n_steps
    x = np.zeros(n_paths)
    v = np.full(n_paths, v0)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        wv = rho * z1 + np.sqrt(1.0 - rho**2) * z2
        vp = np.maximum(v, 0.0)
        x += -0.5 * vp * dt + np.sqrt(vp) * np.sqrt(dt) * z1
        v += kappa * (theta - vp) * dt + sigma * np.sqrt(vp) * np.sqrt(dt) * wv
    FT = F * np.exp(x)
    px = np.empty(K.size)
    se = np.empty(K.size)
    for i, kk in enumerate(K):
        payoff = np.maximum(FT - kk, 0.0)
        px[i] = float(payoff.mean())
        se[i] = float(payoff.std(ddof=1) / np.sqrt(n_paths))
    return px, se


def test_monte_carlo() -> None:
    """Quad pricer within 3 standard errors of a seeded full-truncation Euler MC."""
    v0, kappa, theta, sigma, rho = 0.04, 1.5, 0.06, 0.5, -0.7
    tau = 1.0
    K = np.array([80.0, 100.0, 120.0])
    px, se = _mc_call(tau, v0, kappa, theta, sigma, rho, K, 1_000_000, 250, 42)
    for i, kk in enumerate(K):
        ref = heston.heston_call(F, float(kk), tau, v0, kappa, theta, sigma, rho)
        assert abs(px[i] - ref) < 3.0 * se[i], (kk, px[i], ref, se[i])


def test_grid_matches_quad() -> None:
    """Fixed Gauss-Legendre path (used in calibration) matches the quad reference."""
    v0, kappa, theta, sigma, rho = 0.04, 1.5, 0.06, 0.5, -0.7
    for tau in (0.1, 1.0):
        K = np.array([80.0, 100.0, 120.0])
        pg = heston.heston_call_vec(F, K, tau, v0, kappa, theta, sigma, rho)
        for i, kk in enumerate(K):
            pq = heston.heston_call(F, float(kk), tau, v0, kappa, theta, sigma, rho)
            assert abs(pq - pg[i]) < 1e-6 * max(pq, 1.0), (tau, kk, pq, pg[i])


# Fitted 2020-03-31 and 2020-02-28 parameters (v0, kappa, theta, sigma, rho): large vol-of-vol,
# where the original Heston (1993) form of the CF crosses the branch cut of the complex log.
STRESSED = (
    (0.2411731, 1.5822198, 0.1519294, 1.9864009, -0.8516627),
    (0.2772739, 8.6469402, 0.0222408, 3.5189039, -0.6738301),
)


def test_monte_carlo_stressed() -> None:
    """Quad pricer within 3 SE of full-truncation Euler MC at large vol-of-vol, one year out."""
    K = np.array([80.0, 100.0, 120.0])
    px, se = _mc_call(0.79, *STRESSED[0], K, 400_000, 400, 7)
    for i, kk in enumerate(K):
        ref = heston.heston_call(F, float(kk), 0.79, *STRESSED[0])
        assert abs(px[i] - ref) < 3.0 * se[i], (kk, px[i], ref, se[i])


def test_cf_continuous_stressed() -> None:
    """No jumps in the CF along the integration path (the branch-cut failure mode)."""
    u = np.asarray(np.linspace(0.0, heston.UMAX, 20_001) - 0.5j, dtype=np.complex128)
    for p in STRESSED:
        for tau in (0.1, 0.9, 2.0):
            phi = np.asarray(heston.heston_cf(u, tau, *p))
            assert np.all(np.isfinite(phi))
            step = np.abs(np.diff(phi))
            assert step.max() < 0.05 * np.abs(phi[:-1]).max() + 1e-3, (p, tau, step.max())


def test_prices_within_bounds_stressed() -> None:
    """Calls lie in [max(F - K, 0), F] and the grid matches quad at large vol-of-vol."""
    K = F * np.array([0.6, 0.8, 1.0, 1.2, 1.5])
    for p in STRESSED:
        for tau in (0.1, 0.9, 2.0):
            pg = heston.heston_call_vec(F, K, tau, *p)
            assert np.all(pg >= np.maximum(F - K, 0.0) - 1e-8) and np.all(pg <= F), (p, tau, pg)
            for i, kk in enumerate(K):
                pq = heston.heston_call(F, float(kk), tau, *p)
                assert abs(pq - pg[i]) < 1e-6 * max(pq, 1.0), (p, tau, kk, pq, pg[i])


# Fitted 2017-06-30 parameters: low v0, so at three weeks |phi(u - i/2)| is still 0.2 at u = 200
# and a cutoff there leaves a strike-oscillating error (negative OTM call prices).
LOW_VAR = (0.0063683, 5.4475942, 0.0307617, 1.2603156, -0.7576482)


def _call_long(kk: float, tau: float, p: tuple[float, ...]) -> float:
    k = float(np.log(F / kk))

    def integrand(u: float) -> float:
        phi = heston.heston_cf(u - 0.5j, tau, *p)
        return float(np.real(np.exp(1j * u * k) * phi)) / (u * u + 0.25)

    edges = np.arange(0.0, 10_001.0, 50.0)
    val = sum(
        quad(integrand, a, b, limit=200, epsabs=1e-15)[0]
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    )
    return float(F - np.sqrt(F * kk) / np.pi * val)


def test_low_variance_short_expiry() -> None:
    """Both pricers integrate until the CF has decayed: positive, decreasing, and equal to a
    cutoff-10k reference."""
    tau = 21 / 365
    K = F * np.array([0.95, 1.0, 1.03, 1.05, 1.07, 1.1])
    pg = heston.heston_call_vec(F, K, tau, *LOW_VAR)
    assert np.all(pg > 0.0) and np.all(np.diff(pg) < 0.0), pg
    for i, kk in enumerate(K):
        ref = _call_long(float(kk), tau, LOW_VAR)
        pq = heston.heston_call(F, float(kk), tau, *LOW_VAR)
        assert abs(pg[i] - ref) < 1e-9 * F and abs(pq - ref) < 1e-9 * F, (kk, pg[i], pq, ref)
