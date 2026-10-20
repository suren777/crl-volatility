"""Self-checks for heston.py: run with `uv run pytest`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import heston  # noqa: E402
import numpy as np  # noqa: E402
import shifted_spx  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

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
