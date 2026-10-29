"""MSV: high-order Moments-based Stochastic Volatility model.

Implements Date & Islyaev (2014), "A fast calibrating volatility model for
option pricing" (Brunel preprint). Variance is a Nelson-Siegel term structure
Q(tau) whose level is scaled by a unit-mean lognormal random variable
zeta ~ LN(1, k^2)  (eq 13-14); European options are priced by a 4th-order
Taylor expansion of Black-Scholes in the mean variance (eq 15-17), or exactly
by Gauss-Hermite quadrature (gh_call).
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import OptimizeResult, least_squares
from scipy.stats import norm

FloatArray = npt.NDArray[np.float64]
Params = tuple[float, float, float, float, float]  # a0, a1, a2, lam, k


def draw_zeta(k: float, n: int, rng: np.random.Generator | None = None) -> FloatArray:
    """zeta ~ lognormal with E=1, Var=k^2 (paper section 3, k = sqrt Var(zeta)).

    Log-variance is ln(1+k^2), not k^2; this is what makes the paper's moment
    formulas (11)-(12) exact, e.g. E[(zeta-1)^3] = (1+k^2)^3 - 3(1+k^2) + 2
    = k^4 (3 + k^2).
    """
    rng = rng or np.random.default_rng()
    s2 = np.log1p(k * k)
    zeta = np.exp(np.sqrt(s2) * rng.standard_normal(n) - s2 / 2.0)
    return np.asarray(zeta, dtype=np.float64)


def bs_call(
    S: float, K: npt.ArrayLike, r: float, tau: npt.ArrayLike, V: npt.ArrayLike
) -> FloatArray:
    """Black-Scholes call. V is the *variance rate* (sigma^2), scalar or array."""
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    V = np.asarray(np.maximum(V, 1e-14), dtype=float)
    w = np.sqrt(V * tau)
    d1 = (np.log(S / K) + (r + V / 2.0) * tau) / w
    d2 = d1 - w
    return np.asarray(S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2), dtype=float)


def Q(tau: npt.ArrayLike, a0: float, a1: float, a2: float, lam: float) -> FloatArray:
    """Mean variance E[V_tau] = Q(tau), the Nelson-Siegel curve (eq 14).

    a0, a1, a2 are the squared sigma-hat parameters; lam > 0.
    """
    tau = np.asarray(np.maximum(tau, 1e-12), dtype=np.float64)  # tau=0 -> limit a0+a2
    x = np.exp(-lam * tau)
    f = (1.0 - x) / (lam * tau)
    return np.asarray(a2 + a0 * f + a1 * (f - x), dtype=np.float64)


def g(t: npt.ArrayLike, a0: float, a1: float, a2: float, lam: float) -> FloatArray:
    """Instantaneous variance rate g(t) = d(t Q(t))/dt = a2 + (a0 + a1 lam t) e^{-lam t}.

    So that int_0^T g = T*Q(T) (eq_inst_variance in derivations); this is the
    model's instantaneous variance (eq 13), not the average Q (eq 14).
    """
    t = np.asarray(t, dtype=np.float64)
    return np.asarray(a2 + (a0 + a1 * lam * t) * np.exp(-lam * t), dtype=np.float64)


def bs_call_variance_derivs(
    S: float, K: npt.ArrayLike, r: float, tau: npt.ArrayLike, q: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """First four derivatives of the BS call wrt variance V, evaluated at V=q.

    Closed forms (d1, d2, sigma, nu = K e^{-r tau} phi(d2) sqrt(tau), h = d1 d2):
        C1 = nu / (2 sigma)
        C2 = nu (h - 1) / (4 sigma^3)
        C3 = nu (h^2 - 6h + 3 - sigma^2 tau) / (8 sigma^5)
        C4 = nu (h^3 - 12h^2 + 30h - 15 + sigma^2 tau (6 - 3h)) / (16 sigma^7)
    (paper eq 16, restated for expansion in variance; verified vs finite
    differences in test_msv.py).
    """
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    q = np.asarray(np.maximum(q, 1e-14), dtype=float)
    sigma = np.sqrt(q)
    w = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r + q / 2.0) * tau) / w
    d2 = d1 - w
    h = d1 * d2
    pdf_d2 = np.asarray(norm.pdf(d2), dtype=float)
    nu = K * np.exp(-r * tau) * pdf_d2 * np.sqrt(tau)
    s2, s5, s7 = sigma**2, sigma**5, sigma**7
    c1 = nu / (2.0 * sigma)
    c2 = nu * (h - 1.0) / (4.0 * sigma**3)
    c3 = nu * (h * h - 6.0 * h + 3.0 - s2 * tau) / (8.0 * s5)
    c4 = nu * (h**3 - 12.0 * h * h + 30.0 * h - 15.0 + s2 * tau * (6.0 - 3.0 * h)) / (16.0 * s7)
    return c1, c2, c3, c4


def msv_moments(q: npt.ArrayLike, k: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Central moments 2..4 of V_tau = zeta*q (eq 17)."""
    q = np.asarray(q, dtype=float)
    k2 = k * k
    mu2 = k2 * q**2
    mu3 = k2**2 * (3.0 + k2) * q**3
    k4 = k2**2
    mu4 = k4 * ((1.0 + k2) ** 4 + 2.0 * (1.0 + k2) ** 3 + 3.0 * (1.0 + k2) ** 2 - 3.0) * q**4
    return mu2, mu3, mu4


def msv_call(
    S: float,
    K: npt.ArrayLike,
    r: float,
    tau: npt.ArrayLike,
    a0: float,
    a1: float,
    a2: float,
    lam: float,
    k: float,
    order: int = 4,
) -> FloatArray:
    """Approximate European call under MSV, Taylor order 2..4 (eq 15)."""
    q = Q(tau, a0, a1, a2, lam)
    price = bs_call(S, K, r, tau, q)
    if k > 0 and order >= 2:
        _, c2, c3, c4 = bs_call_variance_derivs(S, K, r, tau, q)
        mu2, mu3, mu4 = msv_moments(q, k)
        price = price + c2 * mu2 / 2.0
        if order >= 3:
            price = price + c3 * mu3 / 6.0
        if order >= 4:
            price = price + c4 * mu4 / 24.0
    return price


def msv_put(
    S: float,
    K: npt.ArrayLike,
    r: float,
    tau: npt.ArrayLike,
    a0: float,
    a1: float,
    a2: float,
    lam: float,
    k: float,
    order: int = 4,
) -> FloatArray:
    """European put via put-call parity."""
    K_a = np.asarray(K, dtype=np.float64)
    tau_a = np.asarray(tau, dtype=np.float64)
    return msv_call(S, K_a, r, tau_a, a0, a1, a2, lam, k, order) + K_a * np.exp(-r * tau_a) - S


def gh_call(
    S: float,
    K: npt.ArrayLike,
    r: float,
    tau: npt.ArrayLike,
    a0: float,
    a1: float,
    a2: float,
    lam: float,
    k: float,
    n_nodes: int = 32,
) -> float | FloatArray:
    """Exact within-model European call: E[BS(zeta*Q)] by Gauss-Hermite quadrature.

    Replaces the Taylor expansion (eq 15) outright: same inputs, no truncation
    error, and being a lognormal mixture of BS prices it is arbitrage-free by
    construction. Still ~1000x faster to calibrate than Heston/Bates.
    """
    scalar = bool(np.isscalar(K) and np.isscalar(tau))
    K_a = np.atleast_1d(np.asarray(K, dtype=float))
    tau_a = np.atleast_1d(np.asarray(tau, dtype=float))
    q = Q(tau_a, a0, a1, a2, lam)
    s2 = np.log1p(k * k)
    x, w = hermegauss(n_nodes)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    zeta = np.exp(np.sqrt(s2) * x - s2 / 2.0)
    V = zeta[:, None] * q[None, :]
    px = np.asarray(
        (w / np.sqrt(2.0 * np.pi)) @ bs_call(S, K_a[None, :], r, tau_a[None, :], V),
        dtype=float,
    )
    return float(px[0]) if scalar else px


def gh_call_fwd(
    F: npt.ArrayLike,
    K: npt.ArrayLike,
    D: npt.ArrayLike,
    tau: npt.ArrayLike,
    a0: float,
    a1: float,
    a2: float,
    lam: float,
    k: float,
    beta: float = 0.0,
    n_nodes: int = 32,
) -> FloatArray:
    """Shifted MSV call in forward form: D * E_zeta[Black(F+beta, K+beta, tau, zeta*Q(tau))].

    beta >= 0 is the displaced-diffusion shift; beta > 0 generates the negative
    implied-vol skew plain MSV lacks. Q here is the variance rate of the
    *shifted* process, so its level differs from plain MSV's — fine.
    """
    if beta < 0:
        raise ValueError("beta must be >= 0")
    F_a = np.atleast_1d(np.asarray(F, dtype=float))
    K_a = np.atleast_1d(np.asarray(K, dtype=float))
    D_a = np.atleast_1d(np.asarray(D, dtype=float))
    tau_a = np.atleast_1d(np.asarray(tau, dtype=float))
    q = Q(tau_a, a0, a1, a2, lam)
    s2 = np.log1p(k * k)
    x, w = hermegauss(n_nodes)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    zeta = np.exp(np.sqrt(s2) * x - s2 / 2.0)
    V = zeta[:, None] * q[None, :]
    Ft = F_a[None, :] + beta
    Kt = K_a[None, :] + beta
    sd = np.sqrt(V * tau_a[None, :])
    d1 = (np.log(Ft / Kt) + V * tau_a[None, :] / 2.0) / sd
    px = np.asarray(Ft * norm.cdf(d1) - Kt * norm.cdf(d1 - sd), dtype=float)
    return np.asarray(D_a * ((w / np.sqrt(2.0 * np.pi)) @ px), dtype=float)


def crl_call(
    F: npt.ArrayLike,
    K: npt.ArrayLike,
    D: npt.ArrayLike,
    tau: npt.ArrayLike,
    a0: float,
    a1: float,
    a2: float,
    lam: float,
    k: float,
    rho: float,
    n_nodes: int | None = None,
) -> FloatArray:
    """Correlated random-level call: D * E_Z[Black(F M(Z)/m, K, tau, (1-rho^2) zeta(Z) Q(tau))].

    One Gaussian factor Z drives both the random level zeta = exp(sZ - s^2/2),
    s^2 = ln(1+k^2), and the return, with correlation rho: the Z-conditional
    forward is F * M(Z) / m with M = exp(rho*sqrt(V*tau)*Z - rho^2*V*tau/2),
    V = zeta*Q(tau), and m = E_quad[M] makes the discrete forward exact.
    rho = 0 is gh_call_fwd(beta=0); rho != 0 prices the smile without a shift.

    Array arguments are paired elementwise (option i uses F[i], K[i], tau[i]), not
    crossed into a K x tau grid: call once per tau to price a surface.
    Reliable for rho <= 0. For rho > 0 and large k the integrand peaks far in the
    Z tail (high level paired with high return) and Gauss-Hermite does not converge.
    n_nodes=None uses 64 nodes, or 256 for rho < -0.9: as rho -> -1 the conditional
    variance (1 - rho^2) V tau vanishes, the integrand develops a kink and convergence
    slows (64 nodes err 0.28 vol pts at rho = -0.99, 256 nodes 0.005).
    """
    if n_nodes is None:
        n_nodes = 256 if rho < -0.9 else 64
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be in (-1, 1)")
    if k <= 0:
        raise ValueError("k must be > 0")
    F_a = np.atleast_1d(np.asarray(F, dtype=float))
    K_a = np.atleast_1d(np.asarray(K, dtype=float))
    D_a = np.atleast_1d(np.asarray(D, dtype=float))
    tau_a = np.atleast_1d(np.asarray(tau, dtype=float))
    q = Q(tau_a, a0, a1, a2, lam)
    s2 = np.log1p(k * k)
    x, w = hermegauss(n_nodes)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    zeta = np.exp(np.sqrt(s2) * x - s2 / 2.0)
    V = zeta[:, None] * q[None, :]
    Vt = V * tau_a[None, :]
    M = np.exp(rho * np.sqrt(Vt) * x[:, None] - 0.5 * rho**2 * Vt)
    m = (w / np.sqrt(2.0 * np.pi)) @ M
    Fj = F_a[None, :] * M / m[None, :]
    v = (1.0 - rho**2) * Vt  # conditional Black total variance
    sd = np.sqrt(v)
    # Fj underflows to 0 at extreme nodes for deep-OTM K; d1 -> -inf then, and
    # the node's Black price -> 0, which is the correct limit.
    with np.errstate(divide="ignore"):
        d1 = (np.log(Fj / K_a[None, :]) + v / 2.0) / sd
    px = np.asarray(Fj * norm.cdf(d1) - K_a[None, :] * norm.cdf(d1 - sd), dtype=float)
    return np.asarray(D_a * ((w / np.sqrt(2.0 * np.pi)) @ px), dtype=float)


def mc_call(
    S: float,
    K: float,
    r: float,
    tau: float,
    a0: float,
    a1: float,
    a2: float,
    lam: float,
    k: float,
    n: int = 200_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte-Carlo ground truth: price = E[ BS( zeta * Q(tau) ) ]  (eq 8).

    For tau beyond the (arbitrarily small) t0, V_tau = zeta * Q(tau) exactly,
    so a single lognormal draw per path suffices.
    """
    rng = rng or np.random.default_rng()
    q = Q(tau, a0, a1, a2, lam)
    zeta = draw_zeta(k, n, rng)
    return float(np.mean(bs_call(S, K, r, tau, zeta * q)))


def _implied_var(S: float, K: float, r: float, tau: float, price: float) -> float:
    """BS variance rate inverting a single call price near ATM (bisection)."""
    lo, hi = 1e-10, 5.0
    flo = float(bs_call(S, K, r, tau, lo)) - price
    fhi = float(bs_call(S, K, r, tau, hi)) - price
    if flo * fhi > 0:
        return 0.02  # price outside bisection range: return a 20% prior
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fmid = float(bs_call(S, K, r, tau, mid)) - price
        if flo * fmid < 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def _default_x0(
    strikes: npt.NDArray[np.float64],
    taus: npt.NDArray[np.float64],
    prices: npt.NDArray[np.float64],
    S: float,
    r: float,
) -> list[float]:
    """Starting point from ATM implied variance by maturity, fitted to Q(tau) at lam=1.

    For each unique maturity the option nearest the forward gives an ATM implied
    variance; (a0, a1, a2) are least-squares fit to Q(tau)=a2+a0 f+a1(f-x),
    lam=1, k=0.3 (a0<=0, common in smiles fitted by NS).
    """
    F = S * np.exp(r * taus)
    ivs: dict[float, float] = {}
    for tau in np.unique(taus):
        mask = taus == tau
        Ks = strikes[mask]
        i = int(np.argmin(np.abs(Ks - F[mask][0])))
        ivs[float(tau)] = _implied_var(S, Ks[i], r, float(tau), float(prices[mask][i]))
    ta = np.array(sorted(ivs), dtype=float)
    va = np.array([ivs[t] for t in ta], dtype=float)
    lam = 1.0
    x = np.exp(-lam * ta)
    f = (1.0 - x) / (lam * ta)
    a = np.linalg.lstsq(np.column_stack([f, f - x, np.ones_like(ta)]), va, rcond=None)[0]
    a = np.clip(a, 1e-6, None)
    return [float(a[0]), float(a[1]), float(a[2]), lam, 0.3]


def calibrate(
    strikes: npt.ArrayLike,
    taus: npt.ArrayLike,
    prices: npt.ArrayLike,
    spreads: npt.ArrayLike,
    S: float,
    r: float,
    k_bounds: tuple[float, float] = (1e-4, 1.5),
    x0: list[float] | None = None,
) -> OptimizeResult:
    """Fit (a0, a1, a2, lam, k) by weighted least squares (section 4.2).

    Prices with the exact Gauss-Hermite pricer gh_call, not the Taylor-4
    msv_call (which goes negative from k ~ 0.8), so the objective is meaningful
    across the whole k_bounds. Minimises sum |C_mkt - C_mod|^2 / spread^4, i.e.
    least_squares residuals (C_mkt - C_mod) / spread^2. Returns scipy
    OptimizeResult with .x and .cost.
    """
    strikes_a = np.asarray(strikes, dtype=float)
    taus_a = np.asarray(taus, dtype=float)
    prices_a = np.asarray(prices, dtype=float)
    spreads_a = np.asarray(spreads, dtype=float)

    def resid(theta: npt.ArrayLike) -> FloatArray:
        a0, a1, a2, lam, k = np.asarray(theta, dtype=np.float64)
        model = gh_call(S, strikes_a, r, taus_a, a0, a1, a2, lam, k)
        return np.asarray((prices_a - model) / spreads_a**2, dtype=np.float64)

    if x0 is None:
        x0 = _default_x0(strikes_a, taus_a, prices_a, S, r)
    lower = [0.0, 0.0, 0.0, 1e-6, k_bounds[0]]
    upper = [np.inf, np.inf, np.inf, np.inf, k_bounds[1]]
    return least_squares(resid, x0, bounds=(lower, upper), xtol=1e-12, ftol=1e-12)


def greeks(
    S: float, K: float, r: float, tau: float, theta: Params, delta_h: float = 1e-3
) -> tuple[float, float]:
    """Delta and gamma by central differences on the MSV price (section 4.3)."""
    a0, a1, a2, lam, k = theta

    def price(s: float) -> float:
        return float(msv_call(s, K, r, tau, a0, a1, a2, lam, k))

    delta = (price(S + delta_h) - price(S - delta_h)) / (2.0 * delta_h)
    gamma = (price(S + delta_h) - 2.0 * price(S) + price(S - delta_h)) / delta_h**2
    return delta, gamma


def simulate(
    S0: float,
    r: float,
    tau: float,
    theta: Params,
    n_paths: int = 10_000,
    n_steps: int = 1_000,
    rng: np.random.Generator | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Exact lognormal increments of dS = r S dt + sqrt(v) S dW, v_t = zeta * g(t) (eq 6, 13).

    Given zeta the variance rate is deterministic, so each step's log-variance
    integral is exact: int_{t_i}^{t_{i+1}} g = (tQ)(t_{i+1}) - (tQ)(t_i) -- no
    discretisation bias. Returns (times, paths) of shape (n_steps+1, n_paths).
    zeta is drawn once per path (v is random only through its level, section 3).
    """
    rng = rng or np.random.default_rng()
    a0, a1, a2, lam, k = theta
    t: FloatArray = np.linspace(0.0, tau, n_steps + 1)
    dt = tau / n_steps
    dv = np.diff(t * Q(t, a0, a1, a2, lam))  # int_{t_i}^{t_{i+1}} g dt, exact
    zeta = draw_zeta(k, n_paths, rng)
    paths: FloatArray = np.empty((n_steps + 1, n_paths))
    paths[0] = S0
    logS: FloatArray = np.log(S0) * np.ones(n_paths)
    for i in range(n_steps):
        incr = (r * dt - zeta * dv[i] / 2.0) + np.sqrt(zeta * dv[i]) * rng.standard_normal(n_paths)
        logS = logS + incr
        paths[i + 1] = np.exp(logS)
    return t, paths


def price_exotic(
    payoff: Callable[[FloatArray, FloatArray], FloatArray],
    S0: float,
    r: float,
    tau: float,
    theta: Params,
    n_paths: int = 10_000,
    n_steps: int = 1_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte-Carlo price of any path-dependent payoff f(times, paths) -> array."""
    t, paths = simulate(S0, r, tau, theta, n_paths, n_steps, rng)
    disc = np.exp(-r * tau)
    return float(disc * np.mean(payoff(t, paths)))


if __name__ == "__main__":
    import time

    # Synthetic "market": true parameters, plausible spreads.
    rng = np.random.default_rng(7)
    true = {"a0": 0.05, "a1": 0.03, "a2": 0.02, "lam": 1.2, "k": 0.35}
    S, r = 100.0, 0.02
    taus = np.repeat(np.linspace(30 / 365, 3.0, 10), 10)
    strikes = np.tile(S * np.linspace(0.8, 1.2, 10), 10)
    base = msv_call(S, strikes, r, taus, true["a0"], true["a1"], true["a2"], true["lam"], true["k"])
    spread = np.maximum(0.02 * base, 0.05)
    mkt = base + rng.uniform(-1, 1, base.size) * spread / 2.0

    t0 = time.perf_counter()
    res = calibrate(strikes, taus, mkt, spread, S, r)
    el = time.perf_counter() - t0
    print(f"calibrated in {el:.3f}s  (paper: ~0.2-1.2s on 2012 hardware)")
    print("true  :", [round(true[p], 4) for p in ("a0", "a1", "a2", "lam", "k")])
    print("fitted:", [round(float(v), 4) for v in res.x])
    print("cost  :", res.cost)
