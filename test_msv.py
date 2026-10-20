"""Self-checks for msv.py: run with `uv run pytest` or `uv run python test_msv.py`."""

import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import brentq
from scipy.stats import norm

import msv

S: float = 100.0
r: float = 0.02
tau: float = 0.75
theta: msv.Params = (0.05, 0.03, 0.02, 1.2, 0.35)  # a0, a1, a2, lam, k
a0, a1, a2, lam, k = theta


def relerr(a: float | npt.NDArray[np.float64], b: float | npt.NDArray[np.float64]) -> float:
    return float(np.abs(a - b) / max(float(np.abs(b)), 1e-12))


def test_bs_derivatives() -> None:
    """Closed-form variance derivatives of BS vs central finite differences."""
    q = 0.04

    def f(v: float) -> msv.FloatArray:
        return msv.bs_call(S, 100.0, r, tau, v)

    c1, c2, c3, c4 = msv.bs_call_variance_derivs(S, 100.0, r, tau, q)
    h = 1e-4
    fd1 = (f(q + h) - f(q - h)) / (2 * h)
    fd2 = (f(q + h) - 2 * f(q) + f(q - h)) / h**2
    assert relerr(c1, fd1) < 1e-6, (c1, fd1)
    assert relerr(c2, fd2) < 1e-3, (c2, fd2)  # 2nd-order FD of a smooth fn
    # 3rd/4th vs high-accuracy FD with Richardson-free 5-point stencils
    fd3 = (f(q + 2 * h) - 2 * f(q + h) + 2 * f(q - h) - f(q - 2 * h)) / (2 * h**3)
    fd4 = (f(q + 2 * h) - 4 * f(q + h) + 6 * f(q) - 4 * f(q - h) + f(q - 2 * h)) / h**4
    assert relerr(c3, fd3) < 5e-3, (c3, fd3)
    assert relerr(c4, fd4) < 5e-2, (c4, fd4)
    print("ok  bs derivatives vs finite differences")


def test_moments() -> None:
    """Analytic central moments of zeta*q vs Monte Carlo."""
    rng = np.random.default_rng(1)
    n = 2_000_000
    z = msv.draw_zeta(k, n, rng)
    q = 0.04
    mu2, mu3, mu4 = msv.msv_moments(q, k)
    for analytic, mc, name in [
        (mu2, np.var(z) * q**2, "mu2"),
        (mu3, np.mean((z - 1) ** 3) * q**3, "mu3"),
        (mu4, np.mean((z - 1) ** 4) * q**4, "mu4"),
    ]:
        assert relerr(float(analytic), float(mc)) < 0.02, (name, analytic, mc)
    print("ok  moments vs Monte Carlo")


def test_price_vs_mc() -> None:
    """Taylor-4 MSV price within 0.5% of E[BS(zeta*Q)] Monte Carlo ground truth."""
    for K in (80.0, 100.0, 120.0):
        taylor = msv.msv_call(S, K, r, tau, *theta)
        mc = msv.mc_call(S, K, r, tau, *theta, n=1_000_000, rng=np.random.default_rng(3))
        assert relerr(taylor, mc) < 0.005, (K, taylor, mc)
    print("ok  Taylor-4 price vs Monte Carlo (3 strikes, <0.5%)")


def test_simulate_matches_gh_call() -> None:
    """simulate's exact lognormal increments price a European call within 3 SE of gh_call."""
    for tau_s in (0.5, 1.25):
        exact = float(msv.gh_call(S, 100.0, r, tau_s, *theta))
        disc = float(np.exp(-r * tau_s))
        payoff_sum, var_sum, n = 0.0, 0.0, 0
        for seed in range(5):  # pool seeds: a single 3-SE draw fails ~0.3% of the time
            _, paths = msv.simulate(
                S, r, tau_s, theta, n_paths=2_000_000, n_steps=1, rng=np.random.default_rng(seed)
            )
            payoff = np.maximum(paths[-1] - 100.0, 0.0)
            payoff_sum += float(payoff.sum())
            var_sum += float(payoff.var()) * payoff.size
            n += payoff.size
        mc = disc * payoff_sum / n
        se = disc * np.sqrt(var_sum / n) / np.sqrt(n)
        assert abs(mc - exact) < 3 * se, (tau_s, mc, exact, se)
    print("ok  simulate vs gh_call: within 3 SE at tau=0.5, 1.25 (pooled over 5 seeds)")


def test_g_integrates_to_tq() -> None:
    """g is the instantaneous rate: numeric int_0^T g dt == T*Q(T) (eq_inst_variance)."""
    for tau_s in (0.5, 1.25, 3.0):
        ts = np.linspace(0.0, tau_s, 2001)
        num = float(np.trapezoid(msv.g(ts, a0, a1, a2, lam), ts))
        assert abs(num - tau_s * float(msv.Q(tau_s, a0, a1, a2, lam))) < 1e-8, (tau_s, num)
    print("ok  g integrates to t*Q(t): numeric int_0^T g dt == T*Q(T) at 0.5, 1.25, 3.0")


def test_calibration_recovers_params() -> None:
    """Calibration on synthetic data recovers the true parameters and reprices well."""
    rng = np.random.default_rng(7)
    true = (0.05, 0.03, 0.02, 1.2, 0.35)
    taus = np.repeat(np.linspace(30 / 365, 3.0, 10), 10)
    strikes = np.tile(S * np.linspace(0.8, 1.2, 10), 10)
    base = np.asarray(msv.gh_call(S, strikes, r, taus, *true), dtype=float)
    spread = np.maximum(0.02 * base, 0.05)
    mkt = base + rng.uniform(-1, 1, base.size) * spread / 2
    res = msv.calibrate(strikes, taus, mkt, spread, S, r)
    fitted = res.x
    fit_px = np.asarray(msv.gh_call(S, strikes, r, taus, *fitted), dtype=float)
    assert np.max(np.abs(fit_px - base)) < 0.2, np.max(np.abs(fit_px - base))
    # parameters recovered to within sensible tolerance of the truth; the a0/a1/lam
    # triple has a known flat direction (0002) so the NS shape trades off internally
    # but k (the volatility-of-variance) is well identified.
    assert abs(fitted[4] - true[4]) < 0.1, (fitted, true)
    assert np.allclose(fitted[:3], true[:3], rtol=0.5, atol=0.05), (fitted, true)
    assert 0.5 < fitted[3] < 3.0, fitted
    print(
        f"ok  calibration: max repricing err={np.max(np.abs(fit_px - base)):.4f}, "
        f"params={np.round(fitted, 3)} vs true={true}"
    )


@settings(deadline=3000)
@given(
    a0=st.floats(0.0, 0.2),
    a1=st.floats(0.0, 0.2),
    a2=st.floats(0.0, 0.2),
    lam=st.floats(1e-6, 4.0),
    k=st.floats(1e-4, 1.5),
    K=st.floats(40.0, 250.0),
    tau=st.floats(30 / 365, 3.0),
)
def test_gh_call_no_negative_inside_calibration_bounds(
    a0: float, a1: float, a2: float, lam: float, k: float, K: float, tau: float
) -> None:
    """gh_call is non-negative and below the (discounted) forward for parameters
    inside the calibrate() bounds: no negative prices anywhere the optimiser may go."""
    px = float(msv.gh_call(S, K, r, tau, a0, a1, a2, lam, k))
    assert 0.0 <= px <= S * np.exp(r * tau), (px, a0, a1, a2, lam, k, K, tau)


def test_bs_call_variance_derivs_finite_at_q0() -> None:
    """bs_call_variance_derivs does not divide by zero at q=0 (floored to 1e-14)."""
    c1, c2, c3, c4 = msv.bs_call_variance_derivs(S, 100.0, r, tau, 0.0)
    assert np.all(np.isfinite([c1, c2, c3, c4])), (c1, c2, c3, c4)
    print(f"ok  bs variance derivatives finite at q=0: {c1:.4f} {c2:.3f} {c3:.1f} {c4:.1f}")


def test_gh_quadrature() -> None:
    """Gauss-Hermite pricing: near-exact vs MC everywhere, incl. where Taylor-4 fails."""
    Ks, taus = np.array([80.0, 100.0, 130.0]), np.array([0.25, 0.75, 2.0])
    gh = msv.gh_call(S, Ks, r, taus, *theta)
    assert isinstance(gh, np.ndarray), "array input must return an array"
    assert gh.shape == (3,) and np.all(np.isfinite(gh)), gh
    rng = np.random.default_rng(3)
    for tau in taus:
        for K in Ks:
            mc = msv.mc_call(S, K, r, tau, *theta, n=1_000_000, rng=rng)
            g = float(msv.gh_call(S, K, r, tau, *theta))
            t4 = msv.msv_call(S, K, r, tau, *theta)
            # absolute error vs MC tiny; and no worse than Taylor-4
            assert abs(g - mc) < 0.01, (K, tau, g, mc)
            assert abs(g - mc) <= abs(t4 - mc) + 1e-3, (K, tau, g, mc, t4)
    # extreme strike where Taylor-4 breaks down badly
    mc = msv.mc_call(S, 200.0, r, 0.25, *theta, n=2_000_000, rng=rng)
    g = float(msv.gh_call(S, 200.0, r, 0.25, *theta))
    assert abs(g - mc) < 0.005, (g, mc)
    print(f"ok  GH quadrature vs MC (incl. K=200 tau=0.25: gh={g:.5f} mc={mc:.5f})")


def black_fwd(F: float, K: float, tau: float, sigma: float) -> float:
    """Forward Black call at volatility sigma (reference for shifted-model checks)."""
    v = sigma * sigma * tau
    sd = np.sqrt(v)
    d1 = (np.log(F / K) + v / 2.0) / sd
    return float(F * norm.cdf(d1) - K * norm.cdf(d1 - sd))


def test_gh_call_fwd_matches_gh_call() -> None:
    """gh_call_fwd with beta=0 in forward form (F=S e^{r tau}, D=e^{-r tau}) == gh_call."""
    taus = np.array([0.25, 0.75, 2.0])
    Ks = np.array([80.0, 100.0, 130.0])
    fwd = msv.gh_call_fwd(S * np.exp(r * taus), Ks, np.exp(-r * taus), taus, *theta)
    spot = msv.gh_call(S, Ks, r, taus, *theta)
    assert isinstance(fwd, np.ndarray) and isinstance(spot, np.ndarray)
    assert np.max(np.abs(fwd - spot)) < 1e-10, (fwd, spot)
    print("ok  gh_call_fwd(beta=0) == gh_call to 1e-10")


def test_gh_call_fwd_small_k_is_shifted_black() -> None:
    """With beta>0 and k tiny, gh_call_fwd collapses to the shifted Black price."""
    theta0: msv.Params = (a0, a1, a2, lam, 1e-6)
    beta, D, F = 500.0, 0.97, 4500.0
    taus = np.array([0.25, 0.75])
    Ks = np.array([4200.0, 5500.0])
    px = msv.gh_call_fwd(F, Ks, D, taus, *theta0, beta=beta)
    sig = np.sqrt(msv.Q(taus, a0, a1, a2, lam))
    ref = np.array(
        [
            D * black_fwd(F + beta, K + beta, t, float(s))
            for K, t, s in zip(Ks, taus, sig, strict=True)
        ]
    )
    assert np.allclose(px, ref, rtol=1e-9, atol=1e-10), (px, ref)
    print("ok  gh_call_fwd(k=1e-6, beta>0) == shifted Black")


def test_gh_call_fwd_negative_skew() -> None:
    """With beta>0 the unshifted-Black IV at ln(K/F)=-0.1 exceeds that at +0.1."""
    F, tau, D, beta = 4500.0, 0.25, 1.0, 900.0

    def iv(x: float) -> float:
        K = F * np.exp(x)
        px = msv.gh_call_fwd(F, K, D, tau, *theta, beta=beta)
        price = float(np.asarray(px)[0])
        return float(brentq(lambda s: black_fwd(F, K, tau, s) - price, 1e-6, 5.0, xtol=1e-12))

    iv_lo, iv_hi = iv(-0.1), iv(0.1)
    assert iv_lo > iv_hi, (iv_lo, iv_hi)
    print(f"ok  shifted MSV skew: IV(-0.1)={iv_lo:.4f} > IV(+0.1)={iv_hi:.4f}")


def test_crl_rho0_matches_gh_call_fwd() -> None:
    """crl_call with rho=0 is exactly gh_call_fwd(beta=0) (same k, NS params)."""
    taus = np.array([0.25, 0.75, 2.0])
    Ks = np.array([80.0, 100.0, 130.0])
    fwd = msv.gh_call_fwd(S * np.exp(r * taus), Ks, np.exp(-r * taus), taus, *theta)
    crl = msv.crl_call(S * np.exp(r * taus), Ks, np.exp(-r * taus), taus, a0, a1, a2, lam, k, 0.0)
    assert isinstance(fwd, np.ndarray) and isinstance(crl, np.ndarray)
    assert np.max(np.abs(crl - fwd)) < 1e-10, (crl, fwd)
    print("ok  crl_call(rho=0) == gh_call_fwd(beta=0) to 1e-10")


def test_crl_martingale() -> None:
    """Deep-OTM call recovers the discounted forward: lim K->0 crl_call = D*F (rho=-0.7).

    K = 1e-9*F, not 1e-8*F: the Black payoff floor -K*N(d2) ~ -K puts the
    relative error at K/F identically, so the 1e-8 tolerance is only reachable
    with K/F < 1e-8.
    """
    for t in (0.25, 1.0, 2.0):
        F, D = S * float(np.exp(r * t)), float(np.exp(-r * t))
        px = msv.crl_call(F, 1e-9 * F, D, t, a0, a1, a2, lam, 1.0, -0.7)
        assert relerr(float(px[0]), D * F) < 1e-8, (t, px, D * F)
    print("ok  crl martingale: deep-OTM call == D*F to 1e-8 rel (rho=-0.7)")


def test_crl_vs_monte_carlo() -> None:
    """crl_call within 3 standard errors of an exact simulation (rho=-0.7, k=1).

    MC follows the epic's ln S_T = ln F - ln m + rho*sqrt(V tau) Z - V tau/2
    + sqrt((1-rho^2) V tau) eps: the -V tau/2 term covers BOTH normals, so the
    eps increment carries no compensator of its own (adding -1/2(1-rho^2)V tau
    as well deflates E[S_T] to 0.9875 F and underprices calls by ~5%).
    """
    rng = np.random.default_rng(5)
    n = 2_000_000
    s2 = np.log1p(1.0)
    t = 1.0
    F, D = 100.0, 0.98
    z = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    vt = float(msv.Q(t, a0, a1, a2, lam)) * t * np.exp(np.sqrt(s2) * z - s2 / 2.0)
    m_exact = float(np.mean(np.exp(-0.7 * np.sqrt(vt) * z - 0.5 * 0.7**2 * vt)))
    s_t = (
        F
        * np.exp(-0.7 * np.sqrt(vt) * z - 0.5 * vt - np.log(m_exact))
        * np.exp(np.sqrt((1.0 - 0.7**2) * vt) * eps)
    )
    assert abs(float(s_t.mean()) / F - 1.0) < 4e-3, float(s_t.mean()) / F
    for K in (0.9 * F, F, 1.1 * F):
        payoff = np.maximum(s_t - K, 0.0)
        mc = D * float(payoff.mean())
        se = D * float(payoff.std(ddof=1)) / np.sqrt(n)
        quad = float(msv.crl_call(F, K, D, t, a0, a1, a2, lam, 1.0, -0.7)[0])
        assert abs(mc - quad) <= 3 * se, (t, K, quad, mc, se)
    print("ok  crl_call vs 2e6-path MC within 3 SE (3 strikes, tau=1, rho=-0.7, k=1)")


def test_crl_no_arbitrage_in_strike() -> None:
    """Call prices decrease and are convex in K (positive mixture of convex Blacks)."""
    Ks = np.linspace(60.0, 160.0, 21)
    for t in (0.25, 1.0, 2.0):
        F, D = S * float(np.exp(r * t)), float(np.exp(-r * t))
        px = msv.crl_call(F, Ks, D, t, a0, a1, a2, lam, 1.0, -0.7)
        assert isinstance(px, np.ndarray)
        assert np.all(np.diff(px) < 0.0), ("decreasing", t)
        assert np.all(np.diff(px, 2) >= -1e-8), ("convex", t)
    print("ok  strike no-arbitrage: decreasing & convex in K (3 taus, rho=-0.7)")


def test_crl_skew_sign() -> None:
    """rho=-0.7: IV at ln(K/F)=-0.1 exceeds +0.1; rho=+0.7 the reverse."""
    F, D, t, kc = 100.0, 1.0, 0.5, 1.0

    def iv(x: float, rho: float) -> float:
        K = F * float(np.exp(x))
        price = float(msv.crl_call(F, K, D, t, a0, a1, a2, lam, kc, rho)[0])
        return float(brentq(lambda s: black_fwd(F, K, t, s) - price, 1e-6, 5.0, xtol=1e-12))

    lo, hi = iv(-0.1, -0.7), iv(0.1, -0.7)
    assert lo > hi, (lo, hi)
    lo_p, hi_p = iv(-0.1, 0.7), iv(0.1, 0.7)
    assert lo_p < hi_p, (lo_p, hi_p)
    print(f"ok  crl skew sign: rho=-0.7 IV(-0.1)={lo:.4f}>IV(+0.1)={hi:.4f}; rho=+0.7 reversed")


def test_crl_call_pairing_semantics() -> None:
    """crl_call pairs arrays elementwise (option i): scalars broadcast, no K x tau grid."""
    ks = np.array([90.0, 100.0, 110.0])
    ts = np.array([0.5, 0.75, 1.0])
    F, D = S * np.exp(r * ts), np.exp(-r * ts)
    paired = msv.crl_call(F, ks, D, ts, a0, a1, a2, lam, k, -0.7)
    singles = np.array(
        [msv.crl_call(F[i], ks[i], D[i], ts[i], a0, a1, a2, lam, k, -0.7) for i in range(3)]
    ).ravel()
    assert np.allclose(paired, singles, atol=1e-10), (paired, singles)
    try:
        msv.crl_call(F, ks[:2], D, ts, a0, a1, a2, lam, k, -0.7)
        raised = False
    except ValueError:
        raised = True
    assert raised, "mismatched array lengths must raise"
    print("ok  crl_call pairing: elementwise (scalars broadcast), mismatched lengths raise")


def test_greeks_and_exotics() -> None:
    """Greeks are finite/sane; barrier < European; Asian runs and prices > 0."""
    delta, gamma = msv.greeks(S, 100.0, r, tau, theta)
    assert 0 < delta < 1 and 0 < gamma < 0.2, (delta, gamma)

    rng = np.random.default_rng(11)
    euro = msv.price_exotic(
        lambda t, p: np.maximum(p[-1] - 100, 0), S, r, tau, theta, n_paths=20_000, rng=rng
    )
    barrier = msv.price_exotic(
        lambda t, p: np.where(p.min(axis=0) < 80.0, 0.0, np.maximum(p[-1] - 100, 0)),
        S,
        r,
        tau,
        theta,
        n_paths=20_000,
        rng=rng,
    )
    asian = msv.price_exotic(
        lambda t, p: np.maximum(p[-1] - p.mean(axis=0), 0),
        S,
        r,
        tau,
        theta,
        n_paths=20_000,
        rng=rng,
    )
    assert barrier < euro, (barrier, euro)
    assert 0 < asian < euro, (asian, euro)
    print(
        f"ok  greeks d={delta:.3f} g={gamma:.4f}; exotic prices "
        f"euro={euro:.2f} barrier={barrier:.2f} asian={asian:.2f}"
    )


if __name__ == "__main__":
    test_bs_derivatives()
    test_moments()
    test_price_vs_mc()
    test_simulate_matches_gh_call()
    test_g_integrates_to_tq()
    test_gh_quadrature()
    test_gh_call_fwd_matches_gh_call()
    test_gh_call_fwd_small_k_is_shifted_black()
    test_gh_call_fwd_negative_skew()
    test_crl_rho0_matches_gh_call_fwd()
    test_crl_martingale()
    test_crl_vs_monte_carlo()
    test_crl_no_arbitrage_in_strike()
    test_crl_skew_sign()
    test_crl_call_pairing_semantics()
    test_calibration_recovers_params()
    test_bs_call_variance_derivs_finite_at_q0()
    test_greeks_and_exotics()
    print("ALL CHECKS PASSED")
