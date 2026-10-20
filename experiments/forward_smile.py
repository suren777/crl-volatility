"""Forward-start smiles under MSV: exact GH prices, implied vols, MC checks.

Forward-start call: strike fixed at T1 to m*S_T1, pays max(S_T2 - m*S_T1, 0) at T2.
Given zeta the variance rate over [T1, T2] is deterministic, zeta * vf with
vf = (T2*Q(T2) - T1*Q(T1)) / (T2 - T1); since E[S_T1 | zeta] = S0*e^{r*T1}
factorizes from the T1-to-T2 return, the time-0 price is exactly

    S0 * E_zeta[ BS(1, m, r, T2-T1, zeta*vf) ],

computed by Gauss-Hermite exactly like msv.gh_call. Run:

    uv run python experiments/forward_smile.py

msv.simulate applies exact lognormal increments of the model (per-step log-variance
integral (tQ)(t_{i+1}) - (tQ)(t_i), epic 0009), so its MC validates the closed
form directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `import msv`

import numpy as np  # noqa: E402
from numpy.polynomial.hermite_e import hermegauss  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.stats import norm  # noqa: E402

import msv  # noqa: E402

S0 = 100.0
R = 0.02
A0, A1, A2, LAM = 0.05, 0.03, 0.02, 1.2
KS = (0.35, 0.6)
T1S = (0.25, 1.0)
DT = 0.25  # T2 - T1
MS = (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2)
N_PATHS, N_STEPS = 200_000, 200


def forward_var_rate(t1: float, t2: float) -> float:
    """Forward variance rate over [t1, t2]: (t2 Q(t2) - t1 Q(t1)) / (t2 - t1)."""
    q1 = float(msv.Q(t1, A0, A1, A2, LAM))
    q2 = float(msv.Q(t2, A0, A1, A2, LAM))
    return (t2 * q2 - t1 * q1) / (t2 - t1)


def forward_call(m: float, t1: float, t2: float, vf: float, k: float) -> float:
    """S0 * E_zeta[BS(1, m, r, t2-t1, zeta*vf)] by Gauss-Hermite (mirrors msv.gh_call)."""
    s2 = np.log1p(k * k)
    x, w = hermegauss(32)
    zeta = np.exp(np.sqrt(s2) * x - s2 / 2.0)
    px = msv.bs_call(1.0, m, R, t2 - t1, zeta * vf)
    return float(S0 * float(w @ px) / np.sqrt(2.0 * np.pi))


def implied_vol(price: float, m: float, tau: float) -> float:
    """Sigma solving BS(1, m, r, tau, sigma^2) = price (price per unit spot)."""

    def f(sigma: float) -> float:
        return float(msv.bs_call(1.0, m, R, tau, sigma * sigma)) - price

    return float(brentq(f, 1e-10, 5.0, xtol=1e-13))


def forward_iv(m: float, t1: float, t2: float, vf: float, k: float) -> float:
    return implied_vol(forward_call(m, t1, t2, vf, k) / S0, m, t2 - t1)


def spot_iv(m: float, tau: float, k: float) -> float:
    """Spot smile at tau with strike K = m * F(tau) (same K/F moneyness as the forward)."""
    fwd = S0 * float(np.exp(R * tau))
    px = float(msv.gh_call(S0, m * fwd, R, tau, A0, A1, A2, LAM, k))
    return implied_vol(px / S0, m * float(np.exp(R * tau)), tau)


def mc_forward_call(m: float, t1: float, t2: float, k: float, seed: int) -> tuple[float, float]:
    """MC price and standard error of the forward-start call via msv.simulate."""
    theta: msv.Params = (A0, A1, A2, LAM, k)
    _, paths = msv.simulate(
        S0, R, t2, theta, n_paths=N_PATHS, n_steps=N_STEPS, rng=np.random.default_rng(seed)
    )
    i1 = round(t1 / t2 * N_STEPS)
    payoff = np.maximum(paths[-1] - m * paths[i1], 0.0)
    disc = float(np.exp(-R * t2))
    px = disc * float(payoff.mean())
    se = disc * float(payoff.std(ddof=1)) / np.sqrt(float(payoff.size))
    return px, se


def simulate_forward_var(t1: float, t2: float, n_steps: int = N_STEPS) -> float:
    """Forward variance rate msv.simulate applies over [t1, t2].

    simulate integrates the instantaneous g(t) exactly per step via the
    antiderivative t*Q(t) (epic 0009), so this equals
    (t2*Q(t2) - t1*Q(t1)) / (t2 - t1); the grid form mirrors the simulate code.
    """
    i1 = round(t1 / t2 * n_steps)
    t = np.linspace(0.0, t2, n_steps + 1)
    tq = t * msv.Q(t, A0, A1, A2, LAM)
    return float(np.diff(tq)[i1:].sum()) / (t2 - t1)


def mc_forward_call_exact(
    m: float, t1: float, t2: float, vf: float, k: float, seed: int
) -> tuple[float, float]:
    """Exact MC of the stated model: zeta once per path, then exact increments.

    Given zeta the variance rate is deterministic, so the T1->T2 log-return is
    exactly normal (same one-draw logic as msv.mc_call) -- no Euler error.
    """
    rng = np.random.default_rng(seed)
    zeta = msv.draw_zeta(k, N_PATHS, rng)
    i1_var = t1 * float(msv.Q(t1, A0, A1, A2, LAM))
    fwd_var = (t2 - t1) * vf
    z1 = rng.standard_normal(N_PATHS)
    s_t1 = S0 * np.exp(R * t1 - 0.5 * zeta * i1_var + np.sqrt(zeta * i1_var) * z1)
    z2 = rng.standard_normal(N_PATHS)
    s_t2 = s_t1 * np.exp(R * (t2 - t1) - 0.5 * zeta * fwd_var + np.sqrt(zeta * fwd_var) * z2)
    payoff = np.maximum(s_t2 - m * s_t1, 0.0)
    disc = float(np.exp(-R * t2))
    px = disc * float(payoff.mean())
    se = disc * float(payoff.std(ddof=1)) / np.sqrt(float(payoff.size))
    return px, se


def main() -> None:
    ok = True
    seed = 42
    print(f"Forward-start smiles under MSV: S0={S0}, r={R}, (a0,a1,a2,lam)=({A0},{A1},{A2},{LAM})")
    for k in KS:
        for t1 in T1S:
            t2 = t1 + DT
            vf = forward_var_rate(t1, t2)
            print(f"\n## k={k}, T1={t1} (T2={t2}), forward variance rate vf={vf:.6f}")
            print("\n| m | forward IV | spot IV (tau=T2-T1, K=m*F) |")
            print("|---|------------|-------------------------------|")
            for m in MS:
                print(f"| {m:.2f} | {forward_iv(m, t1, t2, vf, k):.6f} | {spot_iv(m, DT, k):.6f} |")

            print("\nSymmetry in forward log-moneyness ln(K/F_fwd) = ln(m) - r*(T2-T1) = +/-x:")
            print("| x | IV at m=exp(r*dt+x) | IV at m=exp(r*dt-x) | abs diff |")
            print("|---|---------------------|---------------------|----------|")
            for x in (0.05, 0.1, 0.2):
                iv_up = forward_iv(float(np.exp(R * DT + x)), t1, t2, vf, k)
                iv_dn = forward_iv(float(np.exp(R * DT - x)), t1, t2, vf, k)
                print(f"| {x:.2f} | {iv_up:.6f} | {iv_dn:.6f} | {abs(iv_up - iv_dn):.2e} |")

            print("\nClaim (b): given zeta, the T1-conditional smile expiring at T2 is flat:")
            print("| zeta | sqrt(zeta*vf) | min IV | max IV | spread |")
            print("|------|---------------|--------|--------|--------|")
            s2 = np.log1p(k * k)
            ps = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
            zetas = np.exp(np.sqrt(s2) * norm.ppf(ps) - s2 / 2.0)
            worst = 0.0
            for z in zetas:
                ivs = [implied_vol(float(msv.bs_call(1.0, m, R, DT, z * vf)), m, DT) for m in MS]
                spread = max(ivs) - min(ivs)
                worst = max(worst, spread)
                print(
                    f"| {z:.4f} | {float(np.sqrt(z * vf)):.6f} "
                    f"| {min(ivs):.6f} | {max(ivs):.6f} | {spread:.2e} |"
                )
            flat_ok = worst < 1e-8
            ok = ok and flat_ok
            print(f"max spread across strikes = {worst:.2e} (< 1e-8: {flat_ok})")

            print("\nClaim (c): MC vs closed form (n_paths=200000, n_steps=200):")
            vf_sim = simulate_forward_var(t1, t2)
            print(
                f"note: msv.simulate applies forward variance rate {vf_sim:.6f} over [T1,T2] "
                f"({(vf_sim / vf - 1) * 100:+.2f}% vs model vf={vf:.6f}) -- exact per-step "
                "g-integrals (epic 0009), so its MC must match the closed form up to noise."
            )
            print("| m | closed | MC simulate | diff | 3*SE | inside |")
            print("|---|--------|-------------|------|------|--------|")
            for m in (0.9, 1.0):
                closed = forward_call(m, t1, t2, vf, k)
                mc, se = mc_forward_call(m, t1, t2, k, seed)
                seed += 1
                inside = abs(mc - closed) <= 3 * se
                print(
                    f"| {m:.2f} | {closed:.6f} | {mc:.6f} | {abs(mc - closed):.6f} "
                    f"| {3 * se:.6f} | {inside} |"
                )
            print("| m | closed | MC exact | diff | 3*SE | inside |")
            print("|---|--------|----------|------|------|--------|")
            for m in (0.9, 1.0):
                closed = forward_call(m, t1, t2, vf, k)
                mc, se = mc_forward_call_exact(m, t1, t2, vf, k, seed)
                seed += 1
                inside = abs(mc - closed) <= 3 * se
                ok = ok and inside
                print(
                    f"| {m:.2f} | {closed:.6f} | {mc:.6f} | {abs(mc - closed):.6f} "
                    f"| {3 * se:.6f} | {inside} |"
                )
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
