# Derivation checks

Every derivation the paper relies on is verified by at least two independent means before
it is trusted. The checks live here and run with the main suite (`uv run pytest -q`).

## Principles

1. **Symbolic** — a SymPy check of the identity itself (`simplify(lhs - rhs) == 0`, or
   series coefficients for asymptotics).
2. **Numeric, independent code path** — compared against an implementation that does not
   share the derivation: quadrature vs Monte Carlo, closed form vs finite differences,
   random parameter sweeps with `hypothesis`.
3. **Limits and special cases** must reduce to known results (e.g. ρ = 0 back to the MSV
   mixture, k → 0 back to Black–Scholes).
4. **Asymptotics** — the measured error must shrink at the predicted order, read off a
   log-log fit (`checks.loglog_slope`).
5. **Naming** — tests reference the paper equation label: `test_eq_<label>`.

## Adding a derivation

1. Add the symbolic check in the matching `test_*.py` module: derive the identity from
   first principles in SymPy, name it `test_eq_<label>_symbolic` (or fold it into
   `test_eq_<label>` if one test covers the label).
2. Add the numeric cross-check against independent code — `checks.black_fwd`, raw
   quadrature, or Monte Carlo written out in the test; never the function under test.
   Sweep random parameters with `hypothesis` and a fixed small
   `@settings(max_examples=..., deadline=None)` so the whole suite stays under ~60 s.
3. Check the known limits of the identity as part of the same module.
4. One-off explorations that are not regression tests (property searches) live in scripts
   here and report into a `.md` next to them, e.g. `search_calendar.py` →
   `search_calendar.md`. Numbers in `.md` files come from the script's own output, never
   hand-edited.

Helpers in `checks.py`: `relerr`, `lognormal_moment` (E[ζⁿ] = (1+k²)^{n(n−1)/2}),
`loglog_slope` (convergence order), `black_fwd` (plain-numpy Black, independent of
`msv.py` — use it for cross-checks instead of anything from `msv.py`).

## Type-checking

`MYPYPATH=.:derivations uv run mypy derivations` (strict). SymPy ships no type stubs; its
import lines carry narrow `# type: ignore[import-untyped]`.
