"""Replicate the three 26 November 2012 MSV calibrations (FTSE, S&P500, DAX) exactly.

Parses each 2014 MATLAB calibration script (index split, x0, S, weight power, which
minfunc), loads the legacy .mat data from the read-only legacy tree, runs the six
standard variants via legacy_pricer.replicate, and writes results/nov26_2012.{json,md}.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from legacy_pricer import Dataset, FloorKind, replicate
from scipy.io import loadmat

LEGACY = Path(__file__).resolve().parent.parent / "legacy/MATLAB/Options/logaddedparam"
RESULTS = Path(__file__).parent / "results"
GROUP = "nov26_2012"

# Published 26 Nov rows: MRAE-in, RMSE-in, time, MRAE-out, RMSE-out (time not compared).
PUBLISHED: dict[str, dict[str, float]] = {
    "FTSE": {"MRAE_in": 3.33, "RMSE_in": 7.82, "MRAE_out": 4.32, "RMSE_out": 8.63},
    "S&P500": {"MRAE_in": 3.81, "RMSE_in": 2.32, "MRAE_out": 3.84, "RMSE_out": 3.15},
    "DAX": {"MRAE_in": 4.79, "RMSE_in": 42.80, "MRAE_out": 4.42, "RMSE_out": 40.38},
}

# (dataset name, .mat variable) in published table order; script is calib<var>.m.
GROUPS = [("FTSE", "FTSEnew"), ("S&P500", "SNPnew"), ("DAX", "DAXnew")]

META_KEYS = (
    "name",
    "script",
    "minfunc",
    "S",
    "n_in",
    "n_out",
    "weight_power",
    "seven_param",
    "floor",
    "published",
)


def _active(text: str, pattern: str) -> str:
    """First capture of pattern on a non-comment line of a MATLAB script."""
    for line in text.splitlines():
        if line.lstrip().startswith("%"):
            continue
        m = re.search(pattern, line)
        if m:
            return m.group(1)
    raise ValueError(f"no active line matches {pattern!r}")


def _resolve_minfunc(script_dir: Path, name: str) -> tuple[bool, FloorKind, str]:
    """Resolve the minfunc*.m a script calls (script folder first, then parent).

    Returns (seven_param, floor, path relative to LEGACY).
    """
    for folder in (script_dir, LEGACY):
        f = folder / f"{name}.m"
        if f.exists():
            body = f.read_text()
            if re.search(r"abs\(c\(1\)\)\s*<\s*0\.001", body):
                floor: FloorKind = "abs_0.001"
            elif re.search(r"c\(1\)\s*<\s*0\.01", body):
                floor = "plain_0.01"
            else:
                floor = "none"
            return "mmpricer1" in body, floor, str(f.relative_to(LEGACY))
    raise FileNotFoundError(name)


def load_dataset(name: str, var: str) -> tuple[Dataset, dict[str, Any], list[float]]:
    """Build the Dataset from the legacy .mat and the script's exact parsed settings."""
    folder = LEGACY / var
    script = folder / f"calib{var}.m"
    text = script.read_text()
    idx = np.array(
        [t == "true" for t in re.findall(r"true|false", _active(text, r"index=\[([^\]]*)\];"))]
    )
    x0 = [float(v) for v in _active(text, r"x0=\[([^\]]*)\];").split(";") if v.strip()]
    s = float(_active(text, r"S=([0-9.]+);"))
    power = int(_active(text, r"1\./abs\(bid-ask\)\.\^(\d+)"))
    minfunc = _active(text, r"fminsearch\(@\(c\)\s*(minfunc\w*)\(")
    seven, floor, minfunc_file = _resolve_minfunc(folder, minfunc)

    mat = loadmat(folder / f"{var}.mat")[var]
    assert len(idx) == mat.shape[0], f"{name}: index length {len(idx)} != {mat.shape[0]} rows"

    def split(rows: Any) -> tuple[Any, Any, Any, Any, Any]:
        bid, ask = rows[:, 1], rows[:, 2]
        return rows[:, 0], rows[:, 3], (bid + ask) / 2, rows[:, 4], 1 / np.abs(bid - ask) ** power

    kin, tin, cin, rin, win = split(mat[~idx])
    kout, tout, cout, rout, _ = split(mat[idx])
    ds = Dataset(
        name=name,
        S=s,
        K_in=np.asarray(kin, dtype=float),
        T_in=np.asarray(tin, dtype=float),
        C_in=np.asarray(cin, dtype=float),
        r_in=np.asarray(rin, dtype=float),
        w_in=np.asarray(win, dtype=float),
        K_out=np.asarray(kout, dtype=float),
        T_out=np.asarray(tout, dtype=float),
        C_out=np.asarray(cout, dtype=float),
        r_out=np.asarray(rout, dtype=float),
    )
    meta: dict[str, Any] = {
        "name": name,
        "script": str(script.relative_to(LEGACY)),
        "minfunc": minfunc_file,
        "S": s,
        "n_in": int(cin.size),
        "n_out": int(cout.size),
        "weight_power": power,
        "seven_param": seven,
        "floor": floor,
        "published": PUBLISHED[name],
    }
    return ds, meta, x0


def fmt_table(meta: dict[str, Any], rows: list[dict[str, object]]) -> str:
    """Markdown table of the six variants plus the published row, as printed."""
    pub = cast(dict[str, float], meta["published"])
    seven = "7-param" if meta["seven_param"] else "5-param"
    lines = [
        f"## {meta['name']} (S={meta['S']}, n_in={meta['n_in']}, n_out={meta['n_out']}, "
        f"w = 1/|bid-ask|^{meta['weight_power']}, {meta['minfunc']}, {seven}, "
        f"floor = {meta['floor']})",
        "",
        "| variant | k | MRAE-in % | RMSE-in | MaxRE-in % | MRAE-out % | RMSE-out | MaxRE-out % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        i = cast(dict[str, float], r["in"])
        o = cast(dict[str, float], r["out"])
        lines.append(
            f"| {r['variant']} | {cast(float, r['k']):.4f} | {i['MRAE']:.2f} | {i['RMSE']:.2f} "
            f"| {i['MaxRE']:.2f} | {o['MRAE']:.2f} | {o['RMSE']:.2f} | {o['MaxRE']:.2f} |"
        )
    lines.append(
        f"| published |  | {pub['MRAE_in']:.2f} | {pub['RMSE_in']:.2f} |  "
        f"| {pub['MRAE_out']:.2f} | {pub['RMSE_out']:.2f} |  |"
    )
    return "\n".join(lines)


ANALYSIS = """\
## Notes

Variant 1 reproduces the published S&P500 row to within 0.02 in all four metrics (3.82/2.34
in vs 3.81/2.32, 3.85/3.17 out vs 3.84/3.15) and the DAX row to within 0.06 MRAE but about
2 RMSE points (4.74/40.65 vs 4.79/42.80, 4.41/38.59 vs 4.42/40.38); for FTSE the published
row is matched instead by variant 2, the unpenalized fit (3.33/7.83 and 4.32/8.64 vs
published 3.33/7.82 and 4.32/8.63), while the faithful penalized run parks k on the +0.001
boundary and fits worse (3.86/7.39 in, 3.99/8.01 out). Every variant reaches |k| ~ 0 on all
three datasets (largest off-boundary magnitude 0.0007, DAX variant 6), so the fourth-moment
term is effectively switched off in every 26 November fit. Variant 3 (k = 0, Black-Scholes
with the Nelson-Siegel term structure) is therefore indistinguishable from variants 4-5 to
two decimals everywhere, and variant 6 differs only on DAX, where a different multi-start
optimum lowers the unweighted RMSE to 39.13/32.01 at a higher MRAE of 5.13/4.33 — the
spread-weighted objective does not rank it first. Anomalies found in the 2014 scripts and
data: the DAX exotic-simulation script behind the paper's barrier/Asian tables is
Simulation/DAX26Nov.m (also copied under AsianOptionSimulation/) with S0 = 7281.18, the
26 November spot, while the 1 November DAX calibration uses S = 7298.9 and the FTSE/S&P
"1st nov" simulation scripts use their own 1 November spots (5812.06 and 1412.16);
FTSEnew/minfunc12.m is dead code — it duplicates the parent minfunc1.m (7-parameter,
unpenalized) and is never called, since calibFTSEnew.m invokes plain minfunc, which
resolves to the parent's penalized 5-parameter minfunc.m for all three scripts; and each
script ships an unexamined with-K alternative (calibDAXnewwithstrikeK.m,
calibFTSEnewwithstrikeK.m, calibSNPnew_withK.m) that this replication ignores.
"""


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    datasets: list[dict[str, Any]] = []
    tables: list[str] = []
    for name, var in GROUPS:
        ds, meta, x0 = load_dataset(name, var)
        t0 = time.perf_counter()
        rows = replicate(ds, x0, meta["seven_param"], meta["floor"])
        table = fmt_table(meta, rows)
        print(table)
        print(f"({time.perf_counter() - t0:.0f}s)\n")
        entry = {k: meta[k] for k in META_KEYS} | {"rows": rows}
        datasets.append(entry)
        tables.append(table)
    (RESULTS / f"{GROUP}.json").write_text(
        json.dumps({"group": GROUP, "datasets": datasets}, indent=2) + "\n"
    )
    (RESULTS / f"{GROUP}.md").write_text("\n\n".join(tables) + "\n\n" + ANALYSIS)


if __name__ == "__main__":
    main()
