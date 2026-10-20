"""Replicate the three 1 November 2012 MSV calibrations (FTSE, S&P500, DAX) exactly as the
2014 MATLAB scripts ran them: same data rows, same hard-coded in/out split, same weight
power, same x0, same pricer and same minfunc penalty. Writes results/nov01_2012.json and
prints one markdown table per dataset.
"""

import json
import re
from pathlib import Path
from typing import Any, NamedTuple, cast

import legacy_pricer as lp
import numpy as np
import numpy.typing as npt
from scipy.io import loadmat

LEGACY = Path(__file__).resolve().parent.parent / "legacy/MATLAB/Options/logaddedparam"
OUT = Path(__file__).resolve().parent / "results"
N_STARTS = 20

PUBLISHED: dict[str, dict[str, float]] = {
    "FTSE": {"MRAE_in": 3.30, "RMSE_in": 9.62, "MRAE_out": 2.08, "RMSE_out": 5.77},
    "S&P500": {"MRAE_in": 4.75, "RMSE_in": 3.30, "MRAE_out": 4.73, "RMSE_out": 3.27},
    "DAX": {"MRAE_in": 4.37, "RMSE_in": 20.51, "MRAE_out": 5.60, "RMSE_out": 23.42},
}


class Spec(NamedTuple):
    name: str
    script: str
    mat: str
    t_col: int  # 1-based data column holding T (SANDP500 skips column 4)
    r_col: int


SPECS = [
    Spec("FTSE", "calibration_ftse.m", "FTSE100", t_col=4, r_col=5),
    Spec("S&P500", "snp_cal_new.m", "SANDP500", t_col=5, r_col=6),
    Spec("DAX", "calibDAX.m", "DAX", t_col=4, r_col=5),
]


def must(m: re.Match[str] | None, what: str) -> re.Match[str]:
    assert m is not None, f"could not parse {what}"
    return m


def parse_index(text: str, script: str) -> npt.NDArray[np.bool_]:
    """The active (uncommented) index=[...] vector; true = out-of-sample."""
    m = must(re.search(r"^index=\[(.*?)\];", text, re.MULTILINE), "index vector")
    vals = [v.strip().lower() for v in m.group(1).split(";") if v.strip()]
    assert all(v in ("true", "false") for v in vals), f"{script}: bad index values"
    return np.asarray([v == "true" for v in vals], dtype=bool)


def parse_x0(text: str, script: str) -> npt.NDArray[np.float64]:
    """The active x0 line. snp_cal_new.m uses x0=rand([7 1]); we seed it for
    reproducibility (MATLAB's start was whatever its RNG happened to produce)."""
    m = re.search(r"^x0=\[(.*?)\];", text, re.MULTILINE)
    if m:
        return np.asarray([float(v) for v in m.group(1).split(";") if v.strip()], dtype=float)
    m = re.search(r"^x0=rand\(\[(\d+) 1\]\);", text, re.MULTILINE)
    if m:
        return np.random.default_rng(0).uniform(0.0, 1.0, int(m.group(1)))
    raise ValueError(f"{script}: no active x0 line")


Vec = npt.NDArray[np.float64]


def split(data: Vec, t_col: int, r_col: int) -> tuple[Vec, Vec, Vec, Vec, Vec]:
    return data[:, 0], data[:, 1], data[:, 2], data[:, t_col - 1], data[:, r_col - 1]


def run_one(spec: Spec) -> dict[str, Any]:
    text = (LEGACY / spec.script).read_text()
    raw = loadmat(LEGACY / f"{spec.mat}.mat")[spec.mat]
    data = np.asarray(raw, dtype=float)
    index = parse_index(text, spec.script)
    assert index.size == data.shape[0], f"{spec.script}: {index.size} flags vs {data.shape[0]} rows"
    w = re.search(r"w\s*=\s*1\./abs\(bid-ask\)\.\^(\d+);", text, re.MULTILINE)
    p = int(must(w, "w").group(1))
    s_val = float(must(re.search(r"^S=([0-9.]+);", text, re.MULTILINE), "S").group(1))
    minfunc = "minfunc1.m" if re.search(r"minfunc1\(", text) else "minfunc.m"
    seven_param = bool(re.search(r"mmpricer1\(", text))
    assert (minfunc == "minfunc1.m") == seven_param, f"{spec.script}: pricer/minfunc mismatch"
    # MATLAB resolves minfunc*.m from the script folder first; both live in LEGACY itself.
    floor: lp.FloorKind = "none" if minfunc == "minfunc1.m" else "abs_0.001"
    x0 = parse_x0(text, spec.script)

    inr, outr = data[~index], data[index]
    K_in, bid_in, ask_in, T_in, r_in = split(inr, spec.t_col, spec.r_col)
    K_out, bid_out, ask_out, T_out, r_out = split(outr, spec.t_col, spec.r_col)
    ds = lp.Dataset(
        spec.name,
        s_val,
        K_in,
        T_in,
        (bid_in + ask_in) / 2,
        r_in,
        1.0 / np.abs(bid_in - ask_in) ** p,
        K_out,
        T_out,
        (bid_out + ask_out) / 2,
        r_out,
    )
    rows = lp.replicate(ds, x0, seven_param, floor, n_starts=N_STARTS, seed=0)
    return {
        "name": spec.name,
        "script": spec.script,
        "minfunc": minfunc,
        "S": s_val,
        "n_in": int(ds.C_in.size),
        "n_out": int(ds.C_out.size),
        "weight_power": p,
        "seven_param": seven_param,
        "floor": floor,
        "published": PUBLISHED[spec.name],
        "rows": rows,
    }


def table(entry: dict[str, Any]) -> list[str]:
    pub = cast(dict[str, float], entry["published"])
    lines = [
        f"## {entry['name']} — {entry['script']}, {entry['minfunc']}, "
        f"w = 1/|bid-ask|^{entry['weight_power']}, S = {entry['S']}, "
        f"{entry['n_in']} in / {entry['n_out']} out, seven_param={entry['seven_param']}, "
        f"floor={entry['floor']}",
        "",
        "| variant | k | MRAE in % | RMSE in | MRAE out % | RMSE out |",
        "|---|---|---|---|---|---|",
    ]
    for row in cast(list[dict[str, Any]], entry["rows"]):
        i, o = cast(dict[str, float], row["in"]), cast(dict[str, float], row["out"])
        lines.append(
            f"| {row['variant']} | {row['k']:.4f} | {i['MRAE']:.2f} | {i['RMSE']:.2f} "
            f"| {o['MRAE']:.2f} | {o['RMSE']:.2f} |"
        )
    lines.append(
        f"| published 2014 | — | {pub['MRAE_in']:.2f} | {pub['RMSE_in']:.2f} "
        f"| {pub['MRAE_out']:.2f} | {pub['RMSE_out']:.2f} |"
    )
    return lines


def main() -> None:
    entries = [run_one(spec) for spec in SPECS]
    OUT.mkdir(exist_ok=True)
    (OUT / "nov01_2012.json").write_text(
        json.dumps({"group": "nov01_2012", "datasets": entries}, indent=2) + "\n"
    )
    print("\n\n".join("\n".join(table(e)) for e in entries))


if __name__ == "__main__":
    main()
