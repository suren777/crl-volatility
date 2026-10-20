"""Replicate the nine 2014 MSV calibrations that produced the paper's "July 2013" tables.

The scripts NEW_DATA/MSV/calibration{DAX,FTSE,SNP}_new{1,2,3}.m were run in April 2014 on
Bloomberg data dated 15/16/17 April 2013 (sheets SNP500/FTSE100/DAX of
OptionPricesNEW_still.xlsx). For each script we parse (regex) the xlsread ranges, spot cell,
weight power, x0 and the hard-coded in/out boolean index, rebuild the numeric matrix exactly
as MATLAB's xlsread produced it, port rffix (per-option interest rate), and run the six
standard variants of legacy_pricer.replicate. Writes results/apr2013.json and prints a
markdown table per dataset.

Sheet layout (all three sheets): B..H = day 1 block (Strike, Expiry, T, PX_BID, PX_ASK,
OPT_FINANCE_RATE, IMPLIED_VOLATILITY); I = Security names (text, no numbers); J..N, O..S,
T..X = day 2/3/4 blocks (T, PX_BID, PX_ASK, OPT_FINANCE_RATE, IMPLIED_VOLATILITY).
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from legacy_pricer import Dataset, replicate
from openpyxl import load_workbook  # type: ignore[import-untyped]
from openpyxl.worksheet.worksheet import Worksheet  # type: ignore[import-untyped]
from scipy.optimize import minimize
from scipy.stats import norm

LEGACY = Path(__file__).resolve().parent.parent / "legacy/MATLAB/Options/NEW_DATA/MSV"
XLSX = LEGACY / "OptionPricesNEW_still.xlsx"
RESULTS = Path(__file__).parent / "results"
N_STARTS = 10  # multistarts for variants 3-6; 20 took ~61 s/dataset (~11 min total), too slow
MINFUNC = "NEW_DATA/MSV/minfunc.m (script folder shadows parent logaddedparam/minfunc*.m)"

FloatArray = npt.NDArray[np.float64]

# Published MSV rows (MRAE_in %, RMSE_in, MRAE_out %, RMSE_out), table order FTSE, S&P, DAX.
PUBLISHED: dict[str, list[tuple[str, float, float, float, float]]] = {
    "FTSE": [
        ("25 Jul 2013", 4.15, 9.80, 5.27, 11.11),
        ("26 Jul 2013", 4.45, 9.37, 5.83, 10.80),
        ("29 Jul 2013", 4.52, 9.35, 5.74, 10.73),
    ],
    "SNP": [
        ("25 Jul 2013", 3.97, 3.11, 4.30, 2.34),
        ("26 Jul 2013", 3.86, 3.06, 1.00, 2.43),
        ("29 Jul 2013", 4.71, 2.99, 5.29, 2.30),
    ],
    "DAX": [
        ("25 Jul 2013", 13.81, 110.51, 17.18, 114.35),
        ("26 Jul 2013", 4.24, 17.55, 8.01, 18.17),
        ("29 Jul 2013", 4.34, 18.35, 8.03, 18.75),
    ],
}

# (name, script file, sheet, published index, optional faithful range override)
SCRIPTS = [
    ("FTSE new1", "calibrationFTSE_new1.m", "FTSE100", "FTSE", None),
    ("FTSE new2", "calibrationFTSE_new2.m", "FTSE100", "FTSE", None),
    ("FTSE new3", "calibrationFTSE_new3.m", "FTSE100", "FTSE", None),
    ("SNP new1", "calibrationSNP_new1.m", "SNP500", "SNP", None),
    ("SNP new2", "calibrationSNP_new2.m", "SNP500", "SNP", None),
    ("SNP new3", "calibrationSNP_new3.m", "SNP500", "SNP", None),
    ("DAX new1", "calibrationDAX_new1.m", "DAX", "DAX", None),
    ("DAX new2", "calibrationDAX_new2.m", "DAX", "DAX", None),
    ("DAX new3", "calibrationDAX_new3.m", "DAX", "DAX", None),
    # FTSE/SNP new3's faithful range N5:X104 starts one block column late (see md): after the
    # strike overwrite, D(:,3) is the day-3 PX_BID, not T. M5:S104 (DAX new3's own range) is
    # the correctly aligned day-3 block; same strike overwrite, spot, weights, x0 and split.
    ("FTSE new3 (aligned)", "calibrationFTSE_new3.m", "FTSE100", "FTSE", "M5:S104"),
    ("SNP new3 (aligned)", "calibrationSNP_new3.m", "SNP500", "SNP", "M5:S104"),
]

_RANGE_RE = re.compile(
    r"D=xlsread\('OptionPricesNEW_still\.xlsx','(\w+)','([A-Z]+\d+:[A-Z]+\d+)'\)"
)
_STRIKE_RE = re.compile(
    r"D\(:,1\)=xlsread\('OptionPricesNEW_still\.xlsx','\w+','([A-Z]+\d+:[A-Z]+\d+)'\)"
)
_SPOT_RE = re.compile(r"S=xlsread\('OptionPricesNEW_still\.xlsx','\w+','([A-Z]+\d+)'\)")
_POWER_RE = re.compile(r"w=\s*1\./abs\(bid-ask\)\.\^(\d+);")
_X0_RE = re.compile(r"x0=\[([^\]]*)\]")
_INDEX_RE = re.compile(r"index=\[([^\]]*)\];")


@dataclass(frozen=True)
class Parsed:
    name: str
    script: str
    sheet: str
    range: str
    strike_range: str | None
    spot_cell: str
    power: int
    x0: list[float]
    index: FloatArray  # True = out-of-sample, exactly the script's `index` vector
    seven_param: bool


def parse_script(name: str, script: str, sheet: str, range_override: str | None) -> Parsed:
    text = (LEGACY / script).read_text()
    m = _RANGE_RE.search(text)
    assert m, script
    strikes = _STRIKE_RE.search(text)
    spot = _SPOT_RE.search(text)
    power = _POWER_RE.search(text)
    x0 = _X0_RE.search(text)
    index = _INDEX_RE.search(text)
    assert spot and power and x0 and index, script
    assert "mmpricer1(" not in text, f"{script}: unexpected 7-param pricer"
    assert "minfunc(" in text, script
    return Parsed(
        name=name,
        script=script,
        sheet=sheet,
        range=range_override or m.group(2),
        strike_range=strikes.group(1) if strikes else None,
        spot_cell=spot.group(1),
        power=int(power.group(1)),
        x0=[float(t) for t in x0.group(1).split(";") if t.strip()],
        index=np.array([t.strip() == "true" for t in index.group(1).split(";") if t.strip()]),
        seven_param=False,
    )


def xlsread(ws: Worksheet, rng: str) -> FloatArray:
    """Numeric part of MATLAB xlsread: text/empty -> NaN, then drop LEADING and TRAILING
    rows/columns of the range that hold no numeric value. Interior all-text columns stay as
    NaN (COM-mode behaviour; dropping them would shift every later column left). For these
    sheets no leading/trailing trim fires; the only non-numeric column is the interior
    Security names column I inside H5:H104-derived ranges. Date cells read as serial numbers."""
    m = re.fullmatch(r"([A-Z]+)(\d+)(?::([A-Z]+)(\d+))?", rng)
    assert m, rng
    c1 = ws[m.group(1) + "1"].column
    c2 = ws[m.group(3) + "1"].column if m.group(3) else c1
    r1, r2 = int(m.group(2)), int(m.group(4) or m.group(2))
    grid = np.empty((r2 - r1 + 1, c2 - c1 + 1))
    for i, r in enumerate(range(r1, r2 + 1)):
        for j, c in enumerate(range(c1, c2 + 1)):
            v = ws.cell(row=r, column=c).value
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                grid[i, j] = float(v)
            elif isinstance(v, datetime):  # date cells are numeric serial numbers in xlsread
                grid[i, j] = float(v.toordinal())
            else:
                grid[i, j] = np.nan
    if not np.isfinite(grid).any():
        return grid
    rows = np.isfinite(grid).any(axis=1)
    cols = np.isfinite(grid).any(axis=0)
    r_idx = np.arange(grid.shape[0])[rows.argmax() : grid.shape[0] - rows[::-1].argmax()]
    c_idx = np.arange(grid.shape[1])[cols.argmax() : grid.shape[1] - cols[::-1].argmax()]
    trimmed: FloatArray = grid[np.ix_(r_idx, c_idx)]
    return trimmed


def blsprice(S: float, K: float, r: float, T: float, sigma: float) -> float:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _rfix_one(C: float, S: float, K: float, T: float, iv: float) -> float:
    def rmin(c: FloatArray) -> float:
        y = (C - blsprice(S, K, abs(float(c[0])), T, iv)) ** 2
        return float(y + 1e6 * (c[0] < 1e-4))

    res = minimize(rmin, np.array([0.001]), method="Nelder-Mead")
    return abs(float(res.x[0]))


def rffix(K: FloatArray, T: FloatArray, C: FloatArray, iv: FloatArray, S: float) -> FloatArray:
    """Port of rffix.m: per-option r from Nelder-Mead on rmin, start 0.001, then abs()."""
    return np.array(
        [_rfix_one(float(C[i]), S, float(K[i]), float(T[i]), float(iv[i])) for i in range(K.size)]
    )


def build_dataset(p: Parsed, ws: Worksheet) -> Dataset:
    D = xlsread(ws, p.range)
    if p.strike_range:
        D[:, 0] = xlsread(ws, p.strike_range).ravel()
    assert D.shape[0] == p.index.size, f"{p.name}: index/D size mismatch"
    S = float(xlsread(ws, p.spot_cell).ravel()[0])
    K = D[:, 0]
    T = D[:, 2] / 252.0
    bid, ask = D[:, 4], D[:, 3]  # script's labels are swapped; the mid is unaffected
    C = (bid + ask) / 2.0
    iv = D[:, 6] / 100.0
    w = 1.0 / np.abs(bid - ask) ** p.power
    r = rffix(K, T, C, iv, S)
    idx = p.index.astype(bool)
    return Dataset(
        p.name, S, K[~idx], T[~idx], C[~idx], r[~idx], w[~idx], K[idx], T[idx], C[idx], r[idx]
    )


def best_published(group: str, v1: dict[str, object]) -> dict[str, object]:
    ins = cast(dict[str, float], v1["in"])
    out = cast(dict[str, float], v1["out"])
    cands: dict[str, dict[str, object]] = {}
    for date, mi, ri, mo, ro in PUBLISHED[group]:
        d = (
            abs(ins["MRAE"] - mi)
            + abs(ins["RMSE"] - ri)
            + abs(out["MRAE"] - mo)
            + abs(out["RMSE"] - ro)
        )
        cands[date] = {
            "table_date": date,
            "MRAE_in": mi,
            "RMSE_in": ri,
            "MRAE_out": mo,
            "RMSE_out": ro,
            "l1_distance": d,
        }
    best = min(cands.values(), key=lambda c: cast(float, c["l1_distance"]))
    return {"best": best, "all": cands}


def print_table(entry: dict[str, object]) -> None:
    pub = cast(dict[str, object], entry["published"])
    mf = cast(str, entry["minfunc"])
    print(
        f"\n### {entry['name']} — script {entry['script']}, S={entry['S']}, "
        f"n_in={entry['n_in']}, n_out={entry['n_out']}, "
        f"w=1/|bid-ask|^{entry['weight_power']}, floor plain_0.01, {mf.split(' ')[0]}"
    )
    print(
        f"published {pub['table_date']}: MRAE_in {pub['MRAE_in']} | RMSE_in {pub['RMSE_in']} | "
        f"MRAE_out {pub['MRAE_out']} | RMSE_out {pub['RMSE_out']}"
    )
    print(
        "| variant | k | MRAE in % | RMSE in | MaxRE in % | MRAE out % | RMSE out | MaxRE out % |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for row in cast(list[dict[str, object]], entry["rows"]):
        i = cast(dict[str, float], row["in"])
        o = cast(dict[str, float], row["out"])
        print(
            f"| {row['variant']} | {row['k']:.4g} | {i['MRAE']:.2f} | {i['RMSE']:.2f} | "
            f"{i['MaxRE']:.2f} | {o['MRAE']:.2f} | {o['RMSE']:.2f} | {o['MaxRE']:.2f} |"
        )


def main() -> None:
    wb = load_workbook(XLSX, data_only=True)
    datasets = []
    for name, script, sheet, group, override in SCRIPTS:
        p = parse_script(name, script, sheet, override)
        ds = build_dataset(p, cast(Worksheet, wb[sheet]))
        rows = replicate(ds, p.x0, p.seven_param, "plain_0.01", n_starts=N_STARTS)
        match = best_published(group, rows[0])
        datasets.append(
            {
                "name": ds.name,
                "script": p.script,
                "minfunc": MINFUNC,
                "S": ds.S,
                "n_in": int(ds.K_in.size),
                "n_out": int(ds.K_out.size),
                "weight_power": p.power,
                "seven_param": p.seven_param,
                "floor": "plain_0.01",
                "published": match["best"],
                "published_l1_all": match["all"],
                "rows": rows,
            }
        )
        print_table(datasets[-1])
    RESULTS.mkdir(exist_ok=True)
    out = json.dumps({"group": "apr2013", "datasets": datasets}, indent=2)
    (RESULTS / "apr2013.json").write_text(out)
    print(f"\nwrote {RESULTS / 'apr2013.json'}")


if __name__ == "__main__":
    main()
