"""Emit the Lessons-section replication table as LaTeX from committed results.

Reads replication/results/nov01_2012.json and nov26_2012.json (no recomputation)
and writes paper/tables/replication.tex: a booktabs `tabular` with one row per
(index, date). Columns are the four published MRAE/RMSE in/out metrics, the same
four for the faithful 2014 code and for the k = 0 fit, and the fitted k of the
faithful run. The closer of faithful / k = 0 to the published row (by summed
absolute difference over the four metrics) is bolded.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).parent / "results"
OUT = Path(__file__).parent.parent / "paper" / "tables" / "replication.tex"

FAITHFUL = "1 faithful 2014"
KZERO = "3 k=0 (BS + NS vol term structure)"
METRICS = ("MRAE_in", "RMSE_in", "MRAE_out", "RMSE_out")


@dataclass(frozen=True)
class Row:
    label: str
    published: list[float]
    faithful: list[float]
    kzero: list[float]
    k: float
    bold: str  # "faithful" | "kzero"


def _metrics(row: dict[str, Any]) -> list[float]:
    i, o = row["in"], row["out"]
    return [float(i["MRAE"]), float(i["RMSE"]), float(o["MRAE"]), float(o["RMSE"])]


def _pick(clone: dict[str, Any], variant: str) -> dict[str, Any]:
    return next(r for r in clone["rows"] if r["variant"] == variant)


def load_rows() -> list[Row]:
    rows: list[Row] = []
    for date in ("1 Nov 2012", "26 Nov 2012"):
        fname = {"1 Nov 2012": "nov01", "26 Nov 2012": "nov26"}[date] + "_2012.json"
        data = json.loads((RESULTS / fname).read_text())
        for ds in data["datasets"]:
            pub = [float(ds["published"][m]) for m in METRICS]
            f = _metrics(_pick(ds, FAITHFUL))
            z = _metrics(_pick(ds, KZERO))
            k = float(_pick(ds, FAITHFUL)["k"])
            deff = sum(abs(p - x) for p, x in zip(pub, f, strict=True))
            defz = sum(abs(p - x) for p, x in zip(pub, z, strict=True))
            bold = "faithful" if deff <= defz else "kzero"
            rows.append(Row(f"{ds['name']}, {date}", pub, f, z, k, bold))
    return rows


def cell(nums: list[float], bold: bool) -> str:
    return " & ".join(f"\\textbf{{{x:.2f}}}" if bold else f"{x:.2f}" for x in nums)


def build(rows: list[Row]) -> str:
    header = (
        "\\begin{tabular}{lccccccccccccc}\n"
        "\\toprule\n"
        " & \\multicolumn{4}{c}{Published (2014)} & \\multicolumn{4}{c}{Faithful code}"
        " & \\multicolumn{4}{c}{$k=0$} & \\\\\n"
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\n"
        " & MRAE-in & RMSE-in & MRAE-out & RMSE-out & MRAE-in & RMSE-in & MRAE-out & RMSE-out"
        " & MRAE-in & RMSE-in & MRAE-out & RMSE-out & $k$ \\\\\n"
        "\\midrule\n"
    )
    body = "\n".join(
        f"{r.label.replace('&', '\\&')} & {cell(r.published, False)}"
        f" & {cell(r.faithful, r.bold == 'faithful')}"
        f" & {cell(r.kzero, r.bold == 'kzero')} & {r.k:.3f} \\\\"
        for r in rows
    )
    return header + body + "\n\\bottomrule\n\\end{tabular}\n"


OUT_COMPACT = OUT.with_name("replication_compact.tex")


def build_compact(rows: list[Row]) -> str:
    """In-sample MRAE and out-of-sample RMSE only, so the table fits at full size."""
    header = (
        "\\begin{tabular}{lccccccc}\n"
        "\\toprule\n"
        " & \\multicolumn{2}{c}{Published} & \\multicolumn{2}{c}{Original code}"
        " & \\multicolumn{2}{c}{$k=0$} & \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        " & MRAE-in & RMSE-out & MRAE-in & RMSE-out & MRAE-in & RMSE-out & $k$ \\\\\n"
        "\\midrule\n"
    )
    body = "\n".join(
        f"{r.label.replace('&', '\\&')} & {cell([r.published[0], r.published[3]], False)}"
        f" & {cell([r.faithful[0], r.faithful[3]], r.bold == 'faithful')}"
        f" & {cell([r.kzero[0], r.kzero[3]], r.bold == 'kzero')} & {r.k:.3f} \\\\"
        for r in rows
    )
    return header + body + "\n\\bottomrule\n\\end{tabular}\n"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    OUT.write_text(build(rows))
    OUT_COMPACT.write_text(build_compact(rows))


if __name__ == "__main__":
    main()
