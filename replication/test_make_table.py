"""Tests for make_table: the replication table has 6 rows, matches the JSON, compiles."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest
from make_table import FAITHFUL, KZERO, METRICS, build, build_compact, load_rows

RESULTS = Path(__file__).parent / "results"

Json = dict[str, Any]


def _clone_json(name: str) -> list[Json]:
    return list(json.loads((RESULTS / name).read_text())["datasets"])


def test_six_rows_in_expected_order() -> None:
    rows = load_rows()
    assert [r.label for r in rows] == [
        "FTSE, 1 Nov 2012",
        "S&P500, 1 Nov 2012",
        "DAX, 1 Nov 2012",
        "FTSE, 26 Nov 2012",
        "S&P500, 26 Nov 2012",
        "DAX, 26 Nov 2012",
    ]


def test_values_match_json() -> None:
    rows = load_rows()
    for r in rows:
        date = "1 Nov 2012" if r.label.endswith("1 Nov 2012") else "26 Nov 2012"
        file = "nov01_2012.json" if date == "1 Nov 2012" else "nov26_2012.json"
        ds = {d["name"]: d for d in _clone_json(file)}[r.label.split(", ")[0]]
        pub = [float(ds["published"][m]) for m in METRICS]
        f = _pick(ds, FAITHFUL)
        z = _pick(ds, KZERO)
        assert r.published == pytest.approx(pub)
        assert [*r.faithful] == pytest.approx(
            [f["in"]["MRAE"], f["in"]["RMSE"], f["out"]["MRAE"], f["out"]["RMSE"]]
        )
        assert [*r.kzero] == pytest.approx(
            [z["in"]["MRAE"], z["in"]["RMSE"], z["out"]["MRAE"], z["out"]["RMSE"]]
        )
        assert r.k == pytest.approx(f["k"])


def test_table_contains_six_data_rows() -> None:
    rows = load_rows()
    table = build(rows)
    body = table.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
    assert body.count("\\\\") == len(rows)  # one \\ per data row


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_table_compiles_in_minimal_document() -> None:
    rows = load_rows()
    tex = (
        "\\documentclass{article}\n"
        "\\usepackage{booktabs}\n"
        "\\begin{document}\n" + build(rows) + "\n\\end{document}\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "main.tex").write_text(tex)
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=tmp,
            capture_output=True,
            check=True,
        )
        assert (d / "main.pdf").exists()


def _pick(ds: Json, variant: str) -> Json:
    return next(r for r in ds["rows"] if r["variant"] == variant)


def test_compact_table_has_six_rows_and_seven_numbers() -> None:
    tex = build_compact(load_rows())
    rows = [ln for ln in tex.splitlines() if ln.endswith("\\\\") and "Nov 2012" in ln]
    assert len(rows) == 6
    assert all(ln.count("&") - ln.count("\\&") == 7 for ln in rows)
