import gzip
import json
import sqlite3
from pathlib import Path

import cboe_snapshot
import pytest


def payload(n: int, last_trade: str = "2026-10-16T16:14:59") -> bytes:
    options = [
        {"option": f"SPXW261120C{4000000 + i:08d}", "bid": 1.0, "ask": 1.5} for i in range(n)
    ]
    data = {"last_trade_time": last_trade, "close": 7659.44, "options": options}
    return json.dumps({"timestamp": "x", "data": data}).encode()


def test_saves_once_per_trade_date(tmp_path: Path) -> None:
    raw = payload(2000)
    first = cboe_snapshot.snapshot(tmp_path, lambda: raw)
    assert first == tmp_path / "spx" / "2026-10-16.json.gz"
    with gzip.open(first, "rb") as f:
        assert f.read() == raw
    assert cboe_snapshot.snapshot(tmp_path, lambda: raw) is None


def test_rejects_truncated_chain(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        cboe_snapshot.snapshot(tmp_path, lambda: payload(10))
    assert not (tmp_path / "spx").exists()


def test_sqlite_rows_and_rebuild(tmp_path: Path) -> None:
    raw = payload(2000)
    cboe_snapshot.snapshot(tmp_path, lambda: raw)
    db = tmp_path / "spx.sqlite"
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM quotes").fetchone() == (2000,)
        row = con.execute("SELECT root, expiry, cp, strike, bid FROM quotes LIMIT 1").fetchone()
        assert row == ("SPXW", "2026-11-20", "C", 4000.0, 1.0)
        assert con.execute("SELECT spot FROM snapshots").fetchone() == (7659.44,)
    db.unlink()
    assert cboe_snapshot.rebuild(tmp_path) == 1
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM quotes").fetchone() == (2000,)
