"""Daily snapshot of the Cboe delayed SPX option chain (stdlib only, run from cron).

Writes <out>/spx/<trade_date>.json.gz (lossless archive), keyed by the underlying's last trade
date so weekend and holiday runs are no-ops, and loads each snapshot into <out>/spx.sqlite
(tables `snapshots` and `quotes`, one row per option). The database can always be rebuilt
from the archive with --rebuild. Raw Cboe data may not be redistributed: keep <out> outside
the repo and publish only derived results.
"""

import argparse
import gzip
import json
import os
import re
import sqlite3
import sys
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

URL = "https://cdn-api.cboe.com/api/global/delayed_quotes/options/_SPX.json"
DEFAULT_OUT = Path.home() / "Data" / "article" / "cboe"
SYMBOL = re.compile(r"^([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")
FIELDS = (
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "iv",
    "open_interest",
    "volume",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "theo",
    "last_trade_price",
    "last_trade_time",
)
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS snapshots (
    trade_date TEXT PRIMARY KEY, fetched_at TEXT, spot REAL, prev_close REAL, iv30 REAL,
    n_options INTEGER
);
CREATE TABLE IF NOT EXISTS quotes (
    trade_date TEXT NOT NULL REFERENCES snapshots(trade_date), root TEXT NOT NULL,
    expiry TEXT NOT NULL, cp TEXT NOT NULL, strike REAL NOT NULL, {", ".join(FIELDS)},
    PRIMARY KEY (trade_date, root, expiry, cp, strike)
) WITHOUT ROWID;
"""


def fetch(url: str = URL) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        body: bytes = resp.read()
    return body


def ingest(db: Path, raw: bytes) -> str:
    """Load one raw snapshot into SQLite (idempotent per trade date); return the trade date."""
    data = json.loads(raw)
    d = data["data"]
    trade_date = str(d["last_trade_time"][:10])
    rows = []
    for o in d["options"]:
        m = SYMBOL.match(o["option"])
        if m is None:
            raise ValueError(f"unparsed option symbol {o['option']!r}")
        root, yy, mm, dd, cp, strike = m.groups()
        rows.append(
            (
                trade_date,
                root,
                f"20{yy}-{mm}-{dd}",
                cp,
                int(strike) / 1000,
                *(o.get(f) for f in FIELDS),
            )
        )
    with sqlite3.connect(db) as con:
        con.executescript(SCHEMA)
        con.execute("DELETE FROM quotes WHERE trade_date = ?", (trade_date,))
        con.execute(
            "INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?, ?, ?, ?)",
            (
                trade_date,
                data.get("timestamp"),
                d.get("close"),
                d.get("prev_day_close"),
                d.get("iv30"),
                len(rows),
            ),
        )
        con.executemany(f"INSERT INTO quotes VALUES ({', '.join('?' * (5 + len(FIELDS)))})", rows)
    return trade_date


def snapshot(out: Path, get: Callable[[], bytes] = fetch) -> Path | None:
    """Save one snapshot; return its path, or None if that trade date is already stored."""
    raw = get()
    payload = json.loads(raw)
    data = payload["data"]
    if len(data["options"]) < 1000:
        raise ValueError(f"suspiciously small chain: {len(data['options'])} options")
    trade_date = data["last_trade_time"][:10]
    target = out / "spx" / f"{trade_date}.json.gz"
    if target.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, target)
    ingest(out / "spx.sqlite", raw)
    return target


def rebuild(out: Path) -> int:
    """Re-ingest every archived snapshot; return how many were loaded."""
    files = sorted((out / "spx").glob("*.json.gz"))
    for f in files:
        with gzip.open(f, "rb") as fh:
            ingest(out / "spx.sqlite", fh.read())
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rebuild", action="store_true", help="re-ingest the whole archive")
    args = parser.parse_args()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    if args.rebuild:
        print(f"{now} rebuilt database from {rebuild(args.out)} snapshots")
        return 0
    try:
        path = snapshot(args.out)
    except Exception as exc:
        print(f"{now} ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"{now} {'saved ' + str(path) if path else 'skip (trade date already stored)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
