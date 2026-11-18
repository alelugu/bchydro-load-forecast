"""SQLite access: connect, create the schema, upsert rows, fetch the published file."""
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import config

SCHEMA = Path(__file__).with_name("schema.sql")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: Path | str | None = None, read_only: bool = False) -> sqlite3.Connection:
    path = Path(path or config.DB_PATH)
    if read_only:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db(path: Path | str | None = None) -> None:
    """Create tables and seed the locations. Safe to run again."""
    with connect(path) as con:
        con.executescript(SCHEMA.read_text())
        upsert(con, "locations",
               [dict(location_id=i, name=n, lat=la, lon=lo, pop_weight=w)
                for i, n, la, lo, w in config.LOCATIONS])


def upsert(con: sqlite3.Connection, table: str, rows: list[dict]) -> int:
    """Insert rows, replacing any with the same primary key, so reruns never duplicate."""
    if not rows:
        return 0
    cols = list(rows[0])
    sql = (f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
           f"VALUES ({', '.join(':' + c for c in cols)})")
    con.executemany(sql, rows)
    return len(rows)


def log_ingest(con: sqlite3.Connection, job: str, status: str, rows: int = 0, message: str = "") -> None:
    con.execute("INSERT INTO ingest_log (job, ran_at, status, rows, message) VALUES (?, ?, ?, ?, ?)",
                (job, utcnow(), status, rows, message))


def refresh_db_if_stale() -> bool:
    """Download the published database when DB_URL is set and the local copy is
    missing or older than DB_MAX_AGE_HOURS. Returns True if a new copy was fetched.
    A failed download keeps the copy already on disk."""
    path = Path(config.DB_PATH)
    if not config.DB_URL:
        return False
    if path.exists() and (time.time() - path.stat().st_mtime) < config.DB_MAX_AGE_HOURS * 3600:
        return False
    try:
        resp = requests.get(config.DB_URL, timeout=120)
        resp.raise_for_status()
    except requests.RequestException:
        if path.exists():
            return False
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".download")
    tmp.write_bytes(resp.content)
    tmp.replace(path)
    return True
