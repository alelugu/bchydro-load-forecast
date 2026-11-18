"""BC Hydro hourly load: download the .xls files, archive them, load them into load_hourly.

    python -m bchydro.ingest.load data/raw/BalancingAuthorityLoad_2026.xls   # load a file
    python -m bchydro.ingest.load                                            # download + load "current"

File layout (checked on the 2026 and "current" files, 17 Nov 2026):
  - a few title rows, then a header row: "Date", "HE", "Control Area Load"
  - Date is text MM/DD/YYYY in Pacific local time; HE is hour-ending 1..24; load is MW
  - the yearly file covers whole months; the "current" file holds the month to date
  - 8 Mar 2026 (clocks forward) has 23 rows, with HE 2 left out
So hours are placed by their order within the day, not by the HE label: the first row of a
day is local midnight and each next row is one hour later. That is right on 23-, 24- and
25-hour days as long as the day has the number of rows the calendar says it should.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .. import config, db


def download(url: str, name: str) -> Path:
    """Save a BC Hydro file under data/raw/ and return its path."""
    if not url:
        raise ValueError("BC Hydro URL not set: see bchydro/config.py")
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (course project)"})
    resp.raise_for_status()
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / name
    path.write_bytes(resp.content)
    return path


def download_current() -> Path:
    """Fetch the 'current hourly' file, stamped with the download date so each day's
    copy is kept."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return download(config.BCH_CURRENT_HOURLY_URL, f"current_hourly_{stamp}.xls")


def download_historical() -> list[Path]:
    return [download(url, f"BalancingAuthorityLoad_{year}.xls")
            for year, url in sorted(config.BCH_HISTORICAL_URLS.items())]


def parse_load_xls(path: Path) -> pd.DataFrame:
    """Return one row per hour: ts_utc (start of the hour, tz-aware) and load_mw.
    Raises ValueError if the layout or a day's row count is not what we expect."""
    config.check_tz()
    raw = pd.read_excel(path, header=None, dtype=object)
    is_header = (raw[0].astype(str).str.strip() == "Date") & (raw[1].astype(str).str.strip() == "HE")
    if not is_header.any():
        raise ValueError(f"{path.name}: header row 'Date | HE | Control Area Load' not found")
    df = raw.iloc[is_header.idxmax() + 1:, :3].dropna(how="all")
    df.columns = ["date", "he", "load_mw"]

    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%m/%d/%Y")
    df["load_mw"] = pd.to_numeric(df["load_mw"], errors="raise").astype(float)
    if df["load_mw"].isna().any() or (df["load_mw"] <= 0).any():
        raise ValueError(f"{path.name}: missing or non-positive load values")
    if not df["date"].is_monotonic_increasing:
        raise ValueError(f"{path.name}: dates are not in order")

    midnight = df["date"].dt.tz_localize(config.LOCAL_TZ).dt.tz_convert("UTC")
    next_midnight = (df["date"] + pd.Timedelta(days=1)).dt.tz_localize(config.LOCAL_TZ).dt.tz_convert("UTC")
    expected = ((next_midnight - midnight) / pd.Timedelta(hours=1)).astype(int)
    position = df.groupby("date").cumcount()
    rows_in_day = df.groupby("date")["he"].transform("size")

    last_day = df["date"].max()          # the newest day may still be incomplete
    wrong = df[(rows_in_day != expected) & (df["date"] != last_day)]["date"].dt.date.unique()
    if len(wrong):
        raise ValueError(f"{path.name}: unexpected number of hours on {list(wrong)[:5]}")

    df["ts_utc"] = midnight + pd.to_timedelta(position, unit="h")
    return df[["ts_utc", "load_mw"]].reset_index(drop=True)


def ingest_file(path: Path) -> int:
    df = parse_load_xls(Path(path))
    local = df["ts_utc"].dt.tz_convert(config.LOCAL_TZ)
    now = db.utcnow()
    rows = [dict(ts_utc=u.strftime("%Y-%m-%dT%H:%M:%SZ"), ts_local=l.strftime("%Y-%m-%dT%H:%M:%S"),
                 load_mw=mw, source_file=Path(path).name, ingested_at=now)
            for u, l, mw in zip(df["ts_utc"], local, df["load_mw"])]
    with db.connect() as con:
        n = db.upsert(con, "load_hourly", rows)
        db.log_ingest(con, "load", "ok", n, Path(path).name)
    return n


if __name__ == "__main__":
    db.init_db()
    for f in [Path(a) for a in sys.argv[1:]] or [download_current()]:
        print(f"{f.name}: {ingest_file(f)} rows")
