"""Open-Meteo weather for the four cities -> weather_hourly.

kind = observed          ERA5 reanalysis, training only (about a 5-day lag)
       forecast_archive  forecasts as they were issued, 2022 onward, for validation and replay
       forecast_live     the next 7 days, for live forecasts

    python -m bchydro.ingest.weather          # quick test: fetch the live 7-day forecast
"""
import time

import requests

from .. import config, db


def fetch(kind: str, lat: float, lon: float, start: str | None = None, end: str | None = None) -> list[dict]:
    """Return hourly rows (UTC) for one location. Dates are YYYY-MM-DD; live needs none."""
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(config.WEATHER_VARS),
              "timezone": "GMT", "wind_speed_unit": "kmh"}
    if kind == "forecast_live":
        params["forecast_days"] = 9      # 9 UTC days = 7 whole Pacific days after today
    else:
        params |= {"start_date": start, "end_date": end}
    resp = None
    for attempt in range(4):
        try:
            resp = requests.get(config.OPEN_METEO[kind], params=params, timeout=45)
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 3:                 # the server did not answer four times: give up
                raise
            time.sleep(10 * (attempt + 1))
            continue
        if resp.status_code not in (429, 502, 503, 504):
            break
        time.sleep(65 if resp.status_code == 429 else 15)   # 429 = free-tier rate limit
    resp.raise_for_status()
    time.sleep(2)                            # stay well under the per-minute limit
    h = resp.json()["hourly"]
    return [dict(ts_utc=t + ":00Z", temp_c=h["temperature_2m"][i], rh=h["relative_humidity_2m"][i],
                 wind_kmh=h["wind_speed_10m"][i], cloud_pct=h["cloud_cover"][i],
                 precip_mm=h["precipitation"][i])
            for i, t in enumerate(h["time"])]


def ingest(kind: str, start: str | None = None, end: str | None = None) -> int:
    """Fetch one kind for every location and upsert it. Live rows carry their issue time."""
    issued_at = db.utcnow()[:13] + ":00:00Z" if kind == "forecast_live" else ""
    total = 0
    with db.connect() as con:
        for loc_id, _name, lat, lon, _w in config.LOCATIONS:
            rows = fetch(kind, lat, lon, start, end)
            for r in rows:
                r |= dict(location_id=loc_id, kind=kind, issued_at=issued_at)
            total += db.upsert(con, "weather_hourly", rows)
        db.log_ingest(con, f"weather_{kind}", "ok", total)
    return total


def backfill(kind: str, first_year: int, last_year: int, last_day: str) -> int:
    """One request per city per year, to stay well inside the free tier."""
    total = 0
    for y in range(first_year, last_year + 1):
        end = min(f"{y}-12-31", last_day)
        total += ingest(kind, f"{y}-01-01", end)
        print(f"  {kind} {y}: done")
    return total


if __name__ == "__main__":
    db.init_db()
    print(f"forecast_live rows: {ingest('forecast_live')}")
