"""Daily job (GitHub Actions, 09:30 Pacific, retry at 13:30).

    python -m bchydro.jobs.daily

Each step is independent: a failure in one is logged and the rest still run, so a
late BC Hydro file never costs us the day's weather forecast. Every step is an
upsert, so running the job twice in a day changes nothing.
"""
import sys
from datetime import datetime, timedelta, timezone

from .. import db
from ..ingest import load, weather


def _day(offset: int) -> str:
    return (datetime.now(timezone.utc).date() - timedelta(days=offset)).isoformat()


def step(name: str, fn) -> bool:
    try:
        print(f"[ok]    {name}: {fn()}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001 - log and continue with the other steps
        print(f"[error] {name}: {e}", flush=True)
        with db.connect() as con:
            db.log_ingest(con, name, "error", 0, str(e)[:500])
        return False


def main() -> int:
    db.init_db()
    ok = [
        # BC Hydro: the "current" file holds the month to date, so a missed day is recovered.
        step("load_current", lambda: load.ingest_file(load.download_current())),
        # Next 7 days, stamped with the hour it was issued: the input for live forecasts.
        step("weather_forecast_live", lambda: weather.ingest("forecast_live")),
        # Observed weather is published about 5 days late; re-read a window to fill it in.
        step("weather_observed", lambda: weather.ingest("observed", _day(12), _day(6))),
        step("weather_forecast_archive", lambda: weather.ingest("forecast_archive", _day(4), _day(1))),
        # Model owner adds here: score yesterday, correct, forecast tomorrow.
        #   step("forecast", lambda: forecast_job.run(phase="live")),
    ]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
