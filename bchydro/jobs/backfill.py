"""One-time build of the database from the sources.

    python -m bchydro.jobs.backfill            # everything
    python -m bchydro.jobs.backfill calendar   # one part: calendar | load | weather
"""
import sys
from datetime import date, timedelta

from .. import config, db
from ..ingest import calendar_seed, load, weather


def run_calendar() -> None:
    print(f"calendar rows: {calendar_seed.seed_calendar()}")


def run_load() -> None:
    load.ingest_historical()


def run_weather() -> None:
    today = date.today()
    first = int(config.TRAIN_START[:4])
    # ERA5 lags about 5 days; the forecast archive starts in 2022.
    weather.backfill("observed", first, today.year, (today - timedelta(days=6)).isoformat())
    weather.backfill("forecast_archive", 2022, today.year, (today - timedelta(days=1)).isoformat())


PARTS = {"calendar": run_calendar, "load": run_load, "weather": run_weather}

if __name__ == "__main__":
    db.init_db()
    for name in sys.argv[1:] or PARTS:
        print(f"== {name}")
        PARTS[name]()
