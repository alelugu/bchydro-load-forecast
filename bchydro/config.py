"""Settings shared by every job and by the app. Override with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "bchydro.sqlite"))

# Public URL of the published database (a GitHub Release asset). When set, the
# app downloads it instead of expecting a local file. See README, "How the data
# reaches the Space".
DB_URL = os.getenv("DB_URL", "")
DB_MAX_AGE_HOURS = float(os.getenv("DB_MAX_AGE_HOURS", "6"))

LOCAL_TZ = "America/Vancouver"


def check_tz() -> None:
    """BC moved to permanent daylight time (UTC-7 all year) in 2026. An old time-zone
    database still 'falls back' on 1 Nov 2026 and would shift every later hour by one."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    offset = datetime(2026, 12, 1, 12, tzinfo=ZoneInfo(LOCAL_TZ)).utcoffset().total_seconds() / 3600
    if offset != -7:
        raise RuntimeError("Time-zone database is out of date for BC (expected UTC-7 in "
                           "December 2026). Run: pip install -U tzdata")

# BC Hydro load files. Copy the exact links from:
#   https://www.bchydro.com/energy-in-bc/operations/transmission/transmission-system/balancing-authority-load-data.html
#   (current hourly file) and its "historical transmission data" page (yearly files).
BCH_CURRENT_HOURLY_URL = os.getenv("BCH_CURRENT_HOURLY_URL", "")  # TODO: paste the link
BCH_HISTORICAL_URLS: dict[int, str] = {
    # 2015: "https://www.bchydro.com/....xls",  TODO: paste one link per year, 2015-2026
}

# Weather locations: (id, name, lat, lon, population weight). Weights are rough
# metro-area shares and only feed the population-weighted temperature feature.
LOCATIONS = [
    (1, "Vancouver", 49.2827, -123.1207, 0.72),
    (2, "Victoria", 48.4284, -123.3656, 0.11),
    (3, "Kelowna", 49.8880, -119.4960, 0.07),
    (4, "Prince George", 53.9171, -122.7497, 0.03),
]

WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                "cloud_cover", "precipitation"]
OPEN_METEO = {
    "observed": "https://archive-api.open-meteo.com/v1/archive",
    "forecast_archive": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "forecast_live": "https://api.open-meteo.com/v1/forecast",
}

TRAIN_START = "2015-01-01"
CUTOFF = "2025-11-15"        # frozen model trains up to here; replay starts the next day
RECORD_PEAK_MW = 11300
