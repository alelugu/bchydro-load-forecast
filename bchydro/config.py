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

# BC Hydro load files, from "Balancing Authority Load Data" on bchydro.com.
_BCH = ("https://www.bchydro.com/content/dam/BCHydro/customer-portal/documents/corporate/"
        "suppliers/transmission-system/balancing_authority_load_data/")
BCH_CURRENT_HOURLY_URL = os.getenv("BCH_CURRENT_HOURLY_URL", _BCH + "CurrentHourlyBALoad.xls")
_HIST = _BCH + "Historical%20Transmission%20Data/"
# BC Hydro's file names are not uniform: 2024 onward have a space before the year.
BCH_HISTORICAL_URLS: dict[int, str] = {
    **{y: f"{_HIST}BalancingAuthorityLoad{y}.xls" for y in range(2015, 2024)},
    2024: _HIST + "BalancingAuthorityLoad%202024.xls",
    2025: _HIST + "BalancingAuthorityLoad%202025.xls",
    2026: _BCH + "BalancingAuthorityLoad%202026.xls",   # current year sits one folder up
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
