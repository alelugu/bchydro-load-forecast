"""Fill the calendar table: day of week, BC statutory holidays, daylight hours."""
import math
from datetime import date, timedelta

import holidays

from .. import db

VANCOUVER_LAT = 49.28


def daylight_hours(d: date, lat: float = VANCOUVER_LAT) -> float:
    """Approximate day length from the solar declination (good to a few minutes)."""
    decl = math.radians(23.44) * math.sin(2 * math.pi * (d.timetuple().tm_yday - 81) / 365)
    x = -math.tan(math.radians(lat)) * math.tan(decl)
    return round(24 / math.pi * math.acos(max(-1.0, min(1.0, x))), 2)


def seed_calendar(start_year: int = 2015, end_year: int = 2027) -> int:
    bc = holidays.Canada(subdiv="BC", years=range(start_year, end_year + 1))
    rows, d = [], date(start_year, 1, 1)
    while d.year <= end_year:
        rows.append(dict(date=d.isoformat(), dow=d.weekday(), is_holiday=int(d in bc),
                         holiday_name=bc.get(d), daylight_hours=daylight_hours(d)))
        d += timedelta(days=1)
    with db.connect() as con:
        return db.upsert(con, "calendar", rows)


if __name__ == "__main__":
    db.init_db()
    print(f"calendar rows: {seed_calendar()}")
