"""Features for the hourly load model. One function builds them for training, the
year-long replay and live forecasts, so the three cannot drift apart.

Rule: a feature may only use what an operator has at 09:30 on the issue day.
Every load feature is at least 7 days older than the hour being forecast (BC Hydro
posts yesterday's data on weekdays and nothing on weekends), and the weather is the
forecast, not what actually happened, except when training.

    target hour T (local)  ->  calendar of T
                               weather at T and over the 3 days before T
                               load at T-7d, T-14d, T-21d, T-28d and daily means 8+ days back
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .. import config

BASE_TEMP_C = 18.0
CITY = {1: "van", 2: "vic", 3: "kel", 4: "pg"}
WEIGHTS = {i: w for i, _n, _la, _lo, w in config.LOCATIONS}

# Which weather a target hour uses, in order of preference.
#   train  : forecasts as issued where they exist (2022 on), else what happened (ERA5).
#            Training on forecast weather matches what the model sees when forecasting;
#            on 2024 it cut MAPE from 3.44% to 3.22% versus training on ERA5 alone.
#   replay : forecasts as issued (Open-Meteo archive), as an operator would have had
#   live   : today's 7-day forecast, falling back to the archive for past hours
WEATHER_PREFERENCE = {
    "train": ["forecast_archive", "observed"],
    "replay": ["forecast_archive", "observed"],
    "live": ["forecast_live", "forecast_archive", "observed"],
}

FEATURES = [
    # calendar
    "hour", "hour_sin", "hour_cos", "dow", "is_weekend", "is_holiday", "near_holiday",
    "xmas", "doy_sin", "doy_cos", "daylight_hours",
    # temperature and degree days
    "temp_van", "temp_vic", "temp_kel", "temp_pg", "temp_w",
    "hdh", "cdh", "hdd_day", "cdd_day", "hdd_3d", "temp_w_24h",
    # other weather (Vancouver, where most of the load is)
    "wind_x_hdh", "cloud", "precip", "rh",
    # load history (all at least 7 days back)
    "lag_7d", "lag_14d", "lag_21d", "lag_28d", "same_hour_4wk", "mean_8_14d",
    "growth_ratio",
]


# ---------------------------------------------------------------- loading
def load_series(con: sqlite3.Connection) -> pd.Series:
    """Hourly load in MW, indexed by tz-aware UTC timestamps."""
    df = pd.read_sql("SELECT ts_utc, load_mw FROM load_hourly ORDER BY ts_utc", con)
    s = df.set_index(pd.to_datetime(df["ts_utc"], utc=True))["load_mw"].astype(float)
    return s


def calendar_frame(con: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("SELECT date, is_holiday, daylight_hours FROM calendar", con)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.set_index("date")


def weather_frame(con: sqlite3.Connection, start: pd.Timestamp, end: pd.Timestamp,
                  mode: str, issued_before: str | None = None) -> pd.DataFrame:
    """One row per UTC hour in [start, end] with each city's temperature and
    Vancouver's other variables, using the weather kinds `mode` allows.
    For live forecasts, only forecasts issued before `issued_before` are used."""
    kinds = WEATHER_PREFERENCE[mode]
    q = ("SELECT ts_utc, location_id, kind, issued_at, temp_c, rh, wind_kmh, cloud_pct, precip_mm "
         "FROM weather_hourly WHERE ts_utc BETWEEN ? AND ? AND kind IN (%s)" % ",".join("?" * len(kinds)))
    w = pd.read_sql(q, con, params=[_iso(start), _iso(end), *kinds])
    if issued_before is not None:
        w = w[(w["kind"] != "forecast_live") | (w["issued_at"] <= issued_before)]
    w["rank"] = w["kind"].map({k: i for i, k in enumerate(kinds)})
    # newest live issue first within the same kind
    w = w.sort_values(["ts_utc", "location_id", "rank", "issued_at"], ascending=[True, True, True, False])
    w = w.drop_duplicates(["ts_utc", "location_id"])
    w["ts"] = pd.to_datetime(w["ts_utc"], utc=True)
    temp = w.pivot(index="ts", columns="location_id", values="temp_c").rename(columns=lambda c: f"temp_{CITY[c]}")
    van = w[w["location_id"] == 1].set_index("ts")[["rh", "wind_kmh", "cloud_pct", "precip_mm"]]
    out = temp.join(van, how="outer").sort_index()
    out["temp_w"] = sum(out[f"temp_{CITY[i]}"] * WEIGHTS[i] for i in CITY) / sum(WEIGHTS.values())
    return out


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- targets
def local_day_hours(day) -> pd.DatetimeIndex:
    """The UTC start of every local hour of one local date (23, 24 or 25 hours)."""
    d0 = pd.Timestamp(day).tz_localize(config.LOCAL_TZ)
    d1 = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(config.LOCAL_TZ)
    return pd.date_range(d0.tz_convert("UTC"), d1.tz_convert("UTC"), freq="h", inclusive="left")


def hours_between(first_day, last_day) -> pd.DatetimeIndex:
    days = pd.date_range(first_day, last_day, freq="D")
    return local_day_hours(days[0]).append([local_day_hours(d) for d in days[1:]]) if len(days) else pd.DatetimeIndex([])


# ---------------------------------------------------------------- features
def build(target_utc: pd.DatetimeIndex, load: pd.Series, weather: pd.DataFrame,
          cal: pd.DataFrame, temp_shift_c: float = 0.0) -> pd.DataFrame:
    """Feature table for the given target hours. `load` may contain any history: only
    values 7+ days before each target are read. `temp_shift_c` moves every temperature
    by that many degrees (the dashboard's what-if slider)."""
    idx = pd.DatetimeIndex(target_utc)
    loc = idx.tz_convert(config.LOCAL_TZ)
    f = pd.DataFrame(index=idx)

    # calendar
    f["hour"] = loc.hour
    f["hour_sin"] = np.sin(2 * np.pi * loc.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * loc.hour / 24)
    f["dow"] = loc.dayofweek
    f["is_weekend"] = (loc.dayofweek >= 5).astype(int)
    dates = pd.Index(loc.date)
    hol = cal["is_holiday"]
    f["is_holiday"] = dates.map(hol).fillna(0).astype(int).values
    prev_hol = pd.Index([d - pd.Timedelta(days=1) for d in loc.normalize()]).date
    next_hol = pd.Index([d + pd.Timedelta(days=1) for d in loc.normalize()]).date
    f["near_holiday"] = (pd.Index(prev_hol).map(hol).fillna(0).values + pd.Index(next_hol).map(hol).fillna(0).values > 0).astype(int)
    f["xmas"] = (((loc.month == 12) & (loc.day >= 24)) | ((loc.month == 1) & (loc.day == 1))).astype(int)
    doy = loc.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["daylight_hours"] = dates.map(cal["daylight_hours"]).values

    # weather at the target hour
    w = weather.copy()
    for c in ["temp_van", "temp_vic", "temp_kel", "temp_pg", "temp_w"]:
        w[c] = w[c] + temp_shift_c
    w["hdh"] = (BASE_TEMP_C - w["temp_w"]).clip(lower=0)
    w["cdh"] = (w["temp_w"] - BASE_TEMP_C).clip(lower=0)
    w["temp_w_24h"] = w["temp_w"].rolling(24, min_periods=12).mean()
    wloc = w.index.tz_convert(config.LOCAL_TZ)
    daily_t = w["temp_w"].groupby(wloc.date).mean()
    hdd = (BASE_TEMP_C - daily_t).clip(lower=0)
    cdd = (daily_t - BASE_TEMP_C).clip(lower=0)
    hdd_prev3 = hdd.rolling(3, min_periods=2).sum().shift(1)   # the 3 days before

    wt = w.reindex(idx)
    for c in ["temp_van", "temp_vic", "temp_kel", "temp_pg", "temp_w", "hdh", "cdh", "temp_w_24h"]:
        f[c] = wt[c].values
    f["hdd_day"] = dates.map(hdd).values
    f["cdd_day"] = dates.map(cdd).values
    f["hdd_3d"] = dates.map(hdd_prev3).values
    f["wind_x_hdh"] = (wt["wind_kmh"] * wt["hdh"]).values
    f["cloud"] = wt["cloud_pct"].values
    f["precip"] = wt["precip_mm"].values
    f["rh"] = wt["rh"].values

    # load history, by elapsed hours (UTC), so clock changes cannot misalign it
    for d in (7, 14, 21, 28):
        f[f"lag_{d}d"] = load.reindex(idx - pd.Timedelta(days=d)).values
    f["same_hour_4wk"] = f[["lag_7d", "lag_14d", "lag_21d", "lag_28d"]].mean(axis=1)
    daily_load = load.groupby(load.index.tz_convert(config.LOCAL_TZ).date).mean()
    daily_load.index = pd.to_datetime(daily_load.index)
    roll7 = daily_load.rolling(7, min_periods=5).mean()          # mean of days d-6..d
    roll28 = daily_load.rolling(28, min_periods=21).mean()
    tday = pd.to_datetime(pd.Index(loc.date))
    f["mean_8_14d"] = roll7.reindex(tday - pd.Timedelta(days=8)).values
    recent = roll28.reindex(tday - pd.Timedelta(days=8)).values
    year_ago = roll28.reindex(tday - pd.Timedelta(days=8 + 364)).values
    f["growth_ratio"] = recent / year_ago
    return f[FEATURES]


def training_table(con: sqlite3.Connection, first_day: str, last_day: str, mode: str = "train",
                   load: pd.Series | None = None, cal: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features plus the actual load (`y`) for every hour of [first_day, last_day]."""
    load = load if load is not None else load_series(con)
    cal = cal if cal is not None else calendar_frame(con)
    idx = hours_between(first_day, last_day)
    wx = weather_frame(con, idx[0] - pd.Timedelta(days=4), idx[-1], mode)
    X = build(idx, load, wx, cal)
    X["y"] = load.reindex(idx).values
    return X
