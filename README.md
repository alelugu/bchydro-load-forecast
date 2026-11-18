# BC Hydro Demand Forecasting Dashboard

CPSC 4820 Data Visualization. A live Gradio dashboard that forecasts tomorrow's hourly
electricity demand on the BC Hydro grid and shows the forecast correcting itself as new
data arrives. Audience: BC Hydro grid operations and load-forecasting analysts.

Status: data pipeline and daily automation are running. Model and dashboard are in progress.

## Setup

Python 3.13.

```bash
git clone https://github.com/alelugu/bchydro-load-forecast.git
cd bchydro-load-forecast

# with conda
conda create -n bchydro python=3.13 -y
conda activate bchydro

# or with venv
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

## Get the database

The database is one SQLite file, `data/bchydro.sqlite` (about 65 MB). It is not stored in
git. Use either option:

**A. Download it (seconds).** Save this file into the `data` folder:

https://github.com/alelugu/bchydro-load-forecast/releases/download/data/bchydro.sqlite

**B. Rebuild it from the sources (about 15 minutes).**

```bash
python -m bchydro.jobs.backfill
```

This downloads the BC Hydro yearly files and the Open-Meteo weather from 2015, and builds
the calendar. It can be run again at any time: every write replaces rows by key, so nothing
is duplicated.

## Commands

| Command | What it does |
|---|---|
| `python -m bchydro.jobs.backfill` | Builds the whole database (`calendar`, `load`, `weather` can be given to run one part) |
| `python -m bchydro.jobs.daily` | The daily job: new BC Hydro actuals, 7-day weather forecast, recent weather |
| `python -m bchydro.ingest.load --historical` | Loads every BC Hydro yearly file |
| `python -m bchydro.ingest.load` | Downloads and loads BC Hydro's current-month file |

## Project layout

```
bchydro/
  config.py                 paths, URLs, locations, cutoff date
  schema.sql                every table and the forecast_scores view
  db.py                     connect, create tables, upsert
  ingest/load.py            BC Hydro .xls files -> load_hourly
  ingest/weather.py         Open-Meteo -> weather_hourly
  ingest/calendar_seed.py   holidays and daylight hours -> calendar
  modeling/                 features, models, correction, replay
  jobs/backfill.py          one-time build
  jobs/daily.py             daily job
data/raw/                   BC Hydro files exactly as downloaded (committed)
.github/workflows/daily.yml scheduled job
```

## Database

Times are stored in UTC as text (`2026-11-17T08:00:00Z`), with Pacific local time beside
them for display.

| Table | Key | Contents |
|---|---|---|
| `load_hourly` | `ts_utc` | BC Hydro hourly load in MW, 2015 to yesterday |
| `locations` | `location_id` | Vancouver, Victoria, Kelowna, Prince George, with population weights |
| `weather_hourly` | `ts_utc, location_id, kind, issued_at` | `observed` (2015 on), `forecast_archive` (2022 on), `forecast_live` (next 7 days) |
| `calendar` | `date` | Day of week, BC holidays, daylight hours |
| `model_runs` | `run_id` | One row per model per forecast day |
| `forecasts` | `run_id, target_ts_utc` | Hourly forecasts with their range |
| `ingest_log` | `id` | One row per ingest run |
| `forecast_scores` (view) | | Joins forecasts to actuals and computes the error |

## Daily automation

`.github/workflows/daily.yml` runs at 09:30 and 13:30 Pacific. Each run:

1. downloads the database from the `data` release of this repository,
2. runs `python -m bchydro.jobs.daily`,
3. checks that the database is complete, then uploads it back to the release,
4. commits the day's raw BC Hydro file to `data/raw/`.

If the database cannot be downloaded or fails the check, nothing is published.

## Data notes

- **Time zone.** BC moved to permanent daylight time in 2026: UTC-7 all year, with no clock
  change on 1 November 2026. `config.check_tz()` stops the load if the machine's time-zone
  database is too old to know this (`pip install -U tzdata`).
- **BC Hydro file formats.** The yearly files changed layout over the years. On the March
  clock-change day the skipped hour is left out, set to 0 MW, or left blank, depending on
  the year; on the November day (up to 2025) the repeated hour is labelled `2*`. The loader
  places hours by their order within the day and rejects any day with the wrong number of
  hours.
- **Weather lag.** Observed weather is published about five days late; the daily job fills
  it in as it arrives.

## Data sources

- BC Hydro, Balancing Authority Load Data (hourly gross telemetered load)
- Open-Meteo: Historical Weather, Historical Forecast and Forecast APIs
- Python `holidays` package (BC statutory holidays)
