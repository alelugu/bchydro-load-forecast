# bchydro-load-forecast
# BC Hydro Demand Forecasting Dashboard

CPSC 4820 Data Visualization. A live Gradio dashboard that forecasts tomorrow's hourly
electricity demand on the BC Hydro grid and shows the forecast correcting itself as new
data arrives. Audience: BC Hydro grid operations and load-forecasting analysts.

## Setup

Python 3.13.

```bash
git clone <repo-url> && cd bchydro-load-forecast
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
make install        # pip install -r requirements.txt
make test
```

## Run

| Step | Command | What it does |
|---|---|---|
| 1 | `make init` | Creates `data/bchydro.sqlite` with the schema, locations and calendar |
| 2 | `make backfill` | Downloads BC Hydro load and Open-Meteo weather from 2015 and loads them |
| 3 | `make daily` | The daily job: new actuals, live weather forecast, tomorrow's forecast |
| 4 | `make app` | Starts the dashboard at http://127.0.0.1:7860 |

Without `make`, run the commands inside the `Makefile` directly (for example
`python -m bchydro.jobs.backfill`).

## Project layout

```
app.py                      Gradio dashboard (reads the database only)
bchydro/
  config.py                 paths, URLs, locations, cutoff date
  schema.sql                every table and the forecast_scores view
  db.py                     connect, init, upsert, download the published database
  ingest/load.py            BC Hydro .xls files -> load_hourly
  ingest/weather.py         Open-Meteo -> weather_hourly
  ingest/calendar_seed.py   holidays and daylight hours -> calendar
  modeling/                 features, models, correction, replay (see its README)
  jobs/backfill.py          one-time build
  jobs/daily.py             daily job
data/raw/                   BC Hydro files exactly as downloaded (committed)
tests/
.github/workflows/          daily.yml (scheduled job), ci.yml (tests)
```

## Database

One SQLite file, `data/bchydro.sqlite`. Times are stored in UTC as text
(`2026-11-17T08:00:00Z`), with Pacific local time beside them for display.

| Table | Key | Written by |
|---|---|---|
| `load_hourly` | `ts_utc` | load ingester |
| `locations` | `location_id` | `init_db` |
| `weather_hourly` | `ts_utc, location_id, kind, issued_at` | weather ingester |
| `calendar` | `date` | calendar seed |
| `model_runs` | `run_id` | forecast job / replay |
| `forecasts` | `run_id, target_ts_utc` | forecast job / replay |
| `ingest_log` | `id` | every job |
| `forecast_scores` (view) | | joins forecasts to actuals |

Every write is an upsert on the key, so rerunning a job never duplicates rows.

## How the data reaches the Space

SQLite is a file, so the scheduled job and the hosted app need a shared copy:

1. GitHub Actions (`daily.yml`, 09:30 and 13:30 Pacific) downloads the database from the
   `data` release of this repo, runs the daily job, and uploads it back to that release.
2. The Space sets `DB_URL` to the release asset. The app downloads it on start and again
   whenever its copy is older than `DB_MAX_AGE_HOURS`.

The database itself is not committed. Raw BC Hydro files are, in `data/raw/`.

## One-time repository setup

- Settings -> Actions -> General -> Workflow permissions: **Read and write**.
- Settings -> Secrets and variables -> Actions -> Variables: add `BCH_CURRENT_HOURLY_URL`
  with the link to BC Hydro's "Current Hourly" file.
- Paste the yearly file links into `BCH_HISTORICAL_URLS` in `bchydro/config.py`.
- Run the `daily` workflow once by hand (Actions -> daily -> Run workflow).

## Time zone

BC moved to permanent daylight time in 2026: UTC-7 all year, with no clock change on
1 November 2026. Machines with an old time-zone database get this wrong by an hour;
`config.check_tz()` and the tests fail loudly when that happens (`pip install -U tzdata`).

## Data sources

- BC Hydro, Balancing Authority Load Data (hourly gross telemetered load)
- Open-Meteo: Historical Weather, Historical Forecast and Forecast APIs
- Python `holidays` package (BC statutory holidays)


