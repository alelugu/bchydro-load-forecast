---
title: BC Hydro Demand Forecast
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.32.0
python_version: "3.13"
app_file: app.py
pinned: false
---

# BC Hydro Demand Forecasting Dashboard

CPSC 4820 Data Visualization. A live Gradio dashboard that forecasts tomorrow's hourly
electricity demand on the BC Hydro grid and shows the forecast correcting itself as new
data arrives. Audience: BC Hydro grid operations and load-forecasting analysts.

Status: data pipeline, model, year-long replay, daily live forecast and dashboard are all running.

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

The database is one SQLite file, `data/bchydro.sqlite` (about 85 MB, with the trained models and every forecast). It is not stored in
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
| `python -m bchydro.modeling.validate` | Compares the models on 2024 (trained 2015-2023) and stores the results |
| `python -m bchydro.modeling.forecast_job train` | Fits the frozen model on data up to the cutoff (15 Nov 2025) |
| `python -m bchydro.modeling.forecast_job replay` | Replays every day from the cutoff to the latest actual (about 75 minutes on 2 CPU cores) |
| `python -m bchydro.modeling.forecast_job live` | Today's forecast for tomorrow plus the 7-day outlook (the daily job runs this) |
| `python app.py` | The dashboard, at http://127.0.0.1:7860 |
| `pytest -q` | Checks on clock handling, data availability, leakage and correction rules |

To rebuild the model results from scratch, run the first three model commands in order:
`validate`, `train`, `replay`. Each one replaces its own previous output.

## Project layout

```
bchydro/
  config.py                 paths, URLs, locations, cutoff date
  schema.sql                every table and the forecast_scores view
  db.py                     connect, create tables, upsert
  ingest/load.py            BC Hydro .xls files -> load_hourly
  ingest/weather.py         Open-Meteo -> weather_hourly
  ingest/calendar_seed.py   holidays and daylight hours -> calendar
  modeling/features.py      features (one function for training, replay and live)
  modeling/models.py        baseline, linear explainer, LightGBM + quantiles, model storage
  modeling/validate.py      model selection on 2024
  modeling/correction.py    bias correction, retrain triggers, band recalibration
  modeling/forecast_job.py  the forecast loop: train, replay, live
  dashboard.py              data and charts for the app
app.py                      Gradio dashboard
tests/                      pytest checks
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
| `model_store` | `name` | Trained models (`frozen`, `champion`, `outlook`) and the 2024 validation results |
| `forecast_scores` (view) | | Joins forecasts to actuals and computes the error |

## Daily automation

`.github/workflows/daily.yml` runs twice each morning, at about 09:15 and 12:45 Pacific (GitHub can start
scheduled jobs late). Each run:

1. downloads the database from the `data` release of this repository,
2. runs `python -m bchydro.jobs.daily`,
3. checks that the database is complete, then uploads it back to the release,
4. commits the day's raw BC Hydro file to `data/raw/`.

If the database cannot be downloaded or fails the check, nothing is published.

## Model

Each issue day at 09:30 Pacific forecasts the 24 hours of the next day in four versions:

| Version (`model_name`) | What it is |
|---|---|
| `baseline` | Same hour one week earlier |
| `frozen` | LightGBM trained once on 2015 to 15 Nov 2025; never changes |
| `bias_corrected` | Frozen model + hourly bias correction + band recalibration |
| `retrained` | Champion model, refit weekly or on a trigger, + its own bias correction: the version operators see |

**Features.** Calendar (hour, weekday, BC holidays, Christmas period, season, daylight), temperature in the
four cities and population-weighted, heating and cooling degree-hours, daily HDD/CDD, HDD over the previous
3 days, wind × cold, cloud, rain, humidity, and load 7, 14, 21 and 28 days earlier plus older averages. Load
is never newer than 7 days, because BC Hydro posts nothing on weekends. Training uses archived weather
forecasts where they exist (2022 on), matching what the model sees when forecasting.

**Model selection (2024, trained 2015-2023, forecast weather).** LightGBM 3.22% MAPE, linear degree-day
model 4.52%, same hour last week 5.05%. The raw 10-90% quantile band covered only 46% of hours out of
sample, so it is widened 2.3× (the width that covered 80% in 2024) and recalibrated as the replay runs.

**Correction rules** (`bchydro/modeling/correction.py`):

- Bias: per local hour, the exponentially weighted mean error of the last 28 known days (7-day half-life),
  from 3 days of errors on, capped at ±3% of the forecast.
- Retrain triggers, in this order: a daily peak off by more than 5%; daily mean error with the same sign for
  5 days averaging more than 2% (refit weights the last 60 days 3×); rolling 7-day MAPE above 1.25 × the
  validation MAPE on 2 runs in a row; otherwise every Sunday. Unscheduled triggers wait 3 days after a retrain.
- A refit is trained on data up to 14 days before the newest actual and must beat the current model on those
  14 days; only then is it refit on all data and promoted. Otherwise it is logged as rejected.
- Band: every 7 days, if 14-day coverage is outside 70-90%, the width is scaled toward 80% (at most ±50%).

**Replay.** `forecast_job replay` issues a forecast for every day from 15 Nov 2025 to the day before the
latest actual. On each simulated day it sees only the actuals BC Hydro would have published by 09:30 that
day: yesterday on weekdays, Thursday on weekends, with Monday's update covering Friday to Sunday.

## Dashboard

`python app.py`. Five tabs: **Tomorrow** (hourly forecast and band, alert level, actions, what-if
temperature slider), **Peak risk (7 days)**, **Explore history**, **Prediction correction** (accuracy by
version, rolling error with retrain markers, error heatmap, alert precision/recall/F1, model log) and
**About**. The Watch and High sliders at the top set the alert thresholds as percentiles of past winter
daily peaks. Two buttons download what is on screen as Excel files: the hourly forecast with the 7-day
outlook, and the correction results (accuracy, daily errors, live days, alert accuracy, model log).

The job runs more than once a day, so the dashboard and the correction logic keep one forecast per version
and day: the latest one issued, and a live forecast over a replayed one.

### Hosting on Hugging Face Spaces

1. Create a Gradio Space and push this repository to it (the header at the top of this README is the
   Space's configuration).
2. In the Space's settings, add the variable `DB_URL` =
   `https://github.com/alelugu/bchydro-load-forecast/releases/download/data/bchydro.sqlite`.
3. The app downloads the database at start-up and again whenever its copy is more than 6 hours old
   (`DB_MAX_AGE_HOURS`), so it shows each day's new forecast without a redeploy.

Free accounts can currently host Gradio Spaces only on ZeroGPU hardware; the app needs no GPU.

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
