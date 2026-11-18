-- BC Hydro demand forecasting: one SQLite database for every source,
-- forecast and score. All timestamps are ISO-8601 text in UTC
-- ("2026-11-17T08:00:00Z"); Pacific local time is kept beside them for display.

CREATE TABLE IF NOT EXISTS load_hourly (
    ts_utc      TEXT PRIMARY KEY,          -- start of the hour, UTC
    ts_local    TEXT NOT NULL,             -- same instant, America/Vancouver
    load_mw     REAL NOT NULL,
    source_file TEXT NOT NULL,
    ingested_at TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS locations (
    location_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    lat         REAL NOT NULL,
    lon         REAL NOT NULL,
    pop_weight  REAL NOT NULL              -- for the population-weighted temperature
);

-- kind: observed (ERA5) / forecast_archive (as issued, 2022+) / forecast_live.
-- issued_at is '' when the source has no issue time. It is NOT NULL on purpose:
-- SQLite treats NULLs in a key as distinct, which would break the upsert.
CREATE TABLE IF NOT EXISTS weather_hourly (
    ts_utc      TEXT NOT NULL,
    location_id INTEGER NOT NULL REFERENCES locations(location_id),
    kind        TEXT NOT NULL CHECK (kind IN ('observed', 'forecast_archive', 'forecast_live')),
    issued_at   TEXT NOT NULL DEFAULT '',
    temp_c      REAL,
    rh          REAL,
    wind_kmh    REAL,
    cloud_pct   REAL,
    precip_mm   REAL,
    PRIMARY KEY (ts_utc, location_id, kind, issued_at)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS calendar (
    date           TEXT PRIMARY KEY,       -- local date, YYYY-MM-DD
    dow            INTEGER NOT NULL,       -- 0 = Monday
    is_holiday     INTEGER NOT NULL,
    holiday_name   TEXT,
    daylight_hours REAL NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS model_runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    phase        TEXT NOT NULL CHECK (phase IN ('backtest', 'live')),
    model_name   TEXT NOT NULL,            -- baseline / frozen / bias_corrected / retrained
    issued_at    TEXT NOT NULL,            -- when the forecast was made (simulated in backtest)
    train_end    TEXT,
    n_train_rows INTEGER,
    trigger      TEXT,                     -- why a retrain fired, if it did
    accepted     INTEGER,                  -- 1 if the refit replaced the old model
    params_json  TEXT,
    UNIQUE (phase, model_name, issued_at)
);

CREATE TABLE IF NOT EXISTS forecasts (
    run_id        INTEGER NOT NULL REFERENCES model_runs(run_id),
    target_ts_utc TEXT NOT NULL,
    yhat_mw       REAL NOT NULL,
    yhat_lo_mw    REAL,
    yhat_hi_mw    REAL,
    horizon_h     INTEGER,
    PRIMARY KEY (run_id, target_ts_utc)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_forecasts_target ON forecasts(target_ts_utc);

-- One row per ingest attempt; feeds the "Data last updated" stamp.
CREATE TABLE IF NOT EXISTS ingest_log (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    job      TEXT NOT NULL,
    ran_at   TEXT NOT NULL,
    status   TEXT NOT NULL,                -- ok / error
    rows     INTEGER,
    message  TEXT
);

CREATE VIEW IF NOT EXISTS forecast_scores AS
SELECT r.run_id, r.phase, r.model_name, r.issued_at,
       f.target_ts_utc, f.horizon_h,
       f.yhat_mw, f.yhat_lo_mw, f.yhat_hi_mw,
       l.load_mw                                   AS actual_mw,
       l.load_mw - f.yhat_mw                       AS error_mw,
       ABS(l.load_mw - f.yhat_mw) / l.load_mw * 100 AS abs_pct_error
FROM forecasts f
JOIN model_runs r  ON r.run_id = f.run_id
JOIN load_hourly l ON l.ts_utc = f.target_ts_utc;
