"""How forecasts correct themselves. Pure functions on a history table of past forecasts
with their actuals, so the replay (in memory) and the live job (from the database) run
exactly the same logic.

history columns: model_name, issued_day, target (UTC), local_day, hour (local),
                 raw, yhat, lo, hi, actual, train_end
Thresholds are the starting values from the project definition, tuned on 2024.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BIAS_HALF_LIFE_DAYS = 7
BIAS_WINDOW_DAYS = 28
BIAS_MIN_DAYS = 3
BIAS_CAP = 0.03                 # +-3% of the raw forecast

DRIFT_FACTOR = 1.25             # rolling 7-day MAPE above 1.25 x validation MAPE ...
DRIFT_RUNS = 2                  # ... on 2 runs in a row
PERSIST_DAYS = 5                # daily mean error with the same sign 5 days running ...
PERSIST_PCT = 2.0               # ... averaging more than 2%
PEAK_MISS_PCT = 5.0             # a daily peak off by more than 5%
TRIGGER_COOLDOWN_DAYS = 3       # unscheduled triggers wait this long after any retrain
CHALLENGE_DAYS = 14             # a refit must beat the current model on the last 14 days

BAND_TARGET = 0.80
BAND_LOW, BAND_HIGH = 0.70, 0.90
BAND_MAX_STEP = 1.5             # each recalibration changes the width by at most +-50%
BAND_RECAL_EVERY_DAYS = 7


def known(history: pd.DataFrame, model_name: str, avail_day, since_train_end: str | None = None) -> pd.DataFrame:
    """Rows of one version whose actual was published by the issue day."""
    h = history[(history["model_name"] == model_name) & (history["local_day"] <= avail_day)]
    h = h[h["actual"].notna()]
    if since_train_end is not None:
        h = h[h["train_end"] == since_train_end]
    return h


def hourly_bias(k: pd.DataFrame, avail_day) -> dict[int, float]:
    """Exponentially weighted mean error (actual - raw) per local hour over the last
    28 known days, newest weighted most. Empty until 3 days of errors exist."""
    k = k[k["local_day"] > avail_day - pd.Timedelta(days=BIAS_WINDOW_DAYS)]
    if k["local_day"].nunique() < BIAS_MIN_DAYS:
        return {}
    age = (avail_day - k["local_day"]).dt.days.astype(float)
    w = 0.5 ** (age / BIAS_HALF_LIFE_DAYS)
    e = (k["actual"] - k["raw"]) * w
    return (e.groupby(k["hour"]).sum() / w.groupby(k["hour"]).sum()).to_dict()


def apply_bias(raw: np.ndarray, hours: np.ndarray, bias: dict[int, float]) -> np.ndarray:
    b = np.array([bias.get(int(h), 0.0) for h in hours])
    return np.clip(b, -BIAS_CAP * raw, BIAS_CAP * raw)


def daily_errors(k: pd.DataFrame) -> pd.DataFrame:
    g = k.groupby("local_day")
    out = pd.DataFrame({
        "mean_pct": g.apply(lambda d: ((d["actual"] - d["yhat"]) / d["actual"]).mean() * 100, include_groups=False),
        "mape": g.apply(lambda d: (abs(d["actual"] - d["yhat"]) / d["actual"]).mean() * 100, include_groups=False),
        "peak_pct": g.apply(lambda d: abs(d["actual"].max() - d["yhat"].max()) / d["actual"].max() * 100, include_groups=False),
    })
    return out.sort_index()


def retrain_trigger(k: pd.DataFrame, issue_day, last_retrain_day, validation_mape: float,
                    drift_streak: int) -> tuple[str | None, int]:
    """Which trigger fires at this issue day, if any, and the updated drift streak.
    Order: peak miss, persistent bias, drift, then the Sunday schedule."""
    d = daily_errors(k) if len(k) else pd.DataFrame()
    fired = None
    streak = drift_streak
    if len(d):
        last7 = d.tail(7)
        streak = drift_streak + 1 if (len(last7) >= 5 and last7["mape"].mean() > DRIFT_FACTOR * validation_mape) else 0
        cooling = last_retrain_day is not None and (issue_day - last_retrain_day).days < TRIGGER_COOLDOWN_DAYS
        last5 = d.tail(PERSIST_DAYS)
        if not cooling:
            if d["peak_pct"].iloc[-1] > PEAK_MISS_PCT and d.index[-1] > (last_retrain_day or pd.Timestamp.min):
                fired = "peak_miss"
            elif (len(last5) == PERSIST_DAYS and (np.sign(last5["mean_pct"]).nunique() == 1)
                  and last5["mean_pct"].abs().mean() > PERSIST_PCT):
                fired = "persistent_bias"
            elif streak >= DRIFT_RUNS:
                fired = "drift"
    if fired is None and issue_day.dayofweek == 6:
        fired = "scheduled"
    if fired == "drift":
        streak = 0
    return fired, streak


def recalibrate_band(k: pd.DataFrame, avail_day, scale: float) -> tuple[float, float | None]:
    """Widen or narrow the band when the last 14 days' coverage leaves 70-90%.
    Returns (new scale, coverage seen)."""
    k = k[k["local_day"] > avail_day - pd.Timedelta(days=14)]
    if k["local_day"].nunique() < 7:
        return scale, None
    inside = (k["actual"] >= k["lo"]) & (k["actual"] <= k["hi"])
    cov = float(inside.mean())
    if BAND_LOW <= cov <= BAND_HIGH:
        return scale, cov
    e = k["actual"] - k["yhat"]
    half = np.where(e >= 0, k["hi"] - k["yhat"], k["yhat"] - k["lo"])
    ratio = np.abs(e) / np.maximum(half, 1.0)
    factor = float(np.clip(np.quantile(ratio, BAND_TARGET), 1 / BAND_MAX_STEP, BAND_MAX_STEP))
    return scale * factor, cov
