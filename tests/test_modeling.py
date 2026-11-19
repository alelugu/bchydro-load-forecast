"""Checks on the parts that would silently ruin the results if they broke.

    pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from bchydro import config
from bchydro.modeling import correction as C
from bchydro.modeling import features as F
from bchydro.modeling.forecast_job import available_through


def test_day_lengths_follow_bc_clock_rules():
    assert len(F.local_day_hours("2024-11-03")) == 25      # last fall-back
    assert len(F.local_day_hours("2026-03-08")) == 23      # last spring-forward
    assert len(F.local_day_hours("2026-11-01")) == 24      # no clock change since 2026
    assert len(F.local_day_hours("2026-12-15")) == 24


@pytest.mark.parametrize("issue, expected", [
    ("2026-11-16", "2026-11-15"),   # Monday: the update covers Fri-Sun
    ("2026-11-17", "2026-11-16"),   # Tuesday
    ("2026-11-20", "2026-11-19"),   # Friday
    ("2026-11-21", "2026-11-19"),   # Saturday: nothing new since Thursday
    ("2026-11-22", "2026-11-19"),   # Sunday
])
def test_available_through_matches_bc_hydro_publishing(issue, expected):
    assert available_through(pd.Timestamp(issue)) == pd.Timestamp(expected)


def _synthetic():
    idx = pd.date_range("2025-01-01", "2025-03-31 23:00", freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    load = pd.Series(7000 + 500 * np.sin(np.arange(len(idx)) / 24 * 2 * np.pi) + rng.normal(0, 50, len(idx)), index=idx)
    wx = pd.DataFrame({c: 5.0 for c in ["temp_van", "temp_vic", "temp_kel", "temp_pg", "temp_w"]}, index=idx)
    wx[["rh", "wind_kmh", "cloud_pct", "precip_mm"]] = [80.0, 10.0, 50.0, 0.0]
    days = pd.date_range("2024-12-01", "2025-04-30").date
    cal = pd.DataFrame({"is_holiday": 0, "daylight_hours": 10.0}, index=pd.Index(days, name="date"))
    return load, wx, cal


def test_features_never_use_load_from_the_last_7_days():
    load, wx, cal = _synthetic()
    target = F.local_day_hours("2025-03-20")
    full = F.build(target, load, wx, cal)
    cut = load[load.index < target[0] - pd.Timedelta(days=7) + pd.Timedelta(hours=1)]
    # Remove every hour newer than 7 days before the *first* target hour: nothing may change
    # for that hour, and later hours may only lose values, never gain different ones.
    trimmed = F.build(target, cut, wx, cal)
    pd.testing.assert_series_equal(full.iloc[0], trimmed.iloc[0], check_names=False)


def test_temperature_shift_moves_degree_days():
    load, wx, cal = _synthetic()
    target = F.local_day_hours("2025-03-20")
    a, b = F.build(target, load, wx, cal), F.build(target, load, wx, cal, temp_shift_c=-5)
    assert np.allclose(b["hdh"] - a["hdh"], 5)
    assert (b["lag_7d"] == a["lag_7d"]).all()


def test_bias_is_capped_at_3_percent():
    raw = np.array([8000.0, 8000.0])
    b = C.apply_bias(raw, np.array([17, 18]), {17: 1000.0, 18: -50.0})
    assert b[0] == pytest.approx(240.0) and b[1] == pytest.approx(-50.0)


def test_persistent_bias_triggers_and_cooldown_blocks_it():
    days = pd.date_range("2026-01-01", periods=6)
    k = pd.DataFrame({"local_day": np.repeat(days, 24), "actual": 8300.0, "yhat": 8000.0})
    trig, _ = C.retrain_trigger(k, pd.Timestamp("2026-01-08"), None, 3.0, 0)
    assert trig in ("persistent_bias", "peak_miss")
    trig, _ = C.retrain_trigger(k, pd.Timestamp("2026-01-08"), pd.Timestamp("2026-01-07"), 3.0, 0)
    assert trig is None


def test_tz_database_knows_bc_permanent_time():
    config.check_tz()
