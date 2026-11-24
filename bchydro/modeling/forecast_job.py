"""Forecast, score and correct: the same loop for the year-long replay and for live runs.

    python -m bchydro.modeling.forecast_job train    # fit the frozen model (data to the cutoff)
    python -m bchydro.modeling.forecast_job replay   # replay 16 Nov 2025 -> latest actual, one day at a time
    python -m bchydro.modeling.forecast_job live     # today's forecast (run by the daily job)

Each issue day D (09:30 Pacific) forecasts the 24 hours of D+1 in four versions:
  baseline        same hour last week
  frozen          LightGBM trained once, up to the cutoff; never changes
  bias_corrected  frozen + hourly bias correction + band recalibration
  retrained       the champion model (refit weekly or on a trigger, kept only if it wins)
                  + its own bias correction and band recalibration: the version operators see
Live runs also write a 7-day outlook (model_name 'outlook') for the peak-risk tab.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config, db
from . import correction as C
from . import features as F
from . import models as M

VERSIONS = ["baseline", "frozen", "bias_corrected", "retrained"]
ISSUE_TIME = "09:30"


def issue_ts(day: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(f"{day.date()} {ISSUE_TIME}").tz_localize(config.LOCAL_TZ).tz_convert("UTC")


def available_through(day: pd.Timestamp) -> pd.Timestamp:
    """Last local day of load BC Hydro has published by 09:30 on `day`: yesterday on
    weekdays (Monday's update covers Friday to Sunday), Thursday on Saturday and Sunday."""
    wd = day.dayofweek
    return day - pd.Timedelta(days=1 if wd <= 4 else wd - 3)


def iso(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class VersionState:
    band_scale: float
    drift_streak: int = 0
    last_retrain_day: pd.Timestamp | None = None
    last_band_day: pd.Timestamp | None = None


class Engine:
    def __init__(self, con, phase: str):
        self.con, self.phase = con, phase
        self.load = F.load_series(con)
        self.cal = F.calendar_frame(con)
        v = con.execute("SELECT model_json FROM model_store WHERE name='validation'").fetchone()
        if v is None:
            raise RuntimeError("Run python -m bchydro.modeling.validate first")
        v = json.loads(v[0])
        self.validation_mape = v["chosen"]["mape"]
        self.band_scale0 = v["band_scale"]
        self.frozen = M.load(con, "frozen")
        if self.frozen is None:
            raise RuntimeError("Run python -m bchydro.modeling.forecast_job train first")
        self.champion = M.load(con, "champion") or self.frozen
        self.state = {n: VersionState(self.band_scale0) for n in VERSIONS}
        self.history = pd.DataFrame(columns=["model_name", "issued_day", "target", "local_day", "hour",
                                             "raw", "yhat", "lo", "hi", "actual", "train_end"])
        self._train_cache = None
        self._replay_cache = None

    # ------------------------------------------------------------ features
    def train_rows(self, end_day: pd.Timestamp) -> pd.DataFrame:
        """Training rows up to end_day. ERA5 weather lags about 5 days, so the last
        6 days use the archived forecasts, as they would have on that date."""
        if self._train_cache is None:
            last = self.load.index.max().tz_convert(config.LOCAL_TZ).normalize().tz_localize(None)
            self._train_cache = F.training_table(self.con, config.TRAIN_START, str(last.date()), "train",
                                                 self.load, self.cal)
        tc, rc = self._train_cache, self.replay_rows()
        local = tc.index.tz_convert(config.LOCAL_TZ).tz_localize(None)
        old = tc[local < end_day - pd.Timedelta(days=5)]
        rl = rc.index.tz_convert(config.LOCAL_TZ).tz_localize(None)
        recent = rc[(rl >= end_day - pd.Timedelta(days=5)) & (rl < end_day + pd.Timedelta(days=1))]
        return pd.concat([old, recent])

    def replay_rows(self) -> pd.DataFrame:
        if self._replay_cache is None:
            first = pd.Timestamp(config.CUTOFF) - pd.Timedelta(days=40)
            last = self.load.index.max().tz_convert(config.LOCAL_TZ).normalize().tz_localize(None)
            self._replay_cache = F.training_table(self.con, str(first.date()), str(last.date()), "replay",
                                                  self.load, self.cal)
        return self._replay_cache

    def target_features(self, target_day: pd.Timestamp, issued: pd.Timestamp, temp_shift: float = 0.0) -> pd.DataFrame:
        idx = F.local_day_hours(target_day)
        if self.phase == "backtest" and temp_shift == 0:
            return self.replay_rows().reindex(idx)
        wx = F.weather_frame(self.con, idx[0] - pd.Timedelta(days=4), idx[-1], "live", iso(issued))
        return F.build(idx, self.load, wx, self.cal, temp_shift)

    # ------------------------------------------------------------ retraining
    def maybe_retrain(self, day, avail, k: pd.DataFrame) -> tuple[str | None, bool | None]:
        st = self.state["retrained"]
        trigger, st.drift_streak = C.retrain_trigger(k, day, st.last_retrain_day, self.validation_mape, st.drift_streak)
        if trigger is None:
            return None, None
        eval_start = avail - pd.Timedelta(days=C.CHALLENGE_DAYS - 1)
        window = k[k["local_day"] >= eval_start]
        if window["local_day"].nunique() < C.CHALLENGE_DAYS:      # not enough of our own history yet
            return trigger, None
        st.last_retrain_day = day
        tr = self.train_rows(eval_start - pd.Timedelta(days=1))
        w = M.recent_weight(tr.index, tr.index.max()) if trigger == "persistent_bias" else None
        challenger = M.fit_lgb(tr, tr["y"], quantiles=False, weight=w)
        ev = self.replay_rows().reindex(pd.DatetimeIndex(window["target"]))
        mae_new = float(np.mean(np.abs(window["actual"].to_numpy() - challenger.predict(ev)["yhat"].to_numpy())))
        mae_old = float(np.mean(np.abs(window["actual"] - window["raw"])))
        if mae_new >= mae_old:
            return trigger, False
        tr = self.train_rows(avail)
        w = M.recent_weight(tr.index, tr.index.max()) if trigger == "persistent_bias" else None
        self.champion = M.fit_lgb(tr, tr["y"], weight=w, train_end=str(avail.date()))
        self.champion.meta = dict(trigger=trigger, mae_new=mae_new, mae_old=mae_old)
        return trigger, True

    # ------------------------------------------------------------ one issue day
    def issue(self, day: pd.Timestamp, avail: pd.Timestamp, issued: pd.Timestamp | None = None) -> dict[str, dict]:
        """Forecast day+1 in every version. Returns {version: run} with the forecast frame."""
        issued = issued if issued is not None else issue_ts(day)
        target_day = day + pd.Timedelta(days=1)
        X = self.target_features(target_day, issued)
        hours = X.index.tz_convert(config.LOCAL_TZ).hour.to_numpy()
        horizon = ((X.index - issued) / pd.Timedelta(hours=1)).astype(int)
        runs = {}

        def frame(yhat, lo=None, hi=None, bias=None):
            return pd.DataFrame({"yhat": yhat, "lo": lo, "hi": hi, "bias": bias, "horizon": horizon}, index=X.index)

        runs["baseline"] = dict(model=None, frame=frame(X["lag_7d"].to_numpy()), trigger=None, accepted=None, params={})

        fp = self.frozen.predict(X)
        s0 = self.band_scale0
        runs["frozen"] = dict(model=self.frozen, trigger=None, accepted=None, params=dict(band_scale=s0),
                              frame=frame(fp["yhat"], fp["yhat"] - (fp["yhat"] - fp["lo"]) * s0,
                                          fp["yhat"] + (fp["hi"] - fp["yhat"]) * s0, 0.0))

        for name in ["bias_corrected", "retrained"]:
            st = self.state[name]
            trigger = accepted = None
            if name == "retrained":
                k_all = C.known(self.history, "retrained", avail)
                trigger, accepted = self.maybe_retrain(day, avail, k_all)
            model = self.frozen if name == "bias_corrected" else self.champion
            p = fp if name == "bias_corrected" else model.predict(X)
            # The bias carries across retrains: a refit sees the same weather-forecast
            # skew as the model it replaces, so resetting the bias would undo its gains.
            k = C.known(self.history, name, avail)
            bias =C.apply_bias(p["yhat"].to_numpy(), hours, C.hourly_bias(k, avail))
            cov = None
            if st.last_band_day is None or (day - st.last_band_day).days >= C.BAND_RECAL_EVERY_DAYS:
                new_scale, cov = C.recalibrate_band(C.known(self.history, name, avail), avail, st.band_scale)
                if cov is not None:
                    st.band_scale, st.last_band_day = new_scale, day
            s = st.band_scale
            yhat = p["yhat"].to_numpy() + bias
            runs[name] = dict(model=model, trigger=trigger, accepted=accepted,
                              params=dict(band_scale=s, coverage_14d=cov, drift_streak=st.drift_streak,
                                          last_retrain_day=str(st.last_retrain_day.date()) if st.last_retrain_day is not None else None,
                                          mean_bias_mw=float(np.mean(bias))),
                              frame=frame(yhat, yhat - (p["yhat"] - p["lo"]).to_numpy() * s,
                                          yhat + (p["hi"] - p["yhat"]).to_numpy() * s, bias))

        # remember everything for later corrections
        actual = self.load.reindex(X.index).to_numpy()
        local_day = pd.Timestamp(target_day.date())
        rows = []
        for name, r in runs.items():
            f = r["frame"]
            raw = f["yhat"] - f["bias"].fillna(0) if f["bias"].notna().any() else f["yhat"]
            rows.append(pd.DataFrame({"model_name": name, "issued_day": day, "target": X.index, "local_day": local_day,
                                      "hour": hours, "raw": raw.to_numpy(), "yhat": f["yhat"].to_numpy(),
                                      "lo": f["lo"].to_numpy(), "hi": f["hi"].to_numpy(), "actual": actual,
                                      "train_end": r["model"].train_end if r["model"] is not None else None}))
        self.history = pd.concat([self.history, *rows], ignore_index=True) if len(self.history) else pd.concat(rows, ignore_index=True)
        for r in runs.values():
            r["issued"] = issued
        return runs

    # ------------------------------------------------------------ storage
    def write(self, runs: dict[str, dict]) -> None:
        for name, r in runs.items():
            write_run(self.con, self.phase, name, r)

    def load_history_from_db(self) -> None:
        """Live runs: rebuild the history and each version's state from the database."""
        h = pd.read_sql("""
            SELECT r.run_id, r.phase, r.model_name, r.issued_at, r.train_end, r.params_json, f.target_ts_utc, f.yhat_mw, f.yhat_lo_mw,
                   f.yhat_hi_mw, COALESCE(f.bias_mw, 0) AS bias_mw, l.load_mw
            FROM forecasts f JOIN model_runs r ON r.run_id = f.run_id
            LEFT JOIN load_hourly l ON l.ts_utc = f.target_ts_utc
            WHERE r.model_name IN ('baseline','frozen','bias_corrected','retrained')""", self.con)
        if not len(h):
            return
        h = one_run_per_day(h)
        t = pd.to_datetime(h["target_ts_utc"], utc=True)
        loc = t.dt.tz_convert(config.LOCAL_TZ)
        issued = pd.to_datetime(h["issued_at"], utc=True).dt.tz_convert(config.LOCAL_TZ)
        self.history = pd.DataFrame({
            "model_name": h["model_name"], "issued_day": pd.to_datetime(issued.dt.date), "target": t,
            "local_day": pd.to_datetime(loc.dt.date), "hour": loc.dt.hour, "raw": h["yhat_mw"] - h["bias_mw"],
            "yhat": h["yhat_mw"], "lo": h["yhat_lo_mw"], "hi": h["yhat_hi_mw"], "actual": h["load_mw"],
            "train_end": h["train_end"]})
        last = (pd.read_sql("SELECT model_name, params_json FROM model_runs WHERE run_id IN "
                            "(SELECT MAX(run_id) FROM model_runs GROUP BY model_name)", self.con))
        for _, row in last.iterrows():
            if row["model_name"] in self.state and row["params_json"]:
                p = json.loads(row["params_json"])
                st = self.state[row["model_name"]]
                st.band_scale = p.get("band_scale", st.band_scale)
                st.drift_streak = p.get("drift_streak", 0) or 0
                if p.get("last_retrain_day"):
                    st.last_retrain_day = pd.Timestamp(p["last_retrain_day"])
                st.last_band_day = pd.Timestamp(p["last_band_day"]) if p.get("last_band_day") else None


def one_run_per_day(h: pd.DataFrame) -> pd.DataFrame:
    """Keep one forecast per version and target day. The job runs more than once a day and
    the replay can overlap live days: a live run wins over the replay, then the latest run."""
    day = pd.to_datetime(h["target_ts_utc"], utc=True).dt.tz_convert(config.LOCAL_TZ).dt.date
    k = pd.DataFrame({"model_name": h["model_name"], "day": day, "run_id": h["run_id"],
                      "live": (h["phase"] == "live").astype(int)}).drop_duplicates()
    keep = k.sort_values(["live", "run_id"]).groupby(["model_name", "day"])["run_id"].last()
    return h[h["run_id"].isin(set(keep))]


def write_run(con, phase: str, name: str, r: dict) -> int:
    issued = iso(r["issued"])
    old = con.execute("SELECT run_id FROM model_runs WHERE phase=? AND model_name=? AND issued_at=?",
                      (phase, name, issued)).fetchone()
    if old:
        con.execute("DELETE FROM forecasts WHERE run_id=?", (old[0],))
        con.execute("DELETE FROM model_runs WHERE run_id=?", (old[0],))
    m = r["model"]
    cur = con.execute(
        "INSERT INTO model_runs (phase, model_name, issued_at, train_end, n_train_rows, trigger, accepted, params_json) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (phase, name, issued, m.train_end if m else None, m.n_train_rows if m else None, r["trigger"],
         None if r["accepted"] is None else int(r["accepted"]), json.dumps(r["params"])))
    run_id = cur.lastrowid
    f = r["frame"]
    con.executemany(
        "INSERT INTO forecasts (run_id, target_ts_utc, yhat_mw, yhat_lo_mw, yhat_hi_mw, horizon_h, bias_mw) "
        "VALUES (?,?,?,?,?,?,?)",
        [(run_id, iso(t), float(y), None if pd.isna(lo) else float(lo), None if pd.isna(hi) else float(hi),
          int(h), None if pd.isna(b) else float(b))
         for t, y, lo, hi, h, b in zip(f.index, f["yhat"], f["lo"], f["hi"], f["horizon"], f["bias"]) if not pd.isna(y)])
    return run_id


# ---------------------------------------------------------------- commands
def train_frozen() -> None:
    db.init_db()
    with db.connect() as con:
        load, cal = F.load_series(con), F.calendar_frame(con)
        tr = F.training_table(con, config.TRAIN_START, config.CUTOFF, "train", load, cal)
        m = M.fit_lgb(tr, tr["y"], train_end=config.CUTOFF)
        M.save(con, "frozen", m)
        con.execute("DELETE FROM model_store WHERE name IN ('champion', 'outlook')")
        print(f"frozen model: {m.n_train_rows} rows to {config.CUTOFF}")


def replay(last_issue_day: str | None = None) -> None:
    """Replay every issue day from the cutoff to the last day with a published actual."""
    db.init_db()
    with db.connect() as con:
        con.execute("DELETE FROM forecasts WHERE run_id IN (SELECT run_id FROM model_runs WHERE phase='backtest')")
        con.execute("DELETE FROM model_runs WHERE phase='backtest'")
        con.execute("DELETE FROM model_store WHERE name = 'champion'")
        eng = Engine(con, "backtest")
        last_actual = eng.load.index.max().tz_convert(config.LOCAL_TZ).normalize().tz_localize(None)
        end = pd.Timestamp(last_issue_day) if last_issue_day else last_actual - pd.Timedelta(days=1)
        days = pd.date_range(pd.Timestamp(config.CUTOFF), end, freq="D")
        for i, day in enumerate(days):
            runs = eng.issue(day, available_through(day))
            for n in ("bias_corrected", "retrained"):
                runs[n]["params"]["last_band_day"] = (str(eng.state[n].last_band_day.date())
                                                      if eng.state[n].last_band_day is not None else None)
            eng.write(runs)
            rt = runs["retrained"]
            if rt["trigger"] and rt["accepted"] is not None:
                print(f"{day.date()}  retrain ({rt['trigger']}): {'accepted' if rt['accepted'] else 'rejected'}", flush=True)
            if i % 30 == 0:
                con.commit()
                print(f"{day.date()}  {i + 1}/{len(days)} days", flush=True)
        M.save(con, "champion", eng.champion)
        # store band/drift state on the last runs so the live job continues from here
        for n in ("bias_corrected", "retrained"):
            st = eng.state[n]
            row = con.execute("SELECT run_id, params_json FROM model_runs WHERE model_name=? ORDER BY run_id DESC LIMIT 1", (n,)).fetchone()
            p = json.loads(row[1]) | dict(band_scale=st.band_scale, drift_streak=st.drift_streak,
                                          last_retrain_day=str(st.last_retrain_day.date()) if st.last_retrain_day is not None else None,
                                          last_band_day=str(st.last_band_day.date()) if st.last_band_day is not None else None)
            con.execute("UPDATE model_runs SET params_json=? WHERE run_id=?", (json.dumps(p), row[0]))
        db.log_ingest(con, "replay", "ok", len(days) * len(VERSIONS))
        print(f"replay done: {len(days)} issue days")


def outlook(con, eng: Engine, day: pd.Timestamp, avail: pd.Timestamp, issued: pd.Timestamp) -> None:
    """Hourly forecasts for the next 7 days from a model without 7-day lags, refit on Sundays."""
    m = M.load(con, "outlook")
    if m is None or day.dayofweek == 6:
        tr = eng.train_rows(avail)
        m = M.fit_lgb(tr, tr["y"], features=M.OUTLOOK_FEATURES, train_end=str(avail.date()))
        M.save(con, "outlook", m)
    first = day + pd.Timedelta(days=1)
    idx = F.hours_between(first, first + pd.Timedelta(days=6))
    wx = F.weather_frame(con, idx[0] - pd.Timedelta(days=4), idx[-1], "live", iso(issued))
    X = F.build(idx, eng.load, wx, eng.cal)
    X = X[X["temp_w"].notna()]
    loc_day = X.index.tz_convert(config.LOCAL_TZ).date
    full = pd.Series(loc_day).value_counts()
    X = X[pd.Series(loc_day, index=X.index).map(full).to_numpy() >= 23]      # whole days only
    if X.empty:              # no weather forecast for the coming days: keep the last outlook
        return
    p = m.predict(X)
    s = eng.band_scale0
    f = pd.DataFrame({"yhat": p["yhat"], "lo": p["yhat"] - (p["yhat"] - p["lo"]) * s,
                      "hi": p["yhat"] + (p["hi"] - p["yhat"]) * s, "bias": 0.0,
                      "horizon": ((X.index - issued) / pd.Timedelta(hours=1)).astype(int)}, index=X.index)
    write_run(con, "live", "outlook", dict(model=m, frame=f, trigger=None, accepted=None, issued=issued,
                                          params=dict(band_scale=s)))


def live(now: pd.Timestamp | None = None) -> str:
    """Today's forecast for tomorrow, in every version, plus the 7-day outlook."""
    db.init_db()
    now = now or pd.Timestamp.now(tz="UTC")
    with db.connect() as con:
        eng = Engine(con, "live")
        eng.load_history_from_db()
        day = pd.Timestamp(now.tz_convert(config.LOCAL_TZ).date())
        last = eng.load.index.max().tz_convert(config.LOCAL_TZ)
        avail = pd.Timestamp(last.date()) if last.hour == 23 else pd.Timestamp(last.date()) - pd.Timedelta(days=1)
        issued = pd.Timestamp(now).floor("min")
        runs = eng.issue(day, avail, issued)
        for n in ("bias_corrected", "retrained"):
            lb = eng.state[n].last_band_day
            runs[n]["params"]["last_band_day"] = str(lb.date()) if lb is not None else None
        eng.write(runs)
        if runs["retrained"]["accepted"]:
            M.save(con, "champion", eng.champion)
        outlook(con, eng, day, avail, issued)
        rt = runs["retrained"]["frame"]
        return (f"forecast for {(day + pd.Timedelta(days=1)).date()}: peak {rt['yhat'].max():,.0f} MW; "
                f"actuals to {avail.date()}; retrain: {runs['retrained']['trigger'] or 'none'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "live"
    if cmd == "train":
        train_frozen()
    elif cmd == "replay":
        replay(*sys.argv[2:3])
    elif cmd == "live":
        print(live())
    else:
        sys.exit(__doc__)
