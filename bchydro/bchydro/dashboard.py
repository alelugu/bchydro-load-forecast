"""Data and charts for the Gradio app (app.py). Everything reads the one SQLite database."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config, db
from .modeling import features as F
from .modeling import models as M

TZ = config.LOCAL_TZ
VERSION_LABEL = {"retrained": "Retrained + corrected", "bias_corrected": "Frozen + bias correction",
                 "frozen": "Frozen (no correction)", "baseline": "Same hour last week"}
LABEL_VERSION = {v: k for k, v in VERSION_LABEL.items()}
COLOR = {"actual": "#1f2937", "retrained": "#2563eb", "bias_corrected": "#0d9488",
         "frozen": "#ea580c", "baseline": "#9ca3af", "whatif": "#7c3aed"}
ALERT_COLOR = {"Normal": "#16a34a", "Watch": "#d97706", "High": "#dc2626"}
ALERT_MARK = {"Normal": "🟢", "Watch": "🟠", "High": "🔴"}
ACTIONS = {
    "Normal": "Normal scheduling. Low-demand hours (shaded green) are candidates for exports through Powerex.",
    "Watch": "Check reserve and import availability for the evening peak (16:00-20:00); hold back reservoir water where possible.",
    "High": "Line up imports early, defer planned maintenance, and prepare a conservation appeal.",
}
SHORT_ACTIONS = {"Normal": "Normal scheduling; look for export hours",
                 "Watch": "Check reserves and imports for 16:00-20:00",
                 "High": "Arrange imports, defer maintenance, prepare appeal"}
LAYOUT = dict(template="plotly_white", margin=dict(l=60, r=30, t=50, b=40), height=420,
              legend=dict(orientation="h", y=-0.18), hovermode="x unified",
              font=dict(family="Inter, Arial, sans-serif", size=13))


# ---------------------------------------------------------------- data access
def _mtime() -> float:
    p = Path(config.DB_PATH)
    return p.stat().st_mtime if p.exists() else 0.0


def refresh() -> None:
    """Fetch a newer published database if configured, and drop cached data if it changed."""
    try:
        db.refresh_db_if_stale()      # the cache below is keyed on the file's mtime,
    except Exception:  # noqa: BLE001 - so a new download is picked up automatically
        pass


@lru_cache(maxsize=2)
def _cached(mtime: float) -> dict:
    con = db.connect(read_only=True)
    load = F.load_series(con)
    scores = pd.read_sql("""
        SELECT r.run_id, r.phase, r.model_name, r.issued_at, r.trigger, r.accepted, f.target_ts_utc,
               f.yhat_mw, f.yhat_lo_mw, f.yhat_hi_mw, f.bias_mw, l.load_mw AS actual_mw
        FROM forecasts f JOIN model_runs r ON r.run_id = f.run_id
        LEFT JOIN load_hourly l ON l.ts_utc = f.target_ts_utc
        WHERE r.model_name != 'outlook'""", con)
    t = pd.to_datetime(scores["target_ts_utc"], utc=True).dt.tz_convert(TZ)
    scores["t"], scores["day"], scores["hour"] = t, pd.to_datetime(t.dt.date), t.dt.hour
    scores["err"] = scores["actual_mw"] - scores["yhat_mw"]
    scores["ape"] = scores["err"].abs() / scores["actual_mw"] * 100
    runs = pd.read_sql("SELECT * FROM model_runs", con)
    ingest = pd.read_sql("SELECT * FROM ingest_log ORDER BY id DESC LIMIT 200", con)
    val = con.execute("SELECT model_json FROM model_store WHERE name='validation'").fetchone()
    return dict(load=load, scores=scores, runs=runs, ingest=ingest,
                validation=json.loads(val[0]) if val else None, cal=F.calendar_frame(con))


def data() -> dict:
    return _cached(_mtime())


def winter_daily_peaks() -> pd.Series:
    load = data()["load"]
    loc = load.index.tz_convert(TZ)
    s = load[(loc.month.isin([11, 12, 1, 2])) & (loc < pd.Timestamp(config.CUTOFF, tz=TZ))]
    return s.groupby(s.index.tz_convert(TZ).date).max()


def thresholds(watch_pct: float, high_pct: float) -> tuple[float, float]:
    p = winter_daily_peaks()
    return float(np.percentile(p, watch_pct)), float(np.percentile(p, high_pct))


def level(peak: float, thr: tuple[float, float]) -> str:
    if peak >= thr[1] or peak >= 0.97 * config.RECORD_PEAK_MW:
        return "High"
    return "Watch" if peak >= thr[0] else "Normal"


def status_line() -> str:
    d = data()
    last = d["load"].index.max().tz_convert(TZ)
    ok = d["ingest"][d["ingest"]["status"] == "ok"]
    ran = pd.to_datetime(ok["ran_at"].iloc[0], utc=True).tz_convert(TZ) if len(ok) else None
    live = d["runs"][d["runs"]["phase"] == "live"]
    lf = pd.to_datetime(live["issued_at"].max(), utc=True).tz_convert(TZ) if len(live) else None
    parts = [f"**Actuals to** {last:%a %d %b %Y, %H:00}",
             f"**Data last updated** {ran:%a %d %b %H:%M}" if ran is not None else "",
             f"**Latest live forecast** {lf:%a %d %b %H:%M}" if lf is not None else "**No live forecast yet**"]
    return " · ".join(p for p in parts if p) + " (Pacific time)"


# ---------------------------------------------------------------- Tomorrow tab
def forecast_days() -> list[str]:
    s = data()["scores"]
    s = s[s["model_name"] == "retrained"]
    live = s[s["phase"] == "live"]["day"].unique()
    days = sorted(set(s["day"].unique()), reverse=True)
    return [f"{pd.Timestamp(d):%Y-%m-%d}" + (" (live)" if d in live else "") for d in days]


def _run_for(version: str, day: pd.Timestamp) -> pd.DataFrame:
    s = data()["scores"]
    r = s[(s["model_name"] == version) & (s["day"] == day)]
    if r["phase"].eq("live").any():
        r = r[r["phase"] == "live"]
    r = r[r["run_id"] == r["run_id"].max()]
    return r.sort_values("t")


def whatif_delta(run: pd.DataFrame, version: str, shift: float) -> np.ndarray | None:
    """MW change if every temperature were `shift` degrees different: the model run
    twice on the same weather, once shifted. Corrections stay as they were."""
    if shift == 0 or version == "baseline" or not len(run):
        return None
    con = db.connect(read_only=True)
    model = (M.load(con, "champion") if version == "retrained" else None) or M.load(con, "frozen")
    idx = pd.DatetimeIndex(pd.to_datetime(run["target_ts_utc"], utc=True))
    phase = run["phase"].iloc[0]
    issued = run["issued_at"].iloc[0]
    wx = F.weather_frame(con, idx[0] - pd.Timedelta(days=4), idx[-1], "live" if phase == "live" else "replay",
                         issued if phase == "live" else None)
    d = data()
    base = model.predict(F.build(idx, d["load"], wx, d["cal"]))["yhat"].to_numpy()
    shifted = model.predict(F.build(idx, d["load"], wx, d["cal"], temp_shift_c=shift))["yhat"].to_numpy()
    return shifted - base


def tomorrow(day_label: str, version_label: str, shift: float, watch_pct: float, high_pct: float):
    if not day_label:
        return "No forecasts yet: run the replay and the daily job.", go.Figure(), ""
    day = pd.Timestamp(day_label.split(" ")[0])
    version = LABEL_VERSION[version_label]
    run = _run_for(version, day)
    thr = thresholds(watch_pct, high_pct)
    fig = go.Figure()
    if not len(run):
        return f"No {version_label} forecast for {day:%d %b %Y}.", fig, ""
    x = run["t"]
    if run["yhat_lo_mw"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=run["yhat_hi_mw"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=run["yhat_lo_mw"], fill="tonexty", fillcolor="rgba(37,99,235,0.15)",
                                 line=dict(width=0), name="10-90% band"))
    # lowest 6 hours: export windows
    low = run.nsmallest(6, "yhat_mw")["t"]
    for t in low:
        fig.add_vrect(x0=t - pd.Timedelta(minutes=30), x1=t + pd.Timedelta(minutes=30),
                      fillcolor="rgba(22,163,74,0.08)", line_width=0)
    fig.add_trace(go.Scatter(x=x, y=run["yhat_mw"], name="Forecast", line=dict(color=COLOR.get(version), width=3)))
    delta = whatif_delta(run, version, shift)
    if delta is not None:
        fig.add_trace(go.Scatter(x=x, y=run["yhat_mw"] + delta, name=f"What if {shift:+.0f} °C",
                                 line=dict(color=COLOR["whatif"], dash="dash", width=2)))
    if run["actual_mw"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=run["actual_mw"], name="Actual", line=dict(color=COLOR["actual"], width=2)))
    for name, y, c in [("Watch", thr[0], ALERT_COLOR["Watch"]), ("High", thr[1], ALERT_COLOR["High"]),
                       ("Record 11,300", config.RECORD_PEAK_MW, "#6b7280")]:
        fig.add_hline(y=y, line=dict(color=c, dash="dot", width=1.2), annotation_text=f"{name}",
                      annotation_position="top left")
    fig.update_layout(**LAYOUT, title=f"Hourly forecast for {day:%A %d %B %Y}", yaxis_title="Load (MW)")
    peak_i = run["yhat_mw"].idxmax()
    peak, peak_t = run.loc[peak_i, "yhat_mw"], run.loc[peak_i, "t"]
    lvl = level(peak, thr)
    recent = data()["scores"]
    recent = recent[(recent["model_name"] == version) & recent["actual_mw"].notna() & (recent["day"] < day)]
    recent = recent[recent["day"] >= day - pd.Timedelta(days=7)]
    mape7 = recent["ape"].mean() if len(recent) else float("nan")
    kpi = (f"| Forecast peak | Peak hour | Alert level | Daily energy | Error, last 7 days |\n|---|---|---|---|---|\n"
           f"| **{peak:,.0f} MW** | **{peak_t:%H:00}** | {ALERT_MARK[lvl]} **{lvl}** "
           f"| {run['yhat_mw'].sum() / 1000:,.1f} GWh | {mape7:.1f}% MAPE |")
    if delta is not None:
        kpi += f"\n\nWhat if {shift:+.0f} °C: peak {(run['yhat_mw'] + delta).max():,.0f} MW ({delta[run.index.get_loc(peak_i)]:+,.0f} MW at {peak_t:%H:00})."
    if run["actual_mw"].notna().any():
        kpi += f"\n\nActual peak {run['actual_mw'].max():,.0f} MW; hourly error {run['ape'].mean():.1f}% MAPE."
    cold = ""
    action = (f"### Suggested actions: {ALERT_MARK[lvl]} {lvl}\n{ACTIONS[lvl]}\n\n"
              f"Thresholds: Watch from {thr[0]:,.0f} MW, High from {thr[1]:,.0f} MW "
              f"(percentiles of winter daily peaks, 2015 to the cutoff).{cold}")
    return kpi, fig, action


# ---------------------------------------------------------------- Peak risk tab
def peak_risk(watch_pct: float, high_pct: float, temp_source: str):
    con = db.connect(read_only=True)
    r = con.execute("SELECT run_id, issued_at FROM model_runs WHERE model_name='outlook' ORDER BY run_id DESC LIMIT 1").fetchone()
    if r is None:
        return go.Figure(), pd.DataFrame(), "No 7-day outlook yet: it is written by the daily job."
    f = pd.read_sql("SELECT target_ts_utc, yhat_mw, yhat_lo_mw, yhat_hi_mw FROM forecasts WHERE run_id=?", con, params=[r[0]])
    f["t"] = pd.to_datetime(f["target_ts_utc"], utc=True).dt.tz_convert(TZ)
    f["day"] = f["t"].dt.date
    g = f.loc[f.groupby("day")["yhat_mw"].idxmax()].set_index("day")
    thr = thresholds(watch_pct, high_pct)
    idx = pd.DatetimeIndex(pd.to_datetime(f["target_ts_utc"], utc=True))
    wx = F.weather_frame(con, idx[0], idx[-1], "live", r[1])
    col = {"Provincial (population-weighted)": "temp_w", "Vancouver": "temp_van", "Victoria": "temp_vic",
           "Kelowna": "temp_kel", "Prince George": "temp_pg"}[temp_source]
    tt = wx[col].groupby(wx.index.tz_convert(TZ).date)
    hdd = (F.BASE_TEMP_C - tt.mean()).clip(lower=0).reindex(g.index)
    tmin = tt.min().reindex(g.index)
    g["level"] = [level(p, thr) for p in g["yhat_mw"]]
    labels = [f"{pd.Timestamp(d):%a %d %b}" for d in g.index]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=labels, y=g["yhat_mw"], mode="markers", name="Forecast daily peak (whiskers: 10-90% range)",
                             marker=dict(size=16, color=[ALERT_COLOR[l] for l in g["level"]], line=dict(color="white", width=1.5)),
                             error_y=dict(type="data", symmetric=False, color="#374151", thickness=1.5, width=8,
                                          array=(g["yhat_hi_mw"] - g["yhat_mw"]).clip(lower=0),
                                          arrayminus=(g["yhat_mw"] - g["yhat_lo_mw"]).clip(lower=0))), secondary_y=False)
    fig.add_trace(go.Scatter(x=labels, y=hdd, name=f"Heating degree days ({temp_source})", mode="lines+markers",
                             line=dict(color="#2563eb")), secondary_y=True)
    for name, y, c in [("Watch", thr[0], ALERT_COLOR["Watch"]), ("High", thr[1], ALERT_COLOR["High"])]:
        fig.add_hline(y=y, line=dict(color=c, dash="dot"), annotation_text=name, secondary_y=False)
    lo = min(g["yhat_lo_mw"].min(), thr[0]) * 0.97
    fig.update_layout(**{**LAYOUT, "hovermode": "x"},
                      title=f"Peak outlook, next {len(g)} days (as far as the weather forecast reaches; "
                            f"issued {pd.Timestamp(r[1]).tz_convert(TZ):%a %d %b %H:%M})")
    fig.update_yaxes(title_text="Daily peak (MW)", range=[lo, max(g["yhat_hi_mw"].max(), thr[1]) * 1.03], secondary_y=False)
    fig.update_yaxes(title_text="HDD (°C-days)", secondary_y=True, showgrid=False)
    table = pd.DataFrame({"Day": labels, "Forecast peak (MW)": g["yhat_mw"].round(0).astype(int),
                          "Peak hour": g["t"].dt.strftime("%H:00"), "Coldest (°C)": tmin.round(1).values,
                          "HDD": hdd.round(1).values, "Alert": [f"{ALERT_MARK[l]} {l}" for l in g["level"]], "Action": [SHORT_ACTIONS[l] for l in g["level"]]})
    worst = g["level"].map({"Normal": 0, "Watch": 1, "High": 2}).max()
    msg = {0: f"No Watch or High days in the next {len(g)} days.",
           1: "At least one **Watch** day ahead: check reserves and imports for those evenings.",
           2: "**High** demand expected: arrange imports and defer maintenance now."}[worst]
    return fig, table, msg


# ---------------------------------------------------------------- History tab
def history(start: str, end: str, grain: str, day_types: list[str], city: str):
    d = data()
    load = d["load"]
    a, b = pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1)
    s = load[(load.index >= a) & (load.index < b)]
    if not len(s):
        return go.Figure(), go.Figure(), go.Figure(), "No data in that range."
    con = db.connect(read_only=True)
    wx = F.weather_frame(con, s.index.min() - pd.Timedelta(days=1), s.index.max(), "train")
    col = {"Provincial (population-weighted)": "temp_w", "Vancouver": "temp_van", "Victoria": "temp_vic",
           "Kelowna": "temp_kel", "Prince George": "temp_pg"}[city]
    df = pd.DataFrame({"load": s, "temp": wx[col].reindex(s.index)})
    loc = df.index.tz_convert(TZ)
    hol = pd.Index(loc.date).map(d["cal"]["is_holiday"]).fillna(0).to_numpy().astype(bool)
    kind = np.where(hol, "Holiday", np.where(loc.dayofweek >= 5, "Weekend", "Weekday"))
    df = df[np.isin(kind, day_types)]
    if not len(df):
        return go.Figure(), go.Figure(), go.Figure(), "No days of that type in the range."
    loc = df.index.tz_convert(TZ)
    rule = {"Hourly": None, "Daily": "D", "Weekly": "W"}[grain]
    plot = df.set_index(loc)
    if rule:
        plot = plot.resample(rule).agg({"load": "mean", "temp": "mean"}).dropna()
    f1 = make_subplots(specs=[[{"secondary_y": True}]])
    f1.add_trace(go.Scatter(x=plot.index, y=plot["load"], name="Load (MW)", line=dict(color=COLOR["actual"])), secondary_y=False)
    f1.add_trace(go.Scatter(x=plot.index, y=plot["temp"], name=f"Temperature, {city} (°C)",
                            line=dict(color="#2563eb", width=1)), secondary_y=True)
    f1.update_layout(**LAYOUT, title=f"Load and temperature ({grain.lower()} average)")
    f1.update_yaxes(title_text="MW", secondary_y=False)
    f1.update_yaxes(title_text="°C", secondary_y=True, showgrid=False)

    daily = df.groupby(loc.date).agg(load=("load", "mean"), temp=("temp", "mean"))
    daily["hdd"] = (F.BASE_TEMP_C - daily["temp"]).clip(lower=0)
    f2 = go.Figure(go.Scatter(x=daily["hdd"], y=daily["load"], mode="markers", name="Days",
                              marker=dict(size=5, color=COLOR["retrained"], opacity=0.5)))
    note = ""
    cold = daily[daily["hdd"] > 0]
    if len(cold) > 10:
        slope, icpt = np.polyfit(cold["hdd"], cold["load"], 1)
        xs = np.linspace(0, cold["hdd"].max(), 20)
        f2.add_trace(go.Scatter(x=xs, y=icpt + slope * xs, name=f"Fit: {slope:,.0f} MW per HDD", line=dict(color="#dc2626")))
        note = (f"On days below 18 °C, each extra degree of daily cold added about **{slope:,.0f} MW** of average "
                f"load in this period ({len(cold)} days).")
    f2.update_layout(**{**LAYOUT, "hovermode": "closest"}, title="Daily mean load vs heating degree days",
                     xaxis_title="HDD (°C-days below 18 °C)", yaxis_title="Mean load (MW)")

    shape = df.groupby([loc.month, loc.hour])["load"].mean().unstack(0)
    f3 = go.Figure()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in shape.columns:
        f3.add_trace(go.Scatter(x=shape.index, y=shape[m], name=months[m - 1], mode="lines"))
    f3.update_layout(**{**LAYOUT, "hovermode": "x"}, title="Average daily load shape by month",
                     xaxis_title="Hour of day", yaxis_title="Mean load (MW)")
    return f1, f2, f3, note


# ---------------------------------------------------------------- Correction tab
def correction(phase: str, start: str, end: str, version_labels: list[str], metric: str,
               watch_pct: float, high_pct: float):
    d = data()
    s = d["scores"]
    s = s[s["actual_mw"].notna()]
    if phase != "Both":
        s = s[s["phase"] == ("backtest" if phase == "Replay" else "live")]
    s = s[(s["day"] >= pd.Timestamp(start)) & (s["day"] <= pd.Timestamp(end))]
    versions = [LABEL_VERSION[v] for v in version_labels] or ["retrained"]
    s = s[s["model_name"].isin(versions + ["baseline"])]
    empty = (go.Figure(), go.Figure(), go.Figure(), "No scored forecasts in that range.", pd.DataFrame(), pd.DataFrame(), go.Figure())
    if not len(s):
        return empty

    # 1. predicted vs actual
    f1 = go.Figure()
    act = s.drop_duplicates("t").sort_values("t")
    f1.add_trace(go.Scatter(x=act["t"], y=act["actual_mw"], name="Actual", line=dict(color=COLOR["actual"], width=1.5)))
    for v in versions:
        x = s[s["model_name"] == v].sort_values("t")
        f1.add_trace(go.Scatter(x=x["t"], y=x["yhat_mw"], name=VERSION_LABEL[v], line=dict(color=COLOR[v], width=1.2)))
    f1.update_layout(**{**LAYOUT, "height": 520, "legend": dict(orientation="h", y=-0.36),
                        "margin": dict(l=60, r=30, t=50, b=40)},
                     title="Predicted vs actual hourly load (drag the slider below the chart to move)", yaxis_title="MW")
    last = act["t"].max()
    f1.update_xaxes(range=[max(act["t"].min(), last - pd.Timedelta(days=21)), last], rangeslider=dict(visible=True, thickness=0.08))

    # 2. rolling 7-day metric with retrain markers
    def daily_metric(x: pd.DataFrame) -> pd.Series:
        g = x.groupby("day")
        if metric == "MAPE (%)":
            return g["ape"].mean()
        if metric == "MAE (MW)":
            return g["err"].apply(lambda e: e.abs().mean())
        return g.apply(lambda q: abs(q["actual_mw"].max() - q["yhat_mw"].max()) / q["actual_mw"].max() * 100,
                       include_groups=False)
    f2 = go.Figure()
    for v in dict.fromkeys(versions + ["baseline"]):
        m = daily_metric(s[s["model_name"] == v]).rolling(7, min_periods=4).mean()
        f2.add_trace(go.Scatter(x=m.index, y=m, name=VERSION_LABEL[v],
                                line=dict(color=COLOR[v], width=2.5 if v == "retrained" else 1.5,
                                          dash="dot" if v == "baseline" else "solid")))
    runs = d["runs"]
    rt = runs[(runs["model_name"] == "retrained") & (runs["accepted"] == 1)].copy()
    rt["day"] = pd.to_datetime(pd.to_datetime(rt["issued_at"], utc=True).dt.tz_convert(TZ).dt.date)
    rt = rt[(rt["day"] >= pd.Timestamp(start)) & (rt["day"] <= pd.Timestamp(end))]
    if len(rt):
        f2.add_trace(go.Scatter(x=rt["day"], y=[0] * len(rt), mode="markers", name="Accepted retrain",
                                marker=dict(symbol="triangle-up", size=9, color=COLOR["retrained"]),
                                text=rt["trigger"], hovertemplate="%{x|%d %b %Y}: retrain (%{text})<extra></extra>"))
    f2.update_layout(**{**LAYOUT, "hovermode": "x"}, title=f"Rolling 7-day {metric} (triangles: accepted retrains)",
                     yaxis_title=metric)
    f2.update_yaxes(rangemode="tozero")

    # 3. error heatmap for the main version
    main = versions[0]
    hm = s[s["model_name"] == main].copy()
    hm["week"] = hm["day"].dt.to_period("W").dt.start_time
    piv = (hm.assign(pct=hm["err"] / hm["actual_mw"] * 100)
             .pivot_table(index="hour", columns="week", values="pct", aggfunc="mean"))
    lim = float(np.nanpercentile(np.abs(piv.to_numpy()), 98)) if piv.size else 5
    f3 = go.Figure(go.Heatmap(z=piv.to_numpy(), x=piv.columns, y=piv.index, colorscale="RdBu_r", zmid=0, zmin=-lim, zmax=lim,
                              colorbar=dict(title="actual -<br>forecast, %")))
    f3.update_layout(**{**LAYOUT, "hovermode": "closest"}, title=f"Error by hour and week: {VERSION_LABEL[main]} "
                     "(red: under-forecast)", yaxis_title="Hour of day", xaxis_title="Week")

    # headline tiles
    summ = []
    base_mae = s[s["model_name"] == "baseline"]["err"].abs().mean()
    for v in dict.fromkeys(versions + ["baseline"]):
        x = s[s["model_name"] == v]
        cov = ((x["actual_mw"] >= x["yhat_lo_mw"]) & (x["actual_mw"] <= x["yhat_hi_mw"])).mean() * 100 if x["yhat_lo_mw"].notna().any() else np.nan
        g = x.groupby("day")
        peak_err = (g.apply(lambda q: abs(q["actual_mw"].max() - q["yhat_mw"].max()) / q["actual_mw"].max() * 100,
                            include_groups=False)).mean()
        summ.append({"Version": VERSION_LABEL[v], "MAPE (%)": round(x["ape"].mean(), 2),
                     "MAE (MW)": round(x["err"].abs().mean(), 0), "RMSE (MW)": round(np.sqrt((x["err"] ** 2).mean()), 0),
                     "Bias (%)": round((x["err"] / x["actual_mw"]).mean() * 100, 2),
                     "Peak error (%)": round(peak_err, 2),
                     "Skill vs baseline": round(1 - x["err"].abs().mean() / base_mae, 3),
                     "Band coverage (%)": None if np.isnan(cov) else round(cov, 1), "Days": x["day"].nunique()})
    summary = pd.DataFrame(summ)
    best = summary.iloc[0]
    worst = summary[summary["Version"] == VERSION_LABEL["frozen"]]
    note = f"**{best['Version']}**: {best['MAPE (%)']:.2f}% MAPE over {best['Days']} days"
    if len(worst) and best["Version"] != worst.iloc[0]["Version"]:
        note += f", against {worst.iloc[0]['MAPE (%)']:.2f}% for the frozen model"
    note += f"; skill vs baseline {best['Skill vs baseline']:+.0%}."

    # alert classification
    thr = thresholds(watch_pct, high_pct)
    order = ["Normal", "Watch", "High"]
    cls_rows, cm_fig = [], go.Figure()
    for v in versions:
        x = s[s["model_name"] == v].groupby("day").agg(f=("yhat_mw", "max"), a=("actual_mw", "max"))
        fl, al = x["f"].map(lambda p: level(p, thr)), x["a"].map(lambda p: level(p, thr))
        for cls, pos in [("Watch or High", {"Watch", "High"}), ("High", {"High"})]:
            fp, ap = fl.isin(pos), al.isin(pos)
            tp = int((fp & ap).sum())
            prec = tp / fp.sum() if fp.sum() else np.nan
            rec = tp / ap.sum() if ap.sum() else np.nan
            f1s = 2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec) and not np.isnan(rec) else np.nan
            cls_rows.append({"Version": VERSION_LABEL[v], "Alert class": cls,
                             "Flagged": int(fp.sum()), "Actual": int(ap.sum()), "Caught": tp,
                             "Precision": None if np.isnan(prec) else round(prec, 2),
                             "Recall": None if np.isnan(rec) else round(rec, 2),
                             "F1": None if np.isnan(f1s) else round(f1s, 2)})
        if v == main:
            cm = pd.crosstab(al, fl).reindex(index=order, columns=order, fill_value=0)
            cm_fig = go.Figure(go.Heatmap(z=cm.to_numpy(), x=[f"Forecast {c}" for c in order], y=[f"Actual {c}" for c in order],
                                          text=cm.to_numpy(), texttemplate="%{text}", colorscale="Blues", showscale=False))
            cm_fig.update_layout(**{**LAYOUT, "hovermode": "closest", "height": 340},
                                 title=f"Alert levels, days: {VERSION_LABEL[main]}")
            cm_fig.update_yaxes(autorange="reversed")
    return f1, f2, f3, note, summary, pd.DataFrame(cls_rows), cm_fig


def model_log() -> pd.DataFrame:
    r = data()["runs"]
    r = r[(r["model_name"] == "retrained") & r["trigger"].notna()].copy()
    if not len(r):
        return pd.DataFrame(columns=["Issue day", "Phase", "Trigger", "Result", "Trained to"])
    r["Issue day"] = pd.to_datetime(r["issued_at"], utc=True).dt.tz_convert(TZ).dt.strftime("%a %Y-%m-%d")
    r["Result"] = r["accepted"].map({1: "accepted", 0: "rejected (old model was better)"}).fillna("skipped (not enough history)")
    r = r.rename(columns={"phase": "Phase", "trigger": "Trigger", "train_end": "Trained to"})
    return r.sort_values("issued_at", ascending=False)[["Issue day", "Phase", "Trigger", "Result", "Trained to"]]


def validation_table() -> pd.DataFrame:
    v = data()["validation"]
    if not v:
        return pd.DataFrame()
    t = pd.DataFrame(v["results"]).T.reset_index().rename(columns={"index": "Model"})
    t = t.rename(columns={"mape": "MAPE (%)", "mae": "MAE (MW)", "rmse": "RMSE (MW)", "peak_mape": "Peak error (%)",
                          "peak_hour_off": "Peak hour off (h)", "coverage": "Raw band coverage (%)"})
    return t.round(2)
