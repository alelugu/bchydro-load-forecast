"""Model selection on the 2024 validation year.

    python -m bchydro.modeling.validate

Train: 2015-2023 with observed weather. Validate: 2024 with the forecasts as issued
(what an operator would have had). Prints a table, and stores the validation MAPE in
model_store['validation'] for the drift trigger and the dashboard.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import db
from . import features as F
from . import models as M

TRAIN = ("2015-01-01", "2023-12-31")
VALID = ("2024-01-01", "2024-12-31")


def metrics(y: np.ndarray, yhat: np.ndarray, index: pd.DatetimeIndex, lo=None, hi=None) -> dict:
    ok = ~np.isnan(yhat) & ~np.isnan(y)
    y, yhat, index = y[ok], yhat[ok], index[ok]
    e = y - yhat
    out = dict(mape=float(np.mean(np.abs(e) / y) * 100), mae=float(np.mean(np.abs(e))),
               rmse=float(np.sqrt(np.mean(e ** 2))))
    d = pd.DataFrame({"y": y, "f": yhat}, index=index.tz_convert("America/Vancouver"))
    g = d.groupby(d.index.date)
    peak_y, peak_f = g["y"].max(), g["f"].max()
    out["peak_mape"] = float(np.mean(np.abs(peak_y - peak_f) / peak_y) * 100)
    out["peak_hour_off"] = float(np.mean(np.abs(g["y"].idxmax().map(lambda t: t.hour) - g["f"].idxmax().map(lambda t: t.hour))))
    if lo is not None:
        out["coverage"] = float(np.mean((y >= lo[ok]) & (y <= hi[ok])) * 100)
    return out


def run() -> dict:
    con = db.connect(read_only=True)
    load, cal = F.load_series(con), F.calendar_frame(con)
    tr = F.training_table(con, *TRAIN, mode="train", load=load, cal=cal)
    va = F.training_table(con, *VALID, mode="replay", load=load, cal=cal)
    y = va["y"].to_numpy()
    res = {}
    res["Same hour last week (baseline)"] = metrics(y, M.baseline(va), va.index)
    lin = M.fit_linear(tr, tr["y"])
    res["Linear on degree days"] = metrics(y, M.predict_linear(lin, va), va.index)
    grid = {
        "LightGBM, 300 trees, 31 leaves": dict(n_estimators=300, num_leaves=31, learning_rate=0.06),
        "LightGBM, 700 trees, 63 leaves": dict(),
        "LightGBM, 1200 trees, 127 leaves": dict(n_estimators=1200, num_leaves=127, learning_rate=0.03),
    }
    band_scale = 1.0
    for name, p in grid.items():
        m = M.fit_lgb(tr, tr["y"], quantiles=(p == {}), params=p)
        pr = m.predict(va)
        if p == {}:
            # The raw quantile band is too narrow out of sample (weather-forecast error):
            # find the widening that covers 80% of 2024's hours.
            band_scale = band_scale_for(y, pr)
        lo = pr["lo"].to_numpy() if "lo" in pr else None
        hi = pr["hi"].to_numpy() if "hi" in pr else None
        res[name] = metrics(y, pr["yhat"].to_numpy(), va.index, lo, hi)
    m = M.fit_lgb(tr, tr["y"], features=M.OUTLOOK_FEATURES, quantiles=False)
    res["LightGBM, outlook features (no 7-day lags)"] = metrics(y, m.predict(va)["yhat"].to_numpy(), va.index)

    table = pd.DataFrame(res).T.round(2)
    print(table.to_string())
    print(f"band scale for 80% coverage: {band_scale:.2f}")
    chosen = res["LightGBM, 700 trees, 63 leaves"]
    db.init_db()
    with db.connect() as rw:
        rw.execute("INSERT OR REPLACE INTO model_store (name, saved_at, train_end, model_json) VALUES (?,?,?,?)",
                   ("validation", db.utcnow(), TRAIN[1],
                    json.dumps(dict(results=res, chosen=chosen, band_scale=band_scale))))
    return res


def band_scale_for(y: np.ndarray, pr: pd.DataFrame, target: float = 0.80) -> float:
    for s in np.arange(1.0, 5.01, 0.1):
        lo = pr["yhat"] - (pr["yhat"] - pr["lo"]) * s
        hi = pr["yhat"] + (pr["hi"] - pr["yhat"]) * s
        if np.mean((y >= lo) & (y <= hi)) >= target:
            return round(float(s), 2)
    return 5.0


if __name__ == "__main__":
    run()
