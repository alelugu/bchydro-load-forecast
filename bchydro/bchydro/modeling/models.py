"""The models: a 'same hour last week' baseline, a linear degree-day model that explains
the weather effect, and the main LightGBM model with a 10-90% band from two quantile fits.
Models are stored in the database (model_store) so the daily job and the app share them."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from .. import db
from .features import FEATURES

SEED = 4820
# Settings chosen on the 2024 validation year (python -m bchydro.modeling.validate).
LGB_PARAMS = dict(n_estimators=700, learning_rate=0.04, num_leaves=63, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, n_jobs=2, verbose=-1)
QUANTILE_ESTIMATORS = 300
# Load features that a 7-day-ahead forecast cannot know; the outlook model drops them.
SHORT_LAGS = ["lag_7d", "same_hour_4wk", "mean_8_14d", "growth_ratio"]
OUTLOOK_FEATURES = [c for c in FEATURES if c not in SHORT_LAGS]


def baseline(X: pd.DataFrame) -> np.ndarray:
    """Same hour one week earlier."""
    return X["lag_7d"].to_numpy()


# ---------------------------------------------------------------- linear explainer
LINEAR_NUM = ["hdh", "cdh", "hdd_day", "hdd_3d", "daylight_hours", "lag_14d"]
LINEAR_CAT = ["hour", "dow", "is_holiday"]


def fit_linear(X: pd.DataFrame, y: pd.Series):
    pre = ColumnTransformer([("num", StandardScaler(), LINEAR_NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), LINEAR_CAT)])
    m = make_pipeline(pre, Ridge(alpha=1.0))
    ok = X[LINEAR_NUM].notna().all(axis=1) & y.notna()
    m.fit(X.loc[ok], y[ok])
    return m


def predict_linear(m, X: pd.DataFrame) -> np.ndarray:
    Xf = X.copy()
    Xf[LINEAR_NUM] = Xf[LINEAR_NUM].fillna(Xf[LINEAR_NUM].median())
    return m.predict(Xf)


def mw_per_hdd(con: sqlite3.Connection, first_day: str, last_day: str) -> float:
    """Average MW added per degree-day of cold on winter days: the slope of daily mean
    load against daily HDD. Shown on the dashboard as 'how sensitive demand is to cold'."""
    from .features import training_table
    X = training_table(con, first_day, last_day)
    X = X[X.index.tz_convert("America/Vancouver").month.isin([11, 12, 1, 2])]
    d = X.groupby(X.index.tz_convert("America/Vancouver").date).agg(y=("y", "mean"), hdd=("hdd_day", "first"))
    d = d.dropna()
    return float(np.polyfit(d["hdd"], d["y"], 1)[0])


# ---------------------------------------------------------------- LightGBM
@dataclass
class LoadModel:
    """Point forecast + 10% and 90% quantiles, plus the band scale used by correction."""
    point: lgb.Booster
    lo: lgb.Booster | None
    hi: lgb.Booster | None
    features: list[str]
    train_end: str
    n_train_rows: int
    meta: dict = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xf = X[self.features]
        yhat = self.point.predict(Xf)
        out = pd.DataFrame({"yhat": yhat}, index=X.index)
        if self.lo is not None:
            lo, hi = self.lo.predict(Xf), self.hi.predict(Xf)
            out["lo"] = np.minimum(lo, yhat)
            out["hi"] = np.maximum(hi, yhat)
        return out

    def to_json(self) -> str:
        return json.dumps(dict(point=self.point.model_to_string(),
                               lo=self.lo.model_to_string() if self.lo else None,
                               hi=self.hi.model_to_string() if self.hi else None,
                               features=self.features, train_end=self.train_end,
                               n_train_rows=self.n_train_rows, meta=self.meta))

    @classmethod
    def from_json(cls, s: str) -> "LoadModel":
        d = json.loads(s)
        b = lambda t: lgb.Booster(model_str=t) if t else None  # noqa: E731
        return cls(b(d["point"]), b(d["lo"]), b(d["hi"]), d["features"], d["train_end"],
                   d["n_train_rows"], d.get("meta", {}))


def fit_lgb(X: pd.DataFrame, y: pd.Series, features: list[str] = FEATURES, quantiles: bool = True,
            weight: np.ndarray | None = None, params: dict | None = None, train_end: str = "") -> LoadModel:
    ok = y.notna().to_numpy()
    Xf, yf = X.loc[ok, features], y[ok]
    w = None if weight is None else np.asarray(weight)[ok]
    p = {**LGB_PARAMS, **(params or {})}
    point = lgb.LGBMRegressor(**p).fit(Xf, yf, sample_weight=w).booster_
    lo = hi = None
    if quantiles:
        qp = {**p, "n_estimators": QUANTILE_ESTIMATORS}
        lo = lgb.LGBMRegressor(objective="quantile", alpha=0.1, **qp).fit(Xf, yf, sample_weight=w).booster_
        hi = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **qp).fit(Xf, yf, sample_weight=w).booster_
    return LoadModel(point, lo, hi, list(features), train_end, int(ok.sum()))


def recent_weight(index: pd.DatetimeIndex, end: pd.Timestamp, days: int = 60, factor: float = 3.0) -> np.ndarray:
    """Weight the last `days` days `factor` times more (used by the persistent-bias retrain)."""
    return np.where(index >= end - pd.Timedelta(days=days), factor, 1.0)


# ---------------------------------------------------------------- storage
def save(con: sqlite3.Connection, name: str, model: LoadModel) -> None:
    db.upsert(con, "model_store", [dict(name=name, saved_at=db.utcnow(), train_end=model.train_end,
                                         model_json=model.to_json())])


def load(con: sqlite3.Connection, name: str) -> LoadModel | None:
    row = con.execute("SELECT model_json FROM model_store WHERE name = ?", (name,)).fetchone()
    return LoadModel.from_json(row[0]) if row else None
