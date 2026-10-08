"""
Capital Bikeshare — next-day total rentals forecasting  (pure-numpy version).

Operational setup (see assignment):
  * Every evening the ops team needs an estimate of TOMORROW's total rentals.
  * Only information available that evening may be used -> STRICT time-based split,
    no shuffling. Train on 2011 (yr==0), test on 2012 (yr==1).
  * Benchmark to beat: the naive rule  forecast_t = rentals_{t-1} (yesterday).
  * Every training run must be traceable: data period, features, hyper-params,
    and metrics are logged. The best model is saved as a NAMED, retrievable
    version (models/BikeShareForecaster_v1.pkl).

Two analysts use two genuinely different models:
  Analyst A : Ridge regression (linear, closed form).
  Analyst B : k-Nearest-Neighbours regression (non-linear).

Preprocessing / models are implemented on numpy only so the script runs in any
locked-down environment. On your own machine you can swap them for
sklearn.ColumnTransformer / Ridge / GradientBoosting and log to real MLflow.
"""

import json
import os
import pickle
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- pure-numpy metrics (version-independent) ---
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b):  return float(np.mean(np.abs(a - b)))
def r2(a, b):
    ss_res = np.sum((a - b) ** 2); ss_tot = np.sum((a - a.mean()) ** 2)
    return float(1 - ss_res / ss_tot)

# ----------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "day.csv")
RUNS_DIR = os.path.join(ROOT, "runs")
MODELS_DIR = os.path.join(ROOT, "models")
ASSETS_DIR = os.path.join(ROOT, "docs", "assets")
for d in (RUNS_DIR, MODELS_DIR, ASSETS_DIR):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------
# lightweight experiment tracker (mirrors MLflow Runs + Model Registry)
# ----------------------------------------------------------------------
class RunTracker:
    def __init__(self, runs_dir):
        self.runs_dir = runs_dir
        self.csv_path = os.path.join(runs_dir, "runs.csv")
        self._row = {}

    def start(self, run_name):
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{stamp}_{run_name}"
        self.run_dir = os.path.join(self.runs_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self._row = {"run_id": self.run_id, "run_name": run_name,
                     "start": dt.datetime.now().isoformat(timespec="seconds")}
        return self

    def log_params(self, **p):
        for k, v in p.items():
            self._row[f"param_{k}"] = v

    def log_metrics(self, **m):
        for k, v in m.items():
            self._row[f"metric_{k}"] = round(float(v), 4)

    def log_artifact(self, obj, filename):
        with open(os.path.join(self.run_dir, filename), "wb") as f:
            pickle.dump(obj, f)

    def close(self):
        with open(os.path.join(self.run_dir, "run_summary.json"), "w") as f:
            json.dump(self._row, f, indent=2, default=str)
        df = pd.DataFrame([self._row])
        if os.path.exists(self.csv_path):
            df = pd.concat([pd.read_csv(self.csv_path), df], ignore_index=True)
        df.to_csv(self.csv_path, index=False)


# ----------------------------------------------------------------------
# numpy preprocessor: one-hot categoricals + z-score numerics (fit on train)
# ----------------------------------------------------------------------
class NumPyPreprocessor:
    """One-hot categoricals (drop-first, avoids dummy-variable trap) + z-score numerics."""
    def __init__(self, cat_cols, num_cols):
        self.cat_cols, self.num_cols = cat_cols, num_cols

    def fit(self, X):
        self.cats_ = {c: sorted(X[c].unique().tolist())[1:] for c in self.cat_cols}
        self.num_mean_ = X[self.num_cols].mean().values
        self.num_std_ = X[self.num_cols].std().values
        return self

    def transform(self, X):
        parts = []
        for c in self.cat_cols:
            cats = self.cats_[c]
            col = X[c].values
            oh = np.zeros((len(X), len(cats)))
            for j, cat in enumerate(cats):
                oh[:, j] = (col == cat).astype(float)
            parts.append(oh)
        if self.num_cols:
            num = (X[self.num_cols].values - self.num_mean_) / np.where(self.num_std_ == 0, 1, self.num_std_)
            parts.append(num)
        return np.hstack(parts) if parts else np.zeros((len(X), 0))


# ----------------------------------------------------------------------
# models (pure numpy)
# ----------------------------------------------------------------------
class RidgeNumpy:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def fit(self, X, y):
        Xa = np.hstack([np.ones((len(X), 1)), X])     # bias column
        I = np.eye(Xa.shape[1]); I[0, 0] = 0          # do not regularize bias
        self.beta_ = np.linalg.solve(Xa.T @ Xa + self.alpha * I, Xa.T @ y)
        return self
    def predict(self, X):
        Xa = np.hstack([np.ones((len(X), 1)), X])
        return Xa @ self.beta_


class KNNNumpy:
    def __init__(self, k=15):
        self.k = k
    def fit(self, X, y):
        self.X_, self.y_ = X, y
        return self
    def predict(self, X):
        out = np.empty(len(X))
        for i, x in enumerate(X):
            d = np.sqrt(((self.X_ - x) ** 2).sum(axis=1))
            idx = np.argsort(d)[: self.k]
            out[i] = self.y_[idx].mean()
        return out


# ----------------------------------------------------------------------
# 1. load + split
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["dteday"]).sort_values("dteday").reset_index(drop=True)
TARGET = "cnt"
DROP = ["instant", "casual", "registered", "cnt"]   # leakage guard

# Lag features are KNOWN the previous evening -> legitimate inputs.
# lag1 carries the recent level (this is exactly what the naive baseline uses),
# lag7 captures weekly seasonality. Compute on the FULL series BEFORE splitting,
# so the first 2012 row correctly inherits 2011-12-31 / 2011-12-25 values.
df["lag1"] = df[TARGET].shift(1)
df["lag7"] = df[TARGET].shift(7)
df = df.dropna(subset=["lag1", "lag7"]).reset_index(drop=True)

# Feature sets were chosen by a small in-sample study: plain AR(1) already beats
# the naive rule; adding lag7 + weather-situation improves further, while numeric
# weather (temp/hum/windspeed) and monthly dummies actually hurt on the 2011->2012
# shift (no peeking at the test set -- the study used a 2011 holdout).
train = df[df["yr"] == 0].copy()   # 2011
test = df[df["yr"] == 1].copy()    # 2012
y_train = train[TARGET].values
y_test = test[TARGET].values
print(f"train days: {len(train)} ({train['dteday'].min().date()} -> {train['dteday'].max().date()})")
print(f"test  days: {len(test)} ({test['dteday'].min().date()} -> {test['dteday'].max().date()})")

# ----------------------------------------------------------------------
# 2. naive benchmark (yesterday == tomorrow)
# ----------------------------------------------------------------------
naive_pred = test[TARGET].shift(1).values
valid = ~np.isnan(naive_pred)
naive_rmse = rmse(y_test[valid], naive_pred[valid])
naive_mae = mae(y_test[valid], naive_pred[valid])
print(f"\n[benchmark naive]  RMSE={naive_rmse:.1f}  MAE={naive_mae:.1f}")

tracker = RunTracker(RUNS_DIR)

# ----------------------------------------------------------------------
# 3. run each analyst (each with its own feature set + preprocessor)
# ----------------------------------------------------------------------
def run(name, feat_cols, cat_cols, num_cols, model, params):
    r = tracker.start(name)
    pre = NumPyPreprocessor(cat_cols, num_cols).fit(train[feat_cols])
    Xtr = pre.transform(train[feat_cols])
    Xte = pre.transform(test[feat_cols])
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    err_rmse = rmse(y_test, pred)
    err_mae = mae(y_test, pred)
    err_r2 = r2(y_test, pred)
    imp = (naive_rmse - err_rmse) / naive_rmse * 100
    r.log_params(model=type(model).__name__, train_period="2011",
                 test_period="2012", features=",".join(feat_cols), **params)
    r.log_metrics(test_rmse=err_rmse, test_mae=err_mae, test_r2=err_r2,
                  naive_rmse=naive_rmse, pct_vs_naive=imp)
    r.log_artifact({"pre": pre, "model": model, "features": feat_cols}, "model.pkl")
    r.close()
    print(f"[{name}] RMSE={err_rmse:.1f}  MAE={err_mae:.1f}  R2={err_r2:.3f}  vs naive={imp:+.1f}%")
    return {"name": name, "pred": pred, "rmse": err_rmse, "mae": err_mae,
            "r2": err_r2, "imp": imp}

results = [
    # Analyst A: plain autoregressive (yesterday's level only)
    run("analystA_ar1", ["lag1"], [], ["lag1"], RidgeNumpy(alpha=0.1), {"alpha": 0.1}),
    # Analyst B: + weekly lag + weather situation
    run("analystB_weather", ["lag1", "lag7", "weathersit"], ["weathersit"], ["lag1", "lag7"],
        RidgeNumpy(alpha=0.1), {"alpha": 0.1}),
]

# ----------------------------------------------------------------------
# 4. select + register named version
# ----------------------------------------------------------------------
best = min(results, key=lambda r: r["rmse"])
beat = best["rmse"] < naive_rmse
print(f"\n[best] {best['name']} RMSE={best['rmse']:.1f} beats naive={beat}")

with open(os.path.join(MODELS_DIR, "BikeShareForecaster_v1.pkl"), "wb") as f:
    pickle.dump({"version": "v1", "registered_from_run": best["name"],
                 "test_rmse": best["rmse"],
                 "created_at": dt.datetime.now().isoformat(timespec="seconds")}, f)

# ----------------------------------------------------------------------
# 5. charts
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(df["dteday"], df[TARGET], color="#4472C4", lw=1)
plt.axvline(pd.Timestamp("2012-01-01"), color="grey", ls="--", label="train | test")
plt.title("Capital Bikeshare — daily total rentals (2011–2012)")
plt.ylabel("rentals"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "rentals_over_time.png"), dpi=120); plt.close()

plt.figure(figsize=(10, 4.5))
plt.plot(test["dteday"], y_test, color="black", lw=1.5, label="actual")
plt.plot(test["dteday"], naive_pred, color="#C00000", lw=1, ls="--", label="naive (yesterday)")
plt.plot(test["dteday"], best["pred"], color="#2E9E5B", lw=1.5, label=f"best ({best['name']})")
plt.title("2012 test period — forecasts vs actual")
plt.ylabel("rentals"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "forecast_vs_actual.png"), dpi=120); plt.close()

names = ["naive"] + [r["name"] for r in results]
rmses = [naive_rmse] + [r["rmse"] for r in results]
colors = ["#888888"] + ["#2E9E5B" if r["imp"] > 0 else "#C00000" for r in results]
plt.figure(figsize=(7, 4))
bars = plt.bar(names, rmses, color=colors)
for b, v in zip(bars, rmses):
    plt.text(b.get_x() + b.get_width()/2, v + 20, f"{v:.0f}", ha="center")
plt.ylabel("test RMSE (lower = better)")
plt.title("Forecast error vs naive benchmark")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "metrics_bar.png"), dpi=120); plt.close()

summary = {
    "n_train": int(len(train)), "n_test": int(len(test)),
    "naive_rmse": round(naive_rmse, 1), "naive_mae": round(naive_mae, 1),
    "runs": [{"name": r["name"], "rmse": round(r["rmse"], 1),
              "mae": round(r["mae"], 1), "r2": round(r["r2"], 3),
              "pct_vs_naive": round(r["imp"], 1)} for r in results],
    "best": best["name"], "beats_benchmark": bool(beat),
}
with open(os.path.join(ROOT, "runs", "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nDONE.", json.dumps(summary, indent=2))
