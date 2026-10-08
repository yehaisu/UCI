"""
Capital Bikeshare — next-day total rentals forecasting, tracked with real MLflow.

Operational setup:
  * Forecast tomorrow's total rentals every evening, using only info available then.
  * Strict time split: train 2011, test 2012 (no shuffling, no future leakage).
  * Benchmark to beat: naive  forecast_t = rentals_{t-1}.
  * Every run is traceable (data period, features, hyper-params, metrics, tags).
  * The best model is REGISTERED as a named version "BikeShareForecaster".

MLflow APIs used (per course deck):
  mlflow.set_experiment / start_run / log_params / log_metrics / set_tag /
  log_input (dataset) / pyfunc.log_model / register_model.
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.pyfunc

# ----------------------------------------------------------------------
# paths + tracking setup
# ----------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "day.csv")
MODELS_DIR = os.path.join(ROOT, "models")
ASSETS_DIR = os.path.join(ROOT, "docs", "assets")
for d in (MODELS_DIR, ASSETS_DIR):
    os.makedirs(d, exist_ok=True)

# local SQLite store supports both runs and the Model Registry
mlflow.set_tracking_uri("sqlite:///" + os.path.join(ROOT, "mlflow.db"))
mlflow.set_experiment("CapitalBikeshare")

# ----------------------------------------------------------------------
# numpy preprocessor + models (unchanged, pure numpy)
# ----------------------------------------------------------------------
class NumPyPreprocessor:
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

class RidgeNumpy:
    def __init__(self, alpha=0.1): self.alpha = alpha
    def fit(self, X, y):
        Xa = np.hstack([np.ones((len(X), 1)), X])
        I = np.eye(Xa.shape[1]); I[0, 0] = 0
        self.beta_ = np.linalg.solve(Xa.T @ Xa + self.alpha * I, Xa.T @ y)
        return self
    def predict(self, X):
        Xa = np.hstack([np.ones((len(X), 1)), X])
        return Xa @ self.beta_

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b):  return float(np.mean(np.abs(a - b)))
def r2(a, b):
    return float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))

# pyfunc wrapper so a (preprocessor + numpy model) pair can be logged by MLflow
class Forecaster(mlflow.pyfunc.PythonModel):
    def __init__(self, pre, model, feat_cols):
        self.pre, self.model, self.feat_cols = pre, model, feat_cols
    def predict(self, context, model_input):
        return self.model.predict(self.pre.transform(pd.DataFrame(model_input)[self.feat_cols]))

# ----------------------------------------------------------------------
# load + time split + lag features (known the previous evening)
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["dteday"]).sort_values("dteday").reset_index(drop=True)
TARGET = "cnt"
df["lag1"] = df[TARGET].shift(1)
df["lag7"] = df[TARGET].shift(7)
df = df.dropna(subset=["lag1", "lag7"]).reset_index(drop=True)

train = df[df["yr"] == 0].copy()   # 2011
test = df[df["yr"] == 1].copy()   # 2012
y_train = train[TARGET].values
y_test = test[TARGET].values
print(f"train {len(train)} days | test {len(test)} days")

# ----------------------------------------------------------------------
# naive benchmark
# ----------------------------------------------------------------------
naive_pred = test[TARGET].shift(1).values
v = ~np.isnan(naive_pred)
naive_rmse = rmse(y_test[v], naive_pred[v])
naive_mae = mae(y_test[v], naive_pred[v])
print(f"[naive] RMSE={naive_rmse:.1f}  MAE={naive_mae:.1f}")

# ----------------------------------------------------------------------
# run each analyst under MLflow
# ----------------------------------------------------------------------
EXPERIMENT = [
    ("analystA_ar1", ["lag1"], [], ["lag1"], RidgeNumpy(0.1), {"alpha": 0.1}),
    ("analystB_weather", ["lag1", "lag7", "weathersit"], ["weathersit"], ["lag1", "lag7"],
     RidgeNumpy(0.1), {"alpha": 0.1}),
]

results = []
for name, feat_cols, cat_cols, num_cols, model, params in EXPERIMENT:
    with mlflow.start_run(run_name=name) as run:
        pre = NumPyPreprocessor(cat_cols, num_cols).fit(train[feat_cols])
        Xtr, Xte = pre.transform(train[feat_cols]), pre.transform(test[feat_cols])
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)

        err_rmse, err_mae, err_r2 = rmse(y_test, pred), mae(y_test, pred), r2(y_test, pred)
        imp = (naive_rmse - err_rmse) / naive_rmse * 100

        # --- Data Logging (course deck) ---
        mlflow.log_params({"model": "RidgeNumpy", "train_period": "2011", "test_period": "2012",
                           "features": ",".join(feat_cols), **params})
        mlflow.log_metrics({"test_rmse": err_rmse, "test_mae": err_mae, "test_r2": err_r2,
                            "naive_rmse": naive_rmse, "pct_vs_naive": imp})
        mlflow.set_tag("model_type", "linear_ridge")
        mlflow.set_tag("benchmark", "naive_lag1")
        mlflow.set_tag("beats_benchmark", str(err_rmse < naive_rmse))
        try:  # dataset input (optional; wrapped for robustness)
            mlflow.log_input(mlflow.data.from_pandas(train, targets=TARGET, name="bike_daily_2011"))
        except Exception as e:
            print("log_input skipped:", e)

        # log the model (not registered yet)
        mlflow.pyfunc.log_model(name="model", python_model=Forecaster(pre, model, feat_cols))

        results.append({"name": name, "run_id": run.info.run_id, "pred": pred,
                        "rmse": err_rmse, "mae": err_mae, "r2": err_r2, "imp": imp})
        print(f"[{name}] RMSE={err_rmse:.1f}  R2={err_r2:.3f}  vs naive={imp:+.1f}%")

# ----------------------------------------------------------------------
# register the best model as a NAMED version
# ----------------------------------------------------------------------
best = min(results, key=lambda r: r["rmse"])
mv = mlflow.register_model(f"runs:/{best['run_id']}/model", "BikeShareForecaster")
print(f"[registered] BikeShareForecaster v{mv.version} from {best['name']} "
      f"(RMSE={best['rmse']:.1f}, beats_naive={best['rmse'] < naive_rmse})")

# ----------------------------------------------------------------------
# charts (same as before)
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(df["dteday"], df[TARGET], color="#4472C4", lw=1)
plt.axvline(pd.Timestamp("2012-01-01"), color="grey", ls="--", label="train | test")
plt.title("Capital Bikeshare — daily total rentals (2011–2012)"); plt.ylabel("rentals"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(ASSETS_DIR, "rentals_over_time.png"), dpi=120); plt.close()

plt.figure(figsize=(10, 4.5))
plt.plot(test["dteday"], y_test, color="black", lw=1.5, label="actual")
plt.plot(test["dteday"], naive_pred, color="#C00000", lw=1, ls="--", label="naive (yesterday)")
plt.plot(test["dteday"], best["pred"], color="#2E9E5B", lw=1.5, label=f"best ({best['name']})")
plt.title("2012 test period — forecasts vs actual"); plt.ylabel("rentals"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(ASSETS_DIR, "forecast_vs_actual.png"), dpi=120); plt.close()

names = ["naive"] + [r["name"] for r in results]
rmses = [naive_rmse] + [r["rmse"] for r in results]
colors = ["#888888"] + ["#2E9E5B" if r["imp"] > 0 else "#C00000" for r in results]
plt.figure(figsize=(7, 4))
bars = plt.bar(names, rmses, color=colors)
for b, v in zip(bars, rmses):
    plt.text(b.get_x() + b.get_width()/2, v + 20, f"{v:.0f}", ha="center")
plt.ylabel("test RMSE (lower = better)"); plt.title("Forecast error vs naive benchmark")
plt.tight_layout(); plt.savefig(os.path.join(ASSETS_DIR, "metrics_bar.png"), dpi=120); plt.close()

# summary for the HTML report
import json
summary = {"naive_rmse": round(naive_rmse, 1), "naive_mae": round(naive_mae, 1),
           "best": best["name"], "best_rmse": round(best["rmse"], 1),
           "beats_benchmark": bool(best["rmse"] < naive_rmse),
           "registered_version": int(mv.version),
           "runs": [{"name": r["name"], "rmse": round(r["rmse"], 1), "mae": round(r["mae"], 1),
                     "r2": round(r["r2"], 3), "pct_vs_naive": round(r["imp"], 1)} for r in results]}
with open(os.path.join(ROOT, "runs", "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nDONE.", json.dumps(summary, indent=2))
