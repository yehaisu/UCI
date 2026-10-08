# Capital Bikeshare — Next-Day Rentals Forecast

NUS ISS assignment demo: forecast tomorrow's total bike rentals every evening,

compare against the naive baseline (`tomorrow = yesterday`), and keep a

**traceable record of every training run** with a **named, retrievable model version**.

## Pipeline



* Data: UCI Bike Sharing `data/day.csv` (2011–2012).

* Strict time split: train 2011, test 2012 (no shuffling, no future leakage).

* Drop `casual`/`registered` (they sum to the target `cnt`).

* Features: `lag1` (yesterday, known that evening), `lag7` (same day last week),

  `weathersit`.

* Two analysts:


  * `analystA_ar1` — Ridge on `lag1` only.

  * `analystB_weather` — Ridge on `lag1, lag7, weathersit`.

* Run tracking mimics MLflow: each run logs data period / features / params /

  metrics to `runs/runs.csv` and an isolated `runs/<run_id>/` folder.

* Best model is registered as `models/BikeShareForecaster_v1.pkl`.

## Result (2012 test)



| Run                        | RMSE       | vs naive   |
| -------------------------- | ---------- | ---------- |
| naive baseline             | 1248.0     | —          |
| analystA\_ar1              | 1244.6     | +0.3%      |
| **analystB\_weather (v1)** | **1056.6** | **−15.3%** |

## Run



```
pip install -r requirements.txt
python src/train.py
```

## Publish as GitHub Pages



```
git init && git add . && git commit -m "bikeshare report"
git branch -M main
git remote add origin git@github.com:<you>/<repo>.git
git push -u origin main
```

Then repo **Settings → Pages → Source:&#x20;**`main`**&#x20;/&#x20;**`/docs`.

Visit `https://<you>.github.io/<repo>/`.

## Layout



```
data/        day.csv (+ raw zip)
src/train.py training + experiment tracking (pure numpy)
runs/        runs.csv + per-run artifacts
models/      registered BikeShareForecaster_v1.pkl
docs/        index.html + assets/  -> GitHub Pages site
```