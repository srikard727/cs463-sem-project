# Retail Demand Forecasting (M5, Walmart)

CS 463 Machine Learning semester project, Fall 2026 · Srikar Devesetti

A store manager orders stock once a week. On order day, they need to know how
many units of each product will sell on each day of the coming week. This
project forecasts **daily unit sales for every product in every store, 7 days
ahead**, using only information known on order day.

- **Data:** the Kaggle [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
  dataset: 30,490 product-store series (3,049 products × 10 Walmart stores),
  January 2011 to April 2016.
- **Scope:** all 10 stores in California (4), Texas (3) and Wisconsin (3),
  with a **separate set of models for each state**. Within a state, one model
  learns from all of its stores.
- **Models (per state):** two naive baselines, Linear Regression, Ridge, LASSO,
  Random Forest, XGBoost (squared error and Tweedie loss), a custom
  **two-stage hurdle model**, an MLP, and an LSTM.

## Repository layout

```
├── 01_EDA.ipynb                     Exploratory data analysis
├── 02_Data_Preparation.ipynb        Cleans all 10 stores and builds the features
│
├── 03_Modeling.ipynb                TEMPLATE: classical models for one state (edit this one)
├── 03_Modeling_CA.ipynb             Saved run of the template for California (view only)
├── 03_Modeling_TX.ipynb             Saved run for Texas (view only)
├── 03_Modeling_WI.ipynb             Saved run for Wisconsin (view only)
│
├── 04_Neural_Networks.ipynb         TEMPLATE: MLP and LSTM for one state (edit this one)
├── 04_Neural_Networks_CA/TX/WI.ipynb   Saved runs for each state (view only)
│
├── 05_Evaluation.ipynb              Compares all 33 models, picks the final one, scores the test set
│
├── src/                             Shared, documented Python code
│   ├── config.py                    Paths, states and stores, horizon, split dates, seed
│   ├── data_prep.py                 Loading, cleaning, features, stockout flags, splits
│   ├── models.py                    Model builders and the HurdleRegressor class
│   ├── neural.py                    MLP, LSTM, training loop, input preprocessing
│   └── evaluation.py                Metrics: MAE, RMSE, WAPE, bias, Poisson deviance, RMSSE, R²
│
├── requirements.txt                 Exact library versions
│
│   Created by the notebooks (not in git):
├── data/processed/<state>/          Cleaned train / validation / test files per store
├── models/<state>/                  The 3 shortlisted models (only xgb_tweedie.json, the final model, is in git)
├── results/<state>/                 Training records, saved validation and test forecasts
└── results/evaluation/              Leaderboards, test results and figures from 05
```

### Templates vs. saved runs

`03_Modeling.ipynb` and `04_Neural_Networks.ipynb` are **templates**: each
handles one state, chosen by the `STATE` environment variable. Running a
template for a state saves a copy with that state's outputs, such as
`03_Modeling_WI.ipynb`.

- To change how the models train, **edit the template**, then re-run it.
- Treat the `_CA`, `_TX` and `_WI` files as **read-only records**. Running one
  of them directly with Run All trains **California** (the default), whatever
  the file name says.

## Setup

1. Use Python 3.14 and create a virtual environment:
   ```
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
2. Download the M5 data. The files are not in this repository: two are over
   GitHub's 100 MB file limit, and Kaggle's competition rules don't allow
   sharing the data. You need a Kaggle account. Accept the rules on the
   [competition page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules)
   and create an API token (Kaggle → Settings → API → Create New Token). Then,
   from the project folder:
   ```
   .venv/bin/pip install kaggle
   .venv/bin/kaggle competitions download -c m5-forecasting-accuracy
   unzip m5-forecasting-accuracy.zip sales_train_validation.csv calendar.csv sell_prices.csv
   ```
   Or download them by hand from the competition's Data tab. Either way, these
   three files must be in the project folder:
   `sales_train_validation.csv`, `calendar.csv`, `sell_prices.csv`.
3. In Jupyter or VS Code, select the `.venv` kernel.

## How to run

Run the steps in this order. Each state's 03 run must finish before its 04 run,
because 04 adds to the files 03 creates.

```
# 1. Data preparation for all 10 stores (about 2 minutes)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace 02_Data_Preparation.ipynb

# 2. Classical models, one state per run (about 12-17 minutes each)
for S in WI TX CA; do
  STATE=$S .venv/bin/jupyter nbconvert --to notebook --execute 03_Modeling.ipynb --output 03_Modeling_$S.ipynb
done

# 3. Neural networks, one state per run (about 8 minutes each)
for S in WI TX CA; do
  STATE=$S .venv/bin/jupyter nbconvert --to notebook --execute 04_Neural_Networks.ipynb --output 04_Neural_Networks_$S.ipynb
done

# 4. Evaluation for all three states (about 1 minute)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace 05_Evaluation.ipynb
```

To run a single state, use one line from a loop, e.g.
`STATE=TX .venv/bin/jupyter nbconvert --to notebook --execute 03_Modeling.ipynb --output 03_Modeling_TX.ipynb`.

Keep the computer awake and plugged in during training (on macOS, prefix the
commands with `caffeinate -ims`). Running two trainings at once can run out of
memory on a 16 GB machine.

**Why 03 and 04 are separate:** XGBoost and PyTorch each bundle their own
OpenMP threading library, and loading both in one Python process can freeze or
crash on macOS. Each notebook run starts a fresh process, so they never meet.

## Method summary

**Data preparation** (`src/data_prep.py`)
- One row per product, per store, per day (about 43 million rows after cleaning).
- Removed: days before a product's first price (not on sale yet), Christmas
  (stores closed), and each product's first 98 days (not enough history).
- **Suspected stockouts** (about 16% of rows): zero-sales runs with under a
  0.1% chance at the product's normal Poisson sales rate. They stay in the
  files but are excluded from training.
- Week-over-week price jumps capped at +300%.
- **57–58 features** per state: recent and last-year sales, rolling averages
  and spreads, demand shape (zero share, days since last sale, trends), price
  and discount features, calendar, SNAP days for the store's own state,
  events, department, product age, and store. Every sales feature uses only
  sales at least 7 days old. One-hot groups drop one level to avoid redundant
  columns.
- **Chronological split:** train before 2014-10-08 (from May 2011), validation
  until 2015-07-16, test from 2015-07-17 to 2016-04-24.

**Training** (`03_Modeling.ipynb`, `04_Neural_Networks.ipynb`)
- Clean training rows only (suspected stockouts removed).
- Early stopping (XGBoost, hurdle model, neural networks) uses the last 10% of
  training dates, never the validation set.
- Hyperparameters (Ridge and LASSO α, Tweedie power) are chosen by the lowest
  validation RMSE on non-stockout rows.
- Samples for the slower models: 2M rows (linear models), 1M (Random Forest,
  LSTM), 3M (MLP).
- The Random Forest is also cross-validated with `KFold` and `TimeSeriesSplit`.
- **Hurdle model** (`src.models.HurdleRegressor`): stage 1 is an XGBoost
  classifier for P(sales > 0); stage 2 is an XGBoost Poisson regressor on
  `sales − 1`, trained only on days with sales. Forecast =
  P(sales > 0) × (1 + predicted extra units).
- **Neural networks:** product-store embeddings, Poisson loss, Adam
  (lr 0.001), batch size 4,096, up to 8 epochs. The MLP uses all features; the
  LSTM reads the last 28 known days of raw sales.

**Evaluation and model selection** (`05_Evaluation.ipynb`)
- **Stage 1:** all 33 models (11 types × 3 states) are scored on validation.
- **Stage 2:** model types are ranked by validation RMSE in each state, and
  the top 3 by average rank go to the test set with the 28-day-average
  baseline. The validation winner is the final model. The test set is used
  once.
- **Metrics:** RMSE (selection), WAPE (headline), MAE, bias, Poisson
  deviance, RMSSE and R², all on non-stockout rows. MAPE is left out because
  most days have 0 sales. MAE-type metrics are not used for selection
  because they reward forecasts that run low on zero-heavy data.

## Results

The validation winner in the shortlist (XGBoost Tweedie, hurdle model,
XGBoost squared error) is **XGBoost with Tweedie loss**. Its test scores
(2015-07-17 to 2016-04-24, non-stockout rows):

| State | RMSE | WAPE | Bias | RMSSE | RMSE vs. 28-day baseline |
|---|---|---|---|---|---|
| CA | 2.206 | 66.9% | −1.2% | 0.879 | −7.3% |
| TX | 1.904 | 69.3% | −0.7% | 0.828 | −8.4% |
| WI | 2.165 | 66.9% | −2.2% | 0.862 | −9.4% |

- The baseline under-forecasts by 7–9%. The final model is within about 2%
  of actual sales, which cuts the shelf-price value of under-forecast units
  by 7–10%.
- Errors are lowest for regular sellers and Foods (WAPE about 55–60%) and
  highest for intermittent items and Hobbies (above 100%).
- Every table and figure is in `results/evaluation/`.

## References

- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy
  competition: Results, findings, and conclusions. *International Journal of
  Forecasting*.
- Croston, J. D. (1972). Forecasting and stock control for intermittent demands.
  *Operational Research Quarterly*.
- Mullahy, J. (1986). Specification and testing of some modified count data
  models. *Journal of Econometrics*.
