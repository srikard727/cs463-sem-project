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
├── src/                             Shared, documented Python code
│   ├── config.py                    Paths, states and stores, horizon, split dates, seed
│   ├── data_prep.py                 Loading, cleaning, features, stockout flags, splits
│   ├── models.py                    Model builders and the HurdleRegressor class
│   ├── neural.py                    MLP, LSTM, training loop, input preprocessing
│   └── evaluation.py                Metrics: MAE, RMSE, WAPE, bias
│
├── requirements.txt                 Exact library versions
├── archive/                         Earlier single-store (CA_1) version and old files, kept for reference
│
│   Created by the notebooks (not in git):
├── data/processed/<state>/          Cleaned train / validation / test files per store
├── models/<state>/                  Every trained model for that state
└── results/<state>/                 Training records and saved validation forecasts
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
2. Download the M5 data from the Kaggle competition page and put these three
   files in the project folder:
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

## Status

Data preparation and model training are complete for all three states. The
test set has not been used yet, and model performance results are not
reported in this repository yet.

## References

- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy
  competition: Results, findings, and conclusions. *International Journal of
  Forecasting*.
- Croston, J. D. (1972). Forecasting and stock control for intermittent demands.
  *Operational Research Quarterly*.
- Mullahy, J. (1986). Specification and testing of some modified count data
  models. *Journal of Econometrics*.
