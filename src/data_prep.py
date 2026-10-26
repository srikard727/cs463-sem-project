"""Data preparation: raw M5 files to one feature table per store.

For each store, the pipeline:

1. Builds a products x days matrix of unit sales and one of selling prices.
2. Marks the days before a product's first price as "not on sale yet".
3. Computes the sales-history features on the matrix, shifted by
   ``config.HORIZON`` days so they only use sales known on order day.
4. Reshapes to one row per product per day and adds calendar, event,
   price and product features.
5. Removes Christmas (store closed) and pre-launch days, flags suspected
   stockouts, and drops rows without enough history.
6. Splits by date into train / validation / test and saves Parquet files.

Each store is processed on its own to keep memory use low.
"""

import json

import numpy as np
import pandas as pd

from . import config

ID_COLUMNS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
EVENT_TYPES = ["Cultural", "National", "Religious", "Sporting"]
DEPARTMENTS = ["FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2"]

LAGS = [7, 14, 21, 28]
LAST_YEAR_LAG = 364
ROLLING_MEANS = [7, 28, 56, 91]
ROLLING_STDS = [7, 28]
ZERO_SHARE_WINDOWS = [28, 91]
MAX_DAYS_SINCE = 365
MAX_EVENT_DISTANCE = 60

# One-hot groups drop their first level so no column is a sum of others
# (avoids the "dummy variable trap" for linear models).
SALES_FEATURES = (
    [f"sales_lag_{lag}" for lag in LAGS]
    + ["sales_lag_364", "has_last_year"]
    + [f"rolling_mean_{w}" for w in ROLLING_MEANS]
    + [f"rolling_std_{w}" for w in ROLLING_STDS]
    + [f"zero_share_{w}" for w in ZERO_SHARE_WINDOWS]
    + ["trend_7_28", "trend_28_91", "days_since_sale", "product_age"]
)
PRICE_FEATURES = [
    "sell_price", "price_change", "price_pct_change",
    "price_discount", "price_vs_usual", "price_vs_department",
]
CALENDAR_FEATURES = (
    [f"wday_{d}" for d in range(2, 8)]
    + [f"month_{m}" for m in range(2, 13)]
    + ["day_of_month", "snap"]
)
EVENT_FEATURES = [f"event_type_{t}" for t in EVENT_TYPES] + ["days_until_event", "days_since_event"]
PRODUCT_FEATURES = [f"dept_{d}" for d in DEPARTMENTS[1:]]

# Columns kept in the saved files that are not model inputs
META_COLUMNS = ID_COLUMNS + ["date", "target_sales", "is_suspected_stockout", "stockout_prob", "zero_run_length"]


def store_features(state):
    """Returns the one-hot store columns for a state (first store dropped)."""
    return [f"store_{s}" for s in config.STATES[state][1:]]


def feature_columns(state):
    """Returns the ordered list of model input columns for a state."""
    return (
        SALES_FEATURES + PRICE_FEATURES + CALENDAR_FEATURES
        + EVENT_FEATURES + PRODUCT_FEATURES + store_features(state)
    )


def load_raw():
    """Loads the three raw M5 files.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(sales, calendar, prices)``.
    """
    sales = pd.read_csv(config.SALES_FILE)
    calendar = pd.read_csv(config.CALENDAR_FILE, parse_dates=["date"])
    prices = pd.read_csv(config.PRICES_FILE)
    return sales, calendar, prices


# ---------------------------------------------------------------- matrix helpers
def _shift(matrix, days):
    """Shifts a products x days matrix so the value at day t is the value at t - days."""
    out = np.full(matrix.shape, np.nan, dtype=np.float32)
    out[:, days:] = matrix[:, :-days]
    return out


def _rolling(matrix, window, how="mean"):
    """Rolling mean or standard deviation along the days axis; NaN until the window is full."""
    roll = pd.DataFrame(matrix.T).rolling(window, min_periods=window)
    result = roll.mean() if how == "mean" else roll.std()
    return result.to_numpy(dtype=np.float32).T


def _days_since(event_matrix, cap):
    """Days since the last True value up to and including each day, capped at ``cap``."""
    idx = np.arange(event_matrix.shape[-1])
    last = np.where(event_matrix, idx, -10_000)
    last = np.maximum.accumulate(last, axis=-1)
    return np.minimum(idx - last, cap).astype(np.float32)


# ---------------------------------------------------------------- feature blocks
def _sales_history_features(sales, on_sale):
    """Builds the sales-history features for one store.

    Parameters
    ----------
    sales : numpy.ndarray
        Products x days matrix of units sold.
    on_sale : numpy.ndarray
        Products x days boolean matrix, True once the product has a price.

    Returns
    -------
    dict of str to numpy.ndarray
        Products x days matrices, each using only sales from ``HORIZON`` or
        more days before the target day.
    """
    horizon = config.HORIZON
    history = np.where(on_sale, sales, np.nan).astype(np.float32)
    known = _shift(history, horizon)          # sales known on order day
    feats = {}

    for lag in LAGS:
        feats[f"sales_lag_{lag}"] = _shift(history, lag)
    for window in ROLLING_MEANS:
        feats[f"rolling_mean_{window}"] = _rolling(known, window, "mean")
    for window in ROLLING_STDS:
        feats[f"rolling_std_{window}"] = _rolling(known, window, "std")

    zero = np.where(np.isnan(known), np.nan, (known == 0).astype(np.float32))
    for window in ZERO_SHARE_WINDOWS:
        feats[f"zero_share_{window}"] = _rolling(zero, window, "mean")

    # Same weekday last year; filled with the 28-day average when unavailable
    last_year = _shift(history, LAST_YEAR_LAG)
    feats["has_last_year"] = (~np.isnan(last_year)).astype(np.float32)
    feats["sales_lag_364"] = np.where(np.isnan(last_year), feats["rolling_mean_28"], last_year)

    # Is demand rising or falling? Ratios of a short to a longer average
    feats["trend_7_28"] = np.clip((feats["rolling_mean_7"] + 0.1) / (feats["rolling_mean_28"] + 0.1), 0, 20)
    feats["trend_28_91"] = np.clip((feats["rolling_mean_28"] + 0.1) / (feats["rolling_mean_91"] + 0.1), 0, 20)

    # Days since the last sale, as of order day
    feats["days_since_sale"] = _shift(_days_since(sales > 0, MAX_DAYS_SINCE), horizon)

    # Days since the product first went on sale (known in advance)
    first_day = on_sale.argmax(axis=1)
    feats["product_age"] = (np.arange(sales.shape[1])[None, :] - first_day[:, None]).astype(np.float32)
    return feats


def _price_features(prices_store, item_ids, departments, day_weeks):
    """Builds weekly price features and expands them to products x days matrices.

    Prices are set per week, so the target day's price is treated as known on
    order day (a planned price).
    """
    weeks = np.unique(day_weeks)
    weekly = (
        prices_store.pivot(index="item_id", columns="wm_yr_wk", values="sell_price")
        .reindex(index=item_ids, columns=weeks)
    )
    previous = weekly.shift(1, axis=1)
    change = (weekly - previous).where(weekly.notna())
    pct_change = (change / previous).clip(-1, config.MAX_PRICE_CHANGE)

    usual_max = weekly.T.rolling(52, min_periods=1).max().T          # highest price in the past year
    usual_mean = weekly.T.expanding(min_periods=1).mean().T          # average price so far
    dept_median = weekly.groupby(departments).transform("median")    # typical price in the department

    weekly_feats = {
        "sell_price": weekly,
        "price_change": change.fillna(0).where(weekly.notna()),
        "price_pct_change": pct_change.fillna(0).where(weekly.notna()),
        "price_discount": 1 - weekly / usual_max,
        "price_vs_usual": weekly / usual_mean,
        "price_vs_department": weekly / dept_median,
    }
    positions = np.searchsorted(weeks, day_weeks)
    return {name: frame.to_numpy(dtype=np.float32)[:, positions] for name, frame in weekly_feats.items()}


def _calendar_features(calendar, state, n_days):
    """Builds one value per day for every calendar and event feature."""
    cal = calendar.iloc[:n_days]
    feats = {}
    for d in range(2, 8):
        feats[f"wday_{d}"] = (cal["wday"] == d).to_numpy(np.float32)
    for m in range(2, 13):
        feats[f"month_{m}"] = (cal["month"] == m).to_numpy(np.float32)
    feats["day_of_month"] = cal["date"].dt.day.to_numpy(np.float32)
    feats["snap"] = cal[f"snap_{state}"].to_numpy(np.float32)

    for event_type in EVENT_TYPES:
        feats[f"event_type_{event_type}"] = (
            (cal["event_type_1"] == event_type) | (cal["event_type_2"] == event_type)
        ).to_numpy(np.float32)

    # Distances use the full calendar so events after the last sales day count
    has_event = calendar["event_name_1"].notna().to_numpy()
    since = _days_since(has_event, MAX_EVENT_DISTANCE)
    until = _days_since(has_event[::-1], MAX_EVENT_DISTANCE)[::-1]
    feats["days_since_event"] = since[:n_days]
    feats["days_until_event"] = until[:n_days]
    return feats


# ---------------------------------------------------------------- stockouts
def flag_suspected_stockouts(frame):
    """Flags zero-sales runs that are too unlikely to be real demand.

    A run of ``L`` zero days for a product whose normal rate is ``r`` units a
    day (its 28-day average at the start of the run) has probability
    ``exp(-r * L)`` under a Poisson model. Runs below ``config.STOCKOUT_P``
    are flagged. The flag uses the length of the whole run, so it must only be
    used to choose training rows and for scoring, never as a model input.

    Parameters
    ----------
    frame : pandas.DataFrame
        One store's rows sorted by product and date, with ``target_sales``,
        ``rolling_mean_28`` and ``item_id`` columns.

    Returns
    -------
    pandas.DataFrame
        ``frame`` with ``zero_run_length``, ``stockout_prob`` and
        ``is_suspected_stockout`` columns added.
    """
    is_zero = frame["target_sales"].to_numpy() == 0
    item = frame["item_id"].to_numpy()
    same_item = np.r_[False, item[1:] == item[:-1]]
    previous_zero = np.r_[False, is_zero[:-1]]

    run_start = is_zero & ~(previous_zero & same_item)
    run_id = np.where(is_zero, np.cumsum(run_start), 0)

    run_length = np.bincount(run_id)
    start_rate = np.full(run_length.shape, np.nan)
    start_rate[run_id[run_start]] = frame["rolling_mean_28"].to_numpy()[run_start]
    run_prob = np.exp(-start_rate * run_length)

    frame["zero_run_length"] = np.where(is_zero, run_length[run_id], 0).astype(np.int16)
    frame["stockout_prob"] = np.where(is_zero, run_prob[run_id], np.nan).astype(np.float32)
    frame["is_suspected_stockout"] = (frame["stockout_prob"] < config.STOCKOUT_P).astype(np.int8)
    return frame


# ---------------------------------------------------------------- per store
def build_store_table(store, sales, calendar, prices):
    """Builds the cleaned, feature-complete table for one store.

    Returns
    -------
    tuple
        ``(table, summary)``: the table has one row per product per day, and
        the summary counts the rows removed or flagged at each step.
    """
    state = store.split("_")[0]
    store_sales = sales[sales["store_id"] == store].reset_index(drop=True)
    day_cols = [c for c in store_sales.columns if c.startswith("d_")]
    n_items, n_days = len(store_sales), len(day_cols)
    assert (calendar["d"].iloc[:n_days] == day_cols).all(), "calendar and sales days do not line up"

    units = store_sales[day_cols].to_numpy(np.float32)
    day_weeks = calendar["wm_yr_wk"].iloc[:n_days].to_numpy()

    matrices = _price_features(
        prices[prices["store_id"] == store], store_sales["item_id"],
        store_sales["dept_id"].to_numpy(), day_weeks,
    )
    on_sale = ~np.isnan(matrices["sell_price"])
    matrices.update(_sales_history_features(units, on_sale))
    daily = _calendar_features(calendar, state, n_days)

    christmas = ((calendar["date"].dt.month == 12) & (calendar["date"].dt.day == 25)).to_numpy()[:n_days]
    keep = on_sale & ~christmas[None, :]
    item_idx, day_idx = np.nonzero(keep)          # row-major: sorted by product, then day

    table = pd.DataFrame({col: store_sales[col].to_numpy()[item_idx] for col in ID_COLUMNS})
    table["date"] = calendar["date"].to_numpy()[day_idx]
    table["target_sales"] = units[item_idx, day_idx].astype(np.int16)
    for name, matrix in matrices.items():
        table[name] = matrix[item_idx, day_idx]
    for name, values in daily.items():
        table[name] = values[day_idx]
    for dept in DEPARTMENTS[1:]:
        table[f"dept_{dept}"] = (table["dept_id"] == dept).astype(np.float32)
    for column in store_features(state):
        table[column] = np.float32(column == f"store_{store}")

    table = flag_suspected_stockouts(table)

    enough_history = table["rolling_mean_91"].notna()
    summary = {
        "store": store,
        "product_days": n_items * n_days,
        "removed_not_on_sale": int((~on_sale).sum()),
        "removed_christmas": int((on_sale & christmas[None, :]).sum()),
        "removed_short_history": int((~enough_history).sum()),
    }
    table = table[enough_history].reset_index(drop=True)
    summary["rows_kept"] = len(table)
    summary["suspected_stockouts"] = int(table["is_suspected_stockout"].sum())

    features = feature_columns(state)
    missing = table[features].isna().sum()
    assert missing.sum() == 0, f"{store}: missing feature values\n{missing[missing > 0]}"
    for column in ID_COLUMNS:
        table[column] = table[column].astype("category")
    return table[META_COLUMNS + features], summary


def split_by_date(table):
    """Splits a table into train, validation and test sets by date."""
    train = table[table["date"] < config.TRAIN_END]
    validation = table[(table["date"] >= config.TRAIN_END) & (table["date"] < config.TEST_START)]
    test = table[table["date"] >= config.TEST_START]
    return {"train": train, "validation": validation, "test": test}


def prepare_state(state, sales, calendar, prices):
    """Builds and saves the train / validation / test files for every store in a state.

    Files are written to ``data/processed/<state>/<store>_<split>.parquet``,
    and the state's feature list to ``data/processed/<state>/features.json``.

    Returns
    -------
    pandas.DataFrame
        One summary row per store.
    """
    out_dir = config.PROCESSED_DIR / state
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for store in config.STATES[state]:
        table, summary = build_store_table(store, sales, calendar, prices)
        for split, part in split_by_date(table).items():
            part.to_parquet(out_dir / f"{store}_{split}.parquet", index=False)
            summary[f"{split}_rows"] = len(part)
        summaries.append(summary)
        del table
    with open(out_dir / "features.json", "w") as f:
        json.dump(feature_columns(state), f, indent=2)
    return pd.DataFrame(summaries)


def load_split(state, split, columns=None):
    """Loads one split for every store in a state as a single DataFrame."""
    parts = [
        pd.read_parquet(config.PROCESSED_DIR / state / f"{store}_{split}.parquet", columns=columns)
        for store in config.STATES[state]
    ]
    frame = pd.concat(parts, ignore_index=True)
    for column in ID_COLUMNS:
        if column in frame:
            frame[column] = frame[column].astype("category")
    return frame


def load_features(state):
    """Loads the saved feature list for a state."""
    with open(config.PROCESSED_DIR / state / "features.json") as f:
        return json.load(f)
