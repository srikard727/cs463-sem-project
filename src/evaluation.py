"""Forecast error metrics shared by every notebook.

Two levels of detail:

- :func:`forecast_metrics` returns the four core metrics (MAE, RMSE, WAPE,
  Bias). The modeling notebooks use it for tuning and cross-validation.
- :func:`regression_report` adds Poisson deviance, RMSSE and R² for the final
  evaluation in ``05_Evaluation.ipynb``.

Every function treats negative forecasts as 0, since sales cannot be
negative.
"""

import numpy as np
import pandas as pd

from . import config

# Display order of the metrics in every results table
METRICS = ["MAE", "RMSE", "WAPE", "Bias", "Poisson deviance", "RMSSE", "R2"]


def forecast_metrics(y_true, y_pred):
    """Computes the core error metrics for one set of forecasts.

    Parameters
    ----------
    y_true : array-like
        Actual units sold.
    y_pred : array-like
        Forecast units (negative forecasts are treated as 0).

    Returns
    -------
    dict
        ``MAE`` (average miss in units), ``RMSE`` (penalizes big misses),
        ``WAPE`` (total absolute miss / total units, comparable across
        periods and states) and ``Bias`` (average forecast / average actual
        - 1; negative means under-forecasting).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(0, np.asarray(y_pred, dtype=float))
    error = y_pred - y_true
    return {
        "MAE": np.abs(error).mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "WAPE": np.abs(error).sum() / y_true.sum(),
        "Bias": y_pred.mean() / y_true.mean() - 1,
    }


def poisson_deviance(y_true, y_pred, floor=1e-6):
    """Returns the mean Poisson deviance (lower is better).

    This measures how well the forecast works as the *expected* count of a
    Poisson variable. It is the metric that matches the Poisson and Tweedie
    losses used in training. Forecasts are floored at ``floor`` so that a
    forecast of 0 on a day with sales gives a large but finite penalty.

    Parameters
    ----------
    y_true, y_pred : array-like
        Actual and forecast units.
    floor : float
        Smallest forecast used in the logarithm.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(floor, np.asarray(y_pred, dtype=float))
    log_term = np.where(y_true > 0, y_true * np.log(np.where(y_true > 0, y_true, 1) / y_pred), 0.0)
    return float(np.mean(2 * (log_term - (y_true - y_pred))))


def r_squared(y_true, y_pred):
    """Returns R², the share of the variance in daily sales that the forecast explains.

    Reported for reference only. On zero-heavy count data most of the variance
    comes from a few very high-selling days, so R² mainly measures those days.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(0, np.asarray(y_pred, dtype=float))
    return float(1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())


def rmsse_scales(state, sales=None):
    """Computes the RMSSE scale of every product-store series in a state.

    Follows the M5 competition: the scale is the mean squared day-to-day
    change in sales over the training period (dates before
    ``config.TRAIN_END``), counted from the series' first sale. A series that
    never sold in training has no scale and is left out of RMSSE.

    Parameters
    ----------
    state : str
        State code, e.g. ``"CA"``.
    sales : pandas.DataFrame, optional
        The raw ``sales_train_validation.csv`` table. It is read from disk if
        not given.

    Returns
    -------
    pandas.Series
        Scale per series, indexed by ``"store_id|item_id"``.
    """
    if sales is None:
        sales = pd.read_csv(config.SALES_FILE)
    calendar = pd.read_csv(config.CALENDAR_FILE, usecols=["d", "date"], parse_dates=["date"])
    train_days = calendar.loc[calendar["date"] < config.TRAIN_END, "d"]
    sales = sales[sales["state_id"] == state]

    units = sales[[d for d in train_days if d in sales.columns]].to_numpy(dtype=float)
    started = np.maximum.accumulate(units > 0, axis=1)       # True from the first sale onward
    change = np.where(started[:, 1:] & started[:, :-1], np.diff(units, axis=1) ** 2, np.nan)
    with np.errstate(invalid="ignore"):
        scale = np.nanmean(np.where(np.isnan(change).all(axis=1, keepdims=True), 0, change), axis=1)

    index = sales["store_id"].astype(str) + "|" + sales["item_id"].astype(str)
    return pd.Series(scale, index=index.to_numpy()).replace(0, np.nan).dropna()


def rmsse(y_true, y_pred, series, scales):
    """Returns the root mean squared scaled error, averaged over series.

    Each series' RMSE is divided by how much that series normally changes
    from day to day, so slow and fast sellers count equally. A value of 1
    means the forecast misses by about as much as sales normally change from
    one day to the next. The M5 competition used a sales-weighted version
    (WRMSSE). This is the unweighted mean.

    Parameters
    ----------
    y_true, y_pred : array-like
        Actual and forecast units.
    series : array-like of str
        ``"store_id|item_id"`` key of each row.
    scales : pandas.Series
        Output of :func:`rmsse_scales`.
    """
    frame = pd.DataFrame({
        "series": np.asarray(series),
        "sq_error": (np.maximum(0, np.asarray(y_pred, dtype=float)) - np.asarray(y_true, dtype=float)) ** 2,
    })
    mse = frame.groupby("series", sort=False)["sq_error"].mean()
    scale = scales.reindex(mse.index)
    return float(np.sqrt(mse / scale).dropna().mean())


def regression_report(y_true, y_pred, series=None, scales=None):
    """Computes every metric in :data:`METRICS` for one set of forecasts.

    Parameters
    ----------
    y_true, y_pred : array-like
        Actual and forecast units.
    series : array-like of str, optional
        Series key per row, needed for RMSSE.
    scales : pandas.Series, optional
        RMSSE scales from :func:`rmsse_scales`. If ``series`` or ``scales`` is
        missing, RMSSE is NaN.

    Returns
    -------
    dict
        Metric name to value, in :data:`METRICS` order.
    """
    report = forecast_metrics(y_true, y_pred)
    report["Poisson deviance"] = poisson_deviance(y_true, y_pred)
    report["RMSSE"] = rmsse(y_true, y_pred, series, scales) if series is not None and scales is not None else np.nan
    report["R2"] = r_squared(y_true, y_pred)
    return {name: report[name] for name in METRICS}


def forecast_value(y_true, y_pred, price):
    """Splits forecast errors into the shelf-price value of over- and under-forecast units.

    Over-forecast units are stock bought but not sold. Under-forecast units
    are sales that the order would not have covered.

    Parameters
    ----------
    y_true, y_pred : array-like
        Actual and forecast units.
    price : array-like
        Selling price of each row.

    Returns
    -------
    dict
        ``over ($)`` and ``under ($)``, the dollar totals of each kind of error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(0, np.asarray(y_pred, dtype=float))
    price = np.asarray(price, dtype=float)
    return {
        "over ($)": float((np.maximum(y_pred - y_true, 0) * price).sum()),
        "under ($)": float((np.maximum(y_true - y_pred, 0) * price).sum()),
    }
