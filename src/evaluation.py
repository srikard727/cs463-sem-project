"""Forecast error metrics shared by every notebook."""

import numpy as np


def forecast_metrics(y_true, y_pred):
    """Computes the error metrics for one set of forecasts.

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
