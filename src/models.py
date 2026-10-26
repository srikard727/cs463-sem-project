"""Model definitions for the classical (non-neural) models.

Neural networks live in ``src/neural.py``. PyTorch and XGBoost each bundle
their own OpenMP threading library and can crash when loaded in the same
Python process on macOS, so the two modules are never imported together.
"""

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

# Shared gradient-boosting settings (learning rate, tree size, sampling).
# n_estimators is an upper bound; early stopping picks the real number.
XGB_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    early_stopping_rounds=50,
    n_jobs=-1,
    random_state=config.SEED,
)

RANDOM_FOREST_PARAMS = dict(
    n_estimators=100,
    max_depth=14,
    min_samples_leaf=20,
    max_features=0.5,
    n_jobs=-1,
    random_state=config.SEED,
)


def make_linear(kind="linear", alpha=None):
    """Returns a standardize-then-fit linear model.

    Parameters
    ----------
    kind : {"linear", "ridge", "lasso"}
        Plain least squares, L2-penalized (Ridge) or L1-penalized (LASSO).
    alpha : float, optional
        Penalty strength for Ridge and LASSO.
    """
    if kind == "linear":
        model = LinearRegression()
    elif kind == "ridge":
        model = Ridge(alpha=alpha)
    elif kind == "lasso":
        model = Lasso(alpha=alpha, max_iter=2000)
    else:
        raise ValueError(f"unknown linear model: {kind}")
    return make_pipeline(StandardScaler(), model)


def make_random_forest():
    """Returns the Random Forest regressor (an ensemble of CART trees)."""
    return RandomForestRegressor(**RANDOM_FOREST_PARAMS)


def make_xgb_regressor(objective="reg:squarederror", **extra):
    """Returns an XGBoost regressor with the shared settings.

    Parameters
    ----------
    objective : str
        XGBoost loss, e.g. ``"reg:squarederror"`` or ``"reg:tweedie"``.
    **extra
        Additional parameters, e.g. ``tweedie_variance_power=1.5``.
    """
    return xgb.XGBRegressor(**XGB_PARAMS, objective=objective, **extra)


class HurdleRegressor:
    """Two-stage hurdle model for zero-heavy daily sales.

    Stage 1 is a classifier for the probability that a product sells at
    least one unit. Stage 2 is a regressor trained only on days with sales;
    it predicts ``sales - 1`` with a Poisson loss, so its forecast of units
    given a sale is always at least 1. The final forecast is::

        P(sales > 0) * (1 + predicted extra units)

    Based on the hurdle model for count data (Mullahy, 1986), with XGBoost
    for both stages.

    Parameters
    ----------
    params : dict, optional
        XGBoost settings shared by both stages. Defaults to ``XGB_PARAMS``.
    """

    def __init__(self, params=None):
        self.params = dict(XGB_PARAMS if params is None else params)
        self.classifier_ = None
        self.regressor_ = None

    def fit(self, X, y, eval_set=None):
        """Fits both stages.

        Parameters
        ----------
        X : array-like of shape (n_rows, n_features)
            Training features.
        y : array-like of shape (n_rows,)
            Units sold.
        eval_set : tuple, optional
            ``(X_stop, y_stop)`` rows used only for early stopping.

        Returns
        -------
        HurdleRegressor
            The fitted model.
        """
        y = np.asarray(y)
        sold = y > 0

        self.classifier_ = xgb.XGBClassifier(**self.params, objective="binary:logistic", eval_metric="logloss")
        self.regressor_ = xgb.XGBRegressor(**self.params, objective="count:poisson")

        if eval_set is None:
            self.classifier_.set_params(early_stopping_rounds=None)
            self.regressor_.set_params(early_stopping_rounds=None)
            self.classifier_.fit(X, sold.astype(int), verbose=False)
            self.regressor_.fit(X[sold], y[sold] - 1, verbose=False)
            return self

        X_stop, y_stop = eval_set
        y_stop = np.asarray(y_stop)
        sold_stop = y_stop > 0
        self.classifier_.fit(X, sold.astype(int), eval_set=[(X_stop, sold_stop.astype(int))], verbose=False)
        self.regressor_.fit(
            X[sold], y[sold] - 1,
            eval_set=[(X_stop[sold_stop], y_stop[sold_stop] - 1)], verbose=False,
        )
        return self

    def predict_proba_sale(self, X):
        """Returns stage 1: the probability that at least one unit sells."""
        return self.classifier_.predict_proba(X)[:, 1]

    def predict_units_if_sale(self, X):
        """Returns stage 2: expected units sold on days with a sale (always >= 1)."""
        return 1 + self.regressor_.predict(X)

    def predict(self, X):
        """Returns the expected units sold: P(sale) x units if it sells."""
        return self.predict_proba_sale(X) * self.predict_units_if_sale(X)

    def save(self, directory):
        """Saves both stages to ``directory`` as XGBoost JSON files."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.classifier_.save_model(directory / "stage1_classifier.json")
        self.regressor_.save_model(directory / "stage2_regressor.json")
        with open(directory / "params.json", "w") as f:
            json.dump(self.params, f, indent=2)

    @classmethod
    def load(cls, directory):
        """Loads a hurdle model saved with :meth:`save`."""
        directory = Path(directory)
        with open(directory / "params.json") as f:
            model = cls(json.load(f))
        model.classifier_ = xgb.XGBClassifier()
        model.classifier_.load_model(directory / "stage1_classifier.json")
        model.regressor_ = xgb.XGBRegressor()
        model.regressor_.load_model(directory / "stage2_regressor.json")
        return model
