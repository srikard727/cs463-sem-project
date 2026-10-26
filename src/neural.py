"""Neural network models (PyTorch): an MLP and an LSTM sequence model.

Do not import this module in the same Python process as XGBoost (see
``src/models.py``): the two libraries' OpenMP runtimes can crash together on
macOS.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from . import config
from . import data_prep as dp

# Highest log expected sales the networks may output: exp(8) is about 3,000
# units a day, far above any real day in the data.
MAX_LOG_SALES = 8.0

# Sales-history features the LSTM replaces with its raw sales sequence
HISTORY_FEATURES = [c for c in dp.SALES_FEATURES if c != "product_age"]

# Skewed count-like features that are log-transformed before scaling
LOG_FEATURES = [c for c in dp.SALES_FEATURES if c not in ("has_last_year", "zero_share_28", "zero_share_91")]


def series_key(frame):
    """Returns the product-store identifier ("store|item") for each row."""
    return frame["store_id"].astype(str) + "|" + frame["item_id"].astype(str)


class Preprocessor:
    """Prepares network inputs: log transform, standardization, clipping, series index.

    Parameters
    ----------
    features : list of str
        Model input columns, in order.
    """

    def __init__(self, features):
        self.features = list(features)
        self.log_positions = [self.features.index(c) for c in LOG_FEATURES if c in self.features]
        self.scaler = StandardScaler()
        self.series_ids = None

    def _matrix(self, frame):
        X = frame[self.features].to_numpy(dtype=np.float32, copy=True)
        X[:, self.log_positions] = np.log1p(np.maximum(X[:, self.log_positions], 0))
        return X

    def fit(self, frame):
        """Learns the scaling and the list of product-store series from training rows."""
        self.scaler.fit(self._matrix(frame))
        self.series_ids = sorted(series_key(frame).unique())
        self._series_index = {s: i for i, s in enumerate(self.series_ids)}
        return self

    @property
    def unknown_series(self):
        """Embedding row used for series not seen in training (the average embedding)."""
        return len(self.series_ids)

    def transform(self, frame):
        """Returns standardized features clipped to ±10 std, and each row's series index."""
        X = np.clip(self.scaler.transform(self._matrix(frame)), -10, 10).astype(np.float32)
        series = series_key(frame).map(self._series_index).fillna(self.unknown_series).to_numpy(np.int64)
        return X, series


class DemandMLP(nn.Module):
    """Feed-forward network: features + series embedding -> log expected sales."""

    def __init__(self, n_features, n_series, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(n_series + 1, embedding_dim)   # +1: unknown series
        self.layers = nn.Sequential(
            nn.Linear(n_features + embedding_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, X, series):
        return self.layers(torch.cat([X, self.embedding(series)], dim=1)).squeeze(1)


class DemandLSTM(nn.Module):
    """Sequence model: an LSTM reads the raw sales history, then a dense head
    combines it with known-in-advance features and the series embedding."""

    def __init__(self, n_known, n_series, embedding_dim=16, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(n_series + 1, embedding_dim)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_known + embedding_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, sequence, known, series):
        _, (h, _) = self.lstm(sequence)
        return self.head(torch.cat([h[-1], known, self.embedding(series)], dim=1)).squeeze(1)


def set_unknown_embedding(model):
    """Sets the unknown-series embedding row to the average of the learned rows."""
    with torch.no_grad():
        model.embedding.weight[-1] = model.embedding.weight[:-1].mean(dim=0)
    return model


def fit_network(model, make_batch, n_fit, n_stop, epochs=8, batch_size=4096, patience=2, lr=1e-3):
    """Trains a network with Adam and a Poisson loss, keeping the best epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Network whose output is log expected sales.
    make_batch : callable
        ``make_batch(split, row_indices) -> (inputs_tuple, targets)`` for
        ``split`` in ``{"fit", "stop"}``.
    n_fit, n_stop : int
        Number of training and early-stopping rows.

    Returns
    -------
    tuple
        ``(model, history)``: the model with its best weights, and the
        early-stopping loss of each epoch.
    """
    torch.manual_seed(config.SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.PoissonNLLLoss(log_input=True)
    rng = np.random.default_rng(config.SEED)
    best_loss, best_state, waited, history = np.inf, None, 0, []

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(n_fit)
        for start in range(0, n_fit, batch_size):
            inputs, y = make_batch("fit", order[start:start + batch_size])
            optimizer.zero_grad()
            loss_fn(model(*inputs), y).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            losses = [loss_fn(model(*inputs), y).item() * len(y)
                      for inputs, y in (make_batch("stop", np.arange(s, min(s + 16384, n_stop)))
                                        for s in range(0, n_stop, 16384))]
        stop_loss = sum(losses) / n_stop
        history.append(stop_loss)

        if stop_loss < best_loss:
            best_loss, waited = stop_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def predict_network(model, make_batch, split, n, batch_size=16384):
    """Returns expected units sold for ``n`` rows of ``split``."""
    model.eval()
    with torch.no_grad():
        return np.concatenate([
            torch.exp(model(*make_batch(split, np.arange(s, min(s + batch_size, n)))[0]).clamp(max=MAX_LOG_SALES)).numpy()
            for s in range(0, n, batch_size)
        ])


class SalesHistory:
    """Raw daily sales for every product-store series in a state, for the LSTM.

    Attributes
    ----------
    matrix : numpy.ndarray
        Series x days matrix of ``log(1 + units sold)``.
    """

    def __init__(self, state):
        sales = pd.read_csv(config.SALES_FILE)
        sales = sales[sales["state_id"] == state].reset_index(drop=True)
        day_cols = [c for c in sales.columns if c.startswith("d_")]
        self.matrix = np.log1p(sales[day_cols].to_numpy(np.float32))
        self._row = pd.Series(np.arange(len(sales)), index=sales["store_id"] + "|" + sales["item_id"])
        calendar = pd.read_csv(config.CALENDAR_FILE, usecols=["date"], parse_dates=["date"])
        self._day = pd.Series(np.arange(len(calendar)), index=calendar["date"])

    def rows_and_days(self, frame):
        """Returns each row's series row in ``matrix`` and its target day column."""
        return (series_key(frame).map(self._row).to_numpy(np.int64),
                frame["date"].map(self._day).to_numpy(np.int64))

    def windows(self, rows, days, length):
        """Returns ``length`` days of history ending ``HORIZON`` days before each target day.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_rows, length, 1)``, oldest day first.
        """
        offsets = np.arange(-(config.HORIZON + length - 1), -config.HORIZON + 1)
        return self.matrix[rows[:, None], days[:, None] + offsets][:, :, None]
