"""Project-wide settings for the M5 retail demand forecasting project.

Every notebook and module reads its paths, dates and constants from here,
so a setting only has to be changed in one place.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Raw Kaggle M5 files (download from the "M5 Forecasting - Accuracy" competition)
RAW_DIR = PROJECT_DIR
SALES_FILE = RAW_DIR / "sales_train_validation.csv"
CALENDAR_FILE = RAW_DIR / "calendar.csv"
PRICES_FILE = RAW_DIR / "sell_prices.csv"

# Outputs written by the notebooks
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# ---------------------------------------------------------------- scope
STATES = {
    "CA": ["CA_1", "CA_2", "CA_3", "CA_4"],
    "TX": ["TX_1", "TX_2", "TX_3"],
    "WI": ["WI_1", "WI_2", "WI_3"],
}

# ---------------------------------------------------------------- forecasting setup
# The store manager orders once a week, so the forecast is made 7 days ahead:
# every sales-based feature must be at least HORIZON days old.
HORIZON = 7

# Chronological split: train < TRAIN_END <= validation < TEST_START <= test
TRAIN_END = "2014-10-08"
TEST_START = "2015-07-17"

# A run of zero-sales days is flagged as a suspected stockout when its
# probability under the product's normal Poisson sales rate is below this.
STOCKOUT_P = 0.001

# Largest week-over-week price change kept (+300%); larger jumps are capped.
MAX_PRICE_CHANGE = 3.0

SEED = 42
