"""Forecast-residual monitor: one ridge predicts all 52 standardized channels at t from t-1 and t-2."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from common import CHANNELS, FAULT, KEYS, RESULTS, RUN, Standardizer, load

LAGS = 2
ALPHA = 1.0


def lag_names(lag: int) -> list[str]:
    """Return the column names of the 52 channels lagged by lag samples."""
    return [f"{c}_lag{lag}" for c in CHANNELS]


FEATURES = [c for lag in range(1, LAGS + 1) for c in lag_names(lag)]


def make_table(z: pl.DataFrame) -> pl.DataFrame:
    """Return the keys, the 104 lagged features and the 52 targets; the first LAGS samples of each run are dropped.

    z holds the keys and standardized channels, sorted by fault, run and sample.
    """
    lagged = [pl.col(c).shift(lag).over(FAULT, RUN).alias(name)
              for lag in range(1, LAGS + 1) for c, name in zip(CHANNELS, lag_names(lag))]
    return z.with_columns(lagged).drop_nulls(FEATURES)


class ForecastMonitor:
    """Ridge forecaster fitted on the training runs, with each channel's residual standard deviation."""

    def __init__(self, alpha: float = ALPHA):
        self.model = Ridge(alpha=alpha)
        self.res_std = None

    def fit(self, train: pl.DataFrame) -> ForecastMonitor:
        """Fit the ridge on the training table and store the residual standard deviation per channel."""
        x = train.select(FEATURES).to_numpy()
        y = train.select(CHANNELS).to_numpy()
        self.model.fit(x, y)
        self.res_std = (y - self.model.predict(x)).std(axis=0, ddof=1)
        return self

    def contributions(self, table: pl.DataFrame) -> np.ndarray:
        """Return the squared standardized residual of every channel, one row per sample."""
        x = table.select(FEATURES).to_numpy()
        y = table.select(CHANNELS).to_numpy()
        r = (y - self.model.predict(x)) / self.res_std
        return r ** 2

    def score(self, table: pl.DataFrame) -> pl.DataFrame:
        """Return the keys and score, the sum of the 52 squared standardized residuals."""
        return table.select(KEYS).with_columns(
            score=pl.Series(self.contributions(table).sum(axis=1)))


def run() -> pl.DataFrame:
    """Fit on runs 1 to 300 and write results/scores_ridge.parquet."""
    train, scored = load()
    std = Standardizer(train)
    monitor = ForecastMonitor().fit(make_table(std.transform(train)))
    scores = monitor.score(make_table(std.transform(scored)))

    RESULTS.mkdir(exist_ok=True)
    scores.write_parquet(RESULTS / "scores_ridge.parquet")
    print(f"{scores.height:,} rows scored")
    return scores


if __name__ == "__main__":
    run()
