"""Baseline and tree-based demand forecasting models."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def _validate_horizon(horizon: int) -> None:
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")


def moving_average_forecast(
    history: Sequence[float], horizon: int, window: int = 7
) -> np.ndarray:
    """Forecast each future period using the latest moving average.

    Args:
        history: Historical target observations in chronological order.
        horizon: Number of future observations to predict.
        window: Number of recent observations included in the average.

    Returns:
        A constant forecast array of length ``horizon``.

    Raises:
        ValueError: If history is empty or horizon/window is invalid.
    """
    _validate_horizon(horizon)
    if window <= 0:
        raise ValueError("window must be a positive integer.")
    values = np.asarray(history, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("history must contain finite observations.")
    return np.full(horizon, values[-window:].mean(), dtype=float)


def historical_mean_forecast(
    history: Sequence[float], horizon: int
) -> np.ndarray:
    """Forecast each future period using the full-history average."""
    _validate_horizon(horizon)
    values = np.asarray(history, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("history must contain finite observations.")
    return np.full(horizon, values.mean(), dtype=float)


def seasonal_naive_forecast(
    history: Sequence[float], horizon: int, season_length: int = 7
) -> np.ndarray:
    """Repeat the most recent complete seasonal cycle into the future."""
    _validate_horizon(horizon)
    if season_length <= 0:
        raise ValueError("season_length must be a positive integer.")
    values = np.asarray(history, dtype=float)
    if values.size < season_length or not np.isfinite(values).all():
        raise ValueError("history must contain a finite complete seasonal cycle.")
    return np.resize(values[-season_length:], horizon).astype(float)


class XGBoostForecaster:
    """A tabular XGBoost regressor for features built from past observations."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        """Create a deterministic gradient-boosted tree forecaster."""
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_columns: list[str] = []

    def fit(
        self,
        data: pd.DataFrame,
        feature_columns: Sequence[str],
        target_col: str = "sales",
    ) -> "XGBoostForecaster":
        """Fit the model using only the supplied training observations.

        Args:
            data: Training observations and engineered features.
            feature_columns: Numeric predictor column names.
            target_col: Target column name.

        Returns:
            This fitted forecaster.

        Raises:
            ValueError: If features/target are missing or target has null values.
        """
        missing = [
            column
            for column in [*feature_columns, target_col]
            if column not in data.columns
        ]
        if missing:
            raise ValueError(f"Missing model columns: {missing}")
        if not feature_columns:
            raise ValueError("At least one feature column is required.")
        if data[target_col].isna().any():
            raise ValueError("Training target contains missing values.")
        if not np.isfinite(pd.to_numeric(data[target_col], errors="coerce")).all():
            raise ValueError("Training target must contain finite numeric values.")
        self.feature_columns = list(feature_columns)
        self.model.fit(data[self.feature_columns], data[target_col])
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict demand for rows containing the fitted feature columns."""
        if not self.feature_columns:
            raise RuntimeError("The forecaster must be fitted before prediction.")
        missing = [column for column in self.feature_columns if column not in data]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return self.model.predict(data[self.feature_columns])
