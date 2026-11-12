"""Error metrics and forecast comparison helpers."""

import numpy as np
import pandas as pd


def _validated_arrays(
    actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    actual_values = np.asarray(actual, dtype=float).reshape(-1)
    predicted_values = np.asarray(predicted, dtype=float).reshape(-1)
    if actual_values.size == 0:
        raise ValueError("Metric inputs must not be empty.")
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted values must have the same length.")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Metric inputs must contain only finite values.")
    return actual_values, predicted_values


def mean_absolute_error(
    actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series
) -> float:
    """Return mean absolute error."""
    actual_values, predicted_values = _validated_arrays(actual, predicted)
    return float(np.mean(np.abs(actual_values - predicted_values)))


def root_mean_squared_error(
    actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series
) -> float:
    """Return root mean squared error."""
    actual_values, predicted_values = _validated_arrays(actual, predicted)
    return float(np.sqrt(np.mean(np.square(actual_values - predicted_values))))


def wape(
    actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series
) -> float:
    """Return weighted absolute percentage error as a percentage."""
    actual_values, predicted_values = _validated_arrays(actual, predicted)
    denominator = np.abs(actual_values).sum()
    if denominator == 0:
        raise ValueError("WAPE is undefined when all actual values are zero.")
    return float(np.abs(actual_values - predicted_values).sum() / denominator * 100)


def compare_forecasts(
    actual: np.ndarray | pd.Series,
    forecasts: dict[str, np.ndarray | pd.Series],
) -> pd.DataFrame:
    """Calculate RMSE, MAE, and WAPE for multiple named forecasts.

    Args:
        actual: Actual target values.
        forecasts: Mapping from model name to predicted values.

    Returns:
        A metrics table sorted by WAPE then MAE.
    """
    rows = []
    for model_name, predicted in forecasts.items():
        rows.append(
            {
                "model": model_name,
                "rmse": root_mean_squared_error(actual, predicted),
                "mae": mean_absolute_error(actual, predicted),
                "wape": wape(actual, predicted),
            }
        )
    return pd.DataFrame(rows).sort_values(["wape", "mae"]).reset_index(drop=True)
