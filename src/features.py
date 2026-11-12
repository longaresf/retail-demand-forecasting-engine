"""Leakage-aware feature engineering for retail time series."""

from collections.abc import Sequence

import pandas as pd


def add_calendar_features(
    data: pd.DataFrame, date_col: str = "date"
) -> pd.DataFrame:
    """Add calendar components derived from each observation date.

    Args:
        data: Observations containing a datetime-compatible column.
        date_col: Name of the date column.

    Returns:
        A copy with year, month, day, day-of-week, and weekend features.

    Raises:
        ValueError: If the date column is absent or contains invalid dates.
    """
    if date_col not in data.columns:
        raise ValueError(f"Missing date column: {date_col}")
    result = data.copy()
    dates = pd.to_datetime(result[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{date_col} contains missing or invalid dates.")
    result["year"] = dates.dt.year
    result["month"] = dates.dt.month
    result["day"] = dates.dt.day
    result["day_of_week"] = dates.dt.dayofweek
    result["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype("int8")
    return result


def add_time_series_features(
    data: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "sales",
    group_cols: Sequence[str] | None = None,
    lags: Sequence[int] = (1, 7, 28),
    windows: Sequence[int] = (7, 28),
) -> pd.DataFrame:
    """Add calendar, lag, rolling-mean, and exponentially weighted features.

    All target-derived features are shifted by one observation, so the value
    being predicted is never included in its own predictors.

    Args:
        data: Observations with one or more demand series.
        date_col: Name of the observation date column.
        target_col: Name of the demand column.
        group_cols: Optional series identifiers, such as store and item.
        lags: Positive observation offsets for lag features.
        windows: Positive sizes for rolling and exponential means.

    Returns:
        A copy sorted by series and date with engineered feature columns.

    Raises:
        ValueError: If required columns are absent or lags/windows are invalid.
    """
    groups = list(group_cols or [])
    required = [date_col, target_col, *groups]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if any(lag <= 0 for lag in lags) or any(window <= 0 for window in windows):
        raise ValueError("Lags and window sizes must be positive.")

    result = data.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    if result[date_col].isna().any():
        raise ValueError(f"{date_col} contains missing or invalid dates.")
    result = result.sort_values([*groups, date_col]).reset_index(drop=True)
    result = add_calendar_features(result, date_col)

    for lag in lags:
        name = f"{target_col}_lag_{lag}"
        if groups:
            result[name] = result.groupby(groups, sort=False)[target_col].shift(lag)
        else:
            result[name] = result[target_col].shift(lag)

    for window in windows:
        rolling_name = f"{target_col}_rolling_mean_{window}"
        ewm_name = f"{target_col}_ewm_{window}"
        if groups:
            grouped = result.groupby(groups, sort=False)[target_col]
            result[rolling_name] = grouped.transform(
                lambda series: series.shift(1).rolling(window, min_periods=1).mean()
            )
            result[ewm_name] = grouped.transform(
                lambda series: series.shift(1).ewm(span=window, adjust=False).mean()
            )
        else:
            shifted = result[target_col].shift(1)
            result[rolling_name] = shifted.rolling(window, min_periods=1).mean()
            result[ewm_name] = shifted.ewm(span=window, adjust=False).mean()
    return result
