"""Input validation and preparation for retail demand time series."""

from collections.abc import Sequence

import pandas as pd


def missing_value_report(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values by column.

    Args:
        data: Input observations.

    Returns:
        A DataFrame with missing counts and percentages, including zero-count
        columns.
    """
    counts = data.isna().sum()
    return pd.DataFrame(
        {
            "missing_count": counts,
            "missing_pct": (counts / max(len(data), 1) * 100).round(2),
        }
    )


def clean_sales_data(
    data: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "sales",
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Normalize dates and targets, discard unusable rows, and aggregate duplicates.

    Rows with unparseable dates or non-numeric/missing targets cannot be used to
    train or score a demand model and are removed. Duplicate date/item rows are
    summed, as is appropriate for transaction-level sales.

    Args:
        data: Source observations.
        date_col: Name of the observation date column.
        target_col: Name of the demand column.
        group_cols: Optional series identifiers, such as store and item.

    Returns:
        Cleaned observations sorted by series and date.

    Raises:
        ValueError: If required columns are absent or no valid observations remain.
    """
    groups = list(group_cols or [])
    required = [date_col, target_col, *groups]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    result = data.loc[:, required].copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    result[target_col] = pd.to_numeric(result[target_col], errors="coerce")
    result = result.dropna(subset=[date_col, target_col])
    if result.empty:
        raise ValueError("No valid observations remain after cleaning.")

    keys = [*groups, date_col]
    result = result.groupby(keys, as_index=False, sort=True, dropna=False)[
        target_col
    ].sum()
    return result.sort_values(keys).reset_index(drop=True)


def detect_seasonality(
    values: pd.Series,
    periods: Sequence[int] = (7, 30, 365),
    threshold: float = 0.3,
) -> dict[int, float]:
    """Calculate lag autocorrelation for candidate seasonal periods.

    Args:
        values: Ordered, numeric observations from one time series.
        periods: Positive lags to inspect.
        threshold: Minimum absolute autocorrelation used to flag a period.

    Returns:
        Mapping of candidate periods to their autocorrelation. Lags not shorter
        than the series length are omitted.

    Raises:
        ValueError: If a period is non-positive or threshold is outside [0, 1].
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if any(period <= 0 for period in periods):
        raise ValueError("All periods must be positive.")

    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    scores: dict[int, float] = {}
    for period in periods:
        if period < len(clean_values):
            correlation = clean_values.autocorr(lag=period)
            if pd.notna(correlation) and abs(correlation) >= threshold:
                scores[period] = float(correlation)
    return scores
