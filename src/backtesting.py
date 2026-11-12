"""Chronological expanding-window validation utilities."""

from collections.abc import Iterator

import pandas as pd


def walk_forward_split(
    data: pd.DataFrame,
    date_col: str = "date",
    n_splits: int = 3,
    test_size: int = 28,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield expanding training windows and non-overlapping future test windows.

    Args:
        data: Time-indexed observations.
        date_col: Name of the chronological ordering column.
        n_splits: Number of sequential validation folds.
        test_size: Number of rows in each validation fold.

    Yields:
        ``(train, validation)`` DataFrame pairs in chronological order.

    Raises:
        ValueError: If columns, split sizes, or available history are invalid.
    """
    if date_col not in data.columns:
        raise ValueError(f"Missing date column: {date_col}")
    if n_splits <= 0 or test_size <= 0:
        raise ValueError("n_splits and test_size must be positive.")
    ordered = data.sort_values(date_col).reset_index(drop=True)
    initial_train_size = len(ordered) - n_splits * test_size
    if initial_train_size < 1:
        raise ValueError("Not enough rows for the requested walk-forward splits.")

    for split_index in range(n_splits):
        train_end = initial_train_size + split_index * test_size
        test_end = train_end + test_size
        yield ordered.iloc[:train_end].copy(), ordered.iloc[train_end:test_end].copy()
