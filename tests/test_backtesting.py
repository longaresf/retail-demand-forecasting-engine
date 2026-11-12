import pandas as pd
import pytest

from src.backtesting import walk_forward_split


def test_walk_forward_split_preserves_chronology_and_expands_training() -> None:
    data = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=10), "sales": range(10)}
    ).sample(frac=1, random_state=3)

    splits = list(walk_forward_split(data, n_splits=2, test_size=2))

    assert [len(train) for train, _ in splits] == [6, 8]
    assert splits[0][1]["sales"].tolist() == [6, 7]
    assert splits[1][1]["sales"].tolist() == [8, 9]
    assert splits[0][0]["date"].max() < splits[0][1]["date"].min()


def test_walk_forward_rejects_insufficient_rows() -> None:
    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3)})

    with pytest.raises(ValueError, match="Not enough rows"):
        list(walk_forward_split(data, n_splits=2, test_size=2))
