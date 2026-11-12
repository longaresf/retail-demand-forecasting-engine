import pandas as pd
import pytest

from src.features import add_time_series_features


def test_features_are_sorted_and_do_not_include_current_target() -> None:
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-03"]),
            "sales": [20, 10, 30],
        }
    )

    result = add_time_series_features(
        source, lags=(1,), windows=(2,)
    )

    assert result["sales_lag_1"].tolist()[1:] == [10, 20]
    assert result.loc[1, "sales_rolling_mean_2"] == 10
    assert result.loc[1, "sales_rolling_mean_2"] != result.loc[1, "sales"]
    assert result["is_weekend"].tolist() == [0, 0, 0]


def test_grouped_features_do_not_cross_series_boundaries() -> None:
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
            "store": [1, 1, 2],
            "sales": [10, 20, 99],
        }
    )

    result = add_time_series_features(
        source, group_cols=["store"], lags=(1,), windows=(2,)
    )

    store_two = result.loc[result["store"] == 2].iloc[0]
    assert pd.isna(store_two["sales_lag_1"])
    assert pd.isna(store_two["sales_rolling_mean_2"])


def test_features_reject_invalid_lag() -> None:
    with pytest.raises(ValueError, match="positive"):
        add_time_series_features(
            pd.DataFrame({"date": ["2024-01-01"], "sales": [1]}), lags=(0,)
        )
