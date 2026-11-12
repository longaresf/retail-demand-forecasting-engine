import numpy as np

from data.sample_data import generate_sample_data
from src.backtesting import walk_forward_split
from src.evaluate import compare_forecasts
from src.features import add_time_series_features
from src.models import (
    XGBoostForecaster,
    moving_average_forecast,
    seasonal_naive_forecast,
)


FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "is_weekend",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_mean_28",
]


def test_walk_forward_splits_are_strictly_chronological() -> None:
    data = generate_sample_data(periods=80, stores=1, items=1).sample(
        frac=1, random_state=7
    )

    splits = list(walk_forward_split(data, n_splits=3, test_size=10))

    assert len(splits) == 3
    for train, test in splits:
        assert not train.empty
        assert not test.empty
        assert train["date"].max() < test["date"].min()
    assert splits[0][1]["date"].max() < splits[1][1]["date"].min()
    assert splits[1][1]["date"].max() < splits[2][1]["date"].min()


def test_baselines_and_xgboost_produce_finite_backtest_metrics() -> None:
    data = generate_sample_data(periods=160, stores=1, items=1)
    featured = add_time_series_features(
        data,
        group_cols=["store", "item"],
        lags=(1, 7, 28),
        windows=(7, 28),
    )
    train, test = next(
        walk_forward_split(featured, n_splits=1, test_size=21)
    )
    actual = test["sales"].to_numpy()

    model = XGBoostForecaster(n_estimators=20, max_depth=3).fit(
        train, FEATURE_COLUMNS
    )
    forecasts = {
        "moving_average": moving_average_forecast(
            train["sales"], horizon=len(test), window=7
        ),
        "seasonal_naive": seasonal_naive_forecast(
            train["sales"], horizon=len(test), season_length=7
        ),
        "xgboost": model.predict(test),
    }

    assert all(np.isfinite(prediction).all() for prediction in forecasts.values())
    metrics = compare_forecasts(actual, forecasts)
    assert set(metrics.columns) == {"model", "rmse", "mae", "wape"}
    assert len(metrics) == len(forecasts)
    assert np.isfinite(metrics[["rmse", "mae", "wape"]].to_numpy()).all()


def test_lags_and_rolling_windows_use_past_values_only() -> None:
    data = generate_sample_data(periods=8, stores=1, items=1)
    features = add_time_series_features(
        data, group_cols=["store", "item"], lags=(1, 3), windows=(3,)
    )
    changed_future = data.copy()
    changed_future.loc[7, "sales"] = 100_000
    changed_features = add_time_series_features(
        changed_future, group_cols=["store", "item"], lags=(1, 3), windows=(3,)
    )

    assert np.isnan(features.loc[0, "sales_lag_1"])
    assert np.isnan(features.loc[0, "sales_rolling_mean_3"])
    assert features.loc[1, "sales_lag_1"] == data.loc[0, "sales"]
    assert features.loc[1, "sales_rolling_mean_3"] == data.loc[0, "sales"]
    assert features.loc[3, "sales_rolling_mean_3"] == np.mean(
        data.loc[0:2, "sales"]
    )
    assert features.loc[4:6, ["sales_lag_1", "sales_lag_3", "sales_rolling_mean_3"]].equals(
        changed_features.loc[
            4:6, ["sales_lag_1", "sales_lag_3", "sales_rolling_mean_3"]
        ]
    )
