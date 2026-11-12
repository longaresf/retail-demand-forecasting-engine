import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    compare_forecasts,
    mean_absolute_error,
    root_mean_squared_error,
    wape,
)
from src.models import (
    XGBoostForecaster,
    historical_mean_forecast,
    moving_average_forecast,
    seasonal_naive_forecast,
)


def test_forecast_metrics() -> None:
    actual = np.array([10, 20, 30])
    predicted = np.array([12, 18, 33])

    assert mean_absolute_error(actual, predicted) == pytest.approx(7 / 3)
    assert root_mean_squared_error(actual, predicted) == pytest.approx(np.sqrt(17 / 3))
    assert wape(actual, predicted) == pytest.approx(7 / 60 * 100)


def test_wape_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="all actual values are zero"):
        wape([0, 0], [1, 2])


def test_baseline_forecasts() -> None:
    history = [1, 2, 3, 4]

    assert moving_average_forecast(history, horizon=2, window=2).tolist() == [3.5, 3.5]
    assert historical_mean_forecast(history, horizon=1).tolist() == [2.5]
    assert seasonal_naive_forecast(history, horizon=6, season_length=2).tolist() == [
        3.0,
        4.0,
        3.0,
        4.0,
        3.0,
        4.0,
    ]


def test_xgboost_forecaster_fits_and_predicts() -> None:
    training = pd.DataFrame({"lag": [1, 2, 3, 4], "sales": [2, 4, 6, 8]})
    model = XGBoostForecaster(n_estimators=5, max_depth=1).fit(training, ["lag"])

    prediction = model.predict(pd.DataFrame({"lag": [5]}))

    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_compare_forecasts_orders_best_wape_first() -> None:
    result = compare_forecasts(
        [10, 20], {"less_accurate": [0, 0], "better": [10, 19]}
    )

    assert result.iloc[0]["model"] == "better"
