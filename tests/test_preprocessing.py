import pandas as pd
import pytest

from src.preprocessing import clean_sales_data, detect_seasonality, missing_value_report


def test_clean_sales_data_parses_dates_and_aggregates_duplicates() -> None:
    source = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-01", "2024-01-02", "not-a-date"],
            "item": [1, 1, 1, 1],
            "sales": [2, 4, 3, 5],
        }
    )

    result = clean_sales_data(source, group_cols=["item"])

    assert result["date"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert result["sales"].tolist() == [4, 5]


def test_clean_sales_data_rejects_empty_valid_set() -> None:
    with pytest.raises(ValueError, match="No valid observations"):
        clean_sales_data(pd.DataFrame({"date": ["bad"], "sales": [None]}))


def test_missing_value_report_counts_nulls() -> None:
    report = missing_value_report(pd.DataFrame({"sales": [1, None], "item": [None, None]}))

    assert report.loc["sales", "missing_count"] == 1
    assert report.loc["item", "missing_pct"] == 100


def test_detect_seasonality_finds_weekly_pattern() -> None:
    values = pd.Series([index % 7 for index in range(100)])

    assert 7 in detect_seasonality(values, periods=[7], threshold=0.8)
