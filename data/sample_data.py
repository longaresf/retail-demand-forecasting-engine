"""Deterministic synthetic retail data for local examples and tests."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(
    start_date: str = "2022-01-01",
    periods: int = 730,
    stores: int = 3,
    items: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate daily store-item demand with trend and weekly/yearly seasonality.

    Args:
        start_date: First date in the sample.
        periods: Number of daily observations per store-item pair.
        stores: Number of stores.
        items: Number of products.
        seed: Random seed for reproducibility.

    Returns:
        Long-form observations with date, store, item, and sales columns.
    """
    if min(periods, stores, items) <= 0:
        raise ValueError("periods, stores, and items must all be positive.")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    rows = []
    for store in range(1, stores + 1):
        for item in range(1, items + 1):
            baseline = 8 + store * 2 + item * 1.5
            for index, date in enumerate(dates):
                weekly = 2.5 * np.sin(2 * np.pi * date.dayofweek / 7)
                yearly = 2 * np.sin(2 * np.pi * index / 365.25)
                trend = 0.003 * index
                demand = max(0, baseline + weekly + yearly + trend)
                sales = int(rng.poisson(demand))
                rows.append(
                    {"date": date, "store": store, "item": item, "sales": sales}
                )
    return pd.DataFrame(rows)


def write_sample_data(path: str | Path, **kwargs: int | str) -> Path:
    """Generate sample observations and write them to a CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_data(**kwargs).to_csv(output_path, index=False)
    return output_path
