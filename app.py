"""Interactive Streamlit dashboard for retail demand backtesting."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.sample_data import generate_sample_data
from src.backtesting import walk_forward_split
from src.evaluate import root_mean_squared_error, wape
from src.features import add_time_series_features
from src.models import XGBoostForecaster
from src.preprocessing import clean_sales_data

FEATURE_COLUMNS = [
    "year",
    "month",
    "day_of_week",
    "is_weekend",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_mean_28",
    "sales_ewm_7",
    "sales_ewm_28",
]


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """Create the deterministic local sample dataset."""
    return generate_sample_data()


def main() -> None:
    """Render the demand exploration and temporal holdout dashboard."""
    st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
    st.title("Retail Demand Forecasting Engine")
    st.write(
        "Explora la demanda histórica y compara un modelo XGBoost con las ventas "
        "reales en una partición temporal walk-forward."
    )

    uploaded_file = st.file_uploader(
        "Opcional: carga un CSV con date, store, item y sales",
        type=["csv"],
        help="El archivo train.csv de Store Item Demand Forecasting usa este esquema.",
    )
    if uploaded_file is None:
        source_data = load_sample_data()
        st.caption("Mostrando datos sintéticos reproducibles.")
    else:
        try:
            source_data = clean_sales_data(
                pd.read_csv(uploaded_file), group_cols=["store", "item"]
            )
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"No se pudo cargar el CSV: {error}")
            st.stop()
        st.caption("Mostrando datos del archivo CSV cargado.")

    stores = sorted(source_data["store"].unique().tolist())
    selected_store = st.sidebar.selectbox("Tienda", stores)
    store_data = source_data.loc[source_data["store"] == selected_store]
    items = sorted(store_data["item"].unique().tolist())
    selected_item = st.sidebar.selectbox("Producto", items)
    series = (
        store_data.loc[store_data["item"] == selected_item]
        .sort_values("date")
        .reset_index(drop=True)
    )

    st.subheader(f"Tienda {selected_store} · Producto {selected_item}")
    st.plotly_chart(
        _history_chart(series), use_container_width=True, key="demand-history"
    )

    if len(series) < 21:
        st.warning(
            "Se necesitan al menos 21 observaciones diarias para reservar "
            "7 días de prueba y mantener 14 días de entrenamiento."
        )
        return

    max_test_size = min(90, len(series) - 14)
    test_size = st.sidebar.slider(
        "Días en el período de prueba",
        min_value=7,
        max_value=max_test_size,
        value=min(28, max_test_size),
    )
    featured = add_time_series_features(
        series, lags=(1, 7, 28), windows=(7, 28)
    )
    train, test = next(
        walk_forward_split(featured, n_splits=1, test_size=test_size)
    )
    model = XGBoostForecaster(n_estimators=150, max_depth=5).fit(
        train, FEATURE_COLUMNS
    )
    predictions = model.predict(test)
    actual = test["sales"].to_numpy()
    rmse = root_mean_squared_error(actual, predictions)

    metric_columns = st.columns(2)
    metric_columns[0].metric("RMSE", f"{rmse:.2f}")
    try:
        score_wape = wape(actual, predictions)
    except ValueError:
        metric_columns[1].metric("WAPE", "N/A")
        st.warning("WAPE no está definido porque la demanda real del período es cero.")
    else:
        metric_columns[1].metric("WAPE", f"{score_wape:.2f}%")

    st.subheader("Backtesting: predicción frente a demanda real")
    st.caption(
        "El corte es cronológico. Las variables de lags se calculan solo con "
        "observaciones anteriores; en cada fecha de prueba se asume que ya se "
        "conoce la venta del día previo (evaluación walk-forward de un paso)."
    )
    st.plotly_chart(
        _backtest_chart(series, train, test, predictions),
        use_container_width=True,
        key="backtest",
    )


def _history_chart(series: pd.DataFrame) -> go.Figure:
    """Create an interactive line chart of historical demand."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["sales"],
            mode="lines",
            name="Demanda real",
        )
    )
    figure.update_layout(
        title="Demanda histórica",
        xaxis_title="Fecha",
        yaxis_title="Unidades vendidas",
        hovermode="x unified",
    )
    return figure


def _backtest_chart(
    series: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: np.ndarray,
) -> go.Figure:
    """Plot observed demand and predictions for the chronological test block."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["sales"],
            mode="lines",
            name="Demanda real",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=test["date"],
            y=predictions,
            mode="lines+markers",
            name="XGBoost (prueba)",
        )
    )
    cutoff = train["date"].max()
    figure.add_shape(
        type="line",
        x0=cutoff,
        x1=cutoff,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"dash": "dash"},
    )
    figure.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Unidades vendidas",
        hovermode="x unified",
    )
    return figure


if __name__ == "__main__":
    main()
