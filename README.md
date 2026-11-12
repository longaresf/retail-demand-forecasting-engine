# Retail Demand Forecasting Engine 📈

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](#)
[![Tests](https://github.com/longaresf/retail-demand-forecasting-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/longaresf/retail-demand-forecasting-engine/actions/workflows/tests.yml)

A robust, production-ready time-series forecasting and inventory replenishment engine designed for the retail sector. This repository demonstrates an end-to-end workflow featuring automated data preprocessing, causal feature engineering, gradient boosted trees, and strict chronological backtesting to eliminate data leakage.

## 📋 Project Overview

Building reliable forecasting systems for retail requires more than just fitting a machine learning model; it demands strict chronological validation, feature safety, and systematic performance diagnostics. This project implements an enterprise-grade pipeline that automates time-series feature generation (lags and rolling windows), handles intermittent demand, benchmarks baseline models against **XGBoost**, and executes rigorous **walk-forward validation**.

---

## ✨ Core Pipeline Architecture

```text
CSV / Raw Data Source
       |
       v
preprocessing -> features -> backtesting -> baselines / XGBoost -> evaluate
       |              |            |                             |
 quality,      calendar,     expanding-window               RMSE, MAE,
 dates &       lags &        temporal folds                    WAPE
 missing       windows

    Robust Preprocessing (src/preprocessing.py): Conversion and aggregation of time-series records, missing value reporting, and autocorrelation scoring across candidate periods.

    Causal Feature Engineering (src/features.py): Automatic generation of calendar attributes, lagged features, rolling means, and exponentially weighted moving averages. Target variables are strictly shifted to prevent future leakage.

    Model Exploration & Benchmarking (src/models.py): Implementation of statistical baselines (Moving Average, Historical Mean, Seasonal Naive) alongside an optimized XGBoost regressor wrapper.

    Chronological Backtesting (src/backtesting.py): Walk-forward validation using expanding windows, preserving strict temporal order and ensuring training data never contains future observations.

    Multi-Metric Evaluation (src/evaluate.py): Evaluation using RMSE, MAE, and WAPE (Weighted Absolute Percentage Error), paired with automated model comparison tables.

    Interactive Dashboard (app.py): Streamlit web application enabling interactive exploration of time-series data, visualization of historical sales, and comparison against XGBoost predictions in test folds.

    Automated Testing (tests/test_backtesting_and_models.py): Unit tests validating chronological split integrity, finite metrics, and causal feature boundaries using pytest.

🛠️ Tech Stack

    Language: Python 3.10+

    Data Core: Pandas, NumPy

    Machine Learning: Scikit-Learn, XGBoost, Statsmodels

    Interactive UI & Visualization: Streamlit, Plotly, Matplotlib, Seaborn

    Testing & CI: Pytest, GitHub Actions

📁 Repository Structure
Plaintext

├── data/                       # Kaggle dataset instructions and sample data generator
├── notebooks/                  # Exploratory Data Analysis (EDA) and visualization notebooks
├── src/                        # Core modular Python package (preprocessing, features, models, backtesting, evaluate)
├── tests/                      # Unit tests for backtesting integrity and model pipelines
├── app.py                      # Interactive Streamlit dashboard
├── requirements.txt            # Project Python dependencies
└── README.md                   # Project documentation

🚀 Getting Started
1. Clone the Repository
Bash

git clone [https://github.com/longaresf/retail-demand-forecasting-engine.git](https://github.com/longaresf/retail-demand-forecasting-engine.git)
cd retail-demand-forecasting-engine

2. Configure Virtual Environment & Install Dependencies
Bash

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

3. Run Unit Tests
Bash

python -m pytest -q

📊 Running the Interactive Web App

Launch the local Streamlit dashboard to explore time-series forecasts interactively:
Bash

streamlit run app.py

The application runs on built-in reproducible synthetic data by default or allows you to upload custom CSV files (date, store, item, sales).
💡 Why Temporal Validation Matters Without Data Leakage

In retail and time-series forecasting, random train-test splits introduce future data into training folds, allowing models to memorize trends, promotions, or calendar anomalies that would be unavailable at the actual time of replenishment. This leads to artificially optimistic performance metrics.

Our walk-forward validation framework strictly sorts observations chronologically, trains exclusively on past periods, and shifts lag features so they only rely on previously observed values.
✒️ Author

Francisco Longares — Data Scientist & AI Solutions Developer — longaresf