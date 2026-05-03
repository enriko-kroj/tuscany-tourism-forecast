# Tuscany Tourist Flow Forecasting

Machine learning pipeline to forecast monthly tourist flows across five Tuscan
cities, developed as Bachelor Seminar Project 4 (BSP4) at the University of
Luxembourg under the supervision of Prof. Giacomo Di Tollo.

## Overview

Tourist flows across Arezzo, Florence (Firenze), Lucca, Pisa, and Siena are
predicted using real-world Mastercard payment transaction data sourced from the
Smart Region Toscana platform. Two forecasting approaches are implemented and
compared: SARIMA as a classical statistical baseline and LSTM as a deep learning
approach.

## Dataset

- 335 CSV files — one per city per month (5 cities × 67 months)
- ~329,857 rows of Mastercard payment transaction records
- Source: Smart Region Toscana platform (proprietary — data not published)
- Period: January 2019 — July 2024
- Coverage: 30+ countries of tourist origin, multiple spending categories
  (Hotels, Eating Places, Apparel, Shopping, Transport, and more)
- Includes the COVID-19 disruption period (March–December 2020)

## Models

- **SARIMA** — Seasonal ARIMA with m=12, selected via auto_arima (pmdarima).
  Orders identified per city using AIC minimisation.
- **LSTM** — Two-layer recurrent neural network (64 + 32 units) with dropout
  regularisation and early stopping, implemented in TensorFlow/Keras.

## Key Results

Both models evaluated on a 12-month hold-out test set (Aug 2023 – Jul 2024):

| City    | SARIMA MAPE | LSTM MAPE | Winner (MAPE) |
|---------|-------------|-----------|---------------|
| Arezzo  | 38.6%       | 15.0%     | LSTM ✓        |
| Firenze | 9.3%        | 42.5%     | SARIMA ✓      |
| Lucca   | 112.3%      | 95.6%     | LSTM ✓        |
| Pisa    | 29.4%       | 45.9%     | SARIMA ✓      |
| Siena   | 166.1%      | 82.3%     | LSTM ✓        |

SARIMA outperforms LSTM on absolute errors (RMSE, MAE) for 4 of 5 cities.
LSTM achieves lower percentage error (MAPE) for 3 of 5 cities, particularly
those with explosive non-linear growth in 2024.

**Key finding:** model complexity must be matched to dataset size. With only
43 effective LSTM training sequences and a COVID-19 structural break, SARIMA's
parsimony proved advantageous over deep learning at this sample size.

## Scripts

| Script | Description |
|--------|-------------|
| `compute_stats.py` | Descriptive statistics for all 335 files |
| `phase1_data_prep.py` | Data loading, filtering, aggregation, ADF tests, visualisation |
| `phase2_sarima.py` | SARIMA model fitting and evaluation per city |
| `phase3_lstm.py` | LSTM model training and evaluation per city |

## Tech Stack

Python, pandas, NumPy, scikit-learn, statsmodels, pmdarima,
TensorFlow/Keras, Matplotlib

## Academic Context

Bachelor Seminar Project 4 — University of Luxembourg, 2025–2026
Supervisor: Prof. Giacomo Di Tollo
