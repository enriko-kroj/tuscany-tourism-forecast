# Tuscany Tourist Flow Forecasting

Machine learning pipeline to forecast monthly tourist flows across five
Tuscan cities, developed as Bachelor Semester Project 4 (BSP4) at the
University of Luxembourg under the supervision of Prof. Giacomo Di Tollo.

## Interactive Visualisation

[Rolling Window Walk-Forward Validation — click to open interactive chart](https://htmlpreview.github.io/?https://github.com/enriko-kroj/tuscany-tourism-forecast/blob/main/rolling_window_interactive.html)

Drag the window across the full time series to explore how SARIMA and
LSTM performance changes across different training and test periods.

---

## Overview

Tourist flows across Arezzo, Florence (Firenze), Lucca, Pisa, and Siena
are predicted using real-world Mastercard payment transaction data sourced
from the Smart Region Toscana platform. Four phases are implemented:
data preparation, SARIMA modelling, LSTM modelling, and rolling window
walk-forward validation.

---

## Dataset

- 335 CSV files — one per city per month (5 cities × 67 months)
- 329,857 rows of Mastercard payment transaction records
- Source: Smart Region Toscana platform (proprietary — data not published)
- Period: January 2019 — July 2024
- Coverage: 30+ countries of tourist origin, multiple spending categories
  (Hotels, Eating Places, Apparel, Shopping, Transport, and more)
- Includes the full COVID-19 disruption period (March–December 2020)

---

## Models

- **SARIMA** — Seasonal ARIMA with m=12, selected via auto_arima (pmdarima).
  Orders identified per city using AIC minimisation.
- **LSTM** — Two-layer recurrent neural network (64 + 32 units) with dropout
  regularisation and early stopping, implemented in TensorFlow/Keras.

---

## Results

All models evaluated on a 12-month hold-out test set (Aug 2023 – Jul 2024):

| City    | SARIMA MAPE | LSTM MAPE | Winner (MAPE) |
|---------|-------------|-----------|---------------|
| Arezzo  | 38.6%       | 15.0%     | LSTM          |
| Firenze | 9.3%        | 42.5%     | SARIMA        |
| Lucca   | 112.3%      | 95.6%     | LSTM          |
| Pisa    | 29.4%       | 45.9%     | SARIMA        |
| Siena   | 166.1%      | 82.3%     | LSTM          |

SARIMA outperforms LSTM on absolute errors (RMSE, MAE) for 4 of 5 cities.
LSTM achieves lower percentage error (MAPE) for 3 of 5 cities, particularly
those with explosive non-linear growth in 2024.

**Key finding:** model complexity must be matched to dataset size. With only
43 effective LSTM training sequences and a COVID-19 structural break,
SARIMA's parsimony proved advantageous at this sample size.

### Rolling Window Validation (Phase 4)

Walk-forward validation across 20 windows (train=36m, test=12m, slide=1m):

| City    | SARIMA MAPE (mean±std) | LSTM MAPE (mean±std) | More Consistent |
|---------|------------------------|----------------------|-----------------|
| Arezzo  | 86.6% ± 86.4%          | 87.3% ± 29.9%        | LSTM            |
| Firenze | 43.1% ± 31.5%          | 82.4% ± 82.1%        | SARIMA          |
| Lucca   | 128.5% ± 125.3%        | 89.7% ± 42.3%        | LSTM            |
| Pisa    | 73.9% ± 79.0%          | 95.4% ± 87.0%        | SARIMA          |
| Siena   | 115.8% ± 74.3%         | 92.8% ± 44.0%        | LSTM            |

LSTM shows lower variance (more consistent) across all 5 cities.
Rolling window reverses the MAPE winner for Lucca and Siena vs single-split.

---

## Scripts

| Script | Description |
|--------|-------------|
| `compute_stats.py` | Descriptive statistics for all 335 raw files at three aggregation levels |
| `phase1_data_prep.py` | Data loading, filtering, aggregation, gap checking, COVID flagging, ADF stationarity tests, visualisation |
| `phase2_sarima.py` | SARIMA model fitting via auto_arima, 12-month forecasting, train/test evaluation |
| `phase3_lstm.py` | Two-layer LSTM training and evaluation per city, comparison against SARIMA |
| `phase4_rolling_window.py` | Walk-forward rolling window validation across 20 windows for both models |
| `rolling_window_interactive.html` | Self-contained interactive visualisation of rolling window results |

---

## Tech Stack

Python, pandas, NumPy, scikit-learn, statsmodels, pmdarima,
TensorFlow/Keras, Matplotlib

---

## Academic Context

Bachelor Semester Project 4 — University of Luxembourg, 2025–2026  
Supervisor: Prof. Giacomo Di Tollo  
Research collaborator: Alessio Giorgetti, Università Politecnica delle Marche
