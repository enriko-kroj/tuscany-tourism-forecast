# Tuscany Tourist Flow Forecasting

Machine learning pipeline to forecast monthly tourist flows across five Tuscan cities, developed as Bachelor Semester Project 4 (BSP4) at the University of Luxembourg under the supervision of Prof. Giacomo Di Tollo.

## Interactive Visualisation

**[Rolling Window Walk-Forward Validation — click to open interactive chart](https://htmlpreview.github.io/?https://github.com/enriko-kroj/tuscany-tourism-forecast/blob/main/rolling_window_interactive.html)**

Drag the window across the full time series to explore how SARIMA and LSTM performance changes across different training and test periods.

## Overview

Tourist flows across Arezzo, Florence (Firenze), Lucca, Pisa, and Siena are predicted using real-world Mastercard payment transaction data sourced from the Smart Region Toscana platform. Four phases are implemented: data preparation, SARIMA modelling, LSTM modelling, and rolling window walk-forward validation.

An initial LSTM trained on the raw series performed poorly, producing negative R² (worse than a naive mean baseline). Diagnosis identified two causes — the model could not forecast beyond its training range, and it was over-parameterised for the small sample. Applying seasonal differencing and reducing the network size produced an LSTM that outperforms SARIMA across all five cities.

## Dataset

- 335 CSV files — one per city per month (5 cities × 67 months)
- 329,857 rows of Mastercard payment transaction records
- Source: Smart Region Toscana platform (proprietary — data not published)
- Period: January 2019 — July 2024
- Coverage: 30+ countries of tourist origin, multiple spending categories (Hotels, Eating Places, Apparel, Shopping, Transport, and more)
- Includes the full COVID-19 disruption period (March–December 2020)

## Models

- **SARIMA** — Seasonal ARIMA with m=12, selected via auto_arima (pmdarima). Orders identified per city using AIC minimisation.
- **LSTM** — Two-layer recurrent neural network (16 + 8 units) with dropout regularisation, early stopping, and a fixed random seed for full reproducibility, implemented in TensorFlow/Keras. Trained on the seasonally differenced series (lag 12) rather than raw levels, so the model predicts year-on-year change and is not capped by the range of values seen during training.

## Results

All models evaluated on a 12-month hold-out test set (Aug 2023 – Jul 2024):

| City | SARIMA R² | LSTM R² | SARIMA MAPE | LSTM MAPE |
|------|-----------|---------|-------------|-----------|
| Arezzo | 0.941 | **0.973** | 38.6% | **26.3%** |
| Firenze | 0.947 | **0.950** | **9.3%** | 10.9% |
| Lucca | 0.764 | **0.955** | 112.3% | **41.6%** |
| Pisa | **0.951** | 0.911 | **29.4%** | 41.8% |
| Siena | 0.258 | **0.763** | 166.1% | **113.4%** |

The seasonally-differenced LSTM wins or ties on R² for four of five cities, with the largest gains on Lucca and Siena — the two cities with explosive, record-breaking growth in 2024 that the raw LSTM could not reach. SARIMA retains a clear advantage only on Pisa.

**Key finding:** preprocessing and model capacity, not model family, determined the outcome. The same LSTM architecture went from negative R² to the best-performing model in the study, purely through seasonal differencing and reducing the network to a size appropriate for the ~36 effective training sequences.

## Rolling Window Validation (Phase 4)

Walk-forward validation across 20 windows (train=36m, test=12m, slide=1m), mean ± std:

| City | SARIMA MAPE | LSTM MAPE | SARIMA R² | LSTM R² |
|------|-------------|-----------|-----------|---------|
| Arezzo | 86.6% ± 86.4% | **52.8% ± 31.8%** | 0.531 ± 0.652 | **0.825 ± 0.143** |
| Firenze | 43.1% ± 31.5% | **24.9% ± 15.8%** | −0.475 ± 1.893 | **0.538 ± 0.523** |
| Lucca | 128.5% ± 125.3% | **73.3% ± 49.4%** | 0.365 ± 0.867 | **0.768 ± 0.192** |
| Pisa | 73.9% ± 79.0% | **43.8% ± 20.3%** | −0.041 ± 1.478 | **0.663 ± 0.287** |
| Siena | 115.8% ± 74.3% | **67.2% ± 27.6%** | 0.442 ± 0.481 | **0.708 ± 0.177** |

The LSTM outperforms SARIMA on both MAPE and R² in every city, with substantially lower variance. SARIMA's mean R² is negative for Firenze and Pisa — worse than a naive baseline across the 20 windows — while the LSTM stays solidly positive everywhere.

## Scripts

| Script | Description |
|--------|-------------|
| `compute_stats.py` | Descriptive statistics for all 335 raw files at three aggregation levels |
| `phase1_data_prep.py` | Data loading, filtering, aggregation, gap checking, COVID flagging, ADF stationarity tests, visualisation |
| `phase2_sarima.py` | SARIMA model fitting via auto_arima, 12-month forecasting, train/test evaluation |
| `phase3_lstm.py` | LSTM training and evaluation per city (seasonal differencing, 16/8-unit architecture), comparison against SARIMA |
| `phase4_rolling_window.py` | Walk-forward rolling window validation across 20 windows for both models |
| `rolling_window_interactive.html` | Self-contained interactive visualisation of rolling window results |

## Reproducibility

All LSTM results are fully reproducible: the random seed is fixed at the Python, NumPy, TensorFlow, and OS-environment level, so repeated runs are bit-for-bit identical.

## Tech Stack

Python, pandas, NumPy, scikit-learn, statsmodels, pmdarima, TensorFlow/Keras, Matplotlib

## Academic Context

Bachelor Semester Project 4 — University of Luxembourg, 2025–2026
Supervisor: Prof. Giacomo Di Tollo
Research collaborator: Alessio Giorgetti, Università Politecnica delle Marche
