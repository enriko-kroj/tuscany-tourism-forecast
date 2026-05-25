"""
BSP4 Phase 2: SARIMA Modelling
- auto_arima with seasonal=True, m=12 per city
- Hold-out test set: last 12 months (Aug 2023 – Jul 2024)
- Metrics: RMSE, MAE, MAPE
- Outputs: sarima_results.csv, sarima_forecasts.png, sarima_orders.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pmdarima as pm
from pmdarima.arima import auto_arima

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_CSV  = os.path.expanduser("~/Desktop/BSP4_Statistics/master_timeseries.csv")
OUTPUT_DIR = os.path.expanduser("~/Desktop/BSP4_Statistics")
CITIES     = ["Arezzo", "Firenze", "Lucca", "Pisa", "Siena"]
TEST_N     = 12   # hold-out months

CITY_COLORS = {
    "Arezzo":  "#1f77b4",
    "Firenze": "#ff7f0e",
    "Lucca":   "#2ca02c",
    "Pisa":    "#d62728",
    "Siena":   "#9467bd",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
df = df.sort_values(["city", "date"]).reset_index(drop=True)

# ─────────────────────────────────────────────
# FIT SARIMA PER CITY
# ─────────────────────────────────────────────
results_rows   = []
orders_lines   = []
city_artifacts = {}   # store for plotting

orders_lines.append("SARIMA Orders Selected by auto_arima (seasonal=True, m=12)")
orders_lines.append("=" * 60)
orders_lines.append(f"Hold-out test set: last {TEST_N} months per city")
orders_lines.append("")

print("=" * 60)
print("PHASE 2: SARIMA — auto_arima (seasonal=True, m=12)")
print("=" * 60)

for city in CITIES:
    print(f"\n{'─'*50}")
    print(f"  {city}")
    print(f"{'─'*50}")

    city_df = df[df["city"] == city].reset_index(drop=True)
    series  = city_df["tourist_volume"].values
    dates   = city_df["date"].values

    n       = len(series)
    train   = series[: n - TEST_N]
    test    = series[n - TEST_N :]
    train_dates = dates[: n - TEST_N]
    test_dates  = dates[n - TEST_N :]

    print(f"  Train: {len(train)} months ({pd.Timestamp(train_dates[0]).strftime('%Y-%m')} – {pd.Timestamp(train_dates[-1]).strftime('%Y-%m')})")
    print(f"  Test : {len(test)} months  ({pd.Timestamp(test_dates[0]).strftime('%Y-%m')} – {pd.Timestamp(test_dates[-1]).strftime('%Y-%m')})")

    # ── auto_arima ───────────────────────────
    print(f"  Running auto_arima …")
    model = auto_arima(
        train,
        seasonal=True,
        m=12,
        d=None,          # auto-select via unit-root test
        D=None,          # auto-select seasonal d
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        start_P=0, max_P=2,
        start_Q=0, max_Q=2,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )

    order         = model.order           # (p, d, q)
    seasonal_order = model.seasonal_order  # (P, D, Q, m)
    p, d, q       = order
    P, D, Q, m    = seasonal_order
    aic = model.aic()

    print(f"  Order     : ARIMA({p},{d},{q})")
    print(f"  Seasonal  : ({P},{D},{Q})[{m}]")
    print(f"  AIC       : {aic:.2f}")

    # ── In-sample fitted values ─────────────
    fitted = model.fittedvalues()
    # Fitted may be shorter than train (due to differencing), pad with NaN
    if len(fitted) < len(train):
        pad = np.full(len(train) - len(fitted), np.nan)
        fitted_full = np.concatenate([pad, fitted])
    else:
        fitted_full = fitted

    # ── Forecast (test set) ──────────────────
    forecast, conf_int = model.predict(n_periods=TEST_N, return_conf_int=True, alpha=0.05)
    forecast = np.maximum(forecast, 0)   # clip negative forecasts

    # ── Metrics — test set ───────────────────
    r = rmse(test, forecast)
    m_ = mae(test, forecast)
    mp = mape(test, forecast)

    print(f"  Test  RMSE : {r:.2f}")
    print(f"  Test  MAE  : {m_:.2f}")
    print(f"  Test  MAPE : {mp:.2f}%")

    # ── Metrics — training set ───────────────
    mask = ~np.isnan(fitted_full)
    tr_r  = rmse(train[mask], fitted_full[mask])
    tr_m  = mae(train[mask], fitted_full[mask])
    tr_mp = mape(train[mask], fitted_full[mask])

    print(f"  Train RMSE : {tr_r:.2f}")
    print(f"  Train MAE  : {tr_m:.2f}")
    print(f"  Train MAPE : {tr_mp:.2f}%")

    # ── Store ────────────────────────────────
    results_rows.append({
        "city": city,
        "p": p, "d": d, "q": q,
        "P": P, "D": D, "Q": Q, "m": m,
        "AIC": round(aic, 2),
        "Train_RMSE": round(tr_r, 2),
        "Train_MAE":  round(tr_m, 2),
        "Train_MAPE": round(tr_mp, 2),
        "Test_RMSE":  round(r, 2),
        "Test_MAE":   round(m_, 2),
        "Test_MAPE":  round(mp, 2),
    })

    orders_lines.append(f"{city}")
    orders_lines.append(f"  ARIMA order     : ({p},{d},{q})")
    orders_lines.append(f"  Seasonal order  : ({P},{D},{Q})[{m}]")
    orders_lines.append(f"  AIC             : {aic:.2f}")
    orders_lines.append(f"  RMSE={r:.2f}  MAE={m_:.2f}  MAPE={mp:.2f}%")
    orders_lines.append("")

    city_artifacts[city] = {
        "train": train,
        "test": test,
        "fitted": fitted_full,
        "forecast": forecast,
        "conf_int": conf_int,
        "train_dates": train_dates,
        "test_dates": test_dates,
    }

# ─────────────────────────────────────────────
# SAVE sarima_results.csv
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results_rows)
results_path = os.path.join(OUTPUT_DIR, "sarima_results.csv")
results_df[["city","p","d","q","P","D","Q","m","AIC","Test_RMSE","Test_MAE","Test_MAPE"]].rename(
    columns={"Test_RMSE":"RMSE","Test_MAE":"MAE","Test_MAPE":"MAPE"}
).to_csv(results_path, index=False)
print(f"\nSaved → {results_path}")

full_path = os.path.join(OUTPUT_DIR, "sarima_results_full.csv")
results_df[["city","p","d","q","P","D","Q","m","AIC",
            "Train_RMSE","Train_MAE","Train_MAPE",
            "Test_RMSE","Test_MAE","Test_MAPE"]].to_csv(full_path, index=False)
print(f"Saved → {full_path}")
print(results_df[["city","Train_RMSE","Train_MAE","Train_MAPE",
                   "Test_RMSE","Test_MAE","Test_MAPE"]].to_string(index=False))

# ─────────────────────────────────────────────
# SAVE sarima_orders.txt
# ─────────────────────────────────────────────
orders_path = os.path.join(OUTPUT_DIR, "sarima_orders.txt")
with open(orders_path, "w") as f:
    f.write("\n".join(orders_lines) + "\n")
print(f"\nSaved → {orders_path}")

# ─────────────────────────────────────────────
# PLOT sarima_forecasts.png
# ─────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=False)
fig.suptitle(
    "SARIMA Forecasts vs Actual — Monthly Tourist Volume\n(Hold-out test: last 12 months)",
    fontsize=14, fontweight="bold", y=0.98
)

for ax, city in zip(axes, CITIES):
    art = city_artifacts[city]
    color = CITY_COLORS[city]

    train_dates = pd.to_datetime(art["train_dates"])
    test_dates  = pd.to_datetime(art["test_dates"])
    fitted      = art["fitted"]
    forecast    = art["forecast"]
    conf_int    = art["conf_int"]

    # Train actual
    ax.plot(train_dates, art["train"],
            color=color, linewidth=1.5, label="Actual (train)", alpha=0.8)

    # In-sample fitted
    ax.plot(train_dates, fitted,
            color="grey", linewidth=1.0, linestyle="--", label="SARIMA fitted", alpha=0.7)

    # Test actual
    ax.plot(test_dates, art["test"],
            color=color, linewidth=2.2, linestyle="-", marker="o",
            markersize=5, label="Actual (test)")

    # Forecast
    ax.plot(test_dates, forecast,
            color="black", linewidth=2.0, linestyle="--", marker="s",
            markersize=4, label="SARIMA forecast")

    # 95% confidence interval
    ax.fill_between(test_dates, conf_int[:, 0], conf_int[:, 1],
                    color="black", alpha=0.10, label="95% CI")

    # Shade COVID
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-01"),
               color="red", alpha=0.07, label="COVID period")

    # Vertical line at train/test split
    ax.axvline(x=test_dates[0], color="navy", linewidth=1.2,
               linestyle=":", alpha=0.8)

    # Metrics annotation
    row = results_df[results_df["city"] == city].iloc[0]
    order_str = f"SARIMA({row.p},{row.d},{row.q})({row.P},{row.D},{row.Q})[12]"
    metrics_str = f"Test RMSE={row.Test_RMSE:.1f}  MAE={row.Test_MAE:.1f}  MAPE={row.Test_MAPE:.1f}%"
    ax.set_title(f"{city} — {order_str}   |   {metrics_str}",
                 fontsize=10, fontweight="bold", loc="left")

    ax.set_ylabel("Tourist Volume", fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=7, ncol=3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plot_path = os.path.join(OUTPUT_DIR, "sarima_forecasts.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {plot_path}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2 COMPLETE — Output files:")
for fname in ["sarima_results.csv", "sarima_results_full.csv", "sarima_orders.txt", "sarima_forecasts.png"]:
    fpath = os.path.join(OUTPUT_DIR, fname)
    print(f"  {fname:35s}  {os.path.getsize(fpath):>8,} bytes")
print("=" * 60)
