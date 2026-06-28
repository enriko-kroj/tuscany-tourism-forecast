"""
BSP4 Phase 3: LSTM Modelling — v2 (seasonal differencing + smaller network)
- Seasonal differencing lag=12 before scaling (stationarises the series)
- Architecture: LSTM(16) → Dropout(0.2) → LSTM(8) → Dropout(0.2) → Dense(1)
- Sequence length 12, same train/test split as SARIMA (test = last 12 months)
- MinMaxScaler on differenced series, EarlyStopping patience=20
- Outputs: lstm_results_full.csv, lstm_forecasts.png, lstm_vs_sarima.csv, lstm_comparison.png
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import pmdarima as pm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# ── Config ───────────────────────────────────────────────────
INPUT_CSV  = os.path.expanduser("~/Desktop/BSP4_Statistics/master_timeseries.csv")
SARIMA_CSV = os.path.expanduser("~/Desktop/BSP4_Statistics/sarima_results_full.csv")
OUTPUT_DIR = os.path.expanduser("~/Desktop/BSP4_Statistics")
CITIES     = ["Arezzo", "Firenze", "Lucca", "Pisa", "Siena"]
SEQ_LEN    = 12
SEAS_LAG   = 12
TEST_N     = 12
EPOCHS     = 500
BATCH_SIZE = 16
SEED       = 42

CITY_COLORS = {
    "Arezzo":  "#1f77b4",
    "Firenze": "#ff7f0e",
    "Lucca":   "#2ca02c",
    "Pisa":    "#d62728",
    "Siena":   "#9467bd",
}

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Metric helpers ───────────────────────────────────────────
def rmse(a, b):
    return float(np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2)))

def mae(a, b):
    return float(np.mean(np.abs(np.array(a) - np.array(b))))

def mape(a, b):
    a, b = np.array(a), np.array(b)
    mask = a != 0
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100)

def make_sequences(series, seq_len):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X), np.array(y)

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
df = df.sort_values(["city", "date"]).reset_index(drop=True)

# ── Train LSTM per city ──────────────────────────────────────
results_rows   = []
city_artifacts = {}

print("=" * 65)
print("PHASE 3: LSTM — seas_diff(12), LSTM(16,8), dropout=0.2")
print("=" * 65)

for city in CITIES:
    print(f"\n{'─'*55}")
    print(f"  {city}")
    print(f"{'─'*55}")

    city_df = df[df["city"] == city].reset_index(drop=True)
    series  = city_df["tourist_volume"].values.astype(float)
    dates   = city_df["date"].values

    n           = len(series)
    train       = series[: n - TEST_N]
    test        = series[n - TEST_N :]
    train_dates = dates[: n - TEST_N]
    test_dates  = dates[n - TEST_N :]
    n_train     = len(train)

    print(f"  Train: {n_train} months  "
          f"({pd.Timestamp(train_dates[0]).strftime('%Y-%m')} – "
          f"{pd.Timestamp(train_dates[-1]).strftime('%Y-%m')})")
    print(f"  Test : {len(test)}  months  "
          f"({pd.Timestamp(test_dates[0]).strftime('%Y-%m')} – "
          f"{pd.Timestamp(test_dates[-1]).strftime('%Y-%m')})")

    # ── Seasonal differencing (lag=12) on the training series ──
    # diff_train[j] = train[j+12] - train[j], j = 0..n_train-13
    diff_train = train[SEAS_LAG:] - train[:-SEAS_LAG]   # shape (n_train-12,)

    # ── Scale the differenced series ──────────────────────────
    scaler        = MinMaxScaler()
    diff_train_sc = scaler.fit_transform(diff_train.reshape(-1, 1)).flatten()

    # ── Sequences on differenced, scaled series ───────────────
    X_tr, y_tr = make_sequences(diff_train_sc, SEQ_LEN)
    X_tr = X_tr.reshape(X_tr.shape[0], SEQ_LEN, 1)

    # ── Smaller model ─────────────────────────────────────────
    tf.random.set_seed(SEED)
    model = Sequential([
        LSTM(16, return_sequences=True, input_shape=(SEQ_LEN, 1)),
        Dropout(0.2),
        LSTM(8),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="val_loss", patience=20,
        restore_best_weights=True, verbose=0,
    )

    model.fit(
        X_tr, y_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=0,
    )
    stopped_epoch = early_stop.stopped_epoch if early_stop.stopped_epoch > 0 else EPOCHS
    print(f"  Stopped at epoch {stopped_epoch}")

    # ── In-sample train predictions ───────────────────────────
    # model.predict(X_tr)[i] predicts diff_train[i + SEQ_LEN]
    #   = train[i + SEQ_LEN + SEAS_LAG] - train[i + SEQ_LEN]
    # Invert: pred_orig[i] = pred_diff[i] + train[i + SEQ_LEN]
    # Actual:              = train[i + SEQ_LEN + SEAS_LAG]
    train_pred_sc   = model.predict(X_tr, verbose=0).flatten()
    train_diff_pred = scaler.inverse_transform(train_pred_sc.reshape(-1, 1)).flatten()
    n_insample      = len(train_diff_pred)
    train_pred      = train_diff_pred + train[SEQ_LEN : SEQ_LEN + n_insample]
    train_pred      = np.maximum(train_pred, 0)
    train_actual    = train[SEQ_LEN + SEAS_LAG :]  # = train[24:]

    # ── Rolling multi-step test forecast ──────────────────────
    # Seed with last SEQ_LEN values of differenced scaled train
    rolling = list(diff_train_sc[-SEQ_LEN:])
    test_diff_sc = []
    for _ in range(TEST_N):
        x_in = np.array(rolling[-SEQ_LEN:]).reshape(1, SEQ_LEN, 1)
        p    = model.predict(x_in, verbose=0)[0, 0]
        test_diff_sc.append(p)
        rolling.append(p)

    test_diff_pred = scaler.inverse_transform(
        np.array(test_diff_sc).reshape(-1, 1)
    ).flatten()

    # Invert seasonal differencing:
    # test_pred[k] = test_diff_pred[k] + series[t - 12]
    # where t = n_train + k, so series[t-12] = train[n_train - 12 + k] = train[-TEST_N + k]
    # Since SEAS_LAG == TEST_N == 12, this is simply train[-12:]
    test_pred = test_diff_pred + train[n_train - SEAS_LAG :]
    test_pred = np.maximum(test_pred, 0)

    # ── Metrics ───────────────────────────────────────────────
    tr_r  = rmse(train_actual, train_pred)
    tr_m  = mae(train_actual, train_pred)
    tr_mp = mape(train_actual, train_pred)
    tr_r2 = float(r2_score(train_actual, train_pred))
    te_r  = rmse(test, test_pred)
    te_m  = mae(test, test_pred)
    te_mp = mape(test, test_pred)
    te_r2 = float(r2_score(test, test_pred))

    print(f"  Train RMSE={tr_r:.2f}  MAE={tr_m:.2f}  MAPE={tr_mp:.2f}%  R²={tr_r2:.4f}")
    print(f"  Test  RMSE={te_r:.2f}  MAE={te_m:.2f}  MAPE={te_mp:.2f}%  R²={te_r2:.4f}")

    results_rows.append({
        "city":       city,
        "Train_RMSE": round(tr_r,  2),
        "Train_MAE":  round(tr_m,  2),
        "Train_MAPE": round(tr_mp, 2),
        "Train_R2":   round(tr_r2, 4),
        "Test_RMSE":  round(te_r,  2),
        "Test_MAE":   round(te_m,  2),
        "Test_MAPE":  round(te_mp, 2),
        "Test_R2":    round(te_r2, 4),
    })

    city_artifacts[city] = {
        "train":             train,
        "test":              test,
        "train_pred":        train_pred,
        "test_pred":         test_pred,
        "train_dates":       train_dates,
        "test_dates":        test_dates,
        "insample_dates":    train_dates[SEQ_LEN + SEAS_LAG:],
    }

# ── Save lstm_results_full.csv ────────────────────────────────
results_df = pd.DataFrame(results_rows)
lstm_path  = os.path.join(OUTPUT_DIR, "lstm_results_full.csv")
results_df.to_csv(lstm_path, index=False)
print(f"\nSaved → {lstm_path}")

# ── Print test-set table ──────────────────────────────────────
print("\n" + "=" * 65)
print("TEST-SET METRICS PER CITY")
print("=" * 65)
print(f"{'City':<10}  {'RMSE':>10}  {'MAE':>10}  {'MAPE':>8}  {'R²':>8}")
print("-" * 55)
for _, row in results_df.iterrows():
    print(f"{row.city:<10}  {row.Test_RMSE:>10.2f}  {row.Test_MAE:>10.2f}  "
          f"{row.Test_MAPE:>7.2f}%  {row.Test_R2:>8.4f}")
print("=" * 65)

# ── SARIMA R² on the same single train/test split ────────────
sarima_orders = (
    pd.read_csv(SARIMA_CSV)[["city", "p", "d", "q", "P", "D", "Q"]]
    .set_index("city")
    .astype(int)
)

print("\n" + "=" * 65)
print("SARIMA vs LSTM — SINGLE-SPLIT TEST R² PER CITY")
print("=" * 65)
print(f"{'City':<10}  {'SARIMA R²':>12}  {'LSTM R²':>12}  {'Δ (LSTM−SARIMA)':>16}")
print("-" * 55)

for city in CITIES:
    city_df  = df[df["city"] == city].reset_index(drop=True)
    series   = city_df["tourist_volume"].values.astype(float)
    n        = len(series)
    train    = series[: n - TEST_N]
    test     = series[n - TEST_N :]

    o = sarima_orders.loc[city]
    try:
        mdl = pm.ARIMA(
            order=(o.p, o.d, o.q),
            seasonal_order=(o.P, o.D, o.Q, 12),
            suppress_warnings=True,
        )
        mdl.fit(train)
        s_fc = np.maximum(mdl.predict(n_periods=TEST_N), 0)
        s_r2 = float(r2_score(test, s_fc))
    except Exception as exc:
        print(f"  {city}: SARIMA fit failed — {exc}")
        s_r2 = float("nan")

    l_r2  = results_df.loc[results_df["city"] == city, "Test_R2"].iloc[0]
    delta = l_r2 - s_r2
    print(f"{city:<10}  {s_r2:>+12.4f}  {l_r2:>+12.4f}  {delta:>+16.4f}")

print("=" * 65)

# ── Plot lstm_forecasts.png ───────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=False)
fig.suptitle(
    "LSTM Forecasts vs Actual — Monthly Tourist Volume\n"
    "(Hold-out test: last 12 months | seas_diff(12) | LSTM(16,8) + Dropout(0.2))",
    fontsize=13, fontweight="bold", y=0.98,
)

for ax, city in zip(axes, CITIES):
    art   = city_artifacts[city]
    color = CITY_COLORS[city]

    train_dates   = pd.to_datetime(art["train_dates"])
    test_dates    = pd.to_datetime(art["test_dates"])
    insamp_dates  = pd.to_datetime(art["insample_dates"])

    ax.plot(train_dates, art["train"],
            color=color, linewidth=1.5, label="Actual (train)", alpha=0.8)
    ax.plot(insamp_dates, art["train_pred"],
            color="grey", linewidth=1.0, linestyle="--",
            label="LSTM fitted", alpha=0.7)
    ax.plot(test_dates, art["test"],
            color=color, linewidth=2.2, linestyle="-",
            marker="o", markersize=5, label="Actual (test)")
    ax.plot(test_dates, art["test_pred"],
            color="black", linewidth=2.0, linestyle="--",
            marker="s", markersize=4, label="LSTM forecast")
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-01"),
               color="red", alpha=0.07, label="COVID period")
    ax.axvline(x=test_dates[0], color="navy", linewidth=1.2,
               linestyle=":", alpha=0.8)

    row = results_df[results_df["city"] == city].iloc[0]
    metrics_str = (f"RMSE={row.Test_RMSE:.1f}  MAE={row.Test_MAE:.1f}  "
                   f"MAPE={row.Test_MAPE:.1f}%  R²={row.Test_R2:.3f}")
    ax.set_title(f"{city} — LSTM(16,8)  |  {metrics_str}",
                 fontsize=10, fontweight="bold", loc="left")
    ax.set_ylabel("Tourist Volume", fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=7, ncol=4)

plt.tight_layout(rect=[0, 0, 1, 0.97])
forecast_path = os.path.join(OUTPUT_DIR, "lstm_forecasts.png")
plt.savefig(forecast_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {forecast_path}")

# ── Merge with SARIMA → lstm_vs_sarima.csv ───────────────────
sarima_raw = pd.read_csv(SARIMA_CSV)[
    ["city", "Train_RMSE", "Train_MAE", "Train_MAPE",
     "Test_RMSE",  "Test_MAE",  "Test_MAPE"]
]
sarima_raw.columns = [
    "city",
    "SARIMA_Train_RMSE", "SARIMA_Train_MAE", "SARIMA_Train_MAPE",
    "SARIMA_Test_RMSE",  "SARIMA_Test_MAE",  "SARIMA_Test_MAPE",
]
lstm_ren = results_df.rename(columns={
    "Train_RMSE": "LSTM_Train_RMSE", "Train_MAE": "LSTM_Train_MAE",
    "Train_MAPE": "LSTM_Train_MAPE", "Train_R2":  "LSTM_Train_R2",
    "Test_RMSE":  "LSTM_Test_RMSE",  "Test_MAE":  "LSTM_Test_MAE",
    "Test_MAPE":  "LSTM_Test_MAPE",  "Test_R2":   "LSTM_Test_R2",
})
vs_df   = sarima_raw.merge(lstm_ren, on="city")
vs_path = os.path.join(OUTPUT_DIR, "lstm_vs_sarima.csv")
vs_df.to_csv(vs_path, index=False)
print(f"Saved → {vs_path}")

# ── MAPE comparison bar chart — lstm_comparison.png ──────────
x     = np.arange(len(CITIES))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("SARIMA vs LSTM — MAPE Comparison per City",
             fontsize=14, fontweight="bold")

for ax, split, title in [
    (ax1, "Train", "Training Set MAPE (%)"),
    (ax2, "Test",  "Test Set MAPE (%)"),
]:
    sarima_mape = vs_df[f"SARIMA_{split}_MAPE"].values
    lstm_mape   = vs_df[f"LSTM_{split}_MAPE"].values

    bars1 = ax.bar(x - width / 2, sarima_mape, width,
                   label="SARIMA", color="#4e79a7", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, lstm_mape,   width,
                   label="LSTM",   color="#f28e2b", alpha=0.85, edgecolor="white")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CITIES, fontsize=10)
    ax.set_ylabel("MAPE (%)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8, color="#4e79a7")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8, color="#f28e2b")

plt.tight_layout()
comp_path = os.path.join(OUTPUT_DIR, "lstm_comparison.png")
plt.savefig(comp_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {comp_path}")

# ── Final summary ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("PHASE 3 COMPLETE — Output files:")
for fname in ["lstm_results_full.csv", "lstm_forecasts.png",
              "lstm_vs_sarima.csv", "lstm_comparison.png"]:
    fpath = os.path.join(OUTPUT_DIR, fname)
    print(f"  {fname:35s}  {os.path.getsize(fpath):>8,} bytes")
print("=" * 65)
