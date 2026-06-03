"""
BSP4 Phase 4: Walk-Forward Rolling Window Validation
- Train window: 36 months  |  Test window: 12 months  |  Slide: 1 month
- 67 months total → 20 windows
  First window : train Jan2019–Dec2021, test Jan2022–Dec2022
  Last window  : train Aug2020–Jul2023, test Aug2023–Jul2024
- SARIMA: fixed orders from phase2 (sarima_results_full.csv), re-fit per window
- LSTM : LSTM(64,32)+Dropout(0.2), seq_len=12, same as phase3
- Outputs: rolling_window_results.csv, rolling_window_plot.png
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_CSV  = os.path.expanduser("~/Desktop/BSP4_Statistics/master_timeseries.csv")
SARIMA_CSV = os.path.expanduser("~/Desktop/BSP4_Statistics/sarima_results_full.csv")
OUTPUT_DIR = os.path.expanduser("~/Desktop/BSP4_Statistics")
CITIES     = ["Arezzo", "Firenze", "Lucca", "Pisa", "Siena"]
TRAIN_W    = 36
TEST_W     = 12
SEQ_LEN    = 12
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

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
df = df.sort_values(["city", "date"]).reset_index(drop=True)

# ─────────────────────────────────────────────
# LOAD SARIMA ORDERS (fixed from phase2)
# ─────────────────────────────────────────────
sarima_orders = (
    pd.read_csv(SARIMA_CSV)[["city", "p", "d", "q", "P", "D", "Q"]]
    .set_index("city")
    .astype(int)
)

# ─────────────────────────────────────────────
# WINDOW MATH
# ─────────────────────────────────────────────
n_months  = len(df[df["city"] == CITIES[0]])
n_windows = n_months - TRAIN_W - TEST_W + 1

print("=" * 65)
print("PHASE 4: ROLLING WINDOW WALK-FORWARD VALIDATION")
print("=" * 65)
print(f"  Months per city : {n_months}")
print(f"  Windows         : {n_windows}  (train={TRAIN_W}, test={TEST_W}, slide=1)")
print(f"  Cities × windows: {len(CITIES) * n_windows}  SARIMA fits + {len(CITIES) * n_windows} LSTM fits")
print("=" * 65)

# ─────────────────────────────────────────────
# WALK-FORWARD LOOP
# ─────────────────────────────────────────────
all_rows = []

for city in CITIES:
    print(f"\n{'─'*65}")
    print(f"  {city}")
    print(f"{'─'*65}")

    city_df = df[df["city"] == city].reset_index(drop=True)
    series  = city_df["tourist_volume"].values.astype(float)
    dates   = city_df["date"].values

    o = sarima_orders.loc[city]
    s_order    = (o.p, o.d, o.q)
    s_seasonal = (o.P, o.D, o.Q, 12)

    for w in range(n_windows):
        tr_sl = slice(w,          w + TRAIN_W)
        te_sl = slice(w + TRAIN_W, w + TRAIN_W + TEST_W)

        train = series[tr_sl]
        test  = series[te_sl]
        t0 = pd.Timestamp(dates[tr_sl][0]).strftime("%Y-%m")
        t1 = pd.Timestamp(dates[tr_sl][-1]).strftime("%Y-%m")
        e0 = pd.Timestamp(dates[te_sl][0]).strftime("%Y-%m")
        e1 = pd.Timestamp(dates[te_sl][-1]).strftime("%Y-%m")

        print(f"  W{w+1:02d}: {t0}–{t1} | test {e0}–{e1}", end="  ", flush=True)

        # ── SARIMA (fixed orders) ─────────────────────────────
        try:
            sarima_mdl = pm.ARIMA(
                order=s_order,
                seasonal_order=s_seasonal,
                suppress_warnings=True,
            )
            sarima_mdl.fit(train)
            s_fc   = np.maximum(sarima_mdl.predict(n_periods=TEST_W), 0)
            s_rmse = rmse(test, s_fc)
            s_mae  = mae(test, s_fc)
            s_mape = mape(test, s_fc)
            s_r2   = float(np.clip(r2_score(test, s_fc), -1, 1))
        except Exception as exc:
            print(f"SARIMA ERR: {exc}", end="  ")
            s_rmse = s_mae = s_mape = s_r2 = np.nan

        # ── LSTM ──────────────────────────────────────────────
        scaler   = MinMaxScaler()
        train_sc = scaler.fit_transform(train.reshape(-1, 1)).flatten()

        X_tr, y_tr = make_sequences(train_sc, SEQ_LEN)
        X_tr = X_tr.reshape(X_tr.shape[0], SEQ_LEN, 1)

        tf.random.set_seed(SEED)
        lstm_mdl = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ])
        lstm_mdl.compile(optimizer="adam", loss="mse")
        early_stop = EarlyStopping(
            monitor="val_loss", patience=20,
            restore_best_weights=True, verbose=0,
        )
        lstm_mdl.fit(
            X_tr, y_tr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0,
        )

        # Rolling multi-step forecast
        rolling = list(train_sc[-SEQ_LEN:])
        pred_sc = []
        for _ in range(TEST_W):
            x_in = np.array(rolling[-SEQ_LEN:]).reshape(1, SEQ_LEN, 1)
            p    = lstm_mdl.predict(x_in, verbose=0)[0, 0]
            pred_sc.append(p)
            rolling.append(p)

        l_fc = np.maximum(
            scaler.inverse_transform(np.array(pred_sc).reshape(-1, 1)).flatten(), 0
        )
        l_rmse = rmse(test, l_fc)
        l_mae  = mae(test, l_fc)
        l_mape = mape(test, l_fc)
        l_r2   = float(np.clip(r2_score(test, l_fc), -1, 1))

        tf.keras.backend.clear_session()

        print(f"SARIMA MAPE={s_mape:.1f}% R²={s_r2:.3f}  LSTM MAPE={l_mape:.1f}% R²={l_r2:.3f}")

        all_rows.append({
            "city":          city,
            "window_number": w + 1,
            "train_start":   t0,
            "train_end":     t1,
            "test_start":    e0,
            "test_end":      e1,
            "sarima_rmse":   round(s_rmse, 2),
            "sarima_mae":    round(s_mae,  2),
            "sarima_mape":   round(s_mape, 2),
            "sarima_r2":     round(s_r2,   4),
            "lstm_rmse":     round(l_rmse, 2),
            "lstm_mae":      round(l_mae,  2),
            "lstm_mape":     round(l_mape, 2),
            "lstm_r2":       round(l_r2,   4),
        })

# ─────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────
results_df = pd.DataFrame(all_rows)
csv_path = os.path.join(OUTPUT_DIR, "rolling_window_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nSaved → {csv_path}")

# ─────────────────────────────────────────────
# PLOT 1: rolling_window_plot.png  (MAPE)
# ─────────────────────────────────────────────
windows = np.arange(1, n_windows + 1)
first_city_rows = results_df[results_df["city"] == CITIES[0]].sort_values("window_number")
tick_labels = first_city_rows["test_start"].tolist()

fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
fig.suptitle(
    f"Walk-Forward Rolling Window Validation — MAPE per Window\n"
    f"(Train={TRAIN_W}m, Test={TEST_W}m, slide=1m, {n_windows} windows | 5 Tuscan cities)",
    fontsize=13, fontweight="bold", y=0.99,
)

for ax, city in zip(axes, CITIES):
    city_res = results_df[results_df["city"] == city].sort_values("window_number")
    s_mapes  = city_res["sarima_mape"].values
    l_mapes  = city_res["lstm_mape"].values
    s_mean, l_mean = np.nanmean(s_mapes), np.nanmean(l_mapes)

    ax.plot(windows, s_mapes, color="#4e79a7", linewidth=1.8, marker="o", markersize=4, label="SARIMA")
    ax.plot(windows, l_mapes, color="#f28e2b", linewidth=1.8, marker="s", markersize=4, label="LSTM")
    ax.axhline(s_mean, color="#4e79a7", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.axhline(l_mean, color="#f28e2b", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_title(
        f"{city} — SARIMA mean={s_mean:.1f}%  std={np.nanstd(s_mapes):.1f}%   |   "
        f"LSTM mean={l_mean:.1f}%  std={np.nanstd(l_mapes):.1f}%",
        fontsize=9, fontweight="bold", loc="left",
    )
    ax.set_ylabel("MAPE (%)", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xticks(windows)
axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
axes[-1].set_xlabel("Test window start (YYYY-MM)", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plot_path = os.path.join(OUTPUT_DIR, "rolling_window_plot.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {plot_path}")

# ─────────────────────────────────────────────
# PLOT 2: rolling_window_r2_plot.png  (R²)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
fig.suptitle(
    f"Walk-Forward Rolling Window Validation — R² per Window\n"
    f"(Train={TRAIN_W}m, Test={TEST_W}m, slide=1m, {n_windows} windows | 5 Tuscan cities)",
    fontsize=13, fontweight="bold", y=0.99,
)

for ax, city in zip(axes, CITIES):
    city_res = results_df[results_df["city"] == city].sort_values("window_number")
    s_r2s    = city_res["sarima_r2"].values
    l_r2s    = city_res["lstm_r2"].values
    s_mean, l_mean = np.nanmean(s_r2s), np.nanmean(l_r2s)

    ax.plot(windows, s_r2s, color="#4e79a7", linewidth=1.8, marker="o", markersize=4, label="SARIMA")
    ax.plot(windows, l_r2s, color="#f28e2b", linewidth=1.8, marker="s", markersize=4, label="LSTM")
    ax.axhline(s_mean, color="#4e79a7", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.axhline(l_mean, color="#f28e2b", linestyle="--", linewidth=0.9, alpha=0.6)
    # Reference line at R²=0 (model no better than predicting the mean)
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_title(
        f"{city} — SARIMA mean={s_mean:.3f}  std={np.nanstd(s_r2s):.3f}   |   "
        f"LSTM mean={l_mean:.3f}  std={np.nanstd(l_r2s):.3f}",
        fontsize=9, fontweight="bold", loc="left",
    )
    ax.set_ylabel("R²", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)

axes[-1].set_xticks(windows)
axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
axes[-1].set_xlabel("Test window start (YYYY-MM)", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.97])
r2_plot_path = os.path.join(OUTPUT_DIR, "rolling_window_r2_plot.png")
plt.savefig(r2_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {r2_plot_path}")

# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 78)
print("ROLLING WINDOW SUMMARY — Mean ± Std across all windows per city")
print("=" * 78)

# Block 1: RMSE / MAE / MAPE
header = f"{'City':<10}  {'SARIMA':^33}  {'LSTM':^33}"
sub    = f"{'':10}  {'RMSE':>10}  {'MAE':>10}  {'MAPE':>9}  {'RMSE':>10}  {'MAE':>10}  {'MAPE':>9}"
print(header)
print(sub)
print("-" * 78)
for city in CITIES:
    r = results_df[results_df["city"] == city]
    print(
        f"{city:<10}  "
        f"{r.sarima_rmse.mean():>7.1f}±{r.sarima_rmse.std():>5.1f}  "
        f"{r.sarima_mae.mean():>7.1f}±{r.sarima_mae.std():>4.1f}  "
        f"{r.sarima_mape.mean():>6.1f}±{r.sarima_mape.std():>4.1f}%  "
        f"{r.lstm_rmse.mean():>7.1f}±{r.lstm_rmse.std():>5.1f}  "
        f"{r.lstm_mae.mean():>7.1f}±{r.lstm_mae.std():>4.1f}  "
        f"{r.lstm_mape.mean():>6.1f}±{r.lstm_mape.std():>4.1f}%"
    )

# Block 2: R²
print()
r2_header = f"{'City':<10}  {'SARIMA R²':^20}  {'LSTM R²':^20}"
r2_sub    = f"{'':10}  {'mean':>9}  {'std':>9}  {'mean':>9}  {'std':>9}"
print(r2_header)
print(r2_sub)
print("-" * 52)
for city in CITIES:
    r = results_df[results_df["city"] == city]
    print(
        f"{city:<10}  "
        f"{r.sarima_r2.mean():>9.4f}  {r.sarima_r2.std():>9.4f}  "
        f"{r.lstm_r2.mean():>9.4f}  {r.lstm_r2.std():>9.4f}"
    )

print("=" * 78)
print(f"\nPHASE 4 COMPLETE")
for fname in ["rolling_window_results.csv", "rolling_window_plot.png", "rolling_window_r2_plot.png"]:
    fpath = os.path.join(OUTPUT_DIR, fname)
    print(f"  {fname:38s}  {os.path.getsize(fpath):>8,} bytes")
print("=" * 78)
