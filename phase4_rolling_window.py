import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

"""
BSP4 Phase 4: Walk-Forward Rolling Window Validation — v2
- Train window: 36 months  |  Test window: 12 months  |  Slide: 1 month
- 67 months total → 20 windows
  First window : train Jan2019–Dec2021, test Jan2022–Dec2022
  Last window  : train Aug2020–Jul2023, test Aug2023–Jul2024
- SARIMA: fixed orders from phase2 (sarima_results_full.csv), re-fit per window
- LSTM : seas_diff(12) + LSTM(16,8)+Dropout(0.2), seq_len=12
- R²   : computed PER WINDOW, reported as mean ± std across 20 windows
- Outputs: rolling_window_results.csv, rolling_window_plot.png,
           rolling_window_r2_plot.png
"""

import warnings
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

import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

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
SEAS_LAG   = 12
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
print(f"  LSTM            : seas_diff(12), LSTM(16,8), dropout=0.2")
print(f"  R²              : per-window, reported as mean ± std")
print("=" * 65)

# ─────────────────────────────────────────────
# WALK-FORWARD LOOP
# ─────────────────────────────────────────────
all_rows = []

# Per-city per-window R² accumulators
city_sarima_r2 = {c: [] for c in CITIES}
city_lstm_r2   = {c: [] for c in CITIES}

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
        tr_sl = slice(w,           w + TRAIN_W)
        te_sl = slice(w + TRAIN_W, w + TRAIN_W + TEST_W)

        train = series[tr_sl]
        test  = series[te_sl]
        t0 = pd.Timestamp(dates[tr_sl][0]).strftime("%Y-%m")
        t1 = pd.Timestamp(dates[tr_sl][-1]).strftime("%Y-%m")
        e0 = pd.Timestamp(dates[te_sl][0]).strftime("%Y-%m")
        e1 = pd.Timestamp(dates[te_sl][-1]).strftime("%Y-%m")

        print(f"  W{w+1:02d}: {t0}–{t1} | test {e0}–{e1}", end="  ", flush=True)

        # ── SARIMA (fixed orders) ─────────────────────────────
        s_r2 = np.nan
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
            s_r2   = float(r2_score(test, s_fc))
            city_sarima_r2[city].append(s_r2)
        except Exception as exc:
            print(f"SARIMA ERR: {exc}", end="  ")
            s_rmse = s_mae = s_mape = np.nan

        # ── LSTM with seasonal differencing ───────────────────
        # diff_train[j] = train[j+12] - train[j], j = 0..TRAIN_W-13
        diff_train = train[SEAS_LAG:] - train[:-SEAS_LAG]   # length TRAIN_W - 12 = 24

        scaler        = MinMaxScaler()
        diff_train_sc = scaler.fit_transform(diff_train.reshape(-1, 1)).flatten()

        X_tr, y_tr = make_sequences(diff_train_sc, SEQ_LEN)
        X_tr = X_tr.reshape(X_tr.shape[0], SEQ_LEN, 1)

        tf.random.set_seed(SEED)
        lstm_mdl = Sequential([
            LSTM(16, return_sequences=True, input_shape=(SEQ_LEN, 1)),
            Dropout(0.2),
            LSTM(8),
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

        # Rolling multi-step forecast in differenced space
        rolling = list(diff_train_sc[-SEQ_LEN:])
        pred_diff_sc = []
        for _ in range(TEST_W):
            x_in = np.array(rolling[-SEQ_LEN:]).reshape(1, SEQ_LEN, 1)
            p    = lstm_mdl.predict(x_in, verbose=0)[0, 0]
            pred_diff_sc.append(p)
            rolling.append(p)

        pred_diff = scaler.inverse_transform(
            np.array(pred_diff_sc).reshape(-1, 1)
        ).flatten()

        # Invert seasonal differencing:
        # test_pred[k] = pred_diff[k] + series[t-12]
        # where t = w + TRAIN_W + k, so series[t-12] = train[TRAIN_W - SEAS_LAG + k]
        # = train[-TEST_W + k]  (since SEAS_LAG == TEST_W == 12)
        l_fc = pred_diff + train[TRAIN_W - SEAS_LAG :]
        l_fc = np.maximum(l_fc, 0)

        l_rmse = rmse(test, l_fc)
        l_mae  = mae(test, l_fc)
        l_mape = mape(test, l_fc)
        l_r2   = float(r2_score(test, l_fc))
        city_lstm_r2[city].append(l_r2)

        tf.keras.backend.clear_session()

        print(f"SARIMA MAPE={s_mape:.1f}% R²={s_r2:.3f}  |  "
              f"LSTM MAPE={l_mape:.1f}% R²={l_r2:.3f}")

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
            "sarima_r2":     round(s_r2,   4) if not np.isnan(s_r2) else np.nan,
            "lstm_rmse":     round(l_rmse, 2),
            "lstm_mae":      round(l_mae,  2),
            "lstm_mape":     round(l_mape, 2),
            "lstm_r2":       round(l_r2,   4),
        })

# ─────────────────────────────────────────────
# BUILD DATAFRAME
# ─────────────────────────────────────────────
results_df = pd.DataFrame(all_rows)

# ─────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "rolling_window_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nSaved → {csv_path}")

# ─────────────────────────────────────────────
# PLOT 1: rolling_window_plot.png  (MAPE per window)
# ─────────────────────────────────────────────
windows = np.arange(1, n_windows + 1)
first_city_rows = results_df[results_df["city"] == CITIES[0]].sort_values("window_number")
tick_labels = first_city_rows["test_start"].tolist()

fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
fig.suptitle(
    f"Walk-Forward Rolling Window Validation — MAPE per Window\n"
    f"(Train={TRAIN_W}m, Test={TEST_W}m, slide=1m, {n_windows} windows | "
    f"seas_diff(12) | LSTM(16,8))",
    fontsize=13, fontweight="bold", y=0.99,
)

for ax, city in zip(axes, CITIES):
    city_res = results_df[results_df["city"] == city].sort_values("window_number")
    s_mapes  = city_res["sarima_mape"].values
    l_mapes  = city_res["lstm_mape"].values
    s_mean = np.nanmean(s_mapes); s_std = np.nanstd(s_mapes)
    l_mean = np.nanmean(l_mapes); l_std = np.nanstd(l_mapes)

    ax.plot(windows, s_mapes, color="#4e79a7", linewidth=1.8,
            marker="o", markersize=4, label="SARIMA")
    ax.plot(windows, l_mapes, color="#f28e2b", linewidth=1.8,
            marker="s", markersize=4, label="LSTM")
    ax.axhline(s_mean, color="#4e79a7", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.axhline(l_mean, color="#f28e2b", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_title(
        f"{city} — SARIMA {s_mean:.1f}±{s_std:.1f}%   |   "
        f"LSTM {l_mean:.1f}±{l_std:.1f}%",
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
# PLOT 2: rolling_window_r2_plot.png  (R² mean ± std bar chart)
# ─────────────────────────────────────────────
x     = np.arange(len(CITIES))
width = 0.35

s_r2_means = [np.nanmean(city_sarima_r2[c]) for c in CITIES]
s_r2_stds  = [np.nanstd(city_sarima_r2[c])  for c in CITIES]
l_r2_means = [np.nanmean(city_lstm_r2[c])   for c in CITIES]
l_r2_stds  = [np.nanstd(city_lstm_r2[c])    for c in CITIES]

fig, ax = plt.subplots(figsize=(11, 5))
bars_s = ax.bar(x - width / 2, s_r2_means, width, yerr=s_r2_stds,
                label="SARIMA", color="#4e79a7", edgecolor="white",
                capsize=4, alpha=0.85)
bars_l = ax.bar(x + width / 2, l_r2_means, width, yerr=l_r2_stds,
                label="LSTM",   color="#f28e2b", edgecolor="white",
                capsize=4, alpha=0.85)

ax.set_title(
    f"Walk-Forward Rolling Window — R² Mean ± Std per City\n"
    f"(Train={TRAIN_W}m, Test={TEST_W}m, {n_windows} windows | seas_diff(12) | LSTM(16,8))",
    fontsize=12, fontweight="bold",
)
ax.set_ylabel("R² (mean ± std across windows)", fontsize=10)
ax.set_xlabel("City", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(CITIES, fontsize=10)
ax.axhline(0, color="black", linestyle=":", linewidth=0.9, alpha=0.6)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.35)

for bar, mean_val in zip(bars_s, s_r2_means):
    ax.text(bar.get_x() + bar.get_width() / 2,
            mean_val + 0.01 if mean_val >= 0 else mean_val - 0.03,
            f"{mean_val:.3f}", ha="center",
            va="bottom" if mean_val >= 0 else "top", fontsize=8)
for bar, mean_val in zip(bars_l, l_r2_means):
    ax.text(bar.get_x() + bar.get_width() / 2,
            mean_val + 0.01 if mean_val >= 0 else mean_val - 0.03,
            f"{mean_val:.3f}", ha="center",
            va="bottom" if mean_val >= 0 else "top", fontsize=8)

plt.tight_layout()
r2_plot_path = os.path.join(OUTPUT_DIR, "rolling_window_r2_plot.png")
plt.savefig(r2_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {r2_plot_path}")

# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 82)
print("ROLLING WINDOW SUMMARY — Mean ± Std across all windows per city")
print("=" * 82)

# MAPE mean ± std
print(f"\n{'City':<10}  {'SARIMA MAPE':>16}  {'LSTM MAPE':>16}")
print("-" * 46)
for city in CITIES:
    r = results_df[results_df["city"] == city]
    s_mp = r["sarima_mape"]
    l_mp = r["lstm_mape"]
    print(f"{city:<10}  {s_mp.mean():>8.2f}±{s_mp.std():>5.2f}%  "
          f"{l_mp.mean():>8.2f}±{l_mp.std():>5.2f}%")

# R² mean ± std (per-window)
print(f"\n{'City':<10}  {'SARIMA R²':>18}  {'LSTM R²':>18}")
print("-" * 50)
for city in CITIES:
    s_vals = city_sarima_r2[city]
    l_vals = city_lstm_r2[city]
    s_mean = np.nanmean(s_vals); s_std = np.nanstd(s_vals)
    l_mean = np.nanmean(l_vals); l_std = np.nanstd(l_vals)
    print(f"{city:<10}  {s_mean:>+9.4f}±{s_std:>6.4f}  "
          f"{l_mean:>+9.4f}±{l_std:>6.4f}")

print("=" * 82)
print(f"\nPHASE 4 COMPLETE")
for fname in ["rolling_window_results.csv", "rolling_window_plot.png",
              "rolling_window_r2_plot.png"]:
    fpath = os.path.join(OUTPUT_DIR, fname)
    print(f"  {fname:38s}  {os.path.getsize(fpath):>8,} bytes")
print("=" * 82)
