"""
BSP4 Phase 1: Data Preparation
Steps 1-7 as defined in PROJECT_BRIEF.md
Outputs saved to ~/Desktop/BSP4_Statistics/
"""

import os
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = os.path.expanduser("~/Desktop/Tuscany_Mastercard_Data")
OUTPUT_DIR = os.path.expanduser("~/Desktop/BSP4_Statistics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CITIES = ["Arezzo", "Firenze", "Lucca", "Pisa", "Siena"]

COVID_START = "2020-03-01"
COVID_END   = "2020-12-01"

# ─────────────────────────────────────────────
# STEP 1: Combine all CSV files into master df
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading and combining all CSV files …")

frames = []
files_loaded = 0

for city in CITIES:
    pattern = os.path.join(DATA_DIR, city, "*.csv")
    city_files = sorted(glob.glob(pattern))
    print(f"  {city}: {len(city_files)} files found")
    for fpath in city_files:
        df = pd.read_csv(fpath, low_memory=False)
        # Filter to Comune level only
        df = df[df["area"] == "Comune"].copy()
        # Keep only required columns
        cols_keep = [
            "year_month_date__c",
            "year__c",
            "month__c",
            "indexed_number_of_cards__c",
            "Categoria",
            "Nazione_origine",
            "municipality__c",
        ]
        df = df[[c for c in cols_keep if c in df.columns]]
        df["city"] = city
        frames.append(df)
        files_loaded += 1

master = pd.concat(frames, ignore_index=True)
master["year_month_date__c"] = pd.to_datetime(master["year_month_date__c"])
master["indexed_number_of_cards__c"] = pd.to_numeric(
    master["indexed_number_of_cards__c"], errors="coerce"
)

print(f"  Files loaded  : {files_loaded}")
print(f"  Master rows   : {len(master):,}")
print(f"  Comune rows   : {len(master):,}  (already filtered)")

# ─────────────────────────────────────────────
# STEP 2: Build one time series per city
# (sum indexed_number_of_cards__c per city/month)
# ─────────────────────────────────────────────
print("\nSTEP 2: Aggregating into monthly time series per city …")

ts = (
    master
    .groupby(["city", "year_month_date__c"], as_index=False)["indexed_number_of_cards__c"]
    .sum()
    .rename(columns={
        "year_month_date__c": "date",
        "indexed_number_of_cards__c": "tourist_volume",
    })
    .sort_values(["city", "date"])
    .reset_index(drop=True)
)

for city in CITIES:
    n = len(ts[ts["city"] == city])
    print(f"  {city}: {n} monthly data points")

# ─────────────────────────────────────────────
# STEP 3: Handle missing values / gaps
# ─────────────────────────────────────────────
print("\nSTEP 3: Checking for gaps in monthly sequences …")

full_date_range = pd.date_range(
    start=ts["date"].min(), end=ts["date"].max(), freq="MS"
)

ts_complete_frames = []
for city in CITIES:
    city_df = ts[ts["city"] == city].set_index("date")
    city_df = city_df.reindex(full_date_range)
    city_df["city"] = city
    city_df.index.name = "date"
    missing_months = city_df["tourist_volume"].isna().sum()
    if missing_months > 0:
        print(f"  {city}: {missing_months} missing months — filling via linear interpolation")
        city_df["tourist_volume"] = city_df["tourist_volume"].interpolate(method="linear")
        city_df["gap_filled"] = True
    else:
        print(f"  {city}: no gaps")
        city_df["gap_filled"] = False
    ts_complete_frames.append(city_df.reset_index())

ts = pd.concat(ts_complete_frames, ignore_index=True).sort_values(["city", "date"])

# ─────────────────────────────────────────────
# STEP 4: Handle COVID outlier (2020-03 – 2020-12)
# ─────────────────────────────────────────────
print("\nSTEP 4: Flagging COVID period (2020-03 to 2020-12) …")
# Decision: keep raw values as-is, but add a boolean flag column so models
# can optionally use it as an exogenous dummy variable.
covid_mask = (ts["date"] >= COVID_START) & (ts["date"] <= COVID_END)
ts["covid_period"] = covid_mask

covid_stats = (
    ts[covid_mask]
    .groupby("city")["tourist_volume"]
    .agg(["min", "mean", "max"])
    .round(2)
)
print("  COVID-period tourist_volume stats per city:")
print(covid_stats.to_string())
print("  → Values kept as-is; 'covid_period' flag column added for modelling.")

# ─────────────────────────────────────────────
# STEP 5: Stationarity check — ADF test per city
# ─────────────────────────────────────────────
print("\nSTEP 5: ADF stationarity tests …")

adf_lines = []
adf_lines.append("ADF Stationarity Report — Phase 1 Data Preparation")
adf_lines.append("=" * 60)

for city in CITIES:
    series = ts[ts["city"] == city]["tourist_volume"].dropna().values
    adf_result = adfuller(series, autolag="AIC")
    stat, pval, lags, nobs, crit_vals = (
        adf_result[0], adf_result[1], adf_result[2], adf_result[3], adf_result[4]
    )
    stationary = pval < 0.05
    conclusion = "STATIONARY (p<0.05)" if stationary else "NON-STATIONARY (p≥0.05)"

    # If non-stationary, test first difference
    diff_info = ""
    if not stationary:
        diff_series = np.diff(series)
        adf_d1 = adfuller(diff_series, autolag="AIC")
        d1_stat, d1_pval = adf_d1[0], adf_d1[1]
        d1_conclusion = "STATIONARY after d=1" if d1_pval < 0.05 else "still non-stationary after d=1"
        diff_info = f"\n    First-difference ADF: stat={d1_stat:.4f}, p={d1_pval:.4f} → {d1_conclusion}"

    block = (
        f"\n{city}\n"
        f"  ADF statistic : {stat:.4f}\n"
        f"  p-value       : {pval:.4f}\n"
        f"  Lags used     : {lags}\n"
        f"  N obs         : {nobs}\n"
        f"  Critical vals : 1%={crit_vals['1%']:.3f}, 5%={crit_vals['5%']:.3f}, 10%={crit_vals['10%']:.3f}\n"
        f"  Conclusion    : {conclusion}"
        f"{diff_info}"
    )
    print(block)
    adf_lines.append(block)

stationarity_path = os.path.join(OUTPUT_DIR, "stationarity_report.txt")
with open(stationarity_path, "w") as f:
    f.write("\n".join(adf_lines) + "\n")
print(f"\n  Saved → {stationarity_path}")

# ─────────────────────────────────────────────
# STEP 6: Visualise all 5 city series
# ─────────────────────────────────────────────
print("\nSTEP 6: Generating time series plot …")

CITY_COLORS = {
    "Arezzo":  "#1f77b4",
    "Firenze": "#ff7f0e",
    "Lucca":   "#2ca02c",
    "Pisa":    "#d62728",
    "Siena":   "#9467bd",
}

fig, ax = plt.subplots(figsize=(14, 6))

for city in CITIES:
    city_df = ts[ts["city"] == city].sort_values("date")
    ax.plot(
        city_df["date"],
        city_df["tourist_volume"],
        label=city,
        color=CITY_COLORS[city],
        linewidth=1.8,
        marker="o",
        markersize=3,
    )

# Shade COVID period
ax.axvspan(
    pd.Timestamp(COVID_START),
    pd.Timestamp(COVID_END),
    color="red",
    alpha=0.10,
    label="COVID period (Mar–Dec 2020)",
)

ax.set_title(
    "Monthly Tourist Volume (indexed_number_of_cards__c)\n"
    "Summed across all categories & nationalities — Comune level",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Month", fontsize=11)
ax.set_ylabel("Indexed Number of Cards (summed)", fontsize=11)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45, ha="right")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "timeseries_plot.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"  Saved → {plot_path}")

# ─────────────────────────────────────────────
# STEP 7: Save master_timeseries.csv
# ─────────────────────────────────────────────
print("\nSTEP 7: Saving master_timeseries.csv …")

ts_out = ts[["city", "date", "tourist_volume", "covid_period", "gap_filled"]].copy()
ts_out["date"] = ts_out["date"].dt.strftime("%Y-%m-%d")

out_path = os.path.join(OUTPUT_DIR, "master_timeseries.csv")
ts_out.to_csv(out_path, index=False)

print(f"  Rows saved : {len(ts_out):,}")
print(f"  Saved → {out_path}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 1 COMPLETE — Output files:")
for fname in ["master_timeseries.csv", "timeseries_plot.png", "stationarity_report.txt"]:
    fpath = os.path.join(OUTPUT_DIR, fname)
    size = os.path.getsize(fpath)
    print(f"  {fname:35s} {size:>8,} bytes")
print("=" * 60)
