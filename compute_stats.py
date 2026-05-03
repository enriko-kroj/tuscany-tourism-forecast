"""
BSP4 - Descriptive Statistics
Computes MAX, MIN, AVG, STDDEV for:
1. Each individual file
2. Each city (all months pooled)
3. All data combined

Run: python3 ~/Desktop/compute_stats.py
Output: ~/Desktop/BSP4_Statistics/
"""

import os
import pandas as pd
import numpy as np
import glob

DESKTOP      = os.path.expanduser("~/Desktop")
DATA_FOLDER  = os.path.join(DESKTOP, "Tuscany_Mastercard_Data")
OUTPUT_FOLDER = os.path.join(DESKTOP, "BSP4_Statistics")

CITIES = ["Arezzo", "Firenze", "Lucca", "Pisa", "Siena"]

# Columns to skip (non-numeric or not meaningful)
SKIP_COLS = [
    "Categoria", "Nazione_origine", "area", "region__c",
    "municipality__c", "province__c", "view__c",
    "week_start_date__c", "year_month_date__c", "year__c",
    "month__c", "year__c"
]


def load_csv(filepath):
    """Load a CSV file and return a clean dataframe."""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None


def get_numeric_cols(df):
    """Get numeric columns, excluding non-meaningful ones."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in SKIP_COLS]


def compute_stats(df, label=""):
    """Compute MAX, MIN, AVG, STDDEV for all numeric columns."""
    num_cols = get_numeric_cols(df)
    if not num_cols:
        return pd.DataFrame()

    stats = pd.DataFrame({
        "variable": num_cols,
        "count":    [df[c].count() for c in num_cols],
        "min":      [df[c].min()   for c in num_cols],
        "max":      [df[c].max()   for c in num_cols],
        "mean":     [df[c].mean()  for c in num_cols],
        "std":      [df[c].std()   for c in num_cols],
        "median":   [df[c].median() for c in num_cols],
    })

    # Round to 4 decimal places
    for col in ["min", "max", "mean", "std", "median"]:
        stats[col] = stats[col].round(4)

    return stats


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("=" * 60)
    print("BSP4 — Descriptive Statistics")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────
    # PART 1: Statistics for each individual file
    # ─────────────────────────────────────────────────────────
    print("\nPART 1: Computing stats for each file...")

    all_file_stats = []

    for city in CITIES:
        city_folder = os.path.join(DATA_FOLDER, city)
        csv_files   = sorted(glob.glob(os.path.join(city_folder, "*.csv")))

        for filepath in csv_files:
            filename = os.path.basename(filepath).replace(".csv", "")
            df = load_csv(filepath)
            if df is None:
                continue

            stats = compute_stats(df, label=filename)
            if stats.empty:
                continue

            stats.insert(0, "file",  filename)
            stats.insert(0, "city",  city)
            all_file_stats.append(stats)
            print(f"  ✓ {filename}")

    # Save Part 1
    if all_file_stats:
        df_all_files = pd.concat(all_file_stats, ignore_index=True)
        out1 = os.path.join(OUTPUT_FOLDER, "1_stats_per_file.csv")
        df_all_files.to_csv(out1, index=False)
        print(f"\n  Saved: 1_stats_per_file.csv ({len(df_all_files)} rows)")

    # ─────────────────────────────────────────────────────────
    # PART 2: Statistics per city (all months pooled)
    # ─────────────────────────────────────────────────────────
    print("\nPART 2: Computing stats per city...")

    all_city_stats = []

    for city in CITIES:
        city_folder = os.path.join(DATA_FOLDER, city)
        csv_files   = sorted(glob.glob(os.path.join(city_folder, "*.csv")))

        # Load and pool all months for this city
        dfs = []
        for filepath in csv_files:
            df = load_csv(filepath)
            if df is not None:
                dfs.append(df)

        if not dfs:
            continue

        city_df = pd.concat(dfs, ignore_index=True)
        stats   = compute_stats(city_df)
        if stats.empty:
            continue

        stats.insert(0, "city",        city)
        stats.insert(1, "files_pooled", len(dfs))
        stats.insert(2, "total_rows",   len(city_df))
        all_city_stats.append(stats)

        print(f"  ✓ {city}: {len(dfs)} files, {len(city_df):,} rows")

    # Save Part 2
    if all_city_stats:
        df_cities = pd.concat(all_city_stats, ignore_index=True)
        out2 = os.path.join(OUTPUT_FOLDER, "2_stats_per_city.csv")
        df_cities.to_csv(out2, index=False)
        print(f"\n  Saved: 2_stats_per_city.csv ({len(df_cities)} rows)")

    # ─────────────────────────────────────────────────────────
    # PART 3: Overall statistics (everything pooled)
    # ─────────────────────────────────────────────────────────
    print("\nPART 3: Computing overall statistics...")

    all_dfs = []
    for city in CITIES:
        city_folder = os.path.join(DATA_FOLDER, city)
        csv_files   = sorted(glob.glob(os.path.join(city_folder, "*.csv")))
        for filepath in csv_files:
            df = load_csv(filepath)
            if df is not None:
                df["city"] = city
                all_dfs.append(df)

    if all_dfs:
        full_df    = pd.concat(all_dfs, ignore_index=True)
        stats      = compute_stats(full_df)
        stats.insert(0, "scope",       "ALL CITIES")
        stats.insert(1, "total_files", len(all_dfs))
        stats.insert(2, "total_rows",  len(full_df))

        out3 = os.path.join(OUTPUT_FOLDER, "3_stats_overall.csv")
        stats.to_csv(out3, index=False)
        print(f"  ✓ All cities: {len(all_dfs)} files, {len(full_df):,} rows")
        print(f"\n  Saved: 3_stats_overall.csv")

        # Also print overall stats to terminal for quick review
        print("\n" + "=" * 60)
        print("OVERALL STATISTICS SUMMARY")
        print("=" * 60)
        print(stats[["variable", "count", "min", "max", "mean", "std"]].to_string(index=False))

    print("\n" + "=" * 60)
    print(f"COMPLETE — Output saved to: {OUTPUT_FOLDER}")
    print("Files produced:")
    print("  1_stats_per_file.csv  — stats for each city/month file")
    print("  2_stats_per_city.csv  — stats per city (all months pooled)")
    print("  3_stats_overall.csv   — stats across all data combined")
    print("=" * 60)


if __name__ == "__main__":
    main()
