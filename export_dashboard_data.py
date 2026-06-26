"""
export_dashboard_data.py

Consolidates all pilot outputs (clean simulation, manual baseline
simulation, fragmented spreadsheet sample, forecast comparison, KPI
summary) into a single dashboard_data.json file that the static HTML
dashboard loads directly. This avoids needing a running Python/Dash
server for the deliverable -- the dashboard works as a self-contained
file suitable for the report, poster, or presentation.
"""

import os
import json
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main():
    clean = pd.read_csv(os.path.join(DATA_DIR, "clean_ground_truth.csv"))
    manual = pd.read_csv(os.path.join(DATA_DIR, "manual_baseline_simulation.csv"))
    fragmented = pd.read_csv(os.path.join(DATA_DIR, "fragmented_spreadsheet_sim.csv"))
    forecast = pd.read_csv(os.path.join(OUTPUT_DIR, "forecast_accuracy_comparison.csv"))

    with open(os.path.join(OUTPUT_DIR, "pilot_results.json")) as f:
        results = json.load(f)

    # --- Site-level daily total stock (digital twin vs manual baseline) ---
    dt_daily = clean.groupby("day")["stock_level"].sum().round(1)
    mb_daily = manual.groupby("day")["true_stock_level"].sum().round(1)
    mb_perceived_daily = manual.groupby("day")["perceived_stock_level"].sum().round(1)

    daily_stock = {
        "days": dt_daily.index.tolist(),
        "digital_twin_total_stock": dt_daily.values.tolist(),
        "manual_baseline_true_stock": mb_daily.values.tolist(),
        "manual_baseline_perceived_stock": mb_perceived_daily.values.tolist(),
    }

    # --- Per-SKU daily stock series (for SKU drill-down chart) ---
    per_sku_series = {}
    for sku_id in clean["sku_id"].unique():
        c = clean[clean["sku_id"] == sku_id].sort_values("day")
        m = manual[manual["sku_id"] == sku_id].sort_values("day")
        per_sku_series[sku_id] = {
            "sku_name": c["sku_name"].iloc[0],
            "abc_class": c["abc_class"].iloc[0],
            "xyz_class": c["xyz_class"].iloc[0],
            "days": c["day"].tolist(),
            "digital_twin_stock": c["stock_level"].round(1).tolist(),
            "manual_baseline_true_stock": m["true_stock_level"].round(1).tolist(),
            "manual_baseline_perceived_stock": m["perceived_stock_level"].round(1).tolist(),
            "reorder_point": float(c["reorder_point"].iloc[0]),
            "demand": c["demand"].round(2).tolist(),
        }

    # --- Sample of fragmented spreadsheet rows (for "messy data" table) ---
    frag_sample = (
        fragmented.sort_values(["day", "sku_id"])
        .head(25)[["day", "true_day", "site_id", "sku_id", "stock_level", "source_sheet"]]
        .fillna("—")
        .to_dict(orient="records")
    )

    # --- Forecast comparison table ---
    forecast_table = forecast.to_dict(orient="records")

    dashboard_data = {
        "results": results,
        "daily_stock": daily_stock,
        "per_sku_series": per_sku_series,
        "fragmented_sample": frag_sample,
        "forecast_table": forecast_table,
    }

    out_path = os.path.join(OUTPUT_DIR, "dashboard_data.json")
    with open(out_path, "w") as f:
        json.dump(dashboard_data, f)

    print(f"Dashboard data written to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
