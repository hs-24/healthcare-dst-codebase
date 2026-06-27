"""
generate_multi_site_pilot.py

Runs the same digital-twin-vs-manual-baseline pilot (run_pilot.py +
export_dashboard_data.py combined) independently for three warehouse
sites, so the live dashboard can offer a "Site 1 / Site 2 / Site 3"
widget and demonstrate unified oversight across multiple locations --
directly answering the Warehousing Infrastructure problem in Section 1
of the report ("warehousing operations span multiple locations and
systems with no unified oversight mechanism").

All three sites share the same SKU catalogue (config/warehouse_config.yaml)
so the GDP colour taxonomy / storage zoning stays consistent; only the
site identity and random seed differ, giving each site its own demand
draw and its own (independently corrupted) fragmented-spreadsheet
pattern -- exactly like three real sites running the same product line
through three different local manual processes.

Run:
    python build/generate_multi_site_pilot.py

Writes:
    outputs/dashboard_data_SITE_A.json
    outputs/dashboard_data_SITE_B.json
    outputs/dashboard_data_SITE_C.json
    outputs/multi_site_summary.json   (cross-site KPI rollup)
"""

import sys
import os
import json
import copy
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from digital_twin.warehouse import SKU, Warehouse
from digital_twin.simulation import WarehouseSimulation, ManualBaselineSimulation
from digital_twin.fragmentation import fragment_dataset, compute_data_quality_metrics
from digital_twin.forecasting import forecast_accuracy_comparison, summarize_accuracy_improvement

import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT, "config", "warehouse_config.yaml")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

SITE_PROFILES = [
    {"site_id": "SITE_A", "site_name": "Central Distribution Hub (Site 1)", "random_seed": 42},
    {"site_id": "SITE_B", "site_name": "North Regional Hub (Site 2)", "random_seed": 7},
    {"site_id": "SITE_C", "site_name": "East Satellite Depot (Site 3)", "random_seed": 99},
]


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_warehouse(cfg, site_id, site_name):
    skus = [
        SKU(
            sku_id=s["sku_id"], name=s["name"], category=s["category"],
            abc_class=s["abc_class"], xyz_class=s["xyz_class"], stock=s["initial_stock"],
            reorder_point=s["reorder_point"], reorder_qty=s["reorder_qty"],
            lead_time_days=s["lead_time_days"], demand_mean=s["demand_mean"], demand_std=s["demand_std"],
        )
        for s in cfg["skus"]
    ]
    return Warehouse(site_id=site_id, site_name=site_name, skus=skus)


def run_site_pilot(cfg, profile):
    sim_cfg = cfg["simulation"]
    dq_cfg = cfg["data_quality"]
    seed = profile["random_seed"]

    wh_dt = build_warehouse(cfg, profile["site_id"], profile["site_name"])
    sim_dt = WarehouseSimulation(wh_dt, sim_cfg["duration_days"], seed, sim_cfg["time_step_hours"])
    sim_dt.run_simulation()
    clean_df = sim_dt.get_store_history_dataframe()
    kpi_dt = sim_dt.get_kpi_summary()

    wh_mb = build_warehouse(cfg, profile["site_id"], profile["site_name"])
    sim_mb = ManualBaselineSimulation(
        wh_mb, sim_cfg["duration_days"], seed, sim_cfg["time_step_hours"],
        dq_cfg["reporting_lag_days"], dq_cfg["manual_entry_error_rate"], dq_cfg["missing_entry_rate"],
    )
    sim_mb.run_simulation()
    manual_df = sim_mb.get_store_history_dataframe()
    kpi_mb = sim_mb.get_kpi_summary()

    rng = np.random.default_rng(seed + 1)
    fragmented_df = fragment_dataset(clean_df, dq_cfg, rng)
    dq_metrics = compute_data_quality_metrics(clean_df, fragmented_df)

    forecast_df = forecast_accuracy_comparison(clean_df, fragmented_df, window=7)
    forecast_summary = summarize_accuracy_improvement(forecast_df)

    fill_rate_improvement_pct = kpi_dt["site_fill_rate_pct"] - kpi_mb["site_fill_rate_pct"]
    stockout_reduction = kpi_mb["total_stockout_days"] - kpi_dt["total_stockout_days"]

    results = {
        "simulation_config": {
            "duration_days": sim_cfg["duration_days"],
            "site": profile["site_name"],
            "site_id": profile["site_id"],
            "num_skus": len(cfg["skus"]),
        },
        "data_quality_metrics": dq_metrics,
        "forecast_accuracy_summary": forecast_summary,
        "operational_comparison": {
            "digital_twin_fill_rate_pct": kpi_dt["site_fill_rate_pct"],
            "manual_baseline_fill_rate_pct": kpi_mb["site_fill_rate_pct"],
            "fill_rate_improvement_pct_points": round(fill_rate_improvement_pct, 2),
            "digital_twin_stockout_days": kpi_dt["total_stockout_days"],
            "manual_baseline_stockout_days": kpi_mb["total_stockout_days"],
            "stockout_days_avoided": int(stockout_reduction),
        },
        "kpi_summary_digital_twin": kpi_dt,
        "kpi_summary_manual_baseline": kpi_mb,
    }

    per_sku_series = {}
    for sku_id in clean_df["sku_id"].unique():
        c = clean_df[clean_df["sku_id"] == sku_id].sort_values("day")
        m = manual_df[manual_df["sku_id"] == sku_id].sort_values("day")
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

    frag_sample = (
        fragmented_df.sort_values(["day", "sku_id"])
        .head(25)[["day", "true_day", "site_id", "sku_id", "stock_level", "source_sheet"]]
        .fillna("—")
        .to_dict(orient="records")
    )
    forecast_table = forecast_df.to_dict(orient="records")

    dt_daily = clean_df.groupby("day")["stock_level"].sum().round(1)
    mb_daily = manual_df.groupby("day")["true_stock_level"].sum().round(1)
    mb_perceived_daily = manual_df.groupby("day")["perceived_stock_level"].sum().round(1)
    daily_stock = {
        "days": dt_daily.index.tolist(),
        "digital_twin_total_stock": dt_daily.values.tolist(),
        "manual_baseline_true_stock": mb_daily.values.tolist(),
        "manual_baseline_perceived_stock": mb_perceived_daily.values.tolist(),
    }

    return {
        "results": results,
        "daily_stock": daily_stock,
        "per_sku_series": per_sku_series,
        "fragmented_sample": frag_sample,
        "forecast_table": forecast_table,
    }


def main():
    cfg = load_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {"sites": []}

    for profile in SITE_PROFILES:
        print(f"Running pilot for {profile['site_name']} (seed={profile['random_seed']})...")
        dashboard_data = run_site_pilot(cfg, profile)
        out_path = os.path.join(OUTPUT_DIR, f"dashboard_data_{profile['site_id']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dashboard_data, f)
        print(f"  -> {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")

        oc = dashboard_data["results"]["operational_comparison"]
        dq = dashboard_data["results"]["data_quality_metrics"]
        fc = dashboard_data["results"]["forecast_accuracy_summary"]
        summary["sites"].append({
            "site_id": profile["site_id"],
            "site_name": profile["site_name"],
            "digital_twin_fill_rate_pct": oc["digital_twin_fill_rate_pct"],
            "manual_baseline_fill_rate_pct": oc["manual_baseline_fill_rate_pct"],
            "stockout_days_avoided": oc["stockout_days_avoided"],
            "data_inconsistency_rate_pct": dq["inconsistency_rate_pct"],
            "forecast_accuracy_improvement_pct": fc["forecast_accuracy_improvement_pct"],
        })

    summary_path = os.path.join(OUTPUT_DIR, "multi_site_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCross-site summary written to {summary_path}")


if __name__ == "__main__":
    main()
