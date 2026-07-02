"""
run_pilot.py

Main entry point for the Healthcare Warehouse Digital Twin pilot.
Equivalent in role to the reference repo's `run_simulator.py`:
loads configuration, builds the warehouse, runs the simulation,
generates the fragmented "before centralisation" dataset, computes
forecast accuracy and data quality KPIs, and writes all outputs to
/data and /outputs for use by the dashboard and report.

Usage:
    PYTHONPATH=src python3 run_pilot.py
"""

import sys
import os
import json
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from digital_twin.warehouse import SKU, Warehouse
from digital_twin.simulation import WarehouseSimulation, ManualBaselineSimulation
from digital_twin.fragmentation import fragment_dataset, compute_data_quality_metrics
from digital_twin.forecasting import forecast_accuracy_comparison, summarize_accuracy_improvement

CONFIG_PATH = os.environ.get(
    "WAREHOUSE_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), "config", "warehouse_config.yaml"),
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_warehouse(cfg: dict) -> Warehouse:
    site_cfg = cfg["sites"][0]  # pilot scope: single site
    skus = [
        SKU(
            sku_id=s["sku_id"],
            name=s["name"],
            category=s["category"],
            abc_class=s["abc_class"],
            xyz_class=s["xyz_class"],
            stock=s["initial_stock"],
            reorder_point=s["reorder_point"],
            reorder_qty=s["reorder_qty"],
            lead_time_days=s["lead_time_days"],
            demand_mean=s["demand_mean"],
            demand_std=s["demand_std"],
        )
        for s in cfg["skus"]
    ]
    return Warehouse(site_id=site_cfg["site_id"], site_name=site_cfg["site_name"], skus=skus)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading configuration...")
    cfg = load_config(CONFIG_PATH)
    sim_cfg = cfg["simulation"]
    dq_cfg = cfg["data_quality"]

    print(f"Building warehouse: {cfg['sites'][0]['site_name']} "
          f"({len(cfg['skus'])} SKUs, {sim_cfg['duration_days']}-day pilot)")

    # --- Run 1: Digital Twin (clean, centralised data) ---
    print("Running digital twin simulation (centralised, real-time stock visibility)...")
    warehouse_dt = build_warehouse(cfg)
    sim_dt = WarehouseSimulation(
        warehouse=warehouse_dt,
        duration_days=sim_cfg["duration_days"],
        random_seed=sim_cfg["random_seed"],
        time_step_hours=sim_cfg["time_step_hours"],
    )
    sim_dt.run_simulation()
    clean_df = sim_dt.get_store_history_dataframe()
    reorder_dt_df = sim_dt.get_reorder_log_dataframe()
    kpi_dt = sim_dt.get_kpi_summary()

    # --- Run 2: Manual Baseline (same demand sequence, fragmented/lagged
    #     stock visibility drives reorder decisions instead) ---
    print("Running manual-baseline simulation (fragmented, lagged stock visibility)...")
    warehouse_mb = build_warehouse(cfg)  # fresh, independent SKU state
    sim_mb = ManualBaselineSimulation(
        warehouse=warehouse_mb,
        duration_days=sim_cfg["duration_days"],
        random_seed=sim_cfg["random_seed"],  # same seed -> identical demand draws
        time_step_hours=sim_cfg["time_step_hours"],
        reporting_lag_days=dq_cfg["reporting_lag_days"],
        manual_entry_error_rate=dq_cfg["manual_entry_error_rate"],
        missing_entry_rate=dq_cfg["missing_entry_rate"],
    )
    sim_mb.run_simulation()
    manual_baseline_df = sim_mb.get_store_history_dataframe()
    reorder_mb_df = sim_mb.get_reorder_log_dataframe()
    kpi_mb = sim_mb.get_kpi_summary()

    print("Generating fragmented 'spreadsheet export' dataset for data-quality analysis...")
    rng = np.random.default_rng(sim_cfg["random_seed"] + 1)
    fragmented_df = fragment_dataset(clean_df, dq_cfg, rng)

    print("Computing data quality / inconsistency metrics...")
    dq_metrics = compute_data_quality_metrics(clean_df, fragmented_df)

    print("Computing forecast accuracy comparison (manual baseline vs digital twin)...")
    forecast_df = forecast_accuracy_comparison(clean_df, fragmented_df, window=7)
    forecast_summary = summarize_accuracy_improvement(forecast_df)

    # --- Persist all outputs ---
    clean_df.to_csv(os.path.join(DATA_DIR, "clean_ground_truth.csv"), index=False)
    manual_baseline_df.to_csv(os.path.join(DATA_DIR, "manual_baseline_simulation.csv"), index=False)
    fragmented_df.to_csv(os.path.join(DATA_DIR, "fragmented_spreadsheet_sim.csv"), index=False)
    reorder_dt_df.to_csv(os.path.join(DATA_DIR, "reorder_log_digital_twin.csv"), index=False)
    reorder_mb_df.to_csv(os.path.join(DATA_DIR, "reorder_log_manual_baseline.csv"), index=False)
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, "forecast_accuracy_comparison.csv"), index=False)

    fill_rate_improvement_pct = (
        kpi_dt["site_fill_rate_pct"] - kpi_mb["site_fill_rate_pct"]
    )
    stockout_reduction = kpi_mb["total_stockout_days"] - kpi_dt["total_stockout_days"]

    pilot_results = {
        "simulation_config": {
            "duration_days": sim_cfg["duration_days"],
            "site": cfg["sites"][0]["site_name"],
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
    with open(os.path.join(OUTPUT_DIR, "pilot_results.json"), "w") as f:
        json.dump(pilot_results, f, indent=2)

    # --- Console report ---
    print("\n" + "=" * 60)
    print("PILOT SIMULATION COMPLETE")
    print("=" * 60)
    print(f"{'Metric':<35}{'Manual Baseline':>15}{'Digital Twin':>15}")
    print(f"{'Site fill rate (%)':<35}{kpi_mb['site_fill_rate_pct']:>15}{kpi_dt['site_fill_rate_pct']:>15}")
    print(f"{'Total stockout days':<35}{kpi_mb['total_stockout_days']:>15}{kpi_dt['total_stockout_days']:>15}")
    print(f"{'Reorders placed':<35}{kpi_mb['total_reorders_placed']:>15}{kpi_dt['total_reorders_placed']:>15}")
    print("-" * 65)
    print(f"Data inconsistency rate (simulated manual spreadsheets): {dq_metrics['inconsistency_rate_pct']}%")
    print(f"Duplicate records found:                                 {dq_metrics['duplicate_records']}")
    print(f"Manual baseline forecast WMAPE:                          {forecast_summary['avg_manual_baseline_WMAPE_pct']}%")
    print(f"Digital twin forecast WMAPE:                             {forecast_summary['avg_digital_twin_WMAPE_pct']}%")
    print(f"Forecast accuracy improvement:                           {forecast_summary['forecast_accuracy_improvement_pct']}%")
    print(f"Stockout days avoided:                                   {stockout_reduction}")
    print("=" * 60)
    print(f"\nOutputs written to: {DATA_DIR} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
