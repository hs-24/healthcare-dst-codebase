"""Run the three-site Hospital Supply Chain digital-twin pilot."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from digital_twin.forecasting import (  # noqa: E402
    forecast_accuracy_comparison,
    summarize_accuracy_improvement,
)
from digital_twin.fragmentation import (  # noqa: E402
    compute_data_quality_metrics,
    fragment_dataset,
)
from digital_twin.hospital_data import load_or_build_catalog  # noqa: E402
from digital_twin.simulation import ManualBaselineSimulation, WarehouseSimulation  # noqa: E402
from digital_twin.warehouse import SKU, Warehouse  # noqa: E402

CONFIG_PATH = ROOT / "config" / "warehouse_config.yaml"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _classification(group: pd.DataFrame) -> dict[str, tuple[str, str]]:
    ranked = group.sort_values("demand_mean", ascending=False)["sku_id"].tolist()
    result = {}
    for index, sku_id in enumerate(ranked):
        fraction = index / max(1, len(ranked))
        abc = "A" if fraction < 0.3 else "B" if fraction < 0.7 else "C"
        row = group[group["sku_id"] == sku_id].iloc[0]
        cv = float(row["demand_std"]) / max(float(row["demand_mean"]), 1.0)
        xyz = "X" if cv < 0.15 else "Y" if cv < 0.3 else "Z"
        result[sku_id] = (abc, xyz)
    return result


def build_warehouse(site_rows: pd.DataFrame) -> Warehouse:
    classes = _classification(site_rows)
    skus = []
    for row in site_rows.itertuples(index=False):
        abc, xyz = classes[row.sku_id]
        skus.append(SKU(
            sku_id=row.sku_id,
            name=row.sku_name,
            category=row.item_type,
            abc_class=abc,
            xyz_class=xyz,
            stock=row.initial_stock,
            reorder_point=row.reorder_point,
            reorder_qty=row.reorder_qty,
            lead_time_days=int(row.lead_time_days),
            demand_mean=row.demand_mean,
            demand_std=row.demand_std,
        ))
    first = site_rows.iloc[0]
    return Warehouse(first["site_id"], first["site_name"], skus)


def _run_site(site_rows: pd.DataFrame, cfg: dict, seed: int) -> dict:
    sim_cfg = cfg["simulation"]
    dq_cfg = cfg["data_quality"]

    digital = WarehouseSimulation(
        build_warehouse(site_rows), sim_cfg["duration_days"], seed,
        sim_cfg["time_step_hours"],
    )
    digital.run_simulation()

    manual = ManualBaselineSimulation(
        build_warehouse(site_rows), sim_cfg["duration_days"], seed,
        sim_cfg["time_step_hours"], dq_cfg["reporting_lag_days"],
        dq_cfg["manual_entry_error_rate"], dq_cfg["missing_entry_rate"],
    )
    manual.run_simulation()

    clean = digital.get_store_history_dataframe()
    manual_df = manual.get_store_history_dataframe()
    fragmented = fragment_dataset(clean, dq_cfg, np.random.default_rng(seed + 1000))
    forecast = forecast_accuracy_comparison(clean, fragmented, window=7)
    forecast.insert(0, "site_id", site_rows.iloc[0]["site_id"])

    return {
        "clean": clean,
        "manual": manual_df,
        "fragmented": fragmented,
        "forecast": forecast,
        "reorder_dt": digital.get_reorder_log_dataframe(),
        "reorder_manual": manual.get_reorder_log_dataframe(),
        "kpi_dt": digital.get_kpi_summary(),
        "kpi_manual": manual.get_kpi_summary(),
        "data_quality": compute_data_quality_metrics(clean, fragmented),
        "forecast_summary": summarize_accuracy_improvement(forecast),
    }


def _concat(site_runs: list[dict], key: str) -> pd.DataFrame:
    frames = [run[key] for run in site_runs if not run[key].empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    cfg = load_config()
    catalog = load_or_build_catalog(ROOT)

    print(f"Loaded {len(catalog)} normalized site/SKU records from Kaggle data")
    site_runs = []
    site_results = {}
    base_seed = int(cfg["simulation"]["random_seed"])

    for index, (site_id, rows) in enumerate(catalog.groupby("site_id", sort=True)):
        print(f"Running {site_id}: {rows.iloc[0]['site_name']} ({len(rows)} SKUs)")
        run = _run_site(rows, cfg, base_seed + index * 101)
        site_runs.append(run)
        dt = run["kpi_dt"]
        manual = run["kpi_manual"]
        site_results[site_id] = {
            "site_id": site_id,
            "site_name": rows.iloc[0]["site_name"],
            "num_skus": len(rows),
            "data_quality_metrics": run["data_quality"],
            "forecast_accuracy_summary": run["forecast_summary"],
            "operational_comparison": {
                "digital_twin_fill_rate_pct": dt["site_fill_rate_pct"],
                "manual_baseline_fill_rate_pct": manual["site_fill_rate_pct"],
                "fill_rate_improvement_pct_points": round(dt["site_fill_rate_pct"] - manual["site_fill_rate_pct"], 2),
                "digital_twin_stockout_days": dt["total_stockout_days"],
                "manual_baseline_stockout_days": manual["total_stockout_days"],
                "stockout_days_avoided": manual["total_stockout_days"] - dt["total_stockout_days"],
            },
            "kpi_summary_digital_twin": dt,
            "kpi_summary_manual_baseline": manual,
        }

    clean = _concat(site_runs, "clean")
    manual = _concat(site_runs, "manual")
    fragmented = _concat(site_runs, "fragmented")
    forecast = _concat(site_runs, "forecast")
    reorder_dt = _concat(site_runs, "reorder_dt")
    reorder_manual = _concat(site_runs, "reorder_manual")

    clean.to_csv(DATA_DIR / "clean_ground_truth.csv", index=False)
    manual.to_csv(DATA_DIR / "manual_baseline_simulation.csv", index=False)
    fragmented.to_csv(DATA_DIR / "fragmented_spreadsheet_sim.csv", index=False)
    reorder_dt.to_csv(DATA_DIR / "reorder_log_digital_twin.csv", index=False)
    reorder_manual.to_csv(DATA_DIR / "reorder_log_manual_baseline.csv", index=False)
    forecast.to_csv(OUTPUT_DIR / "forecast_accuracy_comparison.csv", index=False)

    site_values = list(site_results.values())
    aggregate = {
        "simulation_config": {
            "duration_days": cfg["simulation"]["duration_days"],
            "sites": len(site_results),
            "site_names": [site["site_name"] for site in site_values],
            "dataset": "Kaggle Hospital Supply Chain",
        },
        "sites": site_results,
        "network_summary": {
            "digital_twin_fill_rate_pct": round(np.mean([s["operational_comparison"]["digital_twin_fill_rate_pct"] for s in site_values]), 2),
            "manual_baseline_fill_rate_pct": round(np.mean([s["operational_comparison"]["manual_baseline_fill_rate_pct"] for s in site_values]), 2),
            "stockout_days_avoided": int(sum(s["operational_comparison"]["stockout_days_avoided"] for s in site_values)),
            "forecast_accuracy_improvement_pct": round(np.mean([s["forecast_accuracy_summary"]["forecast_accuracy_improvement_pct"] for s in site_values]), 2),
            "data_inconsistency_rate_pct": round(np.mean([s["data_quality_metrics"]["inconsistency_rate_pct"] for s in site_values]), 2),
        },
    }
    with (OUTPUT_DIR / "pilot_results.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    print(f"Three-site pilot complete. Outputs written to {DATA_DIR} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
