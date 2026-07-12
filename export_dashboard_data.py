"""Export three-site simulation outputs for the live dashboard."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def _records_for_site(site_id, clean, manual, fragmented, forecast, catalog, results):
    clean_site = clean[clean["site_id"] == site_id]
    manual_site = manual[manual["site_id"] == site_id]
    frag_site = fragmented[fragmented["site_id"] == site_id]
    forecast_site = forecast[forecast["site_id"] == site_id]
    catalog_site = catalog[catalog["site_id"] == site_id]
    metadata = catalog_site.set_index("sku_id").to_dict(orient="index")

    dt_daily = clean_site.groupby("day")["stock_level"].sum().round(1)
    mb_daily = manual_site.groupby("day")["true_stock_level"].sum().round(1)
    perceived = manual_site.groupby("day")["perceived_stock_level"].sum().round(1)
    per_sku = {}
    alerts = []
    today = date.today()

    for sku_id in sorted(clean_site["sku_id"].unique()):
        c = clean_site[clean_site["sku_id"] == sku_id].sort_values("day")
        m = manual_site[manual_site["sku_id"] == sku_id].sort_values("day")
        meta = metadata[sku_id]
        latest_stock = float(c.iloc[-1]["stock_level"])
        reorder_point = float(c.iloc[-1]["reorder_point"])
        expiry_text = str(meta.get("expiry_date") or "")
        days_to_expiry = None
        if expiry_text and expiry_text != "nan":
            days_to_expiry = (date.fromisoformat(expiry_text) - today).days

        per_sku[sku_id] = {
            "sku_name": c.iloc[0]["sku_name"],
            "abc_class": c.iloc[0]["abc_class"],
            "xyz_class": c.iloc[0]["xyz_class"],
            "category": meta["item_type"],
            "uom": meta["uom"],
            "storage_mode": meta["storage_mode"],
            "batch_number": meta["batch_number"],
            "expiry_date": expiry_text if expiry_text != "nan" else "",
            "days_to_expiry": days_to_expiry,
            "vendor_id": meta["vendor_id"],
            "unit_cost": float(meta["unit_cost"]),
            "max_capacity": float(meta["max_capacity"]),
            "units_per_pallet": max(1, round(float(meta["max_capacity"]) / 12)),
            "source_dataset": meta["source_dataset"],
            "derived_fields": meta["derived_fields"],
            "quality_note": str(meta.get("quality_note") or ""),
            "days": c["day"].tolist(),
            "digital_twin_stock": c["stock_level"].round(1).tolist(),
            "manual_baseline_true_stock": m["true_stock_level"].round(1).tolist(),
            "manual_baseline_perceived_stock": m["perceived_stock_level"].round(1).tolist(),
            "reorder_point": reorder_point,
            "demand": c["demand"].round(2).tolist(),
        }

        if latest_stock <= reorder_point:
            alerts.append({"severity": "critical", "type": "Low stock", "sku_id": sku_id, "message": f"{sku_id} is at or below minimum stock."})
        elif latest_stock <= reorder_point * 1.35:
            alerts.append({"severity": "warning", "type": "Stock watch", "sku_id": sku_id, "message": f"{sku_id} is approaching minimum stock."})
        if days_to_expiry is not None and days_to_expiry <= 90:
            severity = "critical" if days_to_expiry <= 30 else "warning"
            alerts.append({"severity": severity, "type": "Expiry", "sku_id": sku_id, "message": f"Batch {meta['batch_number']} expires in {days_to_expiry} days."})
        if meta.get("quality_note"):
            alerts.append({"severity": "info", "type": "Data validation", "sku_id": sku_id, "message": meta["quality_note"]})

    site_result = results["sites"][site_id]
    latest_total = float(dt_daily.iloc[-1])
    capacity = float(catalog_site["max_capacity"].sum())
    summary = dict(site_result["operational_comparison"])
    summary.update(site_result["forecast_accuracy_summary"])
    summary.update({
        "data_inconsistency_rate_pct": site_result["data_quality_metrics"]["inconsistency_rate_pct"],
        "critical_skus": sum(1 for alert in alerts if alert["severity"] == "critical"),
        "capacity_utilization_pct": round(latest_total / capacity * 100, 1) if capacity else 0,
        "current_stock": round(latest_total, 1),
        "max_capacity": round(capacity, 1),
    })

    frag_columns = ["day", "true_day", "site_id", "sku_id", "stock_level", "source_sheet"]
    return {
        "site_id": site_id,
        "site_name": catalog_site.iloc[0]["site_name"],
        "summary": summary,
        "daily_stock": {
            "days": dt_daily.index.tolist(),
            "digital_twin_total_stock": dt_daily.values.tolist(),
            "manual_baseline_true_stock": mb_daily.values.tolist(),
            "manual_baseline_perceived_stock": perceived.values.tolist(),
        },
        "per_sku_series": per_sku,
        "alerts": alerts,
        "fragmented_sample": frag_site.sort_values(["day", "sku_id"]).head(25)[frag_columns].fillna("-").to_dict(orient="records"),
        "forecast_table": forecast_site.fillna("").to_dict(orient="records"),
    }


def _transfer_recommendations(sites):
    recommendations = []
    sku_ids = sorted({sku for site in sites.values() for sku in site["per_sku_series"]})
    for sku_id in sku_ids:
        positions = []
        for site_id, site in sites.items():
            sku = site["per_sku_series"].get(sku_id)
            if not sku:
                continue
            stock = sku["digital_twin_stock"][-1]
            positions.append((site_id, site["site_name"], stock, sku["reorder_point"], sku))
        deficits = sorted((p for p in positions if p[2] < p[3]), key=lambda p: p[2] - p[3])
        surpluses = sorted((p for p in positions if p[2] > p[3] * 2), key=lambda p: p[2] - p[3], reverse=True)
        if deficits and surpluses:
            destination, source = deficits[0], surpluses[0]
            qty = min(destination[3] * 1.5 - destination[2], source[2] - source[3] * 1.5)
            if qty > 0:
                recommendations.append({
                    "sku_id": sku_id,
                    "sku_name": destination[4]["sku_name"],
                    "source_site": source[0], "source_name": source[1],
                    "destination_site": destination[0], "destination_name": destination[1],
                    "quantity": round(qty, 1), "uom": destination[4]["uom"],
                    "reason": "Destination below minimum while source retains safety surplus",
                })
    return recommendations


def main():
    clean = pd.read_csv(DATA_DIR / "clean_ground_truth.csv")
    manual = pd.read_csv(DATA_DIR / "manual_baseline_simulation.csv")
    fragmented = pd.read_csv(DATA_DIR / "fragmented_spreadsheet_sim.csv")
    forecast = pd.read_csv(OUTPUT_DIR / "forecast_accuracy_comparison.csv")
    catalog = pd.read_csv(DATA_DIR / "processed" / "hospital_inventory_normalized.csv", keep_default_na=False)
    with (OUTPUT_DIR / "pilot_results.json").open(encoding="utf-8") as handle:
        results = json.load(handle)

    sites = {
        site_id: _records_for_site(site_id, clean, manual, fragmented, forecast, catalog, results)
        for site_id in sorted(catalog["site_id"].unique())
    }
    dashboard_data = {
        "source": {
            "name": "Hospital Supply Chain",
            "publisher": "Kaggle user vanpatangan",
            "url": "https://www.kaggle.com/datasets/vanpatangan/hospital-supply-chain",
            "note": "Public dataset fields are enriched for the three-site proof of concept; derived fields are identified per SKU.",
        },
        "network_summary": results["network_summary"],
        "site_list": [{"site_id": key, "site_name": value["site_name"]} for key, value in sites.items()],
        "sites": sites,
        "transfers": _transfer_recommendations(sites),
    }
    out_path = OUTPUT_DIR / "dashboard_data.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(dashboard_data, handle)
    print(f"Dashboard data written to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
