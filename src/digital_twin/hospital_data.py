"""Normalize the public Kaggle Hospital Supply Chain inventory dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


SITE_DEFINITIONS = (
    ("SITE_A", "Central Distribution Hub", 1.00),
    ("SITE_B", "North Regional Hub", 0.72),
    ("SITE_C", "East Satellite Depot", 0.48),
)


def _stable_code(*parts: object, length: int = 10) -> str:
    value = "|".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length].upper()


def _uom(item_name: str) -> str:
    name = item_name.lower()
    if "mask" in name or "glove" in name:
        return "box"
    if "drip" in name:
        return "case"
    return "unit"


def _storage_zone(item_name: str, item_type: str) -> str:
    text = f"{item_name} {item_type}".lower()
    if "x-ray" in text or "ventilator" in text or "equipment" in text:
        return "SECURE_CAGE"
    return "VNA"


def normalize_inventory(raw_path: Path, output_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, parse_dates=["Date"])
    required = {
        "Date", "Item_ID", "Item_Type", "Item_Name", "Current_Stock",
        "Min_Required", "Max_Capacity", "Unit_Cost", "Avg_Usage_Per_Day",
        "Restock_Lead_Time", "Vendor_ID",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Hospital dataset is missing columns: {', '.join(missing)}")

    raw = raw.dropna(subset=["Item_ID", "Item_Name", "Current_Stock"]).copy()
    raw["Item_ID"] = raw["Item_ID"].astype(int)
    raw["site_index"] = raw.index % len(SITE_DEFINITIONS)

    records: list[dict] = []
    for site_index, (site_id, site_name, scale) in enumerate(SITE_DEFINITIONS):
        site_rows = raw[raw["site_index"] == site_index]
        for item_id, group in site_rows.groupby("Item_ID"):
            item_name = group["Item_Name"].mode().iloc[0]
            item_type = group["Item_Type"].mode().iloc[0]
            vendor_id = group["Vendor_ID"].mode().iloc[0]
            capacity = max(50.0, float(group["Max_Capacity"].median()) * scale)
            source_stock = max(0.0, float(group["Current_Stock"].median()) * scale)
            current_stock = min(source_stock, capacity)
            min_required = min(float(group["Min_Required"].median()) * scale, capacity * 0.8)
            demand_mean = max(1.0, float(group["Avg_Usage_Per_Day"].median()) * scale)
            demand_std = max(1.0, demand_mean * 0.18)
            lead_time = max(1, int(round(group["Restock_Lead_Time"].median())))
            reorder_qty = min(capacity, max(min_required, demand_mean * lead_time * 0.75))
            latest_date = group["Date"].max()
            uom = _uom(item_name)
            zone = _storage_zone(item_name, item_type)
            is_perishable = uom in {"box", "case"}
            expiry = latest_date + pd.Timedelta(days=540) if is_perishable else pd.NaT

            records.append({
                "site_id": site_id,
                "site_name": site_name,
                "sku_id": f"HSC-{int(item_id):03d}",
                "sku_name": item_name,
                "item_type": item_type,
                "uom": uom,
                "initial_stock": round(current_stock, 2),
                "source_stock_before_capacity_check": round(source_stock, 2),
                "reorder_point": round(min_required, 2),
                "max_capacity": round(capacity, 2),
                "reorder_qty": round(reorder_qty, 2),
                "lead_time_days": lead_time,
                "demand_mean": round(demand_mean, 2),
                "demand_std": round(demand_std, 2),
                "unit_cost": round(float(group["Unit_Cost"].median()), 2),
                "vendor_id": vendor_id,
                "storage_mode": zone,
                "batch_number": f"B-{site_id[-1]}-{item_id}-{_stable_code(site_id, item_id, latest_date.date(), length=6)}",
                "expiry_date": expiry.date().isoformat() if pd.notna(expiry) else "",
                "source_record_count": int(len(group)),
                "source_date_min": group["Date"].min().date().isoformat(),
                "source_date_max": latest_date.date().isoformat(),
                "source_dataset": "Kaggle Hospital Supply Chain / inventory_data.csv",
                "source_fields": "Item_ID,Item_Name,Item_Type,Current_Stock,Min_Required,Max_Capacity,Unit_Cost,Avg_Usage_Per_Day,Restock_Lead_Time,Vendor_ID,Date",
                "derived_fields": "site_id,uom,storage_mode,batch_number,expiry_date,demand_std,reorder_qty",
                "quality_note": "Initial stock capped at capacity" if source_stock > capacity else "",
            })

    normalized = pd.DataFrame(records).sort_values(["site_id", "sku_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False)
    return normalized


def load_or_build_catalog(root: Path) -> pd.DataFrame:
    raw_path = root / "data" / "raw" / "hospital-supply-chain" / "inventory_data.csv"
    output_path = root / "data" / "processed" / "hospital_inventory_normalized.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path}. Download vanpatangan/hospital-supply-chain from Kaggle."
        )
    return normalize_inventory(raw_path, output_path)
