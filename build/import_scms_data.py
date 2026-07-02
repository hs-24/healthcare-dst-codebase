"""
Build a real-data warehouse config from the SCMS Delivery History dataset.

The SCMS file is a delivery history dataset, not a live stock-on-hand file.
This importer uses its real delivered quantities, dates, countries, product
groups, and item descriptions to parameterise demand. Warehouse-only fields
that are not present in SCMS, such as storage zone, batch number, and expiry,
are generated deterministically and documented in the output metadata.

Outputs:
  config/warehouse_config_scms.yaml
  data/scms_sku_master.csv
  data/scms_demand_profile.csv
  outputs/scms_import_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "raw" / "SCMS_Delivery_History_Dataset.csv"
DEFAULT_CONFIG_OUT = ROOT / "config" / "warehouse_config_scms.yaml"
DEFAULT_MASTER_OUT = ROOT / "data" / "scms_sku_master.csv"
DEFAULT_PROFILE_OUT = ROOT / "data" / "scms_demand_profile.csv"
DEFAULT_SUMMARY_OUT = ROOT / "outputs" / "scms_import_summary.json"


REQUIRED_COLUMNS = [
    "ID",
    "Country",
    "Project Code",
    "Delivered to Client Date",
    "Product Group",
    "Sub Classification",
    "Item Description",
    "Dosage Form",
    "Unit of Measure (Per Pack)",
    "Line Item Quantity",
    "Line Item Value",
]


def clean_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def safe_slug(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", str(value).upper()).strip("-")
    return text[:32] or fallback


def infer_storage_mode(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(col, ""))
        for col in ["Item Description", "Product Group", "Sub Classification", "Dosage Form"]
    ).lower()
    if any(term in text for term in ["cold", "refriger", "vaccine", "insulin"]):
        return "COLD_VAULT"
    return "VNA"


def infer_units_per_pallet(dosage_form: str, daily_mean: float) -> int:
    text = str(dosage_form).lower()
    if "tablet" in text or "capsule" in text:
        return max(100, int(round(daily_mean * 10 / 50) * 50))
    if "test" in text or "kit" in text:
        return 40
    if "bottle" in text or "suspension" in text or "solution" in text:
        return 60
    return 80


def shelf_life_days(row: pd.Series) -> int:
    text = " ".join(
        str(row.get(col, ""))
        for col in ["Item Description", "Product Group", "Sub Classification", "Dosage Form"]
    ).lower()
    if any(term in text for term in ["cold", "refriger", "vaccine", "insulin"]):
        return 365
    if "hiv test" in text or "test kit" in text:
        return 540
    return 730


def xyz_class(monthly_values: pd.Series) -> str:
    mean = monthly_values.mean()
    if mean <= 0 or len(monthly_values) < 2:
        return "Y"
    cv = monthly_values.std(ddof=0) / mean
    if cv <= 0.35:
        return "X"
    if cv <= 0.75:
        return "Y"
    return "Z"


def lead_time_days(rows: pd.DataFrame) -> int:
    po_date = pd.to_datetime(rows.get("PO Sent to Vendor Date"), errors="coerce", dayfirst=True)
    delivered = pd.to_datetime(rows.get("Delivered to Client Date"), errors="coerce", dayfirst=True)
    lead_times = (delivered - po_date).dt.days
    lead_times = lead_times[(lead_times > 0) & (lead_times < 180)]
    if lead_times.empty:
        return 7
    return int(max(3, min(14, round(float(lead_times.median())))))


def build_config(source: Path, top_n: int) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(source, low_memory=False)
    df.columns = [str(col).strip() for col in df.columns]

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"SCMS file is missing required columns: {missing}")

    df["Line Item Quantity"] = clean_number(df["Line Item Quantity"])
    df["Line Item Value"] = clean_number(df["Line Item Value"])
    df["Delivered to Client Date"] = pd.to_datetime(
        df["Delivered to Client Date"], errors="coerce", dayfirst=True
    )
    df = df.dropna(subset=["Delivered to Client Date", "Item Description", "Line Item Quantity"])
    df = df[df["Line Item Quantity"] > 0].copy()
    df["month"] = df["Delivered to Client Date"].dt.to_period("M").astype(str)
    df["sku_key"] = (
        df["Product Group"].astype(str)
        + "|"
        + df["Sub Classification"].astype(str)
        + "|"
        + df["Item Description"].astype(str)
        + "|"
        + df["Unit of Measure (Per Pack)"].astype(str)
    )

    totals = (
        df.groupby("sku_key")
        .agg(
            item_description=("Item Description", "first"),
            product_group=("Product Group", "first"),
            sub_classification=("Sub Classification", "first"),
            dosage_form=("Dosage Form", "first"),
            uom=("Unit of Measure (Per Pack)", "first"),
            total_quantity=("Line Item Quantity", "sum"),
            total_value=("Line Item Value", "sum"),
            first_delivery=("Delivered to Client Date", "min"),
            latest_delivery=("Delivered to Client Date", "max"),
            shipment_count=("ID", "count"),
            country=("Country", lambda s: s.mode().iloc[0] if not s.mode().empty else "SCMS"),
            site_code=("Project Code", lambda s: s.mode().iloc[0] if not s.mode().empty else "SCMS_SITE"),
        )
        .sort_values(["total_value", "total_quantity"], ascending=False)
    )

    totals["value_rank"] = range(1, len(totals) + 1)
    cumulative = totals["total_value"].fillna(0).cumsum()
    total_value = max(float(totals["total_value"].fillna(0).sum()), 1.0)
    totals["abc_class"] = cumulative.div(total_value).apply(
        lambda pct: "A" if pct <= 0.80 else ("B" if pct <= 0.95 else "C")
    )

    selected = totals.head(top_n).copy()
    monthly = (
        df[df["sku_key"].isin(selected.index)]
        .groupby(["sku_key", "month"])["Line Item Quantity"]
        .sum()
        .reset_index()
    )

    skus = []
    master_rows = []
    profile_rows = []

    for idx, (sku_key, row) in enumerate(selected.iterrows(), start=1):
        sku_id = f"SCMS-{idx:03d}"
        sku_monthly = monthly[monthly["sku_key"] == sku_key].sort_values("month")
        monthly_qty = sku_monthly["Line Item Quantity"]
        avg_monthly = float(monthly_qty.mean())
        std_monthly = float(monthly_qty.std(ddof=0)) if len(monthly_qty) > 1 else avg_monthly * 0.25
        demand_mean = max(0.1, avg_monthly / 30.0)
        demand_std = max(0.1, std_monthly / 30.0, demand_mean * 0.10)
        lead_time = lead_time_days(df[df["sku_key"] == sku_key])
        reorder_point = math.ceil((demand_mean * lead_time) + (1.65 * demand_std * math.sqrt(lead_time)))
        reorder_qty = math.ceil(demand_mean * 14)
        initial_stock = math.ceil(max(demand_mean * 21, reorder_point + reorder_qty * 0.5))
        storage_mode = infer_storage_mode(row)
        latest_delivery = row["latest_delivery"]
        expiry_date = (latest_delivery + pd.Timedelta(days=shelf_life_days(row))).date().isoformat()
        batch_number = f"{sku_id}-{safe_slug(row['product_group'], 'SCMS')}-B001"

        skus.append(
            {
                "sku_id": sku_id,
                "source_sku_key": sku_key,
                "name": str(row["item_description"]),
                "category": f"{row['product_group']} / {row['sub_classification']}",
                "abc_class": str(row["abc_class"]),
                "xyz_class": xyz_class(monthly_qty),
                "storage_mode": storage_mode,
                "units_per_pallet": infer_units_per_pallet(row["dosage_form"], demand_mean),
                "initial_stock": float(initial_stock),
                "reorder_point": float(reorder_point),
                "reorder_qty": float(reorder_qty),
                "lead_time_days": int(lead_time),
                "demand_mean": round(float(demand_mean), 2),
                "demand_std": round(float(demand_std), 2),
            }
        )

        master_rows.append(
            {
                "sku_code": sku_id,
                "source_sku_key": sku_key,
                "sku_description": row["item_description"],
                "uom": row["uom"],
                "quantity": round(float(row["total_quantity"]), 2),
                "storage_zone": "cold chain" if storage_mode == "COLD_VAULT" else "warehouse item",
                "expiry_date": expiry_date,
                "batch_number": batch_number,
                "site_code": row["site_code"],
                "site_country": row["country"],
                "field_note": "Quantity/site/product fields are SCMS-derived; storage zone, expiry date, and batch number are deterministic PoC warehouse attributes.",
            }
        )

        for _, month_row in sku_monthly.iterrows():
            profile_rows.append(
                {
                    "sku_code": sku_id,
                    "month": month_row["month"],
                    "quantity": round(float(month_row["Line Item Quantity"]), 2),
                }
            )

    cfg = {
        "simulation": {
            "name": "Healthcare Warehouse PoC - SCMS Real-Data Demand Pilot",
            "duration_days": 30,
            "random_seed": 42,
            "time_step_hours": 24,
        },
        "sites": [
            {
                "site_id": "SCMS_SITE_A",
                "site_name": "SCMS Real-Data Healthcare Distribution Pilot",
                "description": (
                    "SKU demand parameters are derived from the SCMS Delivery History "
                    "dataset. Inventory policy, storage zoning, expiry, and batch fields "
                    "are modelled for this digital twin proof-of-concept."
                ),
            }
        ],
        "skus": skus,
        "data_quality": {
            "duplicate_entry_rate": 0.05,
            "missing_entry_rate": 0.04,
            "manual_entry_error_rate": 0.06,
            "reporting_lag_days": 1,
        },
        "real_data_source": {
            "dataset": "SCMS_Delivery_History_Dataset.csv",
            "source_file": str(source),
            "real_fields_used": [
                "Project Code",
                "Country",
                "Delivered to Client Date",
                "Product Group",
                "Sub Classification",
                "Item Description",
                "Dosage Form",
                "Unit of Measure (Per Pack)",
                "Line Item Quantity",
                "Line Item Value",
            ],
            "simulated_fields": ["storage_mode", "expiry_date", "batch_number"],
        },
    }

    master = pd.DataFrame(master_rows)
    profile = pd.DataFrame(profile_rows)
    summary = {
        "source_file": str(source),
        "source_rows_used": int(len(df)),
        "source_date_min": df["Delivered to Client Date"].min().date().isoformat(),
        "source_date_max": df["Delivered to Client Date"].max().date().isoformat(),
        "unique_source_products": int(totals.shape[0]),
        "selected_skus": int(len(skus)),
        "selected_total_quantity": round(float(master["quantity"].sum()), 2),
        "note": (
            "SCMS is delivery history. It provides real product, site, date, "
            "quantity, and value fields. Expiry/batch/storage fields are generated "
            "as warehouse PoC attributes because they are not present in the source file."
        ),
    }
    return cfg, master, profile, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Import SCMS delivery data into a warehouse config.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Path to SCMS_Delivery_History_Dataset.csv")
    parser.add_argument("--top-n", type=int, default=16, help="Number of SKUs to select for the pilot")
    parser.add_argument("--config-out", default=str(DEFAULT_CONFIG_OUT), help="Generated YAML config path")
    parser.add_argument("--master-out", default=str(DEFAULT_MASTER_OUT), help="Generated SKU master CSV path")
    parser.add_argument("--profile-out", default=str(DEFAULT_PROFILE_OUT), help="Generated monthly demand profile CSV path")
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT), help="Generated import summary JSON path")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"SCMS source file not found: {source}")

    cfg, master, profile, summary = build_config(source, args.top_n)

    config_out = Path(args.config_out)
    master_out = Path(args.master_out)
    profile_out = Path(args.profile_out)
    summary_out = Path(args.summary_out)
    for path in [config_out, master_out, profile_out, summary_out]:
        path.parent.mkdir(parents=True, exist_ok=True)

    with config_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    master.to_csv(master_out, index=False)
    profile.to_csv(profile_out, index=False)
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"SCMS config written to {config_out}")
    print(f"SKU master written to {master_out}")
    print(f"Monthly demand profile written to {profile_out}")
    print(f"Import summary written to {summary_out}")
    print(f"Selected {summary['selected_skus']} SKUs from {summary['unique_source_products']} source products")


if __name__ == "__main__":
    main()
