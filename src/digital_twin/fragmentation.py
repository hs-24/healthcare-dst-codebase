"""
fragmentation.py

Simulates the "fragmented spreadsheet" problem described in Section 1
of the project report: the digital twin tracks ground-truth inventory
state perfectly (as a real centralised system would), and this module
deliberately corrupts a copy of that data to recreate what the current
manual, multi-source Excel process actually produces.

This corrupted dataset is what "Method 1 - Before centralisation"
represents in the dashboard, while the clean digital twin output
represents "Method 2 - After centralisation", letting the report make
a direct, quantified before/after comparison.
"""

import numpy as np
import pandas as pd


def fragment_dataset(clean_df: pd.DataFrame, data_quality_cfg: dict,
                      rng: np.random.Generator) -> pd.DataFrame:
    """
    Take the digital twin's clean daily stock-level records and
    produce a corrupted version simulating manual multi-spreadsheet
    upkeep: duplicated rows, missing fields, and transcription errors.

    Parameters
    ----------
    clean_df : DataFrame with columns
        [day, site_id, sku_id, stock_level, demand, fulfilled, stockout]
    data_quality_cfg : dict from warehouse_config.yaml -> data_quality
    rng : numpy random Generator for reproducibility

    Returns
    -------
    DataFrame in the same shape, but with realistic data-quality issues
    injected, plus a `source_sheet` column to mimic multiple Excel files.
    """
    df = clean_df.copy()
    n = len(df)

    # Keep track of which real-world day each row's stock_level actually
    # corresponds to, BEFORE we shift the reported `day` label. This lets
    # later reconciliation compare reported figures against the correct
    # ground-truth value, instead of against whatever happens to share
    # the post-lag day number (which would wrongly flag almost every row
    # as inconsistent simply because the calendar label moved).
    df["true_day"] = df["day"]

    # --- 1. Reporting lag: spreadsheet figures lag real events ---
    # The sheet is *labelled* with a later day than the data really
    # reflects (i.e. staff log "as of" a date after they actually
    # counted stock), so only the reporting label shifts.
    lag_days = data_quality_cfg.get("reporting_lag_days", 1)
    df["day"] = df["day"] + lag_days

    # --- 2. Assign each row to one of several "source spreadsheets" ---
    # Recreates the observed issue of data spread across multiple files
    # with no centralised system in place.
    sheet_names = ["Inventory_MasterA.xlsx", "Inventory_MasterB.xlsx",
                   "WeeklyStockCount.xlsx", "ManualAdjustments.xlsx"]
    df["source_sheet"] = rng.choice(sheet_names, size=n)

    # --- 3. Manual transcription errors on stock_level ---
    error_rate = data_quality_cfg.get("manual_entry_error_rate", 0.05)
    error_mask = rng.random(n) < error_rate
    # Errors simulate digit transposition / fat-finger entry: scale by
    # a random factor between 0.5x and 1.6x of the true value
    error_factors = rng.uniform(0.5, 1.6, size=n)
    df.loc[error_mask, "stock_level"] = (
        df.loc[error_mask, "stock_level"] * error_factors[error_mask]
    ).round()

    # --- 4. Missing entries (blank cells in the spreadsheet) ---
    missing_rate = data_quality_cfg.get("missing_entry_rate", 0.04)
    missing_mask = rng.random(n) < missing_rate
    df.loc[missing_mask, "stock_level"] = np.nan

    # --- 5. Duplicate rows (same record re-entered in another sheet) ---
    dup_rate = data_quality_cfg.get("duplicate_entry_rate", 0.05)
    dup_mask = rng.random(n) < dup_rate
    duplicates = df[dup_mask].copy()
    if len(duplicates) > 0:
        # Duplicated rows often get re-entered with a *different* source
        # sheet and sometimes a slightly different (re-keyed) value
        dup_rng_factors = rng.uniform(0.95, 1.05, size=len(duplicates))
        duplicates["stock_level"] = (duplicates["stock_level"] * dup_rng_factors).round()
        duplicates["source_sheet"] = rng.choice(sheet_names, size=len(duplicates))
        df = pd.concat([df, duplicates], ignore_index=True)

    return df


def compute_data_quality_metrics(clean_df: pd.DataFrame,
                                  fragmented_df: pd.DataFrame) -> dict:
    """
    Quantify the "before centralisation" data inconsistency rate by
    comparing the fragmented dataset back against ground truth.

    This produces the evidence behind the SMART objective claim:
    "reduce data inconsistency errors by at least 10% (pilot scope)".
    """
    merged = fragmented_df.merge(
        clean_df[["day", "site_id", "sku_id", "stock_level"]],
        left_on=["true_day", "site_id", "sku_id"],
        right_on=["day", "site_id", "sku_id"],
        suffixes=("_reported", "_true"),
        how="left",
    )

    merged["is_missing"] = merged["stock_level_reported"].isna()
    # Use a small tolerance since legitimate floating-point stock levels
    # won't match bit-for-bit even when correct; only flag as inconsistent
    # if the reported figure differs from truth by a meaningful amount.
    merged["is_inconsistent"] = (
        (~merged["is_missing"]) &
        (np.abs(merged["stock_level_reported"] - merged["stock_level_true"]) > 0.5)
    )

    total_records = len(merged)
    inconsistent_records = int(merged["is_inconsistent"].sum())
    missing_records = int(merged["is_missing"].sum())
    duplicate_records = int(fragmented_df.duplicated(
        subset=["day", "site_id", "sku_id"], keep=False
    ).sum())

    inconsistency_rate = (inconsistent_records + missing_records) / total_records

    return {
        "total_records": total_records,
        "inconsistent_records": inconsistent_records,
        "missing_records": missing_records,
        "duplicate_records": duplicate_records,
        "inconsistency_rate_pct": round(inconsistency_rate * 100, 2),
    }
