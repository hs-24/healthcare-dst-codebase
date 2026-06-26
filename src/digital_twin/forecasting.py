"""
forecasting.py

Addresses the "Forecasting & Reporting" problem from Section 1:
the current process uses manual, generalised techniques (e.g. simple
ETS-style averaging) on fragmented data, producing unreliable
forecasts. This module computes two forecasts for comparison:

  1. "Manual baseline"  - a naive moving-average forecast computed on
     the FRAGMENTED (messy spreadsheet) dataset, approximating what
     today's manual Excel process would produce.
  2. "Digital twin"     - the same moving-average forecast computed on
     the CLEAN, centralised dataset the digital twin maintains.

Comparing forecast error (MAE / MAPE / WMAPE) between the two
isolates how much of today's forecasting problem is a *data quality*
problem rather than a *forecasting method* problem -- both forecasts
use an identical, simple method, so any accuracy gap is attributable
to data centralisation alone. This is the central, defensible claim
for the pilot's SMART objective.
"""

import numpy as np
import pandas as pd


def _moving_average_forecast(series: pd.Series, window: int = 7) -> pd.Series:
    """Simple trailing moving-average forecast, shifted by one day
    so a given day's forecast never uses that same day's actual value."""
    return series.rolling(window=window, min_periods=1).mean().shift(1)


def forecast_accuracy_comparison(clean_df: pd.DataFrame,
                                  fragmented_df: pd.DataFrame,
                                  window: int = 7) -> pd.DataFrame:
    """
    Compute per-SKU forecast error metrics (MAE, MAPE, WMAPE) for both
    the manual-baseline (fragmented) and digital-twin (clean) data.

    Forecast target: daily `demand`. Note the fragmented dataset only
    has corrupted `stock_level`, not `demand`, since spreadsheets in
    the real process record stock counts, not demand directly -- so
    the manual baseline's demand proxy is the *day-over-day drop* in
    reported stock, which is itself noisier because of corrupted
    counts. This mirrors how manual processes infer usage indirectly.
    """
    results = []

    for sku_id, clean_sku in clean_df.groupby("sku_id"):
        clean_sku = clean_sku.sort_values("day").reset_index(drop=True)

        # --- Digital twin forecast: direct demand, clean data ---
        clean_sku["forecast"] = _moving_average_forecast(clean_sku["demand"], window)
        dt_eval = clean_sku.dropna(subset=["forecast"])
        dt_mae = float(np.mean(np.abs(dt_eval["demand"] - dt_eval["forecast"])))
        dt_mape = float(np.mean(
            np.abs(dt_eval["demand"] - dt_eval["forecast"]) /
            dt_eval["demand"].replace(0, np.nan)
        ) * 100)
        dt_wmape = float(
            np.sum(np.abs(dt_eval["demand"] - dt_eval["forecast"])) /
            np.sum(dt_eval["demand"]) * 100
        )

        # --- Manual baseline: proxy demand from fragmented stock counts ---
        # Group by true_day (the day the stock count actually reflects),
        # not the lagged reporting label, so the proxy-demand series is
        # built on a correctly time-ordered sequence of real stock counts.
        frag_sku = fragmented_df[fragmented_df["sku_id"] == sku_id].copy()
        frag_sku = (
            frag_sku.groupby("true_day", as_index=False)["stock_level"]
            .mean()  # collapse duplicate spreadsheet rows by averaging
            .rename(columns={"true_day": "day"})
            .sort_values("day")
        )
        frag_sku["stock_level"] = frag_sku["stock_level"].interpolate(
            limit_direction="both"
        )  # fill missing entries the way a human would (interpolate gaps)
        frag_sku["proxy_demand"] = (-frag_sku["stock_level"].diff()).clip(lower=0)
        frag_sku["forecast"] = _moving_average_forecast(frag_sku["proxy_demand"], window)

        # Align proxy demand against true demand for error scoring
        merged = frag_sku.merge(
            clean_sku[["day", "demand"]], on="day", how="inner"
        ).dropna(subset=["forecast"])

        if len(merged) > 0:
            mb_mae = float(np.mean(np.abs(merged["demand"] - merged["forecast"])))
            mb_mape = float(np.mean(
                np.abs(merged["demand"] - merged["forecast"]) /
                merged["demand"].replace(0, np.nan)
            ) * 100)
            mb_wmape = float(
                np.sum(np.abs(merged["demand"] - merged["forecast"])) /
                np.sum(merged["demand"]) * 100
            )
        else:
            mb_mae = mb_mape = mb_wmape = np.nan

        results.append({
            "sku_id": sku_id,
            "sku_name": clean_sku["sku_name"].iloc[0],
            "manual_baseline_MAE": round(mb_mae, 2),
            "digital_twin_MAE": round(dt_mae, 2),
            "manual_baseline_MAPE_pct": round(mb_mape, 2),
            "digital_twin_MAPE_pct": round(dt_mape, 2),
            "manual_baseline_WMAPE_pct": round(mb_wmape, 2),
            "digital_twin_WMAPE_pct": round(dt_wmape, 2),
            # Raw totals retained so the overall summary can compute a
            # demand-weighted average across SKUs, rather than an
            # unweighted mean of each SKU's own WMAPE (which lets a
            # single low-volume, erratic SKU dominate the headline figure).
            "manual_baseline_abs_error_total": float(
                np.sum(np.abs(merged["demand"] - merged["forecast"])) if len(merged) > 0 else np.nan
            ),
            "manual_baseline_demand_total": float(merged["demand"].sum()) if len(merged) > 0 else np.nan,
            "digital_twin_abs_error_total": float(np.sum(np.abs(dt_eval["demand"] - dt_eval["forecast"]))),
            "digital_twin_demand_total": float(dt_eval["demand"].sum()),
        })

    return pd.DataFrame(results)


def summarize_accuracy_improvement(comparison_df: pd.DataFrame) -> dict:
    """
    Roll up the per-SKU comparison into a single headline figure:
    the % improvement in forecast accuracy (via WMAPE reduction)
    achieved by the digital twin vs the manual baseline.

    The overall WMAPE is demand-weighted across SKUs (total absolute
    error / total demand), not an unweighted average of each SKU's own
    WMAPE. This matters because low-volume, erratic (XYZ "Z"-class)
    SKUs naturally produce noisy, sometimes very high WMAPE values on
    small demand bases; an unweighted mean would let one such SKU
    dominate the headline figure even though it represents a small
    share of total warehouse activity.
    """
    mb_total_error = comparison_df["manual_baseline_abs_error_total"].sum()
    mb_total_demand = comparison_df["manual_baseline_demand_total"].sum()
    dt_total_error = comparison_df["digital_twin_abs_error_total"].sum()
    dt_total_demand = comparison_df["digital_twin_demand_total"].sum()

    mb_wmape = (mb_total_error / mb_total_demand) * 100 if mb_total_demand else np.nan
    dt_wmape = (dt_total_error / dt_total_demand) * 100 if dt_total_demand else np.nan
    improvement_pct = ((mb_wmape - dt_wmape) / mb_wmape) * 100 if mb_wmape else np.nan

    return {
        "avg_manual_baseline_WMAPE_pct": round(mb_wmape, 2),
        "avg_digital_twin_WMAPE_pct": round(dt_wmape, 2),
        "forecast_accuracy_improvement_pct": round(improvement_pct, 2),
    }
