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

  3. "Clean-stock control" - the SAME stock-count-derived proxy method
     as (1), but run on the CLEAN, uncorrupted stock series.

All three arms use an identical moving-average method, so method is
held constant throughout. The third arm exists because arms (1) and
(2) do not face an identical *input signal*: the digital twin reads a
demand series directly, whereas any spreadsheet process can only infer
demand from periodic stock counts. That inference is structurally
lossy regardless of data quality (see `_stock_proxy_error`), so
comparing (1) against (2) alone would attribute the whole accuracy gap
to data corruption when most of it is caused by the indirect
observation of demand.

Adding the control arm decomposes the total gap into:
  * a DATA-QUALITY effect  (arm 3 vs arm 2) - what cleaning the
    spreadsheet's errors, duplicates and blanks is actually worth; and
  * an OBSERVABILITY effect (arm 1 vs arm 3) - what capturing demand at
    the point of consumption is worth, over inferring it from stock.

This decomposition, not the headline gap on its own, is the defensible
claim for the pilot's SMART objective.
"""

import numpy as np
import pandas as pd


def _moving_average_forecast(series: pd.Series, window: int = 7) -> pd.Series:
    """Simple trailing moving-average forecast, shifted by one day
    so a given day's forecast never uses that same day's actual value."""
    return series.rolling(window=window, min_periods=1).mean().shift(1)


def _stock_proxy_error(stock_frame: pd.DataFrame, clean_sku: pd.DataFrame,
                        window: int = 7) -> dict:
    """
    Reconstruct daily demand from a series of periodic stock counts, forecast
    it with the same trailing moving average, and score it against true demand.

    A stock-count series can only reveal demand as the day-over-day *drop* in
    recorded stock. That reconstruction is structurally lossy in two ways that
    have nothing to do with data quality:

      * on any day a replenishment arrives, stock rises, so the clipped
        difference reads zero demand; and
      * throughout a stockout, stock is pinned at zero while real demand
        continues, so the proxy again reads zero.

    Both blind spots are present even when every stock count is perfectly
    accurate. Running this same function on the clean stock series therefore
    gives the control needed to separate this observability limit from the
    effect of the spreadsheet corruption itself.
    """
    frame = stock_frame.sort_values("day").copy()
    frame["stock_level"] = frame["stock_level"].interpolate(limit_direction="both")
    frame["proxy_demand"] = (-frame["stock_level"].diff()).clip(lower=0)
    frame["forecast"] = _moving_average_forecast(frame["proxy_demand"], window)

    merged = frame.merge(
        clean_sku[["day", "demand"]], on="day", how="inner"
    ).dropna(subset=["forecast"])

    if len(merged) == 0:
        return {"abs_error": np.nan, "demand_total": np.nan,
                "wmape": np.nan, "mae": np.nan, "mape": np.nan}

    abs_error = float(np.sum(np.abs(merged["demand"] - merged["forecast"])))
    demand_total = float(merged["demand"].sum())
    return {
        "abs_error": abs_error,
        "demand_total": demand_total,
        "wmape": abs_error / demand_total * 100 if demand_total else np.nan,
        "mae": float(np.mean(np.abs(merged["demand"] - merged["forecast"]))),
        "mape": float(np.mean(
            np.abs(merged["demand"] - merged["forecast"]) /
            merged["demand"].replace(0, np.nan)
        ) * 100),
    }


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
        )
        manual = _stock_proxy_error(frag_sku, clean_sku, window)
        mb_mae, mb_mape, mb_wmape = manual["mae"], manual["mape"], manual["wmape"]

        # --- Clean-stock control: identical proxy method, uncorrupted stock ---
        # Everything the manual baseline suffers from EXCEPT the corruption.
        # The gap between this arm and the manual baseline is the true cost of
        # fragmented data; the gap between this arm and the digital twin is the
        # cost of never observing demand directly.
        control = _stock_proxy_error(
            clean_sku[["day", "stock_level"]].copy(), clean_sku, window
        )

        results.append({
            "sku_id": sku_id,
            "sku_name": clean_sku["sku_name"].iloc[0],
            "manual_baseline_MAE": round(mb_mae, 2),
            "digital_twin_MAE": round(dt_mae, 2),
            "manual_baseline_MAPE_pct": round(mb_mape, 2),
            "digital_twin_MAPE_pct": round(dt_mape, 2),
            "manual_baseline_WMAPE_pct": round(mb_wmape, 2),
            "digital_twin_WMAPE_pct": round(dt_wmape, 2),
            "clean_stock_proxy_WMAPE_pct": round(control["wmape"], 2),
            # Raw totals retained so the overall summary can compute a
            # demand-weighted average across SKUs, rather than an
            # unweighted mean of each SKU's own WMAPE (which lets a
            # single low-volume, erratic SKU dominate the headline figure).
            "manual_baseline_abs_error_total": manual["abs_error"],
            "manual_baseline_demand_total": manual["demand_total"],
            "digital_twin_abs_error_total": float(np.sum(np.abs(dt_eval["demand"] - dt_eval["forecast"]))),
            "digital_twin_demand_total": float(dt_eval["demand"].sum()),
            "clean_stock_proxy_abs_error_total": control["abs_error"],
            "clean_stock_proxy_demand_total": control["demand_total"],
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
    cs_total_error = comparison_df["clean_stock_proxy_abs_error_total"].sum()
    cs_total_demand = comparison_df["clean_stock_proxy_demand_total"].sum()

    mb_wmape = (mb_total_error / mb_total_demand) * 100 if mb_total_demand else np.nan
    dt_wmape = (dt_total_error / dt_total_demand) * 100 if dt_total_demand else np.nan
    cs_wmape = (cs_total_error / cs_total_demand) * 100 if cs_total_demand else np.nan
    improvement_pct = ((mb_wmape - dt_wmape) / mb_wmape) * 100 if mb_wmape else np.nan

    # Decompose the headline improvement into its two mechanisms.
    #   data-quality effect : manual baseline -> clean-stock control
    #   observability effect: clean-stock control -> digital twin
    data_quality_effect = ((mb_wmape - cs_wmape) / mb_wmape) * 100 if mb_wmape else np.nan
    observability_effect = ((cs_wmape - dt_wmape) / cs_wmape) * 100 if cs_wmape else np.nan
    total_gap = mb_wmape - dt_wmape
    share_from_data_quality = (
        ((mb_wmape - cs_wmape) / total_gap) * 100 if total_gap else np.nan
    )

    return {
        "avg_manual_baseline_WMAPE_pct": round(mb_wmape, 2),
        "avg_digital_twin_WMAPE_pct": round(dt_wmape, 2),
        "forecast_accuracy_improvement_pct": round(improvement_pct, 2),
        # --- mechanism decomposition (see module docstring) ---
        "avg_clean_stock_proxy_WMAPE_pct": round(cs_wmape, 2),
        "data_quality_effect_pct": round(data_quality_effect, 2),
        "observability_effect_pct": round(observability_effect, 2),
        "share_of_improvement_from_data_quality_pct": round(share_from_data_quality, 2),
    }
