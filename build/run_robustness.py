"""
run_robustness.py

Two experiments that stress-test the headline pilot result in
outputs/pilot_results.json. Neither changes the pilot itself: this script
writes its own output file (outputs/robustness_results.json) and leaves the
single-seed pilot the live dashboard replays completely untouched.

Experiment 1 - REPLICATION
    The pilot is a single random seed per site. This re-runs the identical
    paired design across N independent seeds and reports mean and 95%
    confidence interval for every headline KPI, answering the question
    "how do you know this is not just noise?".

Experiment 2 - SENSITIVITY
    Every fragmentation parameter (transcription-error, missing-entry and
    duplicate-entry rates, and the reporting lag) was assumed, not measured.
    This sweeps them across a range and reports how the measured benefit
    scales, answering the question "did you pick parameters that produce the
    answer you wanted?". A benefit that survives at half the assumed error
    rates is not an artefact of the assumption.

Both experiments call the same library functions as run_pilot.py, so they
cannot silently drift from the pilot's own methodology.

Run:  python build/run_robustness.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from digital_twin.fragmentation import (  # noqa: E402
    compute_data_quality_metrics,
    fragment_dataset,
)
from digital_twin.forecasting import (  # noqa: E402
    forecast_accuracy_comparison,
    summarize_accuracy_improvement,
)
from digital_twin.hospital_data import load_or_build_catalog  # noqa: E402
from digital_twin.simulation import ManualBaselineSimulation, WarehouseSimulation  # noqa: E402
from run_pilot import build_warehouse, load_config  # noqa: E402

OUTPUT_DIR = ROOT / "outputs"

N_REPLICATIONS = 20
SENSITIVITY_SEEDS = 5

# Two-sided 95% t critical values; falls back to the normal approximation
# for sample sizes outside the table.
_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 8: 2.365,
        10: 2.262, 12: 2.201, 15: 2.145, 20: 2.093, 25: 2.064, 30: 2.045}


def _t_crit(n: int) -> float:
    if n < 2:
        return float("nan")
    for size in sorted(_T95):
        if n <= size:
            return _T95[size]
    return 1.96


def summarise(values: list[float]) -> dict:
    """Mean, standard deviation and 95% CI for a list of replication results."""
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "ci95_low": None, "ci95_high": None}
    mean = float(arr.mean())
    if n == 1:
        return {"n": 1, "mean": round(mean, 3), "sd": None,
                "ci95_low": None, "ci95_high": None}
    sd = float(arr.std(ddof=1))
    half = _t_crit(n) * sd / np.sqrt(n)
    return {
        "n": n,
        "mean": round(mean, 3),
        "sd": round(sd, 3),
        "ci95_low": round(mean - half, 3),
        "ci95_high": round(mean + half, 3),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
    }


def run_one_site(site_rows: pd.DataFrame, cfg: dict, seed: int) -> dict:
    """One paired digital-twin / manual-baseline run for a single site."""
    sim_cfg, dq_cfg = cfg["simulation"], cfg["data_quality"]

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
    fragmented = fragment_dataset(clean, dq_cfg, np.random.default_rng(seed + 1000))
    forecast = summarize_accuracy_improvement(
        forecast_accuracy_comparison(clean, fragmented, window=7)
    )
    quality = compute_data_quality_metrics(clean, fragmented)

    kpi_dt, kpi_mb = digital.get_kpi_summary(), manual.get_kpi_summary()
    return {
        "fill_rate_gain_pct_points": kpi_dt["site_fill_rate_pct"] - kpi_mb["site_fill_rate_pct"],
        "stockout_days_avoided": kpi_mb["total_stockout_days"] - kpi_dt["total_stockout_days"],
        "manual_baseline_WMAPE_pct": forecast["avg_manual_baseline_WMAPE_pct"],
        "clean_stock_proxy_WMAPE_pct": forecast["avg_clean_stock_proxy_WMAPE_pct"],
        "digital_twin_WMAPE_pct": forecast["avg_digital_twin_WMAPE_pct"],
        "forecast_improvement_pct": forecast["forecast_accuracy_improvement_pct"],
        "data_quality_effect_pct": forecast["data_quality_effect_pct"],
        "observability_effect_pct": forecast["observability_effect_pct"],
        "share_from_data_quality_pct": forecast["share_of_improvement_from_data_quality_pct"],
        "data_inconsistency_rate_pct": quality["inconsistency_rate_pct"],
    }


def _network_rollup(site_results: list[dict]) -> dict:
    """Aggregate sites the same way run_pilot.py does: sum counts, mean rates."""
    return {
        "fill_rate_gain_pct_points": float(np.mean([s["fill_rate_gain_pct_points"] for s in site_results])),
        "stockout_days_avoided": float(sum(s["stockout_days_avoided"] for s in site_results)),
        "manual_baseline_WMAPE_pct": float(np.mean([s["manual_baseline_WMAPE_pct"] for s in site_results])),
        "clean_stock_proxy_WMAPE_pct": float(np.mean([s["clean_stock_proxy_WMAPE_pct"] for s in site_results])),
        "digital_twin_WMAPE_pct": float(np.mean([s["digital_twin_WMAPE_pct"] for s in site_results])),
        "forecast_improvement_pct": float(np.mean([s["forecast_improvement_pct"] for s in site_results])),
        "data_quality_effect_pct": float(np.mean([s["data_quality_effect_pct"] for s in site_results])),
        "observability_effect_pct": float(np.mean([s["observability_effect_pct"] for s in site_results])),
        "share_from_data_quality_pct": float(np.mean([s["share_from_data_quality_pct"] for s in site_results])),
        "data_inconsistency_rate_pct": float(np.mean([s["data_inconsistency_rate_pct"] for s in site_results])),
    }


def experiment_replication(catalog: pd.DataFrame, cfg: dict) -> dict:
    """Re-run the pilot across N independent seeds."""
    print(f"Experiment 1: replication across {N_REPLICATIONS} seeds")
    per_rep, per_site = [], {}

    for rep in range(N_REPLICATIONS):
        site_results = []
        for index, (site_id, rows) in enumerate(catalog.groupby("site_id", sort=True)):
            # rep 0 reproduces run_pilot.py's seeds exactly.
            seed = int(cfg["simulation"]["random_seed"]) + index * 101 + rep * 7919
            result = run_one_site(rows, cfg, seed)
            result["site_name"] = rows.iloc[0]["site_name"]
            site_results.append(result)
            per_site.setdefault(site_id, []).append(result)
        per_rep.append(_network_rollup(site_results))
        if (rep + 1) % 5 == 0:
            print(f"  ...{rep + 1}/{N_REPLICATIONS} replications complete")

    metrics = [k for k in per_rep[0]]
    return {
        "replications": N_REPLICATIONS,
        "note": ("Replication 0 uses the same seeds as run_pilot.py, so the pilot "
                 "figures reported in the main text are one draw from this distribution."),
        "network": {m: summarise([r[m] for r in per_rep]) for m in metrics},
        "per_site": {
            sid: {
                "site_name": runs[0]["site_name"],
                **{m: summarise([r[m] for r in runs])
                   for m in metrics if m != "stockout_days_avoided"},
                "stockout_days_avoided": summarise([r["stockout_days_avoided"] for r in runs]),
            }
            for sid, runs in per_site.items()
        },
    }


def experiment_sensitivity(catalog: pd.DataFrame, cfg: dict) -> dict:
    """Sweep the assumed fragmentation parameters and re-measure the benefit."""
    print("Experiment 2: sensitivity to assumed data-quality parameters")
    base_dq = cfg["data_quality"]
    results = {"error_rate_multiplier": [], "reporting_lag_days": []}

    def sweep(label: str, variants: list[tuple[str, dict]]) -> None:
        for name, overrides in variants:
            trial_cfg = copy.deepcopy(cfg)
            trial_cfg["data_quality"].update(overrides)
            rollups = []
            for rep in range(SENSITIVITY_SEEDS):
                site_results = []
                for index, (site_id, rows) in enumerate(catalog.groupby("site_id", sort=True)):
                    seed = 42 + index * 101 + rep * 7919
                    site_results.append(run_one_site(rows, trial_cfg, seed))
                rollups.append(_network_rollup(site_results))
            entry = {
                "setting": name,
                "params": overrides,
                "fill_rate_gain_pct_points": summarise([r["fill_rate_gain_pct_points"] for r in rollups]),
                "stockout_days_avoided": summarise([r["stockout_days_avoided"] for r in rollups]),
                "data_quality_effect_pct": summarise([r["data_quality_effect_pct"] for r in rollups]),
                "data_inconsistency_rate_pct": summarise([r["data_inconsistency_rate_pct"] for r in rollups]),
            }
            results[label].append(entry)
            print(f"  {label} = {name:>18s}  "
                  f"fill +{entry['fill_rate_gain_pct_points']['mean']:.2f} pts, "
                  f"{entry['stockout_days_avoided']['mean']:.1f} stockout-days avoided")

    # (a) scale all three error rates together
    multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    sweep("error_rate_multiplier", [
        (f"{m:g}x", {
            "manual_entry_error_rate": round(base_dq["manual_entry_error_rate"] * m, 4),
            "missing_entry_rate": round(base_dq["missing_entry_rate"] * m, 4),
            "duplicate_entry_rate": round(base_dq["duplicate_entry_rate"] * m, 4),
        })
        for m in multipliers
    ])

    # (b) vary only the reporting lag, holding error rates at baseline
    sweep("reporting_lag_days", [
        (f"{lag} day(s)", {"reporting_lag_days": lag}) for lag in [0, 1, 2, 3, 5]
    ])

    return results


def experiment_mechanism(catalog: pd.DataFrame, cfg: dict) -> dict:
    """
    Attribute the operational benefit to its three distinct mechanisms by
    switching each one off in turn.

    The manual baseline suffers from three separate information handicaps:

      1. SNAPSHOT STALENESS. simulation.py takes the manual stock reading
         before the day's demand is fulfilled, whereas the digital twin
         checks its reorder point after. A manual count is a point-in-time
         observation and consumption continues after it, so the manual
         process is always acting on a position that is at least one day of
         demand out of date. This handicap is structural: it remains even at
         reporting_lag_days = 0, so the effective information delay is always
         reporting_lag_days + 1.
      2. REPORTING LAG. The configured additional delay before a count
         reaches the person making the reorder decision.
      3. TRANSCRIPTION / MISSING / DUPLICATE ERRORS. The data-accuracy
         handicap.

    The NULL case switches off (2) and (3). Whatever benefit remains is
    attributable to (1) alone, which also serves as a validity check on the
    paired design: both arms face identical demand draws, so the residual
    must be explained by an information difference and nothing else.
    """
    print("Experiment 3: mechanism attribution (turning each handicap off)")
    zero_errors = {"manual_entry_error_rate": 0.0, "missing_entry_rate": 0.0,
                   "duplicate_entry_rate": 0.0}
    cases = {
        "baseline": {},
        "no_reporting_lag": {"reporting_lag_days": 0},
        "no_data_errors": dict(zero_errors),
        "null_snapshot_only": {"reporting_lag_days": 0, **zero_errors},
    }

    measured = {}
    for name, overrides in cases.items():
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg["data_quality"].update(overrides)
        rollups = []
        for rep in range(SENSITIVITY_SEEDS):
            site_results = []
            for index, (site_id, rows) in enumerate(catalog.groupby("site_id", sort=True)):
                seed = 42 + index * 101 + rep * 7919
                site_results.append(run_one_site(rows, trial_cfg, seed))
            rollups.append(_network_rollup(site_results))
        measured[name] = {
            "params": overrides,
            "fill_rate_gain_pct_points": summarise([r["fill_rate_gain_pct_points"] for r in rollups]),
            "stockout_days_avoided": summarise([r["stockout_days_avoided"] for r in rollups]),
            "data_quality_effect_on_WMAPE_pct": summarise([r["data_quality_effect_pct"] for r in rollups]),
        }
        print(f"  {name:20s} fill +{measured[name]['fill_rate_gain_pct_points']['mean']:.2f} pts, "
              f"{measured[name]['stockout_days_avoided']['mean']:.1f} stockout-days avoided")

    base = measured["baseline"]["stockout_days_avoided"]["mean"]
    snapshot = measured["null_snapshot_only"]["stockout_days_avoided"]["mean"]
    lag = base - measured["no_reporting_lag"]["stockout_days_avoided"]["mean"]
    errors = base - measured["no_data_errors"]["stockout_days_avoided"]["mean"]

    attribution = {
        "total_stockout_days_avoided": round(base, 2),
        "from_snapshot_staleness": round(snapshot, 2),
        "from_reporting_lag": round(lag, 2),
        "from_data_errors": round(errors, 2),
        "share_from_snapshot_staleness_pct": round(snapshot / base * 100, 1) if base else None,
        "share_from_reporting_lag_pct": round(lag / base * 100, 1) if base else None,
        "share_from_data_errors_pct": round(errors / base * 100, 1) if base else None,
    }
    print(f"  -> of {attribution['total_stockout_days_avoided']} stockout-days avoided: "
          f"{attribution['share_from_snapshot_staleness_pct']}% snapshot staleness, "
          f"{attribution['share_from_reporting_lag_pct']}% reporting lag, "
          f"{attribution['share_from_data_errors_pct']}% data errors")

    return {"cases": measured, "attribution": attribution}


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    cfg = load_config()
    catalog = load_or_build_catalog(ROOT)
    print(f"Loaded {len(catalog)} site/SKU records\n")

    payload = {
        "baseline_config": cfg["data_quality"],
        "simulation": {k: cfg["simulation"][k] for k in ("duration_days", "random_seed")},
        "replication": experiment_replication(catalog, cfg),
        "sensitivity": experiment_sensitivity(catalog, cfg),
        "mechanism": experiment_mechanism(catalog, cfg),
    }

    path = OUTPUT_DIR / "robustness_results.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    net = payload["replication"]["network"]
    print(f"\nWritten to {path}\n")
    print("Headline replication results (mean [95% CI]):")
    for metric in ("fill_rate_gain_pct_points", "stockout_days_avoided",
                   "forecast_improvement_pct", "data_quality_effect_pct",
                   "observability_effect_pct", "share_from_data_quality_pct"):
        s = net[metric]
        print(f"  {metric:34s} {s['mean']:8.2f}  [{s['ci95_low']:7.2f}, {s['ci95_high']:7.2f}]")


if __name__ == "__main__":
    main()
