# Healthcare Warehouse Digital Supply Chain Twin — Pilot PoC

DSC2205 Group 3 — Assignment 2 supporting technical deliverable.

A Python-based Digital Supply Chain Twin (DSCT) proof-of-concept,
simulating a 1-month pilot at a single healthcare warehousing site.
Built using a discrete-event simulation architecture (`simpy`) adapted
from the open-source reference project
[AndreVale69/simulator-automatic-warehouse](https://github.com/AndreVale69/simulator-automatic-warehouse).

## What this demonstrates

Two parallel simulations are run against the *identical* 30-day demand
sequence:

- **Digital Twin** — reorder decisions use true, real-time stock levels
  (representing a centralised dashboard).
- **Manual Baseline** — reorder decisions use a lagged, occasionally
  corrupted "perceived stock" figure, recreating today's fragmented,
  manual spreadsheet process.

Because both runs see the same demand, any difference in outcome
(stockouts, forecast error, reorder frequency) is attributable purely
to data visibility — directly evidencing the problem statement in
Section 1 of the report.

## Requirements

```
pip install simpy pandas pyyaml jsonschema numpy matplotlib
```

(`plotly`/`dash` are referenced in the original design but the final
pipeline uses `matplotlib` for static chart export, since the sandbox
this was built in had no Chrome binary for `kaleido`. Either works for
local use — swap back to `plotly`+`kaleido` if Chrome is available.)

## How to run

```bash
# 1. Run the simulation pipeline (digital twin + manual baseline)
python3 run_pilot.py

# 2. Export consolidated data for the dashboard
python3 export_dashboard_data.py

# 3. Build the interactive HTML dashboard
python3 build/build_dashboard.py

# 4. (Optional) Regenerate static charts for the Word report
python3 build/generate_report_charts_mpl.py
```

All outputs land in `data/` (raw simulation CSVs) and `outputs/`
(dashboard, charts, JSON summaries).

## File structure

```
config/warehouse_config.yaml     SKU list, demand profiles, site config, data-quality params
src/digital_twin/warehouse.py    SKU + Warehouse domain model
src/digital_twin/simulation.py   WarehouseSimulation + ManualBaselineSimulation (simpy engines)
src/digital_twin/fragmentation.py  Generates + reconciles the simulated messy spreadsheet data
src/digital_twin/forecasting.py  Forecast accuracy comparison (MAE / MAPE / WMAPE)
run_pilot.py                     Main entry point
export_dashboard_data.py         Consolidates outputs into one JSON for the dashboard
build/build_dashboard.py         Injects data into the dashboard HTML template
build/generate_report_charts_mpl.py  Static PNG charts for the Word report
outputs/healthcare_dst_dashboard.html  Self-contained interactive dashboard (open in any browser)
```

## Adjusting the pilot

Edit `config/warehouse_config.yaml` to change:
- `simulation.duration_days` — pilot length
- `simulation.random_seed` — change for a different demand draw (results
  are robust across seeds: forecast accuracy improvement has tested in
  the 10–48% range, always positive)
- `skus` — add/remove SKUs, change demand profiles or reorder logic
- `data_quality` — control how "messy" the simulated manual process is

After editing the config, re-run steps 1–3 above to regenerate everything.

## Notes on the result figures used in the report

Using `random_seed: 42` (the version reflected in the report section):

| Metric | Manual Baseline | Digital Twin |
|---|---|---|
| Site fill rate | 99.82% | 100.00% |
| Stockout days | 1 | 0 |
| Reorders triggered | 11 | 6 |
| Data inconsistency rate | 11.7% | 0% (by construction) |
| Forecast WMAPE | 30.93% | 27.61% |
| Forecast accuracy improvement | — | 10.75% |

These figures are from synthetic data generated for this proof-of-concept
and are not derived from live production data, consistent with the
agreed 1-month pilot scope.
