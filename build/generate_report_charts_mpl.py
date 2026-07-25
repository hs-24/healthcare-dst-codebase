"""
generate_report_charts.py (matplotlib version)

Generates static PNG charts for embedding in the Word report, from the
live multi-site dashboard_data.json / pilot_results.json produced by
run_pilot.py + export_dashboard_data.py. Uses matplotlib instead of
plotly+kaleido, since kaleido requires a Chrome binary that isn't
available in this sandboxed environment.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.path.dirname(os.path.dirname(__file__))
DASHBOARD_PATH = os.path.join(ROOT, "outputs", "dashboard_data.json")
PILOT_PATH = os.path.join(ROOT, "outputs", "pilot_results.json")
OUT_DIR = os.path.join(ROOT, "outputs", "report_charts")

COLOR_BEFORE = "#5b6b76"
COLOR_AFTER = "#0e7c66"
COLOR_RISK = "#b3641a"
COLOR_GRID = "#e8e2d0"
COLOR_INK = "#13202b"

REPRESENTATIVE_SITE = "SITE_A"
REPRESENTATIVE_SKU = "HSC-108"  # Ventilator, clear stockout-day gap between methods

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.edgecolor": COLOR_INK,
    "axes.labelcolor": COLOR_INK,
    "text.color": COLOR_INK,
    "xtick.color": COLOR_INK,
    "ytick.color": COLOR_INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def style_axes(ax):
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLOR_INK)


def chart_total_stock(dashboard):
    """Network-wide total on-hand stock: sum of all three sites' daily_stock series."""
    sites = list(dashboard["sites"].values())
    days = sites[0]["daily_stock"]["days"]
    n = len(days)

    def summed(key):
        totals = [0.0] * n
        for site in sites:
            series = site["daily_stock"][key]
            for i in range(n):
                v = series[i]
                totals[i] += v if v is not None else 0.0
        return totals

    dt_total = summed("digital_twin_total_stock")
    mb_true = summed("manual_baseline_true_stock")
    mb_perceived = summed("manual_baseline_perceived_stock")

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    ax.plot(days, mb_true, color=COLOR_BEFORE, linewidth=2.5,
            label="Manual baseline (true stock)")
    ax.plot(days, mb_perceived, color=COLOR_BEFORE, linewidth=1.8, linestyle=":",
            label="Manual baseline (perceived / spreadsheet)")
    ax.plot(days, dt_total, color=COLOR_AFTER, linewidth=2.5,
            label="Digital twin (reconciled stock)")
    ax.set_title("Network Total On-Hand Stock — Manual Baseline vs Digital Twin\n(3 sites combined)",
                 fontsize=14, weight="bold", pad=14)
    ax.set_xlabel("Day of 30-day pilot")
    ax.set_ylabel("Total units on hand")
    style_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "chart_total_stock.png"))
    plt.close(fig)


def chart_sku_stockout(dashboard):
    site = dashboard["sites"][REPRESENTATIVE_SITE]
    s = site["per_sku_series"][REPRESENTATIVE_SKU]
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    ax.plot(s["days"], s["manual_baseline_true_stock"], color=COLOR_BEFORE,
            linewidth=2.5, label="Manual baseline (true stock)")
    ax.plot(s["days"], s["manual_baseline_perceived_stock"], color=COLOR_BEFORE,
            linewidth=1.8, linestyle=":", label="Manual baseline (perceived / spreadsheet)")
    ax.plot(s["days"], s["digital_twin_stock"], color=COLOR_AFTER,
            linewidth=2.5, label="Digital twin (reconciled stock)")
    ax.axhline(s["reorder_point"], color=COLOR_RISK, linewidth=1.5, linestyle="--", label="Reorder point")
    ax.axhline(0, color=COLOR_RISK, linewidth=1)
    ax.set_title(f"{site['site_name']} — {s['sku_name']} ({REPRESENTATIVE_SKU})\nWhere Lagged/Corrupted Visibility Causes an Avoidable Stockout",
                 fontsize=13.5, weight="bold", pad=14)
    ax.set_xlabel("Day of pilot")
    ax.set_ylabel(f"Units on hand ({s['uom']})")
    style_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "chart_sku_stockout.png"))
    plt.close(fig)


def chart_forecast_accuracy(pilot):
    sites = pilot["sites"]
    site_ids = list(sites.keys())
    site_names = [sites[sid]["site_name"] for sid in site_ids]
    mb = [sites[sid]["forecast_accuracy_summary"]["avg_manual_baseline_WMAPE_pct"] for sid in site_ids]
    dt = [sites[sid]["forecast_accuracy_summary"]["avg_digital_twin_WMAPE_pct"] for sid in site_ids]

    x = range(len(site_ids))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=160)
    ax.bar([i - width / 2 for i in x], mb, width=width, color=COLOR_BEFORE, label="Manual baseline")
    ax.bar([i + width / 2 for i in x], dt, width=width, color=COLOR_AFTER, label="Digital twin")
    ax.set_xticks(list(x))
    ax.set_xticklabels(site_names, fontsize=10)
    ax.set_title("Forecast Accuracy by Site — Identical Moving-Average Method\nRun on Fragmented vs Centralised Data (WMAPE, lower is better)",
                 fontsize=13, weight="bold", pad=14)
    ax.set_ylabel("Demand-weighted WMAPE (%)")
    style_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "chart_forecast_accuracy.png"))
    plt.close(fig)


def chart_data_quality_summary(pilot):
    sites = pilot["sites"]
    inconsistent = sum(s["data_quality_metrics"]["inconsistent_records"] for s in sites.values())
    missing = sum(s["data_quality_metrics"]["missing_records"] for s in sites.values())
    duplicate = sum(s["data_quality_metrics"]["duplicate_records"] for s in sites.values())
    total = sum(s["data_quality_metrics"]["total_records"] for s in sites.values())

    categories = ["Inconsistent\nvalues", "Missing\nentries", "Duplicate\nrecords"]
    values = [inconsistent, missing, duplicate]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    bars = ax.bar(categories, values, color=COLOR_RISK, width=0.55)
    for bar, v in zip(bars, values):
        ax.annotate(str(v), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=11, weight="bold")
    ax.set_title(f"Simulated Fragmented-Spreadsheet Data Quality Issues\n(All 3 sites combined, n={total} records)",
                 fontsize=13, weight="bold", pad=14)
    ax.set_ylabel("Number of records")
    style_axes(ax)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "chart_data_quality.png"))
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(DASHBOARD_PATH) as f:
        dashboard = json.load(f)
    with open(PILOT_PATH) as f:
        pilot = json.load(f)

    chart_total_stock(dashboard)
    chart_sku_stockout(dashboard)
    chart_forecast_accuracy(pilot)
    chart_data_quality_summary(pilot)

    print("Charts written to", OUT_DIR)
    for f in sorted(os.listdir(OUT_DIR)):
        print(" -", f)


if __name__ == "__main__":
    main()
