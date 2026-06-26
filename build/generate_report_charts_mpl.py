"""
generate_report_charts.py (matplotlib version)

Generates static PNG charts for embedding in the Word report.
Uses matplotlib instead of plotly+kaleido, since kaleido requires a
Chrome binary that isn't available in this sandboxed environment.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "outputs", "dashboard_data.json")
OUT_DIR = os.path.join(ROOT, "outputs", "report_charts")

COLOR_BEFORE = "#5b6b76"
COLOR_AFTER = "#0e7c66"
COLOR_RISK = "#b3641a"
COLOR_GRID = "#e8e2d0"
COLOR_INK = "#13202b"

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


def chart_total_stock(data):
    d = data["daily_stock"]
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    ax.plot(d["days"], d["manual_baseline_true_stock"], color=COLOR_BEFORE,
            linewidth=2.5, label="Manual baseline (true stock)")
    ax.plot(d["days"], d["manual_baseline_perceived_stock"], color=COLOR_BEFORE,
            linewidth=1.8, linestyle=":", label="Manual baseline (perceived / spreadsheet)")
    ax.plot(d["days"], d["digital_twin_total_stock"], color=COLOR_AFTER,
            linewidth=2.5, label="Digital twin (true stock)")
    ax.set_title("Total On-Hand Stock — Manual Baseline vs Digital Twin", fontsize=15, weight="bold", pad=14)
    ax.set_xlabel("Day of pilot")
    ax.set_ylabel("Total units on hand")
    style_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "chart_total_stock.png"))
    plt.close(fig)


def chart_sku_stockout(data):
    s = data["per_sku_series"]["MED-006"]
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    ax.plot(s["days"], s["manual_baseline_true_stock"], color=COLOR_BEFORE,
            linewidth=2.5, label="Manual baseline (true stock)")
    ax.plot(s["days"], s["manual_baseline_perceived_stock"], color=COLOR_BEFORE,
            linewidth=1.8, linestyle=":", label="Manual baseline (perceived / spreadsheet)")
    ax.plot(s["days"], s["digital_twin_stock"], color=COLOR_AFTER,
            linewidth=2.5, label="Digital twin (true stock)")
    ax.axhline(s["reorder_point"], color=COLOR_RISK, linewidth=1.5, linestyle="--", label="Reorder point")
    ax.axhline(0, color=COLOR_RISK, linewidth=1)
    ax.set_title(f"{s['sku_name']} — Where Lagged Visibility Causes a Stockout",
                 fontsize=14, weight="bold", pad=14)
    ax.set_xlabel("Day of pilot")
    ax.set_ylabel("Units on hand")
    style_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "chart_sku_stockout.png"))
    plt.close(fig)


def chart_forecast_accuracy(data):
    rows = data["forecast_table"]
    sku_ids = [r["sku_id"] for r in rows]
    mb = [r["manual_baseline_WMAPE_pct"] for r in rows]
    dt = [r["digital_twin_WMAPE_pct"] for r in rows]

    x = range(len(sku_ids))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    ax.bar([i - width/2 for i in x], mb, width=width, color=COLOR_BEFORE, label="Manual baseline")
    ax.bar([i + width/2 for i in x], dt, width=width, color=COLOR_AFTER, label="Digital twin")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sku_ids)
    ax.set_title("Forecast Accuracy by SKU (WMAPE — lower is better)", fontsize=15, weight="bold", pad=14)
    ax.set_xlabel("SKU")
    ax.set_ylabel("WMAPE (%)")
    style_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "chart_forecast_accuracy.png"))
    plt.close(fig)


def chart_data_quality_summary(data):
    dq = data["results"]["data_quality_metrics"]
    categories = ["Inconsistent\nvalues", "Missing\nentries", "Duplicate\nrecords"]
    values = [dq["inconsistent_records"], dq["missing_records"], dq["duplicate_records"]]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    bars = ax.bar(categories, values, color=COLOR_RISK, width=0.55)
    for bar, v in zip(bars, values):
        ax.annotate(str(v), (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=11, weight="bold")
    ax.set_title(f"Simulated Spreadsheet Data Quality Issues (n={dq['total_records']} records)",
                 fontsize=13.5, weight="bold", pad=14)
    ax.set_ylabel("Number of records")
    style_axes(ax)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "chart_data_quality.png"))
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(DATA_PATH) as f:
        data = json.load(f)

    chart_total_stock(data)
    chart_sku_stockout(data)
    chart_forecast_accuracy(data)
    chart_data_quality_summary(data)

    print("Charts written to", OUT_DIR)
    for f in sorted(os.listdir(OUT_DIR)):
        print(" -", f)


if __name__ == "__main__":
    main()
