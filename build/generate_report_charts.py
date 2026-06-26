"""
generate_report_charts.py

Generates static PNG charts (via plotly + kaleido) for embedding in
the Word report, since the interactive HTML dashboard can't be
embedded directly in a .docx file.
"""

import os
import json
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "outputs", "dashboard_data.json")
OUT_DIR = os.path.join(ROOT, "outputs", "report_charts")

COLOR_BEFORE = "#5b6b76"
COLOR_AFTER = "#0e7c66"
COLOR_RISK = "#b3641a"

FONT = dict(family="Arial, sans-serif", size=14, color="#13202b")


def base_layout(title, xtitle, ytitle):
    return go.Layout(
        title=dict(text=title, font=dict(family="Arial", size=18, color="#13202b")),
        xaxis=dict(title=xtitle, gridcolor="#e8e2d0", zeroline=False),
        yaxis=dict(title=ytitle, gridcolor="#e8e2d0", zeroline=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=FONT,
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=70, r=30, t=60, b=60),
        width=1000,
        height=560,
    )


def chart_total_stock(data):
    d = data["daily_stock"]
    fig = go.Figure(layout=base_layout(
        "Total On-Hand Stock — Manual Baseline vs Digital Twin",
        "Day of pilot", "Total units on hand"
    ))
    fig.add_trace(go.Scatter(
        x=d["days"], y=d["manual_baseline_true_stock"],
        name="Manual baseline (true stock)", line=dict(color=COLOR_BEFORE, width=3)
    ))
    fig.add_trace(go.Scatter(
        x=d["days"], y=d["manual_baseline_perceived_stock"],
        name="Manual baseline (perceived/spreadsheet)",
        line=dict(color=COLOR_BEFORE, width=2, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=d["days"], y=d["digital_twin_total_stock"],
        name="Digital twin (true stock)", line=dict(color=COLOR_AFTER, width=3)
    ))
    fig.write_image(os.path.join(OUT_DIR, "chart_total_stock.png"), scale=2)


def chart_sku_stockout(data):
    """Focus chart on MED-006 where the manual baseline stockout occurs."""
    s = data["per_sku_series"]["MED-006"]
    fig = go.Figure(layout=base_layout(
        f"{s['sku_name']} — Where Lagged Visibility Causes a Stockout",
        "Day of pilot", "Units on hand"
    ))
    fig.add_trace(go.Scatter(
        x=s["days"], y=s["manual_baseline_true_stock"],
        name="Manual baseline (true stock)", line=dict(color=COLOR_BEFORE, width=3)
    ))
    fig.add_trace(go.Scatter(
        x=s["days"], y=s["manual_baseline_perceived_stock"],
        name="Manual baseline (perceived/spreadsheet)",
        line=dict(color=COLOR_BEFORE, width=2, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=s["days"], y=s["digital_twin_stock"],
        name="Digital twin (true stock)", line=dict(color=COLOR_AFTER, width=3)
    ))
    fig.add_trace(go.Scatter(
        x=s["days"], y=[s["reorder_point"]] * len(s["days"]),
        name="Reorder point", line=dict(color=COLOR_RISK, width=1.5, dash="dash")
    ))
    fig.add_hline(y=0, line=dict(color="#b3641a", width=1))
    fig.write_image(os.path.join(OUT_DIR, "chart_sku_stockout.png"), scale=2)


def chart_forecast_accuracy(data):
    rows = data["forecast_table"]
    fig = go.Figure(layout=base_layout(
        "Forecast Accuracy by SKU (WMAPE — lower is better)",
        "SKU", "WMAPE (%)"
    ))
    fig.add_trace(go.Bar(
        x=[r["sku_id"] for r in rows], y=[r["manual_baseline_WMAPE_pct"] for r in rows],
        name="Manual baseline", marker_color=COLOR_BEFORE
    ))
    fig.add_trace(go.Bar(
        x=[r["sku_id"] for r in rows], y=[r["digital_twin_WMAPE_pct"] for r in rows],
        name="Digital twin", marker_color=COLOR_AFTER
    ))
    fig.update_layout(barmode="group")
    fig.write_image(os.path.join(OUT_DIR, "chart_forecast_accuracy.png"), scale=2)


def chart_data_quality_summary(data):
    dq = data["results"]["data_quality_metrics"]
    categories = ["Inconsistent\nvalues", "Missing\nentries", "Duplicate\nrecords"]
    values = [dq["inconsistent_records"], dq["missing_records"], dq["duplicate_records"]]
    fig = go.Figure(layout=base_layout(
        f"Simulated Spreadsheet Data Quality Issues (n={dq['total_records']} records)",
        "Issue type", "Number of records"
    ))
    fig.add_trace(go.Bar(x=categories, y=values, marker_color=COLOR_RISK,
                          text=values, textposition="outside"))
    fig.update_layout(showlegend=False, height=480)
    fig.write_image(os.path.join(OUT_DIR, "chart_data_quality.png"), scale=2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(DATA_PATH) as f:
        data = json.load(f)

    chart_total_stock(data)
    chart_sku_stockout(data)
    chart_forecast_accuracy(data)
    chart_data_quality_summary(data)

    print("Charts written to", OUT_DIR)
    for f in os.listdir(OUT_DIR):
        print(" -", f)


if __name__ == "__main__":
    main()
