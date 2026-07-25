"""
generate_extra_figures_mpl.py

Generates the four figures added after the mechanism analysis:

  Figure 5  chart_process_current.png       current-state information flow
  Figure 6  chart_process_proposed.png      proposed-state information flow
  Figure 7  chart_forecast_decomposition.png  where the forecast gain comes from
  Figure 8  chart_sensitivity.png           robustness to assumed parameters

Figures 5 and 6 are process documentation, not field observation. Every box in
Figure 5 corresponds to a parameter that already exists in the simulation code,
and the delay ladder is annotated with the measured attribution from
outputs/robustness_results.json. The two organisational lanes reflect a
qualitative observation from the team's industry attachment: replenishment
decisions are made by a different team from the warehouse floor staff who pick,
pack and count. No cycle times, headcounts or role names are invented, and no
value-added / non-value-added timings are claimed, because none were measured.

Figures 7 and 8 read outputs/robustness_results.json, so they cannot drift from
the experiment. Run build/run_robustness.py first.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "report_charts")
ROBUSTNESS_PATH = os.path.join(ROOT, "outputs", "robustness_results.json")

COLOR_BEFORE = "#5b6b76"
COLOR_AFTER = "#0e7c66"
COLOR_RISK = "#b3641a"
COLOR_GRID = "#e8e2d0"
COLOR_INK = "#13202b"
COLOR_LANE_A = "#eef1f7"
COLOR_LANE_B = "#f7f0e6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "text.color": COLOR_INK,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def _box(ax, x, y, w, h, text, face, edge, fontsize=9.5, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.3, facecolor=face, edgecolor=edge, zorder=3,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=COLOR_INK, zorder=4,
            weight="bold" if bold else "normal", linespacing=1.45)


def _arrow(ax, start, end, color=COLOR_INK, style="-|>", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color, linestyle=ls,
        shrinkA=2, shrinkB=2, zorder=5,
    ))


GUTTER = 0.075  # reserved left column for lane titles, so labels never collide


def _lane(ax, y, h, label, face, sub=None):
    """Swimlane band with its title rotated into a reserved left gutter."""
    ax.add_patch(FancyBboxPatch(
        (0.005, y), 0.99, h, boxstyle="square,pad=0",
        linewidth=0, facecolor=face, zorder=1,
    ))
    ax.add_patch(FancyBboxPatch(
        (0.005, y), GUTTER - 0.005, h, boxstyle="square,pad=0",
        linewidth=0, facecolor="white", alpha=0.55, zorder=2,
    ))
    # Kept deliberately short: a rotated label longer than the lane height
    # overflows into the neighbouring lane. Fuller detail belongs in the caption.
    ax.text(GUTTER / 2, y + h / 2, label, ha="center", va="center", rotation=90,
            fontsize=9, style="italic", color="#4a4a4a", zorder=3,
            linespacing=1.35)


def chart_process_current(attribution: dict) -> None:
    """Figure 5: current-state manual information flow, two organisational lanes."""
    fig, ax = plt.subplots(figsize=(12.4, 6.4), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _lane(ax, 0.58, 0.40, "Warehouse\nfloor team", COLOR_LANE_A)
    _lane(ax, 0.16, 0.40, "Replenishment\nteam", COLOR_LANE_B)

    xs = [0.09, 0.315, 0.54, 0.765]
    bw, by, bh = 0.21, 0.66, 0.22
    _box(ax, xs[0], by, bw, bh,
         "Stock issued to wards\nthrough the day\n(picking and packing)", "white", COLOR_BEFORE)
    _box(ax, xs[1], by, bw, bh,
         "Periodic manual\nstock count\n(taken before the day's\nremaining issues)", "white", COLOR_RISK)
    _box(ax, xs[2], by, bw, bh,
         "Figure written down\nand keyed into a\nspreadsheet", "white", COLOR_BEFORE)
    _box(ax, xs[3], by, bw, bh,
         "Held across several\nseparate workbooks\n(blanks, re-keyed rows,\nduplicate entries)",
         "white", COLOR_BEFORE)
    for i in range(3):
        _arrow(ax, (xs[i] + bw, by + bh / 2), (xs[i + 1], by + bh / 2))

    # Organisational handoff between the two lanes
    hx = xs[3] + bw / 2
    _arrow(ax, (hx, by), (hx, 0.50), color=COLOR_RISK, lw=2.2)
    ax.text(hx - 0.02, 0.575,
            "handoff between teams: the replenishment\nteam never sees the physical shelf",
            ha="right", va="center", fontsize=9, color=COLOR_RISK,
            style="italic", linespacing=1.4)

    ly, lh = 0.26, 0.22
    _box(ax, xs[3], ly, bw, lh,
         "Consolidate workbooks,\nfill gaps by judgement", "white", COLOR_BEFORE)
    _box(ax, xs[2], ly, bw, lh,
         "Forecast in Excel using\nplanner assumptions", "white", COLOR_RISK)
    _box(ax, xs[1], ly, bw, lh,
         "Reorder decision made\non the reported figure", "white", COLOR_RISK, bold=True)
    _box(ax, xs[0], ly, bw, lh,
         "Purchase order\nraised", "white", COLOR_BEFORE)
    for i in range(3, 0, -1):
        _arrow(ax, (xs[i], ly + lh / 2), (xs[i - 1] + bw, ly + lh / 2))

    ax.text(0.5, 0.095,
            "Information age at the decision point: about 2 days "
            "(1 day because the count precedes the day's issues, 1 day of reporting lag)",
            ha="center", va="center", fontsize=9.5, color=COLOR_RISK, weight="bold")

    snap = attribution["share_from_snapshot_staleness_pct"]
    lag = attribution["share_from_reporting_lag_pct"]
    err = attribution["share_from_data_errors_pct"]
    ax.text(0.5, 0.035,
            f"Measured contribution to avoidable stockouts:  "
            f"count-before-issues {snap}%   |   reporting lag {lag}%   |   "
            f"transcription, blank and duplicate entries {err}%",
            ha="center", va="center", fontsize=9.5, color=COLOR_INK)

    # No figure number in the image itself: the report caption owns the numbering,
    # and baking one in here guarantees they disagree the moment sections move.
    ax.set_title("Current-State Information Flow in a Spreadsheet-Stage Warehouse",
                 fontsize=13, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "chart_process_current.png"), bbox_inches="tight")
    plt.close(fig)


def chart_process_proposed() -> None:
    """Figure 6: proposed-state flow through the four-agent workflow."""
    fig, ax = plt.subplots(figsize=(12.4, 6.4), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _lane(ax, 0.76, 0.22, "Warehouse\nfloor team", COLOR_LANE_A)
    _lane(ax, 0.34, 0.38, "Digital twin\n(automated)", "#e8f3ef")
    _lane(ax, 0.08, 0.22, "Planning\nteam", COLOR_LANE_B)

    _box(ax, 0.09, 0.80, 0.36, 0.14,
         "Consumption captured as it happens\nat the point of issue", "white", COLOR_AFTER, bold=True)
    ax.text(0.47, 0.87, "no separate counting step,\nno re-keying", ha="left", va="center",
            fontsize=9, color=COLOR_AFTER, style="italic", linespacing=1.4)

    ay, ah, aw, gap = 0.42, 0.18, 0.206, 0.02
    axs = [0.09 + i * (aw + gap) for i in range(4)]
    _arrow(ax, (axs[0] + aw / 2, 0.80), (axs[0] + aw / 2, 0.60), color=COLOR_AFTER, lw=2.0)
    for x, (name, desc) in zip(axs, [
        ("DataValidationAgent", "reconciles against last\nvalidated figure"),
        ("ForecastingAgent", "exponential smoothing,\nupdated every tick"),
        ("ReplenishmentAgent", "orders from the validated\nposition and live forecast"),
        ("SupervisorAgent", "severity-ranked\nsingle event log"),
    ]):
        _box(ax, x, ay, aw, ah, f"{name}\n\n{desc}", "white", COLOR_AFTER, fontsize=9)
    for i in range(3):
        _arrow(ax, (axs[i] + aw, ay + ah / 2), (axs[i + 1], ay + ah / 2), color=COLOR_AFTER)

    _arrow(ax, (axs[3] + aw / 2, ay), (axs[3] + aw / 2, 0.27), color=COLOR_AFTER, lw=2.0)

    _box(ax, 0.55, 0.12, 0.425, 0.15,
         "Live multi-site dashboard:\nper-site KPIs, alerts, cross-site transfer suggestions",
         "white", COLOR_AFTER)
    _box(ax, 0.09, 0.12, 0.40, 0.15,
         "Planner reviews exceptions only\nand approves cross-site transfers",
         "white", COLOR_BEFORE)
    _arrow(ax, (0.55, 0.195), (0.49, 0.195), color=COLOR_AFTER)

    ax.text(0.5, 0.035,
            "Information age at the decision point: current by construction. "
            "The count-before-issues delay and the reporting lag are both removed.",
            ha="center", va="center", fontsize=9.5, color=COLOR_AFTER, weight="bold")

    ax.set_title("Proposed-State Information Flow Through the Digital Twin",
                 fontsize=13, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "chart_process_proposed.png"), bbox_inches="tight")
    plt.close(fig)


def chart_forecast_decomposition(net: dict) -> None:
    """Figure 7: the three forecasting arms, with 95% CIs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.4), dpi=160,
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    arms = [
        ("Manual baseline\n(proxy from\nfragmented stock)", net["manual_baseline_WMAPE_pct"], COLOR_BEFORE),
        ("Control\n(same proxy from\nCLEAN stock)", net["clean_stock_proxy_WMAPE_pct"], COLOR_RISK),
        ("Digital twin\n(demand captured\ndirectly)", net["digital_twin_WMAPE_pct"], COLOR_AFTER),
    ]
    xs = range(len(arms))
    means = [a[1]["mean"] for a in arms]
    errs = [[a[1]["mean"] - a[1]["ci95_low"] for a in arms],
            [a[1]["ci95_high"] - a[1]["mean"] for a in arms]]
    ax1.bar(xs, means, color=[a[2] for a in arms], width=0.6,
            yerr=errs, capsize=5, ecolor=COLOR_INK)
    for x, m in zip(xs, means):
        ax1.text(x, m + 1.6, f"{m:.1f}%", ha="center", fontsize=11, weight="bold")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([a[0] for a in arms], fontsize=9.5)
    ax1.set_ylabel("Demand-weighted WMAPE (%)")
    ax1.set_ylim(0, max(means) * 1.25)
    ax1.set_title("Three forecasting arms, identical method\n(20 replications, 95% CI)",
                  fontsize=12, weight="bold", pad=10)
    ax1.grid(True, axis="y", color=COLOR_GRID, linewidth=0.8)
    ax1.set_axisbelow(True)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)

    dq = net["data_quality_effect_pct"]["mean"]
    obs = net["observability_effect_pct"]["mean"]
    ax2.barh([1], [dq], color=COLOR_RISK, height=0.45, label="Data quality")
    ax2.barh([0], [obs], color=COLOR_AFTER, height=0.45, label="Observability")
    ax2.text(dq + 1.5, 1, f"{dq:.1f}%  cleaning the spreadsheet", va="center", fontsize=10.5)
    ax2.text(obs - 2, 0, f"{obs:.1f}%  capturing demand", va="center", ha="right",
             fontsize=10.5, color="white", weight="bold")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Observability\neffect", "Data-quality\neffect"], fontsize=10)
    ax2.set_xlabel("Reduction in forecast error (%)")
    ax2.set_xlim(0, 100)
    ax2.set_title("Where the improvement actually comes from",
                  fontsize=12, weight="bold", pad=10)
    ax2.grid(True, axis="x", color=COLOR_GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "chart_forecast_decomposition.png"), bbox_inches="tight")
    plt.close(fig)


def chart_sensitivity(sens: dict) -> None:
    """Figure 8: robustness of the operational benefit to the assumed parameters."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0), dpi=160)

    err = sens["error_rate_multiplier"]
    xs = [float(e["setting"].rstrip("x")) for e in err]
    ys = [e["stockout_days_avoided"]["mean"] for e in err]
    lo = [e["stockout_days_avoided"]["ci95_low"] or e["stockout_days_avoided"]["mean"] for e in err]
    hi = [e["stockout_days_avoided"]["ci95_high"] or e["stockout_days_avoided"]["mean"] for e in err]
    ax1.plot(xs, ys, marker="o", color=COLOR_RISK, linewidth=2.4, markersize=7)
    ax1.fill_between(xs, lo, hi, color=COLOR_RISK, alpha=0.15)
    ax1.axvline(1.0, color=COLOR_INK, linestyle="--", linewidth=1.2)
    ax1.text(1.06, min(ys), "assumed\nbaseline", fontsize=9, color=COLOR_INK)
    ax1.set_xlabel("Multiplier applied to all three assumed error rates")
    ax1.set_ylabel("Stockout-SKU-days avoided")
    ax1.set_title("Benefit is almost flat in the assumed error rates\n"
                  "(it survives setting them all to zero)", fontsize=11.5, weight="bold", pad=10)
    ax1.set_ylim(0, max(ys) * 1.35)

    lag = sens["reporting_lag_days"]
    lx = [int(e["setting"].split()[0]) for e in lag]
    ly = [e["stockout_days_avoided"]["mean"] for e in lag]
    llo = [e["stockout_days_avoided"]["ci95_low"] or e["stockout_days_avoided"]["mean"] for e in lag]
    lhi = [e["stockout_days_avoided"]["ci95_high"] or e["stockout_days_avoided"]["mean"] for e in lag]
    ax2.plot(lx, ly, marker="o", color=COLOR_AFTER, linewidth=2.4, markersize=7)
    ax2.fill_between(lx, llo, lhi, color=COLOR_AFTER, alpha=0.15)
    ax2.axvline(1, color=COLOR_INK, linestyle="--", linewidth=1.2)
    ax2.text(1.12, min(ly), "assumed\nbaseline", fontsize=9, color=COLOR_INK)
    ax2.set_xlabel("Additional reporting lag (days)")
    ax2.set_ylabel("Stockout-SKU-days avoided")
    ax2.set_title("Benefit scales strongly with information delay\n"
                  "(latency is the binding constraint)", fontsize=11.5, weight="bold", pad=10)
    ax2.set_ylim(0, max(ly) * 1.2)

    for ax in (ax1, ax2):
        ax.grid(True, color=COLOR_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "chart_sensitivity.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(ROBUSTNESS_PATH, encoding="utf-8") as handle:
        rob = json.load(handle)

    chart_process_current(rob["mechanism"]["attribution"])
    chart_process_proposed()
    chart_forecast_decomposition(rob["replication"]["network"])
    chart_sensitivity(rob["sensitivity"])

    print("Extra figures written to", OUT_DIR)
    for name in ["chart_process_current.png", "chart_process_proposed.png",
                 "chart_forecast_decomposition.png", "chart_sensitivity.png"]:
        print(" -", name)


if __name__ == "__main__":
    main()
