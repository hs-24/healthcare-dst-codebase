"""
server.py

Real-time backend for the Healthcare Warehouse Digital Supply Chain
Twin. Replays the 30-day pilot (already computed by run_pilot.py /
export_dashboard_data.py into outputs/dashboard_data.json) one
simulated day per tick, on a background thread, and runs every tick
through the agentic AI workflow (src/digital_twin/agents.py) before
publishing the result.

Endpoints:
  GET /                  three.js live digital twin dashboard
  GET /api/state         current day's per-SKU twin state + KPIs
  GET /api/agent-log      latest agentic workflow events
  POST /api/control       {"action": "play"|"pause"|"reset"|"speed", "value": ...}

Run:
  python server.py
  then open http://127.0.0.1:5000/
"""

import json
import os
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from digital_twin.agents import AgentWorkflow  # noqa: E402

ROOT = Path(__file__).parent
DASHBOARD_DATA_PATH = ROOT / "outputs" / "dashboard_data.json"
CONFIG_PATH = Path(os.environ.get("WAREHOUSE_CONFIG_PATH", ROOT / "config" / "warehouse_config.yaml"))

app = Flask(__name__, static_folder=None)


class LiveTwin:
    def __init__(self, data: dict):
        self.data = data
        self.per_sku = data["per_sku_series"]
        self.results = data["results"]
        self.sku_ids = list(self.per_sku.keys())
        self.n_days = len(self.per_sku[self.sku_ids[0]]["days"])

        self.sku_config = self._load_sku_config(CONFIG_PATH)

        self.workflow = AgentWorkflow()
        self.current_day = 0
        self.playing = True
        self.speed = 1.5  # seconds per simulated day
        self._lock = threading.Lock()
        self._on_order: dict[str, float] = {sid: 0.0 for sid in self.sku_ids}
        self._latest_state: dict = {}
        self._advance_to(0)

    @staticmethod
    def _load_sku_config(path: Path) -> dict:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return {sku["sku_id"]: sku for sku in cfg["skus"]}

    def _advance_to(self, day: int) -> None:
        per_sku_state = {}
        for sku_id in self.sku_ids:
            series = self.per_sku[sku_id]
            cfg = self.sku_config[sku_id]
            true_stock = series["digital_twin_stock"][day]
            perceived = series["manual_baseline_perceived_stock"][day]
            demand = series["demand"][day]

            result = self.workflow.process_tick(
                sku_id,
                perceived_reading=perceived,
                true_stock=true_stock,
                on_order=self._on_order[sku_id],
                observed_demand=demand,
                reorder_point=series["reorder_point"],
                reorder_qty=cfg["reorder_qty"],
                lead_time_days=cfg["lead_time_days"],
            )
            if result["decision"] and result["decision"]["action"] == "REORDER":
                self._on_order[sku_id] += result["decision"]["qty"]

            units_per_pallet = cfg["units_per_pallet"]
            max_stock_ref = cfg["initial_stock"] * 1.4
            pallet_capacity = max(4, round(max_stock_ref / units_per_pallet))

            per_sku_state[sku_id] = {
                "sku_name": series["sku_name"],
                "category": cfg["category"],
                "abc_class": series["abc_class"],
                "xyz_class": series["xyz_class"],
                "storage_mode": cfg["storage_mode"],
                "true_stock": round(true_stock, 1),
                "perceived_stock": perceived,
                "reconciled_stock": result["reconciled_stock"],
                "forecast_demand": result["forecast_demand"],
                "reorder_point": series["reorder_point"],
                "decision": result["decision"],
                "max_stock_ref": max_stock_ref,
                "units_per_pallet": units_per_pallet,
                "pallet_count": round(result["reconciled_stock"] / units_per_pallet, 2),
                "pallet_capacity": pallet_capacity,
            }

        self.current_day = day
        self._latest_state = {
            "day": day,
            "n_days": self.n_days,
            "playing": self.playing,
            "skus": per_sku_state,
            "site_summary": {
                "manual_baseline_fill_rate_pct": self.results["operational_comparison"]["manual_baseline_fill_rate_pct"],
                "digital_twin_fill_rate_pct": self.results["operational_comparison"]["digital_twin_fill_rate_pct"],
                "stockout_days_avoided": self.results["operational_comparison"]["stockout_days_avoided"],
                "forecast_accuracy_improvement_pct": self.results["forecast_accuracy_summary"]["forecast_accuracy_improvement_pct"],
                "data_inconsistency_rate_pct": self.results["data_quality_metrics"]["inconsistency_rate_pct"],
                "digital_twin_total_reorders": self.results["kpi_summary_digital_twin"]["total_reorders_placed"],
                "manual_baseline_total_reorders": self.results["kpi_summary_manual_baseline"]["total_reorders_placed"],
                # KS: reorder count comparison for KPI card
                "digital_twin_total_reorders": self.results["kpi_summary_digital_twin"]["total_reorders_placed"],
                "manual_baseline_total_reorders": self.results["kpi_summary_manual_baseline"]["total_reorders_placed"],
            },
        }

    def tick(self) -> None:
        with self._lock:
            if not self.playing:
                return
            next_day = (self.current_day + 1) % self.n_days
            if next_day == 0:
                self.workflow = AgentWorkflow()
                self._on_order = {sid: 0.0 for sid in self.sku_ids}
            self._advance_to(next_day)

    def state(self) -> dict:
        with self._lock:
            return self._latest_state

    def agent_log(self, n: int = 40) -> list[dict]:
        with self._lock:
            return self.workflow.supervisor.recent(n)

    def control(self, action: str, value=None) -> None:
        with self._lock:
            if action == "play":
                self.playing = True
            elif action == "pause":
                self.playing = False
            elif action == "reset":
                self.workflow = AgentWorkflow()
                self._on_order = {sid: 0.0 for sid in self.sku_ids}
                self._advance_to(0)
            elif action == "speed" and value:
                self.speed = max(0.2, float(value))


def _background_loop(twin: LiveTwin) -> None:
    while True:
        time.sleep(twin.speed)
        twin.tick()


with open(DASHBOARD_DATA_PATH, "r", encoding="utf-8") as f:
    _data = json.load(f)

twin = LiveTwin(_data)
threading.Thread(target=_background_loop, args=(twin,), daemon=True).start()


@app.get("/")
def index():
    return send_from_directory(ROOT / "web", "live_dashboard.html")


@app.get("/web/<path:filename>")
def web_assets(filename):
    return send_from_directory(ROOT / "web", filename)


@app.get("/api/state")
def api_state():
    return jsonify(twin.state())


@app.get("/api/agent-log")
def api_agent_log():
    return jsonify(twin.agent_log())





@app.post("/api/control")
def api_control():
    body = request.get_json(force=True) or {}
    twin.control(body.get("action"), body.get("value"))
    return jsonify({"ok": True, "playing": twin.playing, "speed": twin.speed})


@app.get("/api/sku-series")
def api_sku_series():
    """KS: 30-day DT vs Manual stock trajectory for the stock chart panel."""
    t = _get_twin(request)
    sku_id = request.args.get("sku") or t.sku_ids[0]
    series = t.per_sku.get(sku_id)
    if series is None:
        return jsonify({"error": f"Unknown SKU: {sku_id}"}), 404
    return jsonify({
        "sku_id": sku_id,
        "sku_name": series["sku_name"],
        "days": series["days"],
        "dt_stock": series["digital_twin_stock"],
        "manual_stock": series["manual_baseline_perceived_stock"],
        "reorder_point": series["reorder_point"],
        "demand": series["demand"],
    })


@app.get("/api/audit")
def api_audit():
    """Annotated fragmented spreadsheet data for the Audit Heatmap (#1)."""
    import pandas as pd
    frag_path = ROOT / "data" / "fragmented_spreadsheet_sim.csv"
    clean_path = ROOT / "data" / "clean_ground_truth.csv"
    if not frag_path.exists():
        return jsonify({"rows": [], "error": "Run: py run_pilot.py"})

    frag_df = pd.read_csv(frag_path)
    clean_df = pd.read_csv(clean_path)

    clean_lookup = {
        (int(r["day"]), r["sku_id"]): r["stock_level"]
        for _, r in clean_df.iterrows()
    }
    dup_counts = frag_df.groupby([frag_df["day"].astype(int), frag_df["sku_id"]]).size().to_dict()

    rows = []
    for _, r in frag_df.iterrows():
        reported = None if (hasattr(r["stock_level"], "__float__") and
                            __import__("math").isnan(float(r["stock_level"]))) \
                   else float(r["stock_level"])
        try:
            reported = float(r["stock_level"])
            import math
            if math.isnan(reported):
                reported = None
        except (TypeError, ValueError):
            reported = None

        true_stock = clean_lookup.get((int(r["true_day"]), r["sku_id"]))
        is_dup = dup_counts.get((int(r["day"]), r["sku_id"]), 1) > 1

        if reported is None:
            issue = "missing"
        elif true_stock is not None and abs(reported - true_stock) > 0.5:
            issue = "error"
        elif is_dup:
            issue = "duplicate"
        else:
            issue = "ok"

        rows.append({
            "sku_id": r["sku_id"],
            "reported_day": int(r["day"]),
            "true_day": int(r["true_day"]),
            "stock_level": round(reported) if reported is not None else None,
            "true_stock": round(true_stock) if true_stock is not None else None,
            "source_sheet": str(r.get("source_sheet", "")),
            "issue": issue,
        })

    order = {"error": 0, "missing": 1, "duplicate": 2, "ok": 3}
    rows.sort(key=lambda x: (order.get(x["issue"], 9), x["sku_id"], x["true_day"]))
    return jsonify({"rows": rows[:250]})


@app.get("/api/abc-xyz")
def api_abc_xyz():
    """Forecast accuracy grouped by ABC/XYZ class + optional ARIMA summary (#2 & #4)."""
    import pandas as pd
    import yaml
    accuracy_path = ROOT / "outputs" / "forecast_accuracy_comparison.csv"
    if not accuracy_path.exists():
        return jsonify({"error": "Run: py run_pilot.py", "abc": [], "xyz": []})

    acc_df = pd.read_csv(accuracy_path)
    with open(ROOT / "config" / "warehouse_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sku_map = {s["sku_id"]: s for s in cfg["skus"]}
    acc_df["abc_class"] = acc_df["sku_id"].map(lambda x: sku_map.get(x, {}).get("abc_class", "?"))
    acc_df["xyz_class"] = acc_df["sku_id"].map(lambda x: sku_map.get(x, {}).get("xyz_class", "?"))

    def group_stats(df, col):
        out = []
        for cls, grp in df.groupby(col):
            mw = grp["manual_baseline_WMAPE_pct"].mean()
            dw = grp["digital_twin_WMAPE_pct"].mean()
            out.append({
                "class": cls, "sku_count": len(grp),
                "manual_wmape": round(mw, 1), "dt_wmape": round(dw, 1),
                "improvement": round(((mw - dw) / max(mw, 1)) * 100, 1),
            })
        return out

    arima_summary = None
    if "arima_abs_error_total" in acc_df.columns:
        me = acc_df["manual_baseline_abs_error_total"].sum()
        md = acc_df["manual_baseline_demand_total"].sum()
        de = acc_df["digital_twin_abs_error_total"].sum()
        dd = acc_df["digital_twin_demand_total"].sum()
        ae = acc_df["arima_abs_error_total"].sum()
        ad = acc_df["arima_demand_total"].sum()
        mw = me / md * 100 if md else 0
        dw = de / dd * 100 if dd else 0
        aw = ae / ad * 100 if ad else 0
        arima_summary = {
            "manual_wmape": round(mw, 2), "dt_wmape": round(dw, 2),
            "arima_wmape": round(aw, 2),
            "arima_vs_manual": round((mw - aw) / max(mw, 1) * 100, 2),
        }

    return jsonify({
        "abc": group_stats(acc_df, "abc_class"),
        "xyz": group_stats(acc_df, "xyz_class"),
        "arima_summary": arima_summary,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
