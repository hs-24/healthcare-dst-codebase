"""
server.py

Real-time backend for the Healthcare Warehouse Digital Supply Chain
Twin. Runs three independent sites (Site 1/2/3 -- SITE_A/B/C), each
replaying its own 30-day pilot (computed by
build/generate_multi_site_pilot.py into outputs/dashboard_data_<SITE>.json)
one simulated day per tick, on its own background thread, through the
agentic AI workflow (src/digital_twin/agents.py). Serving all three
sites from a single dashboard is the literal "unified oversight"
answer to the Warehousing Infrastructure problem in the report.

Endpoints:
  GET /                       three.js live digital twin dashboard
  GET /api/sites              site list + latest snapshot KPIs (multi-site overview)
  GET /api/state?site=ID      current day's per-SKU twin state + KPIs for one site
  GET /api/agent-log?site=ID  latest agentic workflow events for one site
  POST /api/control           {"site": "SITE_A", "action": "play"|"pause"|"reset"|"speed", "value": ...}

Run:
  python server.py
  then open http://127.0.0.1:5000/
"""

import json
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from digital_twin.agents import AgentWorkflow  # noqa: E402

ROOT = Path(__file__).parent
DASHBOARD_DATA_PATH = ROOT / "outputs" / "dashboard_data.json"

app = Flask(__name__, static_folder=None)


class LiveTwin:
    def __init__(self, data: dict, site_meta: dict | None = None):
        self.data = data
        self.site_meta = site_meta or {}
        self.per_sku = data["per_sku_series"]
        self.results = data["results"]
        self.sku_ids = list(self.per_sku.keys())
        self.n_days = len(self.per_sku[self.sku_ids[0]]["days"])

        cfg_path = ROOT / "config" / "warehouse_config.yaml"
        self.sku_config = self._load_sku_config(cfg_path)

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
            "site_id": self.site_meta.get("site_id"),
            "site_name": self.site_meta.get("site_name"),
            "site_label": self.site_meta.get("label"),
            "skus": per_sku_state,
            "site_summary": {
                "manual_baseline_fill_rate_pct": self.results["operational_comparison"]["manual_baseline_fill_rate_pct"],
                "digital_twin_fill_rate_pct": self.results["operational_comparison"]["digital_twin_fill_rate_pct"],
                "stockout_days_avoided": self.results["operational_comparison"]["stockout_days_avoided"],
                "forecast_accuracy_improvement_pct": self.results["forecast_accuracy_summary"]["forecast_accuracy_improvement_pct"],
                "data_inconsistency_rate_pct": self.results["data_quality_metrics"]["inconsistency_rate_pct"],
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


SITE_META = [
    {"site_id": "SITE_A", "site_name": "Central Distribution Hub", "label": "Site 1"},
    {"site_id": "SITE_B", "site_name": "North Regional Hub", "label": "Site 2"},
    {"site_id": "SITE_C", "site_name": "East Satellite Depot", "label": "Site 3"},
]

twins: dict[str, LiveTwin] = {}
for meta in SITE_META:
    data_path = ROOT / "outputs" / f"dashboard_data_{meta['site_id']}.json"
    if not data_path.exists():
        data_path = DASHBOARD_DATA_PATH  # fallback to the single-site file
    with open(data_path, "r", encoding="utf-8") as f:
        site_data = json.load(f)
    site_twin = LiveTwin(site_data, site_meta=meta)
    twins[meta["site_id"]] = site_twin
    threading.Thread(target=_background_loop, args=(site_twin,), daemon=True).start()


def _get_twin(req) -> LiveTwin:
    site_id = req.args.get("site", "SITE_A")
    return twins.get(site_id, twins["SITE_A"])


@app.get("/")
def index():
    return send_from_directory(ROOT / "web", "live_dashboard.html")


@app.get("/web/<path:filename>")
def web_assets(filename):
    return send_from_directory(ROOT / "web", filename)


@app.get("/api/sites")
def api_sites():
    out = []
    for meta in SITE_META:
        t = twins[meta["site_id"]]
        state = t.state()
        out.append({**meta, "summary": state.get("site_summary"), "day": state.get("day"), "n_days": state.get("n_days")})
    return jsonify(out)


@app.get("/api/state")
def api_state():
    return jsonify(_get_twin(request).state())


@app.get("/api/agent-log")
def api_agent_log():
    return jsonify(_get_twin(request).agent_log())


@app.post("/api/control")
def api_control():
    body = request.get_json(force=True) or {}
    site_id = body.get("site", "SITE_A")
    t = twins.get(site_id, twins["SITE_A"])
    t.control(body.get("action"), body.get("value"))
    return jsonify({"ok": True, "site": site_id, "playing": t.playing, "speed": t.speed})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
