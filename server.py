"""Live Flask backend for the three-site healthcare digital twin."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from digital_twin.agents import AgentWorkflow  # noqa: E402

DATA_PATH = ROOT / "outputs" / "dashboard_data.json"
app = Flask(__name__, static_folder=None)


class LiveTwin:
    def __init__(self, site_data: dict):
        self.site_id = site_data["site_id"]
        self.site_name = site_data["site_name"]
        self.summary = site_data["summary"]
        self.per_sku = site_data["per_sku_series"]
        self.sku_ids = list(self.per_sku)
        self.n_days = len(self.per_sku[self.sku_ids[0]]["days"])
        self.workflow = AgentWorkflow()
        self.current_day = 0
        self.playing = True
        self.speed = 1.5
        self._lock = threading.Lock()
        self._on_order = {sku_id: 0.0 for sku_id in self.sku_ids}
        self._latest_state = {}
        self._advance_to(0)

    def _advance_to(self, day: int) -> None:
        sku_state = {}
        for sku_id, series in self.per_sku.items():
            true_stock = series["digital_twin_stock"][day]
            perceived = series["manual_baseline_perceived_stock"][day]
            demand = series["demand"][day]
            reorder_point = series["reorder_point"]
            reorder_qty = max(reorder_point, demand * max(1, series.get("lead_time_days", 7)))
            result = self.workflow.process_tick(
                sku_id,
                perceived_reading=perceived,
                true_stock=true_stock,
                on_order=self._on_order[sku_id],
                observed_demand=demand,
                reorder_point=reorder_point,
                reorder_qty=reorder_qty,
                lead_time_days=7,
            )
            if result["decision"] and result["decision"]["action"] == "REORDER":
                self._on_order[sku_id] += result["decision"]["qty"]
            units_per_pallet = max(1, series["units_per_pallet"])
            sku_state[sku_id] = {
                "sku_name": series["sku_name"],
                "category": series["category"],
                "abc_class": series["abc_class"],
                "xyz_class": series["xyz_class"],
                "uom": series["uom"],
                "storage_mode": series["storage_mode"],
                "batch_number": series["batch_number"],
                "expiry_date": series["expiry_date"],
                "days_to_expiry": series["days_to_expiry"],
                "vendor_id": series["vendor_id"],
                "true_stock": round(true_stock, 1),
                "perceived_stock": round(perceived, 1) if perceived is not None else None,
                "reconciled_stock": round(result["reconciled_stock"], 1),
                "forecast_demand": round(result["forecast_demand"], 1),
                "reorder_point": reorder_point,
                "decision": result["decision"],
                "max_capacity": series["max_capacity"],
                "units_per_pallet": units_per_pallet,
                "pallet_count": round(result["reconciled_stock"] / units_per_pallet, 1),
                "pallet_capacity": 12,
                "source_dataset": series["source_dataset"],
                "derived_fields": series["derived_fields"],
            }
        self.current_day = day
        self._latest_state = {
            "site_id": self.site_id, "site_name": self.site_name,
            "day": day, "n_days": self.n_days, "playing": self.playing,
            "skus": sku_state, "site_summary": self.summary,
        }

    def tick(self):
        with self._lock:
            if self.playing:
                next_day = (self.current_day + 1) % self.n_days
                if next_day == 0:
                    self.workflow = AgentWorkflow()
                    self._on_order = {sku_id: 0.0 for sku_id in self.sku_ids}
                self._advance_to(next_day)

    def state(self):
        with self._lock:
            return self._latest_state

    def agent_log(self, count=50):
        with self._lock:
            events = self.workflow.supervisor.recent(count)
            for event in events:
                event["site_id"] = self.site_id
            return events

    def control(self, action, value=None):
        with self._lock:
            if action == "play":
                self.playing = True
            elif action == "pause":
                self.playing = False
            elif action == "reset":
                self.workflow = AgentWorkflow()
                self._on_order = {sku_id: 0.0 for sku_id in self.sku_ids}
                self._advance_to(0)
            elif action == "speed" and value is not None:
                self.speed = max(0.2, float(value))


with DATA_PATH.open(encoding="utf-8") as handle:
    dashboard_data = json.load(handle)

twins = {site_id: LiveTwin(site) for site_id, site in dashboard_data["sites"].items()}


def _background_loop():
    while True:
        time.sleep(0.2)
        now = time.monotonic()
        for twin in twins.values():
            last = getattr(twin, "_last_tick", 0.0)
            if now - last >= twin.speed:
                twin.tick()
                twin._last_tick = now


threading.Thread(target=_background_loop, daemon=True).start()


@app.get("/")
def index():
    return send_from_directory(ROOT / "web", "live_dashboard.html")


@app.get("/web/<path:filename>")
def web_assets(filename):
    return send_from_directory(ROOT / "web", filename)


@app.get("/api/sites")
def api_sites():
    return jsonify({
        "sites": dashboard_data["site_list"],
        "network_summary": dashboard_data["network_summary"],
        "source": dashboard_data["source"],
    })


@app.get("/api/state")
def api_state():
    site_id = request.args.get("site", "SITE_A").upper()
    if site_id == "ALL":
        return jsonify({
            "site_id": "ALL", "site_name": "All Sites",
            "network_summary": dashboard_data["network_summary"],
            "sites": [twin.state() for twin in twins.values()],
            "transfers": dashboard_data["transfers"],
        })
    twin = twins.get(site_id)
    return jsonify(twin.state()) if twin else (jsonify({"error": "Unknown site"}), 404)


@app.get("/api/agent-log")
def api_agent_log():
    site_id = request.args.get("site", "SITE_A").upper()
    if site_id == "ALL":
        events = [event for twin in twins.values() for event in twin.agent_log(20)]
        return jsonify(events[-60:])
    twin = twins.get(site_id)
    return jsonify(twin.agent_log()) if twin else (jsonify({"error": "Unknown site"}), 404)


@app.get("/api/sku-series")
def api_sku_series():
    site_id = request.args.get("site", "SITE_A").upper()
    sku_id = request.args.get("sku", "")
    site = dashboard_data["sites"].get(site_id)
    if not site or sku_id not in site["per_sku_series"]:
        return jsonify({"error": "Unknown site or SKU"}), 404
    return jsonify(site["per_sku_series"][sku_id])


@app.get("/api/alerts")
def api_alerts():
    site_id = request.args.get("site", "ALL").upper()
    if site_id == "ALL":
        return jsonify([
            {**alert, "site_id": key, "site_name": site["site_name"]}
            for key, site in dashboard_data["sites"].items()
            for alert in site["alerts"]
        ])
    site = dashboard_data["sites"].get(site_id)
    return jsonify(site["alerts"]) if site else (jsonify({"error": "Unknown site"}), 404)


@app.get("/api/transfers")
def api_transfers():
    return jsonify(dashboard_data["transfers"])


@app.post("/api/control")
def api_control():
    body = request.get_json(force=True) or {}
    site_id = str(body.get("site", "ALL")).upper()
    targets = twins.values() if site_id == "ALL" else [twins[site_id]] if site_id in twins else []
    if not targets:
        return jsonify({"error": "Unknown site"}), 404
    for twin in targets:
        twin.control(body.get("action"), body.get("value"))
    return jsonify({"ok": True, "site": site_id})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
