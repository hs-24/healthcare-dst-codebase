"""Live Flask backend for the three-site healthcare digital twin."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from digital_twin.agents import AgentWorkflow  # noqa: E402

DATA_PATH = ROOT / "outputs" / "dashboard_data.json"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
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
        "chat_model": OLLAMA_MODEL,
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


def _compact_site_state(twin: LiveTwin, question: str) -> dict:
    state = twin.state()
    terms = question.lower()
    def classify(sku: dict) -> str:
        if sku["reconciled_stock"] <= sku["reorder_point"]:
            return "below minimum"
        if sku["reconciled_stock"] <= sku["reorder_point"] * 1.35:
            return "watch"
        return "healthy"

    include_manual = any(word in terms for word in ("manual", "perceived", "difference"))
    include_expiry = any(word in terms for word in ("expiry", "expire", "batch", "cold"))
    include_decisions = any(
        word in terms for word in ("agent", "decision", "reorder", "expedite", "why")
    )
    include_pilot_metrics = any(
        word in terms for word in (
            "forecast", "accuracy", "wmape", "fill rate", "capacity", "pilot", "manual"
        )
    )
    sku_items = list(state["skus"].items())
    named_items = [
        item for item in sku_items
        if item[0].lower() in terms or item[1]["sku_name"].lower() in terms
    ]
    if named_items:
        selected_items = named_items
    elif include_expiry:
        selected_items = sorted(
            sku_items, key=lambda item: item[1].get("days_to_expiry") or 999999
        )[:3]
    else:
        selected_items = sorted(
            sku_items,
            key=lambda item: item[1]["reconciled_stock"] / max(item[1]["reorder_point"], 1),
        )[:2]

    inventory = []
    for sku_id, sku in selected_items:
        position = {
            "sku_id": sku_id,
            "sku_name": sku["sku_name"],
            "uom": sku["uom"],
            "storage_mode": sku["storage_mode"],
            "digital_twin_stock": sku["reconciled_stock"],
            "forecast_daily_demand": sku["forecast_demand"],
            "reorder_point": sku["reorder_point"],
            "status": classify(sku),
        }
        if include_manual:
            position["manual_perceived_stock"] = sku["perceived_stock"]
        if include_expiry:
            position.update({
                "batch_number": sku["batch_number"],
                "expiry_date": sku["expiry_date"],
                "days_to_expiry": sku["days_to_expiry"],
            })
        if include_decisions:
            position["decision"] = sku["decision"]
        inventory.append(position)
    compact = {
        "site_id": state["site_id"],
        "site_name": state["site_name"],
        "simulation_day": state["day"],
        "current_risk_summary": {
            "below_minimum_skus": sum(classify(sku) == "below minimum" for sku in state["skus"].values()),
            "watch_skus": sum(classify(sku) == "watch" for sku in state["skus"].values()),
            "healthy_skus": sum(classify(sku) == "healthy" for sku in state["skus"].values()),
        },
        "inventory": inventory,
    }
    if include_pilot_metrics:
        compact["historical_pilot_metrics"] = {
            key: state["site_summary"][key]
            for key in (
                "digital_twin_fill_rate_pct",
                "manual_baseline_fill_rate_pct",
                "stockout_days_avoided",
                "avg_digital_twin_WMAPE_pct",
                "forecast_accuracy_improvement_pct",
                "critical_skus",
                "capacity_utilization_pct",
            )
        }
    return compact


def _chat_context(site_id: str, question: str) -> tuple[dict, list[str]]:
    selected = list(twins.values()) if site_id == "ALL" else [twins[site_id]]
    states = [_compact_site_state(twin, question) for twin in selected]
    terms = question.lower()
    include_events = any(
        word in terms for word in ("agent", "decision", "reorder", "expedite", "why")
    )
    include_transfers = any(
        word in terms for word in ("transfer", "move", "surplus", "shortage", "another site")
    )
    include_alerts = any(word in terms for word in ("alert", "exception", "action queue"))
    alerts = [
        {**alert, "site_id": key, "site_name": site["site_name"]}
        for key, site in dashboard_data["sites"].items()
        if site_id == "ALL" or key == site_id
        for alert in site["alerts"]
    ] if include_alerts else []
    events = [
        event
        for twin in selected
        for event in twin.agent_log(5)
    ] if include_events else []
    transfers = dashboard_data["transfers"] if include_transfers else []
    risk_scores = {
        state["site_id"]: (
            state["current_risk_summary"]["below_minimum_skus"],
            state["current_risk_summary"]["watch_skus"],
        )
        for state in states
    }
    highest_score = max(risk_scores.values())
    highest_risk_sites = [
        site_id for site_id, score in risk_scores.items() if score == highest_score
    ]
    risk_comparison = {
        "ranking_rule": "More below-minimum SKUs ranks first; watch SKUs break ties.",
        "site_scores": {
            site_id: {"below_minimum_skus": score[0], "watch_skus": score[1]}
            for site_id, score in risk_scores.items()
        },
        "highest_risk_sites": highest_risk_sites,
        "result": (
            "No site currently has a below-minimum or watch SKU."
            if highest_score == (0, 0)
            else "Tie between the listed sites."
            if len(highest_risk_sites) > 1
            else f"{highest_risk_sites[0]} has the highest current inventory risk."
        ),
    }
    context = {
        "scope": "All Sites" if site_id == "ALL" else states[0]["site_name"],
        "field_definitions": {
            "digital_twin_stock": "Trusted simulated current stock record for the selected day; not a demand forecast.",
            "manual_perceived_stock": "Stock visible in the simulated manual spreadsheet baseline, which may contain errors.",
            "forecast_daily_demand": "Expected daily consumption; separate from the current stock record.",
            "reorder_point": "Precomputed stock threshold used to trigger replenishment review.",
        },
        "authoritative_current_risk_comparison": risk_comparison,
        "sites": states,
        "active_alerts": alerts,
        "agent_events": events,
        "cross_site_transfer_recommendations": transfers,
    }
    evidence = [f"Live digital-twin state: {context['scope']}"]
    if alerts:
        evidence.append(f"Active alert records: {len(alerts)}")
    if events:
        evidence.append(f"Recent agent events: {len(events)}")
    if transfers:
        evidence.append(f"Documented pilot transfer recommendations: {len(transfers)}")
    return context, evidence


def _verified_chat_answer(question: str, context: dict) -> str | None:
    terms = question.lower()
    transfers = context["cross_site_transfer_recommendations"]
    if transfers:
        lines = ["Documented pilot transfer recommendations:"]
        lines.extend(
            f"- {item['sku_id']} ({item['sku_name']}): {item['quantity']} {item['uom']} "
            f"from {item['source_name']} to {item['destination_name']}."
            for item in transfers
        )
        return "\n".join(lines)

    events = context["agent_events"]
    if events and any(term in terms for term in ("agent", "decision", "latest", "expedite")):
        if "latest" in terms:
            events = events[:1]
            lines = ["Latest recorded agent event:"]
        else:
            lines = ["Recent recorded agent events:"]
        lines.extend(
            f"- {event['timestamp']} | {event['site_id']} | {event['agent']} | "
            f"{event['sku_id']} | {event['severity']}: {event['message']}"
            for event in events
        )
        return "\n".join(lines)

    risk_terms = ("risk", "critical", "below minimum", "watch", "stockout")
    if any(term in terms for term in risk_terms):
        comparison = context["authoritative_current_risk_comparison"]
        lines = [comparison["result"]]
        lines.extend(
            f"- {site_id}: {score['below_minimum_skus']} below minimum, "
            f"{score['watch_skus']} watch."
            for site_id, score in comparison["site_scores"].items()
        )
        return "\n".join(lines)
    return None


@app.post("/api/chat")
def api_chat():
    body = request.get_json(silent=True) or {}
    message = str(body.get("message", "")).strip()
    site_id = str(body.get("site", "ALL")).upper()
    if not message:
        return jsonify({"error": "Message is required."}), 400
    if len(message) > 600:
        return jsonify({"error": "Message must be 600 characters or fewer."}), 400
    if site_id != "ALL" and site_id not in twins:
        return jsonify({"error": "Unknown site."}), 404

    history = []
    submitted_history = body.get("history", [])
    if not isinstance(submitted_history, list):
        submitted_history = []
    for item in submitted_history[-6:]:
        role = item.get("role") if isinstance(item, dict) else None
        content = str(item.get("content", ""))[:1200] if isinstance(item, dict) else ""
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})

    context, evidence = _chat_context(site_id, message)
    verified_answer = _verified_chat_answer(message, context)
    if verified_answer:
        return jsonify({
            "answer": verified_answer,
            "evidence": [*evidence, "Verbatim or Python-verified dashboard result"],
            "model": OLLAMA_MODEL,
            "response_mode": "verified dashboard calculation",
        })

    system_prompt = """You are a read-only healthcare supply-chain inventory assistant.
Answer only from the DIGITAL_TWIN_CONTEXT supplied below. Explain the recorded values
and operational recommendations clearly and concisely. Never invent quantities, dates,
sites, SKUs, alerts, forecasts, or agent actions. If the evidence is insufficient, say so.
Do not provide clinical advice. Do not claim to execute orders, transfers, or data changes.
Distinguish digital-twin stock from manual perceived stock and simulation results from
real-world outcomes. The status and current_risk_summary fields are calculated by Python:
copy them exactly and never recalculate or contradict them. Do not infer clinical or patient
safety consequences. For site-risk comparisons, repeat the result in
authoritative_current_risk_comparison exactly. If that result says no site has a risk, return
only that sentence. Otherwise, add at most one sentence using its site_scores. Use plain text
and short bullet points when useful.

DIGITAL_TWIN_CONTEXT:
""" + json.dumps(context, separators=(",", ":"), ensure_ascii=True)
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": message},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 180},
    }
    ollama_request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(ollama_request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
        answer = str(result.get("message", {}).get("content", "")).strip()
        if not answer:
            raise ValueError("Ollama returned an empty answer")
        return jsonify({"answer": answer, "evidence": evidence, "model": OLLAMA_MODEL})
    except HTTPError as exc:
        return jsonify({"error": f"Ollama rejected the request ({exc.code})."}), 502
    except (URLError, TimeoutError):
        return jsonify({
            "error": "Local AI is unavailable. Start Ollama and confirm the model is installed."
        }), 503
    except (ValueError, json.JSONDecodeError):
        return jsonify({"error": "Ollama returned an invalid response."}), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
