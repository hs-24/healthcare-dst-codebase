"""
agents.py

Agentic AI workflow automation for the Healthcare Warehouse Digital
Twin. This module is the direct technical answer to the Section 1
problems in the report:

  - Data Management -> DataValidationAgent
    Reconciles a fragmented/manual spreadsheet reading against the
    last trusted figure and flags duplicates, blanks and out-of-range
    values *before* they reach a decision-maker.

  - Forecasting & Reporting -> ForecastingAgent
    Maintains a live exponential-smoothing demand forecast per SKU so
    forecasts update every tick instead of a periodic manual Excel run.

  - Warehousing Infrastructure -> ReplenishmentAgent
    Makes the reorder/expedite decision autonomously from the
    validated (not perceived) stock figure, removing the lag that
    causes the stockouts evidenced in run_pilot.py.

  - Unified oversight -> SupervisorAgent
    Aggregates the three agents above into one event log and severity
    feed -- the "single pane of glass" the report calls for, instead
    of decisions being scattered across spreadsheets and sites.

Each agent is a small, stateless-per-call class with an explicit
`run()` method, chained by `AgentWorkflow.process_tick()`. No external
LLM call is required for the pilot scope (Section 3: out-of-scope --
third-party integrations); the "agentic" property is the autonomous,
multi-step perceive -> validate -> decide -> report loop running
without a human in it, which is what the SMART objectives measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class AgentEvent:
    timestamp: str
    agent: str
    sku_id: str
    severity: str  # info | warning | critical
    message: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "agent": self.agent,
            "sku_id": self.sku_id,
            "severity": self.severity,
            "message": self.message,
        }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


class DataValidationAgent:
    """
    Reconciles an incoming "spreadsheet" stock reading against the
    last validated figure for that SKU. Flags missing entries and
    readings that imply an implausible day-over-day swing (duplicate
    rows, fat-finger transcription, stale copies from another sheet)
    -- the exact failure modes described under Data Management in
    Section 1.
    """

    name = "DataValidationAgent"
    MAX_PLAUSIBLE_SWING = 0.35  # ±35% day-over-day is treated as suspect

    def __init__(self):
        self._last_validated: dict[str, float] = {}

    def run(self, sku_id: str, perceived_reading: Optional[float],
            true_stock: float) -> tuple[float, list[AgentEvent]]:
        events: list[AgentEvent] = []
        last_good = self._last_validated.get(sku_id, true_stock)

        if perceived_reading is None:
            events.append(AgentEvent(
                _now(), self.name, sku_id, "warning",
                f"Missing spreadsheet entry for {sku_id} — holding last "
                f"validated figure ({last_good:.0f} units) instead of "
                f"blanking the record.",
            ))
            reconciled = last_good
        else:
            swing = abs(perceived_reading - last_good) / max(last_good, 1.0)
            if swing > self.MAX_PLAUSIBLE_SWING:
                events.append(AgentEvent(
                    _now(), self.name, sku_id, "critical",
                    f"Rejected implausible reading for {sku_id}: "
                    f"{perceived_reading:.0f} units is a {swing*100:.0f}% "
                    f"swing from last validated figure. Falling back to "
                    f"true on-hand stock ({true_stock:.0f}).",
                ))
                reconciled = true_stock
            else:
                reconciled = perceived_reading

        self._last_validated[sku_id] = reconciled
        return reconciled, events


class ForecastingAgent:
    """
    Live exponential-smoothing demand forecast, replacing the manual,
    periodic Excel/ETS forecasting run described in Section 1. Updated
    every tick rather than on whatever cadence a human last opened the
    spreadsheet.
    """

    name = "ForecastingAgent"
    ALPHA = 0.3

    def __init__(self):
        self._forecast: dict[str, float] = {}

    def run(self, sku_id: str, observed_demand: float) -> tuple[float, list[AgentEvent]]:
        prev = self._forecast.get(sku_id, observed_demand)
        forecast = self.ALPHA * observed_demand + (1 - self.ALPHA) * prev
        self._forecast[sku_id] = forecast

        events: list[AgentEvent] = []
        if prev > 0 and abs(observed_demand - prev) / prev > 0.6:
            events.append(AgentEvent(
                _now(), self.name, sku_id, "warning",
                f"Demand for {sku_id} shifted sharply ({observed_demand:.1f} "
                f"vs forecast {prev:.1f}) — forecast updated to "
                f"{forecast:.1f}/day.",
            ))
        return forecast, events


class ReplenishmentAgent:
    """
    Autonomous reorder/expedite decision made from the *validated*
    stock figure and the live demand forecast, instead of the lagged
    perceived-stock figure that drives the manual baseline in
    simulation.py. Directly automates the Warehousing Infrastructure
    decision step called out in Section 1.
    """

    name = "ReplenishmentAgent"

    def run(self, sku_id: str, reconciled_stock: float, on_order: float,
            forecast_demand: float, reorder_point: float,
            reorder_qty: float, lead_time_days: int) -> tuple[Optional[dict], list[AgentEvent]]:
        events: list[AgentEvent] = []
        position = reconciled_stock + on_order
        days_of_cover = reconciled_stock / forecast_demand if forecast_demand > 0 else float("inf")

        decision = None
        if position <= reorder_point:
            decision = {"action": "REORDER", "qty": reorder_qty}
            events.append(AgentEvent(
                _now(), self.name, sku_id, "info",
                f"Autonomous reorder placed for {sku_id}: {reorder_qty:.0f} "
                f"units (stock position {position:.0f} <= reorder point "
                f"{reorder_point:.0f}).",
            ))
        elif days_of_cover < lead_time_days:
            decision = {"action": "EXPEDITE_WATCH", "qty": 0}
            events.append(AgentEvent(
                _now(), self.name, sku_id, "warning",
                f"{sku_id} projected to run out in {days_of_cover:.1f} days, "
                f"shorter than the {lead_time_days}-day lead time — "
                f"flagged for expedite review.",
            ))
        return decision, events


class SupervisorAgent:
    """
    Aggregates the per-SKU outputs of the other agents into a single
    feed and severity ranking -- the unified-oversight "single pane of
    glass" the report's Warehousing Infrastructure section says is
    currently missing across sites.
    """

    name = "SupervisorAgent"
    MAX_LOG = 200

    def __init__(self):
        self.log: list[AgentEvent] = []

    def ingest(self, events: list[AgentEvent]) -> None:
        self.log.extend(events)
        if len(self.log) > self.MAX_LOG:
            self.log = self.log[-self.MAX_LOG:]

    def recent(self, n: int = 30) -> list[dict]:
        return [e.to_dict() for e in self.log[-n:][::-1]]


@dataclass
class AgentWorkflow:
    """
    Orchestrates the perceive -> validate -> forecast -> decide ->
    report pipeline for one warehouse, one tick (day) at a time.
    """

    validator: DataValidationAgent = field(default_factory=DataValidationAgent)
    forecaster: ForecastingAgent = field(default_factory=ForecastingAgent)
    replenisher: ReplenishmentAgent = field(default_factory=ReplenishmentAgent)
    supervisor: SupervisorAgent = field(default_factory=SupervisorAgent)

    def process_tick(self, sku_id: str, *, perceived_reading: Optional[float],
                      true_stock: float, on_order: float, observed_demand: float,
                      reorder_point: float, reorder_qty: float,
                      lead_time_days: int) -> dict:
        events: list[AgentEvent] = []

        reconciled_stock, ev = self.validator.run(sku_id, perceived_reading, true_stock)
        events += ev

        forecast_demand, ev = self.forecaster.run(sku_id, observed_demand)
        events += ev

        decision, ev = self.replenisher.run(
            sku_id, reconciled_stock, on_order, forecast_demand,
            reorder_point, reorder_qty, lead_time_days,
        )
        events += ev

        if not events:
            events.append(AgentEvent(
                _now(), self.supervisor.name, sku_id, "info",
                f"{sku_id} nominal — stock {reconciled_stock:.0f} units, "
                f"forecast demand {forecast_demand:.1f}/day.",
            ))

        self.supervisor.ingest(events)

        return {
            "sku_id": sku_id,
            "reconciled_stock": round(reconciled_stock, 1),
            "forecast_demand": round(forecast_demand, 2),
            "decision": decision,
        }
