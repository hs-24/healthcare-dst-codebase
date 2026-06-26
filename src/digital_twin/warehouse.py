"""
warehouse.py

Core domain model for the Healthcare Warehouse Digital Twin.

Design pattern adapted from AndreVale69/simulator-automatic-warehouse:
that project models a physical Automatic Vertical Storage System as a
`Warehouse` object containing bays/trays/items, driven by simpy events.

Here, the same idea is applied to inventory flow instead of physical
storage slots: a `Warehouse` holds a set of `SKU` (stock-keeping unit)
objects, each with its own stock level, reorder logic and demand
profile. The simulation engine (simulation.py) drives daily demand
and replenishment events against this model, exactly as the reference
project drives pick/store events against its bay model.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SKU:
    """A single stock-keeping unit tracked by the warehouse."""

    sku_id: str
    name: str
    category: str
    abc_class: str
    xyz_class: str
    stock: float
    reorder_point: float
    reorder_qty: float
    lead_time_days: int
    demand_mean: float
    demand_std: float

    # Tracking state
    on_order: float = 0.0
    pending_arrivals: list = field(default_factory=list)  # list of (arrival_day, qty)
    stockout_days: int = 0
    total_demand: float = 0.0
    total_fulfilled: float = 0.0

    def draw_daily_demand(self, rng: np.random.Generator) -> float:
        """Sample a day's demand from a normal distribution, floored at 0."""
        demand = rng.normal(self.demand_mean, self.demand_std)
        return max(0.0, demand)

    def receive_arrivals(self, current_day: int) -> None:
        """Move any arrived purchase orders into on-hand stock."""
        still_pending = []
        for arrival_day, qty in self.pending_arrivals:
            if arrival_day <= current_day:
                self.stock += qty
                self.on_order -= qty
            else:
                still_pending.append((arrival_day, qty))
        self.pending_arrivals = still_pending

    def fulfill_demand(self, demand: float) -> float:
        """
        Attempt to fulfill demand from stock.
        Returns the amount actually fulfilled (less than demand = stockout).
        """
        self.total_demand += demand
        fulfilled = min(self.stock, demand)
        self.stock -= fulfilled
        self.total_fulfilled += fulfilled
        if fulfilled < demand:
            self.stockout_days += 1
        return fulfilled

    def check_and_place_reorder(self, current_day: int) -> Optional[float]:
        """
        If stock + on_order has fallen to/below the reorder point,
        place a replenishment order. Returns the order quantity placed,
        or None if no order was needed.
        """
        if (self.stock + self.on_order) <= self.reorder_point:
            self.on_order += self.reorder_qty
            arrival_day = current_day + self.lead_time_days
            self.pending_arrivals.append((arrival_day, self.reorder_qty))
            return self.reorder_qty
        return None

    def check_and_place_reorder_from_perceived_stock(
        self, current_day: int, perceived_stock: Optional[float]
    ) -> Optional[float]:
        """
        Manual-baseline variant of check_and_place_reorder: decision is
        based on a *perceived* stock figure (the latest available
        fragmented spreadsheet reading) rather than the true on-hand
        stock. This recreates the real-world failure mode described in
        the problem statement: staff trust a number that may be stale,
        duplicated, or transcribed incorrectly, and a reorder is missed
        or delayed if that perceived figure looks healthier than reality.

        If no reading is available yet (perceived_stock is None, e.g.
        early in the simulation before the first lagged report lands),
        falls back to true stock so the warehouse isn't flying blind on
        day zero -- a real site would still have *some* starting count.
        """
        reference_level = perceived_stock if perceived_stock is not None else self.stock
        if (reference_level + self.on_order) <= self.reorder_point:
            self.on_order += self.reorder_qty
            arrival_day = current_day + self.lead_time_days
            self.pending_arrivals.append((arrival_day, self.reorder_qty))
            return self.reorder_qty
        return None

    @property
    def fill_rate(self) -> float:
        """Service-level KPI: % of demand fulfilled directly from stock."""
        if self.total_demand == 0:
            return 1.0
        return self.total_fulfilled / self.total_demand


class Warehouse:
    """
    Represents a single warehousing site holding multiple SKUs.

    Mirrors the role of the `Warehouse` class in the reference repo,
    but the "slots" being managed are inventory positions rather than
    physical storage bays.
    """

    def __init__(self, site_id: str, site_name: str, skus: list[SKU]):
        self.site_id = site_id
        self.site_name = site_name
        self.skus: dict[str, SKU] = {sku.sku_id: sku for sku in skus}

    def get_sku(self, sku_id: str) -> SKU:
        return self.skus[sku_id]

    def all_skus(self) -> list[SKU]:
        return list(self.skus.values())

    def site_fill_rate(self) -> float:
        """Aggregate fill rate across all SKUs at this site."""
        total_demand = sum(s.total_demand for s in self.skus.values())
        total_fulfilled = sum(s.total_fulfilled for s in self.skus.values())
        if total_demand == 0:
            return 1.0
        return total_fulfilled / total_demand
