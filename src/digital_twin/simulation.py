"""
simulation.py

Discrete-event simulation engine for the Healthcare Warehouse Digital
Twin, built on `simpy` -- the same event-driven simulation library
used by the reference repo (AndreVale69/simulator-automatic-warehouse).

The reference project's `WarehouseSimulation.run_simulation()` steps
a physical Warehouse object through pick/store events and returns a
`get_store_history_dataframe()`. This module follows the identical
pattern, but the events being simulated are daily demand draws and
replenishment cycles for a healthcare inventory, rather than physical
storage actions.
"""

import simpy
import numpy as np
import pandas as pd
from typing import Optional

from digital_twin.warehouse import Warehouse


class ManualBaselineSimulation:
    """
    Replays the *exact same* daily demand sequence as a corresponding
    WarehouseSimulation run, but reorder decisions are made using a
    corrupted, lagged "perceived stock" figure instead of true on-hand
    stock -- recreating the operational consequence of the fragmented
    spreadsheet problem (Section 1), rather than only its reporting
    symptom.

    This is what makes the before/after comparison fair: both runs
    face identical demand draws, so any difference in stockouts or
    fill rate is attributable purely to data visibility, not to
    random chance in the demand sequence.
    """

    def __init__(self, warehouse: Warehouse, duration_days: int,
                 random_seed: int, time_step_hours: int,
                 reporting_lag_days: int, manual_entry_error_rate: float,
                 missing_entry_rate: float):
        self.warehouse = warehouse
        self.duration_days = duration_days
        self.time_step_hours = time_step_hours
        # Same seed as the clean run ensures identical demand draws,
        # since both simulations call draw_daily_demand() in the same
        # per-SKU order on each day.
        self.rng = np.random.default_rng(random_seed)
        # Separate RNG stream for the data-corruption noise itself, so
        # corruption noise doesn't consume draws from the demand stream
        # and silently desynchronise the two simulations.
        self.corruption_rng = np.random.default_rng(random_seed + 1000)
        self.env = simpy.Environment()
        self.history_records = []
        self.reorder_log = []

        self.reporting_lag_days = reporting_lag_days
        self.manual_entry_error_rate = manual_entry_error_rate
        self.missing_entry_rate = missing_entry_rate

        # Per-SKU buffer of (report_day, perceived_stock) the manual
        # process has "on file" -- simulates a lagged, occasionally
        # wrong or missing spreadsheet reading.
        self._last_known_perceived: dict[str, Optional[float]] = {
            sku.sku_id: sku.stock for sku in warehouse.all_skus()
        }
        self._pending_reports: dict[str, list] = {sku.sku_id: [] for sku in warehouse.all_skus()}

    def _generate_perceived_reading(self, true_stock: float) -> Optional[float]:
        """Corrupt a true stock reading the way a manual spreadsheet would."""
        if self.corruption_rng.random() < self.missing_entry_rate:
            return None  # blank cell -- no usable reading today
        if self.corruption_rng.random() < self.manual_entry_error_rate:
            factor = self.corruption_rng.uniform(0.5, 1.6)
            return round(true_stock * factor)
        return true_stock

    def _daily_process(self):
        for day in range(self.duration_days):
            for sku in self.warehouse.all_skus():
                sku_id = sku.sku_id

                # 1. Receive any arrived purchase orders (physical stock
                #    arrives regardless of what the spreadsheet says)
                sku.receive_arrivals(day)

                # 2. Today's true stock count gets generated and queued
                #    to "arrive" in the perceived-stock buffer after the
                #    configured reporting lag.
                reading = self._generate_perceived_reading(sku.stock)
                if reading is not None:
                    self._pending_reports[sku_id].append(
                        (day + self.reporting_lag_days, reading)
                    )

                # Pull in any reports that have "landed" by today
                still_pending = []
                for report_day, value in self._pending_reports[sku_id]:
                    if report_day <= day:
                        self._last_known_perceived[sku_id] = value
                    else:
                        still_pending.append((report_day, value))
                self._pending_reports[sku_id] = still_pending

                # 3. Draw and fulfil today's demand (identical RNG
                #    sequence to the clean simulation)
                demand = sku.draw_daily_demand(self.rng)
                fulfilled = sku.fulfill_demand(demand)
                stockout = fulfilled < demand

                # 4. Reorder decision based on perceived (not true) stock
                order_placed = sku.check_and_place_reorder_from_perceived_stock(
                    day, self._last_known_perceived[sku_id]
                )
                if order_placed:
                    self.reorder_log.append({
                        "day": day,
                        "site_id": self.warehouse.site_id,
                        "sku_id": sku_id,
                        "order_qty": order_placed,
                        "expected_arrival_day": day + sku.lead_time_days,
                    })

                self.history_records.append({
                    "day": day,
                    "site_id": self.warehouse.site_id,
                    "sku_id": sku_id,
                    "sku_name": sku.name,
                    "abc_class": sku.abc_class,
                    "xyz_class": sku.xyz_class,
                    "true_stock_level": sku.stock,
                    "perceived_stock_level": self._last_known_perceived[sku_id],
                    "on_order": sku.on_order,
                    "demand": round(demand, 2),
                    "fulfilled": round(fulfilled, 2),
                    "stockout": stockout,
                    "reorder_point": sku.reorder_point,
                })

            yield self.env.timeout(self.time_step_hours)

    def run_simulation(self) -> None:
        self.env.process(self._daily_process())
        self.env.run(until=self.duration_days * self.time_step_hours)

    def get_store_history_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history_records)

    def get_reorder_log_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.reorder_log)

    def get_kpi_summary(self) -> dict:
        summary = []
        for sku in self.warehouse.all_skus():
            summary.append({
                "sku_id": sku.sku_id,
                "sku_name": sku.name,
                "abc_class": sku.abc_class,
                "xyz_class": sku.xyz_class,
                "fill_rate_pct": round(sku.fill_rate * 100, 2),
                "stockout_days": sku.stockout_days,
                "total_demand": round(sku.total_demand, 1),
                "total_fulfilled": round(sku.total_fulfilled, 1),
            })
        return {
            "per_sku": summary,
            "site_fill_rate_pct": round(self.warehouse.site_fill_rate() * 100, 2),
            "total_stockout_days": sum(s.stockout_days for s in self.warehouse.all_skus()),
            "total_reorders_placed": len(self.reorder_log),
        }


class WarehouseSimulation:
    """
    Drives a Warehouse object through a multi-day simulation using a
    simpy environment. Each simulated day:
      1. Receives any purchase orders that have arrived
      2. Draws daily demand for every SKU
      3. Fulfils demand from stock (recording stockouts where relevant)
      4. Checks reorder points and places new purchase orders as needed
      5. Records the day's state to history
    """

    def __init__(self, warehouse: Warehouse, duration_days: int,
                 random_seed: int = 42, time_step_hours: int = 24):
        self.warehouse = warehouse
        self.duration_days = duration_days
        self.time_step_hours = time_step_hours
        self.rng = np.random.default_rng(random_seed)
        self.env = simpy.Environment()
        self.history_records = []
        self.reorder_log = []

    def _daily_process(self):
        """The simpy process generator: one iteration per simulated day."""
        for day in range(self.duration_days):
            for sku in self.warehouse.all_skus():
                # 1. Receive any arrived purchase orders
                sku.receive_arrivals(day)

                # 2. Draw and fulfil today's demand
                demand = sku.draw_daily_demand(self.rng)
                fulfilled = sku.fulfill_demand(demand)
                stockout = fulfilled < demand

                # 3. Check reorder point, place PO if needed
                order_placed = sku.check_and_place_reorder(day)
                if order_placed:
                    self.reorder_log.append({
                        "day": day,
                        "site_id": self.warehouse.site_id,
                        "sku_id": sku.sku_id,
                        "order_qty": order_placed,
                        "expected_arrival_day": day + sku.lead_time_days,
                    })

                # 4. Record ground-truth state for this SKU/day
                self.history_records.append({
                    "day": day,
                    "site_id": self.warehouse.site_id,
                    "sku_id": sku.sku_id,
                    "sku_name": sku.name,
                    "abc_class": sku.abc_class,
                    "xyz_class": sku.xyz_class,
                    "stock_level": sku.stock,
                    "on_order": sku.on_order,
                    "demand": round(demand, 2),
                    "fulfilled": round(fulfilled, 2),
                    "stockout": stockout,
                    "reorder_point": sku.reorder_point,
                })

            # advance simpy clock by one simulated day
            yield self.env.timeout(self.time_step_hours)

    def run_simulation(self) -> None:
        """Start and run the simpy environment to completion."""
        self.env.process(self._daily_process())
        self.env.run(until=self.duration_days * self.time_step_hours)

    def get_store_history_dataframe(self) -> pd.DataFrame:
        """
        Returns the clean, ground-truth simulation history as a
        DataFrame -- equivalent in role to the reference repo's
        get_store_history_dataframe(), but tracking inventory state
        rather than physical bay contents.
        """
        return pd.DataFrame(self.history_records)

    def get_reorder_log_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.reorder_log)

    def get_kpi_summary(self) -> dict:
        """
        Aggregate pilot-scope KPIs referenced in the SMART objectives:
        fill rate, stockout days, and total reorders triggered.
        """
        summary = []
        for sku in self.warehouse.all_skus():
            summary.append({
                "sku_id": sku.sku_id,
                "sku_name": sku.name,
                "abc_class": sku.abc_class,
                "xyz_class": sku.xyz_class,
                "fill_rate_pct": round(sku.fill_rate * 100, 2),
                "stockout_days": sku.stockout_days,
                "total_demand": round(sku.total_demand, 1),
                "total_fulfilled": round(sku.total_fulfilled, 1),
            })
        return {
            "per_sku": summary,
            "site_fill_rate_pct": round(self.warehouse.site_fill_rate() * 100, 2),
            "total_stockout_days": sum(s.stockout_days for s in self.warehouse.all_skus()),
            "total_reorders_placed": len(self.reorder_log),
        }
