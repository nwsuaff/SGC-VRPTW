"""Data models for dynamic VRPTW (ORTEC-style order-arrival scenario).

In dynamic dispatch, orders arrive in epochs during the day. A solver must,
at each epoch, decide which open orders to dispatch (assign to vehicle routes)
and which to defer. Some orders are "must-go" -- delaying them will make their
time windows infeasible. The objective is to minimize total driving duration.

Epoch layout (following ORTEC competition):
- EPOCH_DURATION = 3600s (1 hour per epoch)
- MARGIN_DISPATCH = 3600s (1 hour to dispatch a vehicle before it starts driving)
- At epoch e, requests released in epochs [0, e] are "open"
- A must-go order must be dispatched at epoch e, or it will be infeasible at epoch e+1
- The static instance provides the "customer base" from which dynamic orders are sampled
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DynamicOrder(BaseModel):
    """A single order (customer) that arrived in the dynamic scenario.

    Orders are keyed by request_id. The depot is always request_id=0.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: int = Field(description="Unique request ID (0 = depot sentinel)")
    customer_idx: int = Field(
        description="Index into the static base instance's customer list (0-based)"
    )
    x: float = Field(description="X coordinate")
    y: float = Field(description="Y coordinate")
    demand: int = Field(description="Demand amount")
    ready_time: int = Field(description="Earliest service start time (seconds)")
    due_time: int = Field(description="Latest service start time (seconds)")
    service_time: int = Field(description="Service duration at this customer")
    release_epoch: int = Field(
        description="Epoch when this order became available (0 = available from start)"
    )
    is_dispatched: bool = Field(default=False, description="Whether this order has been served")
    must_dispatch: bool = Field(
        default=False,
        description=(
            "True if this order must be dispatched at the current epoch; "
            "delaying it will make the time window infeasible"
        ),
    )


class EpochResult(BaseModel):
    """Result of solving one epoch in the dynamic scenario."""

    model_config = ConfigDict(extra="forbid")

    epoch: int = Field(description="Epoch number (0-indexed)")
    routes: list[list[int]] = Field(
        description="Routes submitted for this epoch. "
        "Each route is [customer1, customer2, ...] (no depot endpoints)."
    )
    orders_dispatched: int = Field(description="Number of orders served in this epoch")
    orders_pending: int = Field(description="Number of open (not yet dispatched) orders")
    served_rate: float = Field(
        description="Cumulative proportion of all orders served so far (0-1)"
    )
    total_distance: float = Field(description="Total driving distance this epoch")
    driving_duration: float = Field(
        description="Total driving duration this epoch (excluding wait/drive-to-depot time)"
    )
    driving_duration_cumulative: float = Field(
        description="Cumulative driving duration across all epochs so far"
    )
    feasible: bool = Field(
        default=True,
        description="Whether the submitted routes are feasible"
    )
    runtime_sec: float = Field(default=0.0, description="Wall-clock time to solve this epoch")
    late_violations: int = Field(default=0, description="Time-window violations")
    capacity_violations: int = Field(default=0, description="Capacity violations")


class DynamicVRPTWInstance(BaseModel):
    """A dynamic VRPTW scenario built from a static base instance.

    The static base provides:
    - Customer coordinates, demands, service times, time windows
    - Duration matrix (real-world road driving times)
    - Vehicle capacity

    Dynamic layers:
    - Epochs define when orders become available
    - Orders sampled from the static base define which customers appear when
    - The solver operates on the union of all orders released so far
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Scenario name, e.g. 'ORTEC-VRPTW-ASYM-xxx-d1-n458-k35'")
    base_instance_name: str = Field(
        description="Name of the static VRPTWInstance this is derived from"
    )
    source: Literal["ortec", "ortec_synthetic"] = Field(
        default="ortec",
        description="Data source"
    )
    num_epochs: int = Field(description="Total number of epochs in this scenario")
    epoch_duration: int = Field(
        default=3600,
        description="Duration of each epoch in seconds"
    )
    margin_dispatch: int = Field(
        default=3600,
        description="Time assumed to dispatch a vehicle before it starts driving (seconds)"
    )
    num_orders_total: int = Field(
        description="Total number of customer orders (excluding depot)"
    )
    num_orders_per_epoch: int | None = Field(
        default=None,
        description="Target number of new orders released per epoch (auto-computed if None)"
    )
    orders: list[DynamicOrder] = Field(
        default_factory=list,
        description="All orders in the scenario, sorted by request_id"
    )
    duration_matrix: list[list[float]] = Field(
        description="Duration matrix from the static base (EXPLICIT, seconds)"
    )
    vehicle_capacity: int = Field(description="Vehicle capacity")
    vehicle_count: int | None = Field(
        default=None,
        description="Maximum vehicles available per epoch"
    )
    metadata: dict = Field(default_factory=dict)

    def get_open_orders(self, through_epoch: int) -> list[DynamicOrder]:
        """Return all orders released by or before `through_epoch` that are not yet dispatched."""
        return [o for o in self.orders if o.release_epoch <= through_epoch and not o.is_dispatched]

    def get_orders_by_epoch(self, epoch: int) -> list[DynamicOrder]:
        """Return new orders that arrive exactly at `epoch`."""
        return [o for o in self.orders if o.release_epoch == epoch]

    def get_depot(self) -> DynamicOrder:
        """Return the depot sentinel order (request_id=0)."""
        return self.orders[0]


@dataclass
class DynamicSolverResult:
    """Complete result of a dynamic dispatch run across all epochs."""

    scenario_name: str
    base_instance_name: str
    num_epochs: int
    epoch_results: list[EpochResult]
    total_distance: float
    total_driving_duration: float
    overall_served_rate: float
    overall_feasible: bool
    total_late_violations: int
    total_capacity_violations: int
    total_runtime_sec: float

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "base_instance_name": self.base_instance_name,
            "num_epochs": self.num_epochs,
            "total_distance": self.total_distance,
            "total_driving_duration": self.total_driving_duration,
            "overall_served_rate": self.overall_served_rate,
            "overall_feasible": self.overall_feasible,
            "total_late_violations": self.total_late_violations,
            "total_capacity_violations": self.total_capacity_violations,
            "total_runtime_sec": self.total_runtime_sec,
            "epochs": [e.model_dump() for e in self.epoch_results],
        }

    def served_rate_by_epoch(self) -> list[float]:
        """Return cumulative served rate at the end of each epoch."""
        return [e.served_rate for e in self.epoch_results]

    def driving_duration_by_epoch(self) -> list[float]:
        """Return cumulative driving duration at the end of each epoch."""
        return [e.driving_duration_cumulative for e in self.epoch_results]
