"""Internal data models for VRPTW.

Customer ID Convention:
- depot_id = 0 internally
- customer IDs are 1, 2, ..., n
- All coordinate/demand/service-time/demand dicts are keyed by node ID (including depot).
- The 'customer_ids' field lists only the non-depot nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class Node(BaseModel):
    """A single node (depot or customer) in a VRPTW instance."""

    model_config = ConfigDict(frozen=True)

    id: int
    x: float
    y: float
    demand: int = 0
    service_time: int = 0
    ready_time: int = 0
    due_time: int = 0


class TimeWindow(BaseModel):
    """A time window defined by earliest and latest times."""

    model_config = ConfigDict(frozen=True)

    open: int
    close: int


class VRPTWInstance(BaseModel):
    """A complete VRPTW instance in our unified internal representation.

    The instance stores all data keyed by node ID. The depot has id=0.
    Customer IDs are 1..n.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Instance name, e.g. 'C101' or 'vrptw_tiny'")
    source: str = Field(description="Data source: 'solomon', 'homberger', 'toy', 'processed'")
    family: str | None = Field(
        default=None,
        description="Instance family: C1, C2, R1, R2, RC1, RC2, or null for toy",
    )
    size: int = Field(description="Number of customers (depot not counted)")
    depot_id: int = Field(default=0, description="Depot node ID (always 0 in this codebase)")
    customer_ids: list[int] = Field(description="Sorted list of customer node IDs (1..n)")
    x_coords: dict[int, float] = Field(description="X coordinate per node ID")
    y_coords: dict[int, float] = Field(description="Y coordinate per node ID")
    demand: dict[int, int] = Field(description="Demand per node ID (0 for depot)")
    service_time: dict[int, int] = Field(description="Service time per node ID (0 for depot)")
    ready_time: dict[int, int] = Field(
        default_factory=dict,
        description="Earliest service start time per node ID (0 for depot)",
    )
    due_time: dict[int, int] = Field(
        default_factory=dict,
        description="Latest service start time per node ID (999999 for depot)",
    )
    vehicle_capacity: int = Field(description="Vehicle capacity Q")
    vehicle_count: int | None = Field(
        default=None,
        description="Max number of vehicles available (None = unlimited)",
    )
    distance_matrix: list[list[float]] | None = Field(
        default=None,
        description="Precomputed distance matrix, indexed by [from][to]. Euclidean if None.",
    )
    duration_matrix: list[list[float]] | None = Field(
        default=None,
        description="Precomputed travel duration matrix. Equal to distance if None.",
    )
    objective: str = Field(
        default="lexicographic_vehicles_distance",
        description="Optimization objective",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional instance metadata",
    )

    def num_nodes(self) -> int:
        """Total number of nodes including depot."""
        return self.size + 1

    def get_all_node_ids(self) -> list[int]:
        """Return [0] + customer_ids."""
        return [self.depot_id] + self.customer_ids

    def dist(self, from_id: int, to_id: int) -> float:
        """Return Euclidean distance between two nodes."""
        if self.distance_matrix is not None:
            return self.distance_matrix[from_id][to_id]
        dx = self.x_coords[from_id] - self.x_coords[to_id]
        dy = self.y_coords[from_id] - self.y_coords[to_id]
        return (dx * dx + dy * dy) ** 0.5

    def travel_duration(self, from_id: int, to_id: int) -> float:
        """Return travel duration between two nodes. Defaults to distance.

        For Solomon-style instances (R1/C1/RC1), distance equals time (unit speed=1).
        For homogeneous instances (R2/C2/RC2), time windows are wide so unit mismatch
        has minimal impact on feasibility. We default to dist-to-time = 1:1.
        """
        if self.duration_matrix is not None:
            return self.duration_matrix[from_id][to_id]
        return self.dist(from_id, to_id)


class RouteSolution(BaseModel):
    """A solution (route plan) for a VRPTW instance."""

    model_config = ConfigDict(extra="forbid")

    routes: list[list[int]] = Field(
        description="List of routes. Each route is [depot, ..., customer, ..., depot].",
    )
    vehicles_used: int = Field(description="Number of vehicles/rides actually used")
    total_distance: float = Field(description="Sum of travel distances across all routes")
    total_duration: float = Field(description="Sum of route durations (makespan or sum)")
    feasible: bool = Field(description="Whether the solution satisfies all constraints")
    late_violations: int = Field(
        default=0,
        description="Number of time-window violations across all routes",
    )
    capacity_violations: int = Field(
        default=0,
        description="Number of capacity violations across all routes",
    )
    runtime_sec: float = Field(default=0.0, description="Wall-clock solve time in seconds")
    first_feasible_sec: float | None = Field(
        default=None,
        description="Time to first feasible solution",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Solver name, seed, and other solver-specific info",
    )

    def route_count(self) -> int:
        """Number of routes."""
        return len(self.routes)

    def is_empty(self) -> bool:
        """True if no routes at all."""
        return len(self.routes) == 0


@dataclass
class ValidationReport:
    """Result of feasibility validation for a RouteSolution."""

    feasible: bool
    late_violations: int
    capacity_violations: int
    route_reports: list[RouteValidationReport]
    total_load_violated: int
    total_time_violated: int


@dataclass
class RouteValidationReport:
    """Validation result for a single route."""

    route_index: int
    route: list[int]
    load: int
    capacity_violated: bool
    time_violations: int
    route_distance: float
    route_duration: float
    arrival_times: dict[int, float]
    service_start_times: dict[int, float]
    waiting_times: dict[int, float]


class ExperimentResult(BaseModel):
    """Result of a single experiment run."""

    model_config = ConfigDict(extra="allow")

    experiment_id: str
    instance_name: str
    solver: str
    vehicles_used: int
    total_distance: float
    total_duration: float
    feasible: bool
    late_violations: int
    capacity_violations: int
    runtime_sec: float
    first_feasible_sec: float | None = None
    gap_to_bks: float | None = None
    lexicographic_gap: float | None = None
    metadata: dict = Field(default_factory=dict)
