"""Diagnostic encoder: extracts structured diagnostics from VRPTW instances and solutions.

This module implements the "Diagnostic Encoder" component of the verifier-gated
LLM guidance protocol. It transforms solver state into interpretable diagnostics
that help the LLM understand where to focus its guidance.

The diagnostics include:
- Route-level: load utilization, time slack, duration
- Customer-level: time-window tightness, waiting time risk, spatial clustering
- Edge-level: high-cost edges that could be improved
- Solver-level: failure modes and patterns
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.schema import VRPTWInstance, RouteSolution


@dataclass
class CustomerDiagnostics:
    """Diagnostics for a single customer."""
    customer_id: int
    time_window_size: float
    time_window_slack: float
    travel_time_from_depot: float
    demand: float
    demand_fraction: float
    is_tight: bool
    spatial_neighbors: list[int]


@dataclass
class RouteDiagnostics:
    """Diagnostics for a single route."""
    route_index: int
    customers: list[int]
    load: float
    load_utilization: float
    total_duration: float
    total_distance: float
    avg_time_window_slack: float
    min_time_window_slack: float
    waiting_time_fraction: float
    has_late_violation: bool


@dataclass
class EdgeDiagnostics:
    """Diagnostics for customer pairs (potential edges)."""
    customer_a: int
    customer_b: int
    distance: float
    is_long: bool
    in_same_route: bool
    route_index: int | None


@dataclass
class SolverDiagnostics:
    """High-level solver diagnostics."""
    total_vehicles: int
    total_distance: float
    avg_route_utilization: float
    tight_customers_count: int
    long_edges_count: int
    high_waiting_routes: list[int]
    low_utilization_routes: list[int]


@dataclass
class FullDiagnostics:
    """Complete diagnostic output for an instance-solution pair."""
    instance_name: str
    solver: SolverDiagnostics
    customer_diags: dict[int, CustomerDiagnostics] = field(default_factory=dict)
    route_diags: dict[int, RouteDiagnostics] = field(default_factory=dict)
    edge_diags: list[EdgeDiagnostics] = field(default_factory=list)

    def to_text_summary(self) -> str:
        """Convert to human-readable text summary for LLM prompt."""
        lines = []

        # Solver-level summary
        lines.append("## Solver Diagnostics Summary")
        lines.append(f"- Total vehicles: {self.solver.total_vehicles}")
        lines.append(f"- Total distance: {self.solver.total_distance:.2f}")
        lines.append(f"- Avg route utilization: {self.solver.avg_route_utilization:.1%}")
        lines.append(f"- Tight time-window customers: {self.solver.tight_customers_count}")
        lines.append(f"- Long distance edges: {self.solver.long_edges_count}")

        if self.solver.high_waiting_routes:
            lines.append(f"- High waiting routes: {self.solver.high_waiting_routes}")
        if self.solver.low_utilization_routes:
            lines.append(f"- Low utilization routes: {self.solver.low_utilization_routes}")

        # Route-level diagnostics
        lines.append("\n## Route-Level Diagnostics")
        if self.route_diags:
            for route_idx, rd in sorted(self.route_diags.items())[:5]:
                lines.append(
                    f"Route {route_idx}: "
                    f"customers={len(rd.customers)}, "
                    f"util={rd.load_utilization:.1%}, "
                    f"duration={rd.total_duration:.1f}, "
                    f"min_slack={rd.min_time_window_slack:.1f}"
                )
            if len(self.route_diags) > 5:
                lines.append(f"... and {len(self.route_diags) - 5} more routes")
        else:
            lines.append("No routes in current solution.")

        # Customer-level diagnostics
        lines.append("\n## Customer-Level Diagnostics")
        tight_customers = [
            (cid, cd) for cid, cd in self.customer_diags.items() if cd.is_tight
        ]
        tight_customers.sort(key=lambda x: x[1].time_window_slack)
        if tight_customers:
            lines.append("Tight time-window customers (most constrained first):")
            for cid, cd in tight_customers[:10]:
                lines.append(
                    f"  Customer {cid}: "
                    f"window={cd.time_window_size:.1f}, "
                    f"slack={cd.time_window_slack:.1f}, "
                    f"demand_fraction={cd.demand_fraction:.2f}"
                )
        else:
            lines.append("No particularly tight customers.")

        # Edge-level diagnostics
        lines.append("\n## Edge-Level Diagnostics")
        if self.edge_diags:
            long_edges = [e for e in self.edge_diags if e.is_long]
            if long_edges:
                lines.append(f"Long edges (>100 distance units): {len(long_edges)}")
                for edge in sorted(long_edges, key=lambda e: e.distance, reverse=True)[:5]:
                    lines.append(
                        f"  Edge {edge.customer_a}-{edge.customer_b}: "
                        f"distance={edge.distance:.1f}"
                    )

        return "\n".join(lines)


def compute_diagnostics(
    instance: "VRPTWInstance",
    solution: "RouteSolution | None",
    long_edge_threshold: float = 100.0,
) -> FullDiagnostics:
    """Compute full diagnostics for an instance and solution.

    Args:
        instance: The VRPTW instance.
        solution: Current solution (may be None).
        long_edge_threshold: Distance threshold for "long edge" classification.

    Returns:
        FullDiagnostics containing all computed diagnostics.
    """
    # Basic solver diagnostics
    solver_diags = _compute_solver_diagnostics(instance, solution)

    # Customer diagnostics
    customer_diags = _compute_customer_diagnostics(instance)

    # Route diagnostics
    route_diags = _compute_route_diagnostics(instance, solution, customer_diags)

    # Edge diagnostics
    edge_diags = _compute_edge_diagnostics(instance, solution, long_edge_threshold)

    return FullDiagnostics(
        instance_name=instance.name,
        solver=solver_diags,
        customer_diags=customer_diags,
        route_diags=route_diags,
        edge_diags=edge_diags,
    )


def _compute_solver_diagnostics(
    instance: "VRPTWInstance",
    solution: "RouteSolution | None",
) -> SolverDiagnostics:
    """Compute solver-level diagnostics."""
    if solution is None or not solution.routes:
        return SolverDiagnostics(
            total_vehicles=0,
            total_distance=0.0,
            avg_route_utilization=0.0,
            tight_customers_count=0,
            long_edges_count=0,
            high_waiting_routes=[],
            low_utilization_routes=[],
        )

    total_demand = sum(instance.demand.get(cid, 0) for cid in instance.customer_ids)
    total_capacity = instance.vehicle_capacity * len(solution.routes)

    utilizations = []
    for route in solution.routes:
        route_load = sum(instance.demand.get(cid, 0) for cid in route)
        utilizations.append(route_load / instance.vehicle_capacity)

    avg_util = sum(utilizations) / len(utilizations) if utilizations else 0.0

    # Count tight customers
    tight_count = sum(
        1 for cid in instance.customer_ids
        if _get_time_window_slack(instance, cid) < 30
    )

    # Find high waiting and low utilization routes
    high_waiting = []
    low_util = []
    for i, route in enumerate(solution.routes):
        route_load = sum(instance.demand.get(cid, 0) for cid in route)
        util = route_load / instance.vehicle_capacity

        if util < 0.5:
            low_util.append(i)

    return SolverDiagnostics(
        total_vehicles=len(solution.routes),
        total_distance=solution.total_distance,
        avg_route_utilization=avg_util,
        tight_customers_count=tight_count,
        long_edges_count=0,
        high_waiting_routes=high_waiting,
        low_utilization_routes=low_util,
    )


def _compute_customer_diagnostics(
    instance: "VRPTWInstance",
) -> dict[int, CustomerDiagnostics]:
    """Compute customer-level diagnostics."""
    diags = {}

    for cid in instance.customer_ids:
        ready = instance.ready_time.get(cid, 0)
        due = instance.due_time.get(cid, 999999)
        window_size = due - ready
        travel_from_depot = instance.dist(instance.depot_id, cid)
        slack = window_size - 2 * travel_from_depot

        demand = instance.demand.get(cid, 0)
        demand_fraction = demand / instance.vehicle_capacity

        # Find spatial neighbors
        neighbors = _find_spatial_neighbors(instance, cid, k=5)

        diags[cid] = CustomerDiagnostics(
            customer_id=cid,
            time_window_size=window_size,
            time_window_slack=slack,
            travel_time_from_depot=travel_from_depot,
            demand=demand,
            demand_fraction=demand_fraction,
            is_tight=slack < 30,
            spatial_neighbors=neighbors,
        )

    return diags


def _compute_route_diagnostics(
    instance: "VRPTWInstance",
    solution: "RouteSolution | None",
    customer_diags: dict[int, CustomerDiagnostics],
) -> dict[int, RouteDiagnostics]:
    """Compute route-level diagnostics."""
    if solution is None:
        return {}

    route_diags = {}

    for route_idx, route in enumerate(solution.routes):
        if not route:
            continue

        load = sum(instance.demand.get(cid, 0) for cid in route)
        load_util = load / instance.vehicle_capacity

        # Calculate route duration
        duration = instance.dist(instance.depot_id, route[0]) if route else 0
        for i in range(len(route) - 1):
            duration += instance.dist(route[i], route[i + 1])
        if route:
            duration += instance.dist(route[-1], instance.depot_id)

        # Calculate distance
        distance = instance.dist(instance.depot_id, route[0]) if route else 0
        for i in range(len(route) - 1):
            distance += instance.dist(route[i], route[i + 1])
        if route:
            distance += instance.dist(route[-1], instance.depot_id)

        # Time window slack statistics
        slacks = [customer_diags[cid].time_window_slack for cid in route if cid in customer_diags]
        avg_slack = sum(slacks) / len(slacks) if slacks else 0
        min_slack = min(slacks) if slacks else 0

        route_diags[route_idx] = RouteDiagnostics(
            route_index=route_idx,
            customers=route,
            load=load,
            load_utilization=load_util,
            total_duration=duration,
            total_distance=distance,
            avg_time_window_slack=avg_slack,
            min_time_window_slack=min_slack,
            waiting_time_fraction=0.0,
            has_late_violation=False,
        )

    return route_diags


def _compute_edge_diagnostics(
    instance: "VRPTWInstance",
    solution: "RouteSolution | None",
    long_edge_threshold: float,
) -> list[EdgeDiagnostics]:
    """Compute edge-level diagnostics."""
    if solution is None:
        return []

    edge_diags = []
    customer_set = set(instance.customer_ids)

    # Find customer-to-route mapping
    route_map: dict[int, int] = {}
    for route_idx, route in enumerate(solution.routes):
        for cid in route:
            route_map[cid] = route_idx

    # Check inter-route edges (edges between customers in different routes)
    # These are candidates for swap/relocate operations
    long_edges = []

    for cid_a in instance.customer_ids:
        for cid_b in instance.customer_ids:
            if cid_a >= cid_b:
                continue

            dist = instance.dist(cid_a, cid_b)

            if dist > long_edge_threshold:
                in_same_route = route_map.get(cid_a) == route_map.get(cid_b)
                route_idx = route_map.get(cid_a) if in_same_route else None

                long_edges.append(EdgeDiagnostics(
                    customer_a=cid_a,
                    customer_b=cid_b,
                    distance=dist,
                    is_long=dist > long_edge_threshold,
                    in_same_route=in_same_route,
                    route_index=route_idx,
                ))

    # Limit to most significant edges
    long_edges.sort(key=lambda e: e.distance, reverse=True)
    return long_edges[:50]


def _get_time_window_slack(instance: "VRPTWInstance", customer_id: int) -> float:
    """Calculate time window slack for a customer."""
    ready = instance.ready_time.get(customer_id, 0)
    due = instance.due_time.get(customer_id, 999999)
    window_size = due - ready
    travel_from_depot = instance.dist(instance.depot_id, customer_id)
    return window_size - 2 * travel_from_depot


def _find_spatial_neighbors(
    instance: "VRPTWInstance",
    customer_id: int,
    k: int = 5,
) -> list[int]:
    """Find k nearest spatial neighbors of a customer."""
    x = instance.x_coords.get(customer_id, 0)
    y = instance.y_coords.get(customer_id, 0)

    distances = []
    for cid in instance.customer_ids:
        if cid == customer_id:
            continue
        cx = instance.x_coords.get(cid, 0)
        cy = instance.y_coords.get(cid, 0)
        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        distances.append((cid, dist))

    distances.sort(key=lambda x: x[1])
    return [cid for cid, _ in distances[:k]]


def compute_infeasibility_reasons(
    instance: "VRPTWInstance",
    solution: "RouteSolution",
) -> dict[str, list[int]]:
    """Identify likely reasons for infeasibility in a solution.

    Returns a dictionary mapping infeasibility type to list of customer/route IDs.
    """
    reasons: dict[str, list[int]] = {
        "capacity_violations": [],
        "time_window_violations": [],
        "tight_window_customers": [],
        "high_demand_customers": [],
    }

    for route_idx, route in enumerate(solution.routes):
        route_load = sum(instance.demand.get(cid, 0) for cid in route)

        if route_load > instance.vehicle_capacity:
            reasons["capacity_violations"].append(route_idx)

        for cid in route:
            due = instance.due_time.get(cid, 999999)
            ready = instance.ready_time.get(cid, 0)
            window = due - ready

            if window < 30:
                reasons["tight_window_customers"].append(cid)

            demand = instance.demand.get(cid, 0)
            if demand > instance.vehicle_capacity * 0.5:
                reasons["high_demand_customers"].append(cid)

    return reasons
