"""Greedy insertion baseline solver.

A deterministic, earliest-feasible insertion heuristic.
"""

from __future__ import annotations

import time
from typing import Literal

from src.domain.schema import RouteSolution, VRPTWInstance
from src.solvers.feasibility_checker import check_feasibility


OrderingRule = Literal["earliest_due_date", "farthest_first", "nearest_neighbor"]
TieBreakRule = Literal["farthest_first", "nearest_neighbor"]


def solve_greedy(
    instance: VRPTWInstance,
    ordering_rule: OrderingRule = "earliest_due_date",
    tie_break_rule: TieBreakRule = "farthest_first",
    max_vehicles: int | None = None,
    allow_early_depot_start: bool = False,
) -> RouteSolution:
    """Solve a VRPTW instance using greedy earliest-feasible insertion.

    The algorithm:
    1. Sort customers by ordering rule.
    2. For each customer, find the best feasible insertion point in existing routes.
    3. If no feasible insertion exists, open a new route.
    4. Repeat until all customers are placed.

    Args:
        instance: The VRPTW instance to solve.
        ordering_rule: How to order customers for insertion attempt.
        tie_break_rule: Secondary sort key when ordering_rule yields ties.
        max_vehicles: Maximum number of vehicles (None = unlimited).
        allow_early_depot_start: Whether routes can start before depot opens.

    Returns:
        RouteSolution with routes and metrics.
    """
    start_time = time.perf_counter()
    depot_id = instance.depot_id
    capacity = instance.vehicle_capacity
    customers = list(instance.customer_ids)

    ordered = _order_customers(instance, customers, ordering_rule, tie_break_rule)

    routes: list[list[int]] = []
    route_loads: list[int] = []
    route_end_times: list[float] = []
    first_feasible_time: float | None = None

    for customer in ordered:
        inserted = False
        best_route_idx = -1
        best_position = -1
        best_extra_time = float("inf")

        demand = instance.demand.get(customer, 0)

        for r_idx in range(len(routes)):
            load_after = route_loads[r_idx] + demand
            if load_after > capacity:
                continue

            positions = _find_insertion_positions(
                instance, routes[r_idx], customer, route_end_times[r_idx]
            )
            if positions is None:
                continue

            for pos in positions:
                extra_time = _insertion_extra_time(instance, routes[r_idx], customer, pos)
                if extra_time < best_extra_time:
                    best_extra_time = extra_time
                    best_route_idx = r_idx
                    best_position = pos
                    inserted = True

        if inserted:
            routes[best_route_idx].insert(best_position, customer)
            route_loads[best_route_idx] += demand
            route_end_times[best_route_idx] = _update_route_end_time(
                instance, routes[best_route_idx], route_end_times[best_route_idx]
            )
        else:
            if max_vehicles is not None and len(routes) >= max_vehicles:
                continue

            new_route = [customer]
            routes.append(new_route)
            route_loads.append(demand)

            start_time_depot = 0.0 if allow_early_depot_start else instance.ready_time.get(depot_id, 0)
            arrival = start_time_depot + instance.travel_duration(depot_id, customer)
            ready = instance.ready_time.get(customer, 0)
            wait = max(0.0, ready - arrival)
            service_end = arrival + wait + instance.service_time.get(customer, 0)
            route_end_times.append(service_end)

        if first_feasible_time is None and len(routes) > 0:
            first_feasible_time = time.perf_counter() - start_time

    total_distance = 0.0
    for route in routes:
        total_distance += _route_distance(instance, route)

    report = check_feasibility(instance, RouteSolution(
        routes=routes,
        vehicles_used=len(routes),
        total_distance=total_distance,
        total_duration=sum(route_end_times),
        feasible=True,
    ))

    runtime = time.perf_counter() - start_time

    return RouteSolution(
        routes=routes,
        vehicles_used=len(routes),
        total_distance=total_distance,
        total_duration=sum(route_end_times),
        feasible=report.feasible,
        late_violations=report.late_violations,
        capacity_violations=report.capacity_violations,
        runtime_sec=runtime,
        first_feasible_sec=first_feasible_time,
        metadata={"solver": "greedy", "ordering_rule": ordering_rule},
    )


def _order_customers(
    instance: VRPTWInstance,
    customers: list[int],
    rule: OrderingRule,
    tie_rule: TieBreakRule,
) -> list[int]:
    """Return customers sorted by the given rule."""
    depot_id = instance.depot_id

    def sort_key(cid: int) -> tuple:
        due = instance.due_time.get(cid, 999999)
        dist = instance.dist(depot_id, cid)

        if rule == "earliest_due_date":
            primary = due
        elif rule == "farthest_first":
            primary = -dist
        elif rule == "nearest_neighbor":
            primary = dist
        else:
            primary = due

        if tie_rule == "farthest_first":
            tie = -instance.dist(depot_id, cid)
        else:
            tie = instance.dist(depot_id, cid)

        return (primary, tie)

    return sorted(customers, key=sort_key)


def _find_insertion_positions(
    instance: VRPTWInstance,
    route: list[int],
    customer: int,
    current_time: float,
) -> list[int] | None:
    """Find all feasible insertion positions for a customer in a route.

    Returns list of positions (indices), or None if depot start is infeasible.
    """
    depot_id = instance.depot_id
    capacity = instance.vehicle_capacity
    demand = instance.demand.get(customer, 0)

    feasible_positions = []
    prev_id = depot_id

    for pos, node_id in enumerate(route):
        travel = instance.travel_duration(prev_id, customer)
        arrival = current_time + travel

        ready = instance.ready_time.get(customer, 0)
        due = instance.due_time.get(customer, 999999)

        if arrival <= due:
            feasible_positions.append(pos)
        elif arrival > due and arrival - due < 0.001:
            feasible_positions.append(pos)

        current_time += instance.travel_duration(prev_id, node_id)
        ready_n = instance.ready_time.get(node_id, 0)
        if current_time < ready_n:
            current_time = ready_n
        current_time += instance.service_time.get(node_id, 0)
        prev_id = node_id

    if feasible_positions:
        return feasible_positions

    travel_from_last = instance.travel_duration(prev_id, customer)
    arrival = current_time + travel_from_last
    due = instance.due_time.get(customer, 999999)
    if arrival <= due:
        return [len(route)]

    return None


def _insertion_extra_time(
    instance: VRPTWInstance,
    route: list[int],
    customer: int,
    position: int,
) -> float:
    """Compute the extra time added by inserting a customer at a position."""
    depot_id = instance.depot_id

    if not route:
        travel = instance.travel_duration(depot_id, customer) + instance.travel_duration(customer, depot_id)
        return travel

    if position == 0:
        prev_id = depot_id
    else:
        prev_id = route[position - 1]

    if position < len(route):
        next_id = route[position]
    else:
        next_id = depot_id

    old_travel = instance.travel_duration(prev_id, next_id)
    new_travel = (
        instance.travel_duration(prev_id, customer)
        + instance.travel_duration(customer, next_id)
    )

    return new_travel - old_travel


def _update_route_end_time(instance: VRPTWInstance, route: list[int], current_time: float) -> float:
    """Recompute the end time for a route after insertion."""
    depot_id = instance.depot_id
    prev_id = depot_id

    for node_id in route:
        current_time += instance.travel_duration(prev_id, node_id)
        ready = instance.ready_time.get(node_id, 0)
        if current_time < ready:
            current_time = ready
        current_time += instance.service_time.get(node_id, 0)
        prev_id = node_id

    return current_time


def _route_distance(instance: VRPTWInstance, route: list[int]) -> float:
    """Compute total travel distance of a route."""
    depot_id = instance.depot_id
    if not route:
        return 0.0
    total = instance.dist(depot_id, route[0])
    for i in range(len(route) - 1):
        total += instance.dist(route[i], route[i + 1])
    total += instance.dist(route[-1], depot_id)
    return total
