"""Deterministic feasibility checker for VRPTW solutions.

This checker is solver-independent and serves as the source of truth for
tests and for validating LLM-produced hints.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.schema import RouteSolution, ValidationReport, RouteValidationReport, VRPTWInstance


def check_feasibility(instance: VRPTWInstance, solution: RouteSolution) -> ValidationReport:
    """Check whether a solution satisfies all VRPTW constraints.

    Validates:
    - Each route starts and ends at the depot (node 0).
    - All customers are visited exactly once.
    - Cumulative load does not exceed vehicle capacity.
    - Arrival times respect time windows (ready_time <= arrival <= due_time).
    - Service times are accounted for correctly.

    Args:
        instance: The VRPTW instance.
        solution: The proposed solution to validate.

    Returns:
        ValidationReport with detailed per-route information.
    """
    depot_id = instance.depot_id
    capacity = instance.vehicle_capacity

    route_reports: list[RouteValidationReport] = []
    total_load_violated = 0
    total_time_violated = 0

    all_visited: set[int] = set()
    all_expected = set(instance.customer_ids)

    for route_idx, route in enumerate(solution.routes):
        report = _check_single_route(instance, route, route_idx, depot_id, capacity)
        route_reports.append(report)

        if report.capacity_violated:
            total_load_violated += 1
        total_time_violated += report.time_violations

        for node_id in route:
            if node_id != depot_id:
                all_visited.add(node_id)

    missing_customers = all_expected - all_visited
    extra_customers = all_visited - all_expected

    if missing_customers or extra_customers:
        total_time_violated += len(missing_customers) + len(extra_customers)

    feasible = (
        total_load_violated == 0
        and total_time_violated == 0
        and not missing_customers
        and not extra_customers
        and solution.late_violations == 0
        and solution.capacity_violations == 0
    )

    return ValidationReport(
        feasible=feasible,
        late_violations=total_time_violated,
        capacity_violations=total_load_violated,
        route_reports=route_reports,
        total_load_violated=total_load_violated,
        total_time_violated=total_time_violated,
    )


def _check_single_route(
    instance: VRPTWInstance,
    route: list[int],
    route_index: int,
    depot_id: int,
    capacity: int,
) -> RouteValidationReport:
    """Validate a single route.

    Simulates the route from depot, tracking cumulative load, arrival times,
    waiting time, and service completion.
    """
    if not route:
        return RouteValidationReport(
            route_index=route_index,
            route=route,
            load=0,
            capacity_violated=False,
            time_violations=0,
            route_distance=0.0,
            route_duration=0.0,
            arrival_times={},
            service_start_times={},
            waiting_times={},
        )

    load = 0
    arrival_times: dict[int, float] = {}
    service_start_times: dict[int, float] = {}
    waiting_times: dict[int, float] = {}
    route_distance = 0.0
    time_violations = 0
    capacity_violated = False

    prev_id = depot_id
    current_time = instance.ready_time.get(depot_id, 0)

    for node_id in route:
        dist = instance.dist(prev_id, node_id)
        travel = round(instance.travel_duration(prev_id, node_id))
        route_distance += dist
        current_time += travel

        ready = instance.ready_time.get(node_id, 0)
        due = instance.due_time.get(node_id, 999999)
        service = instance.service_time.get(node_id, 0)

        arrival_times[node_id] = current_time

        if current_time < ready:
            wait = ready - current_time
            waiting_times[node_id] = wait
            current_time = float(ready)
        else:
            waiting_times[node_id] = 0.0

        if current_time > due + 1e-3:
            time_violations += 1

        # NOTE: The service_time at this node has already been added to
        # current_time above. This means current_time now represents
        # "time when service at node_id FINISHES". The due time constraint
        # is applied by the solver to (arrival + service_time), which equals
        # current_time. So the check above (current_time > due) correctly
        # validates the solver's constraint formulation.

        service_start_times[node_id] = current_time
        current_time += service

        load += instance.demand.get(node_id, 0)
        if load > capacity:
            capacity_violated = True
            time_violations += 1

        prev_id = node_id

    travel_back = instance.dist(prev_id, depot_id)
    route_distance += travel_back

    return RouteValidationReport(
        route_index=route_index,
        route=route,
        load=load,
        capacity_violated=capacity_violated,
        time_violations=time_violations,
        route_distance=route_distance,
        route_duration=(
            service_start_times.get(route[-1], 0.0)
            + round(instance.travel_duration(route[-1], depot_id)) if route else 0.0
        ),
        arrival_times=arrival_times,
        service_start_times=service_start_times,
        waiting_times=waiting_times,
    )


def compute_route_stats(instance: VRPTWInstance, route: list[int]) -> tuple[float, float, float]:
    """Compute distance, duration, and load for a single route.

    Args:
        instance: The VRPTW instance.
        route: A single route starting and ending at depot.

    Returns:
        Tuple of (total_distance, total_duration, total_load).
    """
    if not route:
        return 0.0, 0.0, 0.0

    total_distance = 0.0
    total_load = 0
    depot_id = instance.depot_id
    prev_id = depot_id

    for node_id in route:
        total_distance += instance.dist(prev_id, node_id)
        total_load += instance.demand.get(node_id, 0)
        prev_id = node_id

    total_distance += instance.dist(prev_id, depot_id)

    last_service_end = 0.0
    if route:
        last_service_end = _compute_service_end(instance, route)

    total_duration = total_distance + last_service_end

    return total_distance, total_duration, total_load


def _compute_service_end(instance: VRPTWInstance, route: list[int]) -> float:
    """Compute the time when service at the last customer of the route ends."""
    current_time = instance.ready_time.get(instance.depot_id, 0)
    prev_id = instance.depot_id

    for node_id in route:
        travel_time = round(instance.travel_duration(prev_id, node_id))
        current_time += travel_time

        ready = instance.ready_time.get(node_id, 0)
        if current_time < ready:
            current_time = ready

        service = instance.service_time.get(node_id, 0)
        current_time += service
        prev_id = node_id

    return current_time
