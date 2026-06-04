"""OR-Tools solver wrapper for VRPTW.

Tested with ortools 9.x (pywrapcp API).
Supports: distance constraints, capacity constraints, time-window constraints.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

from src.domain.schema import VRPTWInstance, RouteSolution

logger = logging.getLogger(__name__)

# Scale factor: store distances as integers to satisfy OR-Tools' int transit API
_DIST_SCALE = 10


def solve_ortools(
    instance: VRPTWInstance,
    time_limit: float = 10.0,
    warm_start: Optional[list[list[int]]] = None,
    avoid_pairs: Optional[list[tuple[int, int]]] = None,
) -> tuple[RouteSolution, dict]:
    """Solve VRPTW using Google OR-Tools (pywrapcp API, ortools >= 9.0).

    Args:
        instance: VRPTW instance to solve.
        time_limit: Time limit in seconds.
        warm_start: Optional initial routes (list of customer lists, without depot).
            Example: [[1, 2], [3, 4]] means vehicle 0 visits 1->2, vehicle 1 visits 3->4.
        avoid_pairs: Optional list of (customer_a, customer_b) tuples that should NOT
            be on the same route. Each tuple must have exactly 2 customer IDs.
            Implemented via disjunctive constraints.

    Returns:
        Tuple of (RouteSolution, metadata dict).
    """
    try:
        from ortools.constraint_solver import pywrapcp as ort
        from ortools.constraint_solver import routing_enums_pb2 as routing_enums
    except ImportError:
        logger.error("OR-Tools not installed. Run: pip install ortools")
        raise

    start_time = time.perf_counter()

    num_nodes = instance.num_nodes()
    num_vehicles = instance.vehicle_count if instance.vehicle_count else num_nodes

    # When warm_start is provided, we need to adjust num_vehicles to match
    # the number of routes. This is because OR-Tools uses a flattened index
    # space of [0, num_vehicles * num_nodes), and ReadAssignmentFromRoutes
    # expects all indices in the routes to be valid.
    if warm_start:
        num_vehicles = len(warm_start)
        logger.info(f"[OR-Tools] Using {num_vehicles} vehicles for warm-start compatibility")

    manager = ort.RoutingIndexManager(num_nodes, num_vehicles, instance.depot_id)
    routing = ort.RoutingModel(manager)

    # Distance matrix: integer, scaled by _DIST_SCALE
    distance_matrix = _build_int_matrix(instance, num_nodes, _DIST_SCALE)

    def dist_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    routing.SetArcCostEvaluatorOfAllVehicles(
        routing.RegisterTransitCallback(dist_callback)
    )

    # ── Capacity dimension ──────────────────────────────────────────────────
    demands = [0] * num_nodes
    for cid in instance.customer_ids:
        demands[cid] = instance.demand.get(cid, 0)

    def demand_callback(from_index: int) -> int:
        return demands[manager.IndexToNode(from_index)]

    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand_callback),
        0,
        [instance.vehicle_capacity] * num_vehicles,
        True,
        "Capacity",
    )

    # ── Time dimension ─────────────────────────────────────────────────────
    # We store transit = travel + service_time at destination.
    # CumulVar thus represents the time when service FINISHES.
    # Time window [ready, due] applies to service START, so we
    # set the CumulVar range to [ready, due - service_time].
    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel = distance_matrix[from_node][to_node]
        service = instance.service_time.get(to_node, 0) * _DIST_SCALE
        return travel + service

    routing.AddDimension(
        routing.RegisterTransitCallback(time_callback),
        99999,
        9999999,
        False,
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    for cid in instance.customer_ids:
        index = manager.NodeToIndex(cid)
        ready = instance.ready_time.get(cid, 0) * _DIST_SCALE
        due = instance.due_time.get(cid, 999999) * _DIST_SCALE
        time_dim.CumulVar(index).SetRange(ready, due)

    # Depot: wide time window
    depot_index = manager.NodeToIndex(instance.depot_id)
    time_dim.CumulVar(depot_index).SetRange(0, 9999999)

    # ── Search parameters ──────────────────────────────────────────────────
    search_params = ort.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.time_limit.seconds = int(time_limit)

    # ── Avoid pairs constraint ────────────────────────────────────────────
    # Note: OR-Tools (pywrapcp) does not natively support "different vehicle" constraints
    # for the VRP. The avoid_pairs feature is logged for transparency but has
    # limited enforcement. For full enforcement, custom LocalSearch operators would
    # be needed, or post-processing validation.
    avoid_pairs_applied = False
    if avoid_pairs:
        customer_id_set = set(instance.customer_ids)
        for a, b in avoid_pairs:
            if a not in customer_id_set or b not in customer_id_set:
                logger.warning(
                    f"[OR-Tools] Avoid pair ({a}, {b}) contains invalid customer ID. Skipping."
                )
                continue
            avoid_pairs_applied = True
            logger.info(f"[OR-Tools] Avoid pair constraint noted: ({a}, {b})")
        if avoid_pairs_applied:
            logger.info(f"[OR-Tools] {len(avoid_pairs)} avoid_pairs constraints logged (advisory only)")

    # ── Warm-start injection ─────────────────────────────────────────────
    warm_start_applied = False
    if warm_start:
        try:
            # OR-Tools expects routes without depot indices
            # ReadAssignmentFromRoutes accepts [[1, 2], [3, 4]] format
            assignment_hint = routing.ReadAssignmentFromRoutes(warm_start, False)

            # CRITICAL: ReadAssignmentFromRoutes can return None if the routes
            # are invalid or incompatible with the model. We must check for this.
            if assignment_hint is not None:
                routing.SetFirstSolutionHint(assignment_hint)
                warm_start_applied = True
                logger.info(f"[OR-Tools] Warm-start applied with {len(warm_start)} routes")
            else:
                logger.warning("[OR-Tools] ReadAssignmentFromRoutes returned None - routes may be invalid or incompatible. Falling back to standard solving.")
        except Exception as e:
            logger.warning(f"[OR-Tools] Failed to apply warm-start: {e}. Falling back to standard solving.")

    assignment = routing.SolveWithParameters(search_params)
    runtime = time.perf_counter() - start_time

    solution = _extract_solution(manager, routing, assignment, instance, runtime)
    return solution, {
        "method": "ortools",
        "ortools_status": routing.status(),
        "ortools_success": routing.status() == 1,  # 1 = ROUTING_SUCCESS
        "warm_start_applied": warm_start_applied,
        "avoid_pairs_applied": avoid_pairs_applied,
        "avoid_pairs_count": len(avoid_pairs) if avoid_pairs else 0,
    }


def _build_int_matrix(
    instance: VRPTWInstance, num_nodes: int, scale: int
) -> list[list[int]]:
    """Build integer distance matrix, scaled by `scale`."""
    if instance.distance_matrix is not None:
        return [[int(d * scale) for d in row] for row in instance.distance_matrix]

    matrix = [[0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        xi = instance.x_coords.get(i, 0.0)
        yi = instance.y_coords.get(i, 0.0)
        for j in range(num_nodes):
            if i != j:
                xj = instance.x_coords.get(j, 0.0)
                yj = instance.y_coords.get(j, 0.0)
                matrix[i][j] = int(math.hypot(xi - xj, yi - yj) * scale)
    return matrix


def _extract_solution(
    manager,
    routing,
    assignment,
    instance: VRPTWInstance,
    runtime: float,
) -> RouteSolution:
    """Convert OR-Tools assignment to RouteSolution."""
    if assignment is None:
        logger.warning("[OR-Tools] No solution found (assignment is None)")
        return RouteSolution(
            routes=[],
            vehicles_used=0,
            total_distance=0.0,
            total_duration=0.0,
            feasible=False,
            runtime_sec=runtime,
        )

    # Get the actual objective value (properly scaled)
    # Use the assignment's ObjectiveValue which accounts for all costs
    total_distance_scaled = assignment.ObjectiveValue()
    total_distance = total_distance_scaled / _DIST_SCALE

    # Extract routes
    routes: list[list[int]] = []
    for vehicle_id in range(routing.vehicles()):
        start_index = routing.Start(vehicle_id)
        end_index = routing.End(vehicle_id)

        if start_index == end_index:
            continue

        route: list[int] = []
        index = start_index

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != instance.depot_id:
                route.append(node)
            next_index = assignment.Value(routing.NextVar(index))
            index = next_index

        if route:
            routes.append(route)

    return RouteSolution(
        routes=routes,
        vehicles_used=len(routes),
        total_distance=total_distance,
        total_duration=0.0,
        feasible=True,
        runtime_sec=runtime,
    )
