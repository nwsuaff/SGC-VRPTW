"""PyVRP solver wrapper - PyVRP 0.11.x compatible.

Uses the Model API (pyvrp.Model) which is the correct interface for 0.11.x.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.domain.schema import VRPTWInstance, RouteSolution

logger = logging.getLogger(__name__)


def solve_pyvrp(
    instance: VRPTWInstance,
    time_limit: float = 10.0,
    seed: int = 42,
    warm_start: Optional[list[list[int]]] = None,
    biased_order: Optional[list[int]] = None,
    avoid_pairs: Optional[list[tuple[int, int]]] = None,
) -> tuple[RouteSolution, dict]:
    """Solve VRPTW using PyVRP (v0.11.x Model API).

    Args:
        instance: VRPTW instance to solve.
        time_limit: Time limit in seconds for the genetic algorithm.
        seed: Random seed.
        warm_start: List of route fragments (each a list of customer IDs). 
            Note: PyVRP 0.11.3 does not support initial solution injection.
        biased_order: Customer insertion order (not implemented in 0.11.3).
        avoid_pairs: Not natively supported, logged as warning.

    Returns:
        Tuple of (RouteSolution, metadata dict).
    """
    try:
        from pyvrp import Model, Solution
        from pyvrp.stop import MaxRuntime

        model, node_map = _build_model(instance)

        stop = MaxRuntime(time_limit)  # seconds (NOT milliseconds)

        initial_solution = None
        warm_start_accepted = False

        if warm_start:
            logger.warning("[PyVRP] warm_start is not supported in PyVRP 0.11.x, ignoring.")

        if biased_order:
            logger.warning("[PyVRP] biased_order is not supported in PyVRP 0.11.x, ignoring.")

        if avoid_pairs:
            logger.warning(
                f"[PyVRP] avoid_pairs={avoid_pairs} requested but not natively supported."
            )

        result = model.solve(
            stop=stop,
            seed=seed,
            display=False,
        )

        solution = _from_pyvrp_result(result, node_map)
        solution.metadata["warm_start_applied"] = warm_start_accepted

        return solution, {
            "method": "pyvrp_0.11",
            "warm_start_routes": len(warm_start) if warm_start else 0,
            "biased_order": bool(biased_order),
            "has_initial_solution": False,
        }

    except ImportError as exc:
        logger.error(f"PyVRP import error: {exc}. Run: pip install pyvrp==0.11.3")
        raise


def _build_model(instance: VRPTWInstance) -> tuple["Model", dict[int, any]]:
    """Build a PyVRP Model from a VRPTWInstance.
    
    Returns:
        Tuple of (Model, node_map) where node_map maps customer IDs to PyVRP Client objects.
        Depot is at index 0 in m.locations.
    """
    from pyvrp import Model

    model = Model()

    depot_x = int(instance.x_coords.get(instance.depot_id, 0))
    depot_y = int(instance.y_coords.get(instance.depot_id, 0))
    depot_tw_early = int(instance.ready_time.get(instance.depot_id, 0))
    depot_tw_late = int(instance.due_time.get(instance.depot_id, 999999))

    vehicle_capacity = int(instance.vehicle_capacity)
    vehicle_count = instance.vehicle_count or 10

    model.add_vehicle_type(
        num_available=vehicle_count,
        capacity=vehicle_capacity,
        tw_early=depot_tw_early,
        tw_late=depot_tw_late,
    )

    model.add_depot(
        x=depot_x,
        y=depot_y,
        tw_early=depot_tw_early,
        tw_late=depot_tw_late,
    )

    node_map: dict[int, any] = {}
    for cid in instance.customer_ids:
        client_x = int(instance.x_coords.get(cid, 0))
        client_y = int(instance.y_coords.get(cid, 0))
        demand = int(instance.demand.get(cid, 0))
        service_duration = int(instance.service_time.get(cid, 0))
        tw_early = int(instance.ready_time.get(cid, 0))
        tw_late = int(instance.due_time.get(cid, 999999))

        client = model.add_client(
            x=client_x,
            y=client_y,
            delivery=demand,
            service_duration=service_duration,
            tw_early=tw_early,
            tw_late=tw_late,
        )
        node_map[cid] = client

    locations = list(model.locations)

    if instance.distance_matrix is not None:
        for i, loc_i in enumerate(locations):
            for j, loc_j in enumerate(locations):
                dist = int(instance.distance_matrix[i][j])
                dur = int(instance.duration_matrix[i][j]) if instance.duration_matrix else dist
                model.add_edge(loc_i, loc_j, distance=dist, duration=dur)
    else:
        for i, loc_i in enumerate(locations):
            xi = instance.x_coords.get(i, 0)
            yi = instance.y_coords.get(i, 0)
            for j, loc_j in enumerate(locations):
                xj = instance.x_coords.get(j, 0)
                yj = instance.y_coords.get(j, 0)
                dist = int(((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5)
                model.add_edge(loc_i, loc_j, distance=dist, duration=dist)

    return model, node_map


def _from_pyvrp_result(result, node_map: dict[int, any]) -> RouteSolution:
    """Convert a PyVRP Result to a RouteSolution."""
    best = getattr(result, "best", None)
    if best is None:
        return RouteSolution(
            routes=[],
            vehicles_used=0,
            total_distance=0.0,
            total_duration=0.0,
            feasible=False,
            runtime_sec=0.0,
            first_feasible_sec=None,
        )

    elapsed = getattr(result, "elapsed_time", 0.0)
    if callable(elapsed):
        try:
            elapsed = elapsed()
        except Exception:
            elapsed = 0.0
    elapsed_sec = elapsed / 1000.0 if elapsed else 0.0

    first_feasible_sec = elapsed_sec if best.is_feasible() else None

    routes = []
    for route in best.routes():
        route_iter = list(route)
        if route_iter and hasattr(route_iter[0], "idx"):
            route_ids = [loc.idx for loc in route_iter]
        else:
            route_ids = list(route_iter)
        routes.append(route_ids)

    return RouteSolution(
        routes=routes,
        vehicles_used=len(routes),
        total_distance=best.distance(),
        total_duration=best.duration(),
        feasible=best.is_feasible(),
        runtime_sec=elapsed_sec,
        first_feasible_sec=first_feasible_sec,
    )
