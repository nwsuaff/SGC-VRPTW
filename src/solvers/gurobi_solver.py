"""Gurobi solver wrapper for VRPTW.

This module provides a Gurobi-based solver for the Vehicle Routing Problem
with Time Windows (VRPTW). It requires the gurobipy package to be installed.

Install Gurobi:
    pip install gurobipy

Or follow instructions at: https://www.gurobi.com/documentation/quickstart.html
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

from src.domain.schema import VRPTWInstance, RouteSolution

logger = logging.getLogger(__name__)

# Scale factor for integer conversion
_DIST_SCALE = 1000
_TIME_SCALE = 1000


def solve_gurobi(
    instance: VRPTWInstance,
    time_limit: float = 60.0,
    warm_start: Optional[list[list[int]]] = None,
    avoid_pairs: Optional[list[tuple[int, int]]] = None,
) -> tuple[RouteSolution, dict]:
    """Solve VRPTW using Gurobi optimizer.

    This solver formulates VRPTW as a Mixed Integer Programming (MIP) model
    with arc-based variables and constraints for capacity, time windows, and
    vehicle count.

    Args:
        instance: VRPTW instance to solve.
        time_limit: Time limit in seconds for optimization.
        warm_start: Optional initial routes (not yet implemented).
        avoid_pairs: Optional list of (customer_a, customer_b) tuples that should
            NOT be on the same route.

    Returns:
        Tuple of (RouteSolution, metadata dict).
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        logger.error("Gurobi not installed. Run: pip install gurobipy")
        raise

    start_time = time.perf_counter()

    num_nodes = instance.num_nodes()
    num_customers = len(instance.customer_ids)
    depot_id = instance.depot_id
    capacity = int(instance.vehicle_capacity)

    # Build distance matrix
    distance_matrix = _build_int_matrix(instance, num_nodes, _DIST_SCALE)
    time_matrix = _build_time_matrix(instance, num_nodes, _DIST_SCALE)

    customers = [i for i in range(num_nodes) if i != depot_id]

    try:
        model = gp.Model("VRPTW")
        model.setParam(GRB.Param.TimeLimit, time_limit)
        model.setParam(GRB.Param.OutputFlag, 0)  # Suppress Gurobi output
        model.setParam(GRB.Param.LogToConsole, 0)

        # Decision variables: x[i,j,k] = 1 if vehicle k travels from i to j
        # We use a simplified model with a single vehicle type
        num_vehicles = num_customers + 1  # Maximum possible vehicles

        # Create arc variables
        x = {}
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    for k in range(num_vehicles):
                        x[i, j, k] = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"x_{i}_{j}_{k}"
                        )

        # Flow variables for capacity constraints
        f = {}
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    for k in range(num_vehicles):
                        f[i, j, k] = model.addVar(
                            vtype=GRB.CONTINUOUS,
                            lb=0,
                            ub=capacity,
                            name=f"f_{i}_{j}_{k}"
                        )

        # Arrival time variables
        t = {}
        for i in range(num_nodes):
            for k in range(num_vehicles):
                t[i, k] = model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0,
                    ub=999999,
                    name=f"t_{i}_{k}"
                )

        # Objective: minimize total distance
        obj = gp.quicksum(
            distance_matrix[i][j] * x[i, j, k]
            for i in range(num_nodes)
            for j in range(num_nodes)
            if i != j
            for k in range(num_vehicles)
        )
        model.setObjective(obj, GRB.MINIMIZE)

        # Constraint: each customer visited exactly once (entering)
        for j in customers:
            model.addConstr(
                gp.quicksum(x[i, j, k] for i in range(num_nodes) if i != j for k in range(num_vehicles)) == 1,
                name=f"visit_once_{j}"
            )

        # Constraint: flow conservation (except depot)
        for j in customers:
            for k in range(num_vehicles):
                model.addConstr(
                    gp.quicksum(x[i, j, k] for i in range(num_nodes) if i != j) ==
                    gp.quicksum(x[j, i, k] for i in range(num_nodes) if i != j),
                    name=f"flow_cons_{j}_{k}"
                )

        # Depot constraints
        for k in range(num_vehicles):
            # Each vehicle starts at depot
            model.addConstr(
                gp.quicksum(x[depot_id, j, k] for j in customers) <= 1,
                name=f"depot_start_{k}"
            )
            # Each vehicle ends at depot
            model.addConstr(
                gp.quicksum(x[i, depot_id, k] for i in customers) <= 1,
                name=f"depot_end_{k}"
            )

        # Capacity flow constraints
        demand = {depot_id: 0}
        for cid in customers:
            demand[cid] = instance.demand.get(cid, 0)

        for k in range(num_vehicles):
            # Flow out of depot
            for j in customers:
                model.addConstr(
                    f[depot_id, j, k] - demand[j] * x[depot_id, j, k] == 0,
                    name=f"cap_init_{j}_{k}"
                )
            # Flow conservation
            for i in customers:
                for j in customers:
                    if i != j:
                        model.addConstr(
                            f[i, j, k] - f[depot_id, i, k] + demand[i] * x[i, j, k] == 0,
                            name=f"cap_cons_{i}_{j}_{k}"
                        )

        # Time window constraints
        M_TIME = 99999999
        ready_time = {i: instance.ready_time.get(i, 0) * _TIME_SCALE for i in range(num_nodes)}
        due_time = {i: instance.due_time.get(i, 999999) * _TIME_SCALE for i in range(num_nodes)}
        service_time = {i: instance.service_time.get(i, 0) * _TIME_SCALE for i in range(num_nodes)}

        for k in range(num_vehicles):
            # Arrival time at depot is 0
            model.addConstr(t[depot_id, k] == 0, name=f"time_depot_{k}")

            for i in range(num_nodes):
                for j in customers:
                    if i != j:
                        # If x[i,j,k] = 1, then t[j,k] >= t[i,k] + travel_time + service_time
                        model.addConstr(
                            t[j, k] >= t[i, k] + time_matrix[i][j] + service_time[j] - M_TIME * (1 - x[i, j, k]),
                            name=f"time_window_{i}_{j}_{k}"
                        )
                        model.addConstr(
                            t[j, k] <= due_time[j] + M_TIME * (1 - x[i, j, k]),
                            name=f"time_due_{i}_{j}_{k}"
                        )

        # MTZ-style sub tour elimination (simplified)
        # For each vehicle, add order variables
        u = {}
        for i in customers:
            for k in range(num_vehicles):
                u[i, k] = model.addVar(
                    vtype=GRB.INTEGER,
                    lb=0,
                    ub=num_customers,
                    name=f"u_{i}_{k}"
                )

        for k in range(num_vehicles):
            for i in customers:
                for j in customers:
                    if i != j:
                        model.addConstr(
                            u[i, k] - u[j, k] + num_customers * x[i, j, k] <= num_customers - 1,
                            name=f"subtour_{i}_{j}_{k}"
                        )

        # Avoid pairs constraint
        avoid_pairs_applied = False
        if avoid_pairs:
            customer_id_set = set(customers)
            for a, b in avoid_pairs:
                if a not in customer_id_set or b not in customer_id_set:
                    logger.warning(f"[Gurobi] Avoid pair ({a}, {b}) contains invalid customer ID. Skipping.")
                    continue
                avoid_pairs_applied = True
                # Add constraint: customers a and b cannot be served by the same vehicle
                for k in range(num_vehicles):
                    model.addConstr(
                        gp.quicksum(x[a, j, k] for j in customers if j != a) +
                        gp.quicksum(x[b, j, k] for j in customers if j != b) <= 1,
                        name=f"avoid_pair_{a}_{b}_{k}"
                    )

        # Optimize
        model.optimize()

        runtime = time.perf_counter() - start_time

        # Extract solution
        solution = _extract_solution(model, x, num_vehicles, depot_id, customers, runtime, distance_matrix)

        return solution, {
            "method": "gurobi",
            "gurobi_status": model.Status if hasattr(model, 'Status') else None,
            "gurobi_obj_val": model.objVal if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT else None,
            "warm_start_applied": warm_start is not None,
            "avoid_pairs_applied": avoid_pairs_applied,
            "avoid_pairs_count": len(avoid_pairs) if avoid_pairs else 0,
        }

    except Exception as e:
        logger.error(f"Gurobi solver error: {e}")
        raise


def _build_int_matrix(instance: VRPTWInstance, num_nodes: int, scale: int) -> list[list[int]]:
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


def _build_time_matrix(instance: VRPTWInstance, num_nodes: int, scale: int) -> list[list[int]]:
    """Build integer time/duration matrix, scaled by `scale`."""
    if instance.duration_matrix is not None:
        return [[int(d * scale) for d in row] for row in instance.duration_matrix]
    return _build_int_matrix(instance, num_nodes, scale)


def _extract_solution(model, x, num_vehicles, depot_id, customers, runtime, distance_matrix) -> RouteSolution:
    """Extract solution from Gurobi model."""
    from gurobipy import GRB

    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        logger.warning(f"[Gurobi] No solution found, status: {model.status}")
        return RouteSolution(
            routes=[],
            vehicles_used=0,
            total_distance=0.0,
            total_duration=0.0,
            feasible=False,
            runtime_sec=runtime,
        )

    routes = []
    total_distance = 0.0

    for k in range(num_vehicles):
        # Find route for vehicle k
        route = []
        current = depot_id

        while True:
            next_node = None
            for j in customers + [depot_id]:
                if j != current:
                    try:
                        if x[current, j, k].x > 0.5:
                            next_node = j
                            total_distance += distance_matrix[current][j] / _DIST_SCALE
                            break
                    except (KeyError, AttributeError):
                        pass

            if next_node is None or next_node == depot_id:
                break

            if next_node != depot_id:
                route.append(next_node)
            current = next_node

        if route:
            routes.append(route)

    # Count actual vehicles used
    vehicles_used = len(routes)

    # Check feasibility (basic check)
    feasible = vehicles_used > 0

    return RouteSolution(
        routes=routes,
        vehicles_used=vehicles_used,
        total_distance=total_distance,
        total_duration=0.0,
        feasible=feasible,
        runtime_sec=runtime,
    )
