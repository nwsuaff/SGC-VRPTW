"""Mock LLM client: deterministic hint generation for testing without real API calls."""

from __future__ import annotations

import json
import random
from typing import Any

from src.domain.schema import VRPTWInstance, RouteSolution
from src.llm.hint_schema import LLMHints, empty_hints, SolverControl
from src.llm.prompt_builder import build_llm_prompt


class MockLLMClient:
    """A deterministic mock LLM client for development and testing.

    The mock implements simple, non-trivial heuristics that mirror what a real
    LLM might suggest, without requiring API keys or network access.
    """

    def __init__(self, seed: int = 42, max_priority: int = 5, max_subroutes: int = 3):
        """Initialize the mock client.

        Args:
            seed: Random seed for reproducibility.
            max_priority: Maximum number of priority customers to suggest.
            max_subroutes: Maximum number of locked subroutes to suggest.
        """
        self._rng = random.Random(seed)
        self._max_priority = max_priority
        self._max_subroutes = max_subroutes

    def query(
        self,
        instance: VRPTWInstance,
        incumbent: RouteSolution | None = None,
        **kwargs: Any,
    ) -> LLMHints:
        """Generate hints for a VRPTW instance.

        The mock logic:
        1. Prioritizes customers with earliest due times.
        2. Locks short feasible prefixes from the incumbent (if available).
        3. Proposes 1-2 repair moves if an incumbent exists.
        4. Suggests a small extra time budget.

        Args:
            instance: The VRPTW instance.
            incumbent: Current incumbent solution (may be None).
            **kwargs: Additional arguments (ignored by mock).

        Returns:
            LLMHints with suggested actions.
        """
        priority_customers = self._get_priority_customers(instance)
        locked_subroutes = self._get_locked_subroutes(instance, incumbent)
        avoid_pairs = self._get_avoid_pairs(instance, incumbent)
        suggested_moves = self._get_repair_moves(instance, incumbent)
        solver_control = self._get_solver_control(incumbent)
        rationale = self._build_rationale(instance, incumbent)

        return LLMHints(
            priority_customers=priority_customers,
            locked_subroutes=locked_subroutes,
            avoid_pairs=avoid_pairs,
            suggested_moves=suggested_moves,
            solver_control=solver_control,
            rationale=rationale,
        )

    def query_raw(self, prompt: str) -> str:
        """Return a raw JSON string response.

        Args:
            prompt: The prompt string (not used in mock).

        Returns:
            A JSON string with mock hints.
        """
        hints = self.query(
            VRPTWInstance.model_validate_json(prompt)  # type: ignore
        )
        return hints.model_dump_json()

    def _get_priority_customers(self, instance: VRPTWInstance) -> list[int]:
        """Return customers with earliest due times, sorted ascending."""
        sorted_customers = sorted(
            instance.customer_ids,
            key=lambda cid: instance.due_time.get(cid, 999999),
        )
        return sorted_customers[: self._max_priority]

    def _get_locked_subroutes(self, instance: VRPTWInstance, incumbent: RouteSolution | None) -> list[list[int]]:
        """Lock short feasible prefixes from the incumbent."""
        if incumbent is None or not incumbent.routes:
            return []

        locked: list[list[int]] = []

        for route in incumbent.routes:
            if not route:
                continue

            feasible_prefix: list[int] = []
            cumulative_time = 0.0
            cumulative_load = 0

            for cid in route:
                travel = instance.dist(instance.depot_id if not feasible_prefix else feasible_prefix[-1], cid)
                cumulative_time += travel

                ready = instance.ready_time.get(cid, 0)
                due = instance.due_time.get(cid, 999999)
                demand = instance.demand.get(cid, 0)

                if cumulative_time <= due and cumulative_load + demand <= instance.vehicle_capacity:
                    feasible_prefix.append(cid)
                else:
                    break

            if len(feasible_prefix) >= 2:
                locked.append(feasible_prefix[: min(3, len(feasible_prefix))])

            if len(locked) >= self._max_subroutes:
                break

        return locked

    def _get_avoid_pairs(self, instance: VRPTWInstance, incumbent: RouteSolution | None) -> list[list[int]]:
        """Suggest pairs of distant customers that should not share a route."""
        if incumbent is None or not incumbent.routes:
            return []

        avoid: list[list[int]] = []

        for route in incumbent.routes:
            if len(route) < 3:
                continue
            mid_idx = len(route) // 2
            if mid_idx > 0:
                avoid.append([route[0], route[mid_idx]])
            if mid_idx + 1 < len(route):
                avoid.append([route[mid_idx], route[-1]])

        return avoid[:2]

    def _get_repair_moves(self, instance: VRPTWInstance, incumbent: RouteSolution | None) -> list:
        """Propose repair moves to improve the incumbent."""
        moves = []

        if incumbent is None or not incumbent.routes:
            return moves

        for route_idx, route in enumerate(incumbent.routes):
            if len(route) >= 3:
                first_customer = route[0]
                last_customer = route[-1]

                if self._rng.random() < 0.5:
                    moves.append({
                        "type": "relocate",
                        "customer": last_customer,
                        "target_route": route_idx,
                        "target_position": 0,
                    })
                else:
                    moves.append({
                        "type": "swap",
                        "a": first_customer,
                        "b": last_customer,
                    })

                break

        return moves[:2]

    def _get_solver_control(self, incumbent: RouteSolution | None) -> SolverControl:
        """Suggest solver runtime parameters."""
        extra_seconds = 2 if incumbent is not None else 5
        return SolverControl(
            extra_seconds=extra_seconds,
            intensify=incumbent is not None,
            diversify=incumbent is None,
        )

    def generate_code(self, prompt: str) -> str:
        """Generate code (mock implementation).

        This is a simple mock that generates a basic VRPTW solver template.
        In production, this would call the real LLM API.
        """
        # Simple mock code generator
        code = '''
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def solve(time_matrix, demands, time_windows, service_times, num_vehicle, depot, vehicle_capacity):
    """Mock VRPTW solver."""
    n = len(time_matrix)
    manager = pywrapcp.RoutingIndexManager(n, num_vehicle, depot)
    routing = pywrapcp.RoutingModel(manager)

    def transit_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(transit_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Capacity
    def demand_callback(from_index):
        return demands[manager.IndexToNode(from_index)]

    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand_callback),
        0, [vehicle_capacity] * num_vehicle, True, "Capacity"
    )

    # Time
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node] + service_times[to_node]

    routing.AddDimension(
        routing.RegisterTransitCallback(time_callback),
        99999, 9999999, False, "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")

    for i, (ready, due) in enumerate(time_windows):
        if i != depot:
            time_dim.CumulVar(manager.NodeToIndex(i)).SetRange(ready, due)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_params)

    if solution:
        routes = []
        total_dist = 0
        vehicles_used = 0

        for v in range(num_vehicle):
            idx = routing.Start(v)
            if routing.IsEnd(idx):
                continue
            route = []
            dist = 0
            prev = depot
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node != depot:
                    route.append(node)
                next_idx = solution.Value(routing.NextVar(idx))
                next_node = manager.IndexToNode(next_idx)
                dist += time_matrix[node][next_node]
                idx = next_idx
            if route:
                routes.append(route)
                total_dist += dist
                vehicles_used += 1

        return {"routes": routes, "total_distance": total_dist, "vehicles_used": vehicles_used}
    else:
        return {"routes": [], "total_distance": 0, "vehicles_used": 0}
'''
        return code

    def _build_rationale(self, instance: VRPTWInstance, incumbent: RouteSolution | None) -> str:
        """Build a human-readable rationale."""
        parts = []
        parts.append(
            f"For instance {instance.name} with {instance.size} customers, "
            f"I prioritize customers with earliest due dates to ensure feasibility."
        )
        if incumbent and incumbent.routes:
            parts.append(
                f"The current incumbent uses {incumbent.vehicles_used} vehicles "
                f"with total distance {incumbent.total_distance:.1f}."
            )
            parts.append("I lock feasible prefixes and propose targeted repair moves.")
        else:
            parts.append("Starting from scratch; I suggest a diversified search.")
        return " ".join(parts)
