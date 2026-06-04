"""Code templates for VRPTW problems.

This module provides base templates that the LLM will extend and complete
when generating solver code. The templates follow OR-Tools pywrapcp API.
"""

from __future__ import annotations

import textwrap


# ─────────────────────────────────────────────────────────────────────────────
# Base Template - VRPTW with capacity and time windows
# ─────────────────────────────────────────────────────────────────────────────

VRPTW_BASE_TEMPLATE = textwrap.dedent('''
def solve(time_matrix: list[list[int]], demands: list[int], time_windows: list[tuple[int, int]], 
          service_times: list[int], num_vehicle: int, depot: int, vehicle_capacity: int) -> dict:
    """
    Solve Vehicle Routing Problem with Time Windows (VRPTW).
    
    Args:
        time_matrix: Travel time matrix (n x n), where n is number of nodes.
        demands: Demand for each node (0 for depot).
        time_windows: Time window for each node as (ready_time, due_time).
        service_times: Service time for each node.
        num_vehicle: Number of vehicles available.
        depot: Index of depot node.
        vehicle_capacity: Capacity of each vehicle.
    
    Returns:
        dict with keys: 
            - 'routes': list of routes, each route is a list of customer indices
            - 'total_distance': total travel distance
            - 'vehicles_used': number of vehicles actually used
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    
    # Create the routing index manager
    # YOUR CODE HERE: Initialize manager with proper dimensions
    manager = None
    routing = None
    
    # Define transit callback for time/distance
    def transit_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # YOUR CODE HERE: Return the travel time
        return 0
    
    # Register transit callback
    transit_callback_index = routing.RegisterTransitCallback(transit_callback)
    
    # Set arc cost (distance dimension)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add capacity dimension
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return demands[node]
    
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand_callback),
        0,  # null capacity
        [vehicle_capacity] * num_vehicle,  # vehicle capacities
        True,  # start cumul at zero
        "Capacity"
    )
    
    # Add time dimension
    # YOUR CODE HERE: Add time dimension with proper slack
    time_dimension = None
    
    # Set time window constraints for each location
    for location_idx, (ready, due) in enumerate(time_windows):
        if location_idx == depot:
            continue
        index = manager.NodeToIndex(location_idx)
        # YOUR CODE HERE: Set time window range for this node
        pass
    
    # Set depot time window
    depot_index = manager.NodeToIndex(depot)
    # YOUR CODE HERE: Set depot time window
    pass
    
    # Set first solution strategy
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 60

    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    
    # Extract solution
    if solution:
        # YOUR CODE HERE: Extract routes and compute metrics
        routes = []
        total_distance = 0
        vehicles_used = 0
        
        for vehicle_id in range(num_vehicle):
            index = routing.Start(vehicle_id)
            if routing.IsEnd(index):
                continue
            route = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != depot:
                    route.append(node)
                index = solution.Value(routing.NextVar(index))
            if route:
                routes.append(route)
                vehicles_used += 1
        
        return {{
            "routes": routes,
            "total_distance": total_distance,
            "vehicles_used": vehicles_used
        }}
    else:
        return {{
            "routes": [],
            "total_distance": 0,
            "vehicles_used": 0
        }}
''').strip()


# ─────────────────────────────────────────────────────────────────────────────
# Warm-Start Template - VRPTW with initial solution hint
# ─────────────────────────────────────────────────────────────────────────────

VRPTW_WARMSTART_TEMPLATE = textwrap.dedent('''
def solve(time_matrix: list[list[int]], demands: list[int], time_windows: list[tuple[int, int]], 
          service_times: list[int], num_vehicle: int, depot: int, vehicle_capacity: int,
          initial_routes: list[list[int]] = None) -> dict:
    """
    Solve Vehicle Routing Problem with Time Windows (VRPTW) with warm start.
    
    Args:
        time_matrix: Travel time matrix (n x n).
        demands: Demand for each node (0 for depot).
        time_windows: Time window for each node as (ready_time, due_time).
        service_times: Service time for each node.
        num_vehicle: Number of vehicles available.
        depot: Index of depot node.
        vehicle_capacity: Capacity of each vehicle.
        initial_routes: Initial solution routes to warm-start the solver.
    
    Returns:
        dict with keys: 'routes', 'total_distance', 'vehicles_used'
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    
    # Use initial routes to determine actual number of vehicles needed
    if initial_routes:
        num_vehicle = len(initial_routes)
    
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(time_matrix), num_vehicle, depot)
    routing = pywrapcp.RoutingModel(manager)
    
    # Transit callback
    def transit_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(transit_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Capacity dimension
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return demands[node]
    
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand_callback),
        0,
        [vehicle_capacity] * num_vehicle,
        True,
        "Capacity"
    )
    
    # Time dimension with service times
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node] + service_times[to_node]
    
    routing.AddDimension(
        routing.RegisterTransitCallback(time_callback),
        99999,  # allow waiting time
        9999999,  # max time
        False,  # don't force start cumul to zero
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    
    # Time window constraints
    for location_idx, (ready, due) in enumerate(time_windows):
        index = manager.NodeToNode(location_idx)
        time_dimension.CumulVar(index).SetRange(ready, due)
    
    # Warm start: apply initial routes if provided
    if initial_routes:
        # YOUR CODE HERE: Apply warm-start using routing.ReadAssignmentFromRoutes
        pass
    
    # First solution strategy
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 60

    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    
    # Extract solution
    if solution:
        routes = []
        total_distance = 0
        vehicles_used = 0
        
        for vehicle_id in range(num_vehicle):
            index = routing.Start(vehicle_id)
            if routing.IsEnd(index):
                continue
            route = []
            route_dist = 0
            prev_node = depot
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != depot:
                    route.append(node)
                next_index = solution.Value(routing.NextVar(index))
                if not routing.IsEnd(next_index):
                    next_node = manager.IndexToNode(next_index)
                    route_dist += time_matrix[node][next_node]
                index = next_index
            
            if route:
                routes.append(route)
                total_distance += route_dist
                vehicles_used += 1
        
        return {{
            "routes": routes,
            "total_distance": total_distance,
            "vehicles_used": vehicles_used
        }}
    else:
        return {{
            "routes": [],
            "total_distance": 0,
            "vehicles_used": 0
        }}
''').strip()


# ─────────────────────────────────────────────────────────────────────────────
# Lexicographic Template - Minimize vehicles first, then distance
# ─────────────────────────────────────────────────────────────────────────────

VRPTW_LEXICOGRAPHIC_TEMPLATE = textwrap.dedent('''
def solve(time_matrix: list[list[int]], demands: list[int], time_windows: list[tuple[int, int]], 
          service_times: list[int], num_vehicle: int, depot: int, vehicle_capacity: int) -> dict:
    """
    Solve VRPTW with lexicographic objective: minimize vehicles first, then distance.
    
    Strategy:
    1. First, find minimum number of vehicles needed
    2. Then, minimize total distance with that vehicle count
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    
    # Phase 1: Find minimum vehicles
    min_vehicles = _find_minimum_vehicles(
        time_matrix, demands, time_windows, service_times, 
        num_vehicle, depot, vehicle_capacity
    )
    
    # Phase 2: Optimize distance with fixed vehicle count
    manager = pywrapcp.RoutingIndexManager(len(time_matrix), min_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)
    
    def transit_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(transit_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Capacity dimension
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return demands[node]
    
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand_callback),
        0,
        [vehicle_capacity] * min_vehicles,
        True,
        "Capacity"
    )
    
    # Time dimension
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node] + service_times[to_node]
    
    routing.AddDimension(
        routing.RegisterTransitCallback(time_callback),
        99999,
        9999999,
        False,
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    
    # Time windows
    for location_idx, (ready, due) in enumerate(time_windows):
        index = manager.NodeToNode(location_idx)
        time_dimension.CumulVar(index).SetRange(ready, due)
    
    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 60  # bound Phase 2 search so we return a result

    solution = routing.SolveWithParameters(search_parameters)
    
    # Extract solution
    if solution:
        routes = []
        total_distance = 0
        vehicles_used = 0
        
        for vehicle_id in range(min_vehicles):
            index = routing.Start(vehicle_id)
            if routing.IsEnd(index):
                continue
            route = []
            route_dist = 0
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != depot:
                    route.append(node)
                next_index = solution.Value(routing.NextVar(index))
                if not routing.IsEnd(next_index):
                    next_node = manager.IndexToNode(next_index)
                    route_dist += time_matrix[node][next_node]
                index = next_index
            
            if route:
                routes.append(route)
                total_distance += route_dist
                vehicles_used += 1
        
        return {{
            "routes": routes,
            "total_distance": total_distance,
            "vehicles_used": vehicles_used
        }}
    else:
        return {{"routes": [], "total_distance": 0, "vehicles_used": 0}}


def _find_minimum_vehicles(time_matrix, demands, time_windows, service_times, 
                           num_vehicle, depot, vehicle_capacity):
    """Binary search for minimum number of vehicles needed."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    
    low, high = 1, num_vehicle
    
    while low < high:
        mid = (low + high) // 2
        manager = pywrapcp.RoutingIndexManager(len(time_matrix), mid, depot)
        routing = pywrapcp.RoutingModel(manager)
        
        def tc(fi, ti):
            fn = manager.IndexToNode(fi)
            tn = manager.IndexToNode(ti)
            return time_matrix[fn][tn]
        
        tc_idx = routing.RegisterTransitCallback(tc)
        routing.SetArcCostEvaluatorOfAllVehicles(tc_idx)
        
        def dc(fi):
            return demands[manager.IndexToNode(fi)]
        
        routing.AddDimensionWithVehicleCapacity(
            routing.RegisterUnaryTransitCallback(dc),
            0, [vehicle_capacity] * mid, True, "Cap"
        )
        
        def tfc(fi, ti):
            fn = manager.IndexToNode(fi)
            tn = manager.IndexToNode(ti)
            return time_matrix[fn][tn] + service_times[tn]
        
        routing.AddDimension(routing.RegisterTransitCallback(tfc), 99999, 9999999, False, "Time")
        td = routing.GetDimensionOrDie("Time")
        
        for li, (r, d) in enumerate(time_windows):
            if li != depot:
                td.CumulVar(manager.NodeToIndex(li)).SetRange(r, d)
        
        sp = pywrapcp.DefaultRoutingSearchParameters()
        sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        sp.time_limit.seconds = 10  # 10s per feasibility check; enough for 800-customer instances

        if routing.SolveWithParameters(sp):
            high = mid
        else:
            low = mid + 1
    
    return low
''').strip()


# ─────────────────────────────────────────────────────────────────────────────
# Constraint-Specific Code Snippets
# ─────────────────────────────────────────────────────────────────────────────

VRPTW_CONSTRAINT_SNIPPETS = {
    "Capacitated": '''
# Capacity dimension implementation
def demand_callback(from_index):
    node = manager.IndexToNode(from_index)
    return demands[node]

routing.AddDimensionWithVehicleCapacity(
    routing.RegisterUnaryTransitCallback(demand_callback),
    0,  # null capacity slack
    [vehicle_capacity] * num_vehicle,  # vehicle capacities
    True,  # start cumul at zero
    "Capacity"
)
''',
    "Time Windows": '''
# Time window constraint implementation
time_dimension = routing.GetDimensionOrDie("Time")

# Set time window for each location
for location_idx, (ready, due) in enumerate(time_windows):
    if location_idx == depot:
        continue
    index = manager.NodeToIndex(location_idx)
    time_dimension.CumulVar(index).SetRange(ready, due)

# Set depot time window
depot_index = manager.NodeToIndex(depot)
time_dimension.CumulVar(depot_index).SetRange(
    time_windows[depot][0], time_windows[depot][1]
)
''',
    "Service Time": '''
# Service time in transit callback
def time_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    # Travel time + service time at destination
    return time_matrix[from_node][to_node] + service_times[to_node]

transit_callback_index = routing.RegisterTransitCallback(time_callback)
''',
    "Multiple Depots": '''
# Multiple depots implementation
# For multiple depots, create separate routing models or use depot-specific assignments
# Option 1: Multiple RoutingIndexManagers
depots = [0, 10]  # List of depot node indices
vehicles_per_depot = [num_vehicle // 2, num_vehicle // 2]

for depot_idx, depot in enumerate(depots):
    manager = pywrapcp.RoutingIndexManager(
        len(time_matrix), 
        vehicles_per_depot[depot_idx], 
        depot
    )
    # Create routing model for this depot...
''',
    "Duration Limit": '''
# Duration limit / maximum route time
time_dimension = routing.GetDimensionOrDie("Time")

# Set maximum route duration for each vehicle
for vehicle_id in range(num_vehicle):
    start_index = routing.Start(vehicle_id)
    # Set maximum cumulative time at start (depot)
    time_dimension.CumulVar(start_index).SetRange(0, max_duration)
    # Set maximum cumulative time at end
    end_index = routing.End(vehicle_id)
    time_dimension.CumulVar(end_index).SetUpper(max_duration)
''',
    "Multiple Vehicles": '''
# Multiple vehicles configuration
num_vehicle = 10  # Total number of vehicles
depot = 0  # Single depot

manager = pywrapcp.RoutingIndexManager(len(time_matrix), num_vehicle, depot)
routing = pywrapcp.RoutingModel(manager)

# Or with heterogeneous fleet (different capacities per vehicle)
vehicle_capacities = [200, 300, 400]  # Different capacities per vehicle
routing.AddDimensionWithVehicleCapacity(
    routing.RegisterUnaryTransitCallback(demand_callback),
    0,
    vehicle_capacities,
    True,
    "Capacity"
)
''',
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Convert VRPTWInstance to solve() parameters
# ─────────────────────────────────────────────────────────────────────────────

def instance_to_solve_params(instance, include_distance_matrix=True):
    """Convert VRPTWInstance to parameters for solve() function.
    
    Returns dict with keys matching solve() function signature.
    """
    n = instance.num_nodes()
    
    # Build distance matrix
    if include_distance_matrix and instance.distance_matrix is not None:
        time_matrix = [[int(d) for d in row] for row in instance.distance_matrix]
    else:
        time_matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(int(instance.dist(i, j)))
            time_matrix.append(row)
    
    # Demands (0 for depot)
    demands = [0] * n
    for cid in instance.customer_ids:
        demands[cid] = instance.demand.get(cid, 0)
    
    # Time windows
    time_windows = [(0, 999999)] * n  # Default wide window
    for cid in instance.customer_ids:
        ready = instance.ready_time.get(cid, 0)
        due = instance.due_time.get(cid, 999999)
        time_windows[cid] = (ready, due)
    
    # Service times
    service_times = [0] * n
    for cid in instance.customer_ids:
        service_times[cid] = instance.service_time.get(cid, 0)
    
    # Vehicle info
    num_vehicle = instance.vehicle_count or n
    depot = instance.depot_id
    vehicle_capacity = instance.vehicle_capacity
    
    return {
        "time_matrix": time_matrix,
        "demands": demands,
        "time_windows": time_windows,
        "service_times": service_times,
        "num_vehicle": num_vehicle,
        "depot": depot,
        "vehicle_capacity": vehicle_capacity,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Additional Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def get_constraint_snippet(constraint_name: str) -> str | None:
    """Get the code snippet for a specific constraint.
    
    Args:
        constraint_name: Name of the constraint (e.g., "Capacitated").
    
    Returns:
        Code snippet string or None if not found.
    """
    return VRPTW_CONSTRAINT_SNIPPETS.get(constraint_name)


def build_constraint_context_from_snippets(keywords: list[str]) -> str:
    """Build a context string from constraint snippets.
    
    Args:
        keywords: List of constraint keywords.
    
    Returns:
        Combined context string with all relevant snippets.
    """
    parts = []
    for kw in keywords:
        snippet = get_constraint_snippet(kw)
        if snippet:
            parts.append(f"## {kw}\n{snippet}\n")
    return "\n".join(parts)


def get_template_for_objective(objective: str) -> str:
    """Get the appropriate template for the given objective.
    
    Args:
        objective: Optimization objective.
    
    Returns:
        Template string.
    """
    objective_lower = objective.lower()
    
    if "lexicographic" in objective_lower or "vehicles" in objective_lower:
        return VRPTW_LEXICOGRAPHIC_TEMPLATE
    elif "warmstart" in objective_lower or "initial" in objective_lower:
        return VRPTW_WARMSTART_TEMPLATE
    else:
        return VRPTW_BASE_TEMPLATE


# ─────────────────────────────────────────────────────────────────────────────
# Scene Card Enhanced Prompts (DRoC + Scene Card)
# ─────────────────────────────────────────────────────────────────────────────

def build_scene_card_prompt(instance, scene_card_str: str, template: str) -> str:
    """Build an enhanced prompt with Scene Card information.
    
    This function combines the VRPTW template with Scene Card analysis
    to provide more context-aware code generation.
    
    Args:
        instance: The VRPTW instance.
        scene_card_str: Pre-formatted Scene Card string.
        template: The VRPTW template to use.
    
    Returns:
        Enhanced prompt string for LLM.
    """
    prompt = f"""## VRPTW Code Generation Task

You are generating OR-Tools solver code for a Vehicle Routing Problem with Time Windows (VRPTW).

### Scene Card (Instance Analysis)
Use the following scene analysis to guide your code generation:

{scene_card_str}

### Problem Instance
- Instance: {instance.name}
- Customers: {instance.size}
- Vehicle capacity: {instance.vehicle_capacity}
- Vehicle count: {instance.vehicle_count or 'unlimited'}

### Code Template
Follow this template structure exactly:

```python
{template}
```

### Requirements
1. Read the template carefully and understand the parameter meanings
2. Follow the template format strictly to generate code
3. Ensure all parameters in the template are properly used
4. Do not include additional examples or main function for testing
5. Return the objective value by the 'solve' function
6. Ensure code is syntactically correct and can be executed

### Important Notes
- Critical customers (tight time windows): {scene_card_str.split('Critical Customers')[-1].split('Farthest')[0].strip() if 'Critical Customers' in scene_card_str else 'None identified'}
- Distribution type affects routing strategy: clustered customers may benefit from sector-based approaches
- Tight time windows require careful scheduling to avoid late arrivals

Generate the complete solve() function:"""
    
    return prompt


def build_scene_card_from_instance(instance) -> str:
    """Build a Scene Card string from a VRPTW instance.
    
    This is a convenience function that creates a Scene Card
    without requiring the full SceneCard class.
    
    Args:
        instance: The VRPTW instance.
    
    Returns:
        Formatted Scene Card string.
    """
    from src.llm.scene_card import create_scene_card
    
    card = create_scene_card(instance)
    return card.to_prompt_string()
