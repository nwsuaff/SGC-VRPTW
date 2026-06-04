"""Debug feasibility checker: find which customers cause late violations."""

import sys
sys.path.insert(0, r'e:\vrp')

from src.data.solomon_loader import load_solomon_instance
from src.solvers.pyvrp_solver import solve_pyvrp, _build_model
from src.solvers.feasibility_checker import _check_single_route
from src.domain.schema import RouteSolution

inst = load_solomon_instance(r'e:\vrp\data\VRPTW\Solomon\R101.vrp')
inst.name = 'R101'

print(f"Instance: {inst.name}, {inst.num_nodes()-1} customers, capacity={inst.vehicle_capacity}")
print(f"Family: {getattr(inst, 'family', '?')}")

# Solve
print("\nSolving with PyVRP (10s)...")
sol, _ = solve_pyvrp(inst, time_limit=10.0, seed=42)
print(f"PyVRP says: feasible={sol.feasible}, vehicles={sol.vehicles_used}, dist={sol.total_distance:.0f}")

# Debug each route
print(f"\n{'─'*80}")
total_violations = 0
for ridx, route in enumerate(sol.routes):
    load = 0
    violations_this_route = 0
    arrival_times = []
    prev = 0
    current_time = 0.0

    for node in route:
        dist = inst.dist(prev, node)          # Euclidean
        travel = round(inst.travel_duration(prev, node))  # Rounded to int (matches Solomon's rounded-euclidean convention)
        current_time += travel

        ready = inst.ready_time.get(node, 0)
        due = inst.due_time.get(node, 999999)
        service = inst.service_time.get(node, 0)

        arr = current_time
        if current_time < ready:
            current_time = float(ready)
        if current_time > due:
            violations_this_route += 1
            total_violations += 1
            print(f"  *** VIOLATION Route {ridx+1}, customer {node:3d}: "
                  f"arrival={arr:.1f}, wait_to={current_time:.1f}, due={due}, "
                  f"violation_amount={current_time - due:.1f}")

        current_time += service
        load += inst.demand.get(node, 0)
        prev = node

    if violations_this_route == 0:
        # Only print first 2 routes cleanly
        if ridx < 2:
            total_demand = sum(inst.demand.get(n, 0) for n in route)
            print(f"  Route {ridx+1}: OK  ({len(route)} customers, load={total_demand}/{inst.vehicle_capacity})")

print(f"\n{'─'*80}")
print(f"Total violations found by checker: {total_violations}")

# Now check: what's the ACTUAL travel time PyVRP uses?
# Let's compute a few distances using different formulas
print(f"\n{'─'*80}")
print("Travel time formula comparison (depot(0,0) to customer 2(35.0,35.0)):")
c = 2
xi, yi = 0.0, 0.0
xj, yj = inst.x_coords[c], inst.y_coords[c]
print(f"  Customer {c}: coords=({xj}, {yj})")
print(f"  Our euclidean:  {((xj-xi)**2 + (yj-yi)**2)**0.5:.4f}")
print(f"  Rounded euclid: {round(((xj-xi)**2 + (yj-yi)**2)**0.5):.4f}")
print(f"  Customer {c} TW: ready={inst.ready_time[c]}, due={inst.due_time[c]}")
